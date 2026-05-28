"""이미지 모드 캡처 유스케이스 — 슬라이스 핵심 조립자.

슬라이스 흐름:
  load_frame  → 소스 첫 프레임 추출 (모델 로드 안 함)
  segment     → 클릭 포인트 → 세그멘테이션 → SegmentResult(마스크·centroid)
  compute_box → SegmentResult.centroid + BoxParams → 크롭 박스(순수·즉시)
  make_crop_box → segment + compute_box 조합(하위호환 위임자)
  export       → 크롭 후 파일 저장

설계 원칙:
  - segment: 무거움(모델 IO). 워커 스레드에서 클릭당 1회만 호출.
  - compute_box: 순수·가벼움. 종횡비/크기 슬라이더 변경 시 메인 스레드에서
    재세그 없이 즉시 호출. backend 절대 미호출.

WHY: Protocol 주입(DIP)으로 구체 백엔드·소스를 교체 가능하게 한다.
     테스트에서는 FakeBackend/FakeFrameSource로 치환해 모델 없이 검증한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from easy_capture.core.crop import (
    apply_aspect_lock,
    centroid_of_mask,
    make_crop_box,
)
from easy_capture.core.export.image_export import ExportConfig, crop_array, save_image
from easy_capture.core.segmentation.backend import SegmentationBackend
from easy_capture.core.upscale.backend import UpscaleBackend
from easy_capture.infra.video_io import FrameSource


class EmptyMaskError(Exception):
    """세그멘테이션 결과 마스크가 비어 있을 때 발생한다.

    WHY: 조용한 폴백(클릭 좌표 대체) 대신 명시적 예외로 호출자에게 알린다.
         UI 워커가 이를 잡아 한국어 안내 메시지를 표시한다(계획서 §8).
    """


@dataclass(frozen=True)
class SegmentResult:
    """세그멘테이션 1회 성공 결과.

    mask: bool HxW — 오버레이 표시용
    centroid: (cx, cy) — 박스 계산용

    WHY: frozen=True로 불변 보장. 실수로 캐시된 세그 결과를 덮어쓰는
         버그를 컴파일 타임에 차단한다. 빈 마스크는 담지 않고
         segment()에서 EmptyMaskError를 올린다.
    """

    mask: np.ndarray
    centroid: tuple[float, float]


@dataclass(frozen=True)
class BoxParams:
    """compute_box 입력 묶음 — 매개변수 3개 규칙 준수용.

    box_size: 요청 크롭 (W, H)
    aspect: 종횡비 프리셋 키('1:1', '9:16', '16:9') or None
    frame_shape: (W, H) 프레임 크기 (경계 클램프용)

    WHY: frozen=True로 불변. 슬라이더/콤보 변경마다 새 BoxParams를
         생성하고 이전 값은 버린다 — 실수 변경 방지.
    """

    box_size: tuple[int, int]
    aspect: str | None
    frame_shape: tuple[int, int]


@dataclass(frozen=True)
class CropRequest:
    """크롭 요청 파라미터.

    point: 클릭 좌표 (이미지 기준 x, y)
    box_size: 요청 크롭 크기 (W, H)
    aspect: 종횡비 프리셋 키('9:16', '1:1' 등) or None
    """

    point: tuple[int, int]
    box_size: tuple[int, int]
    aspect: str | None = None


def _frame_wh(frame: np.ndarray) -> tuple[int, int]:
    """프레임 배열에서 (W, H) 순서로 크기를 반환한다.

    WHY: frame.shape[:2]는 (H, W)순서이므로 뒤집어야 core.make_crop_box의
         frame_size=(W, H) 인자와 순서가 일치한다. 변환 누락 방지용 헬퍼.
    """
    h, w = frame.shape[:2]
    return w, h


class ImageCaptureUseCase:
    """이미지 모드 캡처 유스케이스.

    source와 backend는 Protocol 타입으로 주입받는다(DIP).
    구체 구현(SAM2, PyAV 등)은 진입점(router)에서 결정한다.
    """

    def __init__(self, source: FrameSource, backend: SegmentationBackend) -> None:
        """프레임 공급원·세그 백엔드 주입."""
        self._source = source
        self._backend = backend

    def load_frame(self) -> np.ndarray:
        """소스 첫 프레임을 추출한다(모델 로드 안 함).

        WHY: 파일 열기 직후 즉시 프레임을 표시하기 위해
             무거운 모델 로드와 분리한다(지연 로드 전략, ADR 0007).
        """
        return self._source.read_frame(index=0)

    def segment(
        self, frame: np.ndarray, point: tuple[int, int]
    ) -> SegmentResult:
        """클릭 포인트로 세그멘테이션 1회 실행 → SegmentResult 반환(무거움).

        WHY: 이 메서드에서만 backend.segment_image를 호출한다.
             워커 스레드에서 클릭당 1회만 호출하고 결과를 main_window가 보관.
             빈 마스크면 EmptyMaskError(compute_box는 유효한 centroid만 받음).
        """
        mask = self._backend.segment_image(frame, points=[point])
        centroid = centroid_of_mask(mask)
        if centroid is None:
            raise EmptyMaskError("대상을 인식하지 못했어요. 다시 클릭해 주세요.")
        return SegmentResult(mask=mask, centroid=centroid)

    def compute_box(
        self, centroid: tuple[float, float], params: BoxParams
    ) -> tuple[int, int, int, int]:
        """centroid + BoxParams → 크롭 박스(순수·가벼움).

        backend를 절대 호출하지 않는다. 메인 스레드에서 슬라이더/콤보
        변경마다 즉시 재호출해도 멈춤이 없다.

        WHY: 세그(무거움)와 박스 계산(가벼움)의 분리가 이 슬라이스의
             핵심 설계 결정이다(계획서 §1-1). 재세그 카운터 테스트가
             이 순수성을 회귀 가드로 강제한다.
        """
        locked = apply_aspect_lock(*params.box_size, params.aspect)
        return make_crop_box(centroid, locked, params.frame_shape)

    def make_crop_box(
        self, frame: np.ndarray, request: CropRequest
    ) -> tuple[int, int, int, int]:
        """클릭→박스 1회 조합(하위호환 위임자).

        내부는 segment + compute_box 위임으로 재작성(계획서 A안).
        기존 테스트(TestMakeCropBox, TestEndToEndHappyPath)가 그대로 통과한다.

        WHY: 한 번에 박스까지 뽑는 호출부(배치, 향후 자동화)를 위해 유지한다.
             로직은 segment/compute_box 단일 소스에 있고 여기는 위임만.
        """
        seg = self.segment(frame, request.point)
        params = BoxParams(request.box_size, request.aspect, _frame_wh(frame))
        return self.compute_box(seg.centroid, params)

    def export(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
        target: tuple[str, ExportConfig],
        upscaler: UpscaleBackend | None = None,
    ) -> None:
        """크롭 → (옵션) 업스케일 → 저장.

        upscaler가 None이면 기존 즉시 저장 경로(무회귀). 주어지면 크롭 결과만
        업스케일(전체 프레임 아님) 후 저장. 무거운 추론은 호출자(워커)가 책임진다.

        WHY: 업스케일을 별도 단계로 끼워 순수(crop/save)와 무거움(upscale)을 분리.
             upscaler 주입은 DIP — usecase는 UpscaleBackend Protocol에만 의존.
             target 튜플 묶음으로 위치 인자 3개 유지(매개변수 규칙 준수).
        """
        path, config = target
        cropped = crop_array(frame, box)
        image = upscaler.upscale(cropped) if upscaler is not None else cropped
        save_image(image, path, config)
