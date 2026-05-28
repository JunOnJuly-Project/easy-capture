"""이미지 모드 캡처 유스케이스 — 슬라이스 핵심 조립자.

슬라이스 흐름:
  load_frame  → 소스 첫 프레임 추출 (모델 로드 안 함)
  make_crop_box → 클릭 포인트 → 세그멘테이션 → centroid → 박스 산출
  export       → 크롭 후 파일 저장

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
from easy_capture.infra.video_io import FrameSource


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

    def make_crop_box(
        self, frame: np.ndarray, request: CropRequest
    ) -> tuple[int, int, int, int]:
        """클릭 포인트로 세그멘테이션 → centroid → 종횡비잠금 → 크롭박스를 산출한다.

        WHY: 이 메서드 이름은 core.make_crop_box(기하 함수)와 구별을 위해
             유스케이스의 "조립" 역할임을 주석으로 명시한다.
             실제로 core.make_crop_box를 내부에서 호출한다.

        반환: (x1, y1, x2, y2) — 짝수 정렬, 프레임 경계 클램프 보장.
        """
        h, w = frame.shape[:2]
        mask = self._backend.segment_image(frame, points=[request.point])
        centroid = centroid_of_mask(mask)
        if centroid is None:
            # WHY: 빈 마스크는 클릭이 배경에 맞았을 때 발생한다.
            #      centroid가 없으면 클릭 좌표 자체를 중심으로 대체한다.
            centroid = (float(request.point[0]), float(request.point[1]))
        locked_w, locked_h = apply_aspect_lock(*request.box_size, request.aspect)
        return make_crop_box(centroid, (locked_w, locked_h), (w, h))

    def export(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
        target: tuple[str, ExportConfig],
    ) -> None:
        """크롭 후 path에 config 포맷으로 저장한다.

        target: (path, ExportConfig) — 매개변수 3개 이내 규칙을 위해 튜플로 묶음.
        """
        path, config = target
        cropped = crop_array(frame, box)
        save_image(cropped, path, config)
