"""비디오 모드 캡처 유스케이스 — 슬라이스 핵심 조립자.

슬라이스 흐름:
  load_first_frame → 구간 첫 프레임만 추출 (모델 로드/전파 안 함)
  track           → init_session → add_click → propagate → TrackResult (무거움)
  compute_boxes   → TrackResult.centroids → smooth → 프레임별 make_crop_box (순수·가벼움)
  export          → gap_policy(centroids 기반) → crop_frames → encode_frames (GIF/MP4)

설계 원칙(계획서 §3-2):
  - track: 무거움. propagate는 구간당 딱 1회만 워커에서 호출한다.
  - compute_boxes: 순수·가벼움. backend 절대 미호출.
    종횡비/크기 변경 시 재추적 없이 즉시 재호출.
  - 고정 box size: 전 프레임 동일 W×H (GIF/MP4 인코딩 가능 불변식).
  - export valid_flags: TrackResult.centroids(None 여부)에서 도출.
    WHY: compute_boxes는 occlusion 프레임도 fallback 박스를 항상 채우므로
         box 리스트에서 None을 찾는 방식은 항상 전부 True가 된다(리뷰 [중요] 2).
         centroid None 여부만이 실제 추적 실패(occlusion) 여부를 정확히 나타낸다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from easy_capture.core.crop import (
    apply_aspect_lock,
    centroid_of_mask,
    make_crop_box,
    smooth_centroids,
)
from easy_capture.core.export.video_export import (
    VideoExportConfig,
    crop_frames,
    encode_frames,
)
from easy_capture.core.segmentation.video_backend import (
    EmptyTrackError,
    VideoSegmentationBackend,
)
from easy_capture.core.tracking.gap_policy import build_output_indices
from easy_capture.infra.video_io import FrameSource, FrameSpan

# 타입 힌트 전용 alias
CropBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class TrackResult:
    """추적 1회 성공 결과(불변). 이미지 모드 SegmentResult 계승.

    masks: 프레임별 bool HxW numpy 배열 리스트.
    centroids: 프레임별 centroid — 빈 마스크 프레임은 None.

    WHY: frozen=True로 불변 보장. 캐시된 추적 결과를 실수로 덮어쓰는
         버그를 컴파일 타임에 차단한다. 빈 마스크 프레임은 centroids에
         None으로 표현하고 EmptyTrackError는 전 프레임 빈 마스크 시에만 올린다.
    """

    masks: list[np.ndarray]
    centroids: list[tuple[float, float] | None]


@dataclass(frozen=True)
class VideoCropParams:
    """프레임별 박스 계산 입력 묶음(매개변수 3개 규칙). 이미지 BoxParams 계승.

    box_size: 요청 크롭 (W, H).
    aspect: 종횡비 프리셋 키('1:1', '9:16', '16:9') or None.
    smooth_window: smooth_centroids 이동평균 윈도 (기본 5).

    WHY: frozen=True로 불변. 슬라이더 변경마다 새 인스턴스를 생성하고
         이전 값은 버린다 — 실수 변경 방지.
    """

    box_size: tuple[int, int]
    aspect: str | None
    smooth_window: int = 5


class VideoCaptureUseCase:
    """비디오 모드 캡처 유스케이스.

    source와 backend는 Protocol 타입으로 주입받는다(DIP).
    구체 구현(SAM2 video, PyAV 등)은 진입점(router)에서 결정한다.
    """

    def __init__(
        self, source: FrameSource, backend: VideoSegmentationBackend
    ) -> None:
        """프레임 공급원·비디오 세그 백엔드 주입."""
        self._source = source
        self._backend = backend

    def probe_meta(self):
        """소스 메타 정보를 반환한다(UI가 공개 API만 사용하도록 위임).

        WHY: UI가 self._usecase._source.probe()처럼 비공개 필드를 관통하는
             대신 이 메서드를 호출해 캡슐화를 지킨다(리뷰 [중요] 3).
        """
        return self._source.probe()

    def read_span_frames(self, span: FrameSpan) -> list[np.ndarray]:
        """구간 프레임 시퀀스를 추출해 반환한다(UI 공개 API).

        WHY: UI가 self._usecase._source.read_frames(span)처럼 비공개 필드를
             관통하는 대신 이 메서드를 호출해 캡슐화를 지킨다(리뷰 [중요] 3).
        """
        return self._source.read_frames(span)

    def load_first_frame(self, span: FrameSpan) -> np.ndarray:
        """구간 첫 프레임만 추출한다(모델 로드·전파 안 함, 가벼움).

        WHY: 파일 열기 직후 즉시 캔버스에 표시하기 위해
             무거운 모델 로드와 분리한다(지연 로드 전략, ADR 0007).
        """
        first_span = FrameSpan(start=span.start, end=span.start + 1, step=1)
        frames = self._source.read_frames(first_span)
        return frames[0] if frames else self._source.read_frame(span.start)

    def track(
        self, frames: list[np.ndarray], point: tuple[int, int]
    ) -> TrackResult:
        """init_session → add_click → propagate → 프레임별 centroid(무거움).

        구간당 전파(propagate)는 딱 1회만 호출한다.
        전 프레임 빈 마스크면 EmptyTrackError(한국어 메시지).

        Args:
            frames: 구간 전체 RGB 프레임 리스트.
            point: 첫 프레임 클릭 좌표 (x, y).

        Returns:
            TrackResult(masks, centroids).
        """
        session = self._backend.init_session(frames)
        self._backend.add_click(session, point)
        masks = self._backend.propagate(session)
        centroids = [centroid_of_mask(m) for m in masks]
        _raise_if_all_empty(centroids)
        return TrackResult(masks=masks, centroids=centroids)

    def compute_boxes(
        self,
        result: TrackResult,
        params: VideoCropParams,
        frame_size: tuple[int, int],
    ) -> list[CropBox]:
        """centroids → smooth → 프레임별 고정 크기 make_crop_box (순수·가벼움).

        backend를 절대 호출하지 않는다. 종횡비/크기/smooth_window 변경 시
        재추적 없이 즉시 재호출해도 멈춤이 없다.

        고정 box size 불변식: 모든 프레임 박스가 동일 W×H가 되도록
        apply_aspect_lock 결과를 한 번만 계산해 전 프레임에 적용한다.

        Args:
            result: track()이 반환한 TrackResult.
            params: 박스 크기·종횡비·smooth_window 설정.
            frame_size: (W, H) 프레임 크기 (경계 클램프용).

        Returns:
            프레임별 (x1, y1, x2, y2) 크롭 박스 리스트 (전 원소 동일 크기).
        """
        smoothed = smooth_centroids(result.centroids, params.smooth_window)
        # WHY: apply_aspect_lock을 전 프레임 공통으로 1회만 계산해
        #      고정 box size 불변식을 보장한다.
        fixed_size = apply_aspect_lock(*params.box_size, params.aspect)
        return [
            make_crop_box(c or _fallback_center(frame_size), fixed_size, frame_size)
            for c in smoothed
        ]

    def export(
        self,
        frames: list[np.ndarray],
        boxes: list[CropBox],
        target: tuple[str, VideoExportConfig],
        result: TrackResult | None = None,
    ) -> None:
        """gap_policy → 프레임 선택 → crop_frames → encode_frames.

        valid_flags는 TrackResult.centroids(None 여부)에서 도출한다.
        WHY: compute_boxes는 occlusion 프레임도 fallback 박스를 항상 채우므로
             box 리스트에서 None을 찾으면 항상 전부 True가 된다. centroid None
             여부만이 실제 추적 실패 구간을 정확히 나타낸다(리뷰 [중요] 2).
             result=None이면 모든 프레임 유효(하위호환 폴백).

        Args:
            frames: 원본 구간 프레임 리스트.
            boxes: compute_boxes 결과 박스 리스트.
            target: (출력 경로, VideoExportConfig) 튜플.
            result: track()이 반환한 TrackResult (gap_policy 적용용).
        """
        path, config = target
        valid_flags = _valid_flags_from_result(result, len(frames))
        indices = build_output_indices(valid_flags, config.gap_policy)
        selected_frames = [frames[i] for i in indices]
        selected_boxes = [boxes[i] for i in indices]
        crops = crop_frames(selected_frames, selected_boxes)
        encode_frames(crops, path, config)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _valid_flags_from_result(
    result: TrackResult | None, n_frames: int
) -> list[bool]:
    """TrackResult.centroids에서 valid_flags 리스트를 도출한다.

    centroid가 None이 아닌 프레임만 True(추적 성공).
    result가 None이면 전부 True 반환(하위호환 폴백).

    WHY: compute_boxes는 occlusion 프레임에도 fallback 박스를 채워 box가
         절대 None이 되지 않는다. 따라서 box 존재 여부로는 gap을 감지할 수 없고,
         centroid None 여부만이 실제 occlusion 구간을 정확히 표현한다.
    """
    if result is None:
        return [True] * n_frames
    return [c is not None for c in result.centroids]


def _raise_if_all_empty(centroids: list) -> None:
    """전 프레임 centroid가 None이면 EmptyTrackError를 발생시킨다.

    WHY: 조용한 폴백 없이 명시적 예외로 알린다. UI 워커가 잡아
         한국어 안내 메시지를 표시한다(이미지 모드 EmptyMaskError 계승).
    """
    if all(c is None for c in centroids):
        raise EmptyTrackError(
            "전 프레임에서 대상을 추적하지 못했습니다. "
            "다른 클릭 포인트로 다시 시도해 주세요."
        )


def _fallback_center(frame_size: tuple[int, int]) -> tuple[float, float]:
    """centroid가 None(occlusion)일 때 프레임 중앙을 폴백 중심으로 반환한다.

    WHY: smooth_centroids._hold_forward가 직전 위치를 홀드하므로
         실제로 이 분기는 첫 프레임만 None인 극단적 케이스에서만 발생한다.
    """
    w, h = frame_size
    return float(w / 2), float(h / 2)
