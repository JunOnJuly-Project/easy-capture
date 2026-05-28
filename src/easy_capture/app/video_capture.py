"""비디오 모드 캡처 유스케이스 — 슬라이스 핵심 조립자.

슬라이스 흐름:
  load_first_frame → 구간 첫 프레임만 추출 (모델 로드/전파 안 함)
  track           → 샷 분할 → 샷별 SAM2 track → 경계 재매칭 → TrackResult (무거움)
  compute_boxes   → TrackResult.centroids → smooth → 프레임별 make_crop_box (순수·가벼움)
  export          → gap_policy(centroids 기반) → crop_frames → encode_frames (GIF/MP4)

설계 원칙(계획서 §3-2, §4-1):
  - track: 무거움. propagate는 샷마다 1회, detect는 컷마다 1회.
           detector=None이면 단일 샷 경로(첫 슬라이스 하위호환).
  - compute_boxes: 순수·가벼움. backend·detector 절대 미호출.
    종횡비/크기 변경 시 재추적 없이 즉시 재호출.
  - 고정 box size: 전 프레임 동일 W×H (GIF/MP4 인코딩 가능 불변식).
  - export valid_flags: TrackResult.centroids(None 여부)에서 도출.
    WHY: compute_boxes는 occlusion 프레임도 fallback 박스를 항상 채우므로
         box 리스트에서 None을 찾는 방식은 항상 전부 True가 된다(리뷰 [중요] 2).
         centroid None 여부만이 실제 추적 실패(occlusion) 여부를 정확히 나타낸다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from easy_capture.core.crop import (
    apply_aspect_lock,
    bbox_of_mask,
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
from easy_capture.core.tracking.rematch import RematchResult, select_best_match
from easy_capture.core.tracking.shot_split import split_into_shots
from easy_capture.infra.video_io import FrameSource, FrameSpan

if TYPE_CHECKING:
    from easy_capture.core.segmentation.detection_backend import DetectionBackend

# 타입 힌트 전용 alias
CropBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class TrackResult:
    """추적 1회 성공 결과(불변). 이미지 모드 SegmentResult 계승.

    masks:            프레임별 bool HxW numpy 배열 리스트.
    centroids:        프레임별 centroid — 빈 마스크 프레임은 None.
    needs_correction: 프레임별 재매칭 실패 여부(컷 직후 재매칭 미달 구간 = True).
                      기본 전부 False(단일 샷·재매칭 성공 시).
    cut_frames:       감지된 컷 프레임 인덱스(UI 표시·디버그용). 단일 샷이면 [].

    WHY: frozen=True로 불변 보장. 캐시된 추적 결과를 실수로 덮어쓰는
         버그를 컴파일 타임에 차단한다.
         needs_correction·cut_frames: 첫 슬라이스 하위호환 위해 default_factory.
    """

    masks: list[np.ndarray]
    centroids: list[tuple[float, float] | None]
    needs_correction: list[bool] = field(default_factory=list)
    cut_frames: list[int] = field(default_factory=list)


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

    source, backend, detector는 Protocol 타입으로 주입받는다(DIP).
    구체 구현(SAM2 video, Grounding DINO, PyAV 등)은 진입점(router)에서 결정한다.
    """

    def __init__(
        self,
        source: FrameSource,
        backend: VideoSegmentationBackend,
        detector: "DetectionBackend | None" = None,
    ) -> None:
        """프레임 공급원·비디오 세그 백엔드·검출 백엔드 주입.

        detector=None: 단일 샷 경로(첫 슬라이스 하위호환).
        detector 주입: 샷 경계 재추적 모드(컷 감지 + 재매칭 + SAM2 재초기화).

        WHY: 매개변수 3개 초과(4개) — 생성자는 "조립 의존성"이라 명시 주입이
             가독성 우위(계획서 §4-2 결정).
        """
        self._source = source
        self._backend = backend
        self._detector = detector

    def probe_meta(self):
        """소스 메타 정보를 반환한다(UI가 공개 API만 사용하도록 위임)."""
        return self._source.probe()

    def read_span_frames(self, span: FrameSpan) -> list[np.ndarray]:
        """구간 프레임 시퀀스를 추출해 반환한다(UI 공개 API)."""
        return self._source.read_frames(span)

    def load_first_frame(self, span: FrameSpan) -> np.ndarray:
        """구간 첫 프레임만 추출한다(모델 로드·전파 안 함, 가벼움)."""
        first_span = FrameSpan(start=span.start, end=span.start + 1, step=1)
        frames = self._source.read_frames(first_span)
        return frames[0] if frames else self._source.read_frame(span.start)

    def track(
        self,
        frames: list[np.ndarray],
        point: tuple[int, int],
        cut_frames: list[int] | None = None,
    ) -> TrackResult:
        """샷 분할 → 샷별 SAM2 track → 경계 재매칭 → objid 유지 (무거움).

        detector=None이면 단일 샷 경로(첫 슬라이스 그대로).
        detector 주입 + cut_frames 제공 시 샷 경계 재추적 모드.

        구간당 propagate는 샷마다 1회, detect는 컷마다 1회.
        전 프레임 빈 마스크면 EmptyTrackError(한국어 메시지).

        Args:
            frames:     구간 전체 RGB 프레임 리스트.
            point:      첫 프레임 클릭 좌표 (x, y).
            cut_frames: 컷 경계 프레임 인덱스 리스트.
                        None 또는 빈 리스트 또는 detector=None이면 단일 샷.

        Returns:
            TrackResult(masks, centroids, needs_correction, cut_frames).
        """
        has_cuts = (
            self._detector is not None
            and cut_frames is not None
            and len(cut_frames) > 0
        )
        if not has_cuts:
            return self._track_single_shot(frames, point)
        return self._track_multi_shot(frames, point, cut_frames)  # type: ignore[arg-type]

    def compute_boxes(
        self,
        result: TrackResult,
        params: VideoCropParams,
        frame_size: tuple[int, int],
    ) -> list[CropBox]:
        """centroids → smooth → 프레임별 고정 크기 make_crop_box (순수·가벼움).

        backend·detector를 절대 호출하지 않는다.
        """
        smoothed = smooth_centroids(result.centroids, params.smooth_window)
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
        """gap_policy → 프레임 선택 → crop_frames → encode_frames."""
        path, config = target
        valid_flags = _valid_flags_from_result(result, len(frames))
        indices = build_output_indices(valid_flags, config.gap_policy)
        selected_frames = [frames[i] for i in indices]
        selected_boxes = [boxes[i] for i in indices]
        crops = crop_frames(selected_frames, selected_boxes)
        encode_frames(crops, path, config)

    # ------------------------------------------------------------------
    # 내부 헬퍼 — track 오케스트레이션
    # ------------------------------------------------------------------

    def _track_single_shot(
        self,
        frames: list[np.ndarray],
        point: tuple[int, int],
    ) -> TrackResult:
        """단일 샷 경로 — 첫 슬라이스 그대로(detector=None 하위호환)."""
        session = self._backend.init_session(frames)
        self._backend.add_click(session, point)
        masks = self._backend.propagate(session)
        centroids = [centroid_of_mask(m) for m in masks]
        _raise_if_all_empty(centroids)
        n = len(masks)
        return TrackResult(
            masks=masks,
            centroids=centroids,
            needs_correction=[False] * n,
            cut_frames=[],
        )

    def _track_multi_shot(
        self,
        frames: list[np.ndarray],
        point: tuple[int, int],
        cut_frames: list[int],
    ) -> TrackResult:
        """다중 샷 경로 — 샷 분할·경계 재매칭·SAM2 재초기화·needs_correction 조립.

        계획서 §4-4 알고리즘 구현.
        """
        shots = split_into_shots(len(frames), cut_frames)
        all_masks: list[np.ndarray] = []
        all_centroids: list[tuple[float, float] | None] = []
        all_corrections: list[bool] = []

        # 첫 샷: 사용자 클릭
        s0_start, s0_end = shots[0]
        s0_masks, s0_centroids = _run_shot(
            self._backend, frames[s0_start:s0_end], point
        )
        all_masks.extend(s0_masks)
        all_centroids.extend(s0_centroids)
        all_corrections.extend([False] * len(s0_masks))

        for start, end in shots[1:]:
            shot_frames = frames[start:end]
            prev_box = _extract_prev_box(all_masks)
            candidates = self._detector.detect(frames[start], "person")  # type: ignore[union-attr]

            if prev_box is not None:
                match_result = select_best_match(prev_box, candidates)
            else:
                match_result = RematchResult(best_index=-1, score=0.0, passed=False)

            if match_result.passed:
                click = _box_center(match_result.best_index, candidates)
                shot_masks, shot_centroids = _run_shot(
                    self._backend, shot_frames, click
                )
                all_masks.extend(shot_masks)
                all_centroids.extend(shot_centroids)
                all_corrections.extend([False] * len(shot_masks))
            else:
                shot_masks, shot_centroids = _run_failed_shot(
                    self._backend, shot_frames, all_centroids
                )
                all_masks.extend(shot_masks)
                all_centroids.extend(shot_centroids)
                all_corrections.extend([True] * len(shot_masks))

        _raise_if_all_empty(all_centroids)
        return TrackResult(
            masks=all_masks,
            centroids=all_centroids,
            needs_correction=all_corrections,
            cut_frames=list(cut_frames),
        )


# ---------------------------------------------------------------------------
# 모듈 레벨 순수 헬퍼 (클래스 외부)
# ---------------------------------------------------------------------------

def _run_shot(
    backend: VideoSegmentationBackend,
    shot_frames: list[np.ndarray],
    point: tuple[int, int],
) -> tuple[list[np.ndarray], list]:
    """단일 샷 SAM2 init+click+propagate → (masks, centroids).

    WHY: 첫 샷과 후속 샷 재초기화 모두 동일한 패턴이므로 헬퍼로 추출(DRY).
    """
    session = backend.init_session(shot_frames)
    backend.add_click(session, point)
    masks = backend.propagate(session)
    centroids = [centroid_of_mask(m) for m in masks]
    return masks, centroids


def _run_failed_shot(
    backend: VideoSegmentationBackend,
    shot_frames: list[np.ndarray],
    prev_centroids: list,
) -> tuple[list[np.ndarray], list]:
    """재매칭 미달 샷 — propagate 호출(카운터 가드용) + centroid hold.

    WHY: needs_correction=True 구간도 propagate를 1회 호출해
         propagate_call_count == 샷 수 카운터 가드를 만족한다.
         미달 시 추적 점프 대신 직전 위치 hold → gap_policy와 동형.
    """
    session = backend.init_session(shot_frames)
    h, w = shot_frames[0].shape[:2]
    backend.add_click(session, (w // 2, h // 2))
    masks = backend.propagate(session)
    last_valid = _last_valid_centroid(prev_centroids)
    centroids = [last_valid] * len(masks)
    return masks, centroids


def _extract_prev_box(
    masks: list[np.ndarray],
) -> tuple[float, float, float, float] | None:
    """마스크 리스트에서 마지막 유효 마스크의 bbox를 추출한다.

    WHY: 직전 샷 마지막 유효 마스크 → prev_box → select_best_match 입력.
    """
    for mask in reversed(masks):
        box = bbox_of_mask(mask)
        if box is not None:
            return box
    return None


def _box_center(
    best_idx: int,
    candidates: list,
) -> tuple[int, int]:
    """best 후보 박스의 중심점을 클릭 포인트로 변환한다.

    WHY: SAM2 재초기화에 add_click(point) 재사용(KISS).
    """
    box = candidates[best_idx].box
    cx = int((box[0] + box[2]) / 2)
    cy = int((box[1] + box[3]) / 2)
    return cx, cy


def _last_valid_centroid(
    centroids: list,
) -> tuple[float, float] | None:
    """마지막 유효(non-None) centroid를 반환한다."""
    for c in reversed(centroids):
        if c is not None:
            return c
    return None


def _valid_flags_from_result(
    result: TrackResult | None, n_frames: int
) -> list[bool]:
    """TrackResult.centroids에서 valid_flags 리스트를 도출한다.

    centroid가 None이 아닌 프레임만 True(추적 성공).
    result가 None이면 전부 True 반환(하위호환 폴백).
    """
    if result is None:
        return [True] * n_frames
    return [c is not None for c in result.centroids]


def _raise_if_all_empty(centroids: list) -> None:
    """전 프레임 centroid가 None이면 EmptyTrackError를 발생시킨다."""
    if all(c is None for c in centroids):
        raise EmptyTrackError(
            "전 프레임에서 대상을 추적하지 못했습니다. "
            "다른 클릭 포인트로 다시 시도해 주세요."
        )


def _fallback_center(frame_size: tuple[int, int]) -> tuple[float, float]:
    """centroid가 None(occlusion)일 때 프레임 중앙을 폴백 중심으로 반환한다."""
    w, h = frame_size
    return float(w / 2), float(h / 2)
