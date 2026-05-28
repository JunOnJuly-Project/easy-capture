"""GIF/MP4 비디오 export 순수 로직 (ADR 0011).

경계 불변식: torch·PySide6·PyAV·transformers import 금지.
imageio/imageio-ffmpeg는 함수 내부 지연 import로 격리(ADR 0011).
crop_frames는 순수 함수(부수효과 없음).
encode_frames만 IO 부수효과(파일 쓰기)를 가진다.

위치 결정(ADR 0011): image_export.py(Pillow) 선례를 계승해 core/export에 위치.
imageio는 core 경계 불변식에서 허용되는 순수 인코딩 라이브러리.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from easy_capture.core.export.image_export import crop_array
from easy_capture.core.tracking.gap_policy import GapPolicy
from easy_capture.core.timing.timeremap import (
    SpeedSegment,
    build_playback_schedule,
    clamp_durations_for_gif,
)


@dataclass(frozen=True)
class VideoExportConfig:
    """움짤 내보내기 설정(불변). 이미지 ExportConfig 계승.

    fmt: 'gif' 또는 'mp4' (기본 gif)
    fps: 출력 프레임레이트 (기본 12.0)
    gap_policy: occlusion 갭 처리 정책 (기본 BACKGROUND)
    segments: 구간별 배속 설정 (기본 빈 튜플 = 균일 fps 경로). Story 2.
              WHY 빈 튜플: segments=()이면 기존 단일 fps 경로를 그대로 사용해
              하위 호환성을 보장한다. tuple은 frozenness와 해시 가능성을 모두 충족한다.
    """

    fmt: str = "gif"
    fps: float = 12.0
    gap_policy: GapPolicy = GapPolicy.BACKGROUND
    segments: tuple[SpeedSegment, ...] = ()


def crop_frames(
    frames: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
) -> list[np.ndarray]:
    """프레임별 박스로 슬라이스해 동일 크기 크롭 리스트를 반환한다(순수).

    image_export.crop_array를 프레임 루프로 재사용한다(DRY).
    모든 박스의 크기가 동일해야 한다(GIF/MP4 인코딩 전제).

    Args:
        frames: RGB HxWx3 uint8 프레임 리스트.
        boxes: 프레임별 (x1, y1, x2, y2) 크롭 박스 리스트.

    Returns:
        프레임별 크롭 배열 리스트 (전 원소 동일 크기 보장).

    Raises:
        ValueError: 박스 크기가 서로 다를 때 (GIF/MP4 인코딩 불가).
    """
    _validate_box_sizes(boxes)
    return [crop_array(frame, box) for frame, box in zip(frames, boxes)]


def encode_frames(
    crops: list[np.ndarray],
    path: str,
    config: VideoExportConfig,
) -> None:
    """크롭 프레임 시퀀스를 GIF/MP4로 인코딩해 path에 저장한다(imageio 지연 import).

    전 프레임이 동일 W×H여야 한다. 크기 불일치 시 ValueError.

    Args:
        crops: 동일 크기 RGB HxWx3 uint8 크롭 프레임 리스트.
        path: 출력 파일 경로 (.gif 또는 .mp4).
        config: fps·fmt 설정.

    Raises:
        ValueError: 프레임 크기 불일치.
    """
    _validate_uniform_size(crops)
    if config.fmt == "gif":
        duration_arg = _resolve_gif_durations(crops, config)
        _encode_gif(crops, path, duration_arg)
    else:
        _encode_mp4(crops, path, config.fps)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _validate_box_sizes(boxes: list[tuple[int, int, int, int]]) -> None:
    """박스 크기(W×H)가 모두 동일한지 검증한다.

    WHY: GIF/MP4는 전 프레임이 동일 W×H여야 인코딩된다.
         crop_frames 단계에서 조기 감지해 인코딩 오류를 방지한다.
    """
    if not boxes:
        return
    sizes = {(b[2] - b[0], b[3] - b[1]) for b in boxes}
    if len(sizes) > 1:
        raise ValueError(
            f"박스 크기가 서로 다릅니다: {sizes}. "
            "GIF/MP4 인코딩은 전 프레임 동일 크기여야 합니다."
        )


def _validate_uniform_size(crops: list[np.ndarray]) -> None:
    """크롭 프레임 크기(H×W)가 모두 동일한지 검증한다."""
    if not crops:
        return
    shapes = {c.shape[:2] for c in crops}
    if len(shapes) > 1:
        raise ValueError(
            f"크롭 프레임 크기가 서로 다릅니다: {shapes}. "
            "GIF/MP4 인코딩은 전 프레임 동일 크기여야 합니다."
        )


def _resolve_gif_durations(
    crops: list[np.ndarray],
    config: VideoExportConfig,
) -> float | list[float]:
    """GIF duration 인자를 결정한다 — 균일 float 또는 프레임별 float 리스트.

    segments=() → 기존 균일 duration(1000/fps) 반환 (무회귀).
    segments 있음 → build_playback_schedule + clamp_durations_for_gif 경유 후
    프레임별 duration 리스트 반환.

    WHY 별도 헬퍼: encode_frames 20줄 규칙 준수를 위해 분기 로직을 분리한다.
    WHY list(float) 반환: imageio get_writer(duration=[...]) API가 리스트를 지원한다.
    """
    if not config.segments:
        # 기존 균일 경로 — segments=()일 때 동작 불변
        return 1000.0 / config.fps if config.fps > 0 else 1000.0 / 12.0

    # segments 있음: timeremap 경유 per-frame duration 산출
    schedule = build_playback_schedule(
        len(crops), list(config.segments), config.fps
    )
    clamped, _ = clamp_durations_for_gif(schedule)
    return list(clamped.durations_ms)


def _encode_gif(
    crops: list[np.ndarray],
    path: str,
    duration: float | list[float],
) -> None:
    """imageio로 GIF 파일을 생성한다(지연 import).

    duration: 균일 float(ms) 또는 프레임별 float 리스트(ms).
              segments=()이면 float, segments 있으면 리스트가 전달된다.

    WHY: imageio는 함수 내부에서 import해 core가 imageio 없이도
         import 가능하게 한다(ADR 0011 지연 import 원칙).
    WHY duration 리스트: imageio get_writer(duration=[...]) 는 프레임별
         delay를 각각 설정할 수 있어 GIF VFR(가변 재생속도)를 구현한다.
    """
    import imageio  # 지연 import — core 경계 불변식 유지

    with imageio.get_writer(path, mode="I", duration=duration, loop=0) as writer:
        for crop in crops:
            writer.append_data(crop)


def _encode_mp4(crops: list[np.ndarray], path: str, fps: float) -> None:
    """imageio+imageio-ffmpeg로 MP4 파일을 생성한다(지연 import).

    WHY: macro_block_size=1 — imageio-ffmpeg 기본값(16)이 입력 크기를
         16배수로 강제 리사이즈해 크롭 의도 크기를 왜곡한다.
         to_even이 이미 짝수를 보장하므로 H.264/yuv420p 호환은 유지된다.
         트레이드오프: 극구형 플레이어(매우 드문 케이스)에서 재생 문제 가능.
         ADR 0011 지연 import 원칙: imageio-ffmpeg는 함수 내부에서만 import.
    """
    import imageio  # 지연 import

    with imageio.get_writer(
        path,
        fps=fps,
        format="ffmpeg",
        codec="libx264",
        macro_block_size=1,  # 16배수 강제 리사이즈 방지 — 크롭 의도 크기 보존
    ) as writer:
        for crop in crops:
            writer.append_data(crop)
