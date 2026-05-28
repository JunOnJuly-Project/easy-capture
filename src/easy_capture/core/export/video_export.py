"""GIF/MP4 비디오 export 순수 로직 (ADR 0011).

경계 불변식: torch·PySide6·PyAV·transformers import 금지.
imageio/imageio-ffmpeg는 함수 내부 지연 import로 격리(ADR 0011).
crop_frames는 순수 함수(부수효과 없음).
encode_frames만 IO 부수효과(파일 쓰기)를 가진다.

위치 결정(ADR 0011): image_export.py(Pillow) 선례를 계승해 core/export에 위치.
imageio는 core 경계 불변식에서 허용되는 순수 인코딩 라이브러리.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# ---------------------------------------------------------------------------
# 모듈 상수 — 매직넘버 금지
# ---------------------------------------------------------------------------
# VideoExportConfig.fps 기본값과 동일 의미. fps<=0 폴백 경로에서도 재사용(DRY).
_DEFAULT_GIF_FPS: float = 12.0

from easy_capture.core.export.image_export import crop_array
from easy_capture.core.tracking.gap_policy import GapPolicy
from easy_capture.core.timing.timeremap import (
    SpeedSegment,
    TrimRange,
    build_playback_schedule,
    clamp_durations_for_gif,
    schedule_to_cfr_indices,
    shift_segments_into_trim,
    slice_for_trim,
    validate_trim,
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
    trim: 출력 트림 구간 (기본 None = 트림 없음, 전체 출력). Story 4.
          WHY None: trim=None이면 슬라이스·평행이동 없이 기존 동작과 동일하다(무회귀).
    loop_count: GIF 반복 횟수 (기본 0 = 무한 루프). Story 4.
          WHY 0: loop_count=0은 기존 _encode_gif loop=0(GIF 무한 반복) 계약과 동일하다.
          MP4는 컨테이너 루프 미지원이라 loop_count를 조용히 무시한다.
    """

    fmt: str = "gif"
    fps: float = _DEFAULT_GIF_FPS
    gap_policy: GapPolicy = GapPolicy.BACKGROUND
    segments: tuple[SpeedSegment, ...] = ()
    trim: TrimRange | None = None
    loop_count: int = 0


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

    WHY 좌표계 한정(BACKGROUND 전제):
      config.trim·segments는 이 함수에 전달된 crops 시퀀스 상대 [0, n) 좌표다.
      gap_policy=BACKGROUND에서는 crops가 span 전체이므로 span 상대와 일치한다.
      CUT/FREEZE에서는 build_output_indices가 갭을 제거해 crops가 압축되므로
      trim/segments 좌표가 span 상대와 어긋난다. 현재 슬라이스는 BACKGROUND 전제로만
      정합을 보장한다(잠복 버그 — ADR 0013의 2단계 인덱싱 미구현 추적).

    Args:
        crops: 동일 크기 RGB HxWx3 uint8 크롭 프레임 리스트.
        path: 출력 파일 경로 (.gif 또는 .mp4).
        config: fps·fmt·trim·loop_count 설정.

    Raises:
        ValueError: 프레임 크기 불일치, 트림 범위 오류, loop_count 음수.
    """
    _validate_uniform_size(crops)
    validate_trim(config.trim, len(crops))
    _validate_loop_count(config.loop_count)
    crops = slice_for_trim(crops, config.trim)
    local = _with_trim_local_segments(config)
    if local.fmt == "gif":
        duration_arg = _resolve_gif_durations(crops, local)
        _encode_gif(crops, path, (duration_arg, local.loop_count))
    else:
        remapped = _resolve_mp4_frames(crops, local)
        _encode_mp4(remapped, path, local.fps)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _with_trim_local_segments(config: VideoExportConfig) -> VideoExportConfig:
    """segments를 트림-로컬 좌표로 교체한 config 사본을 반환한다.

    WHY: 트림을 먼저 적용한 crops 위에서 segments가 동작하려면, segments도
         트림-로컬 [0, M) 좌표로 평행이동해야 한다(planner 적용 순서).
         trim=None이면 shift_segments_into_trim이 항등 → segments 불변(무회귀).
    """
    return replace(
        config, segments=shift_segments_into_trim(config.segments, config.trim)
    )


def _validate_loop_count(loop_count: int) -> None:
    """GIF 반복 횟수가 유효한지 검증한다(순수).

    0 이상이어야 한다(0=무한, 양수=N회). 음수는 GIF 스펙에 없는 무효값.
    WHY: validate_trim과 대칭 위치에 두어 인코딩 전 입력 계약을 일관되게 지킨다.
         MP4는 loop_count를 무시하지만 잘못된 값이 조용히 통과하면 혼란을 준다.
    """
    if loop_count < 0:
        raise ValueError(
            f"GIF 반복 횟수는 0 이상이어야 합니다(0=무한): {loop_count}"
        )


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
        fps = config.fps if config.fps > 0 else _DEFAULT_GIF_FPS
        return 1000.0 / fps

    # segments 있음: timeremap 경유 per-frame duration 산출
    schedule = build_playback_schedule(
        len(crops), list(config.segments), config.fps
    )
    # WHY 폐기: GIF 클램프 경고 채널은 후속 Task(UI/노트북 경고)에서 별도 산출
    clamped, _ = clamp_durations_for_gif(schedule)
    return list(clamped.durations_ms)


def _resolve_mp4_frames(
    crops: list[np.ndarray],
    config: VideoExportConfig,
) -> list[np.ndarray]:
    """MP4용 프레임 시퀀스를 결정한다 — segments 있으면 복제/드롭, 없으면 원본 반환.

    segments=() → crops 그대로 반환 (무회귀).
    segments 있음 → build_playback_schedule + schedule_to_cfr_indices 경유 후
    복제/드롭된 프레임 리스트 반환.

    WHY 별도 헬퍼: encode_frames 20줄 규칙 + 단일 책임 원칙 준수.
                  GIF의 _resolve_gif_durations와 대칭 구조로 일관성 유지.
    WHY MP4 CFR 방식: MP4는 GIF와 달리 프레임별 delay 지정 불가.
                     속도 변화는 프레임 복제(슬로우)·드롭(패스트)으로 표현한다.
    """
    if not config.segments:
        # segments=() 경로: 기존 동작 불변 보장
        return crops

    schedule = build_playback_schedule(
        len(crops), list(config.segments), config.fps
    )
    cfr_indices = schedule_to_cfr_indices(schedule)
    return [crops[i] for i in cfr_indices]


def _encode_gif(
    crops: list[np.ndarray],
    path: str,
    gif_spec: tuple[float | list[float], int],
) -> None:
    """imageio로 GIF 파일을 생성한다(지연 import).

    gif_spec: (duration, loop) 튜플.
        duration — 균일 float(ms) 또는 프레임별 float 리스트(ms).
                   segments=()이면 float, segments 있으면 리스트가 전달된다.
        loop     — GIF 반복 횟수(0=무한). config.loop_count에서 전달된다.

    WHY gif_spec 튜플: duration·loop를 묶어 매개변수 3개 규칙을 지킨다.
         loop 인자를 따로 받으면 매개변수가 4개가 되어 CLAUDE.md 규칙에 위반된다.
    WHY 지연 import: imageio를 함수 내부에서 import해 core가 imageio 없이도
         import 가능하게 한다(ADR 0011 지연 import 원칙).
    WHY duration 리스트: imageio get_writer(duration=[...]) 는 프레임별
         delay를 각각 설정할 수 있어 GIF VFR(가변 재생속도)를 구현한다.
    """
    import imageio  # 지연 import — core 경계 불변식 유지

    duration, loop = gif_spec
    with imageio.get_writer(path, mode="I", duration=duration, loop=loop) as writer:
        for crop in crops:
            writer.append_data(crop)


def _encode_mp4(crops: list[np.ndarray], path: str, fps: float) -> None:
    """imageio+imageio-ffmpeg로 MP4 파일을 생성한다(지연 import).

    WHY loop_count 미사용: MP4 컨테이너는 GIF NETSCAPE2.0 같은 루프 메타를
         지원하지 않으므로 config.loop_count를 조용히 무시한다(재생기 책임).
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
