"""타임리맵 순수 로직 — 구간별 가변 재생속도 계산.

WHY 독립 모듈:
  인코딩(imageio·av)·UI(PySide6)·추론(torch)과 책임이 직교한다.
  "어떤 프레임을 시간축에서 어떻게 배열할 것인가"는 순수 도메인 로직이므로
  numpy·stdlib만으로 결정적 단위 테스트가 가능해야 한다.
  ADR 0013 참조.

의존: stdlib + (선택) numpy — imageio·torch·PySide6·av import 금지.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

# ---------------------------------------------------------------------------
# 상수 — 매직넘버 금지
# ---------------------------------------------------------------------------
# 배속 허용 범위 (MVP 결정1·결정2)
FACTOR_MIN: float = 0.25
FACTOR_MAX: float = 4.0

# 기준 표시시간 단위 (ms)
MS_PER_SEC: float = 1000.0

# GIF per-frame duration 하한 및 클램프 목표 (결정4)
GIF_DURATION_MIN_MS: float = 10.0
GIF_DURATION_CLAMP_MS: float = 20.0


# ---------------------------------------------------------------------------
# SpeedSegment — 불변 값객체
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpeedSegment:
    """재생 속도 변경 구간 값객체(불변).

    start: 구간 시작 프레임 인덱스 (포함, 배타 끝 방식 [start, end))
    end:   구간 끝 프레임 인덱스 (미포함)
    factor: 배속 (0.25 ~ 4.0). 0.5 = 슬로우모션 2배, 2.0 = 패스트 2배.

    WHY frozen=True: 값객체이므로 생성 후 불변이 보장되어야 한다.
    해시 가능하므로 dict key·frozenset에 사용 가능.
    """

    start: int
    end: int
    factor: float


# ---------------------------------------------------------------------------
# normalize_segments — 정렬·검증·겹침 금지
# ---------------------------------------------------------------------------
def normalize_segments(segments: list[SpeedSegment]) -> list[SpeedSegment]:
    """구간 리스트를 검증하고 start 기준 정렬하여 반환(순수).

    빈 리스트 → 빈 리스트 반환(항등).
    검증 실패 시 한국어 메시지 ValueError 발생.

    WHY 정렬 선행: 겹침 검사는 O(n) 인접 비교이므로 정렬이 전제 조건이다.
    WHY 겹침 금지: MVP 범위에서 우선순위 병합 규칙을 도입하지 않는다(KISS).
    """
    if not segments:
        return []

    _validate_each_segment(segments)

    sorted_segs = sorted(segments, key=lambda s: s.start)
    _validate_no_overlap(sorted_segs)

    return sorted_segs


def _validate_each_segment(segments: list[SpeedSegment]) -> None:
    """개별 구간의 역전·빈 구간·factor 범위를 검증한다."""
    for seg in segments:
        _check_segment_direction(seg)
        _check_segment_factor(seg)


def _check_segment_direction(seg: SpeedSegment) -> None:
    """start < end 역전 금지 검증."""
    if seg.start >= seg.end:
        raise ValueError(
            f"구간 역전 또는 빈 구간: start={seg.start} >= end={seg.end}. "
            "시작 인덱스는 끝 인덱스보다 작아야 합니다."
        )


def _check_segment_factor(seg: SpeedSegment) -> None:
    """배속 factor 범위(0.25 ~ 4.0) 검증."""
    if seg.factor <= 0 or seg.factor < FACTOR_MIN or seg.factor > FACTOR_MAX:
        raise ValueError(
            f"배속 factor={seg.factor}이(가) 허용 범위({FACTOR_MIN}~{FACTOR_MAX}) 밖입니다."
        )


def _validate_no_overlap(sorted_segs: list[SpeedSegment]) -> None:
    """정렬된 구간 리스트에서 겹침을 검사한다."""
    for i in range(len(sorted_segs) - 1):
        curr = sorted_segs[i]
        nxt = sorted_segs[i + 1]
        if curr.end > nxt.start:
            raise ValueError(
                f"구간 겹침: [{curr.start},{curr.end})와 [{nxt.start},{nxt.end}) "
                "가 중복됩니다. 겹치는 구간은 허용되지 않습니다."
            )


# ---------------------------------------------------------------------------
# PlaybackSchedule — 타임리맵 결과 중간 표현(불변)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlaybackSchedule:
    """타임리맵 결과 중간 표현(불변). GIF·MP4 양쪽 산출 가능.

    frame_indices:    출력할 프레임 인덱스 시퀀스.
                      슬로우=복제로 같은 인덱스 반복, 패스트=일부 인덱스 생략.
    durations_ms:     frame_indices와 1:1 대응하는 프레임별 표시시간(ms).
    base_fps:         기준 fps(등속 1프레임 시간 = 1000/base_fps ms).
                      schedule_to_cfr_indices에서 복제/드롭 비율 계산 기준.
                      기본값 30.0은 일반 촬영 기준.

    WHY frozen=True: 인코더가 소비하는 IR이므로 불변이어야 한다.
    WHY 이중 표현: GIF는 durations_ms로 VFR 표현, MP4는 schedule_to_cfr_indices로 변환.
    WHY base_fps 포함: schedule_to_cfr_indices가 PlaybackSchedule만으로
      복제/드롭 비율을 결정할 수 있어야 한다(인터페이스 최소화 원칙).
    ADR 0013 결정2 참조.
    """

    frame_indices: list[int]
    durations_ms: list[float]
    base_fps: float = 30.0


# ---------------------------------------------------------------------------
# build_playback_schedule — 핵심 순수 함수
# ---------------------------------------------------------------------------
def build_playback_schedule(
    n_frames: int,
    segments: list[SpeedSegment],
    base_fps: float,
) -> PlaybackSchedule:
    """프레임 수 + 구간 배속 + 기준 fps → 재생 스케줄(순수).

    segments=[] → 항등: frame_indices=range(n), durations=균일(1000/base_fps).
    구간 밖 프레임은 factor=1.0(등속)으로 처리한다.

    WHY 순수 함수: 인코딩·UI 비의존으로 단위 테스트 가능. 90%+ 커버리지 가드.
    """
    if n_frames == 0:
        return PlaybackSchedule(frame_indices=[], durations_ms=[])

    base_duration = MS_PER_SEC / base_fps
    normalized = normalize_segments(segments)
    factor_map = _build_factor_map(normalized, n_frames)

    durations = [base_duration / factor_map[i] for i in range(n_frames)]
    return PlaybackSchedule(
        frame_indices=list(range(n_frames)),
        durations_ms=durations,
        base_fps=base_fps,
    )


def _build_factor_map(
    sorted_segs: list[SpeedSegment],
    n_frames: int,
) -> list[float]:
    """각 프레임 인덱스에 대응하는 factor 배열을 생성한다.

    구간 밖은 factor=1.0(등속).
    WHY 배열 방식: O(n) 순회로 모든 프레임 factor를 결정 — 구간 수에 무관하게 선형.
    """
    factors = [1.0] * n_frames
    for seg in sorted_segs:
        end = min(seg.end, n_frames)
        for i in range(seg.start, end):
            factors[i] = seg.factor
    return factors


# ---------------------------------------------------------------------------
# schedule_to_cfr_indices — MP4 CFR 변환
# ---------------------------------------------------------------------------
def schedule_to_cfr_indices(schedule: PlaybackSchedule) -> list[int]:
    """MP4 CFR 경로용: 슬로우=프레임 복제, 패스트=등간격 드롭 인덱스 시퀀스.

    각 프레임의 표시시간을 기준 duration(= 1000/base_fps ms)으로 나눈 비율로
    복제 수를 결정한다. Bresenham 누적기로 ±1 오차 내 정확한 길이를 보장한다.

    기준 duration: schedule.base_fps에서 산출한 등속 1프레임 시간.
    WHY base_fps 사용: 슬로우 구간만 있어도 "몇 배 복제"인지 올바르게 판단한다.
      min(durations) 기준이면 모두 슬로우일 때 ratio=1.0이 되어 복제 불가.

    WHY Bresenham 방식: 단순 반올림은 누적 오차가 생긴다.
    시간 축을 정수 단계로 변환하는 전형적인 line-drawing 알고리즘 응용.
    """
    if not schedule.frame_indices:
        return []

    result: list[int] = []
    accumulator: float = 0.0
    base_dur = MS_PER_SEC / schedule.base_fps  # 등속 1프레임 기준 시간

    for fi, dur in zip(schedule.frame_indices, schedule.durations_ms):
        accumulator += dur / base_dur
        count = int(accumulator)
        for _ in range(count):
            result.append(fi)
        accumulator -= count

    return result


# ---------------------------------------------------------------------------
# clamp_durations_for_gif — GIF 10ms 하한 가드
# ---------------------------------------------------------------------------
def clamp_durations_for_gif(
    schedule: PlaybackSchedule,
) -> tuple[PlaybackSchedule, list[int]]:
    """GIF per-frame duration < 10ms 를 20ms로 클램프(순수).

    WHY 20ms 클램프:
      GIF delay는 centisecond 단위로 양자화된다.
      10ms 미만이면 인코더가 delay=0으로 저장하고,
      대부분 뷰어는 delay=0을 "지정 없음"으로 해석해 기본 ~100ms로 강제 적용한다.
      빠르게 만들려던 구간이 오히려 느려지는 역전을 방지한다.
      ADR 0013 결정4 참조.

    Returns:
        (클램프된 스케줄, 클램프 적용된 frame_indices 목록)
        인덱스 목록은 UI·노트북 경고 표시용.
    """
    if not schedule.durations_ms:
        return PlaybackSchedule(frame_indices=[], durations_ms=[]), []

    new_durations: list[float] = []
    clamped_indices: list[int] = []

    for fi, dur in zip(schedule.frame_indices, schedule.durations_ms):
        if dur < GIF_DURATION_MIN_MS:
            new_durations.append(GIF_DURATION_CLAMP_MS)
            clamped_indices.append(fi)
        else:
            new_durations.append(dur)

    clamped_schedule = PlaybackSchedule(
        frame_indices=schedule.frame_indices,
        durations_ms=new_durations,
    )
    return clamped_schedule, clamped_indices
