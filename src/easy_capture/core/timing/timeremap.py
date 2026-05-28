"""타임리맵 순수 로직 — 구간별 가변 재생속도 계산.

WHY 독립 모듈:
  인코딩(imageio·av)·UI(PySide6)·추론(torch)과 책임이 직교한다.
  "어떤 프레임을 시간축에서 어떻게 배열할 것인가"는 순수 도메인 로직이므로
  numpy·stdlib만으로 결정적 단위 테스트가 가능해야 한다.
  ADR 0013 참조.

의존: stdlib — imageio·torch·PySide6·av import 금지.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 상수 — 매직넘버 금지
# ---------------------------------------------------------------------------
# 배속 허용 범위 (MVP 결정1·결정2)
FACTOR_MIN: float = 0.25
FACTOR_MAX: float = 4.0

# 기준 표시시간 단위 (ms)
MS_PER_SEC: float = 1000.0

# GIF per-frame duration 하한 및 클램프 목표 (ADR 0013 결정4)
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
    """배속 factor 범위(0.25 ~ 4.0) 검증.

    WHY 단일 조건: 0/음수는 FACTOR_MIN 미만에 포함된다.
    not (MIN <= factor <= MAX) 가 0, 음수, 초과를 모두 포착한다.
    """
    if not (FACTOR_MIN <= seg.factor <= FACTOR_MAX):
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

    frame_indices:  출력할 프레임 인덱스 시퀀스(tuple).
                    슬로우=복제로 같은 인덱스 반복, 패스트=일부 인덱스 생략.
    durations_ms:   frame_indices와 1:1 대응하는 프레임별 표시시간(ms, tuple).
    base_fps:       기준 fps(등속 1프레임 시간 = 1000/base_fps ms).
                    schedule_to_cfr_indices에서 복제/드롭 비율 계산 기준.
                    기본값 30.0은 일반 촬영 기준.

    WHY frozen=True + tuple: frozen dataclass에 list 필드를 두면 해시 불가이고
      외부에서 내부 컬렉션을 변경할 수 있어 불변 IR 표방과 모순된다.
      tuple로 변경하면 완전한 불변성과 해시 가능성이 보장된다.
      VideoExportConfig.segments를 tuple로 정의한 것과 동일 논리(ADR 0013 결정2).
    WHY base_fps 포함: schedule_to_cfr_indices가 PlaybackSchedule만으로
      복제/드롭 비율을 결정할 수 있어야 한다(인터페이스 최소화 원칙).
    ADR 0013 결정2 참조.
    """

    frame_indices: tuple[int, ...]
    durations_ms: tuple[float, ...]
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
        return PlaybackSchedule(frame_indices=(), durations_ms=())

    base_duration = MS_PER_SEC / base_fps
    normalized = normalize_segments(segments)
    factor_map = _build_factor_map(normalized, n_frames)

    durations = tuple(base_duration / factor_map[i] for i in range(n_frames))
    return PlaybackSchedule(
        frame_indices=tuple(range(n_frames)),
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

    WHY 잔여 보장(최소 1프레임):
      패스트 구간에서 ratio < 1.0이면 누적기가 1.0에 못 미쳐 마지막 프레임이
      결과에 포함되지 않을 수 있다. 입력이 비어있지 않은데 결과가 비면
      MP4에서 해당 구간 전체가 소실되므로, 최소 1프레임(마지막 원본 인덱스)를 보장한다.
    """
    if not schedule.frame_indices:
        return []

    result: list[int] = []
    accumulator: float = 0.0
    base_dur = MS_PER_SEC / schedule.base_fps  # 등속 1프레임 기준 시간

    last_fi = schedule.frame_indices[-1]
    for fi, dur in zip(schedule.frame_indices, schedule.durations_ms):
        accumulator += dur / base_dur
        count = int(accumulator)
        for _ in range(count):
            result.append(fi)
        accumulator -= count

    # 잔여 보장: 루프 후 누적기에 잔여(>0)가 있거나 결과가 비면
    # 마지막 원본 인덱스를 추가해 패스트 구간 소실을 방지한다.
    # WHY: ratio < 1.0인 패스트 프레임은 누적기가 1.0에 못 미쳐 드롭된다.
    # 단일 패스트 프레임이나 끝부분 패스트 구간이 완전 소실되면
    # MP4에서 해당 구간 자체가 사라지는 치명적 결함이 된다.
    if accumulator > 0 or not result:
        result.append(last_fi)

    return result


# ---------------------------------------------------------------------------
# estimate_output_frame_count — MP4 CFR 출력 프레임 수 사전계산 헬퍼
# ---------------------------------------------------------------------------
def estimate_output_frame_count(
    n_selected: int,
    segments: tuple | list,
    fps: float,
) -> int:
    """선택된 프레임 수 + 구간 배속 + fps → 예상 MP4 출력 프레임 수(순수).

    segments=() → n_selected 그대로 반환(항등).
    슬로우 구간이 있으면 복제 수만큼 증가, 패스트 구간이 있으면 드롭만큼 감소.

    WHY 순수 함수: imageio·torch·PySide6 비의존 — UI에서 인코딩 전 경고 표시 가능.
    WHY schedule_to_cfr_indices 위임: 실제 encode와 동일 로직 재사용 — 오차 최소화.
    폭증 경고(GIF·MP4 용량) 사전 계산에 사용된다. ADR 0013 Task 4-3 참조.

    Args:
        n_selected: 선택된(gap_policy 출력) 프레임 수.
        segments:   SpeedSegment 시퀀스 (tuple 또는 list). 빈 시퀀스 허용.
        fps:        기준 출력 fps.

    Returns:
        예상 MP4 출력 프레임 수 (int).
    """
    schedule = build_playback_schedule(n_selected, list(segments), fps)
    return len(schedule_to_cfr_indices(schedule))


# ---------------------------------------------------------------------------
# TrimRange — 출력 트림 구간 값객체(불변)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrimRange:
    """출력 트림 구간 값객체(불변). 출력 시퀀스 상대 [start, end) 좌표계.

    start: 시작(포함). end: 끝(미포함).

    WHY frozen=True: SpeedSegment와 동일한 불변 값객체 계약.
      생성 후 변경 불가하므로 해시 가능하고 안전하게 공유된다.
    WHY "출력 시퀀스 상대" [0, n): trim·segments 좌표는 encode_frames에 전달되는
      출력 crops 시퀀스 기준이다. gap_policy=BACKGROUND에서는 crops가 span 전체와
      동일해 span 상대와 일치한다. CUT/FREEZE에서는 build_output_indices가 갭을
      제거해 crops가 압축되므로 span 상대와 어긋난다. 즉 현재 구현은 BACKGROUND
      전제로만 trim/segments 좌표 정합을 보장한다(잠복 — ADR 0013의 2단계 인덱싱
      미구현 추적).
    """

    start: int
    end: int


# ---------------------------------------------------------------------------
# slice_for_trim — 트림 구간 슬라이스(순수)
# ---------------------------------------------------------------------------
def slice_for_trim(items: list, trim: TrimRange | None) -> list:
    """trim 구간으로 리스트를 슬라이스한다(순수).

    trim=None → items 그대로 반환(항등, 무회귀 경로).
    trim 지정 → items[trim.start:trim.end] 새 리스트 반환.

    WHY 항등 경로: trim=None은 "트림 안 함"이므로 기존 동작을 보존한다.
    WHY 슬라이스 새 객체: 원본 mutate를 막아 부수효과 없음 계약을 지킨다.
    """
    if trim is None:
        return items
    return items[trim.start:trim.end]


# ---------------------------------------------------------------------------
# shift_segments_into_trim — 트림-로컬 평행이동·클리핑(순수)
# ---------------------------------------------------------------------------
def shift_segments_into_trim(segments, trim: TrimRange | None):
    """segments를 트림-로컬 좌표 [0, M)로 평행이동·클리핑한다(순수).

    trim=None → segments 그대로 반환(항등, 무회귀 경로).
    trim 지정 → 각 seg를 [trim.start, trim.end)와 교집합 후 trim.start만큼
    빼서 트림-로컬 좌표로 옮긴다. 교집합이 비면(트림 밖) 드롭한다. factor 보존.

    WHY 트림-로컬: 트림을 먼저 적용해 새 원점(0)이 trim.start가 되므로
      segments도 동일 원점 기준으로 평행이동해야 위치가 맞는다(planner 적용 순서).
    """
    if trim is None:
        return segments
    shifted = (_clip_segment_into_trim(seg, trim) for seg in segments)
    return tuple(seg for seg in shifted if seg is not None)


def _clip_segment_into_trim(seg: SpeedSegment, trim: TrimRange):
    """seg를 트림과 교집합 후 트림-로컬 좌표로 옮긴 SpeedSegment 반환.

    교집합이 비면(lo >= hi) None을 반환해 호출부가 드롭하게 한다.
    WHY 헬퍼 분리: shift_segments_into_trim 20줄·단일 책임 유지.
    """
    lo = max(seg.start, trim.start) - trim.start
    hi = min(seg.end, trim.end) - trim.start
    if lo < hi:
        return SpeedSegment(start=lo, end=hi, factor=seg.factor)
    return None


# ---------------------------------------------------------------------------
# validate_trim — 트림 범위 검증(순수)
# ---------------------------------------------------------------------------
def validate_trim(trim: TrimRange | None, n_frames: int) -> None:
    """트림 범위를 검증한다(순수). 위반 시 한국어 ValueError.

    trim=None → 통과(검증 대상 아님).
    유효 조건: 0 <= start < end <= n_frames.

    WHY 검증 필요: 범위 밖 트림은 슬라이스가 빈 결과·역전 결과를 내
      움짤이 비거나 의도와 달라진다. 인코딩 전 조기 차단한다.
    WHY 한국어 + 수치: 사용자 대면 에러 정책 + 어디가 잘못됐는지 알려준다.
    """
    if trim is None:
        return
    if not (0 <= trim.start < trim.end <= n_frames):
        raise ValueError(
            f"트림 범위가 올바르지 않습니다: [{trim.start},{trim.end}). "
            f"0 <= 시작 < 끝 <= n_frames({n_frames}) 조건을 만족해야 합니다."
        )


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
        return PlaybackSchedule(frame_indices=(), durations_ms=()), []

    new_durations: list[float] = []
    clamped_indices: list[int] = []

    for fi, dur in zip(schedule.frame_indices, schedule.durations_ms):
        if dur < GIF_DURATION_MIN_MS:
            new_durations.append(GIF_DURATION_CLAMP_MS)
            clamped_indices.append(fi)
        else:
            new_durations.append(dur)

    # WHY 새 tuple 생성: frame_indices는 불변 참조 재사용(변경 없음),
    # durations_ms는 클램프 결과 새 tuple로 완전히 새 객체 보장.
    clamped_schedule = PlaybackSchedule(
        frame_indices=schedule.frame_indices,
        durations_ms=tuple(new_durations),
        base_fps=schedule.base_fps,
    )
    return clamped_schedule, clamped_indices
