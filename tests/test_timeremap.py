"""타임리맵 순수 로직 단위 테스트.

대상 모듈: easy_capture.core.timing.timeremap (미구현 — import 실패 RED 정상)

테스트 전략 (계획서 §6-1):
  - imageio / torch / PySide6 비의존 — 순수 stdlib/numpy 모듈 계약 검증
  - TDD RED 단계: 프로덕션 코드 없으므로 전 케이스 xfail(RED) 예상
  - 방법론: Given / When / Then + 한국어 docstring
  - 기존 300+ 테스트 무회귀: 이 파일만 신규 추가, 기존 파일 무수정

WHY try/except + xfail:
  모듈이 없으면 import 자체가 ModuleNotFoundError를 던진다.
  try/except 로 격리해 기존 테스트 컬렉션 차단을 막으면서,
  _HAS_TIMEREMAP=False 시 각 테스트 첫 줄 _require_timeremap() 가
  pytest.xfail() 을 호출해 XFAIL(=예상된 RED) 양상을 명확히 노출한다.
  구현 완료 시 xfail → GREEN 자동 전환(strict=False).
  (기존 test_crop.py skipif 패턴에서 TDD RED 가시성 보강)
"""
from __future__ import annotations

import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# 모듈 격리 — 미구현 시 XFAIL RED (기존 테스트 차단 없음)
# ---------------------------------------------------------------------------
try:
    from easy_capture.core.timing.timeremap import (
        PlaybackSchedule,
        SpeedSegment,
        build_playback_schedule,
        clamp_durations_for_gif,
        normalize_segments,
        schedule_to_cfr_indices,
    )
    _HAS_TIMEREMAP = True
except ModuleNotFoundError:
    SpeedSegment = None  # type: ignore[assignment,misc]
    PlaybackSchedule = None  # type: ignore[assignment,misc]
    normalize_segments = None  # type: ignore[assignment,misc]
    build_playback_schedule = None  # type: ignore[assignment,misc]
    schedule_to_cfr_indices = None  # type: ignore[assignment,misc]
    clamp_durations_for_gif = None  # type: ignore[assignment,misc]
    _HAS_TIMEREMAP = False

_MSG_NOT_IMPL = "easy_capture.core.timing.timeremap 미구현 — RED 예상"


def _require_timeremap() -> None:
    """테스트 본문 첫 줄에 호출 — 미구현이면 xfail 로 RED 표시.

    WHY: pytest.xfail(strict=False) 로 "예상된 실패"를 명확히 표현한다.
         구현 완료 시 자동으로 XPASS → 데코레이터 제거만 하면 GREEN.
    """
    if not _HAS_TIMEREMAP:
        pytest.xfail(_MSG_NOT_IMPL)


# ---------------------------------------------------------------------------
# 테스트 상수 (매직넘버 금지)
# ---------------------------------------------------------------------------
# 기준 fps
BASE_FPS_30 = 30.0
BASE_FPS_12 = 12.0
BASE_FPS_24 = 24.0

# 배속 factor
FACTOR_HALF = 0.5      # 슬로우모션 2배 느림
FACTOR_DOUBLE = 2.0    # 패스트 2배 빠름
FACTOR_NORMAL = 1.0    # 등속
FACTOR_QUARTER = 0.25  # 슬로우모션 4배 느림 (최소)
FACTOR_QUAD = 4.0      # 패스트 4배 빠름 (최대)

# 프레임 수
N_FRAMES_10 = 10
N_FRAMES_20 = 20
N_FRAMES_6 = 6

# 기준 표시시간 상수 (1000 / fps)
DURATION_30FPS_MS = 1000.0 / BASE_FPS_30   # ≈ 33.333 ms
DURATION_12FPS_MS = 1000.0 / BASE_FPS_12   # ≈ 83.333 ms
DURATION_24FPS_MS = 1000.0 / BASE_FPS_24   # ≈ 41.667 ms

# GIF 10ms 하한 / 클램프 목표
GIF_MIN_DURATION_MS = 10.0
GIF_CLAMP_DURATION_MS = 20.0

# 부동소수 비교 허용 오차
APPROX_REL = 1e-6


# ===========================================================================
# 1. SpeedSegment frozen dataclass 계약
# ===========================================================================
class TestSpeedSegment:
    """SpeedSegment(start, end, factor) frozen dataclass 기본 계약."""

    def test_정상_인수로_생성하면_필드가_저장된다(self):
        """Given: start=0, end=5, factor=0.5
        When:  SpeedSegment 생성
        Then:  각 필드가 전달값과 일치한다.

        WHY: frozen dataclass의 기본 계약 검증.
             핵심 값객체이므로 필드 정확성이 최우선.
        """
        _require_timeremap()
        # Given / When
        seg = SpeedSegment(start=0, end=5, factor=FACTOR_HALF)

        # Then
        assert seg.start == 0
        assert seg.end == 5
        assert seg.factor == pytest.approx(FACTOR_HALF)

    def test_frozen이므로_필드_변경_시_예외를_던진다(self):
        """Given: 생성된 SpeedSegment
        When:  start 필드를 변경 시도
        Then:  FrozenInstanceError(또는 AttributeError) 발생

        WHY: 불변 값객체 보장 — 계획서 결정2 'frozen=True' 준수.
        """
        _require_timeremap()
        seg = SpeedSegment(start=0, end=5, factor=FACTOR_HALF)

        with pytest.raises((AttributeError, TypeError)):
            seg.start = 3  # type: ignore[misc]

    def test_동일_값_두_인스턴스는_동등하다(self):
        """Given: 동일 인수로 생성한 두 SpeedSegment
        When:  == 비교
        Then:  True 반환 (frozen dataclass 기본 __eq__)

        WHY: dict key / set 사용을 위한 동등성 보장.
        """
        _require_timeremap()
        a = SpeedSegment(start=2, end=6, factor=FACTOR_DOUBLE)
        b = SpeedSegment(start=2, end=6, factor=FACTOR_DOUBLE)

        assert a == b


# ===========================================================================
# 2. normalize_segments 검증·정규화 순수 함수
# ===========================================================================
class TestNormalizeSegments:
    """normalize_segments(segments, n_frames?) → 정렬·검증된 리스트."""

    def test_빈_리스트는_빈_리스트를_반환한다(self):
        """Given: 빈 segments 리스트
        When:  normalize_segments 호출
        Then:  빈 리스트 반환 (항등 경로)

        WHY: segments=[] → 기존 단일 fps 경로 무회귀 핵심 조건.
        """
        _require_timeremap()
        result = normalize_segments([])

        assert result == []

    def test_단일_유효_구간은_그대로_반환한다(self):
        """Given: 유효한 단일 SpeedSegment
        When:  normalize_segments 호출
        Then:  동일 구간 1개 리스트 반환.
        """
        _require_timeremap()
        seg = SpeedSegment(start=2, end=7, factor=FACTOR_HALF)

        result = normalize_segments([seg])

        assert len(result) == 1
        assert result[0] == seg

    def test_start_기준_정렬이_적용된다(self):
        """Given: start 역순 구간 리스트
        When:  normalize_segments 호출
        Then:  start 오름차순으로 정렬된 리스트 반환.

        WHY: 겹침 검사는 정렬 후 O(n) 비교. 정렬이 선행 조건.
        """
        _require_timeremap()
        seg_late = SpeedSegment(start=6, end=9, factor=FACTOR_NORMAL)
        seg_early = SpeedSegment(start=1, end=4, factor=FACTOR_DOUBLE)

        result = normalize_segments([seg_late, seg_early])

        assert result[0].start < result[1].start

    def test_역전_구간이면_ValueError를_던진다(self):
        """Given: start >= end 인 구간 (역전)
        When:  normalize_segments 호출
        Then:  ValueError — 한국어 메시지에 '역전' 또는 '시작' 포함.

        WHY: 계획서 결정2 'start < end 역전 금지'.
        """
        _require_timeremap()
        bad = SpeedSegment(start=5, end=3, factor=FACTOR_HALF)

        with pytest.raises(ValueError, match="역전|시작|start"):
            normalize_segments([bad])

    def test_동일_start_end_이면_ValueError를_던진다(self):
        """Given: start == end (빈 구간)
        When:  normalize_segments 호출
        Then:  ValueError — 빈 구간 금지.
        """
        _require_timeremap()
        bad = SpeedSegment(start=3, end=3, factor=FACTOR_NORMAL)

        with pytest.raises(ValueError):
            normalize_segments([bad])

    def test_겹치는_두_구간이면_ValueError를_던진다(self):
        """Given: [0,5)과 [3,8)처럼 겹치는 두 구간
        When:  normalize_segments 호출
        Then:  ValueError — 한국어 메시지에 '겹침' 또는 '중복' 포함.

        WHY: 계획서 결정2 '겹침 금지(MVP)'.
             KISS — 겹침 우선순위 규칙은 v1.1 이후.
        """
        _require_timeremap()
        seg_a = SpeedSegment(start=0, end=5, factor=FACTOR_HALF)
        seg_b = SpeedSegment(start=3, end=8, factor=FACTOR_DOUBLE)

        with pytest.raises(ValueError, match="겹침|중복|overlap"):
            normalize_segments([seg_a, seg_b])

    def test_정확히_인접한_구간은_겹침이_아니다(self):
        """Given: [0,5)과 [5,10)처럼 끝과 시작이 맞닿은 구간
        When:  normalize_segments 호출
        Then:  정상 반환 (겹침 아님 — 배타 끝 인덱스 규칙).

        WHY: [start, end) 배타 범위이므로 end == next.start는 겹침이 아닌
             연속 구간이다. 경계값 검증.
        """
        _require_timeremap()
        seg_a = SpeedSegment(start=0, end=5, factor=FACTOR_HALF)
        seg_b = SpeedSegment(start=5, end=10, factor=FACTOR_DOUBLE)

        result = normalize_segments([seg_a, seg_b])

        assert len(result) == 2

    def test_factor_최솟값_0_25는_유효하다(self):
        """Given: factor=0.25 (MVP 하한)
        When:  normalize_segments 호출
        Then:  정상 반환.
        """
        _require_timeremap()
        seg = SpeedSegment(start=0, end=3, factor=FACTOR_QUARTER)

        result = normalize_segments([seg])

        assert len(result) == 1

    def test_factor_최댓값_4_0은_유효하다(self):
        """Given: factor=4.0 (MVP 상한)
        When:  normalize_segments 호출
        Then:  정상 반환.
        """
        _require_timeremap()
        seg = SpeedSegment(start=0, end=3, factor=FACTOR_QUAD)

        result = normalize_segments([seg])

        assert len(result) == 1

    def test_factor_범위_초과이면_ValueError를_던진다(self):
        """Given: factor=5.0 (4.0 초과)
        When:  normalize_segments 호출
        Then:  ValueError — 한국어 메시지에 '배속' 또는 'factor' 포함.

        WHY: 계획서 결정1·결정2 'MVP 배속 0.25~4.0 범위 밖은 ValueError'.
        """
        _require_timeremap()
        bad = SpeedSegment(start=0, end=3, factor=5.0)

        with pytest.raises(ValueError, match="배속|factor|범위"):
            normalize_segments([bad])

    def test_factor_0이면_ValueError를_던진다(self):
        """Given: factor=0.0 (0속도 = 정지)
        When:  normalize_segments 호출
        Then:  ValueError — 0/음수 방어 (reviewer 요구사항).
        """
        _require_timeremap()
        bad = SpeedSegment(start=0, end=3, factor=0.0)

        with pytest.raises(ValueError):
            normalize_segments([bad])

    def test_factor_음수이면_ValueError를_던진다(self):
        """Given: factor=-1.0 (음수 배속)
        When:  normalize_segments 호출
        Then:  ValueError.
        """
        _require_timeremap()
        bad = SpeedSegment(start=0, end=3, factor=-1.0)

        with pytest.raises(ValueError):
            normalize_segments([bad])

    def test_factor_범위_미만이면_ValueError를_던진다(self):
        """Given: factor=0.1 (0.25 미만)
        When:  normalize_segments 호출
        Then:  ValueError.
        """
        _require_timeremap()
        bad = SpeedSegment(start=0, end=3, factor=0.1)

        with pytest.raises(ValueError):
            normalize_segments([bad])


# ===========================================================================
# 3. PlaybackSchedule frozen dataclass 계약
# ===========================================================================
class TestPlaybackSchedule:
    """PlaybackSchedule(frame_indices, durations_ms) frozen dataclass 계약."""

    def test_필드가_올바르게_저장된다(self):
        """Given: frame_indices, durations_ms 리스트
        When:  PlaybackSchedule 생성
        Then:  각 필드가 전달값과 일치한다.

        WHY: 중간 표현(IR)의 기본 계약.
             GIF·MP4 양쪽에서 이 객체를 소비하므로 필드 정확성 필수.
        """
        _require_timeremap()
        indices = [0, 1, 2, 2, 3]    # 슬로우 = 인덱스 2 복제
        durations = [33.3, 33.3, 66.6, 66.6, 33.3]

        sched = PlaybackSchedule(frame_indices=indices, durations_ms=durations)

        assert sched.frame_indices == indices
        assert sched.durations_ms == durations

    def test_frame_indices와_durations_ms는_같은_길이다(self):
        """Given: 길이 5의 frame_indices와 durations_ms
        When:  PlaybackSchedule 생성 후 길이 비교
        Then:  len(frame_indices) == len(durations_ms).

        WHY: 1:1 대응이 깨지면 인코더가 인덱스 밖 접근으로 크래시한다.
        """
        _require_timeremap()
        n = 5
        indices = list(range(n))
        durations = [DURATION_30FPS_MS] * n

        sched = PlaybackSchedule(frame_indices=indices, durations_ms=durations)

        assert len(sched.frame_indices) == len(sched.durations_ms)


# ===========================================================================
# 4. build_playback_schedule — 핵심 순수 함수
# ===========================================================================
class TestBuildPlaybackSchedule:
    """build_playback_schedule(n_frames, segments, base_fps) → PlaybackSchedule."""

    # -----------------------------------------------------------------------
    # 4-1. 항등 케이스 (segments=[])
    # -----------------------------------------------------------------------
    def test_segments_빈리스트면_frame_indices가_range와_같다(self):
        """Given: n_frames=10, segments=[], base_fps=30
        When:  build_playback_schedule 호출
        Then:  frame_indices == list(range(10)) (항등)

        WHY: 계획서 §3 결정3 항등 보장 + 무회귀 핵심 조건.
             segments=() 빈 경우 기존 단일 fps 경로와 동일해야 한다.
        """
        _require_timeremap()
        sched = build_playback_schedule(
            n_frames=N_FRAMES_10,
            segments=[],
            base_fps=BASE_FPS_30,
        )

        assert sched.frame_indices == list(range(N_FRAMES_10))

    def test_segments_빈리스트면_durations_ms가_균일하다(self):
        """Given: n_frames=10, segments=[], base_fps=30
        When:  build_playback_schedule 호출
        Then:  모든 durations_ms == 1000/30 ≈ 33.333 ms (균일).

        WHY: 등속 경로에서 모든 프레임이 동일 시간을 가져야
             기존 GIF duration과 동일한 결과가 나온다.
        """
        _require_timeremap()
        sched = build_playback_schedule(
            n_frames=N_FRAMES_10,
            segments=[],
            base_fps=BASE_FPS_30,
        )

        expected = DURATION_30FPS_MS
        for d in sched.durations_ms:
            assert d == pytest.approx(expected, rel=APPROX_REL)

    def test_segments_빈리스트면_frame_indices_길이가_n_frames와_같다(self):
        """Given: n_frames=6, segments=[]
        When:  build_playback_schedule 호출
        Then:  len(frame_indices) == 6.
        """
        _require_timeremap()
        sched = build_playback_schedule(
            n_frames=N_FRAMES_6,
            segments=[],
            base_fps=BASE_FPS_12,
        )

        assert len(sched.frame_indices) == N_FRAMES_6

    # -----------------------------------------------------------------------
    # 4-2. 0.5x 슬로우모션 구간 — 표시시간 2배
    # -----------------------------------------------------------------------
    def test_슬로우_0_5x_구간의_표시시간이_2배이다(self):
        """Given: n_frames=10, segments=[(2,5,0.5)], base_fps=30
        When:  build_playback_schedule 호출
        Then:  구간 [2,5) 내 durations_ms == 1000/30/0.5 = 66.666 ms.

        WHY: 계획서 §3 결정1 '표시시간 = (1000/base_fps)/factor'.
             0.5x = 2배 느림 = 표시시간 2배.
        """
        _require_timeremap()
        seg = SpeedSegment(start=2, end=5, factor=FACTOR_HALF)
        sched = build_playback_schedule(
            n_frames=N_FRAMES_10,
            segments=[seg],
            base_fps=BASE_FPS_30,
        )

        expected_slow = DURATION_30FPS_MS / FACTOR_HALF   # ≈ 66.666 ms

        # PlaybackSchedule.frame_indices 에서 구간 [2,5)에 해당하는 위치를 찾아 검증
        slow_durations = [
            d for fi, d in zip(sched.frame_indices, sched.durations_ms)
            if 2 <= fi < 5
        ]
        assert len(slow_durations) > 0, "슬로우 구간 프레임이 스케줄에 없음"
        for d in slow_durations:
            assert d == pytest.approx(expected_slow, rel=APPROX_REL)

    def test_슬로우_구간_밖은_등속_표시시간이다(self):
        """Given: n_frames=10, segments=[(2,5,0.5)], base_fps=30
        When:  build_playback_schedule 호출
        Then:  구간 [2,5) 밖 durations_ms == 1000/30 ≈ 33.333 ms.

        WHY: 지정되지 않은 gap 구간은 자동으로 factor=1.0(등속).
        """
        _require_timeremap()
        seg = SpeedSegment(start=2, end=5, factor=FACTOR_HALF)
        sched = build_playback_schedule(
            n_frames=N_FRAMES_10,
            segments=[seg],
            base_fps=BASE_FPS_30,
        )

        expected_normal = DURATION_30FPS_MS

        outside_durations = [
            d for fi, d in zip(sched.frame_indices, sched.durations_ms)
            if not (2 <= fi < 5)
        ]
        assert len(outside_durations) > 0, "구간 밖 프레임이 스케줄에 없음"
        for d in outside_durations:
            assert d == pytest.approx(expected_normal, rel=APPROX_REL)

    # -----------------------------------------------------------------------
    # 4-3. 2.0x 패스트포워드 구간 — 표시시간 절반
    # -----------------------------------------------------------------------
    def test_패스트_2_0x_구간의_표시시간이_절반이다(self):
        """Given: n_frames=10, segments=[(0,4,2.0)], base_fps=30
        When:  build_playback_schedule 호출
        Then:  구간 [0,4) 내 durations_ms == 1000/30/2.0 ≈ 16.666 ms.

        WHY: 계획서 §3 결정1 'factor > 1.0 = 패스트 = 표시시간 감소'.
        """
        _require_timeremap()
        seg = SpeedSegment(start=0, end=4, factor=FACTOR_DOUBLE)
        sched = build_playback_schedule(
            n_frames=N_FRAMES_10,
            segments=[seg],
            base_fps=BASE_FPS_30,
        )

        expected_fast = DURATION_30FPS_MS / FACTOR_DOUBLE  # ≈ 16.666 ms

        fast_durations = [
            d for fi, d in zip(sched.frame_indices, sched.durations_ms)
            if 0 <= fi < 4
        ]
        assert len(fast_durations) > 0
        for d in fast_durations:
            assert d == pytest.approx(expected_fast, rel=APPROX_REL)

    # -----------------------------------------------------------------------
    # 4-4. 다중 구간 — 각 구간 독립 적용
    # -----------------------------------------------------------------------
    def test_다중_구간이_각각_독립적으로_적용된다(self):
        """Given: segments=[(0,2,2.0),(4,6,0.5)], n_frames=10, base_fps=30
        When:  build_playback_schedule 호출
        Then:
          - [0,2) 구간 durations ≈ 16.666 ms (2.0x)
          - [4,6) 구간 durations ≈ 66.666 ms (0.5x)
          - [2,4)과 [6,10) 구간 durations ≈ 33.333 ms (등속)

        WHY: 덕후 시나리오 '인트로 패스트 → 하이라이트 슬로우' 다중 구간.
        """
        _require_timeremap()
        seg_fast = SpeedSegment(start=0, end=2, factor=FACTOR_DOUBLE)
        seg_slow = SpeedSegment(start=4, end=6, factor=FACTOR_HALF)

        sched = build_playback_schedule(
            n_frames=N_FRAMES_10,
            segments=[seg_fast, seg_slow],
            base_fps=BASE_FPS_30,
        )

        dur_fast = DURATION_30FPS_MS / FACTOR_DOUBLE
        dur_slow = DURATION_30FPS_MS / FACTOR_HALF
        dur_normal = DURATION_30FPS_MS

        for fi, d in zip(sched.frame_indices, sched.durations_ms):
            if 0 <= fi < 2:
                assert d == pytest.approx(dur_fast, rel=APPROX_REL), f"fi={fi}: fast 구간 오류"
            elif 4 <= fi < 6:
                assert d == pytest.approx(dur_slow, rel=APPROX_REL), f"fi={fi}: slow 구간 오류"
            else:
                assert d == pytest.approx(dur_normal, rel=APPROX_REL), f"fi={fi}: 등속 구간 오류"

    # -----------------------------------------------------------------------
    # 4-5. 경계값 — start=0, end=n_frames (전체 커버)
    # -----------------------------------------------------------------------
    def test_전체_범위_단일_구간이_올바르게_적용된다(self):
        """Given: segments=[(0, n_frames, 0.5)], 전체 구간 슬로우
        When:  build_playback_schedule 호출
        Then:  모든 durations_ms ≈ 66.666 ms (2배 느림).

        WHY: 경계값 — start=0, end=n_frames 전체 구간이 경계 밖에 없는 케이스.
        """
        _require_timeremap()
        seg = SpeedSegment(start=0, end=N_FRAMES_10, factor=FACTOR_HALF)
        sched = build_playback_schedule(
            n_frames=N_FRAMES_10,
            segments=[seg],
            base_fps=BASE_FPS_30,
        )

        expected = DURATION_30FPS_MS / FACTOR_HALF
        for d in sched.durations_ms:
            assert d == pytest.approx(expected, rel=APPROX_REL)

    def test_구간_경계_프레임이_올바른_구간에_속한다(self):
        """Given: segments=[(3,7,0.5)], n_frames=10
        When:  build_playback_schedule 호출
        Then:  frame_index 3과 6은 슬로우, 2와 7은 등속.

        WHY: [start, end) 배타 범위 경계 정확성 검증.
             off-by-one 버그를 잡는 핵심 케이스.
        """
        _require_timeremap()
        seg = SpeedSegment(start=3, end=7, factor=FACTOR_HALF)
        sched = build_playback_schedule(
            n_frames=N_FRAMES_10,
            segments=[seg],
            base_fps=BASE_FPS_30,
        )

        dur_slow = DURATION_30FPS_MS / FACTOR_HALF
        dur_normal = DURATION_30FPS_MS

        lookup = dict(zip(sched.frame_indices, sched.durations_ms))

        # 구간 경계 내 첫/끝 프레임
        assert lookup[3] == pytest.approx(dur_slow, rel=APPROX_REL), "start=3은 슬로우여야 함"
        assert lookup[6] == pytest.approx(dur_slow, rel=APPROX_REL), "end-1=6은 슬로우여야 함"
        # 구간 경계 밖 인접 프레임
        assert lookup[2] == pytest.approx(dur_normal, rel=APPROX_REL), "2는 등속이어야 함"
        assert lookup[7] == pytest.approx(dur_normal, rel=APPROX_REL), "7은 등속이어야 함 (end=7 배타)"

    # -----------------------------------------------------------------------
    # 4-6. 반환값 타입 계약
    # -----------------------------------------------------------------------
    def test_반환값이_PlaybackSchedule_인스턴스다(self):
        """Given: 임의 유효 인수
        When:  build_playback_schedule 호출
        Then:  반환값은 PlaybackSchedule 인스턴스.
        """
        _require_timeremap()
        sched = build_playback_schedule(
            n_frames=N_FRAMES_6,
            segments=[],
            base_fps=BASE_FPS_12,
        )

        assert isinstance(sched, PlaybackSchedule)

    def test_frame_indices와_durations_ms_길이가_일치한다(self):
        """Given: n_frames=10, 단일 구간
        When:  build_playback_schedule 호출
        Then:  len(frame_indices) == len(durations_ms).

        WHY: 1:1 불일치는 인코더 크래시의 직접 원인.
        """
        _require_timeremap()
        seg = SpeedSegment(start=2, end=5, factor=FACTOR_HALF)
        sched = build_playback_schedule(
            n_frames=N_FRAMES_10,
            segments=[seg],
            base_fps=BASE_FPS_30,
        )

        assert len(sched.frame_indices) == len(sched.durations_ms)


# ===========================================================================
# 5. schedule_to_cfr_indices — MP4 CFR 변환
# ===========================================================================
class TestScheduleToCfrIndices:
    """schedule_to_cfr_indices(schedule) → list[int] (MP4용 복제/드롭 인덱스).

    WHY: MP4는 고정 fps(CFR)라 표시시간을 직접 표현 못 함.
         슬로우=프레임 복제(같은 인덱스 반복), 패스트=프레임 드롭.
         계획서 §3 결정3 'schedule_to_cfr_indices 헬퍼'.
    """

    def test_등속_구간의_cfr_인덱스는_원본과_같다(self):
        """Given: 전 프레임 등속 PlaybackSchedule(durations 균일)
        When:  schedule_to_cfr_indices 호출
        Then:  반환 리스트 == list(range(n_frames)) (복제/드롭 없음).
        """
        _require_timeremap()
        n = N_FRAMES_6
        sched = PlaybackSchedule(
            frame_indices=list(range(n)),
            durations_ms=[DURATION_30FPS_MS] * n,
        )

        result = schedule_to_cfr_indices(sched)

        assert result == list(range(n))

    def test_슬로우_0_5x_구간_프레임이_약_2배_복제된다(self):
        """Given: 슬로우 0.5x 전 구간 PlaybackSchedule (4 프레임)
        When:  schedule_to_cfr_indices 호출
        Then:  반환 리스트 길이 ≈ 4 × 2 = 8 (±1 허용).

        WHY: 0.5x = 2배 느림 = 프레임 복제 2배.
             기준: durations_ms / (1000/base_fps) 비율로 복제수 결정.
             정확한 복제 전략(올림/반올림)은 구현에 따라 ±1 오차 허용.
        """
        _require_timeremap()
        n = 4
        dur_slow = DURATION_30FPS_MS / FACTOR_HALF  # 66.666 ms
        sched = PlaybackSchedule(
            frame_indices=list(range(n)),
            durations_ms=[dur_slow] * n,
        )

        result = schedule_to_cfr_indices(sched)

        expected_len = n * 2  # 0.5x = 2배 복제
        assert abs(len(result) - expected_len) <= 1, (
            f"슬로우 0.5x 복제 수 오류: 예상 {expected_len}±1, 실제 {len(result)}"
        )

    def test_패스트_2_0x_구간_프레임이_약_절반_드롭된다(self):
        """Given: 패스트 2.0x 전 구간 PlaybackSchedule (6 프레임)
        When:  schedule_to_cfr_indices 호출
        Then:  반환 리스트 길이 ≈ 6 × 0.5 = 3 (±1 허용).

        WHY: 2.0x = 2배 빠름 = 프레임 드롭 절반.
        """
        _require_timeremap()
        n = N_FRAMES_6
        dur_fast = DURATION_30FPS_MS / FACTOR_DOUBLE  # 16.666 ms
        sched = PlaybackSchedule(
            frame_indices=list(range(n)),
            durations_ms=[dur_fast] * n,
        )

        result = schedule_to_cfr_indices(sched)

        expected_len = n // 2  # 2.0x = 절반 드롭
        assert abs(len(result) - expected_len) <= 1, (
            f"패스트 2.0x 드롭 수 오류: 예상 {expected_len}±1, 실제 {len(result)}"
        )

    def test_cfr_인덱스는_원본_frame_indices_범위_내에_있다(self):
        """Given: n=6, 슬로우 스케줄
        When:  schedule_to_cfr_indices 호출
        Then:  반환된 모든 인덱스가 0 이상이고 frame_indices의 최대값 이하.

        WHY: 인덱스가 범위 밖이면 crop_frames 접근 시 IndexError.
        """
        _require_timeremap()
        n = N_FRAMES_6
        dur_slow = DURATION_30FPS_MS / FACTOR_HALF
        sched = PlaybackSchedule(
            frame_indices=list(range(n)),
            durations_ms=[dur_slow] * n,
        )

        result = schedule_to_cfr_indices(sched)

        max_valid = max(sched.frame_indices)
        for idx in result:
            assert 0 <= idx <= max_valid, f"cfr 인덱스 범위 초과: {idx}"

    def test_반환값이_list_of_int이다(self):
        """Given: 유효한 PlaybackSchedule
        When:  schedule_to_cfr_indices 호출
        Then:  list[int] 반환.
        """
        _require_timeremap()
        n = 4
        sched = PlaybackSchedule(
            frame_indices=list(range(n)),
            durations_ms=[DURATION_30FPS_MS] * n,
        )

        result = schedule_to_cfr_indices(sched)

        assert isinstance(result, list)
        for idx in result:
            assert isinstance(idx, int), f"int가 아님: {type(idx)}"

    def test_슬로우_구간_복제_프레임은_원본_인덱스_반복이다(self):
        """Given: frame_indices=[3], durations=66.666ms (0.5x 슬로우)
        When:  schedule_to_cfr_indices 호출
        Then:  결과 리스트에 인덱스 3이 2번 포함된다(±복제 전략 허용).

        WHY: '같은 인덱스 반복'이 복제 구현의 핵심 계약.
             crop_frames가 동일 크롭을 반복 적용해 슬로우를 표현.
        """
        _require_timeremap()
        sched = PlaybackSchedule(
            frame_indices=[3],
            durations_ms=[DURATION_30FPS_MS / FACTOR_HALF],
        )

        result = schedule_to_cfr_indices(sched)

        # 인덱스 3이 복제되어 2번 이상 나타나야 함
        assert result.count(3) >= 2, f"슬로우 프레임 3이 {result.count(3)}번만 등장"


# ===========================================================================
# 6. clamp_durations_for_gif — GIF 10ms 하한 가드
# ===========================================================================
class TestClampDurationsForGif:
    """clamp_durations_for_gif(schedule) → (PlaybackSchedule, list[int]).

    WHY: GIF per-frame duration < 10ms → 인코더 delay=0 → 뷰어 ~100ms clamp
         → 빠르게 만들려던 구간이 오히려 느려지는 역전 발생 방어.
         계획서 §3 결정1 '[치명적] GIF 패스트 10ms 하한 가드'.
    """

    def test_10ms_이상_durations는_변경없이_반환된다(self):
        """Given: 모든 durations_ms >= 20ms
        When:  clamp_durations_for_gif 호출
        Then:  durations_ms 변경 없음, 클램프 인덱스 목록 == [].

        WHY: 정상 범위에서 클램프가 발생하지 않음을 확인 (부수효과 없음 계약).
        """
        _require_timeremap()
        n = N_FRAMES_6
        sched = PlaybackSchedule(
            frame_indices=list(range(n)),
            durations_ms=[DURATION_12FPS_MS] * n,   # 83.333 ms — 안전 범위
        )

        clamped, clamp_indices = clamp_durations_for_gif(sched)

        assert clamp_indices == [], "클램프 없어야 하는데 클램프 발생"
        for d in clamped.durations_ms:
            assert d == pytest.approx(DURATION_12FPS_MS, rel=APPROX_REL)

    def test_base30fps_4x_패스트는_8_3ms로_20ms로_클램프된다(self):
        """Given: base 30fps × 4x → duration = 8.333 ms (< 10ms)
        When:  clamp_durations_for_gif 호출
        Then:  해당 durations_ms == 20ms (GIF_CLAMP_DURATION_MS).

        WHY: 계획서 수용 기준 'base 30fps×4x=8.3ms → 20ms로 클램프'.
             핵심 회귀 가드 케이스.
        """
        _require_timeremap()
        dur_4x = DURATION_30FPS_MS / FACTOR_QUAD   # ≈ 8.333 ms
        n = 5
        sched = PlaybackSchedule(
            frame_indices=list(range(n)),
            durations_ms=[dur_4x] * n,
        )

        clamped, clamp_indices = clamp_durations_for_gif(sched)

        for d in clamped.durations_ms:
            assert d == pytest.approx(GIF_CLAMP_DURATION_MS, rel=APPROX_REL), (
                f"클램프 후 duration이 20ms가 아님: {d}"
            )

    def test_클램프_인덱스_목록이_정확히_반환된다(self):
        """Given: 일부 durations만 < 10ms (혼합)
        When:  clamp_durations_for_gif 호출
        Then:  clamp_indices에 < 10ms였던 frame_indices만 포함.

        WHY: UI 경고/노트북 print에서 어느 구간이 클램프됐는지 보고.
             계획서 §3 결정1 '클램프된 프레임 인덱스 목록 반환'.
        """
        _require_timeremap()
        dur_safe = DURATION_12FPS_MS                     # 83.333 ms — 안전
        dur_fast = DURATION_30FPS_MS / FACTOR_QUAD       # 8.333 ms — 클램프 대상

        # frame 0,1: 정상, frame 2,3: 클램프 대상
        sched = PlaybackSchedule(
            frame_indices=[0, 1, 2, 3],
            durations_ms=[dur_safe, dur_safe, dur_fast, dur_fast],
        )

        _, clamp_indices = clamp_durations_for_gif(sched)

        # frame_indices [2, 3]이 클램프 대상
        assert set(clamp_indices) == {2, 3}, (
            f"클램프 인덱스 오류: 예상 {{2,3}}, 실제 {set(clamp_indices)}"
        )

    def test_클램프_후_모든_durations가_10ms_이상이다(self):
        """Given: 일부 durations < 10ms
        When:  clamp_durations_for_gif 호출
        Then:  클램프 후 min(durations_ms) >= 10ms (역전 방어 보장).

        WHY: GIF delay=0 역전(빠르게→느려짐) 방어의 핵심 사후 조건.
        """
        _require_timeremap()
        dur_fast = DURATION_30FPS_MS / FACTOR_QUAD  # 8.333 ms
        sched = PlaybackSchedule(
            frame_indices=list(range(N_FRAMES_10)),
            durations_ms=[dur_fast] * N_FRAMES_10,
        )

        clamped, _ = clamp_durations_for_gif(sched)

        for d in clamped.durations_ms:
            assert d >= GIF_MIN_DURATION_MS, f"클램프 후 10ms 미만 존재: {d}"

    def test_클램프_전후_frame_indices가_불변이다(self):
        """Given: 클램프가 필요한 PlaybackSchedule
        When:  clamp_durations_for_gif 호출
        Then:  반환된 PlaybackSchedule의 frame_indices == 원본 frame_indices.

        WHY: 클램프는 durations_ms만 조정한다. frame_indices 변경은
             프레임 선택 로직 전체를 깨므로 절대 불변.
        """
        _require_timeremap()
        dur_fast = DURATION_30FPS_MS / FACTOR_QUAD
        original_indices = list(range(N_FRAMES_6))
        sched = PlaybackSchedule(
            frame_indices=original_indices,
            durations_ms=[dur_fast] * N_FRAMES_6,
        )

        clamped, _ = clamp_durations_for_gif(sched)

        assert clamped.frame_indices == original_indices, "frame_indices가 변경됨"

    def test_반환값_타입은_PlaybackSchedule과_list이다(self):
        """Given: 유효한 PlaybackSchedule
        When:  clamp_durations_for_gif 호출
        Then:  (PlaybackSchedule, list) 튜플 반환.
        """
        _require_timeremap()
        sched = PlaybackSchedule(
            frame_indices=[0, 1],
            durations_ms=[DURATION_30FPS_MS] * 2,
        )

        result = clamp_durations_for_gif(sched)

        assert isinstance(result, tuple) and len(result) == 2
        clamped, idx_list = result
        assert isinstance(clamped, PlaybackSchedule)
        assert isinstance(idx_list, list)

    def test_정확히_10ms인_duration은_클램프되지_않는다(self):
        """Given: durations_ms = 10.0 (경계값)
        When:  clamp_durations_for_gif 호출
        Then:  10ms는 클램프되지 않음 (< 10ms 조건, 경계 포함 여부 검증).

        WHY: 계획서 '10ms 미만이면 클램프'. 경계값 off-by-one 방어.
             10ms 자체는 유효한 GIF 최소 딜레이.
        """
        _require_timeremap()
        sched = PlaybackSchedule(
            frame_indices=[0],
            durations_ms=[GIF_MIN_DURATION_MS],  # 정확히 10ms
        )

        _, clamp_indices = clamp_durations_for_gif(sched)

        assert clamp_indices == [], f"10ms는 클램프되지 않아야 함. clamp_indices={clamp_indices}"

    def test_20ms_클램프값이_역전_없음을_보장한다(self):
        """Given: base 30fps × 4x = 8.333ms → 20ms로 클램프
        When:  결과 duration을 base_fps=30에서의 등속(33.333ms)와 비교
        Then:  클램프값 20ms < 등속 33.333ms → 여전히 원본보다 빠름(역전 없음).

        WHY: 클램프 목적이 "너무 빨라서 오히려 느려지는 역전 방지"이므로
             클램프 후에도 등속(factor=1.0)보다 빠른(작은) duration이어야 한다.
        """
        _require_timeremap()
        # 이 케이스는 상수 관계만 검증 — _require_timeremap 호출 후 순수 산술
        assert GIF_CLAMP_DURATION_MS < DURATION_30FPS_MS, (
            f"20ms 클램프가 등속({DURATION_30FPS_MS:.1f}ms)보다 느림 — 역전 발생"
        )


# ===========================================================================
# 7. core 경계 불변식 — 순수 모듈 의존성 가드
# ===========================================================================
def _check_module_not_imported(forbidden_module: str) -> bool:
    """격리된 서브프로세스에서 timeremap import 후 forbidden_module 미로드 검증.

    WHY subprocess 격리:
      동일 pytest 세션에서 다른 테스트가 먼저 해당 라이브러리를 로드하면
      sys.modules에 이미 존재하여 오탐이 발생한다.
      새로운 인터프리터를 기동해 timeremap만 import한 상태의 sys.modules를 검사한다.
    """
    check_code = (
        "import sys; "
        "from easy_capture.core.timing.timeremap import build_playback_schedule; "
        f"assert '{forbidden_module}' not in sys.modules and "
        f"not any(k.startswith('{forbidden_module}') for k in sys.modules), "
        f"'{forbidden_module} 로드됨'"
    )
    result = subprocess.run(
        [sys.executable, "-c", check_code],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


class TestCoreModuleBoundary:
    """timeremap 모듈이 순수 core 경계(imageio/torch/PySide6 비의존) 준수 검증.

    WHY: 계획서 §3 결정3 'core 경계 불변식: numpy/stdlib만'.
         인코딩·GUI 라이브러리가 타임리맵 모듈에 유입되면
         core 도메인 격리 원칙(ADR 0013)이 깨진다.

    WHY subprocess 격리 검증:
         pytest 세션 내 다른 테스트가 먼저 해당 라이브러리를 로드하면
         sys.modules에 이미 존재해 오탐이 발생한다.
         격리된 인터프리터에서 timeremap만 import한 상태를 검사한다.
    """

    def test_timeremap_모듈_import_후_imageio가_로드되지_않는다(self):
        """Given: easy_capture.core.timing.timeremap import (격리 프로세스)
        When:  sys.modules 확인
        Then:  'imageio'가 sys.modules에 없다.

        WHY: GIF 인코딩은 core/export/video_export.py 책임.
             timeremap은 순수 로직이므로 imageio 직접 import 금지.
        """
        _require_timeremap()
        assert _check_module_not_imported("imageio"), (
            "timeremap import 시 imageio가 로드됨 — core 경계 위반"
        )

    def test_timeremap_모듈_import_후_torch가_로드되지_않는다(self):
        """Given: easy_capture.core.timing.timeremap import (격리 프로세스)
        When:  sys.modules 확인
        Then:  'torch'가 sys.modules에 없다.

        WHY: 타임리맵은 텐서 연산이 불필요한 순수 시간 도메인 로직.
             torch import는 GPU 메모리 점유·스타트업 지연을 유발한다.
        """
        _require_timeremap()
        assert _check_module_not_imported("torch"), (
            "timeremap import 시 torch가 로드됨 — core 경계 위반"
        )

    def test_timeremap_모듈_import_후_PySide6가_로드되지_않는다(self):
        """Given: easy_capture.core.timing.timeremap import (격리 프로세스)
        When:  sys.modules 확인
        Then:  'PySide6'가 sys.modules에 없다.

        WHY: core 도메인은 UI 레이어와 완전히 분리(DIP 원칙).
             headless(노트북·CI) 환경에서 PySide6 없이 동작해야 한다.
        """
        _require_timeremap()
        assert _check_module_not_imported("PySide6"), (
            "timeremap import 시 PySide6가 로드됨 — core 경계 위반"
        )

    def test_timeremap_모듈_import_후_av가_로드되지_않는다(self):
        """Given: easy_capture.core.timing.timeremap import (격리 프로세스)
        When:  sys.modules 확인
        Then:  'av'(PyAV)가 sys.modules에 없다.

        WHY: MP4 인코딩 라이브러리는 core/export 레이어 책임.
             timeremap이 av를 알면 계층 역전 발생.
        """
        _require_timeremap()
        assert _check_module_not_imported("av"), (
            "timeremap import 시 av(PyAV)가 로드됨 — core 경계 위반"
        )


# ===========================================================================
# 8. ValueError 메시지 한국어 검증 — 방어 케이스 통합
# ===========================================================================
class TestValueErrorMessages:
    """normalize_segments 에러 메시지가 한국어 키워드를 포함하는지 검증.

    WHY: 사용자 대면 에러는 한국어 정책(글로벌 CLAUDE.md).
         영문 메시지는 한국어 UI 안내와 충돌.
    """

    def test_역전_구간_에러_메시지가_한국어이다(self):
        """Given: start > end 역전 구간
        When:  normalize_segments 호출
        Then:  ValueError 메시지가 한국어 단어를 포함한다.
        """
        _require_timeremap()
        bad = SpeedSegment(start=8, end=3, factor=FACTOR_HALF)

        with pytest.raises(ValueError) as exc_info:
            normalize_segments([bad])

        msg = str(exc_info.value)
        has_korean = any(ord(c) >= 0xAC00 for c in msg)
        assert has_korean, f"에러 메시지에 한국어 없음: {msg!r}"

    def test_겹침_에러_메시지가_한국어이다(self):
        """Given: 겹치는 두 구간
        When:  normalize_segments 호출
        Then:  ValueError 메시지가 한국어 단어를 포함한다.
        """
        _require_timeremap()
        seg_a = SpeedSegment(start=0, end=6, factor=FACTOR_HALF)
        seg_b = SpeedSegment(start=4, end=9, factor=FACTOR_DOUBLE)

        with pytest.raises(ValueError) as exc_info:
            normalize_segments([seg_a, seg_b])

        msg = str(exc_info.value)
        has_korean = any(ord(c) >= 0xAC00 for c in msg)
        assert has_korean, f"겹침 에러 메시지에 한국어 없음: {msg!r}"

    def test_배속_범위_초과_에러_메시지가_한국어이다(self):
        """Given: factor=10.0 (범위 초과)
        When:  normalize_segments 호출
        Then:  ValueError 메시지가 한국어 단어를 포함한다.
        """
        _require_timeremap()
        bad = SpeedSegment(start=0, end=3, factor=10.0)

        with pytest.raises(ValueError) as exc_info:
            normalize_segments([bad])

        msg = str(exc_info.value)
        has_korean = any(ord(c) >= 0xAC00 for c in msg)
        assert has_korean, f"배속 범위 초과 에러 메시지에 한국어 없음: {msg!r}"


# ===========================================================================
# 9. n_frames=0 엣지 케이스
# ===========================================================================
class TestEdgeCasesZeroFrames:
    """n_frames=0 엣지 케이스 — 빈 시퀀스 처리."""

    def test_n_frames_0이면_빈_스케줄을_반환한다(self):
        """Given: n_frames=0, segments=[]
        When:  build_playback_schedule 호출
        Then:  frame_indices==[], durations_ms==[].

        WHY: 빈 프레임 시퀀스는 이미 상위에서 걸러져야 하지만
             타임리맵 자체는 방어적으로 빈 스케줄을 반환해야 함.
        """
        _require_timeremap()
        sched = build_playback_schedule(
            n_frames=0,
            segments=[],
            base_fps=BASE_FPS_30,
        )

        assert sched.frame_indices == []
        assert sched.durations_ms == []

    def test_n_frames_0_cfr_변환은_빈_리스트이다(self):
        """Given: 빈 PlaybackSchedule
        When:  schedule_to_cfr_indices 호출
        Then:  빈 리스트 반환.
        """
        _require_timeremap()
        sched = PlaybackSchedule(frame_indices=[], durations_ms=[])

        result = schedule_to_cfr_indices(sched)

        assert result == []

    def test_n_frames_0_gif_클램프는_빈_결과이다(self):
        """Given: 빈 PlaybackSchedule
        When:  clamp_durations_for_gif 호출
        Then:  (빈 PlaybackSchedule, []) 반환.
        """
        _require_timeremap()
        sched = PlaybackSchedule(frame_indices=[], durations_ms=[])

        clamped, clamp_indices = clamp_durations_for_gif(sched)

        assert clamped.frame_indices == []
        assert clamped.durations_ms == []
        assert clamp_indices == []
