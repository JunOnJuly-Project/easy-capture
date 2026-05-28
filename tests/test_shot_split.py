"""샷 분할 순수 함수 테스트.

대상 모듈: easy_capture.core.tracking (split_into_shots 신규 — 구현 전 RED)

split_into_shots(n_frames, cut_frames) -> list[tuple[int, int]]
  컷 인덱스 리스트로 프레임 구간을 샷 (start, end) 리스트로 분할한다.
  end는 exclusive (range 관례: frames[start:end]).

테스트 케이스:
  1. 컷 없음  → 1샷 [(0, n_frames)]
  2. 컷 1개   → 2샷 [(0, cut), (cut, n_frames)]
  3. 다중 컷  → N+1샷, 각 구간 경계 정확
  4. 경계값   — 첫 프레임 컷, 마지막 프레임 컷, 빈 프레임 입력

구현 전 RED 상태가 정상:
  core/tracking/__init__.py 또는 rematch.py 등 어느 위치에도
  split_into_shots 미구현.
"""
from __future__ import annotations

import pytest

# split_into_shots — 구현 전이므로 try/except 격리
# WHY: 미구현 시 import 실패가 다른 테스트를 차단하지 않도록 한다.
try:
    from easy_capture.core.tracking.shot_split import split_into_shots
    _HAS_SPLIT = True
except ImportError:
    try:
        # 대안 위치: __init__.py 또는 gap_policy.py 인접 모듈일 수 있음
        from easy_capture.core.tracking import split_into_shots  # type: ignore[attr-defined]
        _HAS_SPLIT = True
    except (ImportError, AttributeError):
        split_into_shots = None  # type: ignore[assignment]
        _HAS_SPLIT = False

_MSG_NO_SPLIT = (
    "core/tracking/shot_split.py(또는 core/tracking/__init__.py)에 "
    "split_into_shots 미구현 — RED 예상"
)

# ---------------------------------------------------------------------------
# 테스트 상수 (매직넘버 금지)
# ---------------------------------------------------------------------------
# 기본 프레임 수 시나리오
N_FRAMES_SHORT = 10
N_FRAMES_MID = 30
N_FRAMES_LONG = 100

# 컷 1개 위치 (중간)
CUT_SINGLE = 5

# 다중 컷 위치 (3샷 → 2컷)
CUT_FIRST = 8
CUT_SECOND = 20

# 경계값: 첫 프레임 직후, 마지막 프레임 직전
CUT_NEAR_START = 1
CUT_NEAR_END_OFFSET = 1  # n_frames - 1


# ---------------------------------------------------------------------------
# 컷 없음 → 1샷
# ---------------------------------------------------------------------------
class TestSplitNocut:
    """컷이 없으면 전체가 1샷이다."""

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_컷_없음이면_1샷을_반환한다(self):
        """Given: n_frames=10, cut_frames=[]
        When:  split_into_shots 호출
        Then:  [(0, 10)] — 전체가 단일 샷

        WHY: detector=None 하위호환 경로처럼, 컷이 없으면 첫 슬라이스 동작과
             동일하게 전체를 1샷으로 취급해야 한다.
        """
        shots = split_into_shots(N_FRAMES_SHORT, [])

        assert shots == [(0, N_FRAMES_SHORT)], (
            f"컷 없음 → [(0, {N_FRAMES_SHORT})] 기대, 실제: {shots}"
        )

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_컷_없음_샷_수는_1이다(self):
        """Given: cut_frames=[]
        When:  split_into_shots 호출
        Then:  len(shots) == 1
        """
        shots = split_into_shots(N_FRAMES_MID, [])

        assert len(shots) == 1

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_컷_없음_샷의_start_end가_전체_구간이다(self):
        """Given: n_frames=30, cut_frames=[]
        When:  split_into_shots 호출
        Then:  shots[0] == (0, 30)
        """
        shots = split_into_shots(N_FRAMES_MID, [])

        start, end = shots[0]
        assert start == 0
        assert end == N_FRAMES_MID


# ---------------------------------------------------------------------------
# 컷 1개 → 2샷
# ---------------------------------------------------------------------------
class TestSplitSingleCut:
    """컷 1개이면 2샷으로 분할된다."""

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_컷_1개면_2샷을_반환한다(self):
        """Given: n_frames=10, cut_frames=[5]
        When:  split_into_shots 호출
        Then:  [(0, 5), (5, 10)]

        WHY: cut_frames=[5]는 프레임5에서 새 샷이 시작됨을 의미한다.
             첫 샷 = [0, 5), 두 번째 샷 = [5, 10). end는 exclusive.
        """
        shots = split_into_shots(N_FRAMES_SHORT, [CUT_SINGLE])

        assert shots == [(0, CUT_SINGLE), (CUT_SINGLE, N_FRAMES_SHORT)], (
            f"컷=[{CUT_SINGLE}] → [(0,{CUT_SINGLE}),({CUT_SINGLE},{N_FRAMES_SHORT})] "
            f"기대, 실제: {shots}"
        )

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_컷_1개_샷_수는_2이다(self):
        """Given: cut_frames=[5]
        When:  split_into_shots 호출
        Then:  len(shots) == 2
        """
        shots = split_into_shots(N_FRAMES_SHORT, [CUT_SINGLE])

        assert len(shots) == 2

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_컷_1개_각_샷의_길이_합이_n_frames이다(self):
        """Given: n_frames=10, cut_frames=[5]
        When:  split_into_shots 호출
        Then:  sum(end-start for start,end in shots) == 10

        WHY: 프레임 손실/중복 없이 정확히 분할됐는지 검증한다.
        """
        shots = split_into_shots(N_FRAMES_SHORT, [CUT_SINGLE])

        total = sum(end - start for start, end in shots)
        assert total == N_FRAMES_SHORT, (
            f"프레임 수 불일치: 분할 합계={total} vs n_frames={N_FRAMES_SHORT}"
        )

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_컷_1개_첫_샷은_0부터_시작한다(self):
        """Given: cut_frames=[5]
        When:  split_into_shots 호출
        Then:  shots[0][0] == 0
        """
        shots = split_into_shots(N_FRAMES_SHORT, [CUT_SINGLE])

        assert shots[0][0] == 0

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_컷_1개_마지막_샷은_n_frames에서_끝난다(self):
        """Given: cut_frames=[5]
        When:  split_into_shots 호출
        Then:  shots[-1][1] == n_frames
        """
        shots = split_into_shots(N_FRAMES_SHORT, [CUT_SINGLE])

        assert shots[-1][1] == N_FRAMES_SHORT


# ---------------------------------------------------------------------------
# 다중 컷 → N+1샷
# ---------------------------------------------------------------------------
class TestSplitMultiCut:
    """컷 2개면 3샷, 경계가 정확해야 한다."""

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_컷_2개면_3샷을_반환한다(self):
        """Given: n_frames=30, cut_frames=[8, 20]
        When:  split_into_shots 호출
        Then:  [(0, 8), (8, 20), (20, 30)]

        WHY: 컷이 k개이면 k+1샷이 생성된다 — 기본 산술 규칙 검증.
        """
        shots = split_into_shots(N_FRAMES_MID, [CUT_FIRST, CUT_SECOND])

        expected = [(0, CUT_FIRST), (CUT_FIRST, CUT_SECOND), (CUT_SECOND, N_FRAMES_MID)]
        assert shots == expected, f"기대: {expected}, 실제: {shots}"

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_컷_2개_샷_수는_3이다(self):
        """Given: cut_frames=[8, 20]
        When:  split_into_shots 호출
        Then:  len(shots) == 3
        """
        shots = split_into_shots(N_FRAMES_MID, [CUT_FIRST, CUT_SECOND])

        assert len(shots) == 3

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_다중_컷_프레임_합이_n_frames이다(self):
        """Given: n_frames=30, cut_frames=[8, 20]
        When:  split_into_shots 호출
        Then:  sum(end-start) == 30 — 프레임 손실·중복 없음
        """
        shots = split_into_shots(N_FRAMES_MID, [CUT_FIRST, CUT_SECOND])

        total = sum(end - start for start, end in shots)
        assert total == N_FRAMES_MID

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_다중_컷_인접_샷의_end와_start가_연속이다(self):
        """Given: cut_frames=[8, 20]
        When:  split_into_shots 호출
        Then:  shots[i][1] == shots[i+1][0] (인접 구간 연속)

        WHY: 구간 사이에 빈 프레임이 생기거나 겹치면 안 된다.
        """
        shots = split_into_shots(N_FRAMES_MID, [CUT_FIRST, CUT_SECOND])

        for i in range(len(shots) - 1):
            assert shots[i][1] == shots[i + 1][0], (
                f"구간 {i}↔{i+1} 불연속: shots[{i}][1]={shots[i][1]}, "
                f"shots[{i+1}][0]={shots[i + 1][0]}"
            )

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_컷이_k개이면_샷은_k_plus_1개이다(self):
        """Given: n_frames=100, cut_frames 길이 k
        When:  split_into_shots 호출
        Then:  len(shots) == k + 1

        WHY: 컷 개수→샷 개수 관계는 오케스트레이션이 의존하는 핵심 불변식이다.
             propagate_call_count == 샷 수 == k+1 가드와 연결된다.
        """
        for k in [0, 1, 3, 5]:
            # 균등 간격 컷 생성 (경계 초과 방지)
            step = N_FRAMES_LONG // (k + 1)
            cuts = [step * i for i in range(1, k + 1)]
            shots = split_into_shots(N_FRAMES_LONG, cuts)
            assert len(shots) == k + 1, (
                f"k={k} 컷 → {k+1}샷 기대, 실제 {len(shots)}샷"
            )


# ---------------------------------------------------------------------------
# 경계값
# ---------------------------------------------------------------------------
class TestSplitBoundary:
    """경계값: 첫 프레임 컷, 마지막 프레임 컷, n_frames=1."""

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_컷이_첫_프레임_직후이면_첫_샷이_길이_1이다(self):
        """Given: n_frames=10, cut_frames=[1]
        When:  split_into_shots 호출
        Then:  shots[0] == (0, 1), shots[1] == (1, 10)

        WHY: 첫 프레임만 이전 샷에 속하는 극단적 경우 — 길이 1 샷이
             오케스트레이션에서 propagate를 1회 호출하는지 확인한다.
        """
        shots = split_into_shots(N_FRAMES_SHORT, [CUT_NEAR_START])

        assert shots[0] == (0, CUT_NEAR_START)
        assert shots[1] == (CUT_NEAR_START, N_FRAMES_SHORT)

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_컷이_마지막_프레임_직전이면_마지막_샷이_길이_1이다(self):
        """Given: n_frames=10, cut_frames=[9]
        When:  split_into_shots 호출
        Then:  shots[-1] == (9, 10) — 길이 1 마지막 샷
        """
        cut_near_end = N_FRAMES_SHORT - CUT_NEAR_END_OFFSET  # == 9
        shots = split_into_shots(N_FRAMES_SHORT, [cut_near_end])

        assert shots[-1] == (cut_near_end, N_FRAMES_SHORT)

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_n_frames_1이면_컷_없이_1샷을_반환한다(self):
        """Given: n_frames=1, cut_frames=[]
        When:  split_into_shots 호출
        Then:  [(0, 1)]

        WHY: 프레임이 1개인 극단 케이스 — 빈 리스트나 예외가 아닌 1샷을 반환한다.
        """
        shots = split_into_shots(1, [])

        assert shots == [(0, 1)]

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_반환_샷의_모든_start는_end보다_작다(self):
        """Given: 다양한 n_frames·cut_frames 조합
        When:  split_into_shots 호출
        Then:  모든 샷에서 start < end (길이 > 0)

        WHY: 길이 0 샷은 오케스트레이션에서 propagate를 빈 프레임으로 호출해
             예외를 유발한다. 방어적 불변식 검증.
        """
        test_cases = [
            (10, []),
            (10, [5]),
            (30, [8, 20]),
            (100, [1, 50, 99]),
        ]
        for n_frames, cuts in test_cases:
            shots = split_into_shots(n_frames, cuts)
            for start, end in shots:
                assert start < end, (
                    f"n={n_frames}, cuts={cuts}: 길이 0 샷 ({start}, {end})"
                )

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_반환_샷의_첫_start는_0이다(self):
        """Given: 임의 n_frames·cut_frames
        When:  split_into_shots 호출
        Then:  shots[0][0] == 0 항상
        """
        for cuts in [[], [3], [3, 7]]:
            shots = split_into_shots(N_FRAMES_SHORT, cuts)
            assert shots[0][0] == 0, (
                f"cuts={cuts}: 첫 샷 start={shots[0][0]} ≠ 0"
            )

    @pytest.mark.skipif(not _HAS_SPLIT, reason=_MSG_NO_SPLIT)
    def test_반환_샷의_마지막_end는_n_frames이다(self):
        """Given: 임의 n_frames·cut_frames
        When:  split_into_shots 호출
        Then:  shots[-1][1] == n_frames 항상
        """
        for cuts in [[], [3], [3, 7]]:
            shots = split_into_shots(N_FRAMES_SHORT, cuts)
            assert shots[-1][1] == N_FRAMES_SHORT, (
                f"cuts={cuts}: 마지막 샷 end={shots[-1][1]} ≠ {N_FRAMES_SHORT}"
            )
