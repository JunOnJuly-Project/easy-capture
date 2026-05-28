"""샷 경계 재매칭 순수 함수 테스트.

대상 모듈:
  easy_capture.core.tracking.rematch   — iou, rematch_score (기존 회귀)
  easy_capture.core.tracking.rematch   — REMATCH_THRESHOLD, RematchResult,
                                         select_best_match (신규 — 구현 전 RED)

테스트 범주:
  1. iou 회귀 — 기존 테스트 유지
  2. rematch_score 회귀 — 기존 테스트 유지
  3. RematchResult 불변 dataclass 계약
  4. select_best_match — 통과(≥0.5)/미달(<0.5)/다중 후보 argmax/빈 후보 4케이스

구현 전 RED 상태가 정상:
  REMATCH_THRESHOLD·RematchResult·select_best_match 미구현.
"""
from __future__ import annotations

import pytest

from easy_capture.core.tracking import iou, rematch_score

# ---------------------------------------------------------------------------
# 신규 심볼 — 구현 전이므로 try/except 격리
# WHY: 미구현 시 import 실패가 기존 iou/rematch_score 회귀 테스트까지
#      차단하지 않도록 한다. 신규 테스트만 개별 skip/fail 처리된다.
# ---------------------------------------------------------------------------
try:
    from easy_capture.core.tracking.rematch import (
        REMATCH_THRESHOLD,
        RematchResult,
        select_best_match,
    )
    _HAS_SELECT_BEST = True
except ImportError:
    REMATCH_THRESHOLD = None  # type: ignore[assignment]
    RematchResult = None      # type: ignore[assignment]
    select_best_match = None  # type: ignore[assignment]
    _HAS_SELECT_BEST = False

# DetectionBackend·Detection — 구현 전 격리
try:
    from easy_capture.core.segmentation.detection_backend import Detection
    _HAS_DETECTION = True
except ImportError:
    Detection = None  # type: ignore[assignment]
    _HAS_DETECTION = False

_MSG_NO_SELECT_BEST = (
    "core/tracking/rematch.py에 REMATCH_THRESHOLD·RematchResult·select_best_match 미구현 — RED 예상"
)
_MSG_NO_DETECTION = (
    "core/segmentation/detection_backend.py에 Detection 미구현 — RED 예상"
)

# ---------------------------------------------------------------------------
# 테스트 상수 (매직넘버 금지)
# ---------------------------------------------------------------------------
# IoU 검증용 동일 박스
BOX_SAME: tuple = (0, 0, 10, 10)

# IoU 검증용 분리 박스
BOX_FAR: tuple = (20, 20, 30, 30)

# rematch_score 검증: 가까운 후보 — (1,1,11,11)은 (0,0,10,10)과 높은 IoU
BOX_NEAR_CAND: tuple = (1, 1, 11, 11)
# 점수 통과 기준
SCORE_PASS_THRESHOLD = 0.5

# select_best_match 테스트용 직전 박스
PREV_BOX: tuple = (100, 100, 200, 200)

# 가까운 후보 — PREV_BOX와 거의 겹침 → score ≥ 0.5
BOX_CLOSE: tuple = (105, 105, 205, 205)
# 먼 후보 — PREV_BOX와 전혀 겹치지 않음 → score < 0.5
BOX_DISTANT: tuple = (500, 500, 600, 600)


# ---------------------------------------------------------------------------
# 기존 회귀: iou
# ---------------------------------------------------------------------------
class TestIouRegression:
    """iou 함수 기존 계약 회귀 검증."""

    def test_동일한_박스의_IoU는_1이다(self):
        """Given: 동일한 두 박스
        When:  iou 호출
        Then:  1.0 반환
        """
        assert iou(BOX_SAME, BOX_SAME) == 1.0

    def test_겹치지_않는_박스의_IoU는_0이다(self):
        """Given: 완전히 분리된 두 박스
        When:  iou 호출
        Then:  0.0 반환
        """
        assert iou(BOX_SAME, BOX_FAR) == 0.0


# ---------------------------------------------------------------------------
# 기존 회귀: rematch_score
# ---------------------------------------------------------------------------
class TestRematchScoreRegression:
    """rematch_score 함수 기존 계약 회귀 검증."""

    def test_위치만으로_가까운_후보는_먼_후보보다_점수가_높다(self):
        """Given: 위치 기반(feat 없음)
        When:  가까운·먼 후보 각각 rematch_score
        Then:  가까운 점수 > 0.5 > 먼 점수
        """
        near = rematch_score(BOX_SAME, BOX_NEAR_CAND)
        far = rematch_score(BOX_SAME, BOX_FAR)
        assert near > SCORE_PASS_THRESHOLD > far

    def test_외형_특징이_일치하면_점수가_높다(self):
        """Given: 동일 위치 + 동일 특징 벡터
        When:  rematch_score(feat 있음)
        Then:  score > 0.9
        """
        score = rematch_score(BOX_SAME, BOX_SAME, [1, 0], [1, 0])
        assert score > 0.9


# ---------------------------------------------------------------------------
# RematchResult 불변 dataclass 계약
# ---------------------------------------------------------------------------
class TestRematchResult:
    """RematchResult frozen dataclass 계약 검증."""

    @pytest.mark.skipif(not _HAS_SELECT_BEST, reason=_MSG_NO_SELECT_BEST)
    def test_RematchResult_필드가_올바르게_저장된다(self):
        """Given: best_index=1, score=0.7, passed=True
        When:  RematchResult 생성
        Then:  각 필드가 입력값과 일치
        """
        result = RematchResult(best_index=1, score=0.7, passed=True)

        assert result.best_index == 1
        assert result.score == pytest.approx(0.7)
        assert result.passed is True

    @pytest.mark.skipif(not _HAS_SELECT_BEST, reason=_MSG_NO_SELECT_BEST)
    def test_RematchResult는_frozen_dataclass여서_수정_시_오류가_발생한다(self):
        """Given: RematchResult 인스턴스
        When:  필드 수정 시도
        Then:  AttributeError 또는 FrozenInstanceError 발생

        WHY: frozen=True는 판정 결과가 실수로 덮어씌워지는 버그를 방지한다.
        """
        result = RematchResult(best_index=0, score=0.6, passed=True)

        with pytest.raises((AttributeError, TypeError)):
            result.best_index = 99  # type: ignore[misc]

    @pytest.mark.skipif(not _HAS_SELECT_BEST, reason=_MSG_NO_SELECT_BEST)
    def test_RematchResult_미달_기본값은_False이다(self):
        """Given: passed=False
        When:  RematchResult 생성
        Then:  passed == False (타입 일관성)
        """
        result = RematchResult(best_index=-1, score=0.0, passed=False)

        assert result.passed is False
        assert result.best_index == -1
        assert result.score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# select_best_match — 4케이스
# ---------------------------------------------------------------------------
class TestSelectBestMatch:
    """select_best_match 순수 함수 4케이스 검증.

    WHY: 후보 리스트에서 직전 bbox와 best 매칭 후보·점수·통과여부를 판정하는
         이 함수가 정확해야 재추적 오케스트레이션이 올바르게 동작한다.
         GPU 모델과 완전 독립된 순수 단위 테스트로 로직을 100% 검증한다.
    """

    @pytest.mark.skipif(not _HAS_SELECT_BEST, reason=_MSG_NO_SELECT_BEST)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_통과_후보_하나이면_passed_True이다(self):
        """Given: PREV_BOX, 가까운 후보 1개(IoU≥0.5)
        When:  select_best_match 호출
        Then:  best_index=0, score≥0.5, passed=True

        WHY: 컷 직후 동일 인물이 같은 위치에 있으면 재추적이 이어져야 한다.
        """
        candidates = [Detection(box=BOX_CLOSE, score=0.9, feat=None)]

        result = select_best_match(PREV_BOX, candidates)

        assert result.best_index == 0
        assert result.score >= REMATCH_THRESHOLD
        assert result.passed is True

    @pytest.mark.skipif(not _HAS_SELECT_BEST, reason=_MSG_NO_SELECT_BEST)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_미달_후보이면_passed_False이다(self):
        """Given: PREV_BOX, 먼 후보 1개(IoU<0.5)
        When:  select_best_match 호출
        Then:  score<0.5, passed=False

        WHY: 컷 후 인물이 완전히 다른 위치에 있으면 재매칭 실패로 처리해야 한다.
        """
        candidates = [Detection(box=BOX_DISTANT, score=0.8, feat=None)]

        result = select_best_match(PREV_BOX, candidates)

        assert result.score < REMATCH_THRESHOLD
        assert result.passed is False

    @pytest.mark.skipif(not _HAS_SELECT_BEST, reason=_MSG_NO_SELECT_BEST)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_다중_후보_중_가까운_것이_best로_선택된다(self):
        """Given: PREV_BOX, 가까운 후보 + 먼 후보 혼재
        When:  select_best_match 호출
        Then:  best_index가 가까운 후보의 인덱스, passed=True

        WHY: argmax 선택 정확성 — 여러 인물 중 직전 bbox와 가장 겹치는 인물을
             골라야 틀린 인물로 추적이 점프하지 않는다.
        """
        # 먼 후보가 index=0, 가까운 후보가 index=1
        candidates = [
            Detection(box=BOX_DISTANT, score=0.9, feat=None),  # 0 — 먼 것
            Detection(box=BOX_CLOSE, score=0.7, feat=None),    # 1 — 가까운 것
        ]

        result = select_best_match(PREV_BOX, candidates)

        assert result.best_index == 1, (
            f"argmax 실패: best_index={result.best_index}, "
            "가까운 후보(index=1)가 선택돼야 함"
        )
        assert result.passed is True

    @pytest.mark.skipif(not _HAS_SELECT_BEST, reason=_MSG_NO_SELECT_BEST)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_빈_후보_리스트이면_best_index_minus1_score_0_passed_False이다(self):
        """Given: PREV_BOX, 빈 후보 리스트
        When:  select_best_match 호출
        Then:  RematchResult(best_index=-1, score=0.0, passed=False)

        WHY: 컷 후 화면에 인물이 없거나 검출기가 아무것도 찾지 못하면
             재매칭 실패로 처리해야 한다(빈 검출 = 미달과 동일 처리).
        """
        result = select_best_match(PREV_BOX, [])

        assert result.best_index == -1
        assert result.score == pytest.approx(0.0)
        assert result.passed is False

    @pytest.mark.skipif(not _HAS_SELECT_BEST, reason=_MSG_NO_SELECT_BEST)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_커스텀_threshold_적용이_통과_여부에_반영된다(self):
        """Given: 가까운 후보 1개, threshold=0.9(높은 기준)
        When:  select_best_match(threshold=0.9) 호출
        Then:  기본 threshold(0.5)로는 통과하지만 0.9로는 미달

        WHY: threshold를 주입할 수 있어야 Colab에서 민감도를 조정할 수 있다.
        """
        candidates = [Detection(box=BOX_CLOSE, score=0.9, feat=None)]

        result_default = select_best_match(PREV_BOX, candidates)
        result_strict = select_best_match(PREV_BOX, candidates, threshold=0.9)

        # 기본 threshold=0.5로는 통과
        assert result_default.passed is True
        # 기본 점수가 0.9보다 낮으면 엄격 threshold로는 미달
        # (BOX_CLOSE의 IoU가 1.0이 아닐 수 있으므로 score로 조건부 검증)
        if result_default.score < 0.9:
            assert result_strict.passed is False
        # score가 0.9 이상이면 엄격 기준도 통과 — 그래도 passed 필드는 논리적으로 일관
        assert result_strict.score == pytest.approx(result_default.score)

    @pytest.mark.skipif(not _HAS_SELECT_BEST, reason=_MSG_NO_SELECT_BEST)
    def test_REMATCH_THRESHOLD_기본값은_0점5이다(self):
        """Given: REMATCH_THRESHOLD 상수
        When:  값 확인
        Then:  0.5

        WHY: ADR 0006 명시 기본값 — 이 상수가 바뀌면 모든 재매칭 판정이 달라진다.
        """
        assert REMATCH_THRESHOLD == pytest.approx(0.5)
