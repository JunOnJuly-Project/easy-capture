"""컷별 오브젝트 선택 순수 모델 테스트 (수동 교정 슬라이스 — RED).

대상 모듈: easy_capture.core.tracking.cut_selection (신규 — 구현 전 RED)

배경:
  멀티샷 군무에서 자동 재매칭(IoU, feat=None)이 needs_correction 82.7%로
  구조적 실패. 해결책: 각 컷(샷) 시작 프레임에서 Grounding DINO 후보 검출 →
  사용자가 추적 대상을 명시 선택(CutSelection) → 그 선택으로 컷별 재추적.
  자동 재매칭은 폴백.

검증 대상:
  CutSelection            — frozen dataclass(shot_index, point), 불변·동등성.
  index_selections_by_shot — 선택 리스트 → {shot_index: point} 매핑(순수).
  validate_selections     — shot_index 범위·중복 검증, 한국어 ValueError(순수).

설계 경계 불변식:
  순수 core — torch·transformers·PySide6·PyAV·scenedetect 비의존.
  import 자체가 순수성 가드(아래 _ensure_pure_imports 참조).

구현 전 RED 상태가 정상:
  core/tracking/cut_selection.py 미존재 → ImportError로 skip(개별 집계).
"""
from __future__ import annotations

import sys

import pytest

# --- cut_selection 미구현 → try/except 격리 ---
# WHY: 구현 전이므로 import 자체가 실패한다. 이 격리로 기존 테스트를
#      차단하지 않고 신규 테스트만 skip/fail로 개별 집계되게 한다.
try:
    from easy_capture.core.tracking.cut_selection import (
        CutSelection,
        index_selections_by_shot,
        validate_selections,
    )
    _HAS_CUT_SELECTION = True
except ImportError:
    CutSelection = None  # type: ignore[assignment,misc]
    index_selections_by_shot = None  # type: ignore[assignment]
    validate_selections = None  # type: ignore[assignment]
    _HAS_CUT_SELECTION = False

_MSG_NO_CUT_SELECTION = (
    "core/tracking/cut_selection.py에 CutSelection·"
    "index_selections_by_shot·validate_selections 미구현 — RED 예상"
)

# ---------------------------------------------------------------------------
# 테스트 상수 (매직넘버 금지)
# ---------------------------------------------------------------------------
# 샷 인덱스 — 0-기반
SHOT_0 = 0
SHOT_1 = 1
SHOT_2 = 2

# 클릭 포인트 (x, y) — 샷별 서로 다른 좌표로 매핑 정확성 검증
POINT_A = (320, 180)
POINT_B = (100, 200)
POINT_C = (500, 50)

# 전체 샷 수 (범위 검증 기준)
N_SHOTS_3 = 3

# 범위 밖 샷 인덱스 (0 <= i < N_SHOTS_3 위반)
SHOT_OUT_OF_RANGE = 3
SHOT_NEGATIVE = -1

# 순수성 가드 — import 금지 모듈 목록(core 경계 불변식)
_FORBIDDEN_MODULES = ("torch", "transformers", "PySide6", "av", "scenedetect")


# ---------------------------------------------------------------------------
# 순수성(core 경계) 가드
# ---------------------------------------------------------------------------
def _cut_selection_keeps_pure(forbidden_module: str) -> bool:
    """격리 서브프로세스에서 cut_selection만 import 후 forbidden 미로드 검증.

    WHY subprocess: 같은 pytest 세션의 UI 테스트(test_video_window_* 등)가 먼저
    PySide6 등을 로드하면 sys.modules에 잔류해 같은 프로세스 검사는 위양성이 난다.
    새 인터프리터로 격리해 cut_selection import만의 부수효과를 검사한다.
    """
    import subprocess
    import sys as _sys

    check_code = (
        "import sys; "
        "from easy_capture.core.tracking.cut_selection import CutSelection; "
        f"assert not any(k == '{forbidden_module}' or "
        f"k.startswith('{forbidden_module}.') for k in sys.modules), "
        f"'{forbidden_module} 로드됨'"
    )
    result = subprocess.run(
        [_sys.executable, "-c", check_code], capture_output=True, text=True
    )
    return result.returncode == 0


class TestCutSelectionPurity:
    """cut_selection 모듈이 무거운 의존을 끌어오지 않는지 검증(core 경계 불변식)."""

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.parametrize("forbidden", _FORBIDDEN_MODULES)
    def test_cut_selection_import_시_무거운_의존을_로드하지_않는다(self, forbidden):
        """Given: 격리 서브프로세스에서 cut_selection만 import
        When:  sys.modules에서 금지 모듈 확인
        Then:  torch·transformers·PySide6·av·scenedetect 미로드

        WHY: core는 순수 도메인 — GPU/UI/IO 라이브러리에 의존하면 안 된다.
             subprocess 격리로 같은 세션 타 테스트의 잔류 모듈 위양성을 회피한다.
        """
        assert _cut_selection_keeps_pure(forbidden), (
            f"cut_selection이 격리 import에서 금지 모듈 '{forbidden}'을 로드함 — "
            "core 순수성 위반"
        )


# ---------------------------------------------------------------------------
# CutSelection dataclass
# ---------------------------------------------------------------------------
class TestCutSelectionDataclass:
    """CutSelection: frozen dataclass — 값 보관·불변·동등성."""

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_CutSelection은_shot_index와_point를_보관한다(self):
        """Given: shot_index=1, point=(320, 180)
        When:  CutSelection 생성
        Then:  .shot_index == 1, .point == (320, 180)

        WHY: 사용자가 컷마다 선택한 추적 대상을 (샷, 클릭점) 한 단위로 보관한다.
        """
        selection = CutSelection(shot_index=SHOT_1, point=POINT_A)

        assert selection.shot_index == SHOT_1
        assert selection.point == POINT_A

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_CutSelection은_frozen이라_필드_수정_시_예외가_발생한다(self):
        """Given: 유효한 CutSelection 인스턴스
        When:  shot_index 필드 수정 시도
        Then:  FrozenInstanceError(또는 AttributeError) 발생

        WHY: frozen=True dataclass로 사용자 선택이 실수로 덮어씌워지는 버그를
             차단한다. TrackResult·Detection·RematchResult 패턴 계승.
        """
        from dataclasses import FrozenInstanceError

        selection = CutSelection(shot_index=SHOT_0, point=POINT_A)

        with pytest.raises((FrozenInstanceError, AttributeError)):
            selection.shot_index = SHOT_1  # type: ignore[misc]

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_CutSelection은_point_수정_시에도_예외가_발생한다(self):
        """Given: 유효한 CutSelection 인스턴스
        When:  point 필드 수정 시도
        Then:  FrozenInstanceError(또는 AttributeError) 발생
        """
        from dataclasses import FrozenInstanceError

        selection = CutSelection(shot_index=SHOT_0, point=POINT_A)

        with pytest.raises((FrozenInstanceError, AttributeError)):
            selection.point = POINT_B  # type: ignore[misc]

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_같은_값을_가진_CutSelection은_동등하다(self):
        """Given: shot_index·point가 같은 두 인스턴스
        When:  == 비교
        Then:  동등(True)

        WHY: frozen dataclass의 값 동등성(eq) — 캐시 비교·중복 검출에 활용.
        """
        a = CutSelection(shot_index=SHOT_1, point=POINT_A)
        b = CutSelection(shot_index=SHOT_1, point=POINT_A)

        assert a == b

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_shot_index가_다른_CutSelection은_동등하지_않다(self):
        """Given: point는 같지만 shot_index가 다른 두 인스턴스
        When:  == 비교
        Then:  비동등(False)
        """
        a = CutSelection(shot_index=SHOT_0, point=POINT_A)
        b = CutSelection(shot_index=SHOT_1, point=POINT_A)

        assert a != b


# ---------------------------------------------------------------------------
# index_selections_by_shot
# ---------------------------------------------------------------------------
class TestIndexSelectionsByShot:
    """index_selections_by_shot: 선택 리스트 → {shot_index: point} 매핑(순수)."""

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_빈_리스트면_빈_딕셔너리를_반환한다(self):
        """Given: selections=[], n_shots=3
        When:  index_selections_by_shot 호출
        Then:  {} 반환

        WHY: 선택이 하나도 없으면 전 샷이 자동 재매칭 폴백 — 빈 매핑이어야 한다.
        """
        result = index_selections_by_shot([], N_SHOTS_3)

        assert result == {}

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_정상_선택을_shot_index에서_point로_매핑한다(self):
        """Given: 샷0→POINT_A, 샷1→POINT_B, 샷2→POINT_C 선택
        When:  index_selections_by_shot 호출
        Then:  {0: POINT_A, 1: POINT_B, 2: POINT_C}

        WHY: 오케스트레이션이 샷 인덱스로 사용자 선택 클릭점을 O(1) 조회한다.
        """
        selections = [
            CutSelection(shot_index=SHOT_0, point=POINT_A),
            CutSelection(shot_index=SHOT_1, point=POINT_B),
            CutSelection(shot_index=SHOT_2, point=POINT_C),
        ]

        result = index_selections_by_shot(selections, N_SHOTS_3)

        assert result == {SHOT_0: POINT_A, SHOT_1: POINT_B, SHOT_2: POINT_C}

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_일부_샷만_선택하면_해당_샷만_매핑된다(self):
        """Given: 샷1만 선택(샷0·샷2 미선택), n_shots=3
        When:  index_selections_by_shot 호출
        Then:  {1: POINT_B} — 선택된 샷만 키로 존재

        WHY: 혼합 시나리오 — 선택 안 된 샷은 자동 재매칭 폴백으로 처리되므로
             매핑에 포함되지 않아야 한다.
        """
        selections = [CutSelection(shot_index=SHOT_1, point=POINT_B)]

        result = index_selections_by_shot(selections, N_SHOTS_3)

        assert result == {SHOT_1: POINT_B}
        assert SHOT_0 not in result
        assert SHOT_2 not in result

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_반환값은_shot_index를_키_point를_값으로_가진다(self):
        """Given: 샷0 선택
        When:  index_selections_by_shot 호출
        Then:  result[0] == POINT_A (값이 클릭점 튜플)
        """
        selections = [CutSelection(shot_index=SHOT_0, point=POINT_A)]

        result = index_selections_by_shot(selections, N_SHOTS_3)

        assert result[SHOT_0] == POINT_A


# ---------------------------------------------------------------------------
# validate_selections — 범위·중복 검증
# ---------------------------------------------------------------------------
class TestValidateSelections:
    """validate_selections: shot_index 범위·중복 검증, 한국어 ValueError(순수)."""

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_정상_선택이면_예외없이_None을_반환한다(self):
        """Given: 범위 내·중복 없는 선택 3개, n_shots=3
        When:  validate_selections 호출
        Then:  예외 없이 통과(None 반환)

        WHY: 올바른 입력은 통과시켜야 한다 — 검증이 정상 흐름을 막으면 안 된다.
        """
        selections = [
            CutSelection(shot_index=SHOT_0, point=POINT_A),
            CutSelection(shot_index=SHOT_1, point=POINT_B),
            CutSelection(shot_index=SHOT_2, point=POINT_C),
        ]

        result = validate_selections(selections, N_SHOTS_3)

        assert result is None

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_빈_선택이면_예외없이_통과한다(self):
        """Given: selections=[], n_shots=3
        When:  validate_selections 호출
        Then:  예외 없이 통과

        WHY: 선택이 없으면 전 샷 자동 재매칭 폴백 — 유효한 입력이다.
        """
        result = validate_selections([], N_SHOTS_3)

        assert result is None

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_shot_index가_n_shots_이상이면_ValueError가_발생한다(self):
        """Given: shot_index=3, n_shots=3 (0<=i<3 위반)
        When:  validate_selections 호출
        Then:  ValueError 발생

        WHY: 존재하지 않는 샷을 가리키는 선택은 오케스트레이션에서 IndexError를
             유발한다. 진입 시점에 명시적 예외로 차단한다.
        """
        selections = [CutSelection(shot_index=SHOT_OUT_OF_RANGE, point=POINT_A)]

        with pytest.raises(ValueError):
            validate_selections(selections, N_SHOTS_3)

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_범위_밖_ValueError_메시지는_한국어와_수치를_포함한다(self):
        """Given: shot_index=3, n_shots=3
        When:  validate_selections 호출 → ValueError
        Then:  메시지에 한국어 안내와 위반 수치(3)가 포함된다

        WHY: 사용자 대면 에러는 한국어여야 하며(글로벌 지침), 어떤 값이
             범위를 벗어났는지 수치로 알려야 디버깅·UI 안내가 가능하다.
        """
        selections = [CutSelection(shot_index=SHOT_OUT_OF_RANGE, point=POINT_A)]

        with pytest.raises(ValueError, match=str(SHOT_OUT_OF_RANGE)):
            validate_selections(selections, N_SHOTS_3)

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_shot_index가_음수이면_ValueError가_발생한다(self):
        """Given: shot_index=-1, n_shots=3 (0<=i 위반)
        When:  validate_selections 호출
        Then:  ValueError 발생

        WHY: 음수 인덱스는 파이썬에서 뒤에서부터 접근해 조용한 버그가 된다.
             0 하한도 명시적으로 검증한다.
        """
        selections = [CutSelection(shot_index=SHOT_NEGATIVE, point=POINT_A)]

        with pytest.raises(ValueError):
            validate_selections(selections, N_SHOTS_3)

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_중복_shot_index가_있으면_ValueError가_발생한다(self):
        """Given: 같은 shot_index=1을 가리키는 선택 2개, n_shots=3
        When:  validate_selections 호출
        Then:  ValueError 발생

        WHY: 한 샷에 두 개의 추적 대상 선택이 들어오면 어느 것을 쓸지
             모호하다. 중복은 진입 시점에 거부한다.
        """
        selections = [
            CutSelection(shot_index=SHOT_1, point=POINT_A),
            CutSelection(shot_index=SHOT_1, point=POINT_B),
        ]

        with pytest.raises(ValueError):
            validate_selections(selections, N_SHOTS_3)

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_중복_ValueError_메시지는_한국어를_포함한다(self):
        """Given: 중복 shot_index 선택
        When:  validate_selections 호출 → ValueError
        Then:  메시지에 '중복' 한국어 안내가 포함된다

        WHY: 사용자가 같은 샷을 두 번 선택했음을 한국어로 명확히 안내한다.
        """
        selections = [
            CutSelection(shot_index=SHOT_1, point=POINT_A),
            CutSelection(shot_index=SHOT_1, point=POINT_B),
        ]

        with pytest.raises(ValueError, match="중복"):
            validate_selections(selections, N_SHOTS_3)
