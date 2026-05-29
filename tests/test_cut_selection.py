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

# --- validate_negative_points 미구현 → 별도 격리(negative point 슬라이스 — RED) ---
# WHY: negative point("이 점=옆 멤버는 대상 아님", SAM2 label 0) 검증 함수는 신규다.
#      함수 자체가 없으면 negative 검증 테스트만 skip되고 기존 테스트는 통과한다.
try:
    from easy_capture.core.tracking.cut_selection import (
        validate_negative_points,
    )
    _HAS_VALIDATE_NEGATIVES = True
except ImportError:
    validate_negative_points = None  # type: ignore[assignment]
    _HAS_VALIDATE_NEGATIVES = False

_MSG_NO_VALIDATE_NEGATIVES = (
    "core/tracking/cut_selection.py에 validate_negative_points 미구현 — RED 예상"
)

# --- Task 4-1: core 변환 순수 함수 격리(데스크톱 컷별 선택 UI 공유 — RED) ---
# WHY: 데스크톱 GUI는 후보 인덱스(int)만 다루고, 박스 중심 변환·박스 히트테스트·
#      dict(SHOT_TARGETS/NEG_TARGETS)→CutSelection 빌드를 core 순수 함수에 위임한다
#      (노트북 셀 7.5 로직 이식, core→app 역참조 회피). 신규 심볼이 없으면 변환
#      테스트만 skip되고 기존 테스트(위 import 블록)는 통과한다(무회귀 격리).
try:
    from easy_capture.core.tracking.cut_selection import (
        ShotChoice,
        box_center,
        build_selections_from_choices,
        pick_box_at,
    )
    _HAS_CONVERTERS = True
except ImportError:
    ShotChoice = None  # type: ignore[assignment,misc]
    box_center = None  # type: ignore[assignment]
    build_selections_from_choices = None  # type: ignore[assignment]
    pick_box_at = None  # type: ignore[assignment]
    _HAS_CONVERTERS = False

_MSG_NO_CONVERTERS = (
    "core/tracking/cut_selection.py에 box_center·pick_box_at·ShotChoice·"
    "build_selections_from_choices 미구현 — Task 4-1 RED 예상"
)

# CutSelection.negative_points 필드 존재 여부 판별 — 미구현 시 negative 테스트만 skip.
# WHY: negative_points는 신규(하위호환 default ()) 필드다. dataclasses.fields로
#      존재를 판별해, 필드 추가 전에는 negative 테스트만 skip되고 기존은 통과한다.
_HAS_NEGATIVES_FIELD = False
if _HAS_CUT_SELECTION:
    from dataclasses import fields as _dc_fields_neg

    _HAS_NEGATIVES_FIELD = any(
        f.name == "negative_points" for f in _dc_fields_neg(CutSelection)
    )

_MSG_NO_NEGATIVES_FIELD = (
    "CutSelection.negative_points 필드 미구현 — negative point RED 예상"
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

# box 프롬프트 좌표 (x1, y1, x2, y2) — detect 전신 bbox를 흉내 낸 합성 박스.
# WHY: box 프롬프트(detect bbox→SAM2) 도입으로 CutSelection이 point 외에
#      선택 대상의 전신 bbox(box)를 함께 보관해야 한다(Story D).
BOX_A = (100.0, 50.0, 200.0, 300.0)
BOX_B = (300.0, 80.0, 380.0, 320.0)

# negative point 좌표 (x, y) — "이 점=옆 멤버는 대상 아님"(SAM2 label 0).
# WHY: 군무 밀착 구간에서 box+positive만으론 대상+옆사람이 한 덩어리로 합쳐진다.
#      negative point로 옆사람 경계를 가른다(Story A — negative point 슬라이스).
NEG_POINT_A = (250, 200)   # 옆 멤버(우측) 위치
NEG_POINT_B = (420, 60)    # 다른 옆 멤버(좌상단) 위치

# 프레임 크기 (W, H) — negative 좌표 범위 검증 기준
FRAME_W = 640
FRAME_H = 360
FRAME_SIZE = (FRAME_W, FRAME_H)

# 프레임 밖 negative 좌표(0<=x<W, 0<=y<H 위반) — 범위 검증용
NEG_OUT_X = (FRAME_W, 100)       # x == W → 위반(x < W 아님)
NEG_OUT_Y = (100, FRAME_H + 5)   # y >= H → 위반
NEG_NEGATIVE_COORD = (-1, 50)    # 음수 좌표 → 위반

# CutSelection.box 필드 존재 여부 판별 — 미구현 시 box 테스트만 skip(무회귀 격리).
# WHY: box 필드는 신규(하위호환 default None)다. dataclasses.fields로 존재를
#      판별해, 필드 추가 전에는 box 테스트만 skip 처리되고 기존 테스트는 통과한다.
_HAS_BOX_FIELD = False
if _HAS_CUT_SELECTION:
    from dataclasses import fields as _dc_fields

    _HAS_BOX_FIELD = any(f.name == "box" for f in _dc_fields(CutSelection))

_MSG_NO_BOX_FIELD = "CutSelection.box 필드 미구현 — box 프롬프트 RED 예상"

# ---------------------------------------------------------------------------
# Task 4-1: core 변환 순수 함수 테스트 상수 (합성 box — 작게, 매직넘버 금지)
# ---------------------------------------------------------------------------
# box_center 검증용 — 짝수 합성 box로 중심이 정수로 떨어지게 한다.
# (10, 20, 30, 60) → cx=(10+30)/2=20, cy=(20+60)/2=40
BOX_CENTER_INPUT = (10.0, 20.0, 30.0, 60.0)
BOX_CENTER_EXPECTED = (20, 40)

# box_center 정수 절단(int) 검증용 — 홀수 합으로 .5가 버려져 내림된다.
# (0, 0, 11, 21) → cx=11/2=5.5→5, cy=21/2=10.5→10
BOX_CENTER_ODD_INPUT = (0.0, 0.0, 11.0, 21.0)
BOX_CENTER_ODD_EXPECTED = (5, 10)

# pick_box_at 합성 후보 박스 — 서로 겹치지 않는 작은 박스 2개(인덱스 위치 검증).
# HIT_BOX_0: 좌상단, HIT_BOX_1: 우하단. 클릭점이 어느 박스 안인지로 인덱스 매핑.
HIT_BOX_0 = (0.0, 0.0, 40.0, 40.0)       # 중심 (20, 20)
HIT_BOX_1 = (100.0, 100.0, 160.0, 160.0)  # 중심 (130, 130)
HIT_BOXES = [HIT_BOX_0, HIT_BOX_1]

PICK_INSIDE_BOX_0 = (10, 10)    # HIT_BOX_0 안 → idx 0
PICK_INSIDE_BOX_1 = (120, 120)  # HIT_BOX_1 안 → idx 1
PICK_OUTSIDE = (300, 300)       # 어느 박스에도 안 들어감 → None

# 겹침(중첩) 박스 — 큰 박스 안에 작은 박스. 겹치는 점은 '안쪽=최소 넓이' 선택.
# OUTER: 큰 박스(idx 0), INNER: 그 안의 작은 박스(idx 1). 둘 다 포함하는 점은 idx 1.
OVERLAP_OUTER = (0.0, 0.0, 100.0, 100.0)   # 넓이 10000
OVERLAP_INNER = (30.0, 30.0, 50.0, 50.0)   # 넓이 400 (더 작음 = 안쪽)
OVERLAP_BOXES = [OVERLAP_OUTER, OVERLAP_INNER]
PICK_OVERLAP_POINT = (40, 40)               # 두 박스 모두 포함 → 안쪽(idx 1)
PICK_ONLY_OUTER_POINT = (10, 10)            # OUTER만 포함(INNER 밖) → idx 0

# build_selections_from_choices 합성 후보 — 샷별 후보 박스 리스트(작은 박스).
# 샷 2개, 각 샷에 후보 2개. box only(ShotCandidates 아님 — core→app 역참조 회피).
SHOT0_BOX0 = (10.0, 10.0, 30.0, 50.0)    # 중심 (20, 30)
SHOT0_BOX1 = (60.0, 10.0, 100.0, 90.0)   # 중심 (80, 50)
SHOT1_BOX0 = (0.0, 0.0, 20.0, 40.0)      # 중심 (10, 20)
SHOT1_BOX1 = (50.0, 50.0, 90.0, 130.0)   # 중심 (70, 90)
CANDIDATE_BOXES = [
    [SHOT0_BOX0, SHOT0_BOX1],  # 샷 0 후보
    [SHOT1_BOX0, SHOT1_BOX1],  # 샷 1 후보
]
N_SHOTS_2 = 2  # CANDIDATE_BOXES 길이

# 후보 인덱스 (ShotChoice.target_idx / negative_idxs)
CAND_IDX_0 = 0
CAND_IDX_1 = 1
CAND_IDX_OUT_OF_RANGE = 5  # 후보 수(2) 초과 — 자기제외/범위밖 제외 검증용

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
# CutSelection.box — box 프롬프트 필드 (Story D, 하위호환 default None)
# ---------------------------------------------------------------------------
class TestCutSelectionBoxField:
    """CutSelection.box: 선택 대상 전신 bbox를 보관하는 신규 필드(하위호환).

    배경:
      box 프롬프트(detect bbox→SAM2) 도입으로 자동 재매칭/사용자 선택 시
      중심점(point) 대신 전신 bbox를 SAM2 box 프롬프트로 넘겨야 마스크가
      정확해진다(과대·옆사람 팔 포함 회귀 해결). CutSelection이 그 box를
      함께 보관할 수 있도록 box 필드를 추가하되, 기존 (shot_index, point)
      생성 코드는 절대 깨지면 안 되므로 default None으로 하위호환을 보장한다.
    """

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_BOX_FIELD, reason=_MSG_NO_BOX_FIELD)
    def test_box를_생략하면_기본값이_None이다(self):
        """Given: shot_index·point만 지정(box 생략)
        When:  CutSelection 생성
        Then:  .box == None

        WHY: box 필드는 하위호환을 위해 default None이어야 한다. 기존
             CutSelection(shot_index, point) 호출이 box 추가로 깨지면 안 된다.
        """
        selection = CutSelection(shot_index=SHOT_0, point=POINT_A)

        assert selection.box is None

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_BOX_FIELD, reason=_MSG_NO_BOX_FIELD)
    def test_box를_지정하면_그대로_보관한다(self):
        """Given: shot_index·point·box를 지정
        When:  CutSelection 생성
        Then:  .box == BOX_A (지정한 전신 bbox 그대로 보관)

        WHY: box 프롬프트 경로가 사용자 선택 대상의 전신 bbox를 SAM2에
             그대로 전달할 수 있도록 (x1, y1, x2, y2)를 보관해야 한다.
        """
        selection = CutSelection(shot_index=SHOT_0, point=POINT_A, box=BOX_A)

        assert selection.box == BOX_A

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_BOX_FIELD, reason=_MSG_NO_BOX_FIELD)
    def test_box_지정_시에도_frozen이라_수정하면_예외가_발생한다(self):
        """Given: box를 지정한 CutSelection 인스턴스
        When:  box 필드 수정 시도
        Then:  FrozenInstanceError(또는 AttributeError) 발생

        WHY: box도 frozen 불변식에 포함돼 사용자 선택이 실수로 덮어씌워지는
             버그를 차단한다(shot_index·point 패턴 계승).
        """
        from dataclasses import FrozenInstanceError

        selection = CutSelection(shot_index=SHOT_0, point=POINT_A, box=BOX_A)

        with pytest.raises((FrozenInstanceError, AttributeError)):
            selection.box = BOX_B  # type: ignore[misc]

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_BOX_FIELD, reason=_MSG_NO_BOX_FIELD)
    def test_box가_다르면_CutSelection은_동등하지_않다(self):
        """Given: shot_index·point는 같고 box만 다른 두 인스턴스
        When:  == 비교
        Then:  비동등(False)

        WHY: box도 값 동등성(eq)에 포함돼야 box가 바뀐 선택을 별개로 취급한다.
        """
        a = CutSelection(shot_index=SHOT_0, point=POINT_A, box=BOX_A)
        b = CutSelection(shot_index=SHOT_0, point=POINT_A, box=BOX_B)

        assert a != b

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_BOX_FIELD, reason=_MSG_NO_BOX_FIELD)
    def test_box_생략_생성과_box_None_명시_생성은_동등하다(self):
        """Given: box 생략 인스턴스와 box=None 명시 인스턴스
        When:  == 비교
        Then:  동등(True)

        WHY: 하위호환 default None이 명시 None과 동일하게 취급돼야
             기존 코드가 만든 인스턴스와 신규 코드가 만든 인스턴스가 일치한다.
        """
        omitted = CutSelection(shot_index=SHOT_1, point=POINT_B)
        explicit_none = CutSelection(shot_index=SHOT_1, point=POINT_B, box=None)

        assert omitted == explicit_none


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


# ===========================================================================
# Story A: CutSelection.negative_points — negative point 필드 (하위호환 default ())
# ===========================================================================
class TestCutSelectionNegativePointsField:
    """CutSelection.negative_points: 옆 멤버를 '대상 아님'으로 표시하는 신규 필드.

    배경:
      군무 밀착 구간에서 box+positive(point)만으로는 대상+옆사람이 맞닿은 한
      덩어리로 합쳐져 마스크가 부정확하다. negative point("이 점=옆 멤버는
      대상 아님", SAM2 label 0)로 경계를 가른다. CutSelection이 그 negative
      좌표 묶음을 함께 보관하되, 기존 (shot_index, point[, box]) 생성 코드는
      절대 깨지면 안 되므로 default ()로 하위호환을 보장한다(빈 튜플=negative 없음).
    """

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    def test_negative_points를_생략하면_기본값이_빈_튜플이다(self):
        """Given: shot_index·point만 지정(negative_points 생략)
        When:  CutSelection 생성
        Then:  .negative_points == () (빈 튜플)

        WHY: negative_points는 하위호환을 위해 default ()여야 한다. 기존
             CutSelection(shot_index, point) 호출이 필드 추가로 깨지면 안 된다.
        """
        selection = CutSelection(shot_index=SHOT_0, point=POINT_A)

        assert selection.negative_points == ()

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    def test_기존_shot_index_point_box_생성이_negative_필드_추가로_깨지지_않는다(self):
        """Given: 기존 방식대로 shot_index·point·box만 지정
        When:  CutSelection 생성
        Then:  예외 없이 생성되고 .negative_points == ()

        WHY: negative_points 필드는 box 다음에 추가되어도 기존 (shot_index,
             point, box) 위치 인자/키워드 인자 생성 코드를 깨면 안 된다(무회귀).
        """
        selection = CutSelection(shot_index=SHOT_0, point=POINT_A, box=BOX_A)

        assert selection.negative_points == ()
        assert selection.box == BOX_A

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    def test_negative_points를_지정하면_그대로_보관한다(self):
        """Given: shot_index·point·negative_points=(옆멤버1, 옆멤버2)
        When:  CutSelection 생성
        Then:  .negative_points == ((250, 200), (420, 60)) 그대로 보관

        WHY: negative point 경로가 옆 멤버 좌표(label 0)를 SAM2에 그대로 전달
             할 수 있도록 (x, y) 튜플의 묶음을 보관해야 한다.
        """
        negatives = (NEG_POINT_A, NEG_POINT_B)

        selection = CutSelection(
            shot_index=SHOT_0, point=POINT_A, negative_points=negatives
        )

        assert selection.negative_points == negatives

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    def test_negative_points_지정_시에도_frozen이라_수정하면_예외가_발생한다(self):
        """Given: negative_points를 지정한 CutSelection 인스턴스
        When:  negative_points 필드 수정 시도
        Then:  FrozenInstanceError(또는 AttributeError) 발생

        WHY: negative_points도 frozen 불변식에 포함돼 사용자 선택이 실수로
             덮어씌워지는 버그를 차단한다(shot_index·point·box 패턴 계승).
        """
        from dataclasses import FrozenInstanceError

        selection = CutSelection(
            shot_index=SHOT_0, point=POINT_A, negative_points=(NEG_POINT_A,)
        )

        with pytest.raises((FrozenInstanceError, AttributeError)):
            selection.negative_points = (NEG_POINT_B,)  # type: ignore[misc]

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    def test_negative_points가_다르면_CutSelection은_동등하지_않다(self):
        """Given: shot_index·point는 같고 negative_points만 다른 두 인스턴스
        When:  == 비교
        Then:  비동등(False)

        WHY: negative_points도 값 동등성(eq)에 포함돼야 옆 멤버 표시가 바뀐
             선택을 별개로 취급한다(캐시·중복 검출 정확성).
        """
        a = CutSelection(
            shot_index=SHOT_0, point=POINT_A, negative_points=(NEG_POINT_A,)
        )
        b = CutSelection(
            shot_index=SHOT_0, point=POINT_A, negative_points=(NEG_POINT_B,)
        )

        assert a != b

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    def test_negative_생략_생성과_빈_튜플_명시_생성은_동등하다(self):
        """Given: negative_points 생략 인스턴스와 negative_points=() 명시 인스턴스
        When:  == 비교
        Then:  동등(True)

        WHY: 하위호환 default ()가 명시 ()와 동일하게 취급돼야 기존 코드가
             만든 인스턴스와 신규 코드가 만든 인스턴스가 일치한다.
        """
        omitted = CutSelection(shot_index=SHOT_1, point=POINT_B)
        explicit_empty = CutSelection(
            shot_index=SHOT_1, point=POINT_B, negative_points=()
        )

        assert omitted == explicit_empty


# ===========================================================================
# Story A: validate_negative_points — negative 좌표 검증(순수, 한국어 ValueError)
# ===========================================================================
class TestValidateNegativePoints:
    """validate_negative_points: 프레임 밖·positive 동일좌표를 한국어 ValueError로 차단.

    계약(RED):
      validate_negative_points(selection, frame_size) -> None
        - 빈 negatives → 통과(예외 없음).
        - 모든 negative가 프레임 안(0<=x<w, 0<=y<h) + positive(point)와 다름 → 통과.
        - negative 좌표가 프레임 밖 → 한국어 ValueError.
        - negative 좌표가 positive(point)와 동일 → 한국어 ValueError(무의미 클릭).
      순수 — torch/PySide6 미import(기존 _cut_selection_keeps_pure 가드로 커버).
    """

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(
        not _HAS_VALIDATE_NEGATIVES, reason=_MSG_NO_VALIDATE_NEGATIVES
    )
    def test_빈_negatives면_예외없이_통과한다(self):
        """Given: negative_points=() 인 selection, frame_size=(640, 360)
        When:  validate_negative_points 호출
        Then:  예외 없이 통과(None 반환)

        WHY: negative point는 선택 사항이다. 없으면(빈 튜플) 검증을 통과시켜야
             기존 box+positive만 쓰는 경로가 막히지 않는다(무회귀).
        """
        selection = CutSelection(
            shot_index=SHOT_0, point=POINT_A, negative_points=()
        )

        result = validate_negative_points(selection, FRAME_SIZE)

        assert result is None

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(
        not _HAS_VALIDATE_NEGATIVES, reason=_MSG_NO_VALIDATE_NEGATIVES
    )
    def test_프레임_안_negatives면_예외없이_통과한다(self):
        """Given: 프레임 안(0<=x<640, 0<=y<360) negative 2개, positive와 다름
        When:  validate_negative_points 호출
        Then:  예외 없이 통과(None 반환)

        WHY: 정상 negative 좌표는 통과시켜야 한다 — 검증이 정상 흐름을 막으면 안 된다.
        """
        selection = CutSelection(
            shot_index=SHOT_0,
            point=POINT_A,
            negative_points=(NEG_POINT_A, NEG_POINT_B),
        )

        result = validate_negative_points(selection, FRAME_SIZE)

        assert result is None

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(
        not _HAS_VALIDATE_NEGATIVES, reason=_MSG_NO_VALIDATE_NEGATIVES
    )
    def test_negative_x가_프레임_너비_이상이면_ValueError가_발생한다(self):
        """Given: negative_points=((640, 100),) — x == W(640) 위반(x < W 아님)
        When:  validate_negative_points 호출
        Then:  ValueError 발생

        WHY: 프레임 밖 negative 좌표는 SAM2에 무의미하고 좌표계 오류를 유발한다.
             x < w 상한을 명시적으로 검증한다(0-기반 인덱스 경계).
        """
        selection = CutSelection(
            shot_index=SHOT_0, point=POINT_A, negative_points=(NEG_OUT_X,)
        )

        with pytest.raises(ValueError):
            validate_negative_points(selection, FRAME_SIZE)

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(
        not _HAS_VALIDATE_NEGATIVES, reason=_MSG_NO_VALIDATE_NEGATIVES
    )
    def test_negative_y가_프레임_높이_이상이면_ValueError가_발생한다(self):
        """Given: negative_points=((100, 365),) — y >= H(360) 위반
        When:  validate_negative_points 호출
        Then:  ValueError 발생

        WHY: y 축도 0<=y<h 범위를 벗어나면 프레임 밖이다. x와 대칭으로 검증한다.
        """
        selection = CutSelection(
            shot_index=SHOT_0, point=POINT_A, negative_points=(NEG_OUT_Y,)
        )

        with pytest.raises(ValueError):
            validate_negative_points(selection, FRAME_SIZE)

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(
        not _HAS_VALIDATE_NEGATIVES, reason=_MSG_NO_VALIDATE_NEGATIVES
    )
    def test_negative_좌표가_음수이면_ValueError가_발생한다(self):
        """Given: negative_points=((-1, 50),) — x 음수(0<=x 위반)
        When:  validate_negative_points 호출
        Then:  ValueError 발생

        WHY: 음수 좌표는 numpy에서 뒤에서부터 접근해 조용한 버그가 된다.
             0 하한도 명시적으로 검증한다(positive 검증과 동형).
        """
        selection = CutSelection(
            shot_index=SHOT_0,
            point=POINT_A,
            negative_points=(NEG_NEGATIVE_COORD,),
        )

        with pytest.raises(ValueError):
            validate_negative_points(selection, FRAME_SIZE)

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(
        not _HAS_VALIDATE_NEGATIVES, reason=_MSG_NO_VALIDATE_NEGATIVES
    )
    def test_프레임_밖_ValueError_메시지는_한국어를_포함한다(self):
        """Given: 프레임 밖 negative 좌표
        When:  validate_negative_points 호출 → ValueError
        Then:  메시지에 한국어 안내('프레임' 또는 '벗어')가 포함된다

        WHY: 사용자 대면 에러는 한국어여야 한다(글로벌 지침). 어떤 negative
             좌표가 프레임을 벗어났는지 한국어로 안내해야 UI 표시가 가능하다.
        """
        selection = CutSelection(
            shot_index=SHOT_0, point=POINT_A, negative_points=(NEG_OUT_X,)
        )

        with pytest.raises(ValueError, match="프레임|벗어"):
            validate_negative_points(selection, FRAME_SIZE)

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(
        not _HAS_VALIDATE_NEGATIVES, reason=_MSG_NO_VALIDATE_NEGATIVES
    )
    def test_negative가_positive와_동일_좌표이면_ValueError가_발생한다(self):
        """Given: negative_points=(POINT_A,) — positive(point)와 동일 좌표
        When:  validate_negative_points 호출
        Then:  ValueError 발생

        WHY: 같은 점을 동시에 '대상(positive)'이자 '대상 아님(negative)'으로
             지정하면 모순이다. SAM2에 무의미·충돌하는 입력이므로 차단한다.
        """
        selection = CutSelection(
            shot_index=SHOT_0, point=POINT_A, negative_points=(POINT_A,)
        )

        with pytest.raises(ValueError):
            validate_negative_points(selection, FRAME_SIZE)

    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(
        not _HAS_VALIDATE_NEGATIVES, reason=_MSG_NO_VALIDATE_NEGATIVES
    )
    def test_positive와_동일_좌표_ValueError_메시지는_한국어를_포함한다(self):
        """Given: positive와 동일한 negative 좌표
        When:  validate_negative_points 호출 → ValueError
        Then:  메시지에 한국어 안내('동일' 또는 '같')가 포함된다

        WHY: 사용자가 대상 클릭과 같은 점을 negative로 찍었음을 한국어로
             명확히 안내해 모순 입력을 정정하게 한다.
        """
        selection = CutSelection(
            shot_index=SHOT_0, point=POINT_A, negative_points=(POINT_A,)
        )

        with pytest.raises(ValueError, match="동일|같"):
            validate_negative_points(selection, FRAME_SIZE)


# ===========================================================================
# Task 4-1: box_center — (x1,y1,x2,y2) → 정수 중심 (노트북 셀 7.5 _box_center 이식)
# ===========================================================================
class TestBoxCenter:
    """box_center: 박스 (x1, y1, x2, y2)의 정수 중심 (cx, cy)을 반환(순수).

    계약(RED):
      box_center(box) -> tuple[int, int]
        - cx = int((x1 + x2) / 2), cy = int((y1 + y2) / 2) — float 좌표 → 정수.
      배경: SAM2 add_click(point=(x, y))는 정수 좌표를 받는다. 데스크톱 UI는
      후보 박스만 알고, 클릭점 변환은 core가 담당한다(노트북 셀 7.5 _box_center).
    """

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_박스의_중심을_정수_좌표로_반환한다(self):
        """Given: box=(10, 20, 30, 60)
        When:  box_center 호출
        Then:  (20, 40) — cx=(10+30)/2=20, cy=(20+60)/2=40

        WHY: detect 전신 bbox의 중심을 SAM2 클릭점(point)으로 변환하는 핵심 변환.
        """
        assert box_center(BOX_CENTER_INPUT) == BOX_CENTER_EXPECTED

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_중심이_소수면_int로_절단한다(self):
        """Given: box=(0, 0, 11, 21) — 중심이 (5.5, 10.5)
        When:  box_center 호출
        Then:  (5, 10) — int() 절단(내림)

        WHY: SAM2 클릭점은 정수 픽셀이어야 한다. 셀 7.5의 int(...) 절단을 보존한다.
        """
        assert box_center(BOX_CENTER_ODD_INPUT) == BOX_CENTER_ODD_EXPECTED

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_반환값은_정수_튜플이다(self):
        """Given: float 좌표 box
        When:  box_center 호출
        Then:  반환 원소가 모두 int 타입

        WHY: float를 그대로 넘기면 SAM2 add_click이 타입 오류를 낼 수 있다.
        """
        cx, cy = box_center(BOX_CENTER_INPUT)

        assert isinstance(cx, int)
        assert isinstance(cy, int)


# ===========================================================================
# Task 4-1: pick_box_at — 클릭점 포함 박스 중 최소 넓이 인덱스(겹침 시 안쪽)
# ===========================================================================
class TestPickBoxAt:
    """pick_box_at: 점을 포함하는 후보 박스의 인덱스(겹침 시 최소 넓이), 밖이면 None.

    계약(RED):
      pick_box_at(point, boxes) -> int | None
        - point를 포함하는 박스가 1개 → 그 인덱스.
        - 여러 박스가 포함 → 가장 작은 넓이 박스의 인덱스(겹침 시 안쪽 우선).
        - 어느 박스에도 안 들어감 → None.
      배경: 데스크톱 UI는 사용자가 캔버스를 클릭한 좌표만 안다. 그 좌표가 어느
      후보 박스(인덱스)인지 판정하는 히트테스트를 core가 담당한다(UI는 int만 다룸).
    """

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_박스_안의_점이면_그_인덱스를_반환한다(self):
        """Given: 겹치지 않는 박스 2개, 점이 박스 0 안
        When:  pick_box_at 호출
        Then:  0 (점을 포함하는 박스의 인덱스)

        WHY: 사용자가 클릭한 좌표를 후보 인덱스로 변환하는 기본 히트테스트.
        """
        assert pick_box_at(PICK_INSIDE_BOX_0, HIT_BOXES) == CAND_IDX_0

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_다른_박스_안의_점이면_해당_인덱스를_반환한다(self):
        """Given: 겹치지 않는 박스 2개, 점이 박스 1 안
        When:  pick_box_at 호출
        Then:  1 — 인덱스 매핑이 위치별로 정확

        WHY: 인덱스가 항상 0으로 고정되는 버그를 막는다(위치별 분기 검증).
        """
        assert pick_box_at(PICK_INSIDE_BOX_1, HIT_BOXES) == CAND_IDX_1

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_어느_박스에도_없으면_None을_반환한다(self):
        """Given: 박스 2개, 점이 모든 박스 밖
        When:  pick_box_at 호출
        Then:  None — 빈 영역 클릭은 선택 없음

        WHY: 배경(빈 영역) 클릭은 추적 대상이 아니다. UI가 None으로 '선택 해제'를
             판단할 수 있어야 한다(예외가 아니라 None 신호).
        """
        assert pick_box_at(PICK_OUTSIDE, HIT_BOXES) is None

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_겹친_박스_중_점을_포함하면_더_작은_박스의_인덱스를_반환한다(self):
        """Given: 큰 박스(idx 0) 안에 작은 박스(idx 1), 점이 둘 다 포함
        When:  pick_box_at 호출
        Then:  1 (더 작은 넓이 = 안쪽 박스)

        WHY: 군무에서 큰 박스가 작은 박스를 덮을 때, 사용자가 안쪽을 클릭하면
             의도한 안쪽 후보(작은 박스)를 골라야 한다(겹침 모호성 해소 규칙).
        """
        assert pick_box_at(PICK_OVERLAP_POINT, OVERLAP_BOXES) == CAND_IDX_1

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_안쪽_박스_밖이고_바깥_박스만_포함하면_바깥_인덱스를_반환한다(self):
        """Given: 큰 박스(idx 0) 안에 작은 박스(idx 1), 점이 큰 박스만 포함
        When:  pick_box_at 호출
        Then:  0 (작은 박스 밖이므로 유일하게 포함하는 큰 박스)

        WHY: 최소 넓이 규칙이 '포함하는 박스들 중에서만' 적용돼야 한다.
             점을 포함하지 않는 작은 박스를 잘못 고르면 안 된다.
        """
        assert pick_box_at(PICK_ONLY_OUTER_POINT, OVERLAP_BOXES) == CAND_IDX_0

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_빈_박스_목록이면_None을_반환한다(self):
        """Given: boxes=[] (후보 없음)
        When:  pick_box_at 호출
        Then:  None — 후보가 없으면 선택할 수 없음

        WHY: 검출 후보가 0개인 샷에서 클릭이 들어와도 예외 없이 None이어야 한다.
        """
        assert pick_box_at(PICK_INSIDE_BOX_0, []) is None


# ===========================================================================
# Task 4-1: ShotChoice — UI가 다루는 인덱스 묶음(target_idx + negative_idxs)
# ===========================================================================
class TestShotChoice:
    """ShotChoice: 한 샷의 사용자 선택(인덱스만) — target_idx, negative_idxs.

    계약(RED):
      ShotChoice(target_idx: int | None, negative_idxs: tuple[int, ...])
        - target_idx: 대상 후보 인덱스(None=미선택→폴백).
        - negative_idxs: 배제할 옆 멤버 후보 인덱스 묶음.
      배경: 데스크톱 UI는 인덱스(int)만 안다. core가 ShotChoice(인덱스) →
      CutSelection(좌표·box)으로 변환한다. UI/core 경계의 입력 DTO다.
    """

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_target_idx와_negative_idxs를_보관한다(self):
        """Given: target_idx=0, negative_idxs=(1,)
        When:  ShotChoice 생성
        Then:  .target_idx == 0, .negative_idxs == (1,)

        WHY: UI가 고른 대상·배제 후보 인덱스를 한 단위로 묶어 core에 전달한다.
        """
        choice = ShotChoice(target_idx=CAND_IDX_0, negative_idxs=(CAND_IDX_1,))

        assert choice.target_idx == CAND_IDX_0
        assert choice.negative_idxs == (CAND_IDX_1,)

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_target_idx가_None일_수_있다(self):
        """Given: target_idx=None, negative_idxs=()
        When:  ShotChoice 생성
        Then:  .target_idx is None

        WHY: 사용자가 어떤 샷을 선택 안 하면(미선택) target_idx=None → 그 샷은
             자동 재매칭 폴백으로 처리된다(노트북 셀 7.5 SHOT_TARGETS.get None).
        """
        choice = ShotChoice(target_idx=None, negative_idxs=())

        assert choice.target_idx is None


# ===========================================================================
# Task 4-1: build_selections_from_choices — dict(인덱스) → list[CutSelection]
# ===========================================================================
class TestBuildSelectionsFromChoices:
    """build_selections_from_choices: 샷별 후보 박스 + 선택(인덱스) → CutSelection 리스트.

    계약(RED, 노트북 셀 7.5 변환 루프 이식):
      build_selections_from_choices(candidate_boxes, choices) -> list[CutSelection]
        - candidate_boxes: list[list[Box]] — 샷별 후보 박스(box만, ShotCandidates 아님).
        - choices: dict[int, ShotChoice] — {샷 인덱스: 선택}.
        - 샷별: target_idx None 또는 범위 초과 → skip(폴백).
                box = 후보박스[target_idx], point = box_center(box).
                negative_idxs → box_center(자기(target_idx)·범위 밖 인덱스 제외).
        - CutSelection(shot_index, point, box=box, negative_points=negs) 빌드 + 검증.
      core→app 역참조 회피: ShotCandidates(app DTO)가 아니라 box 리스트만 받는다.
    """

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_빈_choices면_빈_리스트를_반환한다(self):
        """Given: candidate_boxes는 있고 choices={}
        When:  build_selections_from_choices 호출
        Then:  [] — 선택이 없으면 전 샷 자동 재매칭 폴백

        WHY: 아무 샷도 선택 안 하면 빈 리스트(자동 경로 신호)여야 한다(무회귀).
        """
        result = build_selections_from_choices(CANDIDATE_BOXES, {})

        assert result == []

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_target_idx가_None인_샷은_건너뛴다(self):
        """Given: 샷0 choice의 target_idx=None
        When:  build_selections_from_choices 호출
        Then:  [] — 미선택 샷은 CutSelection을 만들지 않음

        WHY: target_idx None은 '이 샷 미선택→폴백' 신호다(셀 7.5 ti is None → continue).
        """
        choices = {SHOT_0: ShotChoice(target_idx=None, negative_idxs=())}

        result = build_selections_from_choices(CANDIDATE_BOXES, choices)

        assert result == []

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_일부_샷만_선택하면_그_샷만_CutSelection이_생성된다(self):
        """Given: 샷1만 target_idx=0 선택(샷0 미선택)
        When:  build_selections_from_choices 호출
        Then:  CutSelection 1개, shot_index==1

        WHY: 혼합 시나리오 — 선택한 샷만 CutSelection으로, 나머지는 폴백.
        """
        choices = {SHOT_1: ShotChoice(target_idx=CAND_IDX_0, negative_idxs=())}

        result = build_selections_from_choices(CANDIDATE_BOXES, choices)

        assert len(result) == 1
        assert result[0].shot_index == SHOT_1

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_선택한_샷의_box와_point가_후보박스_중심으로_채워진다(self):
        """Given: 샷0 target_idx=0 (후보박스 SHOT0_BOX0=(10,10,30,50))
        When:  build_selections_from_choices 호출
        Then:  .box == SHOT0_BOX0, .point == box_center(SHOT0_BOX0)=(20, 30)

        WHY: box 프롬프트는 후보 전신 bbox 그대로, point는 그 중심(셀 7.5 변환).
        """
        choices = {SHOT_0: ShotChoice(target_idx=CAND_IDX_0, negative_idxs=())}

        result = build_selections_from_choices(CANDIDATE_BOXES, choices)

        assert result[0].box == SHOT0_BOX0
        assert result[0].point == box_center(SHOT0_BOX0)

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_negative_idxs는_해당_후보박스_중심으로_채워진다(self):
        """Given: 샷0 target_idx=0, negative_idxs=(1,) (배제 후보 SHOT0_BOX1)
        When:  build_selections_from_choices 호출
        Then:  .negative_points == (box_center(SHOT0_BOX1),)=( (80, 50), )

        WHY: 옆 멤버(negative)도 box 중심점(label 0)으로 변환된다(셀 7.5 negs).
        """
        choices = {
            SHOT_0: ShotChoice(target_idx=CAND_IDX_0, negative_idxs=(CAND_IDX_1,))
        }

        result = build_selections_from_choices(CANDIDATE_BOXES, choices)

        assert result[0].negative_points == (box_center(SHOT0_BOX1),)

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_negative가_target과_같은_인덱스면_제외된다(self):
        """Given: 샷0 target_idx=0, negative_idxs=(0,) (대상 자신)
        When:  build_selections_from_choices 호출
        Then:  .negative_points == () — 자기 자신은 negative에서 제외

        WHY: 대상(positive)을 동시에 negative로 찍으면 모순이다. 셀 7.5의
             `ni != ti` 자기제외 규칙을 보존해 무의미·충돌 입력을 거른다.
        """
        choices = {
            SHOT_0: ShotChoice(target_idx=CAND_IDX_0, negative_idxs=(CAND_IDX_0,))
        }

        result = build_selections_from_choices(CANDIDATE_BOXES, choices)

        assert result[0].negative_points == ()

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_범위_밖_negative_인덱스는_제외된다(self):
        """Given: 샷0 target_idx=0, negative_idxs=(1, 5) (5는 후보 수(2) 초과)
        When:  build_selections_from_choices 호출
        Then:  .negative_points == (box_center(SHOT0_BOX1),) — 5는 무시

        WHY: 셀 7.5의 `ni < len(candidates)` 범위밖 제외 규칙. 잘못된 인덱스는
             IndexError 없이 조용히 걸러내야 한다(UI 입력 방어).
        """
        choices = {
            SHOT_0: ShotChoice(
                target_idx=CAND_IDX_0,
                negative_idxs=(CAND_IDX_1, CAND_IDX_OUT_OF_RANGE),
            )
        }

        result = build_selections_from_choices(CANDIDATE_BOXES, choices)

        assert result[0].negative_points == (box_center(SHOT0_BOX1),)

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_target_idx가_후보_수를_초과하면_건너뛴다(self):
        """Given: 샷0 target_idx=5 (후보 수(2) 초과)
        When:  build_selections_from_choices 호출
        Then:  [] — 범위 밖 대상은 CutSelection을 만들지 않음(폴백)

        WHY: 셀 7.5의 `ti >= len(candidates) → continue`. 잘못된 대상 인덱스는
             IndexError 대신 폴백 처리한다(UI 입력 방어).
        """
        choices = {
            SHOT_0: ShotChoice(
                target_idx=CAND_IDX_OUT_OF_RANGE, negative_idxs=()
            )
        }

        result = build_selections_from_choices(CANDIDATE_BOXES, choices)

        assert result == []

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_생성된_CutSelection은_shot_index_순으로_정렬된다(self):
        """Given: 샷1·샷0을 모두 선택(딕셔너리 순서 무관)
        When:  build_selections_from_choices 호출
        Then:  결과의 shot_index가 [0, 1] 오름차순

        WHY: 오케스트레이션(track)이 샷 순서대로 선택을 적용하므로, 결정적
             출력을 위해 shot_index 오름차순으로 빌드한다(딕셔너리 순서 의존 제거).
        """
        choices = {
            SHOT_1: ShotChoice(target_idx=CAND_IDX_0, negative_idxs=()),
            SHOT_0: ShotChoice(target_idx=CAND_IDX_0, negative_idxs=()),
        }

        result = build_selections_from_choices(CANDIDATE_BOXES, choices)

        assert [s.shot_index for s in result] == [SHOT_0, SHOT_1]

    @pytest.mark.skipif(not _HAS_CONVERTERS, reason=_MSG_NO_CONVERTERS)
    def test_빌드_결과는_validate_selections를_통과한다(self):
        """Given: 샷0·샷1을 정상 선택, n_shots=2
        When:  build_selections_from_choices 후 validate_selections 호출
        Then:  예외 없이 통과(범위·중복 검증 만족)

        WHY: 빌드 함수는 검증된 선택만 내보내야 한다(셀 7.5 → CutSelection 빌드 후
             track 진입 전 검증). 범위·중복 위반 없는 선택임을 보장한다.
        """
        choices = {
            SHOT_0: ShotChoice(target_idx=CAND_IDX_0, negative_idxs=()),
            SHOT_1: ShotChoice(target_idx=CAND_IDX_1, negative_idxs=()),
        }

        result = build_selections_from_choices(CANDIDATE_BOXES, choices)

        assert validate_selections(result, N_SHOTS_2) is None
