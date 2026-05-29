"""CutSelectionPanel 위젯 테스트 (offscreen PySide6, TDD — Task 4-3).

대상 모듈: easy_capture.ui.cut_selection_panel.CutSelectionPanel

검증 범위:
  1. set_shots 후 초기 상태 (current_shot_index==0, nav 경계 disable)
  2. set_target — positive 1개 라디오 선택, 교체
  3. toggle_negative — 다수 체크, 토글(추가/제거), 상호배타(positive ↔ negative)
  4. to_choices — 다중 샷 결과 딕셔너리, 미선택 샷 키 없음
  5. shot_changed / selection_changed Signal 방출 검증
  6. 연쇄 정합: build_selections_from_choices가 ValueError 없이 CutSelection 리스트 생성

PySide6 의존: pytest.importorskip으로 미설치 시 skip.
offscreen 플랫폼: os.environ.setdefault("QT_QPA_PLATFORM", "offscreen").
모달 hang 방지: QMessageBox/QFileDialog는 이 테스트에서 미사용이지만,
  위젯 내부에서 사용 시 monkeypatch 패턴을 여기에 기록한다.

WHY: 기존 test_video_window_segment.py의 QApplication 싱글턴 패턴을 동형으로 적용한다.
     QApplication 인스턴스를 모듈 수준에서 한 번만 생성해 중복 생성(충돌)을 방지한다.
"""
from __future__ import annotations

import os

# WHY: PySide6 위젯 import 전 헤드리스 플랫폼 지정 — CI 및 서버 환경 대응
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="PySide6 미설치 — UI 모듈 import 불가")

from PySide6.QtWidgets import QApplication  # noqa: E402

# ---------------------------------------------------------------------------
# QApplication 싱글턴 (테스트 세션 전체에서 1회 생성)
# ---------------------------------------------------------------------------
_app: QApplication | None = None


def _get_app() -> QApplication:
    """QApplication 싱글턴을 반환한다.

    WHY: PySide6 위젯 생성 전 QApplication 인스턴스가 반드시 존재해야 한다.
         pytest fixture 대신 모듈 수준 싱글턴으로 중복 생성을 방지한다.
    """
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


# ---------------------------------------------------------------------------
# 모듈 격리 — CutSelectionPanel 미구현 시 XFAIL RED
# ---------------------------------------------------------------------------
try:
    from easy_capture.ui.cut_selection_panel import CutSelectionPanel  # noqa: E402
    _HAS_PANEL = True
except (ModuleNotFoundError, ImportError):
    CutSelectionPanel = None  # type: ignore[assignment,misc]
    _HAS_PANEL = False

try:
    from easy_capture.app.video_capture import ShotCandidates  # noqa: E402
    _HAS_SHOT_CANDIDATES = True
except (ModuleNotFoundError, ImportError):
    ShotCandidates = None  # type: ignore[assignment,misc]
    _HAS_SHOT_CANDIDATES = False

try:
    from easy_capture.core.tracking.cut_selection import (  # noqa: E402
        ShotChoice,
        build_selections_from_choices,
    )
    _HAS_CORE = True
except (ModuleNotFoundError, ImportError):
    ShotChoice = None  # type: ignore[assignment,misc]
    build_selections_from_choices = None  # type: ignore[assignment]
    _HAS_CORE = False

try:
    from easy_capture.core.segmentation.detection_backend import Detection  # noqa: E402
    _HAS_DETECTION = True
except (ModuleNotFoundError, ImportError):
    Detection = None  # type: ignore[assignment,misc]
    _HAS_DETECTION = False

_MSG_NO_PANEL = "easy_capture.ui.cut_selection_panel.CutSelectionPanel 미구현 — RED 예상"
_MSG_NO_DEPS = "필수 의존(ShotCandidates/ShotChoice/Detection) 미구현 — RED 예상"

# ---------------------------------------------------------------------------
# 테스트 상수 (매직넘버 금지)
# ---------------------------------------------------------------------------
# 후보 인덱스
CAND_IDX_0 = 0
CAND_IDX_1 = 1
CAND_IDX_2 = 2

# 합성 박스 좌표 (x1, y1, x2, y2) — Detection 더블용
BOX_0 = (10.0, 20.0, 80.0, 150.0)
BOX_1 = (90.0, 20.0, 160.0, 150.0)
BOX_2 = (170.0, 20.0, 240.0, 150.0)

# 신뢰도 — Detection 더블용 (실제 추적에서는 무시됨)
SCORE_DEFAULT = 0.9

# 샷 수 (nav 경계 테스트용)
N_SHOTS_1 = 1
N_SHOTS_2 = 2
N_SHOTS_3 = 3


# ---------------------------------------------------------------------------
# 더블 팩토리 — ShotCandidates + Detection (fakes.py의 Detection 패턴 활용)
# ---------------------------------------------------------------------------

def _make_detection(box: tuple) -> "Detection":
    """합성 Detection 더블을 반환한다(box 속성 포함).

    WHY: 실제 GPU/모델 없이 Detection(box=...) 인스턴스를 만들어
         ShotCandidates의 candidates 리스트에 주입한다.
    """
    return Detection(box=box, score=SCORE_DEFAULT)


def _make_shot(
    shot_index: int,
    boxes: list[tuple],
) -> "ShotCandidates":
    """합성 ShotCandidates 더블을 반환한다.

    WHY: first_frame_index는 UI 썸네일용이므로 테스트에서는 shot_index와 동일로 고정한다.
    """
    candidates = [_make_detection(b) for b in boxes]
    return ShotCandidates(
        shot_index=shot_index,
        first_frame_index=shot_index,
        candidates=candidates,
    )


def _make_two_shots() -> "list[ShotCandidates]":
    """후보 2개씩 2샷을 가진 더블 리스트를 반환한다."""
    return [
        _make_shot(0, [BOX_0, BOX_1]),
        _make_shot(1, [BOX_1, BOX_2]),
    ]


def _make_three_shots() -> "list[ShotCandidates]":
    """후보 2개씩 3샷을 가진 더블 리스트를 반환한다."""
    return [
        _make_shot(0, [BOX_0, BOX_1]),
        _make_shot(1, [BOX_1, BOX_2]),
        _make_shot(2, [BOX_0, BOX_2]),
    ]


# ---------------------------------------------------------------------------
# 전제조건 skip 헬퍼
# ---------------------------------------------------------------------------

def _require_panel() -> None:
    """CutSelectionPanel 미구현이면 xfail로 RED 표시."""
    if not _HAS_PANEL:
        pytest.xfail(_MSG_NO_PANEL)


def _require_all() -> None:
    """패널 + 의존 전체 미구현이면 xfail."""
    if not (_HAS_PANEL and _HAS_SHOT_CANDIDATES and _HAS_CORE and _HAS_DETECTION):
        pytest.xfail(_MSG_NO_DEPS)


# ---------------------------------------------------------------------------
# 1. 초기 상태 — set_shots 후 첫 번째 샷, nav 경계 disable
# ---------------------------------------------------------------------------

class TestInitialState:
    """set_shots 호출 직후 초기 상태를 검증한다."""

    def test_set_shots_후_current_shot_index는_0이다(self):
        """Given: 2샷 ShotCandidates 리스트
        When:  set_shots(shots) 호출
        Then:  current_shot_index() == 0

        WHY: 첫 번째 샷(인덱스 0)부터 시작해야 사용자가 순서대로 선택할 수 있다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        shots = _make_two_shots()

        panel.set_shots(shots)

        assert panel.current_shot_index() == 0

    def test_set_shots_후_이전_버튼은_비활성화된다(self):
        """Given: 2샷 주입
        When:  set_shots 호출 후 첫 번째 샷
        Then:  이전 버튼(prev_btn)이 비활성화(disabled)

        WHY: 첫 번째 샷에서는 더 이전 샷이 없으므로 이전 버튼을 비활성화해
             경계 조건을 명확히 한다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()

        panel.set_shots(_make_two_shots())

        assert not panel._prev_btn.isEnabled()

    def test_set_shots_후_마지막_샷이_아니면_다음_버튼은_활성화된다(self):
        """Given: 2샷 주입, 현재 샷 0
        When:  set_shots 호출
        Then:  다음 버튼(next_btn)이 활성화

        WHY: 첫 번째 샷에서 다음 샷이 있으면 다음 버튼이 활성화되어야 한다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()

        panel.set_shots(_make_two_shots())

        assert panel._next_btn.isEnabled()

    def test_단일_샷_주입_시_양쪽_nav_버튼_모두_비활성화(self):
        """Given: 1샷 ShotCandidates 리스트
        When:  set_shots 호출
        Then:  이전·다음 버튼 모두 비활성화

        WHY: 샷이 1개면 이동할 곳이 없으므로 양쪽 경계가 동시에 성립한다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()

        panel.set_shots([_make_shot(0, [BOX_0, BOX_1])])

        assert not panel._prev_btn.isEnabled()
        assert not panel._next_btn.isEnabled()

    def test_set_shots_후_choices는_초기화된다(self):
        """Given: 이전에 선택했던 패널에 새 샷 주입
        When:  set_shots 호출
        Then:  to_choices() == {} (미선택 초기 상태)

        WHY: 새 영상/구간을 로드할 때 이전 선택이 잔류하면 잘못된 추적이 된다.
             set_shots는 항상 선택 상태를 초기화해야 한다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        # 먼저 선택 상태를 만든다
        panel.set_shots(_make_two_shots())
        panel.set_target(CAND_IDX_0)

        # 새 샷 주입 → 상태 초기화
        panel.set_shots(_make_two_shots())

        assert panel.to_choices() == {}


# ---------------------------------------------------------------------------
# 2. set_target — positive 1개 라디오 선택
# ---------------------------------------------------------------------------

class TestSetTarget:
    """set_target(candidate_idx)으로 positive를 1개 지정한다."""

    def test_set_target_후_current_choice의_target_idx가_설정된다(self):
        """Given: 샷 0, 후보 2개
        When:  set_target(1) 호출
        Then:  current_choice().target_idx == 1

        WHY: UI가 지정한 positive 인덱스가 ShotChoice에 정확히 반영되어야
             build_selections_from_choices가 올바른 박스를 선택할 수 있다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())

        panel.set_target(CAND_IDX_1)

        assert panel.current_choice().target_idx == CAND_IDX_1

    def test_다른_target_지정_시_기존_positive가_교체된다(self):
        """Given: target_idx=0 선택 상태
        When:  set_target(1) 호출
        Then:  current_choice().target_idx == 1 (0 → 1 교체)

        WHY: positive는 샷당 1개(라디오 버튼). 새 positive 지정 시 기존을 교체해야
             중복 선택 없이 단일 추적 대상을 보장한다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())
        panel.set_target(CAND_IDX_0)

        panel.set_target(CAND_IDX_1)

        assert panel.current_choice().target_idx == CAND_IDX_1

    def test_set_target_None으로_positive를_해제한다(self):
        """Given: target_idx=0 선택 상태
        When:  set_target(None) 호출
        Then:  current_choice().target_idx == None

        WHY: 사용자가 positive 선택을 취소할 수 있어야 자동 재매칭 폴백으로
             돌아갈 수 있다(ShotChoice.target_idx=None=미선택 폴백).
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())
        panel.set_target(CAND_IDX_0)

        panel.set_target(None)

        assert panel.current_choice().target_idx is None

    def test_positive_지정_시_해당_인덱스는_negative에서_제외된다(self):
        """Given: 후보 0을 negative로 추가한 상태
        When:  set_target(0) — 동일 인덱스를 positive로 지정
        Then:  current_choice().negative_idxs에 0이 없다(상호배타)

        WHY: positive로 지정된 인덱스를 동시에 negative로 두는 것은 모순이다.
             positive 지정 시 해당 인덱스를 negative에서 자동 제거한다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())
        panel.toggle_negative(CAND_IDX_0)  # 먼저 negative로 추가

        panel.set_target(CAND_IDX_0)  # 같은 인덱스를 positive로 지정

        assert CAND_IDX_0 not in panel.current_choice().negative_idxs


# ---------------------------------------------------------------------------
# 3. toggle_negative — 다수 체크박스, 상호배타
# ---------------------------------------------------------------------------

class TestToggleNegative:
    """toggle_negative(candidate_idx)으로 negative를 토글한다."""

    def test_toggle_negative_첫_호출에_negative에_추가된다(self):
        """Given: 초기 상태(negative 없음)
        When:  toggle_negative(1) 호출
        Then:  current_choice().negative_idxs에 1이 포함된다

        WHY: 체크박스 첫 클릭은 '배제 대상 추가' 동작이다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())

        panel.toggle_negative(CAND_IDX_1)

        assert CAND_IDX_1 in panel.current_choice().negative_idxs

    def test_toggle_negative_두_번_호출에_negative에서_제거된다(self):
        """Given: negative_idxs에 1이 있는 상태
        When:  toggle_negative(1) 다시 호출
        Then:  current_choice().negative_idxs에 1이 없다

        WHY: 체크박스 두 번째 클릭은 '배제 해제' 동작이다(토글 특성).
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())
        panel.toggle_negative(CAND_IDX_1)  # 추가

        panel.toggle_negative(CAND_IDX_1)  # 제거

        assert CAND_IDX_1 not in panel.current_choice().negative_idxs

    def test_positive_인덱스를_negative로_토글하면_negative에_추가되고_positive는_해제된다(self):
        """Given: target_idx=0 선택 상태
        When:  toggle_negative(0) — positive 인덱스를 negative 토글
        Then:  0이 negative_idxs에 추가되고 target_idx는 None

        WHY: 설계 원칙 — positive == 기존 negative 클릭 시: negative에서 제거 후 positive.
             반대로, positive를 negative로 토글하면 positive를 해제하고 negative로 이동한다.
             상호배타 보장.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())
        panel.set_target(CAND_IDX_0)

        panel.toggle_negative(CAND_IDX_0)

        choice = panel.current_choice()
        assert CAND_IDX_0 in choice.negative_idxs
        assert choice.target_idx != CAND_IDX_0

    def test_negative_인덱스를_positive로_지정하면_negative에서_제거된다(self):
        """Given: negative_idxs에 0이 있는 상태
        When:  set_target(0) — negative 인덱스를 positive로 지정
        Then:  0이 negative_idxs에서 제거되고 target_idx==0

        WHY: 사용자가 생각을 바꿔 배제 대상을 추적 대상으로 지정하는 시나리오.
             positive 지정 시 해당 인덱스가 negative에서 자동 제거되어야 한다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())
        panel.toggle_negative(CAND_IDX_0)  # 먼저 negative로 추가

        panel.set_target(CAND_IDX_0)  # positive로 전환

        choice = panel.current_choice()
        assert choice.target_idx == CAND_IDX_0
        assert CAND_IDX_0 not in choice.negative_idxs

    def test_negative는_여러_인덱스를_동시에_가질_수_있다(self):
        """Given: 후보 3개 샷
        When:  toggle_negative(0), toggle_negative(2) 호출
        Then:  negative_idxs에 0, 2 모두 포함

        WHY: negative는 다수(체크박스)로 여러 옆 멤버를 동시에 배제할 수 있어야 한다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots([_make_shot(0, [BOX_0, BOX_1, BOX_2])])

        panel.toggle_negative(CAND_IDX_0)
        panel.toggle_negative(CAND_IDX_2)

        negatives = panel.current_choice().negative_idxs
        assert CAND_IDX_0 in negatives
        assert CAND_IDX_2 in negatives


# ---------------------------------------------------------------------------
# 4. to_choices — 전체 샷 선택 딕셔너리
# ---------------------------------------------------------------------------

class TestToChoices:
    """to_choices()가 전체 샷의 ShotChoice 딕셔너리를 반환한다."""

    def test_미선택_상태에서_to_choices는_빈_딕셔너리다(self):
        """Given: set_shots 후 아무것도 선택하지 않은 상태
        When:  to_choices() 호출
        Then:  {} 반환 (미선택 샷은 키 없음)

        WHY: 미선택 샷은 build_selections_from_choices에서 KeyError 없이
             자동 재매칭 폴백으로 처리된다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())

        assert panel.to_choices() == {}

    def test_현재_샷만_선택하면_to_choices에_그_샷만_키로_있다(self):
        """Given: 샷 0에서 target=1 선택
        When:  to_choices() 호출
        Then:  {0: ShotChoice(target_idx=1, negative_idxs=())}

        WHY: 선택한 샷만 딕셔너리에 포함돼야 build_selections_from_choices가
             미선택 샷을 폴백으로 처리할 수 있다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())
        panel.set_target(CAND_IDX_1)

        choices = panel.to_choices()

        assert 0 in choices
        assert choices[0].target_idx == CAND_IDX_1
        assert 1 not in choices

    def test_두_샷_선택_후_to_choices에_두_샷_모두_포함된다(self):
        """Given: 샷 0 → target=0, 샷 1로 이동 → target=1 선택
        When:  to_choices() 호출
        Then:  {0: ShotChoice(target_idx=0), 1: ShotChoice(target_idx=1)}

        WHY: 여러 샷을 순회하며 선택하면 모든 선택이 보존되어야 한다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())

        # 샷 0 선택
        panel.set_target(CAND_IDX_0)
        # 샷 1로 이동
        panel._next_btn.click()
        # 샷 1 선택
        panel.set_target(CAND_IDX_1)

        choices = panel.to_choices()

        assert 0 in choices
        assert 1 in choices
        assert choices[0].target_idx == CAND_IDX_0
        assert choices[1].target_idx == CAND_IDX_1

    def test_negative_포함된_선택이_to_choices에_반영된다(self):
        """Given: 샷 0에서 target=0, negative={1}
        When:  to_choices() 호출
        Then:  choices[0].negative_idxs == (1,)

        WHY: negative 인덱스가 ShotChoice에 정확히 반영되어야 core의
             _build_negative_points가 옆 멤버 좌표를 변환할 수 있다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())
        panel.set_target(CAND_IDX_0)
        panel.toggle_negative(CAND_IDX_1)

        choices = panel.to_choices()

        assert CAND_IDX_1 in choices[0].negative_idxs


# ---------------------------------------------------------------------------
# 5. Signal 방출 검증
# ---------------------------------------------------------------------------

class TestSignals:
    """shot_changed / selection_changed Signal이 올바르게 방출된다."""

    def test_next_버튼_클릭_시_shot_changed_Signal이_방출된다(self):
        """Given: 2샷 주입, 샷 0 상태
        When:  다음 버튼 클릭
        Then:  shot_changed Signal이 1 인자로 방출된다

        WHY: 외부 위젯(video_window)이 샷 변경을 감지해 썸네일을 갱신하려면
             Signal이 새 인덱스를 인자로 방출해야 한다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())
        received: list[int] = []
        panel.shot_changed.connect(received.append)

        panel._next_btn.click()

        assert received == [1]

    def test_prev_버튼_클릭_시_shot_changed_Signal이_방출된다(self):
        """Given: 2샷 주입, 샷 1 상태(next로 이동 후)
        When:  이전 버튼 클릭
        Then:  shot_changed Signal이 0 인자로 방출된다
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())
        panel._next_btn.click()  # 샷 1로 이동
        received: list[int] = []
        panel.shot_changed.connect(received.append)

        panel._prev_btn.click()

        assert received == [0]

    def test_set_target_호출_시_selection_changed_Signal이_방출된다(self):
        """Given: 2샷 주입
        When:  set_target(0) 호출
        Then:  selection_changed Signal이 방출된다

        WHY: 외부에서 선택 변경을 감지해 "추적 준비 완료" 버튼을 활성화하거나
             미리보기를 갱신하는 데 필요하다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())
        received: list[bool] = []
        panel.selection_changed.connect(lambda: received.append(True))

        panel.set_target(CAND_IDX_0)

        assert len(received) == 1

    def test_toggle_negative_호출_시_selection_changed_Signal이_방출된다(self):
        """Given: 2샷 주입
        When:  toggle_negative(1) 호출
        Then:  selection_changed Signal이 방출된다
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        panel.set_shots(_make_two_shots())
        received: list[bool] = []
        panel.selection_changed.connect(lambda: received.append(True))

        panel.toggle_negative(CAND_IDX_1)

        assert len(received) == 1


# ---------------------------------------------------------------------------
# 6. 연쇄 정합 — build_selections_from_choices가 ValueError 없이 동작
# ---------------------------------------------------------------------------

class TestCoreIntegration:
    """panel.to_choices() 결과를 build_selections_from_choices에 넣어 ValueError 없이 동작."""

    def test_두_샷_선택_후_build_selections_from_choices가_성공한다(self):
        """Given: 샷 0 → target=0, 샷 1 → target=1
        When:  build_selections_from_choices(candidate_boxes, panel.to_choices()) 호출
        Then:  ValueError 없이 CutSelection 리스트 반환

        WHY: 패널의 to_choices() 출력이 core의 build_selections_from_choices 입력으로
             그대로 사용 가능해야 UI→core 경계 연쇄 정합이 보장된다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        shots = _make_two_shots()
        panel.set_shots(shots)

        # 두 샷 모두 선택
        panel.set_target(CAND_IDX_0)
        panel._next_btn.click()
        panel.set_target(CAND_IDX_1)

        # candidate_boxes: 샷별 박스 리스트 (Detection.box 추출 — video_window 책임 모사)
        candidate_boxes = [
            [d.box for d in shot.candidates]
            for shot in shots
        ]

        result = build_selections_from_choices(candidate_boxes, panel.to_choices())

        assert isinstance(result, list)
        assert len(result) == 2  # 두 샷 모두 선택했으므로 2개

    def test_미선택_샷이_있어도_build_selections가_ValueError_없이_동작한다(self):
        """Given: 샷 0만 선택, 샷 1은 미선택
        When:  build_selections_from_choices 호출
        Then:  ValueError 없이 CutSelection 리스트 1개 반환

        WHY: 미선택 샷은 폴백(자동 재매칭)으로 처리되어야 한다.
             to_choices()가 미선택 샷을 키 없음으로 반환하면 core가 건너뛴다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        shots = _make_two_shots()
        panel.set_shots(shots)

        # 샷 0만 선택
        panel.set_target(CAND_IDX_0)

        candidate_boxes = [
            [d.box for d in shot.candidates]
            for shot in shots
        ]

        result = build_selections_from_choices(candidate_boxes, panel.to_choices())

        assert len(result) == 1  # 샷 0만 선택 → 1개

    def test_negative_포함_선택도_build_selections가_ValueError_없이_동작한다(self):
        """Given: 샷 0 → target=0, negative={1}
        When:  build_selections_from_choices 호출
        Then:  ValueError 없이 CutSelection 반환, negative_points가 채워진다

        WHY: negative 인덱스가 core의 _build_negative_points로 정확히 전달되어
             SAM2 label 0 입력이 구성되어야 한다.
        """
        _require_all()
        _get_app()
        panel = CutSelectionPanel()
        shots = _make_two_shots()
        panel.set_shots(shots)
        panel.set_target(CAND_IDX_0)
        panel.toggle_negative(CAND_IDX_1)

        candidate_boxes = [
            [d.box for d in shot.candidates]
            for shot in shots
        ]

        result = build_selections_from_choices(candidate_boxes, panel.to_choices())

        assert len(result) == 1
        assert len(result[0].negative_points) == 1  # negative 1개가 좌표로 변환
