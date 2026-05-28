"""ui/segment_table 위젯 스모크 테스트 (offscreen PySide6).

대상 모듈:
  - easy_capture.ui.segment_table.SegmentTableWidget (미구현 — RED 정상)
  - easy_capture.ui.video_window.VideoMainWindow (기존 — 무회귀)

테스트 범위:
  1. SegmentTableWidget 생성·행 추가·행 삭제 기본 동작
  2. "현재 프레임 → 시작/끝" 버튼 존재 및 동작 (Task 5-2)
  3. 배속 ComboBox 프리셋 항목 포함 여부
  4. VideoMainWindow 인스턴스화 무회귀 (기존 테스트 연장선)

PySide6 의존: pytest.importorskip 으로 미설치 시 skip.
offscreen 플랫폼: os.environ.setdefault("QT_QPA_PLATFORM", "offscreen").

WHY try/except + xfail:
  segment_table 미구현 시 ModuleNotFoundError.
  VideoMainWindow 기존 무회귀 케이스는 segment_table 의존이 없으므로
  별도 skip 없이 바로 실행 가능.
  SegmentTableWidget 테스트만 xfail로 RED 표시.
"""
from __future__ import annotations

import os

# WHY: PySide6 위젯 import 전 헤드리스 플랫폼 지정 — CI 및 offscreen 환경 대응
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
# 모듈 격리 — SegmentTableWidget 미구현 시 XFAIL RED
# ---------------------------------------------------------------------------
try:
    from easy_capture.ui.segment_table import SegmentTableWidget  # noqa: E402
    _HAS_WIDGET = True
except ModuleNotFoundError:
    SegmentTableWidget = None  # type: ignore[assignment,misc]
    _HAS_WIDGET = False

# PRESET_FACTORS는 순수 상수 — 미구현 시 None
try:
    from easy_capture.ui.segment_table import PRESET_FACTORS  # noqa: E402
    _HAS_PRESET = True
except (ModuleNotFoundError, ImportError):
    PRESET_FACTORS = None  # type: ignore[assignment]
    _HAS_PRESET = False

_MSG_WIDGET_NOT_IMPL = "easy_capture.ui.segment_table.SegmentTableWidget 미구현 — RED 예상"
_MSG_PRESET_NOT_IMPL = "easy_capture.ui.segment_table.PRESET_FACTORS 미구현 — RED 예상"


def _require_widget() -> None:
    """SegmentTableWidget 미구현이면 xfail로 RED 표시."""
    if not _HAS_WIDGET:
        pytest.xfail(_MSG_WIDGET_NOT_IMPL)


# ---------------------------------------------------------------------------
# 테스트 상수 — 매직넘버 금지
# ---------------------------------------------------------------------------
# 기대 배속 프리셋값 (ComboBox에 표시되어야 하는 최소 목록)
EXPECTED_COMBO_PRESETS: set[float] = {0.25, 0.5, 1.0, 2.0}
# 0.33은 부동소수 허용치로 별도 검사
EXPECTED_COMBO_THIRD: float = 0.33
PRESET_APPROX_ABS: float = 0.01

# 테스트용 프레임 인덱스
FRAME_IDX_START_TEST: int = 5   # "현재 프레임 → 시작" 버튼에 주입할 인덱스
FRAME_IDX_END_TEST: int = 20    # "현재 프레임 → 끝" 버튼에 주입할 인덱스

# 기본 행 데이터 (start, end, factor)
ROW_DATA_DEFAULT: tuple[int, int, float] = (0, 10, 0.5)


# ===========================================================================
# 1. SegmentTableWidget 기본 동작 (생성·행 관리)
# ===========================================================================
class TestSegmentTableWidgetBasic:
    """SegmentTableWidget: QTableWidget 기반 구간 테이블 생성 및 행 관리."""

    def test_위젯이_QApplication_없이_생성되지_않는다_안내(self):
        """Given: QApplication 초기화 여부
        When:  이 테스트가 실행되는 시점
        Then:  QApplication이 존재한다 (이후 위젯 테스트의 전제 조건).

        WHY: offscreen에서도 QApplication이 반드시 먼저 존재해야 한다.
             이 테스트 통과 = 이후 위젯 테스트 환경 검증.
        """
        app = _get_app()
        assert app is not None, "QApplication 초기화 실패 — offscreen 환경 점검 필요"

    def test_SegmentTableWidget_인스턴스화가_예외없이_완료된다(self):
        """Given: QApplication 존재
        When:  SegmentTableWidget() 생성
        Then:  예외 없이 인스턴스가 생성된다.

        WHY: 기본 인스턴스화 스모크 — 위젯 내부 __init__ 크래시 방지.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()

        assert widget is not None

    def test_초기_행_수가_0이다(self):
        """Given: 새로 생성된 SegmentTableWidget
        When:  행 수 확인
        Then:  rowCount() == 0 (빈 테이블).

        WHY: 처음 열었을 때 빈 테이블이어야 사용자가 직접 구간을 추가한다.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()

        assert widget.rowCount() == 0

    def test_행_추가_후_행_수가_증가한다(self):
        """Given: 빈 SegmentTableWidget
        When:  add_row() (또는 동등 메서드) 호출
        Then:  rowCount() == 1.

        WHY: "추가" 버튼 클릭 → 새 행 삽입 기본 동작 검증.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()
        widget.add_row()

        assert widget.rowCount() == 1

    def test_행_추가_두_번_후_행_수가_2이다(self):
        """Given: 빈 SegmentTableWidget
        When:  add_row() 2회 호출
        Then:  rowCount() == 2.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()
        widget.add_row()
        widget.add_row()

        assert widget.rowCount() == 2

    def test_행_삭제_후_행_수가_감소한다(self):
        """Given: 행 1개가 있는 SegmentTableWidget
        When:  행 선택 후 remove_selected_row() (또는 동등 메서드) 호출
        Then:  rowCount() == 0.

        WHY: "삭제" 버튼 클릭 → 선택 행 제거 기본 동작.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()
        widget.add_row()
        # 첫 행 선택 후 삭제
        widget.setCurrentCell(0, 0)
        widget.remove_selected_row()

        assert widget.rowCount() == 0

    def test_빈_테이블에서_삭제_호출해도_크래시없다(self):
        """Given: 빈 SegmentTableWidget (행 없음)
        When:  remove_selected_row() 호출
        Then:  예외 없이 통과 (방어 처리).

        WHY: 사용자가 빈 테이블에서 실수로 삭제 버튼을 누를 수 있다.
             크래시 없이 무시해야 한다.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()

        # 예외 발생 없이 통과해야 함
        widget.remove_selected_row()

        assert widget.rowCount() == 0


# ===========================================================================
# 2. "현재 프레임 → 시작/끝" 버튼 (Task 5-2)
# ===========================================================================
class TestSegmentTableFrameInjectButtons:
    """구간 테이블 "현재 프레임 → 시작/끝" 버튼 계약.

    WHY: 계획서 Task 5-2 [치명적·UX] — 사용자가 미리보기에서 프레임을
         직접 보며 구간을 지정할 수 있어야 한다.
         프레임 번호 암기 불필요 → UX 핵심 기능.
    """

    def test_set_frame_as_start_버튼이_존재한다(self):
        """Given: SegmentTableWidget
        When:  위젯 생성 후 속성/버튼 확인
        Then:  "현재 프레임 → 시작" 버튼 또는 set_frame_as_start 메서드가 존재한다.

        WHY: Task 5-2 수용 기준 — 미리보기 프레임을 보고 버튼으로 구간 지정.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()

        # 메서드 존재 여부로 기능 계약 검증
        has_method = hasattr(widget, "set_frame_as_start")
        assert has_method, (
            "set_frame_as_start 메서드가 없음 — "
            "현재 프레임을 시작으로 설정하는 기능이 필요합니다."
        )

    def test_set_frame_as_end_버튼이_존재한다(self):
        """Given: SegmentTableWidget
        When:  위젯 생성 후 속성 확인
        Then:  "현재 프레임 → 끝" 기능(메서드)이 존재한다.

        WHY: Task 5-2 — 끝 프레임 지정도 같은 방식으로 제공해야 한다.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()

        has_method = hasattr(widget, "set_frame_as_end")
        assert has_method, (
            "set_frame_as_end 메서드가 없음 — "
            "현재 프레임을 끝으로 설정하는 기능이 필요합니다."
        )

    def test_set_frame_as_start_가_선택_행의_시작값을_갱신한다(self):
        """Given: 행 1개 있는 SegmentTableWidget, 0행 선택
        When:  set_frame_as_start(FRAME_IDX_START_TEST) 호출
        Then:  선택 행의 start 값 == FRAME_IDX_START_TEST.

        WHY: 버튼 동작 핵심 계약 — 인덱스가 실제 셀에 주입되어야 한다.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()
        widget.add_row()
        widget.setCurrentCell(0, 0)

        widget.set_frame_as_start(FRAME_IDX_START_TEST)

        # get_row_data(row) -> (start, end, factor) 형식 기대
        row_data = widget.get_row_data(0)
        start_val = row_data[0]
        assert start_val == FRAME_IDX_START_TEST, (
            f"start 주입 실패: 예상 {FRAME_IDX_START_TEST}, 실제 {start_val}"
        )

    def test_set_frame_as_end_가_선택_행의_끝값을_갱신한다(self):
        """Given: 행 1개 있는 SegmentTableWidget, 0행 선택
        When:  set_frame_as_end(FRAME_IDX_END_TEST) 호출
        Then:  선택 행의 end 값 == FRAME_IDX_END_TEST.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()
        widget.add_row()
        widget.setCurrentCell(0, 0)

        widget.set_frame_as_end(FRAME_IDX_END_TEST)

        row_data = widget.get_row_data(0)
        end_val = row_data[1]
        assert end_val == FRAME_IDX_END_TEST, (
            f"end 주입 실패: 예상 {FRAME_IDX_END_TEST}, 실제 {end_val}"
        )

    def test_행_없을_때_set_frame_as_start_호출해도_크래시없다(self):
        """Given: 빈 SegmentTableWidget
        When:  set_frame_as_start(10) 호출
        Then:  예외 없이 통과 (선택 행 없음 방어).

        WHY: 빈 테이블에서 사용자가 실수로 버튼 누를 수 있다.
             크래시 없이 무시해야 한다.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()

        # 예외 없이 통과해야 함
        widget.set_frame_as_start(10)


# ===========================================================================
# 3. 배속 ComboBox 프리셋 항목 검증
# ===========================================================================
class TestSegmentTableSpeedCombo:
    """구간 테이블 내 배속 ComboBox 프리셋 계약.

    WHY: Task 5-1 명시 "배속 ComboBox 프리셋 0.25/0.33/0.5/1/2x".
         ComboBox에 기대 항목이 없으면 사용자가 UI에서 해당 배속을 선택 불가.
    """

    def test_배속_ComboBox에_0_25가_포함된다(self):
        """Given: 행이 있는 SegmentTableWidget의 배속 ComboBox
        When:  ComboBox 항목 확인
        Then:  0.25 항목이 있다.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()
        widget.add_row()
        combo = widget.get_speed_combo(0)

        assert combo is not None, "배속 ComboBox가 없음"
        items = [combo.itemData(i) for i in range(combo.count())]
        assert any(
            abs(item - 0.25) <= PRESET_APPROX_ABS
            for item in items
            if item is not None
        ), f"0.25x 프리셋 없음. 현재 ComboBox 항목: {items}"

    def test_배속_ComboBox에_0_5가_포함된다(self):
        """Given: 행이 있는 SegmentTableWidget의 배속 ComboBox
        When:  ComboBox 항목 확인
        Then:  0.5 항목이 있다.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()
        widget.add_row()
        combo = widget.get_speed_combo(0)

        assert combo is not None
        items = [combo.itemData(i) for i in range(combo.count())]
        assert any(
            abs(item - 0.5) <= PRESET_APPROX_ABS
            for item in items
            if item is not None
        ), f"0.5x 프리셋 없음. 현재 항목: {items}"

    def test_배속_ComboBox에_1_0이_포함된다(self):
        """Given: 행이 있는 SegmentTableWidget의 배속 ComboBox
        When:  ComboBox 항목 확인
        Then:  1.0(등속) 항목이 있다.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()
        widget.add_row()
        combo = widget.get_speed_combo(0)

        assert combo is not None
        items = [combo.itemData(i) for i in range(combo.count())]
        assert any(
            abs(item - 1.0) <= PRESET_APPROX_ABS
            for item in items
            if item is not None
        ), f"1.0x 프리셋 없음. 현재 항목: {items}"

    def test_배속_ComboBox에_2_0이_포함된다(self):
        """Given: 행이 있는 SegmentTableWidget의 배속 ComboBox
        When:  ComboBox 항목 확인
        Then:  2.0x 항목이 있다.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()
        widget.add_row()
        combo = widget.get_speed_combo(0)

        assert combo is not None
        items = [combo.itemData(i) for i in range(combo.count())]
        assert any(
            abs(item - 2.0) <= PRESET_APPROX_ABS
            for item in items
            if item is not None
        ), f"2.0x 프리셋 없음. 현재 항목: {items}"

    def test_배속_ComboBox에_0_33이_포함된다(self):
        """Given: 행이 있는 SegmentTableWidget의 배속 ComboBox
        When:  ComboBox 항목 확인
        Then:  약 0.33 항목이 있다 (부동소수 허용치 0.01).

        WHY: 계획서 Task 5-1 명시 "0.33" 포함.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()
        widget.add_row()
        combo = widget.get_speed_combo(0)

        assert combo is not None
        items = [combo.itemData(i) for i in range(combo.count())]
        assert any(
            abs(item - EXPECTED_COMBO_THIRD) <= PRESET_APPROX_ABS
            for item in items
            if item is not None
        ), f"약 0.33 프리셋 없음. 현재 항목: {items}"

    def test_배속_ComboBox_항목수가_최소_5개이다(self):
        """Given: 행이 있는 SegmentTableWidget의 배속 ComboBox
        When:  항목 수 확인
        Then:  count >= 5 (0.25/0.33/0.5/1.0/2.0 최소 5개).

        WHY: 5개 프리셋이 모두 있어야 계획서 Task 5-1 충족.
        """
        _require_widget()
        _get_app()

        widget = SegmentTableWidget()
        widget.add_row()
        combo = widget.get_speed_combo(0)

        assert combo is not None
        min_preset_count = 5  # 0.25, 0.33, 0.5, 1.0, 2.0
        assert combo.count() >= min_preset_count, (
            f"ComboBox 항목 부족: {combo.count()}개 < {min_preset_count}개"
        )


# ===========================================================================
# 4. VideoMainWindow 무회귀 (기존 위젯 연장선)
# ===========================================================================
class TestVideoMainWindowNoRegression:
    """VideoMainWindow 인스턴스화 무회귀.

    WHY: Story 5 구현 후 VideoMainWindow가 segment_table을 통합할 때
         기존 동작이 깨지지 않았는지 보장한다.
         현재(미구현) 시점에는 VideoMainWindow 자체 무회귀만 검증.
    """

    def test_VideoMainWindow가_더미_팩토리로_인스턴스화된다(self):
        """Given: 더미 usecase_factory
        When:  VideoMainWindow(usecase_factory) 생성
        Then:  예외 없이 인스턴스 생성.

        WHY: video_window.py 603줄 — 기존 테스트 없이 단독 스모크.
             segment_table 통합 전후로 이 테스트가 항상 통과해야 한다.
        """
        from easy_capture.ui.video_window import VideoMainWindow

        _get_app()

        def _dummy_factory(_path: str):
            """더미 usecase 팩토리 — 실제 GPU·파일 비의존."""
            return None

        window = VideoMainWindow(_dummy_factory)

        assert window is not None

    def test_VideoMainWindow_타이틀이_비어있지_않다(self):
        """Given: VideoMainWindow 인스턴스
        When:  windowTitle() 확인
        Then:  비어있지 않은 문자열.

        WHY: 타이틀바가 비면 사용자가 어떤 창인지 구별 불가.
        """
        from easy_capture.ui.video_window import VideoMainWindow

        _get_app()

        window = VideoMainWindow(lambda _: None)

        assert window.windowTitle(), "windowTitle이 비어있음"

    def test_VideoMainWindow에_GAP_POLICY_ITEMS가_모든_정책을_포함한다(self):
        """Given: video_window._GAP_POLICY_ITEMS
        When:  정책 집합 확인
        Then:  GapPolicy.BACKGROUND, CUT, FREEZE 모두 포함.

        WHY: 기존 test_video_window_gap.py 패턴 계승 — segment_table 통합 후 무회귀.
        """
        from easy_capture.core.tracking.gap_policy import GapPolicy
        from easy_capture.ui.video_window import _GAP_POLICY_ITEMS

        policies = {policy for _, policy in _GAP_POLICY_ITEMS}

        assert policies == set(GapPolicy), (
            f"_GAP_POLICY_ITEMS 정책 불일치: 실제={policies}"
        )
