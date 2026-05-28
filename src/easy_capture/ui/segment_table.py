"""슬로우모션 구간 테이블 위젯 및 순수 로직.

순수 함수(PRESET_FACTORS·rows_to_segments·dynamic_fast_cap)는 PySide6 비의존.
SegmentTableWidget만 PySide6 사용.

WHY 분리:
  순수 함수를 위젯 모듈과 분리하면 UI 없이도 단위 테스트 가능하고,
  timeremap.py의 도메인 경계를 지킨다(ADR 0013).

Story 5 Task 5-1 ~ Task 5-3 구현체.
"""
from __future__ import annotations

from easy_capture.core.timing.timeremap import (
    SpeedSegment,
    normalize_segments,
)

# ---------------------------------------------------------------------------
# 상수 — 매직넘버 금지
# ---------------------------------------------------------------------------
# 배속 프리셋: 0.25↔0.5 사이에 0.33(3배 슬로우) 포함.
# WHY: 4.0x는 GIF에서 8ms → 20ms 클램프로 실질 무의미하므로 기본 제외(Task 5-3).
PRESET_FACTORS: tuple[float, ...] = (0.25, 0.33, 0.5, 1.0, 2.0)

# GIF duration 하한 불변식에서 유도된 상한 분자.
# WHY: duration_ms = (1000/fps)/factor ≥ 20ms → factor ≤ 1000/(fps×20) = 50/fps
#      50 = 1000 / GIF_DURATION_CLAMP_MS(20) — 매직넘버 금지를 위해 상수화.
_GIF_CAP_NUMERATOR: float = 50.0  # = 1000ms / 20ms

# SpeedSegment factor 절대 상한 (timeremap FACTOR_MAX와 정합)
_ABSOLUTE_CAP: float = 4.0


# ---------------------------------------------------------------------------
# 순수 함수 — PySide6 비의존
# ---------------------------------------------------------------------------

def rows_to_segments(
    rows: list[tuple[int, int, float]],
) -> tuple[SpeedSegment, ...]:
    """테이블 행 데이터 → SpeedSegment 튜플(순수).

    Args:
        rows: [(start, end, factor), ...] 형식의 행 데이터 리스트.

    Returns:
        normalize_segments를 통해 정렬·검증된 SpeedSegment 튜플.
        빈 rows → 빈 튜플 ().

    Raises:
        ValueError: 겹침·역전·factor 범위 초과 시 (normalize_segments 전파).

    WHY normalize_segments 위임:
        겹침·역전 검증 로직 중복 방지(DRY).
        UI는 ValueError를 잡아 한국어 QMessageBox로 표시한다.
    """
    if not rows:
        return ()
    segments = [SpeedSegment(start=r[0], end=r[1], factor=r[2]) for r in rows]
    validated = normalize_segments(segments)
    return tuple(validated)


def dynamic_fast_cap(base_fps: float) -> float:
    """GIF 패스트 배속 상한을 동적으로 계산한다(순수).

    공식: min(ABSOLUTE_CAP, 50 / base_fps)

    WHY 50/fps 공식:
        GIF per-frame duration 불변식: duration_ms = (1000/fps) / factor ≥ 20ms
        → factor ≤ 1000 / (fps × 20) = 50 / fps
        50 = 1000ms ÷ GIF_DURATION_CLAMP_MS(20ms) — 매직넘버 금지로 상수화.
        절대 상한 4.0으로 클램프해 SpeedSegment FACTOR_MAX와 정합.

    검증 예시:
        fps=12  → min(4.0, 50/12) = min(4.0, 4.17) = 4.0  → duration=20.8ms ≥ 20ms ✓
        fps=50  → min(4.0, 50/50) = min(4.0, 1.0)  = 1.0  → duration=20.0ms ≥ 20ms ✓
        fps=120 → min(4.0, 50/120)= min(4.0, 0.417)= 0.417 → duration=20.0ms ≥ 20ms ✓

    Args:
        base_fps: 원본 영상 fps (> 0).

    Returns:
        GIF 안전 패스트 상한 (float, 0 < cap ≤ 4.0).
    """
    return min(_ABSOLUTE_CAP, _GIF_CAP_NUMERATOR / base_fps)


# ---------------------------------------------------------------------------
# SegmentTableWidget — PySide6 위젯 (import 조건부)
# ---------------------------------------------------------------------------
from PySide6.QtWidgets import (  # noqa: E402
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# 테이블 열 인덱스 상수 — 매직넘버 금지
_COL_START: int = 0
_COL_END: int = 1
_COL_SPEED: int = 2
_COL_COUNT: int = 3

# SpinBox 프레임 인덱스 범위
_SPINBOX_MIN: int = 0
_SPINBOX_MAX: int = 999_999

# ComboBox 프리셋 표시 형식
_SPEED_LABEL_FORMAT = "{:.2f}x"


class SegmentTableWidget(QWidget):
    """구간별 배속 설정 테이블 위젯.

    QTableWidget(시작/끝 SpinBox · 배속 ComboBox) + 추가/삭제 버튼.
    set_frame_as_start/set_frame_as_end로 현재 미리보기 프레임을 구간에 주입.
    to_segments()로 rows_to_segments에 위임해 SpeedSegment 튜플 반환.

    WHY QWidget 캡슐화:
        VideoMainWindow가 비대해지지 않도록 구간 테이블 책임을 분리한다.
        통합은 생성·배선만 담당한다(SRP).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """위젯 초기화 — 테이블 + 버튼 배치."""
        super().__init__(parent)
        self._build_table()
        self._build_buttons()
        self._build_layout()

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def rowCount(self) -> int:  # noqa: N802  — QTableWidget 동형 네이밍
        """현재 테이블 행 수를 반환한다."""
        return self._table.rowCount()

    def setCurrentCell(self, row: int, col: int) -> None:  # noqa: N802
        """테이블의 현재 선택 셀을 설정한다."""
        self._table.setCurrentCell(row, col)

    def add_row(self) -> None:
        """빈 구간 행을 테이블 끝에 추가한다.

        WHY: "추가" 버튼 클릭 → 기본값(0, 0, 0.5x) 행 삽입.
             사용자가 SpinBox·ComboBox로 직접 수정한다.
        """
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._set_row_widgets(row)

    def remove_selected_row(self) -> None:
        """현재 선택된 행을 삭제한다. 선택 없으면 무시(방어).

        WHY: 빈 테이블에서 삭제 버튼 눌러도 크래시 없이 무시해야 한다.
        """
        row = self._table.currentRow()
        if row < 0:
            return
        self._table.removeRow(row)

    def set_frame_as_start(self, frame_idx: int) -> None:
        """현재 선택 행의 시작 프레임을 frame_idx로 설정한다.

        선택 행 없으면 무시(빈 테이블 방어).

        WHY: 미리보기 프레임을 보면서 버튼으로 구간 지정 — UX 핵심(Task 5-2).
        """
        row = self._table.currentRow()
        if row < 0:
            return
        spin = self._get_spinbox(row, _COL_START)
        if spin is not None:
            spin.setValue(frame_idx)

    def set_frame_as_end(self, frame_idx: int) -> None:
        """현재 선택 행의 끝 프레임을 frame_idx로 설정한다.

        선택 행 없으면 무시(빈 테이블 방어).
        """
        row = self._table.currentRow()
        if row < 0:
            return
        spin = self._get_spinbox(row, _COL_END)
        if spin is not None:
            spin.setValue(frame_idx)

    def get_row_data(self, row: int) -> tuple[int, int, float]:
        """지정 행의 (start, end, factor) 튜플을 반환한다.

        WHY: 테스트 및 to_segments에서 행 데이터를 읽을 때 사용한다.
        """
        start_spin = self._get_spinbox(row, _COL_START)
        end_spin = self._get_spinbox(row, _COL_END)
        combo = self._get_combo(row, _COL_SPEED)

        start = start_spin.value() if start_spin else 0
        end = end_spin.value() if end_spin else 0
        factor = combo.currentData() if combo else 1.0
        return (start, end, factor)

    def get_speed_combo(self, row: int) -> QComboBox | None:
        """지정 행의 배속 ComboBox를 반환한다.

        WHY: 테스트에서 ComboBox 항목을 직접 검사하기 위한 접근자.
        """
        return self._get_combo(row, _COL_SPEED)

    def to_segments(self) -> tuple[SpeedSegment, ...]:
        """현재 테이블 전체 행 데이터를 SpeedSegment 튜플로 변환한다.

        rows_to_segments에 위임 → normalize_segments 검증 포함.
        ValueError는 호출자(VideoMainWindow)가 잡아 한국어 QMessageBox 표시.
        """
        rows = [self.get_row_data(r) for r in range(self.rowCount())]
        return rows_to_segments(rows)

    # ------------------------------------------------------------------
    # 빌더 (private)
    # ------------------------------------------------------------------

    def _build_table(self) -> None:
        """QTableWidget을 초기화한다."""
        self._table = QTableWidget(0, _COL_COUNT)
        self._table.setHorizontalHeaderLabels(["시작", "끝", "배속"])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def _build_buttons(self) -> None:
        """추가·삭제 버튼을 초기화한다."""
        self._add_btn = QPushButton("+ 구간 추가")
        self._add_btn.clicked.connect(self.add_row)

        self._del_btn = QPushButton("- 선택 삭제")
        self._del_btn.clicked.connect(self.remove_selected_row)

    def _build_layout(self) -> None:
        """테이블 + 버튼을 VBox 레이아웃으로 배치한다."""
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._del_btn)
        btn_row.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("슬로우모션 구간"))
        layout.addLayout(btn_row)
        layout.addWidget(self._table)

    def _set_row_widgets(self, row: int) -> None:
        """행에 SpinBox(시작·끝) + ComboBox(배속) 위젯을 삽입한다."""
        self._table.setCellWidget(row, _COL_START, self._make_spinbox())
        self._table.setCellWidget(row, _COL_END, self._make_spinbox())
        self._table.setCellWidget(row, _COL_SPEED, self._make_speed_combo())
        # QTableWidget이 아이템 없이 위젯만 있으면 선택이 안 되는 경우 방어
        self._table.setItem(row, _COL_START, QTableWidgetItem(""))

    # ------------------------------------------------------------------
    # 위젯 팩토리 (private)
    # ------------------------------------------------------------------

    def _make_spinbox(self) -> QSpinBox:
        """프레임 인덱스 입력용 SpinBox를 생성한다."""
        spin = QSpinBox()
        spin.setMinimum(_SPINBOX_MIN)
        spin.setMaximum(_SPINBOX_MAX)
        return spin

    def _make_speed_combo(self) -> QComboBox:
        """배속 프리셋 ComboBox를 생성한다.

        WHY itemData에 float 저장:
            표시는 "0.50x" 문자열이지만 to_segments는 float가 필요하다.
            itemData(i)로 float를 꺼내 SpeedSegment에 직접 전달한다.
        """
        combo = QComboBox()
        for factor in PRESET_FACTORS:
            label = _SPEED_LABEL_FORMAT.format(factor)
            combo.addItem(label, factor)
        return combo

    # ------------------------------------------------------------------
    # 내부 접근자 (private)
    # ------------------------------------------------------------------

    def _get_spinbox(self, row: int, col: int) -> QSpinBox | None:
        """지정 셀의 QSpinBox를 반환한다. 없으면 None."""
        widget = self._table.cellWidget(row, col)
        return widget if isinstance(widget, QSpinBox) else None

    def _get_combo(self, row: int, col: int) -> QComboBox | None:
        """지정 셀의 QComboBox를 반환한다. 없으면 None."""
        widget = self._table.cellWidget(row, col)
        return widget if isinstance(widget, QComboBox) else None
