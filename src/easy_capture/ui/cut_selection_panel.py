"""컷별 추적 대상 선택 패널 위젯 (Story 4 Task 4-3).

멀티샷 군무에서 각 샷(컷)의 추적 대상을 사용자가 명시 선택하는 우측 패널.
샷 네비게이션(이전/다음) + 후보별 대상(positive, 라디오 1개)·배제(negative, 체크 다수)
토글을 제공하고, to_choices()로 core 입력 DTO(dict[int, ShotChoice])를 반환한다.

설계 원칙:
  - 패널은 인덱스(int)만 다룬다. 좌표/box 변환은 일절 하지 않는다(core 경계 유지).
    Detection→box 변환은 video_window의 책임이다.
  - 상태(self._choices)를 단일 소스로 두고, 위젯은 _sync_widgets로 반영만 한다.
    위젯 신호 재진입은 blockSignals로 차단해 selection_changed 중복 방출을 막는다.
  - SegmentTableWidget 캡슐화 패턴 계승 — video_window 비대화 방지(SRP).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from easy_capture.core.tracking.cut_selection import ShotChoice

if TYPE_CHECKING:
    from easy_capture.app.video_capture import ShotCandidates

# 첫 샷 인덱스(0-기반 하한) — 매직넘버 금지
_FIRST_SHOT_INDEX = 0

# 샷 표시 라벨 — current는 1-기반(사용자 친화)
_SHOT_LABEL_TEMPLATE = "샷 {current} / {total}"
_SHOT_LABEL_EMPTY = "샷 없음"

# 후보 행 라벨 접두 — "후보 {i}"
_CANDIDATE_LABEL_TEMPLATE = "후보 {index}"
_TARGET_BUTTON_TEXT = "대상"
_NEGATIVE_BUTTON_TEXT = "배제"


class CutSelectionPanel(QWidget):
    """샷별 추적 대상(positive)·배제(negative)를 선택하는 패널.

    shot_changed:      현재 샷 인덱스 변경 시 방출(video_window가 썸네일·박스 갱신).
    selection_changed: positive/negative 토글 시 방출(미리보기·버튼 갱신 트리거).

    WHY: 외부(video_window)가 Signal로 샷 이동·선택 변경을 감지해 캔버스 박스
         색과 추적 준비 상태를 갱신할 수 있게 한다.
    """

    shot_changed = Signal(int)
    selection_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._shots: list = []
        self._current: int = _FIRST_SHOT_INDEX
        self._choices: dict[int, ShotChoice] = {}
        # 현재 샷 후보 위젯 — _rebuild_for_current_shot에서 재생성
        self._radios: list[QRadioButton] = []
        self._checks: list[QCheckBox] = []
        self._radio_group = QButtonGroup(self)
        self._radio_group.setExclusive(True)
        self._build_layout()

    # ------------------------------------------------------------------
    # UI 빌더
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """네비게이션 + 샷 라벨 + 후보 행 컨테이너를 수직 배치한다."""
        root = QVBoxLayout(self)
        self._build_nav(root)
        self._shot_label = QLabel(_SHOT_LABEL_EMPTY)
        root.addWidget(self._shot_label)
        self._rows_layout = QVBoxLayout()
        root.addLayout(self._rows_layout)
        root.addStretch()

    def _build_nav(self, root: QVBoxLayout) -> None:
        """[◀ 이전] [다음 ▶] 네비게이션 버튼 행을 구성한다."""
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("◀ 이전")
        self._prev_btn.clicked.connect(self._on_prev)
        self._prev_btn.setEnabled(False)
        nav.addWidget(self._prev_btn)

        self._next_btn = QPushButton("다음 ▶")
        self._next_btn.clicked.connect(self._on_next)
        self._next_btn.setEnabled(False)
        nav.addWidget(self._next_btn)
        root.addLayout(nav)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def set_shots(self, shots: "list[ShotCandidates]") -> None:
        """샷·후보를 주입하고 선택 상태를 초기화한다(첫 샷부터 시작).

        WHY: 새 영상/구간 로드 시 이전 선택이 잔류하면 잘못된 추적이 된다.
             set_shots는 항상 _choices와 현재 샷을 초기화한다.
        """
        self._shots = list(shots)
        self._current = _FIRST_SHOT_INDEX
        self._choices = {}
        self._rebuild_for_current_shot()
        self._update_nav_buttons()

    def current_shot_index(self) -> int:
        """현재 보고 있는 샷 인덱스(0-기반)를 반환한다."""
        return self._current

    def set_target(self, candidate_idx: int | None) -> None:
        """현재 샷의 positive(추적 대상)를 지정한다(None이면 해제).

        지정한 인덱스는 negative에서 자동 제외한다(상호배타).
        WHY: positive는 샷당 1개(라디오). 새 지정 시 기존을 교체하고,
             같은 인덱스가 negative였다면 모순을 피해 negative에서 뺀다.
        """
        choice = self.current_choice()
        negatives = set(choice.negative_idxs)
        if candidate_idx is not None:
            negatives.discard(candidate_idx)
        self._set_choice(candidate_idx, negatives)
        self._sync_widgets()
        self.selection_changed.emit()

    def toggle_negative(self, candidate_idx: int) -> None:
        """현재 샷의 negative(배제) 여부를 토글한다.

        WHY: 체크박스 토글 동작. 추가 시 그 인덱스가 positive였다면 해제한다
             (상호배타). 이미 negative면 제거(배제 해제).
        """
        add = candidate_idx not in self.current_choice().negative_idxs
        self._apply_negative(candidate_idx, add)

    def current_choice(self) -> ShotChoice:
        """현재 샷의 선택을 반환한다(미선택이면 빈 ShotChoice)."""
        return self._choices.get(
            self._current, ShotChoice(target_idx=None, negative_idxs=())
        )

    def to_choices(self) -> dict[int, ShotChoice]:
        """선택이 있는 샷만 {shot_index: ShotChoice}로 반환한다(core 입력 DTO).

        WHY: 미선택 샷(target None·negative 없음)은 키에서 제외한다.
             build_selections_from_choices가 미선택 샷을 자동 재매칭 폴백으로
             처리하므로 KeyError 없이 정합한다.
        """
        return {
            shot_index: choice
            for shot_index, choice in self._choices.items()
            if choice.target_idx is not None or choice.negative_idxs
        }

    # ------------------------------------------------------------------
    # 내부 — 선택 상태 갱신
    # ------------------------------------------------------------------

    def _apply_negative(self, candidate_idx: int, add: bool) -> None:
        """negative 집합에 인덱스를 추가/제거하고 위젯·Signal을 갱신한다."""
        choice = self.current_choice()
        negatives = set(choice.negative_idxs)
        target = choice.target_idx
        if add:
            negatives.add(candidate_idx)
            if target == candidate_idx:
                target = None  # positive→negative 이동(상호배타)
        else:
            negatives.discard(candidate_idx)
        self._set_choice(target, negatives)
        self._sync_widgets()
        self.selection_changed.emit()

    def _set_choice(self, target_idx: int | None, negatives) -> None:
        """현재 샷의 ShotChoice를 갱신한다(negative는 결정적 정렬 튜플)."""
        self._choices[self._current] = ShotChoice(
            target_idx=target_idx,
            negative_idxs=tuple(sorted(negatives)),
        )

    # ------------------------------------------------------------------
    # 내부 — 네비게이션
    # ------------------------------------------------------------------

    def _on_prev(self) -> None:
        """이전 샷으로 이동한다(경계에서는 무시)."""
        if self._current > _FIRST_SHOT_INDEX:
            self._goto(self._current - 1)

    def _on_next(self) -> None:
        """다음 샷으로 이동한다(경계에서는 무시)."""
        if self._current < len(self._shots) - 1:
            self._goto(self._current + 1)

    def _goto(self, shot_index: int) -> None:
        """샷을 전환하고 후보 행·네비 버튼을 갱신한 뒤 shot_changed를 방출한다."""
        self._current = shot_index
        self._rebuild_for_current_shot()
        self._update_nav_buttons()
        self.shot_changed.emit(shot_index)

    def _update_nav_buttons(self) -> None:
        """현재 샷 위치에 따라 이전/다음 버튼 활성 상태를 갱신한다."""
        n_shots = len(self._shots)
        self._prev_btn.setEnabled(self._current > _FIRST_SHOT_INDEX)
        self._next_btn.setEnabled(self._current < n_shots - 1)

    # ------------------------------------------------------------------
    # 내부 — 후보 행 렌더링
    # ------------------------------------------------------------------

    def _rebuild_for_current_shot(self) -> None:
        """현재 샷의 후보 행(라디오/체크)을 재생성하고 선택 상태를 반영한다."""
        self._clear_rows()
        self._radios = []
        self._checks = []
        self._radio_group = QButtonGroup(self)
        self._radio_group.setExclusive(True)
        self._shot_label.setText(self._shot_label_text())
        for index in range(self._current_candidate_count()):
            self._add_candidate_row(index)
        self._sync_widgets()

    def _shot_label_text(self) -> str:
        """현재 샷 위치 표시 텍스트를 만든다(1-기반)."""
        if not self._shots:
            return _SHOT_LABEL_EMPTY
        return _SHOT_LABEL_TEMPLATE.format(
            current=self._current + 1, total=len(self._shots)
        )

    def _current_candidate_count(self) -> int:
        """현재 샷의 후보 수를 반환한다(샷 없으면 0)."""
        if not self._shots:
            return 0
        return len(self._shots[self._current].candidates)

    def _add_candidate_row(self, index: int) -> None:
        """후보 1개의 [대상 라디오][배제 체크][라벨] 행을 추가한다."""
        row = QWidget()
        layout = QHBoxLayout(row)
        radio = QRadioButton(_TARGET_BUTTON_TEXT)
        self._radio_group.addButton(radio, index)
        radio.toggled.connect(
            lambda checked, i=index: checked and self.set_target(i)
        )
        check = QCheckBox(_NEGATIVE_BUTTON_TEXT)
        check.toggled.connect(
            lambda checked, i=index: self._apply_negative(i, checked)
        )
        layout.addWidget(radio)
        layout.addWidget(check)
        layout.addWidget(QLabel(_CANDIDATE_LABEL_TEMPLATE.format(index=index)))
        layout.addStretch()
        self._rows_layout.addWidget(row)
        self._radios.append(radio)
        self._checks.append(check)

    def _clear_rows(self) -> None:
        """후보 행 컨테이너의 모든 위젯을 제거한다."""
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    # ------------------------------------------------------------------
    # 내부 — 위젯 동기화(상태 → 위젯, 신호 재진입 차단)
    # ------------------------------------------------------------------

    def _sync_widgets(self) -> None:
        """현재 선택 상태를 라디오/체크 위젯에 반영한다(blockSignals로 재진입 차단)."""
        choice = self.current_choice()
        self._sync_radios(choice.target_idx)
        self._sync_checks(choice.negative_idxs)

    def _sync_radios(self, target_idx: int | None) -> None:
        """positive 라디오 상태를 동기화한다(None이면 전체 해제).

        WHY: exclusive 그룹은 프로그램적 전체 해제가 막히므로 setExclusive(False)로
             잠시 풀어 모두 끄고 다시 배타로 되돌린다. blockSignals로 toggled 재진입 차단.
        """
        self._radio_group.setExclusive(False)
        for index, radio in enumerate(self._radios):
            radio.blockSignals(True)
            radio.setChecked(index == target_idx)
            radio.blockSignals(False)
        self._radio_group.setExclusive(True)

    def _sync_checks(self, negative_idxs) -> None:
        """negative 체크 상태를 동기화한다(blockSignals로 toggled 재진입 차단)."""
        for index, check in enumerate(self._checks):
            check.blockSignals(True)
            check.setChecked(index in negative_idxs)
            check.blockSignals(False)
