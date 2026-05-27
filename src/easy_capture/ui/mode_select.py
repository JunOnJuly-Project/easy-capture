"""시작 화면: 이미지(짤) / GIF(움짤) 모드 선택 (기능 분리 진입점)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)


class ModeSelectWindow(QWidget):
    """모드 선택 창. 선택 시 mode_selected('image'|'gif') 시그널 방출."""

    mode_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("easy-capture")
        self.resize(520, 320)
        self._build_ui()

    def _build_ui(self) -> None:
        title = QLabel("무엇을 만들까요?")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        buttons = QHBoxLayout()
        for label, mode in (("🖼  이미지 (짤)", "image"), ("🎞  GIF (움짤)", "gif")):
            btn = QPushButton(label)
            btn.setMinimumHeight(120)
            btn.clicked.connect(lambda _checked=False, m=mode: self.mode_selected.emit(m))
            buttons.addWidget(btn)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(buttons)
