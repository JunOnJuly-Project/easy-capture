"""프레임 표시 + 클릭 캡처 위젯.

RGB numpy 배열을 QImage로 표시하고, 클릭 시 이미지 좌표(위젯 좌표 역변환)를
Signal로 방출한다. 마스크 오버레이(반투명)도 표시한다.
좌표 변환은 ui/coords.py 순수 함수에 위임(테스트 가능성 확보).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from easy_capture.ui.coords import widget_to_image


class FrameCanvas(QWidget):
    """RGB ndarray를 표시하고 클릭 좌표(이미지 기준)를 Signal로 방출하는 위젯."""

    # 이미지 좌표(x, y)로 방출 — UI 이벤트 → usecase 연결 지점
    clicked = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._frame: np.ndarray | None = None
        self._overlay: np.ndarray | None = None  # bool HxW 마스크

    def set_frame(self, frame: np.ndarray) -> None:
        """RGB HxWx3 uint8 배열을 캔버스에 표시한다."""
        self._frame = frame
        self._render()

    def set_overlay(self, mask: np.ndarray | None) -> None:
        """bool HxW 마스크를 반투명 오버레이로 표시한다. None이면 오버레이 제거."""
        self._overlay = mask
        self._render()

    def mousePressEvent(self, event) -> None:
        """클릭 이벤트를 이미지 좌표로 변환 후 Signal 방출."""
        if self._frame is None:
            return
        wx, wy = event.position().x(), event.position().y()
        h, w = self._frame.shape[:2]
        image_pos = widget_to_image(
            (wx, wy),
            (self._label.width(), self._label.height()),
            (w, h),
        )
        if image_pos is not None:
            self.clicked.emit(*image_pos)

    # ------------------------------------------------------------------
    # 내부 렌더링
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """프레임과 오버레이를 합성해 QLabel에 표시한다."""
        if self._frame is None:
            return
        pixmap = self._frame_to_pixmap(self._frame)
        if self._overlay is not None:
            pixmap = self._draw_overlay(pixmap, self._overlay)
        self._label.setPixmap(
            pixmap.scaled(
                self._label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _frame_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """RGB ndarray → QPixmap 변환."""
        h, w, _ = frame.shape
        img = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(img.copy())

    def _draw_overlay(self, base: QPixmap, mask: np.ndarray) -> QPixmap:
        """마스크를 반투명 파란색으로 base 위에 그린다."""
        # WHY: 별도 QPixmap에 마스크를 그려 base에 합성 — 원본 프레임 보존
        overlay_pixmap = QPixmap(base.size())
        overlay_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(overlay_pixmap)
        painter.setOpacity(0.4)
        mask_h, mask_w = mask.shape
        color = QColor(0, 120, 255, 180)
        for y in range(mask_h):
            for x in range(mask_w):
                if mask[y, x]:
                    painter.fillRect(x, y, 1, 1, color)
        painter.end()

        result = QPixmap(base.size())
        result.fill(Qt.GlobalColor.transparent)
        result_painter = QPainter(result)
        result_painter.drawPixmap(0, 0, base)
        result_painter.drawPixmap(0, 0, overlay_pixmap)
        result_painter.end()
        return result
