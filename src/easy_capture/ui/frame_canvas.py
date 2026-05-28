"""프레임 표시 + 클릭 캡처 위젯.

RGB numpy 배열을 QImage로 표시하고, 클릭 시 이미지 좌표(위젯 좌표 역변환)를
Signal로 방출한다. 마스크 오버레이(반투명)도 표시한다.
좌표 변환은 ui/coords.py 순수 함수에 위임(테스트 가능성 확보).

오버레이 합성 방식:
  mask_to_rgba(mask) → (H,W,4) uint8 RGBA 배열(numpy 브로드캐스트)
  → QImage(Format_RGBA8888) 1회 생성 → drawImage 1회.
  파이썬 픽셀 이중루프 제거(perf 개선).

WHY: QImage 버퍼는 rgba 지역변수가 살아 있는 동안만 유효하다.
     합성을 같은 함수(_draw_overlay) 안에서 완료하고 rgba를
     ascontiguousarray로 보장해 조기 GC를 방지한다.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from easy_capture.ui.coords import widget_to_image

# 기본 오버레이 색 (R, G, B) — 계획서 §4-2 명시값
_OVERLAY_COLOR: tuple[int, int, int] = (0, 120, 255)
# 기본 오버레이 알파 — 반투명(계획서 §4-2)
_OVERLAY_ALPHA: int = 110


def mask_to_rgba(
    mask: np.ndarray,
    color: tuple[int, int, int] = _OVERLAY_COLOR,
    alpha: int = _OVERLAY_ALPHA,
) -> np.ndarray:
    """bool HxW 마스크 → (H, W, 4) uint8 RGBA 배열(순수, PySide6 비의존).

    True 픽셀: RGB == color, A == alpha.
    False 픽셀: RGBA == (0, 0, 0, 0) (완전 투명).

    반환 배열은 C-contiguous이므로 QImage 버퍼로 직접 사용 가능하다.

    WHY: 색·알파 결정 로직을 순수 numpy 함수로 분리해 단위 테스트하고,
         QImage/QPainter는 합성만 담당한다. 파이썬 픽셀 루프 대비
         numpy 브로드캐스트로 수십 ms → 수 ms로 단축.
    """
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask, 0] = color[0]
    rgba[mask, 1] = color[1]
    rgba[mask, 2] = color[2]
    rgba[mask, 3] = alpha
    return np.ascontiguousarray(rgba)


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
        """마스크를 반투명 오버레이로 base 위에 그린다(numpy 벡터화).

        mask_to_rgba로 RGBA 배열을 생성하고 QImage 1회 생성 후
        drawImage 1회로 합성한다. 파이썬 픽셀 이중루프 없음.

        WHY: rgba는 QImage가 버퍼를 참조하므로 painter.end() 완료까지
             지역 변수로 살려 두어야 한다(조기 GC 방지).
             ascontiguousarray는 mask_to_rgba에서 이미 보장한다.
        """
        rgba = mask_to_rgba(mask)
        h, w = mask.shape
        overlay_img = QImage(
            rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888
        )

        result = QPixmap(base.size())
        result.fill(Qt.GlobalColor.transparent)
        painter = QPainter(result)
        painter.drawPixmap(0, 0, base)
        painter.drawImage(0, 0, overlay_img)
        painter.end()

        # WHY: rgba를 painter.end() 이후까지 살려 두어 GC를 방지한다.
        del rgba
        return result
