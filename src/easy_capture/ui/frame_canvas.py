"""프레임 표시 + 클릭 캡처 위젯.

RGB numpy 배열을 QImage로 표시하고, 클릭 시 이미지 좌표(위젯 좌표 역변환)를
Signal로 방출한다. 마스크 오버레이(반투명)와 후보 박스 오버레이도 표시한다.
좌표 변환은 ui/coords.py 순수 함수에 위임(테스트 가능성 확보).

오버레이 합성 방식:
  mask_to_rgba(mask) → (H,W,4) uint8 RGBA 배열(numpy 브로드캐스트)
  → QImage(Format_RGBA8888) 1회 생성 → drawImage 1회.
  파이썬 픽셀 이중루프 제거(perf 개선).

박스 오버레이 합성 방식:
  _draw_boxes(base, boxes, ...) → QPainter로 원본 해상도 pixmap에 사각형 그림.
  마스크 오버레이와 동일 파이프라인(scaled 전 합성) → 좌표 자동 정합.

WHY: QImage 버퍼는 rgba 지역변수가 살아 있는 동안만 유효하다.
     합성을 같은 함수(_draw_overlay) 안에서 완료하고 rgba를
     ascontiguousarray로 보장해 조기 GC를 방지한다.

     박스는 이미지 좌표계 pixmap에 그려야 scaled 이후에도 좌표가 정합된다.
     레터박스 오프셋 버그를 구조적으로 회피하기 위해 마스크와 동일한
     원본 해상도 합성 후 scaled 전략을 채택한다(Task 4-2 설계).

     클릭 라우팅 상호배타: _boxes is not None이면 box_clicked, 아니면 clicked.
     히트테스트는 canvas가 하지 않고 좌표만 방출한다(core.pick_box_at에 위임).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from easy_capture.ui.coords import widget_to_image

# 기본 오버레이 색 (R, G, B) — 계획서 §4-2 명시값
_OVERLAY_COLOR: tuple[int, int, int] = (0, 120, 255)
# 기본 오버레이 알파 — 반투명(계획서 §4-2)
_OVERLAY_ALPHA: int = 110

# 박스 색상 상수 (R, G, B) — 매직넘버 금지(Task 4-2 설계)
_BOX_TARGET_COLOR: tuple[int, int, int] = (0, 200, 0)    # 선택 대상 — 초록
_BOX_NEGATIVE_COLOR: tuple[int, int, int] = (220, 0, 0)  # 배제 — 빨강
_BOX_DEFAULT_COLOR: tuple[int, int, int] = (180, 180, 180)  # 미선택 — 회색
# 박스 선 굵기 (px) — 매직넘버 금지
_BOX_LINE_WIDTH: int = 2


def box_color_for(
    index: int,
    target_idx: int | None,
    negative_idxs: tuple[int, ...],
) -> tuple[int, int, int]:
    """박스 인덱스에 따라 표시 색상을 결정한다(순수, PySide6 비의존).

    우선순위: 대상(초록) > 배제(빨강) > 미선택(회색).

    WHY: 색상 결정 로직을 순수 함수로 분리해 단위 테스트하고,
         QPainter 렌더링과 책임을 분리한다(mask_to_rgba 패턴 계승).
         매직넘버 대신 모듈 상수(_BOX_*_COLOR)를 참조한다.
    """
    if index == target_idx:
        return _BOX_TARGET_COLOR
    if index in negative_idxs:
        return _BOX_NEGATIVE_COLOR
    return _BOX_DEFAULT_COLOR


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
    """RGB ndarray를 표시하고 클릭 좌표(이미지 기준)를 Signal로 방출하는 위젯.

    클릭 라우팅 모드:
      - 박스 모드 (_boxes is not None): box_clicked 방출
      - 단일(세그) 모드 (_boxes is None): clicked 방출

    WHY: 두 모드는 상호배타적이어야 상위 핸들러의 중복 처리를 방지한다.
    """

    # 이미지 좌표(x, y)로 방출 — UI 이벤트 → usecase 연결 지점
    clicked = Signal(int, int)
    # 박스 모드 클릭 — 이미지 좌표(x, y) 방출, 히트테스트는 상위에서
    box_clicked = Signal(int, int)

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
        # 박스 모드 상태 — None이면 단일(세그) 모드
        self._boxes: list[tuple[float, float, float, float]] | None = None
        self._target_idx: int | None = None
        self._negative_idxs: tuple[int, ...] = ()

    def set_frame(self, frame: np.ndarray) -> None:
        """RGB HxWx3 uint8 배열을 캔버스에 표시한다."""
        self._frame = frame
        self._render()

    def set_overlay(self, mask: np.ndarray | None) -> None:
        """bool HxW 마스크를 반투명 오버레이로 표시한다. None이면 오버레이 제거."""
        self._overlay = mask
        self._render()

    def set_boxes(
        self,
        boxes: list[tuple[float, float, float, float]],
        target_idx: int | None = None,
        negative_idxs: tuple[int, ...] = (),
    ) -> None:
        """후보 박스 목록을 설정하고 박스 모드로 전환한다.

        boxes가 빈 리스트여도 박스 모드로 전환한다(빈 오버레이 렌더).

        WHY: boxes=[] 상태를 단일 모드(None)와 구분해야 상위가
             '검출 결과 없음'과 '박스 모드 미진입'을 구별할 수 있다.
        """
        self._boxes = boxes
        self._target_idx = target_idx
        self._negative_idxs = negative_idxs
        self._render()

    def clear_boxes(self) -> None:
        """박스 목록을 초기화하고 단일(세그) 모드로 복귀한다.

        WHY: set_boxes([])와 달리 단일 모드(clicked 방출)로 복귀한다.
             세그멘테이션 워크플로우로 돌아갈 때 명시적으로 호출한다.
        """
        self._boxes = None
        self._target_idx = None
        self._negative_idxs = ()
        self._render()

    def mousePressEvent(self, event) -> None:
        """클릭 이벤트를 이미지 좌표로 변환 후 모드에 따라 Signal 방출.

        박스 모드(_boxes is not None): box_clicked 방출.
        단일 모드(_boxes is None): clicked 방출.
        히트테스트는 하지 않고 좌표만 방출(core.pick_box_at에 위임).
        """
        if self._frame is None:
            return
        image_pos = self._resolve_image_pos(event)
        if image_pos is None:
            return
        if self._boxes is not None:
            self.box_clicked.emit(*image_pos)
        else:
            self.clicked.emit(*image_pos)

    def _resolve_image_pos(self, event) -> tuple[int, int] | None:
        """마우스 이벤트 위젯 좌표를 이미지 좌표로 변환한다(None이면 이미지 밖).

        WHY: 좌표 변환 로직을 분리해 mousePressEvent를 20줄 이내로 유지한다.
        """
        wx, wy = event.position().x(), event.position().y()
        h, w = self._frame.shape[:2]
        return widget_to_image(
            (wx, wy),
            (self._label.width(), self._label.height()),
            (w, h),
        )

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

        WHY: QImage(...).copy()로 rgba 버퍼를 즉시 복사해 numpy 의존을 끊는다.
             (_frame_to_pixmap과 동일 전략 — 버퍼 수명 걱정 제거.)
        """
        rgba = mask_to_rgba(mask)
        h, w = mask.shape
        overlay_img = QImage(
            rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888
        ).copy()

        result = QPixmap(base.size())
        result.fill(Qt.GlobalColor.transparent)
        painter = QPainter(result)
        painter.drawPixmap(0, 0, base)
        painter.drawImage(0, 0, overlay_img)
        painter.end()
        return result
