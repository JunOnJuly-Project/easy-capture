"""FrameCanvas 후보 박스 오버레이 + box_clicked Signal 테스트 (Task 4-2).

대상 모듈: easy_capture.ui.frame_canvas
추가 API:
  box_clicked = Signal(int, int)
  def set_boxes(boxes, target_idx=None, negative_idxs=()) -> None
  def clear_boxes() -> None
  def box_color_for(index, target_idx, negative_idxs) -> tuple[int,int,int]

테스트 범위:
  1. 순수 함수 box_color_for — 대상/배제/미선택 3케이스
  2. 위젯 set_boxes/clear_boxes 예외 없이 렌더 완료
  3. 박스 모드 mousePressEvent → box_clicked 방출 (clicked 미방출)
  4. 단일 모드(boxes=None) mousePressEvent → clicked 방출 (box_clicked 미방출)
  5. 공존: set_overlay(mask) + set_boxes 동시 — 크래시 없음
  6. 무회귀: 기존 clicked/set_frame/mask_to_rgba 동작 보존

offscreen 헤드리스 필수(CI·모니터리스 환경).
QApplication 싱글턴: 중복 생성 방지(기존 UI 테스트 패턴 계승).

WHY: 박스 클릭 라우팅(clicked↔box_clicked 상호배타)은 UI 계층 계약 핵심이므로
     Signal 방출 여부를 직접 검증한다. set_boxes 미호출(None) 상태는 레거시 동작
     보장을 위해 반드시 기존 clicked를 방출해야 한다.
"""
from __future__ import annotations

import os

# PySide6 위젯 import 전 헤드리스 플랫폼 지정 — CI·offscreen 환경 대응
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 미설치 — UI 모듈 import 불가")

from PySide6.QtCore import QPoint
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

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


@pytest.fixture
def app():
    """QApplication 싱글턴 픽스처."""
    return _get_app()


# ---------------------------------------------------------------------------
# 테스트용 픽스처 헬퍼
# ---------------------------------------------------------------------------
# 테스트 프레임 크기 (작게 유지해 렌더 오버헤드 최소화)
_FRAME_H = 100
_FRAME_W = 150

# 테스트용 후보 박스 (x1, y1, x2, y2) — 이미지 좌표
_BOX_A: tuple[float, float, float, float] = (10.0, 10.0, 50.0, 80.0)
_BOX_B: tuple[float, float, float, float] = (60.0, 10.0, 120.0, 80.0)
_SAMPLE_BOXES = [_BOX_A, _BOX_B]

# 클릭점 — _BOX_A 내부 이미지 좌표
_CLICK_IN_BOX_A_X = 30
_CLICK_IN_BOX_A_Y = 40


def _make_frame() -> np.ndarray:
    """테스트용 RGB ndarray 프레임을 생성한다."""
    return np.zeros((_FRAME_H, _FRAME_W, 3), dtype=np.uint8)


def _make_mask() -> np.ndarray:
    """테스트용 bool HxW 마스크를 생성한다."""
    mask = np.zeros((_FRAME_H, _FRAME_W), dtype=bool)
    mask[10:30, 10:30] = True
    return mask


def _make_canvas(app):
    """FrameCanvas 인스턴스를 생성하고 최소 크기를 설정한다."""
    from easy_capture.ui.frame_canvas import FrameCanvas

    canvas = FrameCanvas()
    # 좌표 변환이 의미 있으려면 위젯 크기 필요
    canvas.resize(_FRAME_W * 2, _FRAME_H * 2)
    return canvas


def _emit_mouse_click(canvas, img_x: int, img_y: int):
    """이미지 좌표 (img_x, img_y)에 해당하는 위젯 좌표로 마우스 클릭 이벤트를 방출한다.

    WHY: 레터박스 보정 없이 1:1 위젯→이미지 매핑이 성립하도록
         canvas.resize와 frame 크기를 동일 비율로 맞춰 두고,
         이미지 좌표 == 위젯 좌표 + offset 관계를 이용한다.
         실제 좌표 변환은 widget_to_image가 담당하므로,
         위젯 좌표를 직접 시뮬레이션 대신 canvas._label 크기를 기준으로 계산한다.
    """
    from easy_capture.ui.coords import image_to_widget

    lw = canvas._label.width()
    lh = canvas._label.height()
    wx, wy = image_to_widget((img_x, img_y), (lw, lh), (_FRAME_W, _FRAME_H))

    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(wx, wy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(event)


# ===========================================================================
# 1. 순수 함수 box_color_for 테스트
# ===========================================================================
class TestBoxColorFor:
    """box_color_for — 인덱스별 색상 결정 순수 함수 계약."""

    def test_대상_인덱스는_초록색을_반환한다(self):
        """Given: index == target_idx
        When:  box_color_for 호출
        Then:  _BOX_TARGET_COLOR (0, 200, 0)
        """
        from easy_capture.ui.frame_canvas import _BOX_TARGET_COLOR, box_color_for

        result = box_color_for(0, target_idx=0, negative_idxs=())

        assert result == _BOX_TARGET_COLOR

    def test_배제_인덱스는_빨간색을_반환한다(self):
        """Given: index in negative_idxs
        When:  box_color_for 호출
        Then:  _BOX_NEGATIVE_COLOR (220, 0, 0)
        """
        from easy_capture.ui.frame_canvas import _BOX_NEGATIVE_COLOR, box_color_for

        result = box_color_for(1, target_idx=0, negative_idxs=(1, 2))

        assert result == _BOX_NEGATIVE_COLOR

    def test_미선택_인덱스는_기본_회색을_반환한다(self):
        """Given: index not in negative_idxs and index != target_idx
        When:  box_color_for 호출
        Then:  _BOX_DEFAULT_COLOR (180, 180, 180)
        """
        from easy_capture.ui.frame_canvas import _BOX_DEFAULT_COLOR, box_color_for

        result = box_color_for(2, target_idx=0, negative_idxs=(1,))

        assert result == _BOX_DEFAULT_COLOR

    def test_target_idx가_None이면_미선택_색을_반환한다(self):
        """Given: target_idx=None
        When:  box_color_for 호출
        Then:  _BOX_DEFAULT_COLOR
        """
        from easy_capture.ui.frame_canvas import _BOX_DEFAULT_COLOR, box_color_for

        result = box_color_for(0, target_idx=None, negative_idxs=())

        assert result == _BOX_DEFAULT_COLOR


# ===========================================================================
# 2. set_boxes / clear_boxes 렌더 완료 테스트
# ===========================================================================
class TestSetBoxes:
    """set_boxes/clear_boxes — 예외 없이 렌더 완료 계약."""

    def test_set_frame_후_set_boxes_호출이_예외를_발생시키지_않는다(self, app):
        """Given: set_frame으로 프레임 설정 후
        When:  set_boxes(boxes) 호출
        Then:  예외 없이 정상 완료
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())

        # 예외 없으면 통과
        canvas.set_boxes(_SAMPLE_BOXES)

    def test_set_boxes_target_idx_지정이_예외를_발생시키지_않는다(self, app):
        """Given: 프레임 설정 후
        When:  set_boxes(boxes, target_idx=0) 호출
        Then:  예외 없음
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())

        canvas.set_boxes(_SAMPLE_BOXES, target_idx=0)

    def test_set_boxes_negative_idxs_지정이_예외를_발생시키지_않는다(self, app):
        """Given: 프레임 설정 후
        When:  set_boxes(boxes, target_idx=0, negative_idxs=(1,)) 호출
        Then:  예외 없음
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())

        canvas.set_boxes(_SAMPLE_BOXES, target_idx=0, negative_idxs=(1,))

    def test_clear_boxes_호출이_예외를_발생시키지_않는다(self, app):
        """Given: set_boxes 후
        When:  clear_boxes() 호출
        Then:  예외 없음, _boxes가 None/빈 리스트 상태
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())
        canvas.set_boxes(_SAMPLE_BOXES)

        canvas.clear_boxes()

    def test_프레임_없이_set_boxes_호출해도_예외_없다(self, app):
        """Given: 프레임 미설정 상태
        When:  set_boxes 호출
        Then:  예외 없음 (프레임 없으면 렌더 스킵)
        """
        canvas = _make_canvas(app)

        canvas.set_boxes(_SAMPLE_BOXES)

    def test_빈_리스트로_set_boxes_호출해도_예외_없다(self, app):
        """Given: 빈 박스 리스트
        When:  set_boxes([]) 호출
        Then:  예외 없음
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())

        canvas.set_boxes([])


# ===========================================================================
# 3. 박스 모드 — box_clicked 방출 (클릭 라우팅 상호배타)
# ===========================================================================
class TestBoxClickedSignal:
    """박스 모드 mousePressEvent → box_clicked 방출, clicked 미방출."""

    def test_박스_모드에서_클릭_시_box_clicked가_방출된다(self, app):
        """Given: set_boxes 호출(박스 모드), 프레임 설정
        When:  이미지 내 좌표 클릭
        Then:  box_clicked 방출
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())
        canvas.set_boxes(_SAMPLE_BOXES)

        received: list[tuple[int, int]] = []
        canvas.box_clicked.connect(lambda x, y: received.append((x, y)))

        _emit_mouse_click(canvas, _CLICK_IN_BOX_A_X, _CLICK_IN_BOX_A_Y)

        assert len(received) == 1, f"box_clicked 방출 횟수 불일치: {len(received)}"

    def test_박스_모드에서_클릭_시_clicked는_방출되지_않는다(self, app):
        """Given: 박스 모드
        When:  클릭
        Then:  clicked 미방출 (상호배타)

        WHY: clicked는 세그멘테이션 포인트 모드 Signal이다.
             박스 모드와 혼용되면 상위 핸들러가 중복 처리할 수 있다.
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())
        canvas.set_boxes(_SAMPLE_BOXES)

        clicked_received: list[tuple[int, int]] = []
        canvas.clicked.connect(lambda x, y: clicked_received.append((x, y)))

        _emit_mouse_click(canvas, _CLICK_IN_BOX_A_X, _CLICK_IN_BOX_A_Y)

        assert len(clicked_received) == 0, (
            f"박스 모드에서 clicked가 잘못 방출됨: {clicked_received}"
        )

    def test_박스_모드_방출_좌표는_이미지_좌표다(self, app):
        """Given: 박스 모드, 이미지 내 클릭
        When:  box_clicked 방출
        Then:  방출된 (x, y)가 이미지 범위 내(0 이상, 이미지 크기 미만)

        WHY: box_clicked는 이미지 좌표를 방출해야 상위가 pick_box_at에 바로 사용 가능.
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())
        canvas.set_boxes(_SAMPLE_BOXES)

        received: list[tuple[int, int]] = []
        canvas.box_clicked.connect(lambda x, y: received.append((x, y)))

        _emit_mouse_click(canvas, _CLICK_IN_BOX_A_X, _CLICK_IN_BOX_A_Y)

        assert received, "box_clicked가 방출되지 않음"
        x, y = received[0]
        assert 0 <= x < _FRAME_W, f"x 좌표 범위 이상: {x}"
        assert 0 <= y < _FRAME_H, f"y 좌표 범위 이상: {y}"


# ===========================================================================
# 4. 단일 모드(boxes=None) — clicked 방출
# ===========================================================================
class TestSingleModeClickedSignal:
    """단일(세그) 모드 mousePressEvent → clicked 방출, box_clicked 미방출."""

    def test_단일_모드에서_클릭_시_clicked가_방출된다(self, app):
        """Given: set_boxes 미호출(단일 모드), 프레임 설정
        When:  이미지 내 좌표 클릭
        Then:  clicked 방출
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())
        # set_boxes 미호출 → 단일 모드

        received: list[tuple[int, int]] = []
        canvas.clicked.connect(lambda x, y: received.append((x, y)))

        _emit_mouse_click(canvas, _CLICK_IN_BOX_A_X, _CLICK_IN_BOX_A_Y)

        assert len(received) == 1, f"clicked 방출 횟수 불일치: {len(received)}"

    def test_단일_모드에서_클릭_시_box_clicked는_방출되지_않는다(self, app):
        """Given: 단일 모드
        When:  클릭
        Then:  box_clicked 미방출

        WHY: 단일 모드는 세그멘테이션 포인트 선택이므로 box_clicked는 무관하다.
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())

        box_clicked_received: list[tuple[int, int]] = []
        canvas.box_clicked.connect(lambda x, y: box_clicked_received.append((x, y)))

        _emit_mouse_click(canvas, _CLICK_IN_BOX_A_X, _CLICK_IN_BOX_A_Y)

        assert len(box_clicked_received) == 0, (
            f"단일 모드에서 box_clicked가 잘못 방출됨: {box_clicked_received}"
        )

    def test_clear_boxes_후_단일_모드로_복귀한다(self, app):
        """Given: 박스 모드(set_boxes) 후 clear_boxes 호출
        When:  클릭
        Then:  clicked 방출 (단일 모드 복귀)

        WHY: clear_boxes가 모드를 완전히 초기화해야 레거시 세그 동작이 복구된다.
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())
        canvas.set_boxes(_SAMPLE_BOXES)
        canvas.clear_boxes()

        received: list[tuple[int, int]] = []
        canvas.clicked.connect(lambda x, y: received.append((x, y)))

        _emit_mouse_click(canvas, _CLICK_IN_BOX_A_X, _CLICK_IN_BOX_A_Y)

        assert len(received) == 1, (
            f"clear_boxes 후 clicked 미방출 — 단일 모드 복귀 실패: {received}"
        )


# ===========================================================================
# 5. 공존 테스트 — set_overlay + set_boxes 동시
# ===========================================================================
class TestOverlayAndBoxesCoexist:
    """마스크 오버레이 + 박스 동시 렌더 — 크래시 없음 계약."""

    def test_overlay와_boxes_동시_설정이_예외를_발생시키지_않는다(self, app):
        """Given: set_frame + set_overlay(mask) + set_boxes(boxes)
        When:  두 오버레이 동시 활성
        Then:  크래시 없음
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())
        canvas.set_overlay(_make_mask())

        canvas.set_boxes(_SAMPLE_BOXES, target_idx=0, negative_idxs=(1,))

    def test_overlay_후_set_boxes_재렌더가_예외_없다(self, app):
        """Given: 오버레이 설정 후 박스 변경
        When:  set_boxes 재호출
        Then:  예외 없음
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())
        canvas.set_overlay(_make_mask())
        canvas.set_boxes(_SAMPLE_BOXES, target_idx=0)

        # 박스 재설정
        canvas.set_boxes(_SAMPLE_BOXES, target_idx=1, negative_idxs=(0,))

    def test_boxes_후_overlay_재설정이_예외_없다(self, app):
        """Given: 박스 설정 후 오버레이 변경
        When:  set_overlay 재호출
        Then:  예외 없음
        """
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())
        canvas.set_boxes(_SAMPLE_BOXES)
        canvas.set_overlay(_make_mask())


# ===========================================================================
# 6. 무회귀 — 기존 기능 보존
# ===========================================================================
class TestRegression:
    """기존 clicked/set_frame/mask_to_rgba 무회귀 검증."""

    def test_set_frame이_예외_없이_완료된다(self, app):
        """기존 set_frame 동작 무회귀."""
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())

    def test_set_overlay_None으로_오버레이_제거가_예외_없이_완료된다(self, app):
        """기존 set_overlay(None) 동작 무회귀."""
        canvas = _make_canvas(app)
        canvas.set_frame(_make_frame())
        canvas.set_overlay(_make_mask())
        canvas.set_overlay(None)

    def test_mask_to_rgba_기존_동작이_보존된다(self):
        """기존 mask_to_rgba 순수 함수 무회귀."""
        from easy_capture.ui.frame_canvas import mask_to_rgba

        mask = np.zeros((_FRAME_H, _FRAME_W), dtype=bool)
        mask[0, 0] = True

        result = mask_to_rgba(mask)

        assert result.shape == (_FRAME_H, _FRAME_W, 4)
        assert result.dtype == np.uint8
        assert result[0, 0, 3] > 0  # True 픽셀 불투명

    def test_클릭_Signal_시그니처가_int_int이다(self, app):
        """기존 clicked Signal의 (int, int) 시그니처 무회귀."""
        canvas = _make_canvas(app)

        # Signal이 int, int 두 인자 형태인지 connect로 확인
        received: list[tuple] = []
        canvas.clicked.connect(lambda x, y: received.append((x, y)))

        canvas.set_frame(_make_frame())
        _emit_mouse_click(canvas, _CLICK_IN_BOX_A_X, _CLICK_IN_BOX_A_Y)

        assert len(received) == 1
        x, y = received[0]
        assert isinstance(x, int)
        assert isinstance(y, int)
