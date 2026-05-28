"""ui/coords 좌표 변환 순수 함수 테스트.

PySide6 비의존 — 튜플/숫자만 사용하므로 UI 없이 실행 가능.
레터박스·스케일·경계값 케이스를 망라한다.
"""
from __future__ import annotations

import pytest

from easy_capture.ui.coords import (
    ScaleInfo,
    compute_scale_info,
    image_to_widget,
    widget_to_image,
)


# ---------------------------------------------------------------------------
# compute_scale_info 테스트
# ---------------------------------------------------------------------------
class TestComputeScaleInfo:
    """compute_scale_info: 레터박스 스케일·오프셋 계산."""

    def test_정방형_이미지_정방형_위젯은_scale_1이다(self):
        """Given: 100x100 이미지, 100x100 위젯
        Then:  scale=1, offset=(0,0)
        """
        info = compute_scale_info((100, 100), (100, 100))

        assert info.scale == pytest.approx(1.0)
        assert info.offset_x == pytest.approx(0.0)
        assert info.offset_y == pytest.approx(0.0)

    def test_이미지보다_넓은_위젯은_수평_여백이_생긴다(self):
        """Given: 100x100 이미지, 200x100 위젯 (가로로 2배 넓음)
        Then:  scale=1.0, offset_x=50 (좌우 여백 각 50px)
        """
        info = compute_scale_info((200, 100), (100, 100))

        assert info.scale == pytest.approx(1.0)
        assert info.offset_x == pytest.approx(50.0)
        assert info.offset_y == pytest.approx(0.0)

    def test_이미지보다_높은_위젯은_수직_여백이_생긴다(self):
        """Given: 100x100 이미지, 100x200 위젯 (세로로 2배 높음)
        Then:  scale=1.0, offset_y=50 (상하 여백 각 50px)
        """
        info = compute_scale_info((100, 200), (100, 100))

        assert info.scale == pytest.approx(1.0)
        assert info.offset_x == pytest.approx(0.0)
        assert info.offset_y == pytest.approx(50.0)

    def test_2배_스케일_업은_scale_2를_반환한다(self):
        """Given: 50x50 이미지, 100x100 위젯
        Then:  scale=2.0
        """
        info = compute_scale_info((100, 100), (50, 50))

        assert info.scale == pytest.approx(2.0)

    def test_가로_종횡비_이미지는_세로_레터박스가_생긴다(self):
        """Given: 100x50 이미지(16:9 가로), 100x100 위젯
        When:  aspect-fit 적용
        Then:  scale=1.0, offset_y=25 (상하 여백)
        """
        info = compute_scale_info((100, 100), (100, 50))

        assert info.scale == pytest.approx(1.0)
        assert info.offset_y == pytest.approx(25.0)

    def test_영_크기_이미지는_기본값을_반환한다(self):
        """Given: 0x0 이미지 (경계값)
        Then:  scale=1.0, offset=(0,0) — ZeroDivision 없음
        """
        info = compute_scale_info((100, 100), (0, 0))

        assert info.scale == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# widget_to_image 테스트
# ---------------------------------------------------------------------------
class TestWidgetToImage:
    """widget_to_image: 위젯 좌표 → 이미지 픽셀 좌표 역변환."""

    def test_스케일1_오프셋0_좌표는_그대로_반환된다(self):
        """Given: 위젯=이미지 크기(100x100), 클릭 (30, 40)
        Then:  이미지 좌표 == (30, 40)
        """
        result = widget_to_image((30.0, 40.0), (100, 100), (100, 100))

        assert result == (30, 40)

    def test_2배_스케일에서_위젯_중앙은_이미지_중앙이다(self):
        """Given: 50x50 이미지, 100x100 위젯(scale=2)
        When:  위젯 중앙 (50, 50) 클릭
        Then:  이미지 좌표 == (25, 25)
        """
        result = widget_to_image((50.0, 50.0), (100, 100), (50, 50))

        assert result == (25, 25)

    def test_레터박스_여백_클릭은_None을_반환한다(self):
        """Given: 100x100 이미지, 200x100 위젯(수평 여백 각 50px)
        When:  여백 영역 (10, 50) 클릭 (offset_x=50 바깥)
        Then:  None 반환
        """
        result = widget_to_image((10.0, 50.0), (200, 100), (100, 100))

        assert result is None

    def test_이미지_좌상단_모서리는_0_0을_반환한다(self):
        """Given: 이미지와 동일 크기 위젯
        When:  위젯 (0, 0) 클릭
        Then:  이미지 좌표 (0, 0)
        """
        result = widget_to_image((0.0, 0.0), (100, 100), (100, 100))

        assert result == (0, 0)

    def test_이미지_경계_바깥_클릭은_None이다(self):
        """Given: 100x100 이미지와 동일 크기 위젯
        When:  (-1, 50) 클릭 (경계 밖)
        Then:  None 반환
        """
        result = widget_to_image((-1.0, 50.0), (100, 100), (100, 100))

        assert result is None

    def test_수직_레터박스에서_이미지_내부_클릭은_정확한_좌표를_반환한다(self):
        """Given: 100x50 이미지, 100x100 위젯 (상하 여백 각 25px)
        When:  위젯 (50, 50) 클릭 (이미지 정중앙)
        Then:  이미지 좌표 (50, 25) — 이미지 중앙
        """
        result = widget_to_image((50.0, 50.0), (100, 100), (100, 50))

        assert result == (50, 25)


# ---------------------------------------------------------------------------
# image_to_widget 테스트
# ---------------------------------------------------------------------------
class TestImageToWidget:
    """image_to_widget: 이미지 좌표 → 위젯 좌표 변환."""

    def test_스케일1_오프셋0에서_좌표는_그대로_반환된다(self):
        """Given: 위젯=이미지 크기
        Then:  위젯 좌표 == 이미지 좌표
        """
        wx, wy = image_to_widget((30.0, 40.0), (100, 100), (100, 100))

        assert wx == pytest.approx(30.0)
        assert wy == pytest.approx(40.0)

    def test_2배_스케일에서_이미지_중앙은_위젯_중앙이다(self):
        """Given: 50x50 이미지, 100x100 위젯(scale=2)
        When:  이미지 중앙 (25, 25)
        Then:  위젯 좌표 (50, 50)
        """
        wx, wy = image_to_widget((25.0, 25.0), (100, 100), (50, 50))

        assert wx == pytest.approx(50.0)
        assert wy == pytest.approx(50.0)

    def test_레터박스_위젯에서_이미지_원점은_오프셋으로_이동한다(self):
        """Given: 100x100 이미지, 200x100 위젯(offset_x=50)
        When:  이미지 (0, 0)
        Then:  위젯 좌표 (50, 0) — 레터박스 오프셋 포함
        """
        wx, wy = image_to_widget((0.0, 0.0), (200, 100), (100, 100))

        assert wx == pytest.approx(50.0)
        assert wy == pytest.approx(0.0)

    def test_widget_to_image와_image_to_widget은_역함수다(self):
        """Given: 임의 이미지 좌표 (30, 20)
        When:  image_to_widget → widget_to_image 왕복 변환
        Then:  원래 좌표로 복원 (반올림 오차 ±1 허용)
        """
        image_size = (640, 360)
        widget_size = (800, 450)
        original = (320, 180)

        wx, wy = image_to_widget(original, widget_size, image_size)
        recovered = widget_to_image((wx, wy), widget_size, image_size)

        assert recovered is not None
        assert abs(recovered[0] - original[0]) <= 1
        assert abs(recovered[1] - original[1]) <= 1
