"""core/export/image_export 순수 로직 테스트.

대상 모듈: easy_capture.core.export.image_export
의존 라이브러리: Pillow(저장/재로드), numpy(배열 검증)
외부 의존 없음 — PyAV·SAM2·PySide6 비의존.

NOTE: 구현이 아직 없으므로 import 실패로 Red 상태. TDD 정상.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from easy_capture.core.export.image_export import ExportConfig, crop_array, save_image

# ---------------------------------------------------------------------------
# 테스트 픽스처 상수
# ---------------------------------------------------------------------------
# 고정 테스트 프레임 크기
FRAME_H = 120
FRAME_W = 160
FRAME_C = 3

# 크롭 박스 좌표 (x1, y1, x2, y2)
BOX_X1, BOX_Y1, BOX_X2, BOX_Y2 = 10, 20, 50, 80
EXPECTED_CROP_W = BOX_X2 - BOX_X1  # 40
EXPECTED_CROP_H = BOX_Y2 - BOX_Y1  # 60


@pytest.fixture()
def rgb_frame() -> np.ndarray:
    """테스트용 결정적 RGB HxWx3 uint8 배열.

    R채널=x좌표, G채널=y좌표 기반 → 특정 픽셀값 예측 가능.
    """
    frame = np.zeros((FRAME_H, FRAME_W, FRAME_C), dtype=np.uint8)
    frame[:, :, 0] = (np.arange(FRAME_W) % 256).astype(np.uint8)
    frame[:, :, 1] = (np.arange(FRAME_H) % 256).astype(np.uint8)[:, np.newaxis]
    frame[:, :, 2] = 64
    return frame


# ---------------------------------------------------------------------------
# crop_array 테스트
# ---------------------------------------------------------------------------
class TestCropArray:
    """crop_array: RGB 배열을 (x1,y1,x2,y2) 박스로 정확히 슬라이스."""

    def test_크롭_결과_shape가_박스_크기와_일치한다(self, rgb_frame):
        """Given: 160x120 프레임, (10,20,50,80) 박스
        When:  crop_array 호출
        Then:  shape == (60, 40, 3)
        """
        result = crop_array(rgb_frame, (BOX_X1, BOX_Y1, BOX_X2, BOX_Y2))

        assert result.shape == (EXPECTED_CROP_H, EXPECTED_CROP_W, FRAME_C)

    def test_크롭_결과_픽셀값이_원본_슬라이스와_일치한다(self, rgb_frame):
        """Given: 결정적 그라디언트 프레임
        When:  crop_array 호출
        Then:  픽셀값이 numpy 직접 슬라이스와 동일
        """
        result = crop_array(rgb_frame, (BOX_X1, BOX_Y1, BOX_X2, BOX_Y2))
        expected = rgb_frame[BOX_Y1:BOX_Y2, BOX_X1:BOX_X2]

        np.testing.assert_array_equal(result, expected)

    def test_전체_프레임_박스_크롭은_원본과_동일하다(self, rgb_frame):
        """Given: 프레임 전체를 감싸는 박스
        When:  crop_array 호출
        Then:  결과가 원본 배열과 동일
        """
        full_box = (0, 0, FRAME_W, FRAME_H)
        result = crop_array(rgb_frame, full_box)

        np.testing.assert_array_equal(result, rgb_frame)

    def test_단일_픽셀_박스_크롭_shape는_1x1x3이다(self, rgb_frame):
        """Given: 1x1 픽셀 박스
        When:  crop_array 호출
        Then:  shape == (1, 1, 3)
        """
        single_box = (5, 5, 6, 6)
        result = crop_array(rgb_frame, single_box)

        assert result.shape == (1, 1, FRAME_C)

    def test_크롭_결과_dtype은_uint8이다(self, rgb_frame):
        """Given: uint8 프레임
        When:  crop_array 호출
        Then:  결과 dtype도 uint8
        """
        result = crop_array(rgb_frame, (BOX_X1, BOX_Y1, BOX_X2, BOX_Y2))

        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# save_image 테스트
# ---------------------------------------------------------------------------
class TestSaveImage:
    """save_image: RGB 배열을 PNG/JPG로 저장하고 Pillow로 재로드해 검증."""

    def test_PNG_저장_후_재로드_시_크기가_일치한다(self, rgb_frame, tmp_path):
        """Given: RGB 프레임, PNG 포맷 설정
        When:  save_image로 tmp_path에 저장
        Then:  Pillow 재로드 시 width·height 일치
        """
        output_path = str(tmp_path / "test.png")
        config = ExportConfig(fmt="png")

        save_image(rgb_frame, output_path, config)

        img = Image.open(output_path)
        assert img.size == (FRAME_W, FRAME_H)

    def test_PNG_저장_후_재로드_시_모드가_RGB이다(self, rgb_frame, tmp_path):
        """Given: RGB 프레임, PNG 포맷 설정
        When:  save_image로 저장
        Then:  Pillow 재로드 시 모드 == 'RGB'
        """
        output_path = str(tmp_path / "test.png")
        config = ExportConfig(fmt="png")

        save_image(rgb_frame, output_path, config)

        img = Image.open(output_path)
        assert img.mode == "RGB"

    def test_JPG_저장_후_재로드_시_크기가_일치한다(self, rgb_frame, tmp_path):
        """Given: RGB 프레임, JPG 포맷 설정
        When:  save_image로 tmp_path에 저장
        Then:  Pillow 재로드 시 width·height 일치
        """
        output_path = str(tmp_path / "test.jpg")
        config = ExportConfig(fmt="jpg", quality=90)

        save_image(rgb_frame, output_path, config)

        img = Image.open(output_path)
        assert img.size == (FRAME_W, FRAME_H)

    def test_JPG_저장_후_재로드_시_모드가_RGB이다(self, rgb_frame, tmp_path):
        """Given: RGB 프레임, JPG 포맷 설정
        When:  save_image로 저장
        Then:  Pillow 재로드 시 모드 == 'RGB'
        """
        output_path = str(tmp_path / "test.jpg")
        config = ExportConfig(fmt="jpg", quality=85)

        save_image(rgb_frame, output_path, config)

        img = Image.open(output_path)
        assert img.mode == "RGB"

    def test_저장_후_파일이_실제로_존재한다(self, rgb_frame, tmp_path):
        """Given: PNG 설정
        When:  save_image 호출
        Then:  파일이 디스크에 생성됨
        """
        output_path = tmp_path / "output.png"
        config = ExportConfig(fmt="png")

        save_image(rgb_frame, str(output_path), config)

        assert output_path.exists()

    def test_잘못된_fmt_지정_시_예외를_던진다(self, rgb_frame, tmp_path):
        """Given: 지원하지 않는 포맷 문자열 'bmp'
        When:  save_image 호출
        Then:  ValueError 또는 지정된 예외 발생
        """
        output_path = str(tmp_path / "test.bmp")
        config = ExportConfig(fmt="bmp")

        with pytest.raises((ValueError, Exception)):
            save_image(rgb_frame, output_path, config)

    def test_빈_문자열_fmt_지정_시_예외를_던진다(self, rgb_frame, tmp_path):
        """Given: 빈 포맷 문자열
        When:  save_image 호출
        Then:  예외 발생
        """
        output_path = str(tmp_path / "test_empty.png")
        config = ExportConfig(fmt="")

        with pytest.raises((ValueError, Exception)):
            save_image(rgb_frame, output_path, config)

    def test_ExportConfig_기본값은_png_품질95이다(self):
        """Given: ExportConfig를 기본값으로 생성
        When:  fmt·quality 속성 확인
        Then:  fmt='png', quality=95
        """
        config = ExportConfig()

        assert config.fmt == "png"
        assert config.quality == 95
