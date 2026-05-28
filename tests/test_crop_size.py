"""크롭 크기 변환 순수 함수 테스트.

대상 모듈: easy_capture.ui.sizing
대상 함수: crop_ratio_to_size(ratio, frame_shape) -> (w, h)

검증 범위(계획서 §6-2):
  - 경계값: MIN_CROP_RATIO / MAX_CROP_RATIO / DEFAULT_CROP_RATIO → 픽셀 범위 내
  - 단조성: ratio 증가 → 반환 크기 단조 비감소
  - 짝수 보장: 반환 w·h 가 짝수
  - 하한 보장: 0 반환 방지 (최소 변 >= 최소 픽셀)
  - 프레임 가로/세로 다른 비율 시 최소변 기준 환산 일관성

PySide6 비의존 — numpy·순수 함수만 사용.

WHY: 슬라이더 값(비율)을 픽셀로 환산하는 로직을 순수 함수로 분리해
     UI 없이 경계값 전체를 단위 테스트로 빠르게 검증한다.
"""
from __future__ import annotations

import pytest

from easy_capture.ui.sizing import (
    DEFAULT_CROP_RATIO,
    MAX_CROP_RATIO,
    MIN_CROP_RATIO,
    crop_ratio_to_size,
)

# ---------------------------------------------------------------------------
# 테스트 상수
# ---------------------------------------------------------------------------
# 표준 테스트 프레임 크기 (W, H)
_FRAME_SQUARE = (640, 640)        # 정방형
_FRAME_LANDSCAPE = (640, 360)     # 가로형 (16:9)
_FRAME_PORTRAIT = (360, 640)      # 세로형 (9:16)

# 단조성 검증 비율 단계 (최소~최대 사이 고른 구간)
_MONOTONE_RATIOS = [10, 20, 40, 60, 80, 100]

# 반환 크기가 반드시 짝수임을 확인할 테스트용 비율 목록
_EVEN_CHECK_RATIOS = [MIN_CROP_RATIO, DEFAULT_CROP_RATIO, MAX_CROP_RATIO, 33, 67]

# 최소 크롭 픽셀 하한 (0 방지)
_MIN_PIXEL_SIZE = 2


# ---------------------------------------------------------------------------
# 상수 자체 유효성 검증
# ---------------------------------------------------------------------------
class TestSizingConstants:
    """MIN / MAX / DEFAULT 상수가 논리적 범위를 만족하는지 검증.

    WHY: 상수를 잘못 정의하면 슬라이더 전체가 무의미해진다.
         상수 범위 테스트를 분리해 구현자가 상수를 바꿀 때 즉시 알 수 있게 한다.
    """

    def test_MIN_CROP_RATIO는_양의_정수이다(self):
        """Given: MIN_CROP_RATIO 상수
        Then:  1 이상 정수 (슬라이더 0 위치에서도 최소 픽셀 보장)
        """
        assert isinstance(MIN_CROP_RATIO, int)
        assert MIN_CROP_RATIO >= 1

    def test_MAX_CROP_RATIO는_MIN보다_크다(self):
        """Given: MIN, MAX 상수
        Then:  MIN < MAX
        """
        assert MAX_CROP_RATIO > MIN_CROP_RATIO

    def test_DEFAULT_CROP_RATIO는_MIN과_MAX_사이에_있다(self):
        """Given: MIN, DEFAULT, MAX 상수
        Then:  MIN <= DEFAULT <= MAX
        """
        assert MIN_CROP_RATIO <= DEFAULT_CROP_RATIO <= MAX_CROP_RATIO


# ---------------------------------------------------------------------------
# crop_ratio_to_size 경계값 테스트
# ---------------------------------------------------------------------------
class TestCropRatioToSizeBoundary:
    """MIN / MAX / DEFAULT 비율에서 반환값 범위·짝수·하한을 검증."""

    def test_MIN_비율에서_반환값은_양의_짝수이다(self):
        """Given: ratio=MIN_CROP_RATIO, 표준 가로 프레임
        When:  crop_ratio_to_size 호출
        Then:  w > 0, h > 0, w % 2 == 0, h % 2 == 0
        """
        # --- Given / When ---
        w, h = crop_ratio_to_size(MIN_CROP_RATIO, _FRAME_LANDSCAPE)

        # --- Then ---
        assert w > 0, f"MIN 비율에서 w={w} (0 이하 금지)"
        assert h > 0, f"MIN 비율에서 h={h} (0 이하 금지)"
        assert w % 2 == 0, f"w={w}이 홀수"
        assert h % 2 == 0, f"h={h}이 홀수"

    def test_MAX_비율에서_반환값은_프레임_크기_이내이다(self):
        """Given: ratio=MAX_CROP_RATIO, 표준 가로 프레임
        When:  crop_ratio_to_size 호출
        Then:  w <= 프레임 너비, h <= 프레임 높이
        """
        # --- Given / When ---
        frame_w, frame_h = _FRAME_LANDSCAPE
        w, h = crop_ratio_to_size(MAX_CROP_RATIO, _FRAME_LANDSCAPE)

        # --- Then ---
        assert w <= frame_w, f"w={w}이 프레임 너비 {frame_w} 초과"
        assert h <= frame_h, f"h={h}이 프레임 높이 {frame_h} 초과"

    def test_DEFAULT_비율에서_반환값은_프레임_범위_내_짝수이다(self):
        """Given: ratio=DEFAULT_CROP_RATIO, 표준 가로 프레임
        When:  crop_ratio_to_size 호출
        Then:  0 < w <= frame_w, 0 < h <= frame_h, 짝수
        """
        # --- Given / When ---
        frame_w, frame_h = _FRAME_LANDSCAPE
        w, h = crop_ratio_to_size(DEFAULT_CROP_RATIO, _FRAME_LANDSCAPE)

        # --- Then ---
        assert 0 < w <= frame_w
        assert 0 < h <= frame_h
        assert w % 2 == 0
        assert h % 2 == 0

    def test_MIN_비율_반환_크기는_최소_하한_이상이다(self):
        """Given: ratio=MIN_CROP_RATIO
        When:  crop_ratio_to_size 호출
        Then:  w >= _MIN_PIXEL_SIZE, h >= _MIN_PIXEL_SIZE (0 방지 하한)
        """
        # --- Given / When ---
        w, h = crop_ratio_to_size(MIN_CROP_RATIO, _FRAME_LANDSCAPE)

        # --- Then ---
        assert w >= _MIN_PIXEL_SIZE, f"w={w}이 최소 하한 {_MIN_PIXEL_SIZE} 미만"
        assert h >= _MIN_PIXEL_SIZE, f"h={h}이 최소 하한 {_MIN_PIXEL_SIZE} 미만"


# ---------------------------------------------------------------------------
# crop_ratio_to_size 짝수 보장 테스트
# ---------------------------------------------------------------------------
class TestCropRatioToSizeEvenness:
    """다양한 비율에서 짝수 출력이 일관되게 보장되는지 검증."""

    @pytest.mark.parametrize("ratio", _EVEN_CHECK_RATIOS)
    def test_임의_비율에서_반환값은_항상_짝수이다(self, ratio):
        """Given: _EVEN_CHECK_RATIOS의 각 비율
        When:  crop_ratio_to_size 호출
        Then:  w % 2 == 0, h % 2 == 0

        WHY: yuv420p 인코딩은 짝수 해상도를 요구한다.
             모든 슬라이더 위치에서 짝수가 보장돼야 저장 시 오류가 없다.
        """
        # --- Given / When ---
        w, h = crop_ratio_to_size(ratio, _FRAME_LANDSCAPE)

        # --- Then ---
        assert w % 2 == 0, f"ratio={ratio}에서 w={w}이 홀수"
        assert h % 2 == 0, f"ratio={ratio}에서 h={h}이 홀수"

    @pytest.mark.parametrize("ratio", _EVEN_CHECK_RATIOS)
    def test_세로_프레임에서도_반환값은_항상_짝수이다(self, ratio):
        """Given: 세로형(9:16) 프레임, 다양한 비율
        When:  crop_ratio_to_size 호출
        Then:  w % 2 == 0, h % 2 == 0
        """
        # --- Given / When ---
        w, h = crop_ratio_to_size(ratio, _FRAME_PORTRAIT)

        # --- Then ---
        assert w % 2 == 0, f"ratio={ratio}에서 w={w}이 홀수 (세로 프레임)"
        assert h % 2 == 0, f"ratio={ratio}에서 h={h}이 홀수 (세로 프레임)"


# ---------------------------------------------------------------------------
# crop_ratio_to_size 단조성 테스트
# ---------------------------------------------------------------------------
class TestCropRatioToSizeMonotonicity:
    """비율 증가 시 반환 크기가 단조 비감소인지 검증.

    WHY: 슬라이더를 오른쪽으로 움직이면 박스가 반드시 커지거나 같아야 한다.
         작아지면 사용자 기대를 위배해 UX 버그가 된다.
    """

    def test_비율_증가_시_너비가_단조_비감소이다(self):
        """Given: _MONOTONE_RATIOS 오름차순 비율, 표준 가로 프레임
        When:  각 비율로 crop_ratio_to_size 호출
        Then:  w[i] <= w[i+1] (단조 비감소)
        """
        # --- Given / When ---
        widths = [crop_ratio_to_size(r, _FRAME_LANDSCAPE)[0] for r in _MONOTONE_RATIOS]

        # --- Then ---
        for i in range(len(widths) - 1):
            assert widths[i] <= widths[i + 1], (
                f"단조성 위반: ratio={_MONOTONE_RATIOS[i]}→w={widths[i]}, "
                f"ratio={_MONOTONE_RATIOS[i + 1]}→w={widths[i + 1]}"
            )

    def test_비율_증가_시_높이가_단조_비감소이다(self):
        """Given: _MONOTONE_RATIOS 오름차순 비율, 표준 가로 프레임
        When:  각 비율로 crop_ratio_to_size 호출
        Then:  h[i] <= h[i+1] (단조 비감소)
        """
        # --- Given / When ---
        heights = [crop_ratio_to_size(r, _FRAME_LANDSCAPE)[1] for r in _MONOTONE_RATIOS]

        # --- Then ---
        for i in range(len(heights) - 1):
            assert heights[i] <= heights[i + 1], (
                f"단조성 위반: ratio={_MONOTONE_RATIOS[i]}→h={heights[i]}, "
                f"ratio={_MONOTONE_RATIOS[i + 1]}→h={heights[i + 1]}"
            )

    def test_MIN과_MAX_비율의_크기_차이가_존재한다(self):
        """Given: MIN, MAX 비율
        When:  각각 crop_ratio_to_size 호출
        Then:  MAX 결과 > MIN 결과 (슬라이더 전 범위에서 크기 변화 존재)

        WHY: MIN==MAX이면 슬라이더가 아무 효과가 없다(UX 버그).
        """
        # --- Given / When ---
        w_min, h_min = crop_ratio_to_size(MIN_CROP_RATIO, _FRAME_LANDSCAPE)
        w_max, h_max = crop_ratio_to_size(MAX_CROP_RATIO, _FRAME_LANDSCAPE)

        # --- Then ---
        assert w_max > w_min or h_max > h_min, (
            f"MIN({w_min}x{h_min})과 MAX({w_max}x{h_max})가 동일하거나 역전됨"
        )


# ---------------------------------------------------------------------------
# crop_ratio_to_size 프레임 비율 의존성 테스트
# ---------------------------------------------------------------------------
class TestCropRatioToSizeFrameShape:
    """프레임 가로/세로 비율이 다를 때 최소변 기준 환산이 일관적인지 검증.

    WHY: 계획서 §3-1 — "슬라이더 값(예: 10~100%)을 프레임 최소변 기준 픽셀로 환산".
         가로형 프레임의 최소변은 높이, 세로형의 최소변은 너비이다.
         두 경우 모두 동일한 비율에서 동일한 '최소변 기준 픽셀'이 나와야
         프레임 방향에 무관하게 일관된 박스 비율을 제공한다.
    """

    def test_정방형_프레임에서_DEFAULT_비율_크기는_정방형에_가깝다(self):
        """Given: 정방형 프레임(640x640), DEFAULT 비율
        When:  crop_ratio_to_size 호출
        Then:  w == h (정방형 프레임에서 최소변 기준이면 w=h)
        """
        # --- Given / When ---
        w, h = crop_ratio_to_size(DEFAULT_CROP_RATIO, _FRAME_SQUARE)

        # --- Then ---
        assert w == h, f"정방형 프레임에서 w({w}) != h({h})"

    def test_가로_프레임_MIN_비율에서_크기는_프레임_범위_내이다(self):
        """Given: 가로형 프레임(640x360), MIN 비율
        When:  crop_ratio_to_size 호출
        Then:  0 < w <= 640, 0 < h <= 360
        """
        # --- Given / When ---
        w, h = crop_ratio_to_size(MIN_CROP_RATIO, _FRAME_LANDSCAPE)

        # --- Then ---
        assert 0 < w <= 640
        assert 0 < h <= 360

    def test_세로_프레임_MIN_비율에서_크기는_프레임_범위_내이다(self):
        """Given: 세로형 프레임(360x640), MIN 비율
        When:  crop_ratio_to_size 호출
        Then:  0 < w <= 360, 0 < h <= 640
        """
        # --- Given / When ---
        w, h = crop_ratio_to_size(MIN_CROP_RATIO, _FRAME_PORTRAIT)

        # --- Then ---
        assert 0 < w <= 360
        assert 0 < h <= 640

    def test_동일_비율에서_정방형_반환_크기는_가로_프레임보다_작거나_같다(self):
        """Given: 정방형(640x640), 가로형(640x360), 동일 비율 DEFAULT
        When:  각각 crop_ratio_to_size 호출
        Then:  정방형 결과 >= 가로형 결과

        WHY: 최소변이 클수록 동일 비율에서 더 큰 박스를 만든다.
             640x640의 최소변(640) > 640x360의 최소변(360)이므로
             정방형 결과가 더 크거나 같아야 한다.
        """
        # --- Given / When ---
        w_sq, h_sq = crop_ratio_to_size(DEFAULT_CROP_RATIO, _FRAME_SQUARE)
        w_ls, h_ls = crop_ratio_to_size(DEFAULT_CROP_RATIO, _FRAME_LANDSCAPE)

        # --- Then ---
        assert w_sq >= w_ls, (
            f"정방형 w={w_sq}이 가로형 w={w_ls}보다 작음 (최소변 기준 환산 위반)"
        )
        assert h_sq >= h_ls

    def test_반환값은_항상_튜플이다(self):
        """Given: 임의 비율과 프레임
        When:  crop_ratio_to_size 호출
        Then:  길이 2 튜플, 각 요소 정수
        """
        # --- Given / When ---
        result = crop_ratio_to_size(DEFAULT_CROP_RATIO, _FRAME_LANDSCAPE)

        # --- Then ---
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)
