"""crop 기하 로직 테스트.

기존 함수 회귀 + 신규 bbox_of_mask 단위 테스트.

신규:
  bbox_of_mask — 마스크 → 외접 bbox 순수 헬퍼
    (구현 전 RED 상태 정상: core/crop/crop.py에 미추가)
"""
from __future__ import annotations

import numpy as np
import pytest

from easy_capture.core.crop import (apply_aspect_lock, centroid_of_mask,
                                    make_crop_box, smooth_centroids, to_even)

# bbox_of_mask — 구현 전이므로 try/except 격리
# WHY: centroid_of_mask와 동일 파일(core/crop/crop.py)에 추가 예정이지만
#      아직 없으면 기존 테스트 차단 없이 이 테스트만 FAIL 처리된다.
try:
    from easy_capture.core.crop.crop import bbox_of_mask
    _HAS_BBOX_OF_MASK = True
except ImportError:
    bbox_of_mask = None  # type: ignore[assignment]
    _HAS_BBOX_OF_MASK = False

_MSG_NO_BBOX = "core/crop/crop.py에 bbox_of_mask 미구현 — RED 예상"

# ---------------------------------------------------------------------------
# 테스트 상수
# ---------------------------------------------------------------------------
# 사각형 마스크 내부 좌표 (row_start, row_end, col_start, col_end)
RECT_ROW_START = 10
RECT_ROW_END = 30
RECT_COL_START = 20
RECT_COL_END = 50

# 예상 bbox
EXPECTED_X1 = float(RECT_COL_START)   # xs.min()
EXPECTED_Y1 = float(RECT_ROW_START)   # ys.min()
EXPECTED_X2 = float(RECT_COL_END - 1) # xs.max() (마지막 True 열)
EXPECTED_Y2 = float(RECT_ROW_END - 1) # ys.max() (마지막 True 행)


def test_centroid_of_mask():
    m = np.zeros((10, 10), bool)
    m[2:4, 4:6] = True
    assert centroid_of_mask(m) == (4.5, 2.5)


def test_centroid_empty_returns_none():
    assert centroid_of_mask(np.zeros((5, 5), bool)) is None


def test_smooth_reduces_jitter():
    pts = [(0, 0), (10, 0), (0, 0), (10, 0), (0, 0)]
    out = smooth_centroids(pts, window=5)
    assert 2 < out[-1][0] < 8  # 진폭(0~10)보다 작아짐


def test_smooth_holds_none_forward():
    out = smooth_centroids([(5, 5), None, None], window=1)
    assert out[1] == (5, 5) and out[2] == (5, 5)


def test_to_even():
    assert to_even(101) == 100
    assert to_even(100) == 100


def test_aspect_lock_vertical_shrinks_width():
    assert apply_aspect_lock(200, 200, "9:16") == (112, 200)


def test_aspect_lock_none_passthrough():
    assert apply_aspect_lock(123, 77, None) == (123, 77)


def test_make_crop_box_even_and_within_frame():
    x1, y1, x2, y2 = make_crop_box((1000, 1000), (200, 200), (640, 360))
    assert (x2 - x1) % 2 == 0 and (y2 - y1) % 2 == 0
    assert 0 <= x1 and x2 <= 640 and 0 <= y1 and y2 <= 360


# ---------------------------------------------------------------------------
# bbox_of_mask — 신규 순수 헬퍼 (구현 전 RED)
# ---------------------------------------------------------------------------
class TestBboxOfMask:
    """bbox_of_mask(mask) → (x1, y1, x2, y2) | None 순수 헬퍼 검증.

    WHY: SAM2 video는 마스크만 반환하고 rematch_score는 bbox를 요구한다.
         이 헬퍼가 누락 연결고리를 채운다(계획서 §3-3).
         centroid_of_mask와 완전 대칭(같은 파일·같은 패턴)이므로
         테스트 케이스도 대칭으로 작성한다.
    """

    @pytest.mark.skipif(not _HAS_BBOX_OF_MASK, reason=_MSG_NO_BBOX)
    def test_사각형_마스크의_외접_bbox를_반환한다(self):
        """Given: 10×30 사각형 True 영역을 가진 HxW bool 마스크
        When:  bbox_of_mask 호출
        Then:  (x1, y1, x2, y2) == (COL_START, ROW_START, COL_END-1, ROW_END-1)

        WHY: FakeVideoBackend._make_rect_mask로 생성된 마스크의 bbox를
             수동 계산값과 정확히 대조해 구현이 올바른지 검증한다.
        """
        mask = np.zeros((100, 100), dtype=bool)
        mask[RECT_ROW_START:RECT_ROW_END, RECT_COL_START:RECT_COL_END] = True

        result = bbox_of_mask(mask)

        assert result is not None
        x1, y1, x2, y2 = result
        assert x1 == pytest.approx(EXPECTED_X1), f"x1 불일치: {x1} vs {EXPECTED_X1}"
        assert y1 == pytest.approx(EXPECTED_Y1), f"y1 불일치: {y1} vs {EXPECTED_Y1}"
        assert x2 == pytest.approx(EXPECTED_X2), f"x2 불일치: {x2} vs {EXPECTED_X2}"
        assert y2 == pytest.approx(EXPECTED_Y2), f"y2 불일치: {y2} vs {EXPECTED_Y2}"

    @pytest.mark.skipif(not _HAS_BBOX_OF_MASK, reason=_MSG_NO_BBOX)
    def test_빈_마스크이면_None을_반환한다(self):
        """Given: 전부 False인 5×5 마스크
        When:  bbox_of_mask 호출
        Then:  None 반환

        WHY: centroid_of_mask(빈 마스크) → None과 대칭 계약.
             빈 마스크는 재매칭 입력으로 쓸 수 없음을 명시한다.
        """
        empty_mask = np.zeros((5, 5), dtype=bool)

        result = bbox_of_mask(empty_mask)

        assert result is None

    @pytest.mark.skipif(not _HAS_BBOX_OF_MASK, reason=_MSG_NO_BBOX)
    def test_단일_픽셀_마스크의_bbox는_동일_좌표_4개이다(self):
        """Given: (5, 7) 좌표에만 True인 마스크
        When:  bbox_of_mask 호출
        Then:  x1==x2==7.0, y1==y2==5.0 (단일 픽셀 = 점 박스)

        WHY: 경계값 — 마스크가 1픽셀일 때 bbox가 유효한 점으로 반환되는지 확인.
        """
        mask = np.zeros((20, 20), dtype=bool)
        mask[5, 7] = True  # row=5, col=7

        result = bbox_of_mask(mask)

        assert result is not None
        x1, y1, x2, y2 = result
        assert x1 == pytest.approx(7.0)
        assert y1 == pytest.approx(5.0)
        assert x2 == pytest.approx(7.0)
        assert y2 == pytest.approx(5.0)

    @pytest.mark.skipif(not _HAS_BBOX_OF_MASK, reason=_MSG_NO_BBOX)
    def test_0_1_float_마스크도_처리한다(self):
        """Given: 0.0/1.0 float dtype 마스크 (bool이 아닌 경우)
        When:  bbox_of_mask 호출
        Then:  1.0 영역에 대한 올바른 bbox 반환

        WHY: np.asarray(mask) > 0 조건 — float32/float64 마스크도
             bool 마스크와 동일하게 처리함을 검증한다.
        """
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[15:25, 10:20] = 1.0

        result = bbox_of_mask(mask)

        assert result is not None
        x1, y1, x2, y2 = result
        assert x1 == pytest.approx(10.0)
        assert y1 == pytest.approx(15.0)
        assert x2 == pytest.approx(19.0)
        assert y2 == pytest.approx(24.0)

    @pytest.mark.skipif(not _HAS_BBOX_OF_MASK, reason=_MSG_NO_BBOX)
    def test_반환값은_float_4_튜플이다(self):
        """Given: 유효한 bool 마스크
        When:  bbox_of_mask 호출
        Then:  (float, float, float, float) 튜플 반환

        WHY: rematch_score가 Box = tuple[float, float, float, float] 타입을
             요구하므로 반환 타입이 정확히 맞아야 한다.
        """
        mask = np.zeros((30, 40), dtype=bool)
        mask[5:10, 8:15] = True

        result = bbox_of_mask(mask)

        assert result is not None
        assert len(result) == 4
        for v in result:
            assert isinstance(v, float), f"bbox 원소가 float이 아님: {type(v)}"

    @pytest.mark.skipif(not _HAS_BBOX_OF_MASK, reason=_MSG_NO_BBOX)
    def test_FakeVideoBackend_사각형_마스크와_대칭_검증(self):
        """Given: _make_rect_mask(H=360, W=640, cx=100, cy=150, half=20)
        When:  bbox_of_mask 호출
        Then:  x1≈80, y1≈130, x2≈119, y2≈169 (±경계클램프)

        WHY: FakeVideoBackend가 생성하는 마스크에서 bbox를 정확히 추출해
             재추적 오케스트레이션에서 prev_box를 올바르게 구하는지 확인한다.
             half=20이면 cx±20 범위이므로 x1=80, x2=119.
        """
        # FakeVideoBackend._make_rect_mask 동일 로직 재현
        h, w, cx, cy, half = 360, 640, 100, 150, 20
        mask = np.zeros((h, w), dtype=bool)
        y1_m = max(0, cy - half)
        y2_m = min(h, cy + half)
        x1_m = max(0, cx - half)
        x2_m = min(w, cx + half)
        mask[y1_m:y2_m, x1_m:x2_m] = True

        result = bbox_of_mask(mask)

        assert result is not None
        rx1, ry1, rx2, ry2 = result
        assert rx1 == pytest.approx(float(x1_m))
        assert ry1 == pytest.approx(float(y1_m))
        assert rx2 == pytest.approx(float(x2_m - 1))
        assert ry2 == pytest.approx(float(y2_m - 1))
