"""오버레이 벡터화 순수 함수 테스트.

대상 모듈: easy_capture.ui.frame_canvas
대상 함수: mask_to_rgba(mask, color, alpha) -> (H, W, 4) uint8

검증 범위(계획서 §6-3):
  - shape: (H, W, 4), dtype: uint8
  - True 픽셀: RGB == 지정 색, A == 지정 alpha (불투명)
  - False 픽셀: A == 0 (투명)
  - 빈 마스크(all False): 전체 알파 0
  - 단일 픽셀 True: 정확히 1픽셀만 불투명
  - 기본값 동작: 색·알파 기본값으로 호출 시 유효한 RGBA 반환

PySide6 import 없음 — numpy 순수 함수만 테스트한다.
QImage 합성 자체는 테스트하지 않는다(렌더는 수동 스모크).

WHY: 이중 for-loop 제거 후 numpy RGBA 생성이 올바른지 검증하기 위해
     순수 함수를 분리 추출하고 여기서 단위 테스트한다.
"""
from __future__ import annotations

import numpy as np
import pytest

# 계획서 §4-2: _mask_to_rgba → 공개 모듈 함수 mask_to_rgba로 추출
# 함수명: 계획서 §4-2에서 `_mask_to_rgba`로 명시됐으나,
#         공개 함수로 분리하면 모듈 수준 함수 `mask_to_rgba`로 노출한다.
#         (언더스코어 없이 공개 — 테스트 전용 비공개 접근 회피)
from easy_capture.ui.frame_canvas import mask_to_rgba

# ---------------------------------------------------------------------------
# 테스트 상수
# ---------------------------------------------------------------------------
# 기본 오버레이 색 (R, G, B) — 계획서 §4-2 명시값
_DEFAULT_COLOR = (0, 120, 255)
# 기본 알파값 — 계획서 §4-2 명시값
_DEFAULT_ALPHA = 110

# 테스트 마스크 크기
_SMALL_H = 4
_SMALL_W = 6
_SINGLE_H = 1
_SINGLE_W = 1

# 투명 픽셀 알파
_ALPHA_TRANSPARENT = 0


# ---------------------------------------------------------------------------
# shape · dtype 테스트
# ---------------------------------------------------------------------------
class TestMaskToRgbaShape:
    """반환 배열의 shape·dtype을 검증."""

    def test_HxW_마스크_입력_시_HxWx4_배열을_반환한다(self):
        """Given: bool (4, 6) 마스크
        When:  mask_to_rgba 호출
        Then:  shape == (4, 6, 4)
        """
        # --- Given ---
        mask = np.zeros((_SMALL_H, _SMALL_W), dtype=bool)

        # --- When ---
        result = mask_to_rgba(mask)

        # --- Then ---
        assert result.shape == (_SMALL_H, _SMALL_W, 4), (
            f"shape 불일치: {result.shape} vs 기대 ({_SMALL_H},{_SMALL_W},4)"
        )

    def test_반환_dtype은_uint8이다(self):
        """Given: bool 마스크
        When:  mask_to_rgba 호출
        Then:  dtype == uint8

        WHY: QImage(Format_RGBA8888)는 uint8 버퍼를 요구한다.
             dtype이 다르면 QImage 생성 시 메모리 오류가 발생한다.
        """
        # --- Given ---
        mask = np.zeros((_SMALL_H, _SMALL_W), dtype=bool)

        # --- When ---
        result = mask_to_rgba(mask)

        # --- Then ---
        assert result.dtype == np.uint8, f"dtype 불일치: {result.dtype}"

    def test_1x1_마스크도_1x1x4_배열을_반환한다(self):
        """Given: bool (1, 1) 단일 픽셀 마스크
        When:  mask_to_rgba 호출
        Then:  shape == (1, 1, 4)
        """
        # --- Given ---
        mask = np.zeros((_SINGLE_H, _SINGLE_W), dtype=bool)

        # --- When ---
        result = mask_to_rgba(mask)

        # --- Then ---
        assert result.shape == (_SINGLE_H, _SINGLE_W, 4)

    def test_메모리_연속성이_보장된다(self):
        """Given: bool 마스크
        When:  mask_to_rgba 호출
        Then:  반환 배열이 C-contiguous (QImage 버퍼 안정성)

        WHY: QImage는 데이터 버퍼를 참조하므로 C-order 연속 배열이어야
             stride 불일치로 인한 화면 깨짐을 방지할 수 있다.
        """
        # --- Given ---
        mask = np.ones((_SMALL_H, _SMALL_W), dtype=bool)

        # --- When ---
        result = mask_to_rgba(mask)

        # --- Then ---
        assert result.flags["C_CONTIGUOUS"], "반환 배열이 C-contiguous가 아님"


# ---------------------------------------------------------------------------
# True 픽셀 색·알파 검증
# ---------------------------------------------------------------------------
class TestMaskToRgbaTruePixels:
    """True 픽셀이 지정된 색·알파값을 가지는지 검증."""

    def test_True_픽셀의_RGB는_기본_색상과_같다(self):
        """Given: (0, 0) 위치만 True인 마스크, 기본 색
        When:  mask_to_rgba 호출
        Then:  result[0, 0, :3] == _DEFAULT_COLOR
        """
        # --- Given ---
        mask = np.zeros((_SMALL_H, _SMALL_W), dtype=bool)
        mask[0, 0] = True

        # --- When ---
        result = mask_to_rgba(mask, color=_DEFAULT_COLOR, alpha=_DEFAULT_ALPHA)

        # --- Then ---
        r, g, b = result[0, 0, 0], result[0, 0, 1], result[0, 0, 2]
        assert r == _DEFAULT_COLOR[0], f"R 불일치: {r}"
        assert g == _DEFAULT_COLOR[1], f"G 불일치: {g}"
        assert b == _DEFAULT_COLOR[2], f"B 불일치: {b}"

    def test_True_픽셀의_알파는_지정값과_같다(self):
        """Given: (0, 0) 위치만 True인 마스크, alpha=110
        When:  mask_to_rgba 호출
        Then:  result[0, 0, 3] == 110
        """
        # --- Given ---
        mask = np.zeros((_SMALL_H, _SMALL_W), dtype=bool)
        mask[0, 0] = True

        # --- When ---
        result = mask_to_rgba(mask, color=_DEFAULT_COLOR, alpha=_DEFAULT_ALPHA)

        # --- Then ---
        assert result[0, 0, 3] == _DEFAULT_ALPHA, (
            f"알파 불일치: {result[0, 0, 3]} vs {_DEFAULT_ALPHA}"
        )

    def test_전체_True_마스크의_모든_픽셀_RGB는_지정_색이다(self):
        """Given: all-True 마스크
        When:  mask_to_rgba 호출
        Then:  모든 픽셀 RGB == _DEFAULT_COLOR
        """
        # --- Given ---
        mask = np.ones((_SMALL_H, _SMALL_W), dtype=bool)

        # --- When ---
        result = mask_to_rgba(mask, color=_DEFAULT_COLOR, alpha=_DEFAULT_ALPHA)

        # --- Then ---
        np.testing.assert_array_equal(
            result[:, :, :3],
            np.full((_SMALL_H, _SMALL_W, 3), _DEFAULT_COLOR, dtype=np.uint8),
            err_msg="all-True 마스크에서 RGB 불일치",
        )

    def test_전체_True_마스크의_모든_픽셀_알파는_지정값이다(self):
        """Given: all-True 마스크
        When:  mask_to_rgba 호출
        Then:  모든 픽셀 A == _DEFAULT_ALPHA
        """
        # --- Given ---
        mask = np.ones((_SMALL_H, _SMALL_W), dtype=bool)

        # --- When ---
        result = mask_to_rgba(mask, color=_DEFAULT_COLOR, alpha=_DEFAULT_ALPHA)

        # --- Then ---
        np.testing.assert_array_equal(
            result[:, :, 3],
            np.full((_SMALL_H, _SMALL_W), _DEFAULT_ALPHA, dtype=np.uint8),
            err_msg="all-True 마스크에서 알파 불일치",
        )

    def test_커스텀_색상으로_호출_시_해당_색이_적용된다(self):
        """Given: True 픽셀, 커스텀 색 (255, 0, 128)
        When:  mask_to_rgba(color=(255, 0, 128)) 호출
        Then:  True 픽셀 RGB == (255, 0, 128)
        """
        # --- Given ---
        CUSTOM_COLOR = (255, 0, 128)
        mask = np.zeros((_SMALL_H, _SMALL_W), dtype=bool)
        mask[1, 2] = True

        # --- When ---
        result = mask_to_rgba(mask, color=CUSTOM_COLOR, alpha=200)

        # --- Then ---
        assert tuple(result[1, 2, :3]) == CUSTOM_COLOR


# ---------------------------------------------------------------------------
# False 픽셀 투명 검증
# ---------------------------------------------------------------------------
class TestMaskToRgbaFalsePixels:
    """False 픽셀이 완전 투명(알파=0)인지 검증."""

    def test_False_픽셀의_알파는_0이다(self):
        """Given: (0, 0)만 True, 나머지 False인 마스크
        When:  mask_to_rgba 호출
        Then:  (0, 1) 픽셀 알파 == 0 (투명)
        """
        # --- Given ---
        mask = np.zeros((_SMALL_H, _SMALL_W), dtype=bool)
        mask[0, 0] = True

        # --- When ---
        result = mask_to_rgba(mask, color=_DEFAULT_COLOR, alpha=_DEFAULT_ALPHA)

        # --- Then ---
        assert result[0, 1, 3] == _ALPHA_TRANSPARENT, (
            f"False 픽셀 알파가 0이 아님: {result[0, 1, 3]}"
        )

    def test_전체_False_마스크는_모든_픽셀_알파가_0이다(self):
        """Given: all-False 마스크 (빈 마스크)
        When:  mask_to_rgba 호출
        Then:  모든 픽셀 A == 0

        WHY: 세그 실패 후 오버레이가 완전 투명해야 프레임을 가리지 않는다.
        """
        # --- Given ---
        mask = np.zeros((_SMALL_H, _SMALL_W), dtype=bool)

        # --- When ---
        result = mask_to_rgba(mask, color=_DEFAULT_COLOR, alpha=_DEFAULT_ALPHA)

        # --- Then ---
        np.testing.assert_array_equal(
            result[:, :, 3],
            np.zeros((_SMALL_H, _SMALL_W), dtype=np.uint8),
            err_msg="빈 마스크에서 알파가 0이 아닌 픽셀 존재",
        )

    def test_전체_False_마스크는_RGB도_0이다(self):
        """Given: all-False 마스크
        When:  mask_to_rgba 호출
        Then:  모든 픽셀 RGB == (0, 0, 0)

        WHY: zeros 초기화 배열에서 True 위치만 색을 입히므로
             False는 반드시 (0,0,0,0)이어야 한다.
        """
        # --- Given ---
        mask = np.zeros((_SMALL_H, _SMALL_W), dtype=bool)

        # --- When ---
        result = mask_to_rgba(mask, color=_DEFAULT_COLOR, alpha=_DEFAULT_ALPHA)

        # --- Then ---
        np.testing.assert_array_equal(
            result[:, :, :3],
            np.zeros((_SMALL_H, _SMALL_W, 3), dtype=np.uint8),
            err_msg="빈 마스크에서 RGB가 0이 아닌 픽셀 존재",
        )


# ---------------------------------------------------------------------------
# 단일 픽셀 마스크 검증
# ---------------------------------------------------------------------------
class TestMaskToRgbaSinglePixel:
    """단일 픽셀 True 마스크에서 정확히 1픽셀만 불투명인지 검증."""

    def test_단일_픽셀_True_시_불투명_픽셀이_정확히_1개이다(self):
        """Given: (2, 3) 위치만 True인 마스크
        When:  mask_to_rgba 호출
        Then:  알파 > 0 인 픽셀 수 == 1
        """
        # --- Given ---
        mask = np.zeros((_SMALL_H, _SMALL_W), dtype=bool)
        mask[2, 3] = True

        # --- When ---
        result = mask_to_rgba(mask, color=_DEFAULT_COLOR, alpha=_DEFAULT_ALPHA)

        # --- Then ---
        opaque_count = int((result[:, :, 3] > 0).sum())
        assert opaque_count == 1, (
            f"불투명 픽셀 수 불일치: {opaque_count} (기대: 1)"
        )

    def test_단일_픽셀_True의_위치가_정확하다(self):
        """Given: (2, 3) 위치만 True
        When:  mask_to_rgba 호출
        Then:  result[2, 3, 3] == _DEFAULT_ALPHA, 나머지 위치 알파 == 0
        """
        # --- Given ---
        TRUE_Y, TRUE_X = 2, 3
        mask = np.zeros((_SMALL_H, _SMALL_W), dtype=bool)
        mask[TRUE_Y, TRUE_X] = True

        # --- When ---
        result = mask_to_rgba(mask, color=_DEFAULT_COLOR, alpha=_DEFAULT_ALPHA)

        # --- Then ---
        assert result[TRUE_Y, TRUE_X, 3] == _DEFAULT_ALPHA, (
            f"True 위치 알파 불일치: {result[TRUE_Y, TRUE_X, 3]}"
        )
        # 나머지 위치는 투명
        alpha_channel = result[:, :, 3].copy()
        alpha_channel[TRUE_Y, TRUE_X] = 0  # True 위치 제거 후 모두 0이어야 함
        assert alpha_channel.sum() == 0, "True 이외 위치에 불투명 픽셀 존재"

    def test_1x1_단일_픽셀_True_마스크는_해당_픽셀만_불투명이다(self):
        """Given: (1, 1) shape, 전부 True 마스크
        When:  mask_to_rgba 호출
        Then:  shape (1,1,4), 유일한 픽셀의 알파 == _DEFAULT_ALPHA
        """
        # --- Given ---
        mask = np.ones((_SINGLE_H, _SINGLE_W), dtype=bool)

        # --- When ---
        result = mask_to_rgba(mask, color=_DEFAULT_COLOR, alpha=_DEFAULT_ALPHA)

        # --- Then ---
        assert result.shape == (1, 1, 4)
        assert result[0, 0, 3] == _DEFAULT_ALPHA


# ---------------------------------------------------------------------------
# 기본 인자(default) 동작 검증
# ---------------------------------------------------------------------------
class TestMaskToRgbaDefaults:
    """color·alpha 기본값으로 호출 시 유효한 RGBA가 반환되는지 검증."""

    def test_기본값으로_호출_시_True_픽셀이_불투명이다(self):
        """Given: True 픽셀 존재, color·alpha 기본값
        When:  mask_to_rgba(mask) 호출 (인자 생략)
        Then:  True 픽셀의 알파 > 0 (기본 알파가 0이 아닌 값임)
        """
        # --- Given ---
        mask = np.zeros((_SMALL_H, _SMALL_W), dtype=bool)
        mask[0, 0] = True

        # --- When ---
        result = mask_to_rgba(mask)

        # --- Then ---
        assert result[0, 0, 3] > 0, "기본 알파가 0 — 오버레이가 완전 투명해 보이지 않음"

    def test_기본값으로_호출_시_False_픽셀_알파는_0이다(self):
        """Given: (0,0)만 True, 기본값 호출
        When:  mask_to_rgba(mask) 호출
        Then:  (0, 1) 알파 == 0
        """
        # --- Given ---
        mask = np.zeros((_SMALL_H, _SMALL_W), dtype=bool)
        mask[0, 0] = True

        # --- When ---
        result = mask_to_rgba(mask)

        # --- Then ---
        assert result[0, 1, 3] == _ALPHA_TRANSPARENT

    def test_기본값_shape_dtype이_올바르다(self):
        """Given: any 마스크, 기본값 호출
        When:  mask_to_rgba(mask) 호출
        Then:  shape (H, W, 4), dtype uint8
        """
        # --- Given ---
        mask = np.zeros((_SMALL_H, _SMALL_W), dtype=bool)

        # --- When ---
        result = mask_to_rgba(mask)

        # --- Then ---
        assert result.shape == (_SMALL_H, _SMALL_W, 4)
        assert result.dtype == np.uint8
