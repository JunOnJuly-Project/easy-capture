"""Swin2SR reconstruction 텐서를 RGB uint8 이미지로 정규화(순수 numpy).

경계 불변식: torch·transformers import 금지. (3,H,W) float ndarray만 받는다.
WHY: 정규화 산술(clip·*255·축전치·uint8)을 infra에서 떼어내 순수 함수로 두면
     torch 없이 단위 테스트할 수 있고, Real-ESRGAN 등 다른 백엔드도 재사용한다(DRY).
"""
from __future__ import annotations

import numpy as np

# 정규화 상수 — 매직넘버 금지
_MAX_UINT8 = 255
_FLOAT_MIN = 0.0
_FLOAT_MAX = 1.0


def reconstruction_to_rgb_uint8(recon_chw: np.ndarray) -> np.ndarray:
    """(3, H, W) float [0,1) 근사 → (H, W, 3) uint8 RGB.

    Given: 채널 우선(CHW) float 배열(범위가 [0,1] 밖으로 약간 벗어날 수 있음).
           np.asarray로 변환하므로 torch 텐서 입력도 처리 가능.
    When:  [0,1] 클램프 → *255 반올림 → uint8 → HWC 전치
    Then:  (H, W, 3) uint8, 값 0~255

    WHY: Swin2SR reconstruction이 [0,1] 밖으로 약간 벗어날 수 있으므로(계획서 §3-2)
         클램프 없이 uint8 변환 시 언더/오버플로우로 잘못된 색이 나온다.
    """
    arr = np.asarray(recon_chw, dtype=np.float64)  # torch 텐서 → numpy 안전 변환
    clamped = np.clip(arr, _FLOAT_MIN, _FLOAT_MAX)
    scaled = np.rint(clamped * _MAX_UINT8).astype(np.uint8)  # (3, H, W)
    return np.transpose(scaled, (1, 2, 0))                   # (H, W, 3)
