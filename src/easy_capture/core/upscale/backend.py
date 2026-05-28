"""업스케일 백엔드 인터페이스 (ADR 0004 / ADR 0007).

SwinIR/Swin2SR(기본) 또는 Real-ESRGAN(옵션, v1.x)을 교체한다.
SegmentationBackend와 대칭 패턴: Protocol은 core(순수), 구현은 infra.
경계 불변식: torch·transformers·PySide6·av import 금지.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class UpscaleBackend(Protocol):
    """업스케일 백엔드 Protocol.

    device: 추론 디바이스 ('cpu' or 'cuda')
    scale: 모델이 적용하는 고정 배율(예: 2, 4). repo가 결정.

    SegmentationBackend.device 패턴과 정합.
    scale을 속성으로 노출해 UI가 결과 크기를 미리 계산할 수 있게 한다.
    """

    device: str
    scale: int  # 모델이 적용하는 고정 배율(예: 2, 4). repo가 결정.

    def upscale(self, image_rgb: np.ndarray) -> np.ndarray:
        """RGB HxWx3 uint8 → (H*scale, W*scale, 3) uint8 RGB 반환(무거움)."""
        ...
