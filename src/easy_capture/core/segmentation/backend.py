"""세그멘테이션 백엔드 인터페이스 (ADR 0007).

SAM2(이미지+비디오) 또는 경량 모델(이미지 전용)을 디바이스/모드에 따라 교체한다.
비디오 메서드는 선택적이며, 미지원 백엔드는 supports_video()=False.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SegmentationBackend(Protocol):
    device: str

    def segment_image(self, frame: np.ndarray, points=None, boxes=None) -> np.ndarray:
        """단일 프레임에서 프롬프트(클릭 point/box)로 마스크 생성."""
        ...

    def supports_video(self) -> bool:
        """프레임 전파(비디오 추적) 지원 여부."""
        ...
