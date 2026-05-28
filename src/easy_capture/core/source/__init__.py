"""core.source 패키지 — 프레임 공급원 추상 공개 API."""
from easy_capture.core.source.frame_source import (
    FrameMeta,
    FrameSource,
    FrameSpan,
    UnsupportedSourceError,
)

__all__ = ["FrameMeta", "FrameSource", "FrameSpan", "UnsupportedSourceError"]
