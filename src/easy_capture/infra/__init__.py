"""외부 의존(디바이스·모델·비디오 IO) 어댑터."""
from easy_capture.infra.device import (detect_device, select_sam2_repo,
                                       supports_video_tracking)

__all__ = ["detect_device", "select_sam2_repo", "supports_video_tracking"]
