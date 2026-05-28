"""유스케이스 레이어 — 슬라이스 흐름 오케스트레이션.

UI는 이 레이어만 호출한다. core/infra 구체 타입에 의존하지 않는다(DIP).
"""
from easy_capture.app.image_capture import CropRequest, EmptyMaskError, ImageCaptureUseCase

__all__ = ["CropRequest", "EmptyMaskError", "ImageCaptureUseCase"]
