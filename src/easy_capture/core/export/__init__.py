"""이미지 인코딩 도메인 (순수 로직).

Pillow만 사용. torch·PySide6·PyAV·transformers 비의존(경계 불변식).
"""
from easy_capture.core.export.image_export import ExportConfig, crop_array, save_image

__all__ = ["ExportConfig", "crop_array", "save_image"]
