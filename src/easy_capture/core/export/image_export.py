"""이미지 인코딩 순수 로직 (PNG/JPG, Pillow only).

경계 불변식: torch·PySide6·PyAV·transformers import 금지.
crop_array는 순수 함수(부수효과 없음).
save_image만 IO 부수효과(파일 쓰기)를 가진다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# 지원 포맷 목록 — 확장 시 여기만 수정 (OCP)
_SUPPORTED_FORMATS: frozenset[str] = frozenset({"png", "jpg", "jpeg"})

# Pillow save 시 포맷 문자열 매핑 (jpg → JPEG)
_PILLOW_FORMAT_MAP: dict[str, str] = {
    "png": "PNG",
    "jpg": "JPEG",
    "jpeg": "JPEG",
}


@dataclass(frozen=True)
class ExportConfig:
    """이미지 저장 설정.

    fmt: 'png' 또는 'jpg' (기본 png)
    quality: JPG 압축 품질 1~95 (PNG는 무시)
    color_space: 색공간 태깅 레이블 (data-flow §2)
    """

    fmt: str = "png"
    quality: int = 95
    color_space: str = "sRGB"


def crop_array(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """RGB 배열을 (x1, y1, x2, y2) 박스로 슬라이스한다.

    WHY: make_crop_box가 짝수·경계 클램프를 이미 보장하므로
         여기서는 단순 numpy 슬라이스만 수행한다.
    반환값은 뷰가 아닌 복사본(downstream 변경으로부터 격리).
    """
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2].copy()


def save_image(frame: np.ndarray, path: str, config: ExportConfig) -> None:
    """RGB numpy 배열을 path에 config 포맷으로 저장한다(Pillow).

    지원 포맷: 'png', 'jpg'/'jpeg'. 그 외 포맷은 ValueError.
    디렉토리/권한 오류는 상위 레이어로 전파(래핑 안 함).
    """
    fmt = config.fmt.lower()
    _validate_format(fmt)

    pillow_fmt = _PILLOW_FORMAT_MAP[fmt]
    img = Image.fromarray(frame, mode="RGB")

    save_kwargs: dict = {}
    if pillow_fmt == "JPEG":
        save_kwargs["quality"] = config.quality

    img.save(path, format=pillow_fmt, **save_kwargs)


def _validate_format(fmt: str) -> None:
    """지원하지 않는 포맷이면 ValueError를 발생시킨다."""
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"지원하지 않는 포맷: '{fmt}'. "
            f"지원 포맷: {sorted(_SUPPORTED_FORMATS)}"
        )
