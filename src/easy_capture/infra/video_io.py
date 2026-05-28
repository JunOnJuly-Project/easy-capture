"""프레임 추출 인프라 (PyAV/Pillow, ADR 0005).

이미지 또는 동영상 파일에서 단일 프레임을 RGB(BT.709 정규화) numpy 배열로 추출한다.
app 레이어는 FrameSource Protocol에만 의존한다(DIP).

지원:
- 이미지 파일 (.jpg/.jpeg/.png/.bmp/.webp 등): Pillow로 읽음
- 동영상 파일 (.mp4/.mov/.avi 등): PyAV로 PTS 시크 후 1프레임 디코드
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# 이미지 파일로 처리할 확장자 집합
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"}
)


class UnsupportedSourceError(Exception):
    """지원하지 않는 파일 형식이거나 손상된 소스일 때 발생한다."""


@dataclass(frozen=True)
class FrameMeta:
    """프레임/소스 메타 (UI 표시·검증용).

    is_video: True면 동영상, False면 이미지 파일
    fps: 이미지 소스면 None
    """

    width: int
    height: int
    is_video: bool
    fps: float | None  # 이미지면 None


@runtime_checkable
class FrameSource(Protocol):
    """단일 프레임 공급원 추상(테스트 치환용).

    app 레이어는 이 Protocol에만 의존한다.
    반환 배열: RGB HxWx3 uint8.
    """

    def probe(self) -> FrameMeta:
        """소스 메타 정보를 반환한다."""
        ...

    def read_frame(self, index: int = 0) -> np.ndarray:
        """RGB HxWx3 uint8 배열을 반환한다."""
        ...


def open_source(path: str) -> FrameSource:
    """확장자로 이미지/영상을 판별 후 적절한 FrameSource 구현체를 생성한다(팩토리).

    미지원/손상 파일은 UnsupportedSourceError(한국어 메시지) 발생.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        return _ImageFileSource(path)
    return _VideoFrameSource(path)


# ---------------------------------------------------------------------------
# 내부 구현체 (공개 API 아님)
# ---------------------------------------------------------------------------

class _ImageFileSource:
    """Pillow로 이미지 파일을 읽는 FrameSource 구현체."""

    def __init__(self, path: str) -> None:
        self._path = path

    def probe(self) -> FrameMeta:
        """Pillow로 이미지 크기를 읽어 FrameMeta를 반환한다."""
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(self._path) as img:
                w, h = img.size
                return FrameMeta(width=w, height=h, is_video=False, fps=None)
        except (FileNotFoundError, UnidentifiedImageError, Exception) as exc:
            raise UnsupportedSourceError(
                f"이미지 파일을 읽을 수 없습니다: {self._path}\n원인: {exc}"
            ) from exc

    def read_frame(self, index: int = 0) -> np.ndarray:
        """이미지를 RGB numpy 배열로 변환한다. index는 무시된다."""
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(self._path) as img:
                return np.array(img.convert("RGB"), dtype=np.uint8)
        except (FileNotFoundError, UnidentifiedImageError, Exception) as exc:
            raise UnsupportedSourceError(
                f"이미지 파일을 읽을 수 없습니다: {self._path}\n원인: {exc}"
            ) from exc


class _VideoFrameSource:
    """PyAV로 동영상 파일에서 지정 프레임을 추출하는 FrameSource 구현체.

    ADR 0005: PTS 기반 시크 후 1프레임 디코드, RGB BT.709 full 정규화.
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def probe(self) -> FrameMeta:
        """PyAV로 스트림 메타를 읽어 FrameMeta를 반환한다."""
        try:
            import av
        except ImportError as exc:
            raise UnsupportedSourceError(
                "동영상 파일 처리에 PyAV가 필요합니다. "
                "'pip install av' 후 재시도하세요."
            ) from exc

        try:
            with av.open(self._path) as container:
                stream = container.streams.video[0]
                w = stream.codec_context.width
                h = stream.codec_context.height
                fps = float(stream.average_rate) if stream.average_rate else None
                return FrameMeta(width=w, height=h, is_video=True, fps=fps)
        except Exception as exc:
            raise UnsupportedSourceError(
                f"동영상 파일을 읽을 수 없습니다: {self._path}\n원인: {exc}"
            ) from exc

    def read_frame(self, index: int = 0) -> np.ndarray:
        """index 번째 프레임을 RGB BT.709 배열로 반환한다.

        WHY: PTS 시크를 사용하면 임의 프레임을 전체 디코드 없이 추출 가능하다.
        이미지 모드 슬라이스에서는 index=0(첫 프레임)만 사용한다.
        """
        try:
            import av
        except ImportError as exc:
            raise UnsupportedSourceError(
                "동영상 파일 처리에 PyAV가 필요합니다."
            ) from exc

        try:
            with av.open(self._path) as container:
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = "NONKEY"
                for frame in container.decode(video=0):
                    rgb = frame.to_ndarray(format="rgb24")
                    return rgb.astype(np.uint8)
        except Exception as exc:
            raise UnsupportedSourceError(
                f"동영상 프레임을 디코드할 수 없습니다: {self._path}\n원인: {exc}"
            ) from exc

        raise UnsupportedSourceError(
            f"동영상에서 프레임을 찾을 수 없습니다: {self._path}"
        )
