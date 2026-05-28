"""프레임 추출 인프라 (PyAV/Pillow, ADR 0005).

이미지 또는 동영상 파일에서 단일 프레임 또는 구간 프레임 시퀀스를
RGB(BT.709 정규화) numpy 배열로 추출한다.
app 레이어는 FrameSource Protocol에만 의존한다(ADR 0008 DIP).

지원:
- 이미지 파일 (.jpg/.jpeg/.png/.bmp/.webp 등): Pillow로 읽음
- 동영상 파일 (.mp4/.mov/.avi 등): PyAV로 PTS 시크 후 1프레임 or 구간 디코드

WHY: FrameSource·FrameSpan·FrameMeta·UnsupportedSourceError 추상은
     core/source/frame_source.py로 이동했다(ADR 0008 정합).
     하위 호환성을 위해 여기서 re-export한다 — 기존 import 경로가 동작한다.
     구체 구현(_ImageFileSource, _VideoFrameSource)과 팩토리(open_source)는 여기 잔류.
"""
from __future__ import annotations

import os

import numpy as np

# 추상은 core에서 가져온다 (ADR 0008) — 구현만 여기 잔류
from easy_capture.core.source.frame_source import (
    FrameMeta,
    FrameSource,
    FrameSpan,
    UnsupportedSourceError,
)

# re-export: 기존 infra.video_io import 경로 하위 호환 유지
__all__ = [
    "FrameMeta",
    "FrameSource",
    "FrameSpan",
    "UnsupportedSourceError",
    "open_source",
]

# 이미지 파일로 처리할 확장자 집합
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"}
)

# 메모리 보호용 구간 최대 프레임 수 (fps 30 × 30초)
_MAX_SPAN_FRAMES = 900


def open_source(path: str) -> FrameSource:
    """확장자로 이미지/영상을 판별 후 적절한 FrameSource 구현체를 생성한다(팩토리).

    미지원/손상 파일은 UnsupportedSourceError(한국어 메시지) 발생.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        return _ImageFileSource(path)
    return _VideoFrameSource(path)


# ---------------------------------------------------------------------------
# 내부 헬퍼 (공개 API 아님)
# ---------------------------------------------------------------------------

def _parse_fps(stream) -> float | None:
    """스트림에서 fps를 산출한다(r_frame_rate 우선, average_rate 폴백).

    WHY: average_rate는 VFR 영상에서 부정확할 수 있다. r_frame_rate가 더 신뢰성 높다.
    """
    if stream.guessed_rate:
        return float(stream.guessed_rate)
    if stream.average_rate:
        return float(stream.average_rate)
    return None


def _decode_span(path: str, span: FrameSpan) -> list[np.ndarray]:
    """PyAV로 구간 프레임을 순차 디코드해 RGB 리스트로 반환한다.

    WHY: _VideoFrameSource.read_frames에서 분리해 함수 20줄 규칙 준수.
         step 다운샘플 + _MAX_SPAN_FRAMES 가드로 메모리 폭증을 방지한다.
    """
    import av  # 함수 내 지연 import (호출 시점에만 필요)

    result: list[np.ndarray] = []
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "DEFAULT"
            for idx, frame in enumerate(container.decode(stream)):
                if idx < span.start:
                    continue
                if idx >= span.end:
                    break
                if (idx - span.start) % span.step == 0:
                    result.append(frame.to_ndarray(format="rgb24").astype(np.uint8))
                if len(result) >= _MAX_SPAN_FRAMES:
                    # WHY: 메모리 폭증 방지 — 사용자에게 경고 로그 출력
                    import warnings
                    warnings.warn(
                        f"구간 프레임이 최대치({_MAX_SPAN_FRAMES})를 초과해 잘렸습니다. "
                        "step을 늘려 구간을 줄이세요.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    break
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise UnsupportedSourceError(
            f"동영상 구간을 디코드할 수 없습니다: {path}\n원인: {exc}"
        ) from exc
    return result


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
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
            raise UnsupportedSourceError(
                f"이미지 파일을 읽을 수 없습니다: {self._path}\n원인: {exc}"
            ) from exc

    def read_frame(self, index: int = 0) -> np.ndarray:
        """이미지를 RGB numpy 배열로 변환한다. index는 무시된다."""
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(self._path) as img:
                return np.array(img.convert("RGB"), dtype=np.uint8)
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
            raise UnsupportedSourceError(
                f"이미지 파일을 읽을 수 없습니다: {self._path}\n원인: {exc}"
            ) from exc

    def read_frames(self, span: FrameSpan) -> list[np.ndarray]:
        """이미지 소스는 구간 무관 단일 프레임 1개 리스트를 반환한다(LSP 준수).

        WHY: FrameSource Protocol에 read_frames가 추가됐을 때 이미지 소스도
             구현 의무가 생긴다. UnsupportedSourceError 대신 1개 리스트를 반환해
             호출자가 이미지/영상 분기를 하지 않아도 되게 한다(LSP 안전).
        """
        return [self.read_frame()]


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
                # WHY: r_frame_rate가 VFR 영상에서도 정확하다(average_rate 폴백).
                fps = _parse_fps(stream)
                return FrameMeta(width=w, height=h, is_video=True, fps=fps)
        except (FileNotFoundError, OSError, ValueError, IndexError) as exc:
            raise UnsupportedSourceError(
                f"동영상 파일을 읽을 수 없습니다: {self._path}\n원인: {exc}"
            ) from exc

    def read_frame(self, index: int = 0) -> np.ndarray:
        """동영상 첫 프레임을 RGB BT.709 배열로 반환한다.

        현재 슬라이스는 첫 프레임(index=0)만 지원한다.
        index 인자는 차후 타임라인 슬라이스에서 PTS 시크로 구현 예정.
        WHY: skip_frame="NONKEY" 제거 — 첫 프레임은 보통 키프레임이며
             일부 컨테이너에서 빈 루프 위험이 있다(리뷰 [중요] 4).
        """
        try:
            import av
        except ImportError as exc:
            raise UnsupportedSourceError(
                "동영상 파일 처리에 PyAV가 필요합니다."
            ) from exc

        try:
            with av.open(self._path) as container:
                for frame in container.decode(video=0):
                    return frame.to_ndarray(format="rgb24").astype(np.uint8)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise UnsupportedSourceError(
                f"동영상 프레임을 디코드할 수 없습니다: {self._path}\n원인: {exc}"
            ) from exc

        raise UnsupportedSourceError(
            f"동영상에서 프레임을 찾을 수 없습니다: {self._path}"
        )

    def read_frames(self, span: FrameSpan) -> list[np.ndarray]:
        """FrameSpan 구간 프레임 시퀀스를 RGB HxWx3 uint8 리스트로 반환한다.

        PyAV로 start PTS 시크 후 end까지 순차 디코드한다(ADR 0005).
        step 다운샘플로 메모리 사용량을 줄인다.

        WHY: SAM2 video init_video_session은 전체 프레임을 한 번에 요구하므로
             스트리밍 대신 리스트로 보관한다. _MAX_SPAN_FRAMES 가드로 폭증 방지.
        """
        try:
            import av
        except ImportError as exc:
            raise UnsupportedSourceError(
                "동영상 파일 처리에 PyAV가 필요합니다."
            ) from exc

        frames = _decode_span(self._path, span)
        return frames
