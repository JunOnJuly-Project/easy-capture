"""프레임 공급원 추상 계층 (core, ADR 0008).

FrameSource Protocol·FrameSpan·FrameMeta·UnsupportedSourceError를 core에 정의한다.

다른 백엔드 추상(SegmentationBackend·VideoSegmentationBackend·DetectionBackend·
UpscaleBackend)와 동일하게 core에 위치시켜 ADR 0008 "app은 core Protocol에만 의존"
원칙을 FrameSource에도 적용한다.

의존성 규칙:
  - 이 모듈: numpy + stdlib만 (torch/av/PySide6/PIL 금지)
  - 구체 구현 (_ImageFileSource, _VideoFrameSource): infra/video_io.py
  - 팩토리 (open_source): infra/video_io.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


class UnsupportedSourceError(Exception):
    """지원하지 않는 파일 형식이거나 손상된 소스일 때 발생한다."""


@dataclass(frozen=True)
class FrameSpan:
    """구간 추출 파라미터 묶음 (매개변수 3개 규칙용).

    start: 시작 프레임 인덱스 (포함)
    end: 종료 프레임 인덱스 (미포함, exclusive)
    step: 다운샘플 간격 (기본 1 — 모든 프레임)

    WHY: read_frames(start, end) 대신 dataclass로 묶어 매개변수 3개 규칙을 준수하고
         향후 stride·fps 힌트 등 확장 시 시그니처 변경 없이 필드만 추가한다.
    """

    start: int
    end: int  # exclusive
    step: int = 1  # 다운샘플(움짤 프레임 수 절감, 기본 1)


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

    app 레이어는 이 Protocol에만 의존한다(ADR 0008 DIP).
    반환 배열: RGB HxWx3 uint8.

    WHY: 다른 백엔드 추상(SegmentationBackend 등)이 모두 core에 있듯이
         FrameSource도 core에 위치해야 app → core 단방향 의존성이 유지된다.
         infra에 있던 기존 정의는 infra/video_io.py에서 여기서 re-export한다.
    """

    def probe(self) -> FrameMeta:
        """소스 메타 정보를 반환한다."""
        ...

    def read_frame(self, index: int = 0) -> np.ndarray:
        """RGB HxWx3 uint8 배열을 반환한다."""
        ...

    def read_frames(self, span: FrameSpan) -> list[np.ndarray]:
        """FrameSpan 구간 프레임 시퀀스를 RGB HxWx3 uint8 리스트로 반환한다.

        이미지 소스: [read_frame()] 단일 프레임 1개 리스트 반환 (LSP 준수).
        비디오 소스: start PTS 시크 후 end까지 순차 디코드.
        """
        ...
