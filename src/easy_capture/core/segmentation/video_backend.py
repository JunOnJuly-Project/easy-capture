"""비디오 세그멘테이션 백엔드 Protocol (ADR 0010).

이미지 모드 SegmentationBackend와 별도로 분리된 비디오 전용 추상.

설계 원칙(ADR 0010):
  - ISP: 이미지 백엔드가 쓰지 않는 비디오 메서드로 오염되지 않도록 분리.
  - session은 opaque object — transformers 타입을 core/app에 노출하지 않는다.
  - core 경계 불변식: torch·transformers·PySide6·PyAV 비의존.
  - @runtime_checkable: FakeVideoBackend 주입 시 isinstance 검사 통과.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


class EmptyTrackError(Exception):
    """전파 결과 전 프레임이 빈 마스크일 때 발생한다.

    WHY: 전 프레임 추적 실패 시 조용한 폴백 없이 명시적 예외로 알린다.
         이미지 모드 EmptyMaskError 계승 — UI 워커가 잡아 한국어 안내를 표시한다.
    """


@runtime_checkable
class VideoSegmentationBackend(Protocol):
    """단일 샷 프레임 전파 추적 추상(테스트 치환용).

    호출 계약: init_session → add_click(frame 0) → propagate.
    반환 마스크: bool HxW (centroid_of_mask 계약과 동일).

    session을 불투명(opaque) object로 사용 — SAM2 InferenceSession 타입을
    core/app에 노출하지 않는다(transformers 타입 누출 금지).
    """

    device: str

    def init_session(self, frames: list[np.ndarray]) -> object:
        """구간 프레임 시퀀스로 추적 세션을 연다(무거움 — 모델 지연 로드).

        Args:
            frames: RGB HxWx3 uint8 프레임 리스트 (구간 전체).

        Returns:
            opaque session 객체 (타입은 구현체 내부 전용).
        """
        ...

    def add_click(self, session: object, point: tuple[int, int]) -> None:
        """첫 프레임(frame_idx=0)에 전경 클릭 1점을 등록한다.

        Args:
            session: init_session이 반환한 opaque 세션.
            point: (x, y) 이미지 좌표계 클릭 포인트.
        """
        ...

    def propagate(self, session: object) -> list[np.ndarray]:
        """세션을 끝까지 전파해 프레임별 bool HxW 마스크 리스트를 반환한다(무거움).

        Args:
            session: add_click으로 포인트가 등록된 opaque 세션.

        Returns:
            프레임 수만큼의 bool dtype HxW numpy 배열 리스트.
        """
        ...
