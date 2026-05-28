"""재검출 백엔드 Protocol (ADR 0012).

샷 경계 직후 단일 프레임에서 동일 클래스 후보를 재검출하는 추상.
Grounding DINO 실구현(infra)을 core에서 분리하기 위한 인터페이스.

설계 원칙(ADR 0012 · ADR 0010 계승):
  - ISP: VideoSegmentationBackend에 검출 메서드를 끼워 넣지 않고 별도 Protocol 신설.
  - session 불필요(무상태): 1프레임 → 후보 리스트 출력. SAM2와 달리 opaque session 없음.
  - core 경계 불변식: torch·transformers·PySide6·PyAV·scenedetect 비의존.
  - @runtime_checkable: FakeDetectionBackend 주입 시 isinstance 검사 통과.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# Box 타입 alias — rematch.py Box와 동일
Box = tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class Detection:
    """재검출 후보 1개(불변). 위치(box) + 선택적 외형특징(feat).

    box:   (x1, y1, x2, y2) 픽셀 좌표 — rematch_score pos_sim 입력.
    score: 검출기 자체 신뢰도(후보 정렬·필터용, 재매칭 점수와 별개).
    feat:  외형 임베딩(있으면 rematch_score cls_sim 입력, 없으면 None=위치만 평가).

    WHY: frozen dataclass로 묶어 후보 1개의 위치·신뢰도·외형특징을 한 단위로 다룬다.
         매개변수 폭증 방지. TrackResult·SegmentResult 패턴 계승.
         feat=None 허용: Grounding DINO만으로는 외형 임베딩이 약할 수 있으므로
         1차 구현은 위치(IoU) 기반, feat는 확장점만 열어둔다(KISS).
    """

    box: Box
    score: float
    feat: "np.ndarray | None" = field(default=None)


@runtime_checkable
class DetectionBackend(Protocol):
    """컷 직후 단일 프레임에서 동일 클래스 후보를 재검출하는 추상(테스트 치환용).

    무상태(stateless): session 없음 — 1프레임 입력 → 후보 리스트 출력.
    모델 로드는 무거우므로 지연 로드(_ensure_loaded)는 Sam2VideoBackend 패턴 계승.

    WHY(ISP): VideoSegmentationBackend가 쓰지 않는 검출 메서드로 오염되지 않게 분리.
    WHY(OCP): 향후 YOLO·전용 re-ID 교체 시 Protocol만 구현하면 된다.
    """

    device: str

    def detect(self, frame: np.ndarray, prompt: str) -> list[Detection]:
        """프레임에서 prompt에 해당하는 후보들을 검출한다(무거움 — GPU 권장).

        Args:
            frame:  RGB HxWx3 uint8 단일 프레임 (컷 직후 첫 프레임).
            prompt: Grounding DINO 텍스트 프롬프트(기본 'person').

        Returns:
            Detection 리스트(검출 없으면 빈 리스트). score 내림차순 정렬.
        """
        ...
