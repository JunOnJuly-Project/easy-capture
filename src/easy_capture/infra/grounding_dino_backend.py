"""Grounding DINO 기반 재검출 백엔드 (infra, Colab 후행).

DetectionBackend Protocol 실구현.
PoC h2_cut_retrack.py 패턴 + transformers GroundingDino 매핑.

설계 원칙:
  - 지연 로드: 생성자는 repo·device·box_threshold 보관만.
               첫 detect 호출 시 _ensure_loaded()로 모델·프로세서 로드.
               Sam2VideoBackend·Sam2ImageBackend 패턴 계승.
  - torch·transformers를 메서드 내부 import → 미설치 환경에서 import 자체 오류 없음.
  - feat=None(1차 구현): 위치(IoU) 기반 재매칭. 외형 임베딩은 정확도 부족 시 후속 보강.

WHY: GPU 블로커(Grounding DINO 실추론 Colab 후행)를 Protocol 뒤로 격리.
     자동 테스트는 FakeDetectionBackend로 완전 검증하고,
     실추론은 Colab poc/colab/ 수동 검증 후행(ADR 0012, ADR 0007 이중 경로).

IMPORTANT: 이 파일은 자동 CI에서 실행되지 않는다.
           Colab GPU 환경에서 poc/colab/ 셀로 수동 검증한다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from easy_capture.core.segmentation.detection_backend import Detection

if TYPE_CHECKING:
    # 타입 힌트 전용 import — 런타임에는 지연 로드
    pass

# 기본 Grounding DINO 모델 저장소 (transformers Hub ID)
_DEFAULT_GDINO_REPO = "IDEA-Research/grounding-dino-base"
# 검출 신뢰도 임계값 기본값
_DEFAULT_BOX_THRESHOLD = 0.3
_DEFAULT_TEXT_THRESHOLD = 0.25


class GroundingDinoBackend:
    """Grounding DINO DetectionBackend 구현 (Colab 후행, 지연 로드).

    생성자: repo·device·thresholds 보관만.
    첫 detect 호출 시 _ensure_loaded()로 모델·프로세서 로드(무거움).

    WHY: SAM2 백엔드와 동일한 지연 로드 패턴 — Colab 셀에서 임포트 후
         실제 detect 호출 전까지 GPU 메모리를 점유하지 않는다.
    """

    device: str

    def __init__(
        self,
        repo: str = _DEFAULT_GDINO_REPO,
        device: str = "cuda",
        box_threshold: float = _DEFAULT_BOX_THRESHOLD,
        text_threshold: float = _DEFAULT_TEXT_THRESHOLD,
    ) -> None:
        """설정 보관만. 모델 로드는 첫 detect 호출 시 수행한다.

        Args:
            repo:           transformers Hub 모델 ID.
            device:         추론 디바이스('cuda' 또는 'cpu').
            box_threshold:  박스 신뢰도 임계값.
            text_threshold: 텍스트 매칭 임계값.
        """
        self.device = device
        self._repo = repo
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        """첫 detect 호출 시 모델·프로세서를 지연 로드한다.

        WHY: 생성자에서 로드하면 import만 해도 GPU 메모리가 점유되고
             CLI 응답이 느려진다. 지연 로드로 필요할 때만 로드한다.
             Sam2VideoBackend._ensure_loaded 패턴 완전 계승.
        """
        if self._model is not None:
            return
        # torch·transformers는 메서드 내부 import — 미설치 환경 보호
        import torch
        from transformers import (
            AutoProcessor,
            GroundingDinoForObjectDetection,
        )

        self._processor = AutoProcessor.from_pretrained(self._repo)
        self._model = GroundingDinoForObjectDetection.from_pretrained(self._repo)
        self._model = self._model.to(self.device)
        self._model.eval()

    def detect(self, frame: np.ndarray, prompt: str = "person.") -> list[Detection]:
        """프레임에서 prompt에 해당하는 후보들을 검출한다(무거움 — GPU 권장).

        PoC h2_cut_retrack.py 패턴 → transformers post_process_grounded_object_detection.

        Args:
            frame:  RGB HxWx3 uint8 단일 프레임 (컷 직후 첫 프레임).
            prompt: Grounding DINO 텍스트 프롬프트.
                    WHY: "person." 형태로 마침표를 붙이는 것이 Grounding DINO 관례.
                         마침표 없이 "person"만 사용하면 검출 누락이 빈번히 발생한다
                         (리뷰 [중요] 3 수정).

        Returns:
            Detection 리스트(검출 없으면 빈 리스트). score 내림차순 정렬.

        WHY: feat=None(1차 구현) — 위치 기반 재매칭만.
             외형 임베딩(cls_sim)은 정확도 부족 확인 시 후속 보강(ADR 0006 v1.1).
        """
        import torch
        from PIL import Image as PILImage

        self._ensure_loaded()

        # numpy RGB → PIL Image
        pil_image = PILImage.fromarray(frame)
        h, w = frame.shape[:2]

        inputs = self._processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # WHY: transformers 5.9.0 시그니처: (outputs, input_ids, threshold=, text_threshold=, target_sizes=)
        #      box_threshold 키워드가 아닌 threshold가 첫 인자명(리뷰 [중요] 1 수정).
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[(h, w)],
        )[0]

        detections = [
            Detection(
                box=tuple(float(v) for v in box.tolist()),  # type: ignore[arg-type]
                score=float(score),
                feat=None,  # 1차 구현: 위치 기반만
            )
            for box, score in zip(results["boxes"], results["scores"])
        ]
        # score 내림차순 정렬
        return sorted(detections, key=lambda d: d.score, reverse=True)
