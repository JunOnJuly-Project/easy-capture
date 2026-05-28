"""Swin2SR 초해상도 백엔드 (transformers, ADR 0004).

UpscaleBackend Protocol 구현체. SegmentationBackend↔Sam2ImageBackend와 동일 패턴.
생성자는 repo·device·scale 보관만, 모델 로드는 첫 upscale 호출까지 지연(지연 로드).
배율은 repo에 의해 고정(x2 모델=2배). "배율 선택"=repo 선택.
"""
from __future__ import annotations

import numpy as np

from easy_capture.core.upscale.normalize import reconstruction_to_rgb_uint8


class Swin2srUpscaleBackend:
    """Swin2SR 업스케일 백엔드. UpscaleBackend Protocol을 준수한다."""

    def __init__(self, repo: str, device: str, scale: int) -> None:
        """repo·device·scale 보관만. 모델은 아직 로드하지 않는다(지연).

        WHY: 모델 로드(수백 MB)는 '저장' 클릭 후 워커에서 1회만 수행해
             UI 응답성을 보장한다. scale은 repo가 결정하므로 주입받아 보관.
        """
        self.device = device
        self.scale = scale
        self._repo = repo
        self._model = None
        self._processor = None

    def upscale(self, image_rgb: np.ndarray) -> np.ndarray:
        """RGB HxWx3 uint8 → (H*scale, W*scale, 3) uint8 RGB(무거움)."""
        self._ensure_loaded()
        return self._run_inference(image_rgb)

    def _ensure_loaded(self) -> None:
        """모델 미로드 시 from_pretrained로 지연 로드(이중 로드 방지).

        WHY: 이중 로드를 방지하기 위해 None 체크 후 할당한다.
             torch/transformers는 미사용 시 무거운 import를 피하기 위해 지연 import.
        """
        if self._model is not None:
            return
        from transformers import (
            Swin2SRForImageSuperResolution,
            Swin2SRImageProcessor,
        )
        self._processor = Swin2SRImageProcessor.from_pretrained(self._repo)
        self._model = (
            Swin2SRForImageSuperResolution.from_pretrained(self._repo)
            .to(self.device)
            .eval()
        )

    def _run_inference(self, image_rgb: np.ndarray) -> np.ndarray:
        """모델 추론 → reconstruction 텐서를 순수 정규화 함수로 RGB uint8 변환.

        WHY: reconstruction 텐서 추출(infra 책임)과
             정규화 산술(core 순수 함수 책임)을 분리한다(SRP).
             infra는 detach·cpu·float·numpy 변환까지만 담당.
        """
        import torch

        src_h, src_w = image_rgb.shape[:2]
        inputs = self._processor(image_rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._model(**inputs)
        # WHY: Swin2SRImageProcessor가 입력을 8의 배수로 패딩하므로 출력은
        #      (1, 3, H'*scale, W'*scale)(H'/W'=패딩 크기)일 수 있다. 원본 기준
        #      (H*scale, W*scale)로 재크롭해 Protocol 계약 크기를 보장한다(리뷰 [중요]).
        recon_np = out.reconstruction[0].detach().cpu().float().numpy()  # (3, H'*s, W'*s)
        rgb = reconstruction_to_rgb_uint8(recon_np)
        return rgb[: src_h * self.scale, : src_w * self.scale]
