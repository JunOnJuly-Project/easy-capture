"""SAM2 이미지 세그멘테이션 백엔드 (transformers 5.9.0, ADR 0007).

SegmentationBackend Protocol 구현체.
생성자는 repo·device 보관만 수행하고, 무거운 모델 로드는
첫 segment_image 호출 시점까지 지연한다(지연 로드 전략).

CPU 환경: ~1~3s/장. ADR 0007에 따라 이미지 모드에서만 사용.
"""
from __future__ import annotations

import numpy as np


class Sam2ImageBackend:
    """SAM2 image 세그멘테이션 백엔드.

    SegmentationBackend Protocol을 준수한다.
    transformers Sam2Model + Sam2Processor를 통해
    클릭 포인트 → bool HxW 마스크를 반환한다.
    """

    def __init__(self, repo: str, device: str) -> None:
        """repo·device 보관만. 모델은 아직 로드하지 않는다(지연).

        WHY: 모델 로드(~수 GB)는 UI 표시 이후로 미뤄 UX 응답성을 보장한다.
             첫 클릭 시 1회 로드 지연은 안내 메시지로 사용자에게 고지한다.
        """
        self.device = device
        self._repo = repo
        self._model = None
        self._processor = None

    def segment_image(
        self,
        frame: np.ndarray,
        points=None,
        boxes=None,
    ) -> np.ndarray:
        """프레임 + 클릭 포인트로 bool HxW 마스크를 반환한다.

        첫 호출 시 _ensure_loaded()로 모델을 지연 로드한다.
        반환값은 core.centroid_of_mask가 요구하는 bool HxW ndarray.
        """
        self._ensure_loaded()
        return self._run_inference(frame, points)

    def supports_video(self) -> bool:
        """이미지 전용 백엔드 — 비디오 프레임 전파 미지원."""
        return False

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """모델이 아직 로드되지 않았으면 from_pretrained로 로드한다.

        WHY: 이중 로드를 방지하기 위해 None 체크 후 할당한다.
             멀티스레드 환경(워커 QThread)에서 첫 호출만 로드되도록
             단순 플래그 대신 객체 존재 여부를 확인한다.
        """
        if self._model is not None:
            return

        import torch
        from transformers import Sam2Model, Sam2Processor

        self._processor = Sam2Processor.from_pretrained(self._repo)
        self._model = (
            Sam2Model.from_pretrained(self._repo)
            .to(self.device)
            .eval()
        )

    def _run_inference(
        self, frame: np.ndarray, points
    ) -> np.ndarray:
        """모델 추론을 실행하고 bool HxW 마스크를 반환한다."""
        import torch

        h, w = frame.shape[:2]
        input_points, input_labels = self._build_prompt(points, w, h)

        inputs = self._processor(
            images=frame,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        return self._to_mask(outputs, h, w, inputs)

    def _build_prompt(
        self, points, width: int, height: int
    ) -> tuple:
        """클릭 포인트를 processor 입력 형식으로 변환한다.

        input_points: [[[x, y]]] — 배치/오브젝트/포인트 중첩 리스트
        input_labels: [[1]]      — 1=전경 포인트
        포인트가 없으면 프레임 중앙을 기본 포인트로 사용한다.
        """
        if points is not None and len(points) > 0:
            px, py = int(points[0][0]), int(points[0][1])
        else:
            px, py = width // 2, height // 2

        input_points = [[[[px, py]]]]
        input_labels = [[[1]]]
        return input_points, input_labels

    def _to_mask(self, outputs, height: int, width: int, inputs) -> np.ndarray:
        """post_process_masks 결과를 bool HxW ndarray로 정규화한다.

        WHY: SAM2 multimask 출력(num_masks=3)의 순서는 IoU score 정렬을
             보장하지 않는다(리뷰 [중요] 2). iou_scores argmax로
             실제 best mask를 선택한다.
             iou_scores shape: (batch_size, point_batch_size, num_masks)
             → [0, 0, :].argmax() 로 마스크 인덱스 결정.
        core와의 계약: bool HxW ndarray.
        """
        original_sizes = inputs["original_sizes"]
        masks = self._processor.post_process_masks(
            outputs.pred_masks,
            original_sizes=original_sizes,
        )
        # iou_scores: (batch=1, point_batch=1, num_masks) → best 인덱스 선택
        best_idx = int(outputs.iou_scores[0, 0].argmax())
        # masks[0]: (num_objects, num_masks, H, W) → 첫 오브젝트의 best 마스크
        best_mask = masks[0][0][best_idx]
        mask_np = best_mask.cpu().numpy() if hasattr(best_mask, "cpu") else np.asarray(best_mask)
        return mask_np.astype(bool)
