"""SAM2 비디오 추적 백엔드 (transformers 5.9.0, ADR 0010).

VideoSegmentationBackend Protocol 구현체.
생성자는 repo·device 보관만 수행하고, 모델 로드는
첫 init_session 호출 시점까지 지연한다(지연 로드 전략, ADR 0007).

GPU 블로커: CPU에서 ≈0.10fps. 실추론 검증은 Colab GPU 후행(ADR 0007).
PoC 패턴 매핑(poc/h1_track.py Sam2VideoModel/Sam2VideoProcessor):
  init_session → processor.init_video_session(video=frames, ...)
  add_click    → processor.add_inputs_to_inference_session(...)
  propagate    → model.propagate_in_video_iterator(session, ...) → bool HxW 리스트

session은 opaque object — transformers Sam2VideoInferenceSession 타입을
core/app에 노출하지 않는다(ADR 0010 경계 불변식).
"""
from __future__ import annotations

import numpy as np


class Sam2VideoBackend:
    """SAM2 video 추적 백엔드.

    VideoSegmentationBackend Protocol을 준수한다.
    transformers Sam2VideoModel + Sam2VideoProcessor를 통해
    첫 프레임 클릭 → 프레임별 bool HxW 마스크 리스트를 반환한다.

    실추론은 GPU 환경(Colab) 후행 검증. 코드만 PoC 패턴대로 정확히 작성.
    """

    def __init__(self, repo: str, device: str) -> None:
        """repo·device 보관만. 모델은 아직 로드하지 않는다(지연).

        WHY: 모델 로드(~수 GB)는 UI 표시 이후로 미뤄 UX 응답성을 보장한다.
             Sam2ImageBackend._ensure_loaded 패턴을 그대로 계승한다.
        """
        self.device = device
        self._repo = repo
        self._model = None
        self._processor = None

    def init_session(self, frames: list[np.ndarray]) -> object:
        """구간 프레임 시퀀스로 SAM2 video 추적 세션을 연다(무거움).

        첫 호출 시 _ensure_loaded()로 모델을 지연 로드한다.
        PoC 패턴: processor.init_video_session(video=frames, ...).

        Args:
            frames: RGB HxWx3 uint8 프레임 리스트 (구간 전체).

        Returns:
            opaque Sam2VideoInferenceSession (core/app에 타입 노출 금지).
        """
        self._ensure_loaded()
        import torch  # 지연 import — infra 내부에서만 사용

        session = self._processor.init_video_session(
            video=frames,
            inference_device=self.device,
            dtype=torch.float32,
        )
        return session

    def add_click(self, session: object, point: tuple[int, int]) -> None:
        """첫 프레임(frame_idx=0)에 전경 클릭 1점을 등록한다.

        PoC 패턴: processor.add_inputs_to_inference_session(
            session, frame_idx=0, obj_ids=1,
            input_points=[[[[x, y]]]], input_labels=[[[1]]]
        ).

        Args:
            session: init_session이 반환한 opaque 세션.
            point: (x, y) 이미지 좌표계 클릭 포인트.
        """
        x, y = int(point[0]), int(point[1])
        self._processor.add_inputs_to_inference_session(
            session,
            frame_idx=0,
            obj_ids=1,
            input_points=[[[[x, y]]]],
            input_labels=[[[1]]],
        )

    def propagate(self, session: object) -> list[np.ndarray]:
        """세션을 끝까지 전파해 프레임별 bool HxW 마스크 리스트를 반환한다(무거움).

        PoC 패턴: model.propagate_in_video_iterator(session, start_frame_idx=0)
        → post_process_masks([out.pred_masks], original_sizes=[(h, w)])[0]
        → bool HxW numpy 배열.

        Args:
            session: add_click으로 포인트가 등록된 opaque 세션.

        Returns:
            프레임별 bool dtype HxW numpy 배열 리스트.
        """
        import torch  # 지연 import

        masks: list[np.ndarray] = []
        with torch.inference_mode():
            for out in self._model.propagate_in_video_iterator(
                session, start_frame_idx=0
            ):
                mask = self._extract_mask(out, session)
                masks.append(mask)
        return masks

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """모델이 아직 로드되지 않았으면 from_pretrained로 로드한다.

        WHY: Sam2ImageBackend._ensure_loaded 패턴 계승.
             None 체크로 이중 로드 방지. 워커 QThread 첫 호출에서만 실행.
        """
        if self._model is not None:
            return

        import torch  # 지연 import
        from transformers import Sam2VideoModel, Sam2VideoProcessor  # 지연 import

        self._processor = Sam2VideoProcessor.from_pretrained(self._repo)
        self._model = (
            Sam2VideoModel.from_pretrained(self._repo)
            .to(self.device)
            .eval()
        )

    def _extract_mask(self, out, session: object) -> np.ndarray:
        """propagate_in_video_iterator 한 스텝 출력에서 bool HxW 마스크를 추출한다.

        PoC 패턴:
          post = processor.post_process_masks(
              [out.pred_masks], original_sizes=[(h, w)]
          )[0]
          bool HxW = (post > 0).squeeze().

        WHY: session에서 원본 크기(h, w)를 추출해 post_process_masks에 전달한다.
             session.input_frames_size가 없으면 pred_masks shape 역산 폴백.
             배치/오브젝트 차원(1,1,H,W) → (H,W)로 squeeze해 bool 변환.
        """
        # 원본 크기 추출 — session 구현 의존성이 없도록 pred_masks shape 사용
        h_w = _infer_original_size(out)
        post = self._processor.post_process_masks(
            [out.pred_masks],
            original_sizes=[h_w],
        )[0]
        arr = post.cpu().numpy() if hasattr(post, "cpu") else np.asarray(post)
        # (1, 1, H, W) 또는 (1, H, W) → (H, W) squeeze
        mask_2d = arr.squeeze()
        return mask_2d.astype(bool)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _infer_original_size(out) -> tuple[int, int]:
    """propagate 출력에서 원본 (H, W) 크기를 추론한다.

    WHY: session에서 원본 크기를 꺼내는 안전한 방법이 없을 때
         pred_masks shape에서 역산한다. 마지막 두 차원이 (H, W)이다.
    """
    shape = out.pred_masks.shape
    # shape: (batch, obj, H, W) 또는 (batch, H, W) — 뒤에서 2개 차원
    h, w = int(shape[-2]), int(shape[-1])
    return h, w
