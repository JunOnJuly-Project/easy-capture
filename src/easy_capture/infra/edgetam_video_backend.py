"""EdgeTAM 비디오 추적 백엔드 (AC-06 fps PoC, ADR 0010 연계).

EdgeTAM(Meta, CVPR 2025)은 SAM2의 memory attention을 2D Spatial Perceiver로
경량화한 모델이다. transformers는 `EdgeTamVideoModel` + **`Sam2VideoProcessor`
재사용**으로 통합했고, init_video_session/add_inputs/propagate API가 SAM2 video와
동일하다. 따라서 Sam2VideoBackend를 상속해 모델 로드(_ensure_loaded)만 교체한다.

상태: PoC — Colab T4에서 fps·AC-01·box/negative 지원을 측정해 채택 여부 결정
      (docs/plans/ac06-fps-improvement.md §2-B·§5). transformers>=5.10 필요.
라이선스: Apache 2.0. 체크포인트: yonigozlan/EdgeTAM-hf(public, transformers 호환).
      (문서가 안내하는 yonigozlan/edgetam-video-1은 gated/미존재로 401.)
"""
from __future__ import annotations

from easy_capture.infra.sam2_video_backend import Sam2VideoBackend


class EdgetamVideoBackend(Sam2VideoBackend):
    """EdgeTAM 비디오 백엔드 — SAM2 video API 호환, 모델만 경량.

    VideoSegmentationBackend Protocol을 부모와 동일하게 준수한다.
    init_session·add_click·add_box·propagate·dtype/autocast 처리는 부모
    (Sam2VideoBackend)를 그대로 계승하고, _ensure_loaded만 EdgeTAM 모델로 교체한다.

    WHY 상속: transformers가 Sam2VideoProcessor를 EdgeTAM에 재사용하므로 입력
         조립(input_points/boxes/labels)·마스크 후처리가 SAM2와 동일하다. 모델
         클래스(EdgeTamVideoModel)와 가중치 repo만 다르다 → _ensure_loaded만 override.
    """

    def _ensure_loaded(self) -> None:
        """EdgeTAM 모델·프로세서를 from_pretrained로 지연 로드한다.

        WHY: 프로세서는 Sam2VideoProcessor를 재사용하고(transformers 통합),
             모델만 EdgeTamVideoModel을 쓴다. None 체크로 이중 로드 방지(부모 패턴).
        """
        if self._model is not None:
            return

        from transformers import EdgeTamVideoModel, Sam2VideoProcessor  # 지연 import

        self._processor = Sam2VideoProcessor.from_pretrained(self._repo)
        self._model = (
            EdgeTamVideoModel.from_pretrained(self._repo)
            .to(self.device)
            .eval()
        )
