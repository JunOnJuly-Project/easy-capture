"""VideoCaptureUseCase.export 업스케일 결합 테스트 (비디오 업스케일 슬라이스).

이미지 모드 export(upscaler=)와 대칭. 비디오 크롭 결과에 업스케일을 적용한다.
실제 Swin2SR은 GPU 무거움 → Fake 업스케일러(정수 배율 nearest)로 결합/무회귀만 검증.
실모델 검증은 Colab 노트북 후행(ADR 0009/0018 경로).
"""
from __future__ import annotations

import numpy as np
import pytest

imageio = pytest.importorskip("imageio", reason="imageio 미설치 — GIF 라운드트립 불가")

from easy_capture.app.video_capture import VideoCaptureUseCase  # noqa: E402
from easy_capture.core.export.video_export import VideoExportConfig  # noqa: E402
from tests.fixtures.fakes import FakeFrameSource, FakeVideoBackend  # noqa: E402

_N_FRAMES = 3
_CROP = 10  # 박스 10×10
_SCALE = 2


class _FakeUpscaler:
    """UpscaleBackend 더블 — 정수 배율 nearest 확대 + 호출 카운터."""

    def __init__(self, scale: int = _SCALE) -> None:
        self.scale = scale
        self.calls = 0

    def upscale(self, image_rgb: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.repeat(
            np.repeat(image_rgb, self.scale, axis=0), self.scale, axis=1
        )


def _make_usecase() -> VideoCaptureUseCase:
    # export는 source/backend를 호출하지 않는다(frames·boxes 직접 입력) — 더블로 충분.
    return VideoCaptureUseCase(
        source=FakeFrameSource(), backend=FakeVideoBackend(), detector=None
    )


def _frames_boxes():
    frames = [np.zeros((40, 40, 3), dtype=np.uint8) for _ in range(_N_FRAMES)]
    boxes = [(0, 0, _CROP, _CROP)] * _N_FRAMES
    return frames, boxes


def test_업스케일러_주입_시_크롭이_확대되어_저장된다(tmp_path):
    uc = _make_usecase()
    frames, boxes = _frames_boxes()
    up = _FakeUpscaler(scale=_SCALE)
    out = tmp_path / "o.gif"

    uc.export(frames, boxes, (str(out), VideoExportConfig(fmt="gif")), upscaler=up)

    assert up.calls == _N_FRAMES  # 프레임마다 1회
    imgs = imageio.mimread(str(out))
    assert imgs[0].shape[:2] == (_CROP * _SCALE, _CROP * _SCALE)


def test_업스케일러_None이면_원본_크기다_무회귀(tmp_path):
    uc = _make_usecase()
    frames, boxes = _frames_boxes()
    out = tmp_path / "o.gif"

    uc.export(frames, boxes, (str(out), VideoExportConfig(fmt="gif")))

    imgs = imageio.mimread(str(out))
    assert imgs[0].shape[:2] == (_CROP, _CROP)
