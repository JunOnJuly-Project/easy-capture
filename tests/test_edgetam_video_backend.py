"""EdgetamVideoBackend 구조 테스트 (AC-06 EdgeTAM PoC).

EdgeTAM은 SAM2 video API 호환이라 Sam2VideoBackend를 상속한다.
실제 추론(transformers>=5.10·GPU)은 Colab 노트북 후행 — 여기서는 상속·dtype
계승·Protocol 형태만 검증한다.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch 미설치")

from easy_capture.infra.edgetam_video_backend import (  # noqa: E402
    EdgetamVideoBackend,
)
from easy_capture.infra.sam2_video_backend import Sam2VideoBackend  # noqa: E402


def test_sam2_백엔드를_상속한다():
    # SAM2 video API 호환 → init_session/add_box/propagate 등을 그대로 계승
    assert issubclass(EdgetamVideoBackend, Sam2VideoBackend)


def test_dtype_주입을_계승한다():
    backend = EdgetamVideoBackend(
        repo="yonigozlan/edgetam-video-1", device="cuda", dtype="float16"
    )
    assert backend._dtype == "float16"
    assert backend.device == "cuda"


def test_기본_dtype은_float32_무회귀():
    backend = EdgetamVideoBackend(repo="yonigozlan/edgetam-video-1", device="cpu")
    assert backend._dtype == "float32"


def test_잘못된_dtype은_ValueError():
    with pytest.raises(ValueError, match="지원하지 않는 dtype"):
        EdgetamVideoBackend(repo="r", device="cuda", dtype="fp16")


def test_핵심_메서드를_보유한다():
    # Protocol 형태(부모 계승) — init_session/add_click/add_box/propagate
    for name in ("init_session", "add_click", "add_box", "propagate"):
        assert callable(getattr(EdgetamVideoBackend, name))
