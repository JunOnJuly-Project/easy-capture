"""Sam2VideoBackend dtype 주입 단위 테스트 (AC-06 fps 측정 준비).

대상:
  - _validate_dtype: 지원 목록 검증(torch 비의존)
  - _resolve_torch_dtype: 문자열 → torch.dtype 매핑
  - 생성자 dtype 보관 + 기본값 무회귀(float32)
  - _autocast_context: fp32/cpu는 no-op, fp16/bf16+cuda는 autocast

검증 원칙:
  - 실제 SAM2 추론(transformers·GPU)은 검증하지 않는다 — 노트북(Colab) 후행.
  - dtype 분기·매핑은 순수/CPU torch로 충분히 단위 검증 가능하다.
"""
from __future__ import annotations

import contextlib

import pytest

torch = pytest.importorskip("torch", reason="torch 미설치 — dtype 매핑 검증 불가")

from easy_capture.infra.sam2_video_backend import (  # noqa: E402
    Sam2VideoBackend,
    _resolve_torch_dtype,
    _validate_dtype,
)


class TestValidateDtype:
    """_validate_dtype이 지원 목록만 통과시킨다."""

    @pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
    def test_지원_dtype은_통과한다(self, dtype: str):
        _validate_dtype(dtype)  # 예외 없으면 통과

    @pytest.mark.parametrize("dtype", ["fp16", "float64", "int8", ""])
    def test_미지원_dtype은_ValueError(self, dtype: str):
        with pytest.raises(ValueError, match="지원하지 않는 dtype"):
            _validate_dtype(dtype)


class TestResolveTorchDtype:
    """_resolve_torch_dtype이 문자열을 정확한 torch.dtype으로 매핑한다."""

    def test_float32_매핑(self):
        assert _resolve_torch_dtype("float32") is torch.float32

    def test_float16_매핑(self):
        assert _resolve_torch_dtype("float16") is torch.float16

    def test_bfloat16_매핑(self):
        assert _resolve_torch_dtype("bfloat16") is torch.bfloat16


class TestConstructorDtype:
    """생성자가 dtype을 보관하고 기본값은 float32(무회귀)다."""

    def test_기본_dtype은_float32다(self):
        backend = Sam2VideoBackend(repo="fake/repo", device="cpu")
        assert backend._dtype == "float32"

    def test_지정_dtype을_보관한다(self):
        backend = Sam2VideoBackend(repo="fake/repo", device="cuda", dtype="float16")
        assert backend._dtype == "float16"

    def test_생성_시_잘못된_dtype은_ValueError(self):
        with pytest.raises(ValueError, match="지원하지 않는 dtype"):
            Sam2VideoBackend(repo="fake/repo", device="cuda", dtype="fp16")


class TestAutocastContext:
    """_autocast_context가 dtype·device에 따라 no-op/autocast를 고른다."""

    def test_float32는_nullcontext다(self):
        backend = Sam2VideoBackend(repo="fake/repo", device="cuda", dtype="float32")
        ctx = backend._autocast_context(torch)
        assert isinstance(ctx, contextlib.nullcontext)

    def test_cpu는_fp16이라도_nullcontext다(self):
        # cpu autocast는 fp16 미지원 → no-op로 우회(무회귀)
        backend = Sam2VideoBackend(repo="fake/repo", device="cpu", dtype="float16")
        ctx = backend._autocast_context(torch)
        assert isinstance(ctx, contextlib.nullcontext)

    def test_cuda_fp16은_autocast다(self):
        # autocast 객체 생성은 CUDA 부재에서도 가능(enter 시점에만 디바이스 영향)
        backend = Sam2VideoBackend(repo="fake/repo", device="cuda", dtype="float16")
        ctx = backend._autocast_context(torch)
        assert isinstance(ctx, torch.autocast)
