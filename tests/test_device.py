"""디바이스 감지/티어 선택 테스트."""
from easy_capture.infra import (detect_device, select_sam2_dtype,
                                select_sam2_repo, supports_video_tracking)


def test_detect_device_returns_valid():
    assert detect_device() in ("cpu", "cuda")


def test_select_dtype_cuda_is_fp16():
    # Colab T4 측정: fp16이 AC-01 100% 유지 + 2.67× (ADR 0017)
    assert select_sam2_dtype("cuda") == "float16"


def test_select_dtype_cpu_is_fp32():
    # cpu는 autocast 미지원 → 안전 폴백 fp32
    assert select_sam2_dtype("cpu") == "float32"


def test_select_dtype_unknown_falls_back_fp32():
    assert select_sam2_dtype("mps") == "float32"


def test_select_repo_cpu_is_tiny():
    assert "tiny" in select_sam2_repo("cpu")


def test_select_repo_differs_by_device():
    assert select_sam2_repo("cuda") != select_sam2_repo("cpu")


def test_supports_video_only_on_gpu():
    assert supports_video_tracking("cuda") is True
    assert supports_video_tracking("cpu") is False
