"""디바이스 감지/티어 선택 테스트."""
from easy_capture.infra import (detect_device, select_sam2_repo,
                                supports_video_tracking)


def test_detect_device_returns_valid():
    assert detect_device() in ("cpu", "cuda")


def test_select_repo_cpu_is_tiny():
    assert "tiny" in select_sam2_repo("cpu")


def test_select_repo_differs_by_device():
    assert select_sam2_repo("cuda") != select_sam2_repo("cpu")


def test_supports_video_only_on_gpu():
    assert supports_video_tracking("cuda") is True
    assert supports_video_tracking("cpu") is False
