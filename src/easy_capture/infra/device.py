"""디바이스 자동 감지 및 SAM2 티어 선택 (ADR 0007).

비디오 추적은 GPU 전제. CPU 는 이미지 모드 위주(단발/초단편 경고).
"""
from __future__ import annotations

# 디바이스별 기본 SAM2 가중치 (CPU 는 가장 가벼운 tiny)
SAM2_REPO_BY_DEVICE = {
    "cuda": "facebook/sam2.1-hiera-base-plus",
    "cpu": "facebook/sam2.1-hiera-tiny",
}


def detect_device() -> str:
    """CUDA 가용 시 'cuda', 아니면 'cpu'. torch 미설치 시에도 안전하게 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch 부재/로드 실패 시 CPU 폴백
        return "cpu"


def select_sam2_repo(device: str) -> str:
    """디바이스에 맞는 SAM2 repo id."""
    return SAM2_REPO_BY_DEVICE.get(device, SAM2_REPO_BY_DEVICE["cpu"])


def supports_video_tracking(device: str) -> bool:
    """비디오 추적 실용성 여부. CPU 는 0.1fps 수준이라 비실용(경고 대상)."""
    return device == "cuda"
