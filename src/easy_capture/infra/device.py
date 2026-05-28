"""디바이스 자동 감지 및 SAM2 티어 선택 (ADR 0007).

비디오 추적은 GPU 전제. CPU 는 이미지 모드 위주(단발/초단편 경고).
업스케일 모델 카탈로그(UPSCALE_MODELS)도 이 모듈에 둔다(ADR 0009).
"""
from __future__ import annotations

from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# 업스케일 모델 카탈로그 (ADR 0004 / ADR 0009)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UpscaleModel:
    """업스케일 모델 카탈로그 1항목. UI 라벨·repo·고정 배율을 묶는다.

    WHY: UI 라벨·repo·scale을 단일 소스(이 튜플)에서 관리해 DRY 준수.
         매직 문자열 중복 방지. Real-ESRGAN 추가 시 이 튜플만 수정(OCP).
    """

    label: str   # UI 표시 ("x2 (범용·선명)")
    repo: str    # HuggingFace repo id
    scale: int   # 고정 배율


# 업스케일 모델 카탈로그 — UI 콤보 항목의 단일 소스
# WHY: x2/x4 외 추가는 이 튜플에만 항목을 더한다(OCP).
UPSCALE_MODELS: tuple[UpscaleModel, ...] = (
    UpscaleModel("x2 (범용·선명)", "caidas/swin2SR-classical-sr-x2-64", 2),
    UpscaleModel("x4 (실사·강한 확대)", "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr", 4),
)
