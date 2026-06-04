"""디바이스 자동 감지 및 SAM2 티어 선택 (ADR 0007).

비디오 추적은 GPU 전제. CPU 는 이미지 모드 위주(단발/초단편 경고).
업스케일 모델 카탈로그(UPSCALE_MODELS)도 이 모듈에 둔다(ADR 0009).
"""
from __future__ import annotations

from dataclasses import dataclass

# 디바이스별 기본 SAM2 가중치
# WHY cuda=small: 멀티샷 군무 GPU 게이트(AC-01 100%·needs_correction 0)와 fp16 측정
#   (ADR 0017)이 모두 hiera-small로 통과했다. base-plus는 더 무거운데(VRAM·속도)
#   게이트 미검증이라, 검증된 small을 기본으로 정합화한다. CPU는 가장 가벼운 tiny.
SAM2_REPO_BY_DEVICE = {
    "cuda": "facebook/sam2.1-hiera-small",
    "cpu": "facebook/sam2.1-hiera-tiny",
}

# 디바이스별 추론 정밀도 (ADR 0017) — cuda 는 fp16(Colab T4 측정 AC-06 2.67×·
# AC-01 100%·needs_correction 0 유지), cpu 는 fp32(autocast 미지원·무의미).
# bf16 은 T4 하드웨어 가속이 없어 채택 안 함(측정 0.95×). 필요 시 백엔드 인자로 주입.
SAM2_DTYPE_BY_DEVICE = {
    "cuda": "float16",
    "cpu": "float32",
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


def select_sam2_dtype(device: str) -> str:
    """디바이스에 맞는 SAM2 추론 정밀도 문자열(ADR 0017).

    cuda='float16'(fp16 가속), 그 외='float32'(안전 폴백, autocast no-op).
    """
    return SAM2_DTYPE_BY_DEVICE.get(device, "float32")


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
