"""프레임 인덱스 ↔ mm:ss 시간 표시 변환 (순수, PySide6 비의존).

타임라인/구간 편집 UI에서 프레임 번호 옆에 사람이 읽기 쉬운 mm:ss를 보여준다.
fps 기반 변환이며 표시는 초 단위 해상도다(프레임 정밀도는 SpinBox가 담당).

WHY 순수 분리: ui/coords·ui/sizing 패턴 계승. Qt 없이 단위 테스트 가능.
"""
from __future__ import annotations

_SEC_PER_MIN = 60
_FALLBACK = "00:00"


def frame_to_mm_ss(frame_index: int, fps: float) -> str:
    """프레임 인덱스를 "mm:ss" 문자열로 변환한다(fps<=0·음수면 00:00 폴백).

    Args:
        frame_index: 0-기반 프레임 인덱스.
        fps: 초당 프레임 수.

    Returns:
        "mm:ss" (예: 90프레임·30fps → "00:03").
    """
    if fps <= 0 or frame_index < 0:
        return _FALLBACK
    total_sec = int(frame_index / fps)
    return f"{total_sec // _SEC_PER_MIN:02d}:{total_sec % _SEC_PER_MIN:02d}"


def mm_ss_to_frame(text: str, fps: float) -> int:
    """"mm:ss"(또는 "ss") 문자열을 프레임 인덱스로 변환한다(반올림).

    Args:
        text: "1:30" 또는 "90"(초) 형식.
        fps: 초당 프레임 수.

    Returns:
        반올림된 프레임 인덱스.

    Raises:
        ValueError: 형식 오류·음수·초 60 이상·fps<=0.
    """
    if fps <= 0:
        raise ValueError(f"fps는 0보다 커야 합니다: {fps}")
    return round(_parse_to_seconds(text) * fps)


def _parse_to_seconds(text: str) -> int:
    """"mm:ss"/"ss" 문자열을 총 초로 파싱한다(형식·범위 검증).

    "ss" 단독은 임의 초(예 "90"=90초), "mm:ss"는 초가 0~59여야 한다.
    """
    try:
        nums = [int(p) for p in text.strip().split(":")]
    except ValueError:
        raise ValueError(f"mm:ss 형식이 아닙니다: {text!r}")
    if len(nums) == 1:
        return _checked_single(nums[0], text)
    if len(nums) == 2:
        return _checked_min_sec(nums[0], nums[1], text)
    raise ValueError(f"mm:ss 형식이 아닙니다: {text!r}")


def _checked_single(seconds: int, text: str) -> int:
    """단독 초("ss") 검증 — 음수만 거부."""
    if seconds < 0:
        raise ValueError(f"음수 시간: {text!r}")
    return seconds


def _checked_min_sec(minutes: int, seconds: int, text: str) -> int:
    """"mm:ss" 검증 — 분 음수 금지·초 0~59."""
    if minutes < 0 or not 0 <= seconds < _SEC_PER_MIN:
        raise ValueError(f"mm:ss 범위 오류(초는 0~59): {text!r}")
    return minutes * _SEC_PER_MIN + seconds
