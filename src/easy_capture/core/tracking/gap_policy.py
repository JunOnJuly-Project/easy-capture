"""occlusion 갭 채우기 정책: 프레임별 추적 유효 여부를 출력 프레임 인덱스로 변환.

- CUT: 갭 프레임 제외(시간 점프) — 오브젝트 항상 보임.
- BACKGROUND: 전 프레임 출력, 갭 동안 크롭 위치만 고정(장면 계속).
- FREEZE: 갭을 마지막 유효 프레임으로 대체(정지).
"""
from __future__ import annotations

from enum import Enum


class GapPolicy(str, Enum):
    CUT = "cut"
    BACKGROUND = "background"
    FREEZE = "freeze"


def build_output_indices(valid_flags: list[bool], policy: GapPolicy) -> list[int]:
    """추적 유효 플래그 리스트 → 출력할 원본 프레임 인덱스 리스트."""
    if policy is GapPolicy.BACKGROUND:
        return list(range(len(valid_flags)))
    if policy is GapPolicy.CUT:
        return [i for i, v in enumerate(valid_flags) if v]
    if policy is GapPolicy.FREEZE:
        out: list[int] = []
        last: int | None = None
        for i, v in enumerate(valid_flags):
            if v:
                last = i
                out.append(i)
            elif last is not None:
                out.append(last)
        return out
    raise ValueError(f"알 수 없는 정책: {policy}")
