"""마스크 후처리 — 최대 연결성분 추출 순수 로직 (마스크 정제 슬라이스).

배경:
  멀티샷 군무에서 SAM2 마스크가 부정확·과대(대상 배 + 옆사람 팔)했다. box
  프롬프트(detect bbox→SAM2)로 1차 개선하되, 그래도 인접한 다른 사람의 작은
  파편이 마스크에 섞일 수 있다. 해결: 마스크에서 가장 큰 연결성분(largest
  connected component)만 남기고 작은 파편을 제거한다.

설계 경계 불변식:
  순수 core — numpy만. torch·transformers·PySide6·PyAV·scipy 비의존.
  WHY scipy 금지: scipy.ndimage.label은 무거운 의존을 core에 끌어온다.
       largest_component는 numpy + 명시적 BFS flood-fill만으로 구현 가능하다(KISS).
"""
from __future__ import annotations

import numpy as np

# 4-연결 이웃 오프셋 (상, 하, 좌, 우) — 대각선 제외(8-연결 아님)
_NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def largest_component(mask: np.ndarray) -> np.ndarray:
    """bool HxW 마스크에서 가장 큰 4-연결 성분만 남긴다.

    빈 마스크(all False)는 그대로 반환한다(예외 없음). dtype/shape를 보존한다.

    Args:
        mask: bool HxW numpy 배열(True=전경).

    Returns:
        최대 연결성분만 True인 bool HxW 배열(입력과 동일 shape/dtype).

    WHY: 옆사람 팔 같은 인접 파편이 마스크에 섞이면 크롭이 과대해진다.
         가장 큰 연결성분(=대상 본체)만 남겨 마스크 과대를 해결한다.
         4-연결(상하좌우)만 연결로 보아 대각선 파편은 별개 성분으로 분리한다.
    """
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        component = _flood_fill(mask, visited, (int(y), int(x)))
        if len(component) > len(best):
            best = component
    return _component_to_mask(best, mask.shape)


def _flood_fill(
    mask: np.ndarray,
    visited: np.ndarray,
    start: tuple[int, int],
) -> list[tuple[int, int]]:
    """시작 픽셀에서 4-연결 BFS로 한 성분의 좌표 리스트를 수집한다.

    visited를 갱신해 같은 픽셀을 두 번 방문하지 않는다(효율화).
    """
    h, w = mask.shape
    stack = [start]
    visited[start[0], start[1]] = True
    component: list[tuple[int, int]] = []
    while stack:
        y, x = stack.pop()
        component.append((y, x))
        for dy, dx in _NEIGHBOR_OFFSETS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                stack.append((ny, nx))
    return component


def _component_to_mask(
    component: list[tuple[int, int]],
    shape: tuple[int, ...],
) -> np.ndarray:
    """성분 좌표 리스트를 bool 마스크로 복원한다(빈 성분은 전부 False)."""
    result = np.zeros(shape, dtype=bool)
    for y, x in component:
        result[y, x] = True
    return result
