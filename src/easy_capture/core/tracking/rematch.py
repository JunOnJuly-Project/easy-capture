"""샷 경계 재매칭 점수 (ADR 0006).

score = w_pos·IoU(직전 bbox, 후보 bbox) + w_cls·외형유사도.
외형 특징이 없으면 위치(IoU)만으로 평가한다.
"""
from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2)


def iou(a: Box, b: Box) -> float:
    """두 bbox 의 IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _cosine(u, v) -> float:
    u, v = np.asarray(u, float), np.asarray(v, float)
    denom = float(np.linalg.norm(u) * np.linalg.norm(v)) or 1.0
    return float(np.dot(u, v) / denom)


def rematch_score(prev_box: Box, cand_box: Box, prev_feat=None, cand_feat=None,
                  w_pos: float = 0.7, w_cls: float = 0.3) -> float:
    """동일인 재매칭 점수. 특징이 없으면 위치(IoU)만 사용."""
    pos = iou(prev_box, cand_box)
    if prev_feat is None or cand_feat is None:
        return pos
    return w_pos * pos + w_cls * _cosine(prev_feat, cand_feat)
