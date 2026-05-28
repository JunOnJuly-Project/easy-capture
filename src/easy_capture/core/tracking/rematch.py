"""샷 경계 재매칭 점수 (ADR 0006).

score = w_pos·IoU(직전 bbox, 후보 bbox) + w_cls·외형유사도.
외형 특징이 없으면 위치(IoU)만으로 평가한다.

신규:
  REMATCH_THRESHOLD — 재매칭 통과 임계값(ADR 0006).
  RematchResult     — 재매칭 1회 판정 결과(불변 dataclass).
  select_best_match — 후보 리스트 → best 선택 + threshold 통과 판정(순수).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2)

# 재매칭 통과 임계값 (ADR 0006 — w_pos=0.7·w_cls=0.3 기준, 위치 IoU 기반)
# WHY: 0.5는 보수적 시작값. Colab 실측 후 오탐/미탐 비율 확인 시 보정 예정(ADR 0006 갱신 대기).
REMATCH_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class RematchResult:
    """재매칭 1회 판정 결과(불변).

    best_index: 최고 점수 후보의 인덱스(후보 없으면 -1).
    score:      최고 점수(후보 없으면 0.0).
    passed:     score >= threshold 여부(=동일인 재매칭 성공).

    WHY: frozen=True는 판정 결과가 실수로 덮어씌워지는 버그를 방지한다.
         TrackResult·Detection 패턴 계승.
    """

    best_index: int
    score: float
    passed: bool


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


def select_best_match(
    prev_box: Box,
    candidates: list,
    threshold: float = REMATCH_THRESHOLD,
) -> RematchResult:
    """후보 리스트에서 직전 bbox와 best 매칭 후보·점수·통과여부를 판정한다(순수).

    각 후보에 rematch_score(prev_box, cand.box, prev_feat?, cand.feat)를 적용해
    최댓값 후보를 고르고 threshold로 통과 여부를 결정한다.
    후보가 비면 RematchResult(best_index=-1, score=0.0, passed=False).

    Args:
        prev_box:   직전 샷 마지막 유효 마스크에서 추출한 bbox.
        candidates: Detection 리스트(box + score + feat). 빈 리스트 허용.
        threshold:  통과 기준(기본 REMATCH_THRESHOLD=0.5).

    WHY: 순수 함수 — backend·IO·torch 미의존. GPU 모델과 완전 독립.
         후보 리스트와 박스만 입력 → 결정적 출력. 단위 테스트로 100% 검증 가능.
    """
    if not candidates:
        return RematchResult(best_index=-1, score=0.0, passed=False)

    scores = [
        rematch_score(prev_box, cand.box, None, cand.feat)
        for cand in candidates
    ]
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    return RematchResult(
        best_index=best_idx,
        score=best_score,
        passed=best_score >= threshold,
    )
