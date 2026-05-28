"""크롭 기하 로직: centroid 산출, 떨림 완화, 종횡비 잠금, 경계 클램프, 짝수 정렬.

순수 함수만 둔다(모델·UI·IO 비의존)로 테스트가 쉽다.
"""
from __future__ import annotations

import numpy as np

# 종횡비 프리셋 (가로:세로)
ASPECT_PRESETS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1), "9:16": (9, 16), "16:9": (16, 9),
}


def centroid_of_mask(mask) -> tuple[float, float] | None:
    """불리언/0-1 마스크의 무게중심 (cx, cy). 빈 마스크면 None."""
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def bbox_of_mask(mask) -> tuple[float, float, float, float] | None:
    """불리언/0-1 마스크의 외접 bbox (x1, y1, x2, y2). 빈 마스크면 None.

    centroid_of_mask와 완전 대칭 — 같은 파일·같은 패턴.
    WHY: SAM2 video는 마스크만 반환하고 rematch_score는 bbox를 요구한다.
         이 헬퍼가 직전 샷 마지막 마스크 → prev_box 변환의 누락 연결고리를 채운다.
         (계획서 §3-3)
    """
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _hold_forward(points: list):
    """None 값을 직전 유효 좌표로 채운다(occlusion 시 마지막 위치 홀드)."""
    out, last = [], None
    for p in points:
        last = p if p is not None else last
        out.append(last)
    return out


def smooth_centroids(points: list, window: int = 5) -> list:
    """N-프레임 이동평균으로 centroid 떨림을 완화한다."""
    if window < 1:
        raise ValueError("window 는 1 이상이어야 한다")
    filled = _hold_forward(points)
    out = []
    for i in range(len(filled)):
        seg = [p for p in filled[max(0, i - window + 1): i + 1] if p is not None]
        if not seg:
            out.append(None)
            continue
        cx = sum(p[0] for p in seg) / len(seg)
        cy = sum(p[1] for p in seg) / len(seg)
        out.append((cx, cy))
    return out


def apply_aspect_lock(w: int, h: int, aspect) -> tuple[int, int]:
    """요청 (w,h) 를 종횡비에 맞춰 박스 안쪽으로 축소. aspect None 이면 그대로."""
    if aspect is None:
        return w, h
    aw, ah = ASPECT_PRESETS[aspect] if isinstance(aspect, str) else aspect
    if w * ah > h * aw:        # 현재가 목표보다 가로로 넓음 → 가로 축소
        w = round(h * aw / ah)
    else:                       # 세로로 김 → 세로 축소
        h = round(w * ah / aw)
    return int(w), int(h)


def to_even(v) -> int:
    """yuv420p 인코딩을 위해 짝수로 내림."""
    n = int(round(v))
    return n - (n % 2)


def make_crop_box(center, size, frame_size) -> tuple[int, int, int, int]:
    """center(cx,cy) 중심 size(w,h) 박스를 프레임 경계 안으로 클램프 + 짝수 정렬.

    반환: (x1, y1, x2, y2). w/h 는 짝수이며 프레임을 넘지 않는다.
    """
    cx, cy = center
    fw, fh = frame_size
    w = min(to_even(size[0]), to_even(fw))
    h = min(to_even(size[1]), to_even(fh))
    x1 = max(0, min(int(round(cx - w / 2)), fw - w))
    y1 = max(0, min(int(round(cy - h / 2)), fh - h))
    return x1, y1, x1 + w, y1 + h
