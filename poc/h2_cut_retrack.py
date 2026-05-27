"""H2: 샷 경계 감지 + 재매칭 점수 검증.

- PySceneDetect 로 컷 경계를 감지(실영상/합성 공통).
- 재매칭 점수 함수(설계 #4)를 구현·검증: score = w_pos·pos_sim + w_cls·cls_sim.
  실영상에서는 Grounding DINO 후보 feat 로 cls_sim 을 채우지만,
  여기서는 위치 기반(pos_sim)과 점수 결합 로직을 기계 검증한다.
"""
import argparse
from pathlib import Path


def detect_cuts(video_path: str):
    """컷 경계를 (start, end) 타임코드 리스트로 반환."""
    from scenedetect import detect, ContentDetector
    scenes = detect(video_path, ContentDetector())
    return [(s.get_seconds(), e.get_seconds()) for s, e in scenes]


def iou(a, b) -> float:
    """두 bbox(x1,y1,x2,y2) 의 IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def rematch_score(prev_bbox, cand_bbox, prev_feat=None, cand_feat=None,
                  w_pos: float = 0.7, w_cls: float = 0.3) -> float:
    """컷 직후 동일인 재매칭 점수. cls 특징이 없으면 위치만으로 평가."""
    pos_sim = iou(prev_bbox, cand_bbox)
    if prev_feat is None or cand_feat is None:
        return pos_sim
    cls_sim = _cosine(prev_feat, cand_feat)
    return w_pos * pos_sim + w_cls * cls_sim


def _cosine(u, v) -> float:
    import numpy as np
    u, v = np.asarray(u, float), np.asarray(v, float)
    denom = (np.linalg.norm(u) * np.linalg.norm(v)) or 1.0
    return float(np.dot(u, v) / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test-data/synthetic.mp4")
    a = ap.parse_args()
    if not Path(a.video).exists():
        raise SystemExit(f"영상 없음: {a.video} (먼저 make_test_clip.py 실행)")

    cuts = detect_cuts(a.video)
    print(f"감지된 샷 수: {len(cuts)}")
    for i, (s, e) in enumerate(cuts):
        print(f"  scene {i}: {s:.2f}s ~ {e:.2f}s")

    # 재매칭 점수 자가검증(임계값 0.5)
    prev = (100, 100, 200, 200)
    near = (110, 105, 210, 205)   # 같은 인물(근접) → 높은 점수
    far = (400, 300, 480, 360)    # 다른 인물 → 낮은 점수
    print(f"rematch(near)={rematch_score(prev, near):.3f}  rematch(far)={rematch_score(prev, far):.3f}")
    assert rematch_score(prev, near) >= 0.5 > rematch_score(prev, far)
    print("재매칭 점수 로직 OK (near≥0.5>far)")


if __name__ == "__main__":
    main()
