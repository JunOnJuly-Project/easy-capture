"""H1: SAM2 video predictor 클릭 추적 검증 (transformers Sam2Video).

클릭 포인트로 0번 프레임에서 객체를 잡고 후속 프레임에 전파하여,
설치·동작 여부와 CPU 추적 속도(fps)·추적 유지율을 측정한다.
"""
import argparse
import time

import cv2
import numpy as np
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor


def load_frames_rgb(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def mask_area(post_masks) -> int:
    """post_process_masks 결과(텐서/배열)에서 양성 픽셀 수."""
    arr = post_masks.cpu().numpy() if hasattr(post_masks, "cpu") else np.asarray(post_masks)
    return int((arr > 0).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test-data/synthetic.mp4")
    ap.add_argument("--model", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--x", type=int, default=None)
    ap.add_argument("--y", type=int, default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_frames_rgb(a.video)
    n = len(frames)
    h, w = frames[0].shape[:2]
    x = a.x if a.x is not None else int(0.1 * w)   # 합성 클립: 파랑 원 시작 위치
    y = a.y if a.y is not None else h // 2
    print(f"frames={n} size={w}x{h} click=({x},{y}) device={device}")

    t0 = time.time()
    processor = Sam2VideoProcessor.from_pretrained(a.model)
    model = Sam2VideoModel.from_pretrained(a.model).to(device).eval()
    print(f"모델 로드 {time.time() - t0:.1f}s")

    session = processor.init_video_session(video=frames, inference_device=device, dtype=torch.float32)
    processor.add_inputs_to_inference_session(
        session, frame_idx=0, obj_ids=1,
        input_points=[[[[x, y]]]], input_labels=[[[1]]],
    )

    t1 = time.time()
    tracked, areas = 0, []
    with torch.inference_mode():
        for i, out in enumerate(model.propagate_in_video_iterator(session, start_frame_idx=0)):
            if i == 0:
                print("pred_masks shape:", tuple(out.pred_masks.shape))
            post = processor.post_process_masks([out.pred_masks], original_sizes=[(h, w)])[0]
            area = mask_area(post)
            areas.append(area)
            tracked += area > 0
    dt = time.time() - t1
    print(f"추적 유지: {tracked}/{n} 프레임 ({100*tracked/n:.0f}%)")
    print(f"전파 속도: {n/dt:.2f} fps (CPU) · 총 {dt:.1f}s")
    print(f"마스크 면적 min/max: {min(areas)}/{max(areas)}")


if __name__ == "__main__":
    main()
