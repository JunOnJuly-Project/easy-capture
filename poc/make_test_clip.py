"""합성 테스트 클립 생성: 이동 도형 + 하드 컷 + 일시 소실(+오디오 톤).

외부 영상 없이 SAM2 포인트 추적·PySceneDetect 컷 감지·오디오 mux 를
기계적으로 검증하기 위한 하니스. 실제 군무/인물 identity 검증은
사용자 제공 실영상으로 별도 수행한다.
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


def build_frames(w: int, h: int, fps: int, seconds: int):
    """파랑 원(좌→우), 빨강 원(우→좌, 일시 소실), 중간 하드 컷 프레임 생성."""
    n = fps * seconds
    cut = n // 2
    occ_start, occ_end = int(n * 0.70), int(n * 0.80)
    frames = []
    for i in range(n):
        bg = (30, 30, 30) if i < cut else (90, 60, 20)  # 컷 전/후 배경 급변
        img = np.full((h, w, 3), bg, np.uint8)
        bx = int(w * (0.1 + 0.8 * (i / n)))
        cv2.circle(img, (bx, h // 2), 40, (220, 120, 0), -1)   # 파랑(BGR)
        if not (occ_start <= i < occ_end):
            rx = int(w * (0.9 - 0.8 * (i / n)))
            cv2.circle(img, (rx, h // 3), 35, (0, 0, 220), -1)  # 빨강(BGR)
        frames.append(img)
    return frames, cut, (occ_start, occ_end)


def write_video(path: Path, frames, fps: int) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def add_audio(src: Path, dst: Path, seconds: int) -> None:
    """imageio-ffmpeg 번들 ffmpeg 으로 440Hz 사인 톤을 mux (H4 검증용)."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-i", str(src),
           "-f", "lavfi", "-t", str(seconds), "-i", "sine=frequency=440:sample_rate=44100",
           "-c:v", "copy", "-c:a", "aac", "-shortest", str(dst)]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test-data/synthetic.mp4")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=360)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seconds", type=int, default=6)
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames, cut, occ = build_frames(a.w, a.h, a.fps, a.seconds)
    tmp = out.with_suffix(".noaudio.mp4")
    write_video(tmp, frames, a.fps)
    if a.no_audio:
        tmp.replace(out)
    else:
        add_audio(tmp, out, a.seconds)
        tmp.unlink(missing_ok=True)
    print(f"wrote {out} | frames={len(frames)} cut@{cut} occ={occ} fps={a.fps}")


if __name__ == "__main__":
    main()
