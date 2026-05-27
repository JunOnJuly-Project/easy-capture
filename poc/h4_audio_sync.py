"""H4: PyAV 메타 프로브 + 크롭 구간 MP4(오디오 동기) mux 검증.

- PyAV 로 영상 메타(fps, 프레임수, VFR 추정, 오디오 유무, 길이) 확인.
- ffmpeg 으로 [start,end] 구간을 짝수 해상도 크롭 + yuv420p + 원본 오디오 동기 출력.
- 출력 재프로브로 video/audio 길이 차(동기 프록시)를 측정.
"""
import argparse
import subprocess
from pathlib import Path


def probe(video_path: str) -> dict:
    import av
    container = av.open(video_path)
    v = next((s for s in container.streams if s.type == "video"), None)
    a = next((s for s in container.streams if s.type == "audio"), None)
    avg = float(v.average_rate) if v and v.average_rate else None
    base = float(v.base_rate) if v and getattr(v, "base_rate", None) else None
    info = {
        "fps_avg": avg,
        "fps_base": base,
        "vfr_suspected": (avg is not None and base is not None and abs(avg - base) > 0.01),
        "video_frames": v.frames if v else 0,
        "video_duration_s": float(v.duration * v.time_base) if v and v.duration else None,
        "has_audio": a is not None,
        "audio_duration_s": float(a.duration * a.time_base) if a and a.duration else None,
    }
    container.close()
    return info


def export_clip(src: str, dst: str, start: float, end: float, crop_w: int, crop_h: int) -> None:
    """ffmpeg: 구간 크롭(짝수 정렬) + yuv420p + 오디오 동기 mux."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cw, ch = crop_w - (crop_w % 2), crop_h - (crop_h % 2)  # 짝수 강제
    vf = f"crop={cw}:{ch}:0:0,format=yuv420p"
    cmd = [ff, "-y", "-ss", str(start), "-to", str(end), "-i", src,
           "-vf", vf, "-c:v", "libx264", "-c:a", "aac",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test-data/synthetic.mp4")
    ap.add_argument("--out", default="test-data/clip_out.mp4")
    a = ap.parse_args()
    if not Path(a.video).exists():
        raise SystemExit(f"영상 없음: {a.video}")

    src_info = probe(a.video)
    print("입력 메타:", src_info)

    export_clip(a.video, a.out, start=1.0, end=4.0, crop_w=300, crop_h=300)
    out_info = probe(a.out)
    print("출력 메타:", out_info)

    if out_info["has_audio"] and out_info["audio_duration_s"] and out_info["video_duration_s"]:
        drift = abs(out_info["audio_duration_s"] - out_info["video_duration_s"])
        print(f"video/audio 길이차(동기 프록시): {drift*1000:.1f} ms")
    else:
        print("출력에 오디오 없음 — 무음 폴백 경로 점검 필요")


if __name__ == "__main__":
    main()
