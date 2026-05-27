"""환경 프로브: torch/디바이스와 transformers 모델 클래스 가용성 확인 (H1 사전점검).

설치 직후 실행해 SAM2/Grounding DINO 가 현재 transformers 버전에서
어떤 이름으로 노출되는지 파악한다.
"""
import importlib


def check_torch() -> None:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"cpu threads: {torch.get_num_threads()}")


def _report_attrs(mod, names: list[str]) -> None:
    for n in names:
        mark = "OK" if hasattr(mod, n) else "--"
        print(f"  [{mark}] transformers.{n}")


def check_transformers() -> None:
    import transformers
    print(f"transformers: {transformers.__version__}")
    _report_attrs(transformers, [
        "Sam2Model", "Sam2Processor",
        "Sam2VideoModel", "Sam2VideoProcessor",
        "GroundingDinoForObjectDetection",
        "AutoProcessor", "AutoModel",
    ])


def check_others() -> None:
    for m in ["scenedetect", "av", "cv2", "PIL", "numpy", "imageio", "imageio_ffmpeg"]:
        try:
            mod = importlib.import_module(m)
            print(f"  [OK] {m} {getattr(mod, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001 - 프로브이므로 광범위 캐치 의도
            print(f"  [FAIL] {m}: {e}")


if __name__ == "__main__":
    print("=== torch ===");        check_torch()
    print("=== transformers ==="); check_transformers()
    print("=== others ===");       check_others()
