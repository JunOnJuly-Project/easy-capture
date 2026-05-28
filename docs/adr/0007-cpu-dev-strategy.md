# ADR 0007 — CPU 개발 전략 + 모델 백엔드 추상화 (이중 경로)

- 상태: 채택
- 날짜: 2026-05-27

## 맥락
개발 PC 가 **CPU 전용**인데 PoC 실측상 SAM2 비디오 추적이 CPU 에서 **0.10 fps**(프레임당 ~10초)로 비실용적이다([poc/REPORT.md](../../poc/REPORT.md)). 조사 결과:
- 경량 promptable 모델(EdgeSAM·MobileSAM·FastSAM·RepViT-SAM)은 CPU 이미지 추론이 빠르지만(수십 ms) **모두 이미지 전용 — 비디오 프레임 전파(메모리) 미지원**.
- SAM2 의 ONNX/INT8 최적화는 3~4× 향상에 그쳐 CPU 비디오는 여전히 비실용(~2.5s/frame).
- EdgeTAM(Meta, CVPR 2025)은 온디바이스 비디오 추적 후보이나 매우 신규·라이선스 미확정.

## 결정
**이중 경로(Dual-Path)** + **디바이스/모드 기반 백엔드 선택**으로 개발이 막히지 않게 한다.

1. **이미지(짤) 모드 — CPU 개발 가능**: SAM2 image predictor(CPU ~1~3s/장, 미리보기 수준 허용). 더 빠른 반복이 필요하면 경량 백엔드(MobileSAM/EdgeSAM)를 **선택적 CPU 백엔드**로 추가.
2. **GIF(움짤)·비디오 모드 — GPU 전제**: 로컬 CUDA 또는 클라우드 GPU(Colab/RunPod/Vast.ai). 로컬 검증은 `poc/colab/` 노트북으로.
3. **모델 백엔드 추상화**: `SegmentationBackend` 인터페이스로 SAM2/경량모델을 디바이스·모드에 따라 런타임 교체. 비디오 메서드는 선택(Optional)로 두어 이미지 전용 백엔드도 수용.

```python
class SegmentationBackend(Protocol):
    device: str
    def segment_image(self, image, points=None, boxes=None) -> Mask: ...
    # 비디오 추적은 지원 백엔드만(미지원 시 NotSupported)
    def supports_video(self) -> bool: ...
    def init_video_session(self, frames): ...      # SAM2 등
    def propagate(self, session): ...
```

## 대안
- SAM2 자체 경량화/재학습: ROI 낮음(여전히 CPU 비디오 비실용).
- EdgeTAM 도입: 유망하나 신규·라이선스 미확정 → v1.1 평가 항목.
- 전면 클라우드 개발: 가능하나(월 $75~100) 비용·네트워크 지연.

## 결과
- MVP 개발: **이미지 모드+UI 는 CPU 로컬, 비디오 모드는 클라우드 GPU 검증**의 이중 경로.
- `architecture.md` 의 모델 추상화 IF 를 `SegmentationBackend`(비디오 Optional)로 구체화.
- 디바이스 자동 감지 결과로 백엔드·모델 티어 선택([infra/device]).
- EdgeTAM·경량 CPU 비디오 추적은 v1.1 후보로 백로그.

---

## 보완 (2026-05-28, 구현 확정)

본 ADR의 원안(`§결정 3`)에서 단일 `SegmentationBackend`에 비디오 메서드를 Optional로 두는 설계를 제시했다. 구현 단계에서 이 설계의 두 가지 문제점이 드러났다.

- **ISP 위반**: 이미지 전용 백엔드(`Sam2ImageBackend`, 경량 MobileSAM 등)가 사용하지 않는 비디오 메서드를 빈 몸통으로 구현해야 한다.
- **호출자 분기 복잡도**: `supports_video()` 체크와 `hasattr` 분기가 유스케이스 곳곳에 분산된다.

이에 따라 원안 설계는 두 후속 ADR로 **구체화·대체**되었다.

| 후속 ADR | 내용 |
|---|---|
| [ADR 0010](0010-video-segmentation-backend.md) | `VideoSegmentationBackend`를 독립 Protocol로 분리 (ISP 준수, opaque session) |
| [ADR 0012](0012-detection-backend.md) | `DetectionBackend`를 독립 Protocol로 분리 (컷 재검출, 무상태) |

`SegmentationBackend`는 이미지 단일 프레임 마스크 전용으로 유지되며, `supports_video()` 플래그는 "이미지 백엔드의 비디오 미지원 표식"으로만 보존된다. 비디오 유스케이스(`VideoCaptureUseCase`)는 `VideoSegmentationBackend`만 주입받는다.
