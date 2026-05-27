# easy-capture — 아키텍처

> 최종 업데이트: 2026-05-27 · 관련: [데이터플로우](data-flow.md) · [리소스](resources.md) · [ADR](adr/)

---

## 1. 레이어 구조

```
┌────────────────────────────────────────────────────┐
│ UI (PySide6)                                       │
│  스플래시 · 모드선택 · 메인윈도 · 캔버스 · 타임라인 · 교정 │
└───────────────▲────────────────────────────────────┘
                │ 시그널/슬롯, 진행 콜백
┌───────────────┴────────────────────────────────────┐
│ Application (유스케이스 오케스트레이션)              │
│  ImageCaptureUseCase · GifCaptureUseCase · Session  │
└───────────────▲────────────────────────────────────┘
                │
┌───────────────┴────────────────────────────────────┐
│ Core (도메인 로직, UI/IO 비의존)                     │
│  segmentation · tracking · correction · crop ·      │
│  upscale · export                                   │
└───────────────▲────────────────────────────────────┘
                │
┌───────────────┴────────────────────────────────────┐
│ Infra (외부 의존)                                    │
│  video_io(PyAV/ffprobe) · model_registry · device   │
└────────────────────────────────────────────────────┘
```

의존 방향은 위→아래 단방향(DIP). Core 는 추상 인터페이스에만 의존하고, 구체 모델/디코더는 Infra 가 주입한다.

---

## 2. 모듈 책임

| 모듈 | 책임 |
|---|---|
| `ui/` | 스플래시·모드선택·메인윈도·프레임 캔버스(클릭/박스 드래그)·타임라인(시작/끝 핸들)·교정 모드. 워커 진행률·ETA 표시 |
| `app/usecase` | 모드별 흐름 오케스트레이션, Session 상태 관리 |
| `core/segmentation` | Grounding DINO 검출 + SAM2 image predictor 마스크 (단일 프레임) |
| `core/tracking` | SAM2 video predictor 전파 + occlusion 정책 + **샷 경계 재매칭** |
| `core/correction` | 지정 프레임부터 부분 재추적, 이전 성공 구간 보존·병합 |
| `core/crop` | centroid 산출, N-프레임 이동평균 떨림완화, 경계 클램프, 짝수 정렬, LANCZOS4 리사이즈, 종횡비 잠금 |
| `core/upscale` | Real-ESRGAN / SwinIR 공통 인터페이스, 타일링 |
| `core/export` | PNG/JPG, GIF(팔레트·디더·크기예측), MP4(yuv420p·오디오 mux), 갭 채우기 정책 적용 |
| `infra/video_io` | PyAV 디코드(PTS·VFR 대응), ffprobe 메타, 구간 스트리밍 |
| `infra/model_registry` | 모델 다운로드 매니저(진행 콜백·재시도·캐시 검증), 티어 선택 |
| `infra/device` | CUDA 감지, 모델 티어·해상도 자동 조정, 처리 예상시간 산출 |

---

## 3. 핵심 인터페이스 (추상화)

```python
class Detector(Protocol):              # Grounding DINO 등 교체 가능
    def detect(self, frame, prompt: str) -> list[Detection]: ...

class SegmentationBackend(Protocol):   # SAM2 / 경량모델(이미지 전용) 교체 (ADR 0007)
    device: str
    def segment_image(self, frame, points=None, boxes=None) -> Mask: ...
    def supports_video(self) -> bool: ...          # 경량 백엔드는 False
    def init_video_session(self, frames): ...      # SAM2 등 비디오 지원 시
    def propagate(self, session) -> Iterator[FrameMask]: ...

class Upscaler(Protocol):              # SwinIR(기본) / Real-ESRGAN(옵션) 공통
    def upscale(self, image, scale: int) -> Image: ...

class VideoReader(Protocol):           # PyAV / decord 교체 가능
    def probe(self) -> VideoMeta: ...
    def read_range(self, start, end) -> Iterator[Frame]: ...
```

> 검출기·세그멘테이션 백엔드·업스케일러는 인터페이스로 추상화 → **디바이스(CPU/GPU)·모드(이미지/비디오)** 에 따라 런타임 선택([ADR 0007](adr/0007-cpu-dev-strategy.md)). 이미지 모드는 CPU 백엔드 허용, 비디오 추적은 GPU 백엔드(SAM2). 업스케일 기본 = SwinIR.

---

## 4. 메모리 관리

- **구간 스트리밍**: 전체 영상 디코드 금지. `read_range(start, end)` 로 선택 구간만 PTS seek 후 순차 디코드.
- **버퍼 상한**: 프레임 버퍼 최대 N개 유지. 초과 시 디스크 스풀 또는 재디코드.
- **VRAM 보호**: SAM2/업스케일 처리 전 가용 VRAM 추정 → 부족 시 해상도 자동 다운스케일 → 타일링 → CPU 폴백 순.
- **멤버 순차 처리**: 같은 구간 재선택 시 디코드 캐시 재사용(재디코드 회피).

---

## 5. 동시성

- **추적·업스케일·인코딩**은 `QThread`/워커에서 실행, UI 스레드 비블로킹.
- 워커 → UI 진행률·ETA·중간 미리보기는 **시그널**로 전달.
- 취소(Cancel) 지원: 워커는 협조적 취소 플래그를 폴링.

---

## 6. 확장 대비 (v1.1 Out 항목 수용)

- `Session.tracks` 가 다중 트랙을 이미 보관 → **배치 일괄 처리** 확장 시 UI/오케스트레이션만 추가.
- `core/crop` 의 종횡비 잠금 IF → **SNS 프리셋**은 비율·용량 프로파일 추가로 확장.
- `core/upscale` 공통 IF → **temporal smoothing**은 프레임 시퀀스 입력 변형으로 확장.
