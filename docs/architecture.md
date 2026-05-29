# easy-capture — 아키텍처

> 최종 업데이트: 2026-05-28 · 관련: [데이터플로우](data-flow.md) · [리소스](resources.md) · [ADR](adr/)

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
│  ImageCaptureUseCase · VideoCaptureUseCase · Session │
└───────────────▲────────────────────────────────────┘
                │
┌───────────────┴────────────────────────────────────┐
│ Core (도메인 로직, UI/IO 비의존)                     │
│  segmentation · source · tracking(cut_selection 포함)│
│  crop(mask_refine 포함) · upscale · export · timing  │
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
| `core/segmentation` | **3종 Protocol 추상**: `SegmentationBackend`(이미지 단일 프레임 마스크), `VideoSegmentationBackend`(단일 샷 프레임 전파 추적, ADR 0010), `DetectionBackend`(컷 경계 재검출·무상태, ADR 0012) |
| `core/source` | `FrameSource`·`FrameSpan`·`FrameMeta` Protocol — 이미지/영상 프레임 공급 추상 |
| `core/tracking` | 갭 정책(gap_policy, 기본=BACKGROUND)·재매칭 점수(rematch\_score)·샷 분할(split\_into\_shots)·떨림완화(smooth\_boxes)·**컷별 오브젝트 명시 선택(cut\_selection: `CutSelection`·`index_selections_by_shot`·`validate_selections`, ADR 0006 보강)**. SAM2 video 구현체는 infra에 위치 |
| `core/correction` | **(미구현·대체됨)** 자동 재매칭이 멀티샷에서 구조적 한계를 보여, 별도 부분 재추적 모듈 대신 **컷별 명시 선택(`core/tracking/cut_selection`)**으로 대체했다(ADR 0006 보강). 사용자가 각 컷의 추적 대상을 직접 지정 → 컷별 재추적 |
| `core/crop` | centroid 산출, N-프레임 이동평균 떨림완화, 경계 클램프, 짝수 정렬, LANCZOS4 리사이즈, 종횡비 잠금. **마스크 정제(`mask_refine.largest_component`: 가장 큰 4-연결 성분만 남겨 인접 멤버 파편 제거, numpy 순수·scipy/cv2 비의존, ADR 0014)** |
| `core/upscale` | `UpscaleBackend` Protocol(ADR 0009) + 순수 정규화 함수(`reconstruction_to_rgb_uint8`). SwinIR/Real-ESRGAN 구현체는 infra에 위치 |
| `core/export` | PNG/JPG, GIF(팔레트·디더·크기예측), MP4(yuv420p·오디오 mux), 갭 채우기 정책 적용 |
| `core/timing` | 구간별 가변 재생속도(슬로우/패스트) + 트림(출력 구간 제한) + GIF 루프 순수 로직 — **구현 완료**(`build_playback_schedule`·`schedule_to_cfr_indices`·`clamp_durations_for_gif`·`slice_for_trim`, numpy/stdlib만, ADR 0013) |
| `infra/sam2_image_backend` | `Sam2ImageBackend` — `SegmentationBackend` 구현 (transformers 5.9.0 `Sam2Model`) |
| `infra/sam2_video_backend` | `Sam2VideoBackend` — `VideoSegmentationBackend` 구현 (transformers 5.9.0 `Sam2VideoModel`) |
| `infra/grounding_dino_backend` | `GroundingDinoBackend` — `DetectionBackend` 구현 (transformers 5.9.0) |
| `infra/swin2sr_upscale_backend` | `Swin2srUpscaleBackend` — `UpscaleBackend` 구현 (transformers 5.9.0) |
| `infra/video_io` | PyAV 디코드(PTS·VFR 대응), ffprobe 메타, 구간 스트리밍. `FrameSource` Protocol 구현체 포함 |
| `infra/model_registry` | 모델 다운로드 매니저(진행 콜백·재시도·캐시 검증), 티어 선택 |
| `infra/device` | CUDA 감지, 모델 티어·해상도 자동 조정, 처리 예상시간 산출 |

---

## 3. 핵심 인터페이스 (추상화)

ADR 0010·0012에 따라 세그멘테이션 책임은 3종 Protocol로 분리되어 있다. 이미지 전용·비디오 추적·컷 재검출은 각각 독립 인터페이스다.

```python
# core/segmentation/backend.py  — 이미지 단일 프레임 마스크 (ADR 0007·0008)
class SegmentationBackend(Protocol):
    device: str
    def segment_image(self, frame, points=None, boxes=None) -> Mask: ...
    def supports_video(self) -> bool: ...   # 이미지 전용 백엔드는 False

# core/segmentation/video_backend.py  — 단일 샷 프레임 시퀀스 전파 (ADR 0010·0014)
class VideoSegmentationBackend(Protocol):
    device: str
    def init_session(self, frames: list[np.ndarray]) -> object: ...
    def add_click(self, session: object, point: tuple[int, int]) -> None: ...
    def add_box(self, session: object, box: tuple[float, float, float, float]) -> None: ...
    # box 프롬프트(detect 전신 bbox→SAM2): add_click보다 정확한 전신 마스크(ADR 0014)
    # add_click은 무변경 유지(단일샷 폴백·무회귀)
    def propagate(self, session: object) -> list[np.ndarray]: ...
    # 구현체: infra/sam2_video_backend (transformers 5.9.0 Sam2VideoModel)

# core/segmentation/detection_backend.py  — 컷 경계 재검출, 무상태 (ADR 0012)
class DetectionBackend(Protocol):
    device: str
    def detect(self, frame: np.ndarray, prompt: str) -> list[Detection]: ...
    # 구현체: infra/grounding_dino_backend (transformers 5.9.0)

# core/upscale/backend.py  — 업스케일 공통 인터페이스 (ADR 0009)
class UpscaleBackend(Protocol):
    def upscale(self, image: np.ndarray) -> np.ndarray: ...
    # 구현체: infra/swin2sr_upscale_backend (기본), infra/realesrgan_backend (옵션)

# core/source/frame_source.py  — 이미지/영상 프레임 공급 추상 (ADR 0008)
class FrameSource(Protocol):
    def read_frame(self) -> FrameMeta: ...
    # FrameSpan: 구간 지정. FrameMeta: 프레임+메타.
    # 구현체: infra/video_io (Pillow·PyAV)
```

> 세그멘테이션·검출·업스케일·프레임소스는 인터페이스로 추상화 → **디바이스(CPU/GPU)·모드(이미지/비디오)** 에 따라 런타임 선택([ADR 0007](adr/0007-cpu-dev-strategy.md)·[ADR 0010](adr/0010-video-segmentation-backend.md)·[ADR 0012](adr/0012-detection-backend.md)). 이미지 모드는 CPU 백엔드 허용, 비디오 추적은 GPU 백엔드 필요. 업스케일 기본 = SwinIR/Swin2SR([ADR 0009](adr/0009-upscale-export-integration.md)).

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
