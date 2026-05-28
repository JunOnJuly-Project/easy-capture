# 비디오(움짤) 모드 첫 수직 슬라이스 — 개발 계획

> 작성: 2026-05-28 · 단계: `/develop` 파이프라인 planner · 관련: [아키텍처](../architecture.md) · [데이터플로우](../data-flow.md) · [유즈플로우](../use-flow.md) · [ADR 0005](../adr/0005-video-io-pyav-vfr.md) · [ADR 0006](../adr/0006-shot-boundary-reid.md) · [ADR 0007](../adr/0007-cpu-dev-strategy.md) · [이미지 모드 슬라이스](image-mode-slice.md)

---

## 0. 이 슬라이스의 한 줄 정의

**단일 샷 영상 구간에서 "첫 프레임 클릭 → SAM2 video 프레임 전파 → 프레임별 centroid → 떨림완화 → 프레임별 크롭 → GIF/MP4"** 의 척추를 CPU에서 (FakeVideoBackend로) 관통시킨다. SAM2 video 실추론은 GPU 블로커이므로 **Protocol 뒤에 격리하고 Colab 후행 검증**한다.

### GPU 블로커 대응 원칙 (이 슬라이스의 제1 제약)

- SAM2 video는 CPU에서 ≈0.10fps로 **로컬 실추론 불가**(ADR 0007). 따라서 추론 자체가 아니라 **그 주변의 오케스트레이션·export·UI·기하**를 먼저 완성한다.
- 실모델은 `VideoSegmentationBackend` Protocol 뒤에 두고 `FakeVideoBackend`로 치환해 **전부 CPU 단위/통합 테스트**한다.
- 자동 테스트에서 실모델은 **완전히 제외**(import 자체 회피). SAM2 video 실추론 검증은 `poc/colab/`(Colab GPU) **후행 수동**.
- 이미지 모드의 성공 패턴을 그대로 계승: ① 무거움(전파)/가벼움(박스) 분리, ② Fake 스파이 카운터, ③ `EmptyMaskError` 명시적 예외.

---

## 1. 슬라이스 범위 (가장 얇게 — 단일 샷 가정)

```
모드선택(gif) ──▶ 비디오 메인윈도 라우팅
  └─ 영상 파일 열기 (probe로 메타·fps 확인)
  └─ 구간 선택 (시작/끝 — 프레임 인덱스 또는 초)
  └─ 구간 프레임 시퀀스 추출 (video_io 확장: read_frames(start,end))
  └─ 첫 프레임을 캔버스에 표시 + 클릭으로 대상 지정 (좌표 변환 = 이미지 모드 재사용)
  └─ SAM2 video tracking: init_session → add_click(frame 0) → propagate (무거움, 워커)
  └─ 프레임별 마스크 → centroid_of_mask (기존 재사용)
  └─ smooth_centroids 떨림완화 (기존 재사용)
  └─ 프레임별 크롭 박스: apply_aspect_lock + make_crop_box (기존 재사용)
  └─ gap_policy: 단순 BACKGROUND 기본 (build_output_indices, 기존 재사용)
  └─ GIF/MP4 export (신규: 프레임 시퀀스 인코딩)
```

### In / Out (이 슬라이스 경계)

| 구분 | 포함 (In) | 제외 (Out → 후속 슬라이스, §7) |
|---|---|---|
| 입력 | 영상 1개, **사용자 지정 단일 구간**(start/end 프레임) | 멀티 구간, VFR→CFR 강제 변환, 썸네일 타임라인 |
| 검출 | **첫 프레임 클릭 1점**(전경) | Grounding DINO 자동 검출, 다중 객체, 박스 프롬프트 |
| 추적 | **단일 샷 가정** SAM2 video 전파(첫→끝) | 샷경계 감지+재추적(ADR 0006, `rematch.py`) |
| occlusion | `gap_policy` **BACKGROUND 기본 1종**만 | CUT/FREEZE 선택 UI, gap 구간 사용자 교정 |
| 기하 | centroid·smooth·aspect_lock·make_crop_box (기존) | 사용자 박스 드래그 보정, 키프레임 보간 |
| 출력 | **GIF / MP4**(무음, 단일 fps) | 오디오 동기(H4), 업스케일 결합(ADR 0009), 팔레트 고급 옵션 |

> 의도적 최소화: 첫 슬라이스의 목적은 **"추적→크롭→움짤" 경계 관통**이지 기능 완성이 아니다. `rematch.py`(IoU·rematch_score)는 **이번 슬라이스 미사용**(샷경계 재추적용 후속).

---

## 2. 비디오 백엔드 추상화 결정 (산출물 1)

### 2-1. 결정: 별도 `VideoSegmentationBackend` Protocol 신설 (ISP)

ADR 0007은 `SegmentationBackend`에 `init_video_session`/`propagate`를 **Optional**로 예고했으나, **별도 Protocol로 분리**한다.

- **근거(ISP·OCP)**: 이미지 전용 백엔드(`Sam2ImageBackend`, 향후 MobileSAM 등)가 쓰지 않는 비디오 메서드로 오염되는 것을 막는다. `supports_video()` + Optional 메서드 조합은 호출자가 런타임 `hasattr`/플래그 분기를 해야 해 LSP가 약해진다.
- **위치**: `core/segmentation/video_backend.py` (core가 추상 소유 — torch/transformers 비의존 유지). 구현은 `infra/sam2_video_backend.py`.
- 기존 `SegmentationBackend`(이미지)는 **변경 없음**. `supports_video()`는 이미지 백엔드의 "이건 비디오 아님" 표식으로 그대로 둔다.
- **→ ADR 트리거**: "비디오 세그 백엔드 Protocol을 이미지 백엔드와 분리(별도 Protocol)" 결정은 ADR로 기록할 가치가 있다(작성은 documenter, §9).

### 2-2. Protocol 시그니처 (계획 — 함수 20줄·매개변수 3개 준수)

```python
# core/segmentation/video_backend.py  (core, torch/transformers 비의존)
from typing import Protocol, runtime_checkable
import numpy as np

class EmptyTrackError(Exception):
    """전파 결과 전 프레임이 빈 마스크일 때(대상 추적 실패). 이미지 모드 EmptyMaskError 계승."""

@runtime_checkable
class VideoSegmentationBackend(Protocol):
    """단일 샷 프레임 전파 추적 추상(테스트 치환용).

    호출 계약: init_session → (첫 프레임 클릭) → propagate.
    반환 마스크: bool HxW (centroid_of_mask 계약과 동일).
    """
    device: str

    def init_session(self, frames: list[np.ndarray]) -> object:
        """구간 프레임 시퀀스로 추적 세션을 연다(무거움 — 모델 지연 로드)."""
        ...

    def add_click(self, session: object, point: tuple[int, int]) -> None:
        """첫 프레임(frame_idx=0)에 전경 클릭 1점을 등록한다."""
        ...

    def propagate(self, session: object) -> list[np.ndarray]:
        """세션을 끝까지 전파해 프레임별 bool HxW 마스크 리스트를 반환한다(무거움)."""
        ...
```

- **session을 불투명(opaque) object로 둔다**: SAM2 `Sam2VideoInferenceSession` 타입을 core/app에 노출하지 않는다(transformers 타입 누출 금지). app은 `object`로만 들고 다닌다.
- **3-메서드 분리 이유**: `add_click`을 `init_session`과 분리하면 향후 다중 클릭/객체(후속 슬라이스)를 메서드 변경 없이 확장(OCP). 이번엔 1점만.
- **매개변수 3개 규칙**: `add_click(session, point)` 2개, `propagate(session)` 1개로 모두 만족.

### 2-3. `infra/sam2_video_backend.py` — 실구현 (Colab 후행 검증, 자동 테스트 제외)

PoC `h1_track.py` 패턴을 그대로 매핑(transformers `Sam2VideoModel`/`Sam2VideoProcessor`):

| Protocol 메서드 | transformers 호출(PoC 확인) |
|---|---|
| `init_session(frames)` | `_ensure_loaded()` 지연 로드 후 `processor.init_video_session(video=frames, inference_device=device, dtype=torch.float32)` |
| `add_click(session, point)` | `processor.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=1, input_points=[[[[x,y]]]], input_labels=[[[1]]])` |
| `propagate(session)` | `for out in model.propagate_in_video_iterator(session, start_frame_idx=0): post_process_masks([out.pred_masks], original_sizes=[(h,w)])[0]` → bool HxW 리스트 |

- **지연 로드 계승**: 생성자는 `repo`/`device` 보관만(`Sam2ImageBackend` 패턴). `_ensure_loaded()`로 첫 `init_session`에서 로드.
- **정확한 시그니처는 구현 단계 venv/Colab에서 재확인**(transformers 버전별 차이 가능). PoC는 5.9.0에서 동작 확인됨.
- 빈 마스크 정규화: `_to_mask(post) -> bool HxW`는 이미지 백엔드 `_to_mask`와 동일 규약(DRY 후보 — `core` 순수 헬퍼로 추출 검토하나 이번 슬라이스는 각 백엔드 내부 유지, KISS).

### 2-4. `FakeVideoBackend` 격리 전략 (테스트 더블)

`tests/fixtures/fakes.py`에 `FakeBackend` 패턴 그대로 추가:

```python
class FakeVideoBackend:
    """VideoSegmentationBackend 준수 테스트 더블. torch/transformers 비의존.

    스파이 카운터 — 이미지 모드 FakeBackend.segment_call_count 패턴 계승:
      init_call_count, add_click_call_count, propagate_call_count
    propagate는 프레임 수만큼 결정적 마스크 리스트 반환.
    드리프트 시뮬레이션: 프레임 i마다 클릭점이 일정 속도로 이동한 사각형(추적 흉내).
    empty_after: 이 인덱스 이후 빈 마스크 반환(occlusion/추적실패 경로 테스트).
    """
    device = "cpu"
    def __init__(self, drift=(2, 0), empty_after=None): ...
    def init_session(self, frames) -> object: ...     # 카운트 +1, frames 보관(개수·click 기반 생성)
    def add_click(self, session, point) -> None: ...   # 카운트 +1, click 보관
    def propagate(self, session) -> list[np.ndarray]: ...  # 카운트 +1, N개 결정적 마스크
```

- **핵심 회귀 가드**: `propagate_call_count == 1`(구간당 전파 1회만), `init_call_count == 1` 단언으로 "무거운 전파를 중복 호출하지 않음"을 강제. 이미지 모드 `segment_call_count` 가드의 비디오판.
- **드리프트**로 만든 centroid 시퀀스에 `smooth_centroids`를 적용해 떨림완화 효과를 결정적으로 검증.
- **`empty_after`**로 빈 마스크 → `valid_flags=False` → `gap_policy` BACKGROUND 경로를 결정적으로 검증.

---

## 3. 신규/확장 모듈 책임·공개 API (산출물 2)

> 시그니처는 계획. 함수 20줄·매개변수 3개 이내. 초과 시 frozen dataclass로 묶는다(`CropRequest`·`BoxParams` 패턴 계승).

### 3-1. `infra/video_io.py` — 구간 프레임 시퀀스 추출 (확장)

기존 `FrameSource` Protocol(`probe`/`read_frame`)에 **구간 추출 메서드를 추가**한다.

```python
class FrameSource(Protocol):
    def probe(self) -> FrameMeta: ...
    def read_frame(self, index: int = 0) -> np.ndarray: ...    # [기존] 단일 프레임
    def read_frames(self, span: FrameSpan) -> list[np.ndarray]: ...  # [신규] 구간 시퀀스

@dataclass(frozen=True)
class FrameSpan:
    """추출 구간 (매개변수 3개 규칙용 묶음). 인덱스 기반(start 포함, end 미포함)."""
    start: int
    end: int          # exclusive
    step: int = 1     # 다운샘플(움짤 프레임 수 절감용, 기본 1)
```

- **`read_frames`를 Protocol에 추가하면 `_ImageFileSource`도 구현 의무**가 생긴다 → 이미지 소스는 `read_frames`에서 `[read_frame()]`(단일 프레임 1개 리스트) 반환 또는 `UnsupportedSourceError`("이미지는 구간 추출 불가"). **결정: 단일 프레임 1개 리스트 반환**(LSP 안전, 호출자 분기 불필요).
- `_VideoFrameSource.read_frames`: PyAV로 **start PTS 시크 후 end까지 순차 디코드**(ADR 0005). 기존 `read_frame`의 "첫 프레임만" 한계를 구간으로 확장. RGB BT.709 정규화 동일.
- **메모리 주의**(§6): 구간 전체를 메모리에 리스트로 보관 → `step` 다운샘플 + 구간 길이 가드(probe fps × 권장 최대 초). 스트리밍 제너레이터는 SAM2 video가 전체 프레임을 한 번에 요구(`init_video_session(video=frames)`)하므로 이번 슬라이스에선 리스트 유지(KISS), 메모리 가드만 둔다.
- **fps 산출 개선(백로그)**: `average_rate`는 VFR 부정확 → `r_frame_rate` 폴백 검토(HANDOFF 백로그 반영, 출력 fps에 직결되므로 이번 슬라이스에서 처리).

### 3-2. `app/video_capture.py` — VideoCaptureUseCase (신규 오케스트레이션)

이미지 모드 `ImageCaptureUseCase`의 **무거움/가벼움 분리**를 그대로 계승. UI는 이 유스케이스만 호출.

```python
@dataclass(frozen=True)
class TrackResult:
    """추적 1회 성공 결과(불변). 이미지 모드 SegmentResult 계승."""
    masks: list[np.ndarray]              # 프레임별 bool HxW
    centroids: list[tuple[float, float] | None]  # None=빈 마스크 프레임

@dataclass(frozen=True)
class VideoCropParams:
    """프레임별 박스 계산 입력 묶음(매개변수 3개 규칙). 이미지 BoxParams 계승."""
    box_size: tuple[int, int]
    aspect: str | None
    smooth_window: int = 5               # smooth_centroids 윈도

class VideoCaptureUseCase:
    def __init__(self, source: FrameSource, backend: VideoSegmentationBackend) -> None: ...

    def load_first_frame(self, span: FrameSpan) -> np.ndarray:
        """구간 첫 프레임만 추출(클릭 표시용). 모델 로드·전파 안 함(가벼움)."""

    def track(self, frames: list[np.ndarray], point: tuple[int, int]) -> TrackResult:
        """init→add_click→propagate→프레임별 centroid (무거움 — 워커에서 1회).
        전 프레임 빈 마스크면 EmptyTrackError."""

    def compute_boxes(self, result: TrackResult, params: VideoCropParams) -> list[CropBox]:
        """centroids→smooth→프레임별 박스(순수·가벼움). backend 절대 미호출.
        종횡비/크기 변경 시 재추적 없이 즉시 재호출(이미지 compute_box 계승)."""

    def export(self, frames, boxes, target: tuple[str, "VideoExportConfig"]) -> None:
        """gap_policy로 출력 인덱스 결정 → 프레임별 크롭 → GIF/MP4 인코딩."""
```

- **무거움(track) / 가벼움(compute_boxes) 분리가 이 슬라이스의 핵심 설계**(이미지 §1-1 계승). 추적은 워커 1회, 박스 재계산은 메인 스레드 즉시. `propagate_call_count` 회귀 가드가 순수성 강제.
- `track`은 `frames`를 인자로 받는다(UI가 `load_first_frame`으로 먼저 표시한 그 구간 frames 재사용 — 이중 디코드 방지). 매개변수 3개: `(frames, point)`.
- `centroid_of_mask`가 None(빈 마스크)이면 centroid 리스트에 None을 담는다 → `compute_boxes`에서 `smooth_centroids`의 `_hold_forward`가 직전 위치로 홀드(기존 로직), `valid_flags`는 `gap_policy`로 전달.

### 3-3. `core/export/video_export.py` — GIF/MP4 인코딩 (신규)

**위치 결정**: core에 두되 **인코딩 의존(imageio/imageio-ffmpeg)을 함수 내부 지연 import**로 격리한다.

- **결정 근거**: 이미지 모드 `core/export/image_export.py`가 이미 Pillow(순수 인코딩)를 core에서 허용하는 선례가 있다. GIF는 Pillow로도 가능하나 MP4는 ffmpeg 바인딩 필요. **core 경계 불변식(torch/PySide6/PyAV/transformers 금지)** 중 PyAV는 infra 소유이므로, MP4 인코딩은 **PyAV 대신 `imageio`+`imageio-ffmpeg`**를 쓰고 core/export 안에서 지연 import한다. 이로써 core는 PyAV/torch/PySide6 비의존을 유지하면서 인코딩만 담당.
- **대안(검토)**: MP4 인코딩을 `infra/video_encode.py`에 두는 안. core의 "외부 라이브러리 비의존" 원칙엔 더 보수적이나, ① 이미지 export 선례와 비대칭, ② 인코딩은 순수 변환(부수효과=파일쓰기뿐)이라 도메인 로직 성격. **→ ADR 트리거**: "GIF/MP4 인코딩 위치(core/export + imageio 지연 import)" 결정을 ADR로 기록(작성은 documenter, §9). 리뷰에서 뒤집히면 infra로 이동(모듈 1개만 이동, app은 Protocol 미사용이라 영향 적음 — 단 그 경우 export 함수도 Protocol 뒤로).

```python
@dataclass(frozen=True)
class VideoExportConfig:
    """움짤 내보내기 설정(불변). 이미지 ExportConfig 계승."""
    fmt: str = "gif"          # "gif" | "mp4"
    fps: float = 12.0         # 출력 프레임레이트
    gap_policy: GapPolicy = GapPolicy.BACKGROUND   # 기본 1종(이번 슬라이스)

def crop_frames(frames, boxes) -> list[np.ndarray]:
    """프레임별 박스로 슬라이스(순수). image_export.crop_array를 프레임 루프로 재사용(DRY)."""

def encode_frames(crops, path, config) -> None:
    """크롭 프레임 시퀀스를 GIF/MP4로 인코딩(imageio 지연 import). 부수효과=파일쓰기."""
```

- **크롭 크기 통일 제약**: GIF/MP4는 전 프레임이 **동일 W×H**여야 인코딩된다. `make_crop_box`는 경계 클램프로 프레임 끝에서 박스가 작아질 수 있음 → `compute_boxes`에서 **구간 내 고정 박스 크기**(첫 박스 크기로 통일, 중심만 이동) 정책을 둔다. 이를 `core/crop`에 작은 순수 헬퍼(`unify_box_sizes(boxes)` 또는 `compute_boxes` 내부)로 추가. **결정: `compute_boxes`가 size를 한 번 계산해 전 프레임 동일 size로 `make_crop_box` 호출**(KISS, 신규 함수 최소화).
- `crop_frames`는 `image_export.crop_array`를 그대로 프레임 루프로 호출(DRY, 중복 슬라이스 금지).

### 3-4. 비디오 UI (신규 — 이미지 윈도와 별도)

이미지 모드와 별도 윈도/유스케이스 권장(기존 `ImageMainWindow`는 단일 프레임 전제). 좌표 변환(`ui/coords.py`)·`FrameCanvas`는 재사용.

```
ui/video_window.py   VideoMainWindow(usecase_factory)
  - 파일열기 → probe → 구간 슬라이더(start/end) + 미리보기(첫 프레임 캔버스)
  - 첫 프레임 클릭(FrameCanvas.clicked 재사용) → 대상 지정
  - "추적 실행" → _TrackWorker(QThread) (무거움, propagate)
  - 종횡비/크기/smooth_window 조정 → compute_boxes 즉시 재계산(재추적 없음)
  - "저장" → fmt(gif/mp4)·fps 선택 → _ExportWorker(QThread)
```

- **워커 패턴 계승**: `_SegWorker`/`_UpscaleSaveWorker`(이미지 `main_window.py`)를 그대로 본떠 `_TrackWorker`(track), `_ExportWorker`(export)를 둔다. Signal·run·예외 처리 동형.
- **구간 미리보기**: 첫 슬라이스는 **구간 첫 프레임 1장 + start/end 슬라이더**만(타임라인 썸네일 스트립은 후속). `load_first_frame(span)`만 호출(가벼움).
- 라우터(`app/router.py`) `_on_mode`의 `"gif"` 분기를 **"준비 중" 안내 → `_launch_video_mode()` 조립**으로 교체. composition root에서 `Sam2VideoBackend`/`open_source` 주입.

---

## 4. 의존성 방향 (산출물 3)

```
┌──────────────────────────────────────────────┐
│ UI (PySide6)  VideoMainWindow · FrameCanvas(재) │
└───────────────┬──────────────────────────────┘
                │ 시그널/슬롯 (_TrackWorker, _ExportWorker)
┌───────────────▼──────────────────────────────┐
│ Application  VideoCaptureUseCase              │
└───────────────┬──────────────────────────────┘
                │ Protocol 의존 (구체 미의존)
┌───────────────▼──────────────────────────────┐
│ Core (UI/IO/torch 비의존)                      │
│  tracking/gap_policy(재) · crop/crop(재)        │
│  segmentation/video_backend(신규 Protocol)     │
│  export/video_export(신규, imageio 지연import)  │
└───────────────┬──────────────────────────────┘
                │ Protocol 구현체 주입 (composition root)
┌───────────────▼──────────────────────────────┐
│ Infra  video_io(PyAV, read_frames 확장)         │
│  · sam2_video_backend(신규, torch/transformers) │
│  · device(재)                                  │
└──────────────────────────────────────────────┘
```

### 경계 불변식 (이미지 모드 계승, 회귀 가드 대상)

- **core는 torch·transformers·PySide6·PyAV·av를 import하지 않는다.** `video_export`는 `imageio`/`imageio-ffmpeg`만, 함수 내부 지연 import.
- **`VideoSegmentationBackend` Protocol은 core**, SAM2 video 구현은 **infra**. session은 opaque object로 core/app에 transformers 타입 누출 금지.
- **app은 Protocol(`FrameSource`·`VideoSegmentationBackend`)에만 의존**, 구체는 router 주입. 테스트는 `FakeVideoBackend`·`FakeFrameSource` 주입.
- **UI는 app 유스케이스만 호출.** 좌표 변환은 `ui/coords.py` 순수 함수 재사용.

---

## 5. 작업 분해 (산출물 4 — 인터페이스→테스트→구현 커밋 순서)

> 순서 원칙(전역): **인터페이스 → 테스트 → 구현 → 리팩터**. 1줄 = 1 논리적 변경 = 1 커밋. 브랜치: `feature/video/tracking-slice`.

### 도메인: segmentation video backend (core Protocol — 먼저, 의존 없음)
- [ ] `core/segmentation/video_backend.py` 인터페이스(`VideoSegmentationBackend` Protocol, `EmptyTrackError`, docstring) — 우선순위: 높음
- [ ] `tests/fixtures/fakes.py`에 `FakeVideoBackend`(Protocol 준수, 스파이 카운터·drift·empty_after) — 우선순위: 높음
- [ ] Protocol 계약 테스트(`isinstance(FakeVideoBackend(), VideoSegmentationBackend)`) — 우선순위: 높음

### 도메인: video_io 구간 추출 (infra)
- [ ] `infra/video_io.py` `FrameSpan` dataclass + `FrameSource.read_frames` 시그니처 추가(이미지 소스는 단일프레임 리스트 위임) — 우선순위: 높음
- [ ] `test_video_io_frames.py`(importorskip("av") 가드, 합성 클립 fixture로 구간 길이·step 검증; 이미지 소스 위임 검증) — 우선순위: 중간
- [ ] `_VideoFrameSource.read_frames`(PTS 시크→end 순차 디코드, RGB) + fps `r_frame_rate` 폴백 — 우선순위: 중간

### 도메인: usecase (app — 임계 경로)
- [ ] `app/video_capture.py` 인터페이스(`VideoCaptureUseCase`·`TrackResult`·`VideoCropParams`, docstring) — 우선순위: 높음
- [ ] `test_video_capture.py`(FakeVideoBackend+FakeFrameSource: track→centroids, compute_boxes 순수성[propagate_call_count==1], smooth 적용, gap_policy BACKGROUND, EmptyTrackError) — 우선순위: 높음
- [ ] `VideoCaptureUseCase` 구현(track=무거움/compute_boxes=가벼움 조립, 고정 box size 통일) → 테스트 통과 — 우선순위: 높음

### 도메인: video export (core)
- [ ] `core/export/video_export.py` 인터페이스(`VideoExportConfig`·`crop_frames`·`encode_frames`, docstring) — 우선순위: 높음
- [ ] `test_video_export.py`(합성 프레임 시퀀스 → GIF/MP4 인코드→디코드 라운드트립: 프레임수·크기 일치; gap_policy 출력 인덱스 반영) — 우선순위: 높음
- [ ] `crop_frames`(crop_array 재사용) + `encode_frames`(imageio GIF/MP4) 구현 → 테스트 — 우선순위: 높음

### 도메인: infra SAM2 video (Colab 후행 — 자동 테스트 제외)
- [ ] `infra/sam2_video_backend.py` 골격(생성자 보관·`_ensure_loaded` 스텁, PoC h1 매핑 주석) — 우선순위: 중간
- [ ] `poc/colab/` 노트북에 image-backend 대비 video-backend 호출 검증 셀 추가(수동 GPU) — 우선순위: 중간
- [ ] `segment` 실구현(transformers init/add/propagate→bool 마스크) — 우선순위: 낮음 (Colab 수동, 자동 CI 제외)

### 도메인: UI
- [ ] `ui/video_window.py`(파일열기·구간 슬라이더·미리보기·클릭·_TrackWorker·_ExportWorker·종횡비/크기/smooth 즉시 재계산) — 우선순위: 중간
- [ ] `app/router.py` `"gif"` 분기를 `_launch_video_mode()`(composition root 조립)로 교체 — 우선순위: 중간

### 마무리
- [ ] CPU 자동 스모크(`pytest` — Fake 경로 전 구간) + GIF/MP4 산출물 수동 확인 절차 HANDOFF 기록 — 우선순위: 높음
- [ ] HANDOFF.md 갱신(비디오 슬라이스 척추 완료, SAM2 video 실추론은 Colab 미검증 명시) — 우선순위: 높음 (커밋-문서 동기화 필수)

### 임계 경로
```
video_backend(Protocol) → FakeVideoBackend → video_capture 테스트 → video_capture 구현
                                                      │
                       video_export(인터페이스→테스트→구현) ─┘ (병렬 가능)
video_io.read_frames 와 sam2_video_backend 는 Protocol 뒤에 숨어 병렬 진행.
UI는 usecase·export 완성 후. SAM2 실구현은 최후행(Colab, CI 무관).
```
- **핵심**: `FakeVideoBackend` → `video_capture` → `video_export`가 척추. 이 셋이 CPU에서 녹색이면 슬라이스 가치(추적→크롭→움짤 오케스트레이션)는 GPU 없이 확보된다.

---

## 6. 테스트 전략 (산출물 5)

### 6-1. 격리 원칙 (이미지 모드 계승)

| 대상 | 격리 방법 |
|---|---|
| **SAM2 video 백엔드** | `FakeVideoBackend`(VideoSegmentationBackend 준수). `propagate`는 drift 기반 결정적 마스크 리스트. torch/transformers 전혀 로드 안 함. `propagate_call_count`/`init_call_count` 스파이로 "전파 1회만" 회귀 가드 |
| **PyAV 구간 디코드** | `FakeFrameSource`(read_frames=고정 시퀀스). 실 PyAV는 `importorskip("av")` + 합성 클립 fixture 통합 테스트 |
| **GIF/MP4 인코딩** | 합성 프레임 시퀀스 → `encode_frames` → imageio로 **디코드 라운드트립**(프레임 수·크기·fmt 일치). `imageio-ffmpeg` 미설치 시 MP4는 `importorskip`, GIF는 Pillow 폴백 가능 |
| **PySide6 위젯** | 좌표 변환은 기존 `ui/coords` 순수 함수 재사용 테스트. 윈도는 offscreen 스모크 |
| **SAM2 video 실추론** | **자동 테스트 완전 제외.** Colab(`poc/colab/`) 수동. PySceneDetect·Grounding DINO 이번 슬라이스 미사용 |

### 6-2. 핵심 단위 테스트 (CPU, 모델 무관 — 슬라이스 가치의 증거)

- `test_video_capture.py` (가장 중요):
  - `track`: FakeVideoBackend drift → `TrackResult.centroids`가 프레임마다 이동(추적 흉내) 확인
  - **순수성 가드**: `compute_boxes`를 종횡비·크기·smooth_window 바꿔 여러 번 호출해도 `propagate_call_count == 1`, `init_call_count == 1` (재추적 안 함 — 이미지 `segment_call_count` 가드의 비디오판)
  - `smooth_centroids` 적용: drift centroid에 window 적용 시 인접 프레임 박스 중심 분산이 감소
  - **고정 box size**: 전 프레임 박스 W×H 동일(GIF/MP4 인코딩 가능 불변식)
  - `gap_policy` BACKGROUND: `empty_after`로 빈 마스크 구간 생성 → 출력 인덱스가 전 프레임 유지(`build_output_indices` 재사용 결과 일치)
  - `EmptyTrackError`: 전 프레임 빈 마스크 시 발생
  - Protocol 계약: `isinstance(FakeVideoBackend(), VideoSegmentationBackend)`
- `test_video_export.py`:
  - `crop_frames`: N 프레임 + N 박스 → N개 동일 크기 크롭(crop_array 재사용 회귀)
  - **GIF 라운드트립**: encode→imageio.mimread → 프레임 수·(H,W) 일치
  - **MP4 라운드트립**(importorskip): encode→디코드 → 프레임 수 근사·크기 일치(코덱 손실 허용 오차)
  - `gap_policy` 반영: BACKGROUND 출력 인덱스대로 프레임 선택

### 6-3. 통합 (조건부) / 수동
- `test_video_io_frames.py`: `importorskip("av")` + 합성 클립으로 `read_frames(FrameSpan)` 구간 길이·step 검증.
- **SAM2 video 실추론**: `poc/colab/` GPU 수동. AC-01 추적 유지율(≥80%)·AC-06 GPU fps를 `poc/REPORT.md` 미검증 칸에 기록. **자동 CI 제외**.

### 6-4. 커버리지
- 비디오 도메인 순수 로직(`video_capture` 조립·`video_export`·기존 crop/gap_policy 조합) **80%+**. SAM2 video 래퍼는 Colab 수동 스모크로 happy path.

---

## 7. 리스크 및 후속 슬라이스 (산출물 6·7)

### 7-1. 리스크 및 완화책

| 리스크 | 영향 | 완화책 |
|---|---|---|
| **GPU 미검증 위에 코드를 쌓음** | SAM2 video 실 API/마스크 형식이 Fake와 어긋나면 통합 시 재작업 | ① Protocol을 PoC `h1_track.py` **실측 호출에 정확히 매핑**(§2-3 표) ② `_to_mask` bool HxW 계약을 이미지 백엔드와 동일화 ③ `infra/sam2_video_backend` 골격을 **PoC 시점에 Colab에서 1회 실호출 검증**(스모크) 후 척추 구현 ④ Fake가 실모델 출력 형태(프레임별 bool HxW 리스트)를 모사 |
| **메모리 — 구간 전체 프레임 보관** | 4K·긴 구간에서 RAM 폭증(`init_video_session(video=frames)`가 전체 요구) | ① `FrameSpan.step` 다운샘플 ② probe fps×권장 최대 초(예: 10초)로 구간 길이 가드 + 한국어 경고 ③ 표시용은 첫 프레임 1장만(`load_first_frame`) ④ 장구간/스트리밍은 후속 |
| **GIF 용량·프레임 수 폭증** | 수백 프레임 GIF가 수십 MB | ① `fps`/`step`으로 프레임 수 제한 ② 권장 길이 안내 ③ MP4 우선 권장(GIF는 호환용) ④ 팔레트 최적화는 후속 |
| **VFR fps 부정확 → 움짤 속도 어긋남** | 재생 속도 왜곡 | `average_rate`→`r_frame_rate` 폴백(§3-1), 출력 `fps`는 사용자 조정 가능 |
| **빈 마스크/추적 실패** | centroid None 연속 | `_hold_forward`(직전 위치 홀드)+`gap_policy` BACKGROUND, 전부 실패면 `EmptyTrackError` 한국어 안내 |
| **크롭 박스 크기 프레임별 불일치** | 인코딩 실패 | `compute_boxes`에서 고정 size 통일(§3-3) + 테스트 불변식 |
| **추적 워커 장시간(GPU에서도 수 초~분)** | UI 멈춤 오인 | `_TrackWorker` 비블로킹 + 진행 표시(이미지 워커 패턴) |

### 7-2. 후속 슬라이스 목록 (이번 명시적 제외 → 다음)

1. **샷경계 재추적**(ADR 0006): PySceneDetect 컷 감지 → 컷 직후 `core/tracking/rematch.py`(`iou`·`rematch_score`)로 동일 객체 재매칭 → SAM2 재초기화·object_id 유지. **이번 슬라이스의 "단일 샷 가정"을 푸는 핵심 후속.**
2. **Grounding DINO 자동 검출/재매칭**(ADR 0003): 클릭 없이/컷 후보 박스 자동 검출 → rematch_score 결합.
3. **occlusion gap UI**: `gap_policy` CUT/FREEZE 선택 UI + gap 구간 사용자 교정(이번엔 BACKGROUND 기본 1종만).
4. **오디오 동기**(PoC H4): 선택 구간 오디오를 PTS 동기로 MP4에 머지(움짤→짧은 클립).
5. **업스케일 결합**(ADR 0009): 프레임별/크롭별 Swin2SR 업스케일을 export에 결합(이미지 모드 `export(upscaler=)` 패턴을 시퀀스로 확장).
6. **타임라인 UI 고도화**: 썸네일 스트립, 멀티 구간, VFR→CFR 변환 옵션.

---

## 8. 기술 결정사항 요약

| 결정 | 근거 |
|---|---|
| `VideoSegmentationBackend` **별도 Protocol**(core) | ISP — 이미지 백엔드 오염 방지, LSP 강화. ADR 0007 Optional 예고를 분리로 구체화 → **ADR 트리거** |
| session **opaque object** | transformers `Sam2VideoInferenceSession` 타입을 core/app에 누출 금지(경계 불변식) |
| `track`(무거움)/`compute_boxes`(가벼움) 분리 | 이미지 모드 핵심 설계 계승. `propagate_call_count` 회귀 가드로 순수성 강제 |
| GIF/MP4 export = **core/export + imageio 지연 import** | 이미지 `image_export`(Pillow) 선례. core는 PyAV/torch/PySide6 비의존 유지 → **ADR 트리거**(뒤집히면 infra 이동) |
| `read_frames(FrameSpan)`로 video_io 확장 | 기존 `FrameSource` Protocol 확장(매개변수 dataclass 묶음). 이미지 소스는 단일프레임 위임(LSP) |
| 고정 box size 통일 | GIF/MP4 동일 프레임 크기 인코딩 불변식. `make_crop_box`는 size 입력 그대로 사용 |
| 비디오 UI 별도 윈도(`VideoMainWindow`) | 이미지 단일프레임 전제와 구간/추적/시퀀스 요구가 상이(SRP). `FrameCanvas`·`coords`는 재사용 |
| `rematch.py` **이번 미사용** | 단일 샷 가정. 샷경계 재추적은 후속 슬라이스 |

---

## 9. ADR 트리거 (작성은 documenter)

- **(트리거) ADR 0010 후보**: "비디오 세그 백엔드 Protocol 분리" — `SegmentationBackend`(이미지)와 별도 `VideoSegmentationBackend`(비디오) 신설. ADR 0007의 "비디오 메서드 Optional" 결정을 ISP 근거로 **분리로 갱신/보완**.
- **(트리거) ADR 0011 후보**: "GIF/MP4 인코딩 위치" — `core/export/video_export.py` + `imageio`/`imageio-ffmpeg` 지연 import 채택. core 경계 불변식과의 정합성·대안(infra) 트레이드오프 기록.

> 본 계획서는 코드/ADR 본문을 작성하지 않는다. ADR 작성은 `/develop` documenter 단계, 구현은 developer/tester 단계.

---

## 10. 완료 정의 (DoD)

- [ ] `pytest`로 **Fake 경로 전 구간 녹색**: `FakeVideoBackend`+`FakeFrameSource`로 구간추출→track→smooth→compute_boxes→gap_policy→GIF/MP4 인코드/디코드 라운드트립.
- [ ] **순수성 회귀 가드**: `compute_boxes` 반복 호출 시 `propagate_call_count == 1`.
- [ ] core가 torch/transformers/PySide6/PyAV/av를 import하지 않음(경계 검증, `video_export`는 imageio 지연 import).
- [ ] 기존 166개 테스트 무회귀.
- [ ] `infra/sam2_video_backend.py` 골격 + PoC `h1_track.py` 매핑 주석(실추론은 Colab 미검증 명시).
- [ ] HANDOFF.md 갱신 + 후속 슬라이스(샷경계 재추적) 진입 가능 상태.
</content>
</invoke>
