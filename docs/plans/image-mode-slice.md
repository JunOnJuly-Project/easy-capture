# 이미지 모드 첫 수직 슬라이스 — 개발 계획

> 작성: 2026-05-28 · 단계: `/develop` 파이프라인 planner · 관련: [아키텍처](../architecture.md) · [데이터플로우](../data-flow.md) · [유즈플로우](../use-flow.md) · [ADR 0005](../adr/0005-video-io-pyav-vfr.md) · [ADR 0007](../adr/0007-cpu-dev-strategy.md)

---

## 1. 개요

CPU 개발 PC에서 동작 가능한 **이미지(짤) 모드 end-to-end happy path** 첫 수직 슬라이스를 구현한다. GPU가 필요한 비디오 추적 경로(블로커)를 우회해 "파일 열기 → 프레임 표시 → 클릭 → 마스크 → 크롭 → 저장"의 전 구간을 얇게 관통시켜, 레이어 경계(UI→app→core→infra)와 백엔드 추상화가 실제로 맞물리는지 검증한다.

### 슬라이스 범위 (happy path)

```
모드선택(image) ──▶ 메인윈도 라우팅
  └─ 파일 열기 (동영상 또는 이미지)
  └─ 한 프레임을 UI 캔버스에 표시 (RGB 정규화)
  └─ 사용자가 피사체 클릭 (캔버스 좌표 → 이미지 좌표)
  └─ SAM2 image 백엔드(CPU)로 마스크 생성 (지연 로드, 워커 비블로킹)
  └─ core/crop 으로 centroid 중심 크롭 박스 산출 (기존 함수 재사용)
  └─ core/export 로 PNG/JPG 저장
```

### In / Out (이 슬라이스 경계)

| 구분 | 포함 (In) | 제외 (Out, 다음 슬라이스) |
|---|---|---|
| 입력 | 단일 프레임 추출(이미지 파일 / 영상 첫·지정 프레임) | 타임라인 UI, 구간 선택, VFR CFR 변환 |
| 검출 | 클릭 포인트 프롬프트(SAM2 직접) | Grounding DINO 자동 검출, 후보 박스 표시 |
| 크롭 | centroid 중심, 종횡비 잠금(고정 프리셋 1개) | 떨림완화(비디오 전용), 사용자 W×H 드래그 조정 UI |
| 출력 | PNG / JPG 단일 이미지 | GIF / MP4, 업스케일 |
| 추적 | 없음(이미지 모드는 추적 없음) | SAM2 video, tracking, 샷경계 재매칭 |

> 종횡비 프리셋·크롭 크기 슬라이더 등 UX 디테일은 의도적으로 최소화한다. 첫 슬라이스의 목적은 **경계 관통**이지 기능 완성이 아니다.

---

## 2. 도메인 분석 및 의존성 방향

### 레이어 의존 (단방향, DIP 준수)

```
┌──────────────────────────────────────────────┐
│ UI (PySide6)                                  │
│  MainWindow(image) · FrameCanvas              │
└───────────────┬──────────────────────────────┘
                │ 시그널/슬롯, 진행 콜백
┌───────────────▼──────────────────────────────┐
│ Application (오케스트레이션)                   │
│  ImageCaptureUseCase                          │
└───────────────┬──────────────────────────────┘
                │ Protocol 의존 (구체 타입 미의존)
┌───────────────▼──────────────────────────────┐
│ Core (도메인 로직, UI/IO 비의존)               │
│  crop(기존) · export(신규) · segmentation/IF   │
└───────────────┬──────────────────────────────┘
                │ Protocol 구현체 주입
┌───────────────▼──────────────────────────────┐
│ Infra (외부 의존)                              │
│  video_io(PyAV/ffprobe) · sam2_image_backend  │
│  · device(기존)                               │
└──────────────────────────────────────────────┘
```

### 경계 규칙 (불변식)

- **core 는 PySide6·PyAV·torch·transformers 를 import 하지 않는다.** core/export 는 Pillow(순수 이미지 인코딩)만 허용한다. core는 numpy 배열과 `SegmentationBackend` Protocol에만 의존한다.
- **SAM2 image 백엔드는 `core/segmentation/backend.py`의 `SegmentationBackend` Protocol을 준수**하지만, 무거운 torch/transformers 의존을 가지므로 **infra에 구현체를 둔다.** core에는 Protocol(추상)만, infra에 구체(SAM2)를 두어 DIP를 만족한다.
- **app 유스케이스는 Protocol에만 의존**하고, 구체 백엔드/리더는 진입점(`__main__` 또는 라우터)에서 주입(생성자 주입)한다. 테스트에서는 가짜 구현으로 치환한다.
- **UI는 app 유스케이스만 호출**하고 core/infra를 직접 만지지 않는다. 좌표 변환·표시 외 도메인 로직은 UI에 두지 않는다.

> 결정: `SegmentationBackend` Protocol의 물리적 위치는 기존대로 `core/segmentation/backend.py`에 유지한다(core가 추상을 소유). SAM2 구현은 `infra/sam2_image_backend.py`에 둔다. 이로써 core는 transformers 비의존을 유지한다.

---

## 3. 디렉토리 구조 (신규/변경)

```
src/easy_capture/
├── __main__.py                      # [변경] 라우터 연결
├── app/                             # [신규] 유스케이스 레이어
│   ├── __init__.py
│   ├── router.py                    # [신규] 모드선택→메인윈도 라우팅
│   └── image_capture.py             # [신규] ImageCaptureUseCase
├── core/
│   ├── crop/crop.py                 # [재사용] 변경 없음
│   ├── segmentation/backend.py      # [재사용] Protocol 변경 없음
│   └── export/                      # [신규]
│       ├── __init__.py
│       └── image_export.py          # [신규] PNG/JPG 인코딩(순수)
├── infra/
│   ├── device.py                    # [재사용] 백엔드 선택에 활용
│   ├── video_io.py                  # [신규] PyAV/ffprobe 프레임 추출
│   └── sam2_image_backend.py        # [신규] SAM2 image predictor 구현
└── ui/
    ├── mode_select.py               # [재사용] 변경 없음
    ├── main_window.py               # [신규] 이미지 모드 메인윈도
    └── frame_canvas.py              # [신규] 프레임 표시 + 클릭 캡처

tests/
├── test_crop.py                     # [재사용]
├── test_device.py                   # [재사용]
├── test_image_export.py             # [신규] 인코딩 순수 로직
├── test_image_capture.py            # [신규] 유스케이스(가짜 백엔드/리더)
├── test_video_io_meta.py            # [신규] 메타·좌표 순수 부분(조건부)
└── fixtures/
    ├── __init__.py
    └── fakes.py                     # [신규] FakeBackend, FakeFrameSource
```

> `app/` 레이어는 아키텍처 문서(§1)에 이미 예고되어 있으나 스캐폴딩에는 없다. 이 슬라이스에서 신설한다.

---

## 4. 신규 모듈별 책임 / 공개 API 시그니처

> 시그니처는 계획이며, 함수 20줄·매개변수 3개 이내 규칙을 만족하도록 설계했다. 매개변수가 3개를 넘는 곳은 작은 dataclass(설정 객체)로 묶는다.

### 4-1. `infra/video_io.py` — 프레임 추출 (PyAV/ffprobe, ADR 0005)

책임: 이미지/동영상 파일에서 **단일 프레임을 RGB(BT.709 정규화)** numpy 배열로 추출. 이미지 모드 슬라이스에서는 "한 프레임"만 필요하므로 구간 스트리밍은 다음 슬라이스로 미룬다.

```python
@dataclass(frozen=True)
class FrameMeta:
    """프레임/소스 메타 (UI 표시·검증용)."""
    width: int
    height: int
    is_video: bool
    fps: float | None        # 이미지면 None

class FrameSource(Protocol):
    """단일 프레임 공급원 추상(테스트 치환용). app 은 이 Protocol 에 의존."""
    def probe(self) -> FrameMeta: ...
    def read_frame(self, index: int = 0) -> np.ndarray: ...   # RGB HxWx3 uint8

def open_source(path: str) -> FrameSource:
    """확장자/ffprobe 로 이미지/영상 판별 후 적절한 FrameSource 생성(팩토리)."""

# 내부 구현(공개 아님): _ImageFileSource(Pillow), _VideoFrameSource(PyAV)
#  - _VideoFrameSource.read_frame: PTS 기반 시크 후 1프레임 디코드 → RGB 변환
#  - 색공간: 디코드 결과를 RGB BT.709 full 로 정규화 (data-flow §2)
```

- 매개변수 규칙: `read_frame(index)`는 기본값 0(첫 프레임). 영상의 임의 시점은 다음 슬라이스(타임라인)에서 확장.
- 검증/에러: 미지원 코덱·손상 파일은 `UnsupportedSourceError`(한국어 메시지)로 변환해 상위로 전달(error-handling §1).

### 4-2. `infra/sam2_image_backend.py` — SAM2 image 백엔드 (ADR 0007)

책임: transformers(Sam2Model/Sam2Processor)로 단일 프레임 + 클릭 포인트 → 마스크. **무거운 모델 로드는 첫 `segment_image` 호출 시점까지 지연**(생성자에서 로드 금지).

```python
class Sam2ImageBackend:                       # SegmentationBackend Protocol 구현
    device: str

    def __init__(self, repo: str, device: str) -> None:
        """repo·device 보관만. 모델은 아직 로드하지 않는다(지연)."""

    def segment_image(self, frame, points=None, boxes=None) -> np.ndarray:
        """프레임 + 클릭 포인트로 마스크(bool HxW) 반환. 첫 호출 시 모델 로드."""

    def supports_video(self) -> bool:
        return False                           # 이미지 전용

    # 내부: _ensure_loaded() — Sam2Processor/Sam2Model.from_pretrained(repo).to(device)
    #       _to_mask(post) — post_process_masks 결과를 bool HxW 로 정규화
```

- 디바이스/repo는 `infra/device.detect_device()` + `select_sam2_repo(device)`로 결정해 주입.
- 반환 마스크 형식은 `core/crop.centroid_of_mask`가 받는 `np.where(mask>0)` 규약(bool/0-1 HxW)에 맞춘다. **core와의 계약은 마스크 배열 형식뿐**(transformers 타입 노출 금지).

### 4-3. `core/export/image_export.py` — 이미지 인코딩 (순수)

책임: numpy RGB 배열을 크롭 박스로 자르고 PNG/JPG로 저장. **Pillow만 사용**(torch/PySide6 비의존). UI/IO 부수효과는 파일 쓰기뿐.

```python
@dataclass(frozen=True)
class ExportConfig:
    fmt: str = "png"          # "png" | "jpg"
    quality: int = 95         # jpg 한정
    color_space: str = "sRGB" # 태깅 (data-flow §2)

def crop_array(frame, box) -> np.ndarray:
    """RGB 배열을 (x1,y1,x2,y2) 박스로 슬라이스. 경계는 box 가 이미 보장."""

def save_image(frame, path, config) -> None:
    """RGB 배열을 path 에 config 형식으로 저장(Pillow). 디렉토리/권한 오류는 상위 전달."""
```

- `crop_array`는 순수 함수(반환 배열). `save_image`만 IO. 분리해 크롭 결과를 테스트로 검증 가능하게 한다.
- `make_crop_box`(crop.py)가 짝수·경계 클램프를 이미 보장 → export는 단순 슬라이스만.

### 4-4. `app/image_capture.py` — ImageCaptureUseCase (오케스트레이션)

책임: 슬라이스 흐름을 조립. UI는 이 유스케이스만 호출. core/infra 구체 타입 미의존(Protocol 주입).

```python
@dataclass(frozen=True)
class CropRequest:
    point: tuple[int, int]          # 클릭 좌표(이미지 기준)
    box_size: tuple[int, int]       # 요청 크롭 W×H
    aspect: str | None = None       # 종횡비 프리셋 키 or None

class ImageCaptureUseCase:
    def __init__(self, source: FrameSource, backend: SegmentationBackend) -> None:
        """프레임 공급원·세그 백엔드 주입(DIP). 둘 다 Protocol 타입."""

    def load_frame(self) -> np.ndarray:
        """첫 프레임 추출(UI 표시용). 모델 로드 안 함(지연 유지)."""

    def make_crop_box(self, frame, request) -> tuple[int, int, int, int]:
        """클릭→마스크→centroid→종횡비잠금→크롭박스. core 함수 조합(20줄 이내)."""

    def export(self, frame, box, target) -> None:
        """크롭 후 저장. target=(path, ExportConfig). core/export 위임."""
```

- `make_crop_box`는 백엔드 호출(무거움)을 포함 → UI는 이를 워커 스레드에서 호출한다(§아키텍처 §5).
- core의 `make_crop_box`(기하)와 이름이 겹치므로, 유스케이스 메서드는 그 함수를 **호출**하는 얇은 조립자임을 주석으로 명시(혼동 방지). 필요 시 `compute_crop_box`로 개명 검토.

### 4-5. `app/router.py` — 모드 라우팅

책임: `ModeSelectWindow.mode_selected` 시그널 수신 → 'image'면 메인윈도 생성·표시. 'gif'는 아직 미구현 안내(토스트/다이얼로그).

```python
class AppRouter:
    def __init__(self, app) -> None: ...
    def start(self) -> None:
        """모드선택 창 표시 + 시그널 연결."""
    def _on_mode(self, mode: str) -> None:
        """'image' → MainWindow 생성/표시, 'gif' → '준비 중' 안내."""
```

- 라우터가 의존성(백엔드·소스 팩토리)을 조립해 유스케이스/윈도에 주입하는 **조립 루트(composition root)** 역할.

### 4-6. `ui/main_window.py` — 이미지 모드 메인윈도

책임: 파일 열기 버튼 → 캔버스 표시 → 클릭 위임 → "저장" 버튼. 워커로 세그 호출(UI 비블로킹), 진행/에러 표시.

```python
class ImageMainWindow(QMainWindow):
    def __init__(self, usecase_factory) -> None:
        """경로가 정해지면 usecase 를 만들기 위한 팩토리 주입."""
    # 슬롯: _on_open_file, _on_canvas_click, _on_export
    # 워커: _SegWorker(QThread/QRunnable) — segment+crop 박스 산출, 결과 시그널
```

- 매개변수 3개 규칙: 윈도 구성은 작은 빌더 메서드(`_build_toolbar`, `_build_canvas`)로 분할.

### 4-7. `ui/frame_canvas.py` — 프레임 캔버스

책임: RGB numpy → QImage 표시, 클릭 시 **위젯 좌표 → 이미지 좌표 변환**(스케일/오프셋 보정) 후 시그널 방출. 마스크 오버레이(반투명) 표시.

```python
class FrameCanvas(QWidget):
    clicked = Signal(int, int)            # 이미지 좌표(x, y)
    def set_frame(self, frame) -> None:   # RGB ndarray → 표시
    def set_overlay(self, mask) -> None:  # 마스크 반투명 오버레이
    # 내부: _widget_to_image(pos) — 표시 스케일 역변환(순수 계산, 테스트 가능)
```

- **좌표 변환 로직은 순수 함수로 분리**(`_widget_to_image`)해 PySide6 없이 단위 테스트 가능하게 한다(테스트 전략 §5 참조). 가능하면 `ui/coords.py`로 추출.

---

## 5. 테스트 전략

### 5-1. 격리 원칙 — 무거운/외부 의존을 가짜로 치환

| 대상 | 격리 방법 |
|---|---|
| **SAM2 백엔드** | `tests/fixtures/fakes.py`의 `FakeBackend`(SegmentationBackend Protocol 준수). `segment_image`는 입력 프레임 크기에 맞춰 **결정적 더미 마스크**(예: 중앙 사각형) 반환. torch/transformers 전혀 로드 안 함. `@runtime_checkable` Protocol이므로 `isinstance(fake, SegmentationBackend)`로 계약 검증 가능 |
| **PyAV/ffprobe** | `FakeFrameSource`(FrameSource Protocol 준수). `read_frame`은 고정 numpy 배열 반환. 실제 PyAV 디코드는 **선택적 통합 테스트**로 분리(작은 합성 클립 fixture, PyAV 미설치 시 `pytest.importorskip`) |
| **Pillow 저장** | `tmp_path` fixture로 실제 파일 저장 후 재로드 검증(가벼움, 외부 모델 무관) |
| **PySide6 위젯** | 좌표 변환·크롭 산출 등 **순수 로직만 단위 테스트**. 위젯 렌더링은 스모크 수준(생성·표시)으로 최소화, CI는 offscreen(`QT_QPA_PLATFORM=offscreen`) |

### 5-2. 단위 테스트 (순수 로직, 모델/IO 제외)

- `test_image_export.py`
  - `crop_array`: 박스대로 정확히 슬라이스(shape·픽셀 일치)
  - `save_image`: PNG/JPG 저장 후 재로드 시 크기·모드 일치, 잘못된 fmt 거부
- `test_image_capture.py` (가장 중요 — 슬라이스 핵심 조립 검증)
  - `FakeBackend`+`FakeFrameSource` 주입 → `make_crop_box`가 마스크 centroid 중심·종횡비·짝수·경계 클램프를 만족(기존 crop 함수와 조합 결과)
  - `export`: 가짜 소스 프레임을 tmp_path에 저장 → 결과 존재·크기 확인
  - **end-to-end(순수) happy path**: load_frame→make_crop_box→export 전 구간을 가짜 의존으로 관통
  - Protocol 계약: `isinstance(FakeBackend(), SegmentationBackend)` 통과 확인
- `ui/coords` 좌표 변환: 위젯↔이미지 스케일 역변환(레터박스/스케일 경우 경계값)

### 5-3. 통합 테스트 (happy path 100%, 조건부)

- `test_video_io_meta.py`: 작은 합성 영상/이미지 fixture로 `open_source().probe()`·`read_frame()` 검증. **PyAV 미설치/디코드 실패 시 `pytest.importorskip("av")` 로 skip**(CI 환경 무관성).
- SAM2 실모델 통합은 **이 슬라이스의 자동 테스트에서 제외**(CPU 1~3s/장 + 모델 다운로드). 대신 수동 스모크 절차를 HANDOFF에 기록(`python -m easy_capture` → 클릭 → 저장).

### 5-4. 커버리지 목표

- 이미지 모드 도메인은 L1~L2 수준(프로토타입~애그리거트). **순수 로직(crop·export·usecase·coords) 80%+**, IO/모델 래퍼는 통합/수동 스모크로 happy path 보장.

---

## 6. 작업 분해 (체크리스트 — 최소 구현 단위 커밋 순서)

> 순서 원칙(전역 지침): **인터페이스 → 테스트 → 구현 → 리팩터**. 각 줄 = 한 논리적 변경 = 한 커밋. 브랜치: `feature/image-mode/{기능}`.

### 도메인: export (core, 의존성 없음 — 먼저)
- [ ] `core/export/__init__.py` + `image_export.py` 인터페이스(시그니처·docstring·`ExportConfig`) — 우선순위: 높음
- [ ] `tests/test_image_export.py` 작성(crop_array·save_image) — 우선순위: 높음
- [ ] `image_export` 구현(Pillow) → 테스트 통과 — 우선순위: 높음

### 도메인: fixtures (테스트 기반)
- [ ] `tests/fixtures/fakes.py` — `FakeBackend`(SegmentationBackend 준수), `FakeFrameSource`(FrameSource 준수) — 우선순위: 높음

### 도메인: video_io (infra)
- [ ] `infra/video_io.py` 인터페이스(`FrameSource` Protocol, `FrameMeta`, `open_source` 팩토리 시그니처) — 우선순위: 높음
- [ ] `test_video_io_meta.py`(importorskip 가드, 합성 fixture) — 우선순위: 중간
- [ ] `_ImageFileSource`(Pillow) 구현 → 테스트 — 우선순위: 높음
- [ ] `_VideoFrameSource`(PyAV PTS 시크 1프레임 + RGB 정규화) 구현 → 통합 테스트(조건부) — 우선순위: 중간

### 도메인: segmentation 백엔드 (infra)
- [ ] `infra/sam2_image_backend.py` 골격(생성자 보관만, 지연 로드 _ensure_loaded 스텁) — 우선순위: 높음
- [ ] Protocol 계약 테스트(`isinstance ... SegmentationBackend`, FakeBackend로) — 우선순위: 높음
- [ ] `segment_image` 구현(transformers Sam2Model/Processor, post→bool 마스크) — 우선순위: 중간 (CPU 수동 스모크로 확인)

### 도메인: usecase (app)
- [ ] `app/__init__.py` + `image_capture.py` 인터페이스(`ImageCaptureUseCase`, `CropRequest`) — 우선순위: 높음
- [ ] `test_image_capture.py`(가짜 의존 주입, end-to-end 순수 happy path) — 우선순위: 높음
- [ ] `ImageCaptureUseCase` 구현(core 함수 조립: segment→centroid→aspect_lock→make_crop_box→export) → 테스트 통과 — 우선순위: 높음

### 도메인: UI
- [ ] `ui/coords.py`(위젯↔이미지 좌표 변환 순수 함수) + 테스트 — 우선순위: 중간
- [ ] `ui/frame_canvas.py`(set_frame/set_overlay/clicked) — 우선순위: 중간
- [ ] `ui/main_window.py`(파일열기·캔버스·저장·_SegWorker 비블로킹) — 우선순위: 중간
- [ ] `app/router.py`(모드선택→메인윈도, 조립 루트) — 우선순위: 중간
- [ ] `__main__.py` 라우터 연결 — 우선순위: 중간

### 마무리
- [ ] CPU 수동 스모크(`python -m easy_capture` 전체 흐름) 절차를 HANDOFF 기록 — 우선순위: 높음
- [ ] HANDOFF.md "현재 진행 상태" 갱신(이미지 슬라이스 완료) — 우선순위: 높음 (커밋-문서 동기화 필수)

> 의존성: export·fixtures → usecase 테스트 가능. video_io·backend는 usecase와 병렬 진행 가능(둘 다 Protocol 뒤에 숨음). UI는 usecase 완성 후. **임계 경로: export → fakes → usecase 인터페이스/테스트 → usecase 구현.**

---

## 7. 기술 결정사항

| 결정 | 근거 |
|---|---|
| `SegmentationBackend` Protocol은 core에, SAM2 구현은 infra에 | core의 torch/transformers 비의존 유지(클린 경계). DIP. |
| `FrameSource` Protocol 신설(infra) | app이 PyAV 구체 타입 대신 추상에 의존 → 가짜 소스로 테스트. `read_frame(index=0)`로 매개변수 최소화 |
| `app/` 유스케이스 레이어 신설 | UI에 도메인 조립 로직이 새는 것 방지(SRP). 아키텍처 §1에 이미 예고됨 |
| 모델 지연 로드(첫 segment 호출 시) | ADR 0007: 무거운 로드는 UI 표시 이후로. 첫 프레임 표시는 모델 없이 즉시 |
| 세그 호출은 워커 스레드 | 아키텍처 §5: CPU SAM2 1~3s 블로킹 → UI 프리징 방지 |
| 좌표 변환을 순수 함수(`coords.py`)로 분리 | PySide6 없이 단위 테스트 가능(테스트 가능성↑) |
| 종횡비 프리셋 1개로 시작 | 첫 슬라이스는 경계 관통 목적. 기존 `ASPECT_PRESETS` 그대로 사용, UI 확장은 다음 |

---

## 8. 리스크 및 완화책

| 리스크 | 영향 | 완화책 |
|---|---|---|
| **CPU SAM2 지연(1~3s/장) UX** | 클릭 후 멈춤처럼 보임 | ① 워커 스레드 + 진행 스피너/ "분석 중…" 표시 ② 모델 지연 로드(첫 클릭만 추가 지연, 1회 고지) ③ 첫 프레임 표시는 모델과 무관하게 즉시 |
| **PyAV 설치/ffmpeg 의존** | 영상 입력 불가 | ① 이미지 파일 경로(Pillow)는 PyAV 없이 동작 → 슬라이스 핵심은 PyAV 없이도 검증 ② `open_source` 팩토리로 이미지/영상 분기, 영상 실패 시 한국어 안내(error-handling §1) ③ 테스트는 `importorskip("av")` 가드 |
| **대용량 프레임 메모리(4K 등)** | 메모리 압박 | ① 단일 프레임만 보관(전체 디코드 금지, 아키텍처 §4) ② 표시용 QImage는 캔버스 크기로 다운스케일, 원본은 크롭/저장 시에만 사용 ③ 초고해상도 경고(error-handling §1)는 다음 슬라이스 |
| **모델 다운로드 실패/오프라인** | 첫 클릭 실패 | 재시도·한국어 안내(error-handling §6). 이 슬라이스는 수동 스모크에서 1회 다운로드 확인 |
| **빈/배경 클릭 → 빈 마스크** | centroid None | `centroid_of_mask` None 처리 → "대상을 인식하지 못했어요. 다시 클릭해 주세요"(error-handling §2) |
| **좌표 변환 오류(스케일/레터박스)** | 엉뚱한 곳 세그 | `coords` 순수 함수 + 경계값 단위 테스트로 사전 차단 |

---

## 9. 방법론 가이드 (이 슬라이스 적용)

- **클린 경계(부분 clean-arch)**: core=도메인(crop·export·세그 추상), infra=외부(PyAV·SAM2·device), app=유스케이스, ui=표현. 의존은 ui→app→core←infra(구현 주입). core는 외부 라이브러리 비의존.
- **DIP / 주입**: app은 `SegmentationBackend`·`FrameSource` Protocol에만 의존, 구체는 `router`(조립 루트)에서 주입. 테스트는 가짜 주입.
- **SOLID/클린코드**: 함수 20줄·매개변수 3개 이내(초과 시 dataclass로 묶음: `CropRequest`·`ExportConfig`·`FrameMeta`). DRY(crop 기하 재사용, 중복 인코딩 금지). KISS(첫 슬라이스는 최소 기능).
- **커밋 순서**: 인터페이스→테스트→구현→리팩터. Conventional Commits 한국어, 도메인 스코프(`feat(export):`, `feat(app):` 등). 커밋-문서 동기화(HANDOFF) 필수.

---

## 10. 완료 정의 (DoD)

- [ ] `python -m easy_capture` → 이미지 모드 → 이미지/영상 파일 열기 → 프레임 표시 → 클릭 → (CPU SAM2) 마스크 → 크롭 → PNG/JPG 저장이 수동 스모크로 동작
- [ ] 순수 로직(export·usecase·coords) 단위 테스트 80%+ 통과, 기존 20개 테스트 무회귀
- [ ] core가 torch/transformers/PySide6/av를 import하지 않음(경계 검증)
- [ ] HANDOFF.md 갱신 + 다음 슬라이스(타임라인/구간/GIF) 진입 가능 상태
```