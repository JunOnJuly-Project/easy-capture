# ADR 0008 — app/ 유스케이스 레이어 신설

- 상태: 채택
- 날짜: 2026-05-28

## 맥락

스캐폴딩(커밋 `49de89d`)에는 `core/`, `infra/`, `ui/` 세 레이어만 존재한다. [아키텍처 문서 §1](../architecture.md)에 Application 레이어(`app/`)가 예고되어 있었으나, 실제 디렉터리와 ADR 명문화는 없었다.

이 상태에서 이미지 모드 첫 수직 슬라이스([image-mode-slice.md](../plans/image-mode-slice.md))를 구현하면 다음 문제가 발생한다.

1. **SRP 위반**: "클릭 → 세그먼테이션 → centroid 산출 → 크롭 박스 계산 → 내보내기"의 도메인 오케스트레이션 흐름을 묶을 레이어가 없어, 해당 로직이 UI(`MainWindow`)로 새어 들어간다.
2. **테스트 곤란**: 오케스트레이션 로직이 `QMainWindow` 안에 있으면 PySide6를 전체 로드해야 테스트할 수 있다. 가짜 백엔드/소스를 주입하는 격리 단위 테스트를 작성하기 어렵다.
3. **DIP 불이행**: UI가 `SegmentationBackend` 구현체(SAM2)나 `FrameSource` 구현체(PyAV)를 직접 생성·조립하면, UI가 infra 구체 타입에 의존하게 된다.

## 결정

`src/easy_capture/app/` 레이어를 신설한다.

### 구조

```
app/
├── __init__.py
├── router.py          # 조립 루트(composition root): 모드선택→메인윈도 라우팅, 의존성 주입
└── image_capture.py   # ImageCaptureUseCase: 이미지 모드 오케스트레이션
```

### 의존 방향

```
ui  ──▶  app  ──▶  core  (Protocol/추상)
                    ▲
         infra  ────┘  (구현 주입)
```

- **ui → app**: UI는 `ImageCaptureUseCase`만 호출. core/infra를 직접 참조하지 않는다.
- **app → core**: 유스케이스는 `SegmentationBackend`·`FrameSource` Protocol(추상)에만 의존. 구체 타입 import 금지.
- **infra → core**: 구현체(`Sam2ImageBackend`, `_VideoFrameSource`)가 core Protocol을 만족. core는 infra를 역방향으로 참조하지 않는다.
- **core 비의존 불변식**: core는 `torch`, `transformers`, `PySide6`, `av`를 import하지 않는다. `Pillow`(순수 이미지 인코딩)만 허용.

### 핵심 컴포넌트

```python
class ImageCaptureUseCase:
    def __init__(self, source: FrameSource, backend: SegmentationBackend) -> None:
        """프레임 공급원·세그 백엔드를 Protocol 타입으로 주입(DIP)."""

    def load_frame(self) -> np.ndarray: ...
    def make_crop_box(self, frame, request: CropRequest) -> tuple[int, int, int, int]: ...
    def export(self, frame, box, target) -> None: ...
```

구체 의존성 조립은 `app/router.py`(`AppRouter._on_mode`)에서만 수행한다. 테스트에서는 `FakeBackend`·`FakeFrameSource`로 치환해 모델/IO 없이 end-to-end 유스케이스를 검증한다.

## 대안

**(a) UI에서 직접 조립** — `MainWindow`가 `Sam2ImageBackend`·`_VideoFrameSource`를 생성하고 `core` 함수를 순서대로 호출한다.
- 거부 이유: 도메인 오케스트레이션이 표현 레이어에 혼입되어 SRP를 위반한다. PySide6 없이 오케스트레이션 로직을 단위 테스트할 수 없다.

**(b) core에 오케스트레이션 추가** — `core/` 안에 `orchestrate_image_capture()` 같은 함수를 두어 흐름을 조립한다.
- 거부 이유: core가 `FrameSource`(infra 구현에 의존하는 팩토리) 또는 IO 호출을 포함하게 되면 "core = 외부 라이브러리 비의존" 불변식이 깨진다. 또한 core의 책임(도메인 로직)이 오케스트레이션(흐름 조립)과 뒤섞인다.

## 결과

### 긍정적 영향

- **테스트 가능성**: `FakeBackend`·`FakeFrameSource` 주입으로 torch/PySide6/PyAV 없이 `ImageCaptureUseCase` end-to-end 단위 테스트 가능. 순수 로직 80%+ 커버리지 달성 경로가 열린다.
- **확장성**: 비디오 모드 유스케이스(`VideoCaptureUseCase`)·GIF 모드 유스케이스도 같은 `app/` 레이어에 추가하면 된다. 의존 구조 변경 없이 확장.
- **명확한 경계**: UI는 유스케이스 API만 알면 된다. core/infra 교체(예: 경량 백엔드로 전환, [ADR 0007](0007-cpu-dev-strategy.md))가 UI 코드에 영향을 주지 않는다.
- **아키텍처 문서와 정합**: [architecture.md §1](../architecture.md)에서 예고한 레이어 구조가 코드에 실현되어 문서와 구현의 일치성이 확보된다.

### 부정적 영향 / 트레이드오프

- **파일 수 증가**: `app/__init__.py`, `router.py`, `image_capture.py` 3개 파일이 추가된다. 작은 프로젝트 초기에는 over-engineering으로 보일 수 있다.
- **팩토리 부담**: 조립 루트(`router.py`)가 구체 타입을 알아야 하므로, 새 infra 구현체 추가 시 라우터도 수정해야 한다. 의존성 주입 컨테이너 없이 수동 조립이므로 규모 확장 시 관리 비용이 생긴다(현재 규모에서는 허용 가능).

---

## 보완 (2026-05-28, 구현 확정)

본 ADR에서 `FrameSource` Protocol을 core 추상으로 두어 UI와 infra가 직접 의존하지 않게 한다는 원칙을 명시했다. 초기 구현에서 `FrameSource`·`FrameSpan`·`FrameMeta`가 `infra/video_io.py`에 임시 위치했으나, **이후 `core/source/frame_source.py`로 이동**되어 본 ADR의 원안(core 추상)과 정합이 확보되었다.

- `core/source/frame_source.py`: `FrameSource`·`FrameSpan`·`FrameMeta` Protocol/데이터클래스 (torch·PyAV·PySide6 비의존)
- `infra/video_io.py`: `FrameSource` Protocol 구현체(Pillow 이미지·PyAV 영상) — core를 역방향으로 참조하지 않는 의존 방향 유지

이로써 "app → core(Protocol/추상)만 의존, infra → core(Protocol 구현)" 불변식이 `FrameSource` 계층에서도 성립한다.
