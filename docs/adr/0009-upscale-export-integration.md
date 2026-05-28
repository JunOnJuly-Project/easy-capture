# ADR 0009 — 업스케일 export 통합: 옵션 백엔드 주입 + core 순수 변환 함수

- 상태: 채택
- 날짜: 2026-05-28

## 맥락

[ADR 0004](0004-upscaler-realesrgan-swinir.md)에서 SwinIR/Swin2SR·Real-ESRGAN을 `core/upscale` 공통 인터페이스로 추상화하기로 결정했다. [ADR 0008](0008-app-usecase-layer.md)에서는 `ImageCaptureUseCase.export`가 crop→save 흐름을 담당하는 유스케이스로 확정되었다. 이 두 결정을 잇는 "업스케일을 export 흐름 어디에, 어떻게 끼울 것인가"를 명문화해야 한다.

구체적으로 두 가지 설계 문제가 남아 있었다.

1. **통합 지점 결정**: 업스케일은 Swin2SR 모델 로드만 수 초가 걸리는 무거운 옵션 기능이다. "안 쓰면 비용 없음"을 보장하면서 `ImageCaptureUseCase.export` 흐름에 자연스럽게 끼워야 한다. UI 또는 infra에서 직접 호출하면 [ADR 0008](0008-app-usecase-layer.md)의 `core 비의존 불변식`과 `DIP`를 깨뜨린다.

2. **모델 출력 정규화 책임 소재**: Swin2SR는 `reconstruction` 텐서를 CHW float 범위로 반환한다. 이 텐서를 RGB uint8 `np.ndarray`로 변환하는 로직이 infra 구현 내부에 중복될 경우, 출력 범위·채널 가정이 달라지면 여러 곳을 수정해야 한다. 또한 infra에 묻혀 있으면 torch 없이 단위 테스트할 수 없다.

## 결정

### 1. `UpscaleBackend` Protocol — core/upscale 소유, infra 구현

```python
# core/upscale/backend.py
class UpscaleBackend(Protocol):
    def upscale(self, image: np.ndarray) -> np.ndarray:
        """RGB uint8 입력 → RGB uint8 출력. 배율은 repo(모델) 선택."""
```

- 인터페이스는 `core/upscale/`에 위치한다([ADR 0004](0004-upscaler-realesrgan-swinir.md) "core/upscale 공통 인터페이스"와 정합).
- Swin2SR 구현체(`Swin2SRBackend`)는 `infra/upscale/`에 위치한다. core는 torch·transformers를 import하지 않는다는 [ADR 0008](0008-app-usecase-layer.md)의 `core 비의존 불변식`을 유지한다.
- [ADR 0007](0007-cpu-dev-strategy.md)의 백엔드 추상화·지연 로드 원칙에 따라, 구현체는 첫 `upscale` 호출 시 모델을 지연 로드한다. 미사용 시 모델이 메모리에 올라가지 않는다.

### 2. `ImageCaptureUseCase.export`에 옵션 주입

```python
class ImageCaptureUseCase:
    def __init__(
        self,
        source: FrameSource,
        backend: SegmentationBackend,
        upscaler: UpscaleBackend | None = None,   # 옵션 주입
    ) -> None: ...

    def export(self, frame, box, target) -> None:
        """crop → (upscale, upscaler 있을 때만) → save."""
```

- `upscaler=None`이면 crop→save를 직행한다. 업스케일 미사용 경로에 분기 복잡도가 없다.
- `upscaler`가 주입되면 crop→upscale→save 순으로 실행한다.
- 조립(`router.py`)에서만 `Swin2SRBackend`를 생성하여 주입한다. UI는 `UpscaleBackend` Protocol을 직접 참조하지 않는다.
- "안 쓰면 안 부른다"는 crop-UX 분리 철학([ADR 0004](0004-upscaler-realesrgan-swinir.md))과 일관된다. 업스케일 미사용 시 기존 export 동작에 회귀가 없다.

### 3. `reconstruction_to_rgb_uint8` — core 순수 함수로 격리

```python
# core/upscale/convert.py
def reconstruction_to_rgb_uint8(tensor: "np.ndarray") -> np.ndarray:
    """Swin2SR reconstruction 출력(CHW float) → RGB uint8 변환.

    WHY: infra 구현마다 클램프·채널 순서 처리를 중복하면 모델 출력 범위
    가정이 바뀔 때 여러 파일을 수정해야 한다. 이 함수 한 곳만 고치면 된다.
    torch 비의존(numpy만 사용)이므로 단위 테스트에 venv 없이 검증 가능.
    """
```

- `core/upscale/convert.py`에 위치한다. torch 비의존(numpy만 사용)이므로 core 비의존 불변식을 만족한다.
- infra의 `Swin2SRBackend.upscale`는 모델 추론 후 이 함수를 호출하여 uint8 변환을 위임한다. 추론 전후 변환 로직이 분리되어 각각 독립 테스트 가능하다.
- Swin2SR `reconstruction` 텐서의 실제 출력 범위·채널 순서는 구현 단계 venv 설치 후 검증하며, 결과에 따라 이 함수의 구현을 조정한다(가정이 틀려도 수정 지점이 한 곳).

## 대안

**(a) UI/infra에서 직접 업스케일+저장**

UI 또는 infra 레이어가 crop 결과를 받아 직접 `Swin2SRBackend`를 호출하고 저장한다.

- 거부 이유: 도메인 오케스트레이션(crop→upscale→save 순서)이 표현·인프라 레이어에 혼입된다. [ADR 0008](0008-app-usecase-layer.md)의 SRP·DIP 결정을 역행하며, UI 없이 업스케일 흐름을 단위 테스트할 수 없다.

**(b) export를 항상 업스케일 경유**

`UpscaleBackend`에 "no-op 패스스루" 구현을 두고 항상 `upscale()`을 거치게 한다.

- 거부 이유: 업스케일을 쓰지 않는 경로에 불필요한 함수 호출·모델 객체가 개입된다. 무업스케일 사용자에게 의미 없는 복잡도를 강제하며, crop→save 직행 경로의 무회귀 보장이 모호해진다.

## 결과

### 긍정적 영향

- **무회귀 보장**: `upscaler=None`(기본값)이면 기존 crop→save 경로를 그대로 통과한다. 업스케일 슬라이스 미구현 단계에서도 기존 테스트가 깨지지 않는다.
- **테스트 가능성**: `FakeUpscaleBackend`를 주입하면 torch·transformers 없이 `export` 전체 흐름을 단위 테스트할 수 있다. `reconstruction_to_rgb_uint8`은 numpy만으로 독립 테스트 가능하다.
- **확장성**: Real-ESRGAN 등 추가 백엔드도 `UpscaleBackend` Protocol을 구현하면 `router.py` 주입 교체만으로 끼울 수 있다([ADR 0004](0004-upscaler-realesrgan-swinir.md) Real-ESRGAN 옵션 경로와 정합). 배율(x2/x4)은 모델 repo 선택으로 결정한다.
- **단일 수정 지점**: 모델 출력 형식 가정이 바뀌어도 `reconstruction_to_rgb_uint8` 한 곳만 수정한다.

### 부정적 영향 / 트레이드오프

- **router.py 수정 필요**: 업스케일 활성화 시 `router.py`에서 `Swin2SRBackend`를 생성해 주입하는 코드가 추가된다. 의존성 주입 컨테이너 없이 수동 조립이므로 백엔드 종류가 늘면 라우터가 비대해질 수 있다(현재 규모에서는 허용 가능).
- **Swin2SR 출력 형식 미확정**: `reconstruction_to_rgb_uint8`의 CHW float 가정은 구현 단계 venv 검증 전까지 잠정적이다. 형식이 다를 경우 함수 구현을 수정해야 하나, 수정 범위는 이 함수 하나로 한정된다.
