# ADR 0012 — 컷 재검출용 DetectionBackend Protocol 신설

- 상태: 채택
- 날짜: 2026-05-28

## 맥락

[ADR 0006](0006-shot-boundary-reid.md)은 컷 경계에서 Grounding DINO로 후보를 재검출한 뒤 재매칭 점수로 SAM2를 재초기화한다는 전략을 채택했다. [샷경계 재추적 계획서](../plans/video-shot-retrack.md) §2-1이 이 전략을 구현 단계에서 구체화하면서 두 가지 충돌이 드러났다.

1. **책임 범위 불일치**: [ADR 0010](0010-video-segmentation-backend.md)의 `VideoSegmentationBackend`는 단일 샷 프레임 시퀀스의 마스크 전파를 담당한다. "컷 직후 1프레임에서 후보 bbox를 열거한다"는 재검출 책임은 세그멘테이션(마스크)과 다르다. 같은 Protocol에 검출 메서드를 추가하면 ADR 0010이 ISP를 지키기 위해 분리한 경계를 다시 허문다.

2. **core 비의존 불변식 위협**: Grounding DINO 구현은 `torch`·`transformers`에 의존한다. 이 의존성이 `VideoSegmentationBackend` 구현체 안으로 병합되면, Protocol 자체나 core 추상이 무거운 ML 의존성을 끌어올 위험이 생긴다([ADR 0008](0008-app-usecase-layer.md) core 불변식 계승).

ADR 0010이 이미지 세그와 비디오 세그를 ISP 근거로 분리한 선례가 있다. 재검출은 그보다 더 다른 책임이므로 별도 Protocol 신설이 자연스러운 귀결이다.

## 결정

`core/segmentation/detection_backend.py`에 **`DetectionBackend`를 독립 Protocol로 신설**한다. Grounding DINO 구현체는 `infra/grounding_dino_backend.py`에 위치한다.

### Protocol 시그니처

```python
# core/segmentation/detection_backend.py  (torch / transformers 비의존)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2) — rematch.py Box와 동일 alias


@dataclass(frozen=True)
class Detection:
    """재검출 후보 1개(불변). 위치(box) + 선택적 외형특징(feat).

    box:   (x1, y1, x2, y2) 픽셀 좌표 — rematch_score pos_sim 입력.
    score: 검출기 자체 신뢰도(후보 정렬·필터용, 재매칭 점수와 별개).
    feat:  외형 임베딩(있으면 rematch_score cls_sim 입력, 없으면 None=위치만 평가).
    """

    box: Box
    score: float
    feat: np.ndarray | None = field(default=None, compare=False)


@runtime_checkable
class DetectionBackend(Protocol):
    """컷 직후 단일 프레임에서 동일 클래스 후보를 재검출하는 추상(테스트 치환용).

    VideoSegmentationBackend와의 차이:
      - 무상태(stateless): opaque session 불필요 — 1프레임 입력 → 후보 리스트 출력.
      - 모델 로드는 무거우므로 지연 로드(_ensure_loaded) 패턴은 구현체 측에서 계승.
    """

    device: str

    def detect(self, frame: np.ndarray, prompt: str) -> list[Detection]:
        """프레임에서 prompt 클래스에 해당하는 후보들을 검출한다(무거움 — GPU).

        Args:
            frame:  RGB HxWx3 uint8 단일 프레임 (컷 직후 첫 프레임).
            prompt: 텍스트 프롬프트 (기본 'person', 다중 클래스 확장 대비 인자화).

        Returns:
            Detection 리스트, score 내림차순 정렬. 검출 없으면 빈 리스트.
        """
        ...
```

### 핵심 설계 결정

- **stateless 설계(SAM2와의 차이)**: 검출은 1프레임 입력 → 후보 리스트 출력의 단발 호출이다. `VideoSegmentationBackend`의 `init_session → add_click → propagate` 상태 전이가 필요 없다. opaque session을 반환하지 않으므로 app 레이어에 session 수명 관리 부담을 주지 않는다.

- **`Detection` frozen dataclass로 후보 묶기**: bbox·score·feat를 개별 반환값이나 튜플로 흩뿌리지 않고 단일 값 객체로 묶는다. `TrackResult`·`SegmentResult`·`RematchResult` 패턴 계승. `feat=None` 허용으로 위치 기반 재매칭과 외형특징 기반 재매칭을 동일 인터페이스에서 처리한다.

- **`feat=None`이면 위치(IoU)만 평가**: `select_best_match`(`core/tracking/rematch.py`)가 `Detection.feat`를 `rematch_score`의 `cand_feat` 인자로 전달한다. `feat`가 None이면 기존 `rematch_score` 분기(`cls_sim` 계산 생략)가 그대로 작동한다. 이번 구현은 위치 기반이 기본이며, feat는 정확도 부족 확인 시 후속 보강을 위한 확장점만 열어둔다(KISS).

- **`Box` alias 공유**: `rematch.py`의 `Box` 타입 alias와 동일하게 `(x1, y1, x2, y2) float` 4-튜플로 정의한다. core 내부 타입이 일관되어 임피던스 불일치가 없다.

- **`@runtime_checkable`**: `isinstance(FakeDetectionBackend(...), DetectionBackend)` 계약 테스트를 가능하게 한다. `FakeVideoBackend` 선례와 동일.

- **infra에 구현체 위치**: `infra/grounding_dino_backend.py`가 `DetectionBackend`를 구현한다. [ADR 0007](0007-cpu-dev-strategy.md) 지연 로드 원칙 계승(`_ensure_loaded()` — 첫 `detect` 호출 시 모델 로드). 자동 CI 완전 제외(import 자체 회피), 실추론은 Colab 후행.

- **`VideoCaptureUseCase` 생성자 확장**: `detector: DetectionBackend | None = None` 파라미터를 추가한다. `detector=None`이면 컷 감지·재검출을 건너뛰고 첫 슬라이스의 단일 샷 경로로 동작한다(하위호환). 자세한 오케스트레이션은 [계획서](../plans/video-shot-retrack.md) §4를 참조한다.

- **테스트 더블**: `tests/fixtures/fakes.py`의 `FakeDetectionBackend`가 Protocol을 준수하며 torch/transformers 없이 시나리오별 후보 리스트 주입·`detect_call_count` 스파이를 제공한다. `FakeVideoBackend`의 검출판.

## 대안

**(a) VideoSegmentationBackend에 detect 메서드 추가**

`VideoSegmentationBackend`에 `detect(frame, prompt)` 메서드를 추가한다.

- 거부 이유: ADR 0010이 ISP를 근거로 이미지/비디오 Protocol을 분리했는데, 동일한 Protocol에 책임이 다른 검출 메서드를 추가하면 그 분리 근거를 스스로 허문다. `FakeVideoBackend`가 마스크 전파 역할과 검출 역할을 함께 구현해야 해 테스트 더블이 비대해진다. 향후 YOLO·전용 re-ID 등 다른 검출기로 교체할 때 `VideoSegmentationBackend` 전체를 교체해야 하므로 OCP도 위반한다.

**(b) SAM2 자체 후보 열거 사용**

SAM2의 automatic mask generator를 재검출에 활용한다.

- 거부 이유: SAM2 automatic mask generator는 텍스트 프롬프트 없이 전 픽셀 후보를 생성하므로 "같은 클래스(person)의 후보 열거"에 적합하지 않다. SAM2는 프롬프트 기반 추적 모델이며, 클래스 인식 자동 후보 열거는 Grounding DINO처럼 CLIP 기반 텍스트-비전 모델에 더 어울린다(ADR 0003, ADR 0006 근거 계승). automatic mask generator를 쓰면 후보 수가 폭발적으로 많아져 재매칭 연산 비용이 급증한다.

## 결과

### 긍정적 영향

- **ISP 준수**: `VideoSegmentationBackend`는 마스크 전파 책임만, `DetectionBackend`는 후보 검출 책임만 진다. 검출기 교체(YOLO·전용 re-ID) 시 `DetectionBackend`만 교체하면 된다(OCP).
- **GPU 블로커 우회(두 번째)**: `FakeDetectionBackend`로 GPU 없이 컷 재매칭 오케스트레이션 전체를 CPU 단위 테스트로 검증할 수 있다. Grounding DINO 실추론은 Colab 후행([ADR 0007](0007-cpu-dev-strategy.md) 이중 경로 원칙 계승). 이로써 이번 슬라이스에서 GPU에 새로 묶이는 컴포넌트가 하나 더 늘었지만, Protocol 격리 덕분에 재추적 오케스트레이션 전체를 CPU에서 결정적으로 검증할 수 있다.
- **`detect_call_count` 회귀 가드**: `FakeDetectionBackend` 스파이 카운터로 "검출은 컷 경계마다 정확히 1회만" 불변식을 테스트 수준에서 강제한다. `propagate_call_count`(비디오)·`segment_call_count`(이미지) 가드의 검출판.
- **core 비의존 불변식 유지**: `detection_backend.py`는 numpy만 의존하며 torch·transformers·PyAV·PySide6를 import하지 않는다.
- **threshold·가중치 보정 분리**: `REMATCH_THRESHOLD=0.5`·`w_pos=0.7`·`w_cls=0.3` 기본값은 `core/tracking/rematch.py`에 상수로 고정되어 있다. Colab H2 실추론 후 실측 기반으로 보정하며, Protocol 인터페이스 변경 없이 상수 값만 갱신한다.

### 부정적 영향 / 트레이드오프

- **Protocol 파일 추가**: `core/segmentation/` 안에 `detection_backend.py`가 새로 생긴다. ADR 0010과 동일한 트레이드오프.
- **두 번째 GPU 미검증 레이어**: SAM2 video 실추론(첫 슬라이스 미검증)에 이어 Grounding DINO 실추론도 Colab 후행으로 쌓인다. Fake 위에 코드가 누적되는 리스크는 PoC `h2_cut_retrack.py` 검증 패턴을 `infra/grounding_dino_backend.py` 골격 주석에 정확히 매핑하고, Colab 검증을 두 슬라이스 통합 1회로 수행해 관리한다([계획서](../plans/video-shot-retrack.md) §8-1 리스크 1).
- **`feat=None` 1차 구현의 정확도 한계**: 위치(IoU) 기반 재매칭만으로는 인물이 겹치거나 구도가 크게 바뀐 컷에서 오탐 가능성이 있다. `Detection.feat` 확장점을 열어두었으므로 Colab 검증 후 정확도 부족 시 외형 임베딩을 추가한다(ADR 0006 cls_sim 보조 신호).

### 후속 연계

- [ADR 0006](0006-shot-boundary-reid.md) — 이 ADR이 ADR 0006의 "Grounding DINO 재검출"을 아키텍처로 구체화한다. ADR 0006 보완 섹션(2026-05-28) 참조.
- [ADR 0010](0010-video-segmentation-backend.md) — `VideoSegmentationBackend` 분리 선례 및 ISP 근거 제공.
- [ADR 0007](0007-cpu-dev-strategy.md) — 지연 로드·이중 경로(CPU Fake + Colab GPU) 원칙 계승.
- [ADR 0003](0003-grounding-dino-labeling.md) — Grounding DINO 채택 근거. 이번 ADR은 그 사용 범위를 라벨링 보조에서 재추적 재검출로 확장한다.
- [샷경계 재추적 계획서](../plans/video-shot-retrack.md) §2·§9: 이 ADR이 "ADR 0012 후보(DetectionBackend 신설)" 트리거를 해소한다.
