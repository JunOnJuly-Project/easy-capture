# ADR 0010 — 비디오 세그먼테이션 백엔드 Protocol 분리

- 상태: 채택 (2026-05-29 [ADR 0014]로 `add_box` 확장)
- 날짜: 2026-05-28

## 맥락

[ADR 0007](0007-cpu-dev-strategy.md)은 `SegmentationBackend`에 비디오 메서드(`init_video_session` / `propagate`)를 `supports_video()` 플래그와 Optional 메서드 조합으로 예고했다. 이 설계가 실제 구현 단계에 접어들자 두 가지 문제가 뚜렷해졌다.

1. **이미지 전용 백엔드 오염**: `Sam2ImageBackend`, 향후 MobileSAM·EdgeSAM 등 이미지 전용 구현체가 비디오 메서드를 빈 몸통(`raise NotImplementedError`)으로 채워야 한다. 쓰지 않는 메서드를 구현하도록 강제하므로 ISP(Interface Segregation Principle) 위반이다.

2. **호출자 분기 복잡도**: `supports_video()`를 확인하고 `hasattr`/플래그 분기를 거쳐야 비디오 메서드를 호출할 수 있다. 분기가 프로토콜 사용 측(유스케이스·테스트)에 분산되어 LSP가 약해진다.

[비디오 모드 첫 수직 슬라이스 계획서](../plans/video-tracking-slice.md) §2-1·§8은 이 이유로 별도 Protocol 신설을 명시하고, ADR 작성을 documenter에게 위임했다.

## 결정

`core/segmentation/video_backend.py`에 **`VideoSegmentationBackend`를 독립 Protocol로 신설**한다.

### Protocol 시그니처

```python
# core/segmentation/video_backend.py  (torch / transformers 비의존)
from typing import Protocol, runtime_checkable
import numpy as np

class EmptyTrackError(Exception):
    """전파 결과 전 프레임이 빈 마스크일 때(대상 추적 실패).
    이미지 모드 EmptyMaskError와 동일한 명시적 예외 계약.
    """

@runtime_checkable
class VideoSegmentationBackend(Protocol):
    """단일 샷 프레임 시퀀스 전파 추적 추상.

    호출 계약: init_session → add_click(frame_idx=0) → propagate.
    반환 마스크: bool HxW (centroid_of_mask 계약과 동일).
    테스트에서는 FakeVideoBackend로 치환하여 torch/transformers 비의존 검증.
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

### 핵심 설계 결정

- **session을 opaque object로 격리**: transformers의 `Sam2VideoInferenceSession` 타입을 core/app에 노출하지 않는다. app 레이어는 `object`로만 들고 다니므로 transformers import가 core 경계를 넘지 않는다.

- **3-메서드 분리(OCP)**: `add_click`을 `init_session`과 별도 메서드로 분리하면 향후 다중 클릭·다중 객체(후속 슬라이스)를 Protocol 변경 없이 확장할 수 있다.

- **매개변수 3개 규칙 준수**: `add_click(session, point)` 2개, `propagate(session)` 1개.

- **기존 `SegmentationBackend` 무변경**: 이미지 전용 Protocol은 그대로 둔다. `supports_video()`는 이미지 백엔드의 "비디오 미지원" 표식으로 유지하되, 비디오 유스케이스(`VideoCaptureUseCase`)는 `VideoSegmentationBackend`만 주입받는다.

- **infra에 구현체 위치**: `infra/sam2_video_backend.py`가 `VideoSegmentationBackend`를 구현한다. [ADR 0007](0007-cpu-dev-strategy.md) 지연 로드 원칙을 계승한다(`_ensure_loaded()` — 첫 `init_session` 호출 시 모델 로드).

- **테스트 더블**: `tests/fixtures/fakes.py`의 `FakeVideoBackend`가 Protocol을 준수하며 torch/transformers 없이 드리프트·빈 마스크·스파이 카운터를 제공한다.

## 대안

**(a) ADR 0007 원안 유지 — 단일 Protocol에 Optional 비디오 메서드**

`SegmentationBackend`에 `supports_video()` + `init_video_session`/`propagate`를 Optional로 남긴다.

- 거부 이유: 이미지 전용 백엔드가 쓰지 않는 메서드를 빈 몸통으로 구현해야 하므로 ISP 위반이다. 호출 측이 `supports_video()` 분기를 매번 삽입해야 하며, 분기 누락 시 런타임 오류가 발생한다. `FakeVideoBackend`도 이미지 메서드까지 함께 구현해야 해 테스트 더블이 무거워진다.

**(b) 추상 클래스 상속**

`SegmentationBackend`를 ABC로 바꾸고 `VideoSegmentationBackend`를 서브클래스로 정의한다.

- 거부 이유: Protocol의 구조적 서브타이핑(structural subtyping) 이점을 잃는다. `FakeVideoBackend`가 추상 클래스 상속을 강제받아 테스트 더블 구현 비용이 올라가고, `runtime_checkable`을 통한 `isinstance` 계약 검증도 불가능해진다.

## 결과

### 긍정적 영향

- **ISP 준수**: 이미지 전용 백엔드는 비디오 메서드를 알지 못한다. 새 경량 이미지 백엔드 추가 시 `VideoSegmentationBackend` 구현 의무가 없다.
- **GPU 블로커 우회**: `FakeVideoBackend`로 GPU 없이 추적 오케스트레이션(`VideoCaptureUseCase.track → compute_boxes → export`) 전체를 CPU 단위 테스트로 검증할 수 있다. SAM2 video 실추론은 Colab 후행 검증([ADR 0007](0007-cpu-dev-strategy.md) 이중 경로 원칙 계승).
- **`propagate_call_count` 회귀 가드**: `FakeVideoBackend` 스파이 카운터로 "무거운 전파를 구간당 1회만 호출" 불변식을 테스트 수준에서 강제한다. 이미지 모드 `segment_call_count` 가드의 비디오판.
- **core 비의존 불변식 유지**: `video_backend.py`는 numpy만 의존하며 torch·transformers·PyAV·PySide6를 import하지 않는다([ADR 0008](0008-app-usecase-layer.md) core 불변식 계승).

### 부정적 영향 / 트레이드오프

- **Protocol 파일 추가**: `core/segmentation/` 안에 `video_backend.py`가 새로 생긴다. 작은 프로젝트 초기에는 파일 수 증가처럼 보일 수 있다.
- **SAM2 video 실추론 미검증**: Protocol 분리 자체는 CPU에서 완전 검증되나, `Sam2VideoInferenceSession` opaque 객체의 실제 동작은 Colab GPU에서 후행 확인한다. PoC `h1_track.py` 시그니처 매핑 주석을 `infra/sam2_video_backend.py` 골격에 남겨 불일치 리스크를 관리한다.

### 후속 연계

- [ADR 0011](0011-video-export-location.md) — GIF/MP4 인코딩 위치 결정(같은 비디오 슬라이스 패키지).
- [비디오 슬라이스 계획서](../plans/video-tracking-slice.md) §5 임계 경로: `VideoSegmentationBackend` Protocol → `FakeVideoBackend` → `VideoCaptureUseCase` 순 구현.
- 샷경계 재추적(후속 슬라이스): `VideoSegmentationBackend`는 단일 샷 가정. 멀티 샷 재추적 시 `init_session`을 컷마다 재호출하는 패턴으로 Protocol 변경 없이 확장 가능(OCP).
- [ADR 0014](0014-box-prompt-mask-refine.md) — `VideoSegmentationBackend`에 `add_box`(box 프롬프트) 추가. 3-메서드 분리(add_click 별도) 원칙의 연장으로 ISP/OCP 정합. `add_click` 무변경. 본 ADR을 확장(Superseded 아님).
