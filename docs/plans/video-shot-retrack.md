# 비디오 모드 "샷경계 재추적" 슬라이스 — 개발 계획

> 작성: 2026-05-28 · 단계: `/develop` 파이프라인 planner · 관련: [아키텍처](../architecture.md) · [데이터플로우](../data-flow.md) · [ADR 0006](../adr/0006-shot-boundary-reid.md) · [ADR 0007](../adr/0007-cpu-dev-strategy.md) · [ADR 0010](../adr/0010-video-segmentation-backend.md) · [비디오 첫 슬라이스](video-tracking-slice.md) · PoC: [`poc/h2_cut_retrack.py`](../../poc/h2_cut_retrack.py)

---

## 0. 이 슬라이스의 한 줄 정의

비디오 첫 슬라이스가 가정했던 **"단일 샷"을 푼다.** 구간 안에 샷 전환(컷)이 있을 때:

```
구간 프레임 ─▶ [PySceneDetect] 컷 프레임 인덱스 리스트 (infra, CPU 검증 가능)
            ─▶ 컷으로 구간을 "샷"들로 분할
            ─▶ 샷마다 SAM2 video track (무거움)
            ─▶ 샷 경계에서: 직전 샷 마지막 bbox vs 컷 직후 Grounding DINO 후보들
               · rematch_score(w_pos=0.7·w_cls=0.3) 계산 (core 순수)
               · max ≥ threshold(0.5) → SAM2 재초기화·object_id 유지 → 추적 이어짐
               · 미달 → "재매칭 실패(needs_correction)" 플래그 (수동 교정 UI는 다음 슬라이스)
            ─▶ 컷이 섞인 구간에서도 끊기지 않는 추적 → 크롭 → GIF/MP4
```

**핵심 가치**: 뮤직비디오·직캠은 1~2초마다 컷이 바뀐다(ADR 0006 맥락, 덕후 페르소나 치명적 지적). 컷을 넘어 같은 인물을 자동으로 이어 잡지 못하면 실사용 가치가 급감한다. 이 슬라이스가 그 척추를 CPU에서(Fake로) 관통시킨다.

### GPU 블로커 대응 원칙 (제1 제약 — 이중 경로 계승)

비디오 첫 슬라이스와 동일하게 **GPU 의존을 Protocol 뒤로 격리하고 CPU에서 오케스트레이션을 완성**한다. 이번 슬라이스는 GPU 의존이 **둘로 늘었다**:

| 컴포넌트 | GPU 필요? | 검증 경로 |
|---|---|---|
| **SAM2 video** (`Sam2VideoBackend`) | 🔴 필요 (CPU ≈0.10fps) | Protocol 뒤 격리, `FakeVideoBackend` CPU 테스트, 실추론 Colab 후행 |
| **Grounding DINO 재검출** (`Sam2VideoBackend`→신규 `DetectionBackend`) | 🔴 필요 | **신규** `DetectionBackend` Protocol 뒤 격리, `FakeDetectionBackend` CPU 테스트, 실추론 Colab 후행 |
| **PySceneDetect 컷 감지** (`infra/shot_detect.py`) | 🟢 **CPU 가능** | 합성 클립으로 **CPU 통합 테스트 가능** (조건부 importorskip) |
| **rematch_score·best 선택·threshold 판정** (`core/tracking/rematch.py`) | 🟢 **CPU·순수** | 순수 단위 테스트로 **완전 검증** |
| **샷 분할·경계 재매칭 오케스트레이션** (`app/video_capture.track` 확장) | 🟢 **CPU·Fake** | `FakeVideoBackend`+`FakeDetectionBackend`로 **컷 시나리오 전부 검증** |

> 결론: 이번 슬라이스에서 **새로 GPU에 묶이는 것은 Grounding DINO 재검출 하나뿐**이며, 그것마저 `DetectionBackend` Protocol 뒤로 밀어내면 **재추적 오케스트레이션 전체를 CPU에서 결정적으로 검증**할 수 있다. PySceneDetect는 CPU에서 그대로 돌아가므로 컷 감지 정확도도 합성 클립으로 검증 가능하다. GPU 미검증 위에 쌓이는 위험은 §7-1에서 별도로 다룬다.

---

## 1. 슬라이스 범위 (In / Out)

| 구분 | 포함 (In) | 제외 (Out → 후속, §8) |
|---|---|---|
| 컷 감지 | PySceneDetect `ContentDetector` 기본 → 컷 프레임 인덱스 리스트(infra) | 적응형 임계값 튜닝 UI, FadeDetect/ThresholdDetect 선택 |
| 재검출 | 컷 직후 프레임에서 `DetectionBackend`로 후보 bbox(+feat) 리스트 | 클릭 없는 전구간 자동 검출, 다중 인물 동시 추적 |
| 재매칭 | `rematch_score` 후보 평가 → best 선택 → threshold(0.5) 판정 (core 순수) | 가중치/임계값 사용자 조정 UI, pose matching 보조 신호 |
| 재초기화 | 통과 시 SAM2 새 세션 재초기화·`object_id` 유지(다음 샷 이어 추적) | re-ID 전용 네트워크(ADR 0006 v1.1) |
| 실패 처리 | 미달 시 **`needs_correction` 플래그만** (추적 중단·UI 표시) | **수동 교정 UI**(박스 드래그·재클릭) — 다음 슬라이스 |
| 출력 | 컷 섞인 구간 → 끊김 없는 추적 → 크롭 → GIF/MP4 (기존 export 재사용) | 오디오 동기, 업스케일 결합 |

> 의도적 최소화: 이번 슬라이스 목적은 **"컷을 넘는 자동 재추적"의 척추 관통**이다. 미달 구간 **교정 UI는 만들지 않고 플래그까지만** 한다(ADR 0006 "수동 교정 유도"의 1단계). 교정 UI는 후속.

---

## 2. 산출물 1 — `DetectionBackend` Protocol · `FakeDetectionBackend` · 샷 감지(infra)

### 2-1. 결정: 신규 `DetectionBackend` Protocol 신설 (ISP·DIP — ADR 트리거)

비디오 첫 슬라이스가 `SegmentationBackend`(이미지)와 `VideoSegmentationBackend`(비디오)를 **별도 Protocol로 분리**(ADR 0010, ISP)한 선례를 그대로 계승한다. 재검출(Grounding DINO)은 **세그멘테이션과 책임이 다르다**(마스크가 아니라 후보 박스+외형특징을 검출). 따라서 `VideoSegmentationBackend`에 검출 메서드를 끼워 넣지 않고 **별도 `DetectionBackend` Protocol을 신설**한다.

- **위치**: `core/segmentation/detection_backend.py` (core가 추상 소유 — torch/transformers 비의존 유지). 구현은 `infra/grounding_dino_backend.py`.
- **근거(ISP·OCP)**: SAM2 video 백엔드가 쓰지 않는 검출 메서드로 오염되지 않게 한다. 향후 다른 검출기(YOLO·전용 re-ID)로 교체해도 Protocol만 구현하면 된다.
- **session 불필요(SAM2와 차이)**: 검출은 **1프레임 입력 → 후보 리스트 출력**의 무상태(stateless) 호출이다. opaque session이 필요 없다. 단 모델 로드는 무겁다 → 지연 로드(`_ensure_loaded`)는 동일하게 계승.
- **→ ADR 트리거**: "재검출 백엔드 Protocol 신설(`DetectionBackend`, core 추상·infra 구현)"는 ADR 0006의 "Grounding DINO 재검출"을 아키텍처로 구체화하는 결정이다. ADR 0012 후보로 기록(작성은 documenter, §9).

### 2-2. Protocol 시그니처 (계획 — 함수 20줄·매개변수 3개 준수)

```python
# core/segmentation/detection_backend.py  (core, torch/transformers 비의존)
from typing import Protocol, runtime_checkable
import numpy as np

Box = tuple[float, float, float, float]   # (x1, y1, x2, y2) — rematch.py Box와 동일 alias

@dataclass(frozen=True)
class Detection:
    """재검출 후보 1개(불변). 위치(box) + 선택적 외형특징(feat).

    box:  (x1, y1, x2, y2) 픽셀 좌표 — rematch_score pos_sim 입력.
    score: 검출기 자체 신뢰도(후보 정렬·필터용, 재매칭 점수와 별개).
    feat: 외형 임베딩(있으면 rematch_score cls_sim 입력, 없으면 None=위치만 평가).
    """
    box: Box
    score: float
    feat: np.ndarray | None = None

@runtime_checkable
class DetectionBackend(Protocol):
    """컷 직후 단일 프레임에서 동일 클래스 후보를 재검출하는 추상(테스트 치환용).

    무상태(stateless): session 없음 — 1프레임 입력 → 후보 리스트 출력.
    모델 로드는 무거우므로 지연 로드(Sam2VideoBackend._ensure_loaded 패턴 계승).
    """
    device: str

    def detect(self, frame: np.ndarray, prompt: str) -> list[Detection]:
        """프레임에서 prompt(예: 'person')에 해당하는 후보들을 검출한다(무거움 — GPU).

        Args:
            frame:  RGB HxWx3 uint8 단일 프레임 (컷 직후 첫 프레임).
            prompt: Grounding DINO 텍스트 프롬프트(기본 'person').

        Returns:
            Detection 리스트(검출 없으면 빈 리스트). score 내림차순 정렬.
        """
        ...
```

- **매개변수 3개 규칙**: `detect(frame, prompt)` 2개로 만족. `prompt`는 향후 다중 클래스 확장 대비 인자화(이번엔 'person' 기본).
- **`Detection`을 frozen dataclass로** 묶어 후보 1개의 위치·신뢰도·외형특징을 한 단위로 다룬다(매개변수 폭증 방지, `TrackResult`/`SegmentResult` 패턴 계승).
- **`feat=None` 허용**: Grounding DINO만으로는 외형 임베딩이 약할 수 있다. `feat`가 None이면 `rematch_score`가 자동으로 위치(IoU)만 평가한다(기존 `rematch_score` 분기 그대로). PoC `h2_cut_retrack.py`도 위치 기반으로 자가검증한다.

### 2-3. `infra/grounding_dino_backend.py` — 실구현 (Colab 후행, 자동 테스트 제외)

PoC `h2_cut_retrack.py`(`detect_cuts`/`rematch_score`)는 PySceneDetect·재매칭을 검증했고, 검출 호출 패턴은 설치된 transformers `GroundingDinoForObjectDetection`/`GroundingDinoImageProcessor`에 매핑한다.

| Protocol 메서드 | transformers 호출(구현 단계 venv/Colab 재확인) |
|---|---|
| `detect(frame, prompt)` | `_ensure_loaded()` 지연 로드 → `processor(images=frame, text=prompt, ...)` → `model(**inputs)` → `processor.post_process_grounded_object_detection(...)` → 박스·score를 `Detection` 리스트로 정규화 |

- **지연 로드 계승**: 생성자는 `repo`/`device`/`box_threshold` 보관만. `_ensure_loaded()`로 첫 `detect`에서 로드(`Sam2VideoBackend`·`Sam2ImageBackend` 패턴 동일).
- **외형특징(feat)**: 1차 구현은 `feat=None`(위치 기반 재매칭). 외형 임베딩(crop ROI를 작은 backbone에 통과 등)은 정확도가 부족할 때만 후속 보강(ADR 0006 cls_sim 보조 신호). **이번 슬라이스는 위치 기반이 기본, feat는 확장점만 열어둔다(KISS).**
- **정확한 시그니처는 구현 단계 venv/Colab에서 재확인**(transformers 버전별 차이 가능). 자동 CI 완전 제외(import 자체 회피).

### 2-4. `FakeDetectionBackend` — 테스트 더블

`tests/fixtures/fakes.py`에 `FakeVideoBackend` 패턴 그대로 추가:

```python
class FakeDetectionBackend:
    """DetectionBackend 준수 테스트 더블. torch/transformers 비의존.

    생성자에 시나리오별 후보 리스트를 주입해 결정적 검출을 흉내 낸다.
      - candidates: 프레임 무관 고정 후보 리스트(또는 호출별 시퀀스)
      - detect_call_count: detect 호출 횟수 스파이(컷 수만큼만 호출되는지 가드)
    """
    device = "cpu"
    def __init__(self, candidates: list[Detection], feat_mode="none"): ...
    def detect(self, frame, prompt) -> list[Detection]: ...   # 카운트 +1, 결정적 후보 반환
```

- **결정적 시나리오 주입**: 컷 직후 후보를 테스트가 직접 정의 → `rematch_score`·best 선택·threshold 판정 결과를 수동 계산과 대조. 시나리오 3종:
  - **통과**: 직전 bbox와 IoU 높은 후보 1개(또는 best가 ≥0.5) → 재초기화·objid 유지.
  - **미달**: 모든 후보가 직전 bbox와 멀어 max<0.5 → `needs_correction` 플래그.
  - **다중 후보**: 가까운 것·먼 것 혼재 → best가 정확히 가장 가까운 후보로 선택되는지(argmax 정확성).
  - **빈 검출**: `detect`가 `[]` 반환(컷 후 인물 없음) → 미달과 동일하게 `needs_correction`.
- **`detect_call_count` 스파이**: "검출은 컷 경계마다 정확히 1회만"을 단언하는 회귀 가드(이미지 `segment_call_count`/비디오 `propagate_call_count`의 검출판).

### 2-5. `infra/shot_detect.py` — 컷 감지 (PySceneDetect, CPU 검증 가능)

PySceneDetect는 CPU에서 동작하므로 **infra이지만 합성 클립으로 CPU 통합 테스트가 가능**하다(GPU 블로커 아님).

```python
# infra/shot_detect.py  (PySceneDetect=scenedetect 0.7, 지연 import)
def detect_cut_frames(frames: list[np.ndarray], threshold: float = 27.0) -> list[int]:
    """프레임 시퀀스에서 컷 경계 프레임 인덱스 리스트를 반환한다(순수 데이터 출력).

    PySceneDetect ContentDetector로 장면을 감지한 뒤, 각 장면의 시작 프레임 인덱스
    (첫 장면 0 제외)를 "컷 인덱스"로 반환한다. 예: [0,80) [80,150) → 컷=[80].

    Args:
        frames: 구간 RGB 프레임 리스트(이미 추출된 시퀀스 — video_io 재사용).
        threshold: ContentDetector 민감도(기본 27.0, 클수록 둔감).

    Returns:
        컷이 시작되는 프레임 인덱스 오름차순 리스트. 컷 없으면 [].
    """
```

- **입력은 "이미 추출된 프레임 리스트"**: `track`이 이미 `frames`를 들고 있으므로(첫 슬라이스 `track(frames, point)`), 파일을 다시 디코드하지 않고 메모리 프레임으로 감지한다(이중 디코드 방지, DRY). PySceneDetect는 파일 경로 API(`scenedetect.detect`)와 프레임 푸시 API(`SceneManager`+`ContentDetector`)를 모두 제공 — **프레임 푸시 API**를 써서 메모리 프레임 입력을 지원한다.
- **출력 = 순수 데이터(인덱스 리스트)**: infra가 PySceneDetect 타입을 core/app에 누출하지 않는다(`scenedetect` 타입 → `list[int]`로 정규화). 첫 슬라이스의 "session opaque" 정신 계승 — 외부 라이브러리 타입 경계 차단.
- **지연 import**: `import scenedetect`는 함수 내부에서(video_io의 `import av` 패턴 계승). 테스트는 `importorskip("scenedetect")` 가드.
- **컷 인덱스 의미 정규화**: "샷의 시작 프레임"으로 통일. `[0]`은 첫 프레임이므로 컷이 아님(제외). `split_into_shots`(§4) 헬퍼가 이 리스트로 `[(start,end), ...]` 샷 구간을 만든다.

---

## 3. 산출물 2 — 재매칭 판정 순수 함수 (`rematch.py` 재사용·확장)

### 3-1. 기존 `rematch.py` 그대로 재사용

`core/tracking/rematch.py`의 `iou`·`rematch_score(prev_box, cand_box, prev_feat, cand_feat, w_pos=0.7, w_cls=0.3)`는 **변경 없이 그대로 재사용**한다. 후보 1개 평가는 이미 완성돼 있다(테스트 `test_rematch.py` 통과 중).

### 3-2. 신규 순수 함수: best 후보 선택 + threshold 판정 (`rematch.py` 인접)

`rematch_score`는 후보 1개만 본다. 오케스트레이션이 필요로 하는 것은 **"후보 리스트 + 직전 bbox → best 후보·점수·통과여부"** 이다. 이를 `rematch.py`에 순수 함수로 추가한다(같은 도메인·순수성 유지).

```python
# core/tracking/rematch.py 에 추가 (core 순수, torch/UI 비의존)

# 재매칭 통과 임계값 (ADR 0006 — w_pos=0.7·w_cls=0.3 기준)
REMATCH_THRESHOLD: float = 0.5

@dataclass(frozen=True)
class RematchResult:
    """재매칭 1회 판정 결과(불변).

    best_index: 최고 점수 후보의 인덱스(후보 없으면 -1).
    score: 최고 점수(후보 없으면 0.0).
    passed: score >= threshold 여부(=동일인 재매칭 성공).
    """
    best_index: int
    score: float
    passed: bool

def select_best_match(
    prev_box: Box,
    candidates: list[Detection],      # core는 Detection 타입에만 의존(torch X)
    threshold: float = REMATCH_THRESHOLD,
) -> RematchResult:
    """후보 리스트에서 직전 bbox와 best 매칭 후보·점수·통과여부를 판정한다(순수).

    각 후보에 rematch_score(prev_box, cand.box, prev_feat?, cand.feat)를 적용해
    최댓값 후보를 고르고 threshold로 통과 여부를 결정한다.
    후보가 비면 RematchResult(best_index=-1, score=0.0, passed=False).
    """
```

- **순수·테스트 용이**: backend·IO·torch 미의존. 후보 리스트와 박스만 입력 → 결정적 출력. 통과/미달/다중후보/빈리스트 4케이스를 단위 테스트로 완전 검증.
- **`prev_feat` 주입 경로**: 직전 샷 외형특징은 1차 구현에서 None(위치 기반). `select_best_match` 시그니처는 `prev_box`만 받고, `prev_feat`가 필요해지면 `Detection`처럼 묶어 확장(이번엔 위치 기반, KISS). → 매개변수 3개 준수.
- **함수 20줄**: 루프 1개 + argmax + 판정 → 20줄 이내. 초과 시 `_score_candidate` 헬퍼 분리.
- **`Detection` import 방향**: `Detection`은 `core/segmentation/detection_backend.py`(core)에 있으므로 `core/tracking/rematch.py`가 import해도 **core→core**라 경계 위반 아님. (만약 의존 사이클이 우려되면 `Detection`을 `core/tracking`으로 옮기거나 `rematch`가 `(box, feat)` 튜플만 받게 한다 — 구현 단계 결정, 기본은 Detection 직접 사용.)

### 3-3. 마스크 → bbox 순수 헬퍼 (누락 연결고리 — 중요)

**문제**: SAM2 video는 **마스크만** 반환하고 첫 슬라이스는 `centroid_of_mask`로 **centroid만** 뽑는다. 그러나 `rematch_score`는 **`prev_box`(bbox)** 를 요구한다. 즉 "직전 샷 마지막 프레임의 bbox"가 현재 자산에 없다.

**해결**: `core/crop/crop.py`에 마스크→bbox 순수 헬퍼를 추가한다(centroid_of_mask 인접, DRY).

```python
# core/crop/crop.py 에 추가 (순수)
def bbox_of_mask(mask) -> tuple[float, float, float, float] | None:
    """불리언/0-1 마스크의 외접 bbox (x1, y1, x2, y2). 빈 마스크면 None."""
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
```

- **재매칭 흐름**: 직전 샷 마지막 유효 마스크 → `bbox_of_mask` → `prev_box` → `select_best_match(prev_box, candidates)`.
- **순수·결정적**: `centroid_of_mask`와 완전 대칭(같은 파일·같은 패턴). FakeVideoBackend 사각형 마스크로 bbox를 정확히 예측해 테스트 가능.

---

## 4. 산출물 3 — `track` 확장 오케스트레이션 (샷 분할·경계 재매칭·objid 유지·미달 플래그)

### 4-1. 무거움/가벼움 분리 불변식 유지 (첫 슬라이스 핵심 계승)

이번 슬라이스가 절대 깨면 안 되는 첫 슬라이스의 설계:

- **무거움(track)**: SAM2 `propagate`(샷마다 1회) + Grounding DINO `detect`(컷마다 1회) — **모두 `track` 안에서만**. 워커 1회 실행.
- **가벼움·순수(compute_boxes)**: centroids→smooth→박스. backend·detection **절대 미호출**. 종횡비/크기/smooth 변경 시 재추적 없이 즉시 재계산. **`propagate_call_count`/`detect_call_count` 회귀 가드가 순수성을 강제.**

> 즉 컷 감지·재검출·재매칭은 **전부 track 안**으로 들어가고, `compute_boxes`·`export`는 **첫 슬라이스 그대로**(centroids·valid_flags만 본다). 이게 이번 설계의 등뼈.

### 4-2. `VideoCaptureUseCase` 변경 (생성자에 `DetectionBackend` 추가)

```python
class VideoCaptureUseCase:
    def __init__(
        self,
        source: FrameSource,
        backend: VideoSegmentationBackend,
        detector: DetectionBackend | None = None,   # 신규 — None이면 단일 샷 모드(첫 슬라이스 호환)
    ) -> None: ...
```

- **`detector=None` 하위호환**: detector가 None이면 컷 감지·재매칭을 건너뛰고 첫 슬라이스의 단일 샷 `track` 그대로 동작(기존 216개 테스트 무회귀). router가 `Sam2VideoBackend`+`GroundingDinoBackend`를 둘 다 주입하면 재추적 모드.
- **매개변수 3개 초과(4개)**: 생성자는 "조립 의존성"이라 dataclass로 묶기보다 명시 주입이 가독성 우위. 단 규칙 엄수가 필요하면 `VideoDeps(source, backend, detector)` frozen dataclass로 묶는다(구현 단계 리뷰 결정).

### 4-3. `TrackResult` 확장 (재매칭 실패 플래그 담기)

```python
@dataclass(frozen=True)
class TrackResult:
    masks: list[np.ndarray]
    centroids: list[tuple[float, float] | None]
    # 신규: 프레임별 재매칭 실패 여부(컷 직후 재매칭 미달 구간 = True).
    #   기본 전부 False(단일 샷·재매칭 성공 시). 첫 슬라이스 호환 위해 default_factory.
    needs_correction: list[bool] = field(default_factory=list)
    # 신규: 감지된 컷 프레임 인덱스(UI 표시·디버그용). 단일 샷이면 [].
    cut_frames: list[int] = field(default_factory=list)
```

- **`needs_correction`**: 컷 직후 재매칭이 미달한 프레임 구간을 True로 표시. 첫 슬라이스 호환(필드 추가만, 기본 빈 리스트). export `valid_flags` 도출은 **centroids None 여부 그대로**(첫 슬라이스 불변식 유지) — `needs_correction`은 **UI 표시·교정 유도용 별도 신호**이지 gap_policy 입력이 아니다(책임 분리).
- **`cut_frames`**: UI가 "여기서 컷, 여기서 재매칭 실패"를 표시(§6). 디버깅·검증에도 유용.

### 4-4. `track` 확장 알고리즘 (무거움 — 워커 1회)

```python
def track(self, frames, point) -> TrackResult:
    """샷 분할 → 샷별 SAM2 track → 경계 재매칭 → objid 유지/실패 플래그 (무거움).

    detector=None이면 단일 샷 경로(첫 슬라이스 그대로).
    """
    # 1) 컷 감지 (detector 있을 때만) → 샷 구간 리스트
    # 2) 첫 샷: 사용자 클릭 point로 SAM2 track (첫 슬라이스 로직 재사용)
    # 3) 각 후속 샷 경계마다:
    #    a. 직전 샷 마지막 유효 마스크 → bbox_of_mask → prev_box
    #    b. 컷 직후 프레임 → detector.detect(frame, "person") → 후보 리스트
    #    c. select_best_match(prev_box, candidates) → RematchResult
    #    d. passed: 후속 샷을 best.box 중심 클릭으로 SAM2 재초기화·track (objid 유지)
    #       미달:   후속 샷 프레임을 needs_correction=True, centroid는 hold(직전 위치)
    # 4) 샷별 결과 이어붙여 전체 길이 masks/centroids/needs_correction 조립
```

- **함수 20줄 준수**: `track`은 오케스트레이션만(샷 루프). 실제 작업은 헬퍼로 분리:
  - `_track_first_shot(frames, point)` — 첫 샷 SAM2 track (첫 슬라이스 로직).
  - `_rematch_at_boundary(prev_mask, cut_frame)` — bbox→detect→select_best_match → RematchResult.
  - `_track_shot_from_box(frames, box)` — best.box를 클릭점으로 변환해 다음 샷 SAM2 재초기화·track.
  - `_mark_failed_shot(n)` — 미달 샷을 needs_correction=True·centroid None(hold)로 채움.
- **재초기화 방식 결정 (산출물 명시 항목)**: **샷마다 새 SAM2 세션 + best.box 중심점 add_click**(opaque session 폐기 후 재생성). 이유: ① `VideoSegmentationBackend`가 이미 `init_session→add_click→propagate` 계약 — 재초기화는 그냥 다음 샷 프레임으로 새 세션을 여는 것과 동형(메서드 추가 없음, OCP). ② SAM2 video 세션은 init 시 video=frames를 통째로 받으므로 "샷 단위 세션"이 자연스럽다. ③ best.box → 박스 중심점을 클릭점으로 변환(`add_click(point)` 재사용, 박스 프롬프트 메서드 신설 불필요 — KISS). **대안(메모리 프롬프트로 같은 세션 이어가기)은 transformers API·정확도 불확실 → Colab 검증 후 v1.1.**
- **`object_id` 유지 의미**: 출력 산출물 관점에서 object_id는 "하나의 크롭 시퀀스로 이어진다"는 뜻. 샷마다 SAM2 obj_id는 1로 새로 잡지만, **centroids 리스트를 한 줄로 이어붙여 단일 추적 대상처럼 export**한다 → 사용자에겐 끊김 없는 하나의 추적(objid 유지).
- **카운터 가드**: `propagate_call_count == 샷 개수`(샷마다 1회), `detect_call_count == 컷 개수`(경계마다 1회). compute_boxes 반복 호출 시 두 카운터 모두 불변 — 첫 슬라이스 "재추적 안 함" 가드의 확장판.
- **샷 분할 헬퍼**(순수, core 후보): `split_into_shots(n_frames, cut_frames) -> list[(start,end)]`는 순수 인덱스 변환 → `core/tracking`에 둘 수 있다(gap_policy 인접). PySceneDetect 무의존이므로 core 순수 + 단위 테스트.

### 4-5. `compute_boxes`·`export` — 변경 없음 (첫 슬라이스 그대로)

- `compute_boxes`: `result.centroids`만 본다 → **무변경**. 샷 경계에서 이어붙인 centroids를 그대로 smooth·박스화. 고정 box size 불변식 유지.
- `export`: `valid_flags = centroids None 여부` → **무변경**. 미달 샷은 centroid hold(직전 위치)이므로 gap_policy가 첫 슬라이스대로 처리.
- **WHY 무변경이 중요**: 재추적 복잡도를 `track` 안에 가두고 하류(compute_boxes·export·UI 박스 재계산)는 첫 슬라이스 인터페이스를 그대로 쓴다 → 회귀 표면적 최소화, 리뷰·테스트 부담 최소.

---

## 5. 산출물 4 — 의존성 방향 (core 비의존 검증)

```
┌──────────────────────────────────────────────────────────┐
│ UI (PySide6)  VideoMainWindow(재) — 컷/재매칭실패 표시만 추가 │
└───────────────┬──────────────────────────────────────────┘
                │ 시그널/슬롯 (_TrackWorker 재사용 — track 확장)
┌───────────────▼──────────────────────────────────────────┐
│ Application  VideoCaptureUseCase.track 확장(샷 분할·재매칭) │
└───────────────┬──────────────────────────────────────────┘
                │ Protocol 의존만 (VideoSegmentationBackend·DetectionBackend·FrameSource)
┌───────────────▼──────────────────────────────────────────┐
│ Core (UI/IO/torch/transformers/scenedetect 전부 비의존)     │
│  tracking/rematch(재사용 + select_best_match·RematchResult)│
│  tracking/gap_policy(재) · split_into_shots(신규 순수)      │
│  crop/crop(재 + bbox_of_mask 신규 순수)                     │
│  segmentation/detection_backend(신규 Protocol + Detection)  │
│  segmentation/video_backend(재) · export/video_export(재)   │
└───────────────┬──────────────────────────────────────────┘
                │ Protocol 구현체 주입 (composition root = router)
┌───────────────▼──────────────────────────────────────────┐
│ Infra                                                      │
│  shot_detect(신규, scenedetect 지연 import — CPU 검증 가능) │
│  grounding_dino_backend(신규, torch/transformers — Colab)   │
│  sam2_video_backend(재, torch/transformers — Colab)         │
│  video_io(재) · device(재)                                  │
└──────────────────────────────────────────────────────────┘
```

### 경계 불변식 (ADR 0010 계승, 회귀 가드 대상)

- **core는 torch·transformers·PySide6·PyAV·av·scenedetect를 import하지 않는다.** `Detection`·`RematchResult`·`select_best_match`·`bbox_of_mask`·`split_into_shots`는 numpy만 의존(순수).
- **`DetectionBackend` Protocol은 core**, Grounding DINO 구현은 **infra**. PySceneDetect도 **infra**(`shot_detect`)이며 출력은 `list[int]`(순수 데이터)로 정규화 — scenedetect 타입 누출 금지.
- **app은 Protocol 3종에만 의존**(`FrameSource`·`VideoSegmentationBackend`·`DetectionBackend`). 구체는 router 주입. 테스트는 `FakeVideoBackend`+`FakeDetectionBackend`+`FakeFrameSource` 주입 + `shot_detect`는 합성 클립 또는 cut_frames 직접 주입으로 격리.
- **UI는 app 유스케이스만 호출.** 컷/재매칭실패 표시는 `TrackResult.cut_frames`·`needs_correction`(순수 데이터)만 읽는다.

---

## 6. 산출물 5 — 작업 분해 (인터페이스→테스트→구현 커밋 순서)

> 순서 원칙(전역): **인터페이스 → 테스트 → 구현 → 리팩터**. 1줄 = 1 논리적 변경 = 1 커밋. 브랜치: `feature/video/shot-retrack`. L2 도메인 → `/develop` 팀 파이프라인(planner→developer→tester→reviewer→documenter).

### 도메인 A: 재매칭 판정 순수 함수 (core — 먼저, 의존 없음, GPU 무관)
- [ ] (인터페이스) `core/crop/crop.py`에 `bbox_of_mask` 추가(centroid_of_mask 대칭, docstring) — 우선순위: **높음**
- [ ] (테스트) `test_crop.py`에 `bbox_of_mask` 테스트(사각형 마스크 외접 박스·빈 마스크 None) — 우선순위: 높음
- [ ] (인터페이스) `core/tracking/rematch.py`에 `RematchResult`·`REMATCH_THRESHOLD`·`select_best_match` 시그니처+docstring — 우선순위: **높음**
- [ ] (테스트) `test_rematch.py` 확장: best 선택(다중 후보 argmax)·통과(≥0.5)·미달(<0.5)·빈 후보(-1/False) — 우선순위: **높음**
- [ ] (구현) `select_best_match` 구현 → 테스트 통과 — 우선순위: **높음**
- [ ] (인터페이스+테스트+구현) `core/tracking`에 `split_into_shots(n_frames, cut_frames)` 순수 함수 + 단위 테스트(컷 없음/1컷/다중컷/경계값) — 우선순위: 높음

### 도메인 B: DetectionBackend Protocol + Fake (core 추상 + 테스트 더블 — GPU 무관)
- [ ] (인터페이스) `core/segmentation/detection_backend.py`(`DetectionBackend` Protocol·`Detection` dataclass·docstring, `@runtime_checkable`) — 우선순위: **높음**
- [ ] (테스트 더블) `tests/fixtures/fakes.py`에 `FakeDetectionBackend`(시나리오 후보 주입·`detect_call_count` 스파이) — 우선순위: **높음**
- [ ] (테스트) Protocol 계약 테스트(`isinstance(FakeDetectionBackend(...), DetectionBackend)`) — 우선순위: 높음

### 도메인 C: 컷 감지 (infra — CPU 검증 가능)
- [ ] (인터페이스) `infra/shot_detect.py` `detect_cut_frames(frames, threshold)` 시그니처+docstring(scenedetect 지연 import 골격) — 우선순위: 높음
- [ ] (테스트) `test_shot_detect.py`: `importorskip("scenedetect")` + 합성 "두 장면" 클립 fixture(앞 절반 빨강/뒤 절반 파랑 등)로 컷 인덱스 검출 검증 + 단일색 클립 컷=[] 검증 — 우선순위: 중간
- [ ] (구현) `detect_cut_frames`(SceneManager+ContentDetector 프레임 푸시 → 장면 시작 인덱스 정규화) → 테스트 — 우선순위: 중간

### 도메인 D: usecase track 확장 (app — 임계 경로, GPU 무관)
- [ ] (인터페이스) `VideoCaptureUseCase.__init__`에 `detector` 파라미터 추가 + `TrackResult`에 `needs_correction`/`cut_frames` 필드(default 빈 리스트, 하위호환) — 우선순위: **높음**
- [ ] (테스트) `test_video_capture.py` 확장 — **이번 슬라이스의 증거**:
      - 단일 샷(detector=None) 무회귀(기존 동작 동일)
      - 컷 1개+재매칭 통과: 두 샷 centroids 이어짐·`propagate_call_count==2`·`detect_call_count==1`·needs_correction 전부 False
      - 컷 1개+재매칭 미달: 후속 샷 needs_correction=True·centroid hold
      - 다중 후보: best가 정확히 가까운 후보로 선택(`select_best_match` 통합)
      - 빈 검출: needs_correction=True
      - **순수성 가드**: compute_boxes 반복 호출 시 `propagate_call_count`/`detect_call_count` 불변 — 우선순위: **높음**
- [ ] (구현) `track` 확장(샷 루프 + `_track_first_shot`/`_rematch_at_boundary`/`_track_shot_from_box`/`_mark_failed_shot` 헬퍼) → 테스트 통과 — 우선순위: **높음**
- [ ] (검증) `compute_boxes`·`export` 무변경 확인(centroids·valid_flags 경로 회귀 가드) — 우선순위: 높음

### 도메인 E: infra 실구현 (Colab 후행 — 자동 테스트 제외)
- [ ] (골격) `infra/grounding_dino_backend.py`(`_ensure_loaded` 스텁·`detect` PoC h2 매핑 주석·지연 로드) — 우선순위: 중간
- [ ] (Colab) `poc/colab/`에 컷 재추적 검증 셀: 컷 섞인 짧은 군무 MV 클립 → shot_detect→SAM2 샷별 track→Grounding DINO detect→select_best_match→재초기화 end-to-end (수동 GPU) — 우선순위: 중간
- [ ] (구현) `detect` 실구현(processor/model→post_process→Detection 정규화) → Colab 수동, 자동 CI 제외 — 우선순위: 낮음

### 도메인 F: composition root + UI 최소 변경
- [ ] (조립) `app/router.py` `_build_video_usecase_factory`에 `GroundingDinoBackend` 생성·주입(detector 인자) — 우선순위: 중간
- [ ] (UI) `ui/video_window.py`: `TrackResult.cut_frames`/`needs_correction` 표시(상태바 "N개 컷 감지, M구간 재매칭 실패 — 교정은 추후 지원" 안내 + 캔버스 경계 마커 선택). `_TrackWorker`는 재사용(track 시그니처 불변) — 우선순위: 중간

### 마무리
- [ ] (스모크) `pytest` 전 구간 녹색(Fake 경로) + 컷 시나리오 3종 통과 + 기존 216개 무회귀 — 우선순위: **높음**
- [ ] (문서) ADR 0012(DetectionBackend Protocol) + 필요 시 ADR 0006 보완(재추적 1단계=플래그) — 우선순위: 높음
- [ ] (문서) HANDOFF.md 갱신(샷경계 재추적 척추 완료·Grounding DINO/SAM2 video 실추론 Colab 미검증 명시·교정 UI는 다음) — 우선순위: **높음** (커밋-문서 동기화 필수)

### 임계 경로

```
A(rematch/bbox_of_mask/split_into_shots 순수) ─┐
B(DetectionBackend Protocol + FakeDetection)  ─┼─▶ D(track 확장 테스트→구현) ─▶ 스모크
C(shot_detect, CPU)  ───── (병렬, 독립) ───────┘          │
                                                          ▼
                                          F(router 주입 + UI 표시)
E(Grounding DINO 실구현)는 최후행(Colab, CI 무관) — Protocol 뒤에 숨어 척추와 병렬.
```

- **핵심**: A + B + D가 척추. `FakeDetectionBackend`+`FakeVideoBackend`로 **컷 시나리오(통과/미달/다중후보/빈검출)가 CPU에서 전부 녹색**이면, 재추적 오케스트레이션의 가치는 GPU 없이 확보된다. C(컷 감지)는 CPU에서 합성 클립으로 추가 검증. E(Grounding DINO 실추론)만 Colab 후행.

---

## 7. 산출물 6 — 테스트 전략

### 7-1. 격리 원칙 (첫 슬라이스 계승 + 검출 더블 추가)

| 대상 | 격리 방법 |
|---|---|
| **SAM2 video** | `FakeVideoBackend`(재). 샷마다 `init_session→add_click→propagate`. `propagate_call_count==샷수`·`init_call_count==샷수` 스파이 |
| **Grounding DINO 재검출** | `FakeDetectionBackend`(신규). 시나리오별 후보 리스트 주입. `detect_call_count==컷수` 스파이. torch/transformers 전혀 로드 안 함 |
| **PySceneDetect 컷 감지** | ① 단위: `detect_cut_frames`를 우회하고 `cut_frames`를 track에 직접 주입(=오케스트레이션만 검증). ② 통합: `importorskip("scenedetect")`+합성 2장면 클립으로 `detect_cut_frames` 자체 검증(**CPU 가능**) |
| **재매칭 판정** | `select_best_match`·`bbox_of_mask`·`rematch_score`·`split_into_shots` **순수 단위 테스트로 완전 검증**(모델 무관) |
| **PySide6 위젯** | cut_frames/needs_correction 표시는 순수 데이터 읽기 → offscreen 스모크 |
| **Grounding DINO·SAM2 video 실추론** | **자동 테스트 완전 제외.** Colab(`poc/colab/`) 수동 |

### 7-2. 핵심 단위 테스트 (CPU, 모델 무관 — 슬라이스 가치의 증거)

- **`test_rematch.py`(확장)**: `select_best_match` — 다중 후보 중 best=가장 가까운 후보(argmax), 통과(≥0.5 passed=True), 미달(<0.5 passed=False), 빈 후보(best_index=-1·score=0·passed=False). `RematchResult` 불변 검증.
- **`test_crop.py`(확장)**: `bbox_of_mask` — 사각형 마스크 외접 박스 좌표 정확, 빈 마스크 None.
- **`test_video_capture.py`(확장, 가장 중요)** — `FakeVideoBackend`+`FakeDetectionBackend`+직접 주입 `cut_frames`:
  - **통과 시나리오**: 컷 1개, 후보 중 best≥0.5 → 두 샷 centroids 이어짐, `needs_correction` 전부 False, `propagate_call_count==2`, `detect_call_count==1`.
  - **미달 시나리오**: 모든 후보 IoU<임계 → 후속 샷 `needs_correction=True`, centroid hold(직전 위치 유지).
  - **다중 후보**: 가까운/먼 후보 혼재 → best가 정확히 가까운 것, 통과.
  - **빈 검출**: `detect`가 `[]` → `needs_correction=True`.
  - **단일 샷 호환**: `detector=None` → 첫 슬라이스 동작 동일, `detect_call_count==0`, `cut_frames==[]`.
  - **순수성 회귀 가드(핵심)**: compute_boxes를 종횡비·크기·smooth 바꿔 여러 번 호출해도 `propagate_call_count`·`detect_call_count` **불변** — 재추적·재검출 안 함.
  - **다중 컷 누적**: 컷 2개(3샷) → centroids 3샷 이어붙임, `propagate_call_count==3`, `detect_call_count==2`.
- **`test_shot_detect.py`(신규, 조건부)**: `importorskip("scenedetect")` + 합성 2장면 클립 → `detect_cut_frames`가 경계 인덱스 1개 검출; 단일색 클립 → `[]`.

### 7-3. 통합(조건부) / 수동(Colab)

- 컷 감지 통합은 CPU에서 합성 클립으로 가능(7-2 shot_detect). **나머지 GPU 모델 실추론은 `poc/colab/` 수동**: 컷 섞인 짧은 군무 MV 클립으로 end-to-end(컷감지→샷별 SAM2→Grounding DINO 재검출→재매칭→재초기화) 1회 검증. ADR 0006 AC(재추적 성공률·오탐률)를 `poc/REPORT.md` H2 칸에 기록. **자동 CI 제외.**

### 7-4. 커버리지

- 재추적 순수 로직(`select_best_match`·`bbox_of_mask`·`split_into_shots`·`track` 오케스트레이션 조립) **80%+**(L2). Grounding DINO·SAM2 래퍼는 Colab 수동 스모크로 happy path만.

---

## 8. 산출물 8 — 리스크 및 후속

### 8-1. 리스크 및 완화책

| 리스크 | 영향 | 완화책 |
|---|---|---|
| **GPU 미검증 위에 코드를 쌓음(누적)** — 이번 슬라이스가 **두 번째 GPU 의존(Grounding DINO)을 Fake 위에 추가** | SAM2 video(첫 슬라이스 미검증) + Grounding DINO 실 API/출력이 Fake와 어긋나면 통합 시 연쇄 재작업 | ① 척추(A·B·D)를 GPU 무관 CPU 테스트로 완성해 **재추적 로직의 정확성은 모델과 독립 보장** ② `DetectionBackend`를 PoC `h2_cut_retrack.py` 검증 패턴에 정확히 매핑 ③ **Colab 검증을 첫 슬라이스(SAM2 video)와 이번 슬라이스(Grounding DINO)를 묶어 한 번에** 수행(개별 검증 비용 절감, HANDOFF 우선순위 1) ④ `Detection.box`/`score` 정규화 계약을 PoC 출력 형태에 맞춰 고정 |
| **재매칭 오탐(False Positive)** — 다른 인물을 같은 인물로 오인 | 컷 후 엉뚱한 대상으로 추적 점프(덕후 페르소나 치명적) | ① threshold=0.5 보수적 시작, Colab 실측으로 보정(ADR 0006 갱신) ② 빈 검출·미달 시 **추적 점프 대신 needs_correction 플래그**(틀린 추적보다 "실패 표시"가 안전) ③ 외형특징(feat) 보조 신호를 정확도 부족 시 추가(cls_sim) |
| **재매칭 미탐(False Negative)** — 같은 인물인데 미달 처리 | 끊김 없어야 할 구간이 needs_correction | ① 위치(IoU) 기반은 컷 전후 구도 유사 시 강함 ② 미탐은 **교정 UI(다음 슬라이스)로 복구 가능** — 이번엔 플래그가 안전망 ③ threshold·가중치 Colab 보정 |
| **컷 감지 민감도** — 과검출(빠른 모션을 컷으로)·미검출(소프트 디졸브) | 과검출=불필요한 재매칭 호출/끊김, 미검출=샷 전환 놓쳐 추적 실패 | ① `ContentDetector` threshold 기본 27 + 합성 클립으로 민감도 확인 ② 과검출이어도 재매칭이 통과하면 추적 유지(IoU 높음) → 치명적이지 않음 ③ 미검출은 첫 슬라이스 occlusion 경로(hold)로 부분 흡수 ④ threshold 튜닝 UI는 후속 |
| **재초기화(새 세션) 비용** — 샷마다 SAM2 init_session(무거움) | 컷 많은 구간에서 추적 시간 급증(GPU에서도) | ① 워커 비동기(첫 슬라이스 `_TrackWorker` 재사용) ② 진행 표시 ③ 메모리 프롬프트로 같은 세션 이어가기는 v1.1(API 검증 후) |
| **마스크→bbox 누락 연결고리** — 첫 슬라이스는 centroid만, 재매칭은 bbox 필요 | 구현 시 prev_box 산출 누락 | `bbox_of_mask` 순수 헬퍼를 **도메인 A 최우선**으로 추가(§3-3) + 단위 테스트 |
| **`track` 함수 비대화** — 샷 루프+재검출+재매칭+재초기화 | 20줄 규칙 위반·가독성 | 헬퍼 4분할(§4-4) + frozen dataclass로 중간 결과 묶기 |

### 8-2. 후속 슬라이스 (이번 명시적 제외 → 다음)

1. **수동 교정 UI** (이번 슬라이스의 직접 후속): `needs_correction` 구간에서 사용자가 박스 드래그/재클릭으로 대상 재지정 → 해당 샷부터 재추적(ADR 0006 "수동 교정 유도" 2단계, `use-flow` §5). `cut_frames`·`needs_correction` 데이터가 이미 준비됨.
2. **Grounding DINO 자동 검출(클릭리스)**(ADR 0003): 첫 샷도 클릭 없이 detect로 후보 제시 → 사용자 선택.
3. **외형특징(re-ID) 보강**: `Detection.feat`를 채워 `cls_sim` 활성화(오탐 감소). 정확도 부족 확인 시(ADR 0006 v1.1).
4. **occlusion gap UI**: CUT/FREEZE 선택 + gap 구간 교정(첫 슬라이스 BACKGROUND 기본 유지 중).
5. **오디오 동기**(PoC H4): 컷 재추적된 구간 오디오를 PTS 동기로 MP4 머지.
6. **업스케일 결합**(ADR 0009): 크롭 시퀀스에 Swin2SR 적용(이미지 `export(upscaler=)` 시퀀스 확장).

---

## 9. ADR 트리거 (작성은 documenter)

- **(트리거) ADR 0012 후보**: "재검출 백엔드 Protocol 신설(`DetectionBackend`)" — `VideoSegmentationBackend`와 별도 추상(core)·Grounding DINO 구현(infra). ADR 0010(비디오 세그 분리)·ADR 0006(Grounding DINO 재검출)을 ISP 근거로 구체화. session 불요(stateless)·지연 로드·`feat` 확장점 기록.
- **(보완) ADR 0006 갱신**: 재추적의 "수동 교정 유도"를 **1단계=needs_correction 플래그(이번), 2단계=교정 UI(후속)**로 단계화 명시. threshold/가중치는 Colab 실측 후 보정 칸 추가.
- **(기록) 재초기화 방식**: "샷마다 새 세션 + best.box 중심점 add_click" 채택 근거(메모리 프롬프트 대안은 API 검증 후 v1.1) — ADR 0012 또는 0010 보완에 한 줄 기록.

> 본 계획서는 코드/ADR 본문을 작성하지 않는다. ADR 작성은 documenter, 구현은 developer/tester 단계.

---

## 10. 완료 정의 (DoD)

- [ ] `pytest`로 **Fake 경로 컷 시나리오 전 구간 녹색**: `FakeVideoBackend`+`FakeDetectionBackend`로 컷감지(주입)→샷별 track→재검출→`select_best_match`→재초기화/실패플래그→이어붙인 centroids→compute_boxes→export.
- [ ] **순수성 회귀 가드**: compute_boxes 반복 호출 시 `propagate_call_count`·`detect_call_count` 불변.
- [ ] **컷 시나리오 4종**(통과/미달/다중후보/빈검출) + 단일 샷 호환 + 다중 컷 누적 테스트 통과.
- [ ] core가 torch/transformers/PySide6/PyAV/av/**scenedetect**를 import하지 않음(경계 검증).
- [ ] `infra/grounding_dino_backend.py` 골격 + PoC `h2_cut_retrack.py` 매핑 주석(실추론 Colab 미검증 명시).
- [ ] `detect_cut_frames` 합성 클립 통합 테스트(조건부 importorskip) 녹색.
- [ ] 기존 216개 테스트 무회귀.
- [ ] ADR 0012 + HANDOFF.md 갱신 + 후속(수동 교정 UI) 진입 가능 상태.
</content>
</invoke>
