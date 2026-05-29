# ADR 0015 — SAM2 negative point 프롬프트 도입 (옆 멤버 배제)

- 상태: 채택 (2026-05-29 GPU 게이트 검증 완료 — hiera-small 전제)
- 날짜: 2026-05-29

## 맥락

Colab GPU 게이트(멀티샷 군무 직캠)에서 두 가지 문제가 동시에 드러났다.

1. **군무 밀착 구간 마스크 부정확**: box 프롬프트([ADR 0014](0014-box-prompt-mask-refine.md) 결정 1) + positive point만으로는, 추적 대상과 바로 옆 멤버가 화면에서 **맞닿은 구간**에서 SAM2가 둘을 **하나의 덩어리로 합쳐** 세그먼트한다. 사용자 관찰로는 클립 앞부분이 "대상과 옆사람이 구분 안 됨" 상태로 나온다. box는 전신 영역을 정확히 지정하지만, 밀착 시 box 안에 옆사람 몸의 일부가 함께 들어오면 positive 신호만으로는 경계를 가르지 못한다.

2. **largest_component 후처리의 효과 제한 + 치명적 성능**: [ADR 0014](0014-box-prompt-mask-refine.md) 결정 2의 `largest_component`(최대 4-연결 성분만 잔존)는 마스크에서 **떨어져 나온 파편**만 거른다. 위 1번처럼 대상과 옆 멤버가 **하나의 연결 성분으로 합쳐진 덩어리**는 가장 큰 성분 그대로 통과시켜 분리하지 못한다(효과 제한). 또한 core 경계 불변식(scipy·cv2 배제)을 지키려 순수 Python BFS로 구현한 결과 720p에서 **약 440ms/frame**으로 측정되어, 추적 처리량이 **2.3fps → 0.7fps(7분 영상 기준)**로 급락했다(치명적 성능).

[ADR 0014](0014-box-prompt-mask-refine.md)는 negative point를 "UI·자동 선정 복잡도 급증, box + largest_component로 충분"이라 가정해 "잔여 회귀 시 후속"으로 **보류**했었다. GPU 게이트 실측은 이 가정이 틀렸음을 보였다 — largest_component로는 합침을 못 풀고, 성능까지 무너진다. 따라서 보류 판단을 번복하고 negative point를 정식 채택한다.

## 결정

### 1. SAM2 negative point 프롬프트 도입

SAM2에 "이 점(=옆 멤버)은 대상이 **아니다**"라는 신호(label 0, negative point)를 함께 전달한다. negative point는 positive·box가 만든 영역에서 해당 지점 주변을 **밀어내** 경계를 형성하므로, 대상과 옆 멤버가 붙어 있어도 둘 사이를 가른다. largest_component가 풀지 못한 "연결된 합침"을 SAM2 추론 단계에서 직접 분리한다.

- transformers 5.9.0 `Sam2VideoProcessor.add_inputs_to_inference_session`는 box + positive + negative를 **단일 호출 1회**로 조립한다. 검증 근거(`processing_sam2_video.py`): box와 point를 함께 넘기면 내부에서 box corner-points와 point를 concat 처리하고(라인 699-709 — `input_boxes`·`input_points` cat 경로), `input_labels`는 별도 검증 없이 그대로 전달된다(라인 120-168 — label 무검증). 따라서 호출 1회에서 `input_boxes`(3-레벨 중첩) + `input_points`(4-레벨 중첩) + `input_labels`(3-레벨 중첩)를 함께 구성하고, **label 1=positive·0=negative**로 구분하며 `clear_old_inputs=True`로 한 번에 등록한다.

### 2. Protocol에 `negatives` 인자 추가 (별도 메서드 신설 금지)

`VideoSegmentationBackend`([ADR 0010](0010-video-segmentation-backend.md))의 기존 두 진입 메서드에 default 빈 튜플 `negatives` 인자를 추가한다.

```python
def add_box(session, box, negatives=()) -> None: ...
def add_click(session, point, negatives=()) -> None: ...
```

- **별도 메서드(`add_negative_points` 등)를 신설하지 않는다.** transformers가 box·positive·negative를 `clear_old_inputs=True`로 **1회 호출에 함께** 조립해야 하기 때문이다. negative를 별도 호출로 추가하면 box/positive를 덮어써 버린다(대안 (a) 참조). 따라서 동일 호출에서 함께 전달할 수 있도록 기존 메서드의 인자로 받는다.
- `negatives=()`(default 빈 튜플)이면 기존과 바이트 동일하게 동작하므로 **무회귀**다.
- [ADR 0010](0010-video-segmentation-backend.md)의 3-메서드 분리(add_click·add_box·propagate) 원칙을 인자 추가로 연장한다(메서드 수 불변 → OCP 정합, ISP 유지).

### 3. `CutSelection.negative_points` 필드 확장

`core/tracking/cut_selection.py`의 `CutSelection` VO에 옆 멤버 좌표 묶음을 보관하는 필드를 추가한다.

```python
@dataclass(frozen=True)
class CutSelection:
    shot_index: int
    point: tuple[int, int]
    box: tuple[float, float, float, float] | None = None
    negative_points: tuple[tuple[int, int], ...] = ()   # 신규(default 빈=하위호환)
```

- `negative_points` default `()`로 기존 `(shot_index, point[, box])` 생성 코드를 깨지 않는다(하위호환).
- 순수 검증 함수 `validate_negative_points(selection, frame_size)`를 추가한다 — negative 좌표가 프레임 범위(`0 <= x < W, 0 <= y < H`)를 벗어나거나 positive(point)와 **동일 좌표**이면 한국어 `ValueError`를 발생시킨다(기존 `validate_selections` 방어 패턴 계승). 빈 negatives는 통과(무회귀).

### 4. largest_component 추적 후처리 철회 ([ADR 0014](0014-box-prompt-mask-refine.md) 결정 2 대체)

추적 경로(propagate 후 마스크 확정)에서 `largest_component` 호출을 **철회**한다. 옆 멤버 분리 책임을 negative point가 전적으로 맡는다.

- 함수(`core/crop/mask_refine.py`)와 단위 테스트(`test_mask_refine.py`)는 **deprecated 상태로 보존**한다(infra `cv2.connectedComponents` 폴백 후속 백로그용 — 대안 (c) 참조). 추적 경로에서만 호출을 제거한다.
- 720p 440ms/frame 부담이 사라져 추적 처리량이 **2.3fps로 복구**된다.

### 5. 노트북 UX — `NEG_TARGETS`로 negative point 지정

노트북(Colab) 게이트에서는 컷별로 배제할 옆 후보 인덱스 목록 `NEG_TARGETS`를 받아, 해당 후보 box의 **중심 좌표를 negative point로** 변환해 `CutSelection.negative_points`에 채운다. 사용자는 "추적 대상"(positive)과 "배제할 옆 멤버"(negative)를 컷별로 후보 인덱스로 지정하면 되며, 좌표 계산은 노트북이 수행한다.

## 대안

**(a) `add_negative_points` 별도 메서드 신설**

negative 등록을 `add_box`/`add_click`과 분리된 별도 메서드로 둔다.

- 거부 이유: transformers `add_inputs_to_inference_session`는 `clear_old_inputs=True`로 호출마다 이전 입력을 비운다. negative를 별도 호출로 추가하면 직전에 등록한 box/positive를 **덮어써 버려** 함께 조립할 수 없다. box·positive·negative는 반드시 **1회 호출**로 동시 전달해야 하므로 별도 메서드가 성립하지 않는다.

**(b) `add_points`로 다중 포인트 일반화**

positive/negative를 구분 없이 받는 범용 `add_points(session, points, labels)`로 인터페이스를 통합한다.

- 거부 이유: 기존 `add_click(session, point)` 시그니처를 깨 **무회귀를 위반**한다. 단일 positive point 경로(단일샷·폴백)가 광범위하게 호출되고 있어 시그니처 변경은 회귀 위험이 크다. default `negatives=()` 인자 추가가 무회귀를 보장하면서 동일 표현력을 제공한다.

**(c) `cv2.connectedComponents` 기반 연결성분 후처리**

순수 Python BFS 대신 OpenCV의 최적화된 연결성분 라벨링으로 largest_component를 가속한다.

- 거부 이유(이번 슬라이스 한정): cv2는 infra 의존이라 core 경계 불변식을 위반하므로 infra 폴백으로만 둘 수 있다. 그러나 cv2로 가속해도 **연결된 합침을 분리하지 못하는 효과 한계(맥락 2)는 그대로**다. negative point가 합침 분리라는 근본 문제를 해결하므로, cv2 폴백은 떨어진 파편 제거가 필요해질 경우를 대비한 **후속 백로그**로 남긴다(즉시 채택 아님).

## 결과

### 긍정적 영향

- **밀착 구간 분리 기대**: negative point가 SAM2 추론 단계에서 대상과 옆 멤버의 경계를 가르므로, largest_component가 못 풀던 "연결된 합침"을 직접 해소한다(클립 앞부분 "구분 안 됨" 직접 타겟).
- **성능 복구**: largest_component 추적 후처리 철회로 720p 440ms/frame 부담이 사라져 추적 처리량이 **2.3fps로 복구**된다.
- **무회귀**: `negatives=()`(default 빈)이면 `add_box`/`add_click` 기존 동작 그대로, `CutSelection.negative_points=()`이면 기존 선택 코드 그대로다. **588 passed**(무회귀 확인).
- **SOLID**: OCP(메서드 수 불변·인자 추가로 확장), ISP([ADR 0010](0010-video-segmentation-backend.md) 메서드 분리 유지), DIP(app은 Protocol만 의존 — transformers 조립은 infra 내부).

### 부정적 영향 / 트레이드오프 / 리스크

- **리스크 R1 — 전파 중 재합침(현실화 → 모델 상향으로 해소)**: negative point는 **첫 프레임(frame_idx=0)에만** 지정된다. 전파가 진행되며 대상과 옆 멤버가 다시 가까워지면 negative 신호 없이 재합침할 수 있다는 우려가 **GPU 게이트(2026-05-29)에서 `hiera-tiny`로 현실화**됐다 — tiny는 첫 프레임 negative를 줘도 밀착 인물 분리가 부족해 **전파 중 다시 합쳐졌다**. **`hiera-small`로 모델을 상향하자 해소**됐다(멀티샷 군무 300프레임 **AC-01 100% · needs_correction 0**). 운영 전제: **negative point는 `hiera-small` 이상 모델에서 효과적**이며, `hiera-tiny`에서는 밀착 군무 분리에 불충분하다.
- **노트북 UX 복잡도 증가**: 사용자가 positive 외에 배제할 옆 멤버(`NEG_TARGETS`)까지 컷별로 지정해야 한다. 기본은 빈 값이라 필요한 밀착 구간에서만 선택적으로 쓴다.
- **GPU 미검증**: SAM2 video 실추론은 CI 미실행이라 box+positive+negative 1회 조립의 실제 분리 효과는 **노트북 게이트가 필수**다.

### 검증 경로 (게이트) — ✅ 통과 (2026-05-29)

GPU 의존(track)이라 노트북(Colab/Kaggle) 우선 검증: 밀착 구간에서 box+positive **대비** box+positive+negative의 **분리 정확도** + 전파 중 **재합침 여부** + 추적 처리량 재측정이 데스크톱 적용 게이트였다.

**결과**: `hiera-small` + 컷별 선택(box + negative point) + 올바른 `shot_index` 키로 멀티샷 군무(300프레임, 컷 6→4샷) **AC-01 100%(300/300) · needs_correction 0 · AC-06 2.0fps**(largest_component 철회로 0.7→2.0fps 복구). `hiera-tiny`는 재합침으로 미달 → small 상향이 게이트 통과 조건. 데스크톱 적용 시 **노트북 SAM2_REPO를 small 기본화**한다.

## 연계

- [ADR 0010](0010-video-segmentation-backend.md) — `VideoSegmentationBackend.add_box`/`add_click`에 `negatives` 인자 추가(메서드 수 불변, 3-메서드 분리 원칙 연장). 본 ADR이 0010을 확장(Superseded 아님).
- [ADR 0006](0006-shot-boundary-reid.md) — 컷별 명시 선택(`CutSelection`)에 "배제점"(negative_points)을 더한다. positive 선택 경로는 무변경, negative는 옵트인.
- [ADR 0014](0014-box-prompt-mask-refine.md) — 본 ADR이 0014 결정 2(largest_component 마스크 후처리)를 **철회·대체**한다. 0014 결정 1·3(box 프롬프트)은 유지하며, 옆 멤버 분리 책임을 largest_component에서 negative point로 이관한다.
