# ADR 0016 — 데스크톱 컷별 선택 UI 아키텍처

- 상태: 채택 (Story 4 Task 4-1 구현 확정)
- 날짜: 2026-05-29

## 맥락

[ADR 0006](0006-shot-boundary-reid.md) 보완(2026-05-29)에서 자동 재매칭이 멀티샷 군무에서 **needs_correction 82.7%** 로 구조적 실패하여, 컷별 명시 선택 경로(`CutSelection`)를 1순위 전략으로 채택했다. 노트북(Colab/Kaggle) 검증에서 `hiera-small` + 컷별 명시 선택 + negative point([ADR 0015](0015-negative-point-prompt.md))로 **GPU 게이트 AC-01 100% · needs_correction 0** 를 달성한 상태다.

데스크톱 GUI에서 동일 경로를 재현하려면 다음 설계 결정이 필요했다.

1. **캔버스 히트테스트 책임**: 클릭 좌표를 받아 "어느 박스인지"를 캔버스 자체가 판정할지, 외부에 위임할지.
2. **워커 분리 방식**: 컷 감지·후보 검출·SAM2 전파를 하나의 워커로 묶을지, 분리할지.
3. **UI–core 경계**: UI 레이어가 `Detection` 객체를 직접 사용할지, core가 제공하는 변환 함수를 통해 간접 참조할지.
4. **박스·마스크 렌더링 좌표계**: 박스와 마스크 오버레이를 독립 좌표계로 합성할지, 동일 이미지 좌표계에서 렌더 후 함께 스케일할지.
5. **단일샷 무회귀**: 컷이 없는 영상에서 기존 단일 클릭 모드를 깨지 않는 방법.

## 결정

### 1. 캔버스는 좌표만 방출, 히트테스트는 video_window가 core 함수에 위임

`FrameCanvas`는 클릭 이벤트가 발생하면 `box_clicked(x, y)` Signal로 **이미지 좌표만 방출**한다. "어느 박스인지"를 판정하는 히트테스트(`core.pick_box_at`) — 겹침 시 최소 넓이 박스 선택 포함 — 는 `video_window`가 호출한다. `FrameCanvas`는 core 함수를 일절 import하지 않는다.

- 캔버스의 책임을 "픽셀 렌더링 + 좌표 방출"로 한정해 **단일 책임**을 유지한다.
- `pick_box_at`이 core 순수 함수이므로 Qt 의존 없이 단위 테스트할 수 있다.
- 박스 목록이 바뀌어도 캔버스 코드는 무변경(OCP).

대안 (a) "캔버스가 직접 히트테스트 + Detection 보유"는 core 의존을 캔버스에 끌어들여 레이어 경계를 침범하므로 기각했다(아래 대안 참조).

### 2. 워커 2분할: _DetectWorker(가벼움) vs _TrackWorker(무거움)

백그라운드 워커를 두 개로 분리한다.

- **`_DetectWorker`**: `detect_cuts` + `detect_cut_candidates`(Grounding DINO 후보 열거)를 수행한다. SAM2 propagate를 포함하지 않으므로 빠르게 완료되며, 완료 시 컷별 선택 UI(후보 박스 오버레이 + 선택 패널)를 즉시 표시한다.
- **`_TrackWorker`**: 사용자가 컷별 선택을 확정한 뒤 호출하며 SAM2 propagate(무거운 연산)를 수행한다. `video_path`·`span`으로 `detect_cuts`를 재호출해 컷 정보를 확보한다(중복이나 결정적이므로 `selections`의 `shot_index`와 항상 정합).

이 분리로 사용자는 수초 내에 후보 박스 UI를 확인하고 선택할 수 있으며, SAM2 대기는 선택 완료 후에만 발생한다.

단일 워커에서 검출·선택 대기·전파를 직렬 처리하면 UI가 선택 전부터 수분간 블로킹된다 — 대안 (b)로 기각했다.

### 3. core 변환 공유 + ui→core box-only 경계

- `core.tracking.build_selections_from_choices`가 사용자 선택 인덱스(`ShotChoice.target_idx`, `negative_idxs`) → `CutSelection`(좌표 + box) 변환을 전담한다. 인덱스-좌표 변환 로직이 UI에 중복 구현되지 않는다(DRY).
- `video_window`의 `_candidate_boxes_from`은 `Detection` 리스트에서 박스 좌표만 추출해 UI에 전달한다. **UI는 `Detection` 객체를 직접 보유하거나 core에 넘기지 않는다**. 이 추출 함수가 UI와 core 사이의 경계를 강제한다(DIP — UI는 좌표 타입만 의존).
- `ShotChoice`는 UI 레이어의 선택 상태 VO로, core `CutSelection`과 분리된 별도 타입이다. 변환은 반드시 `build_selections_from_choices`를 통과한다.

### 4. 박스 오버레이는 이미지 좌표계에서 렌더 후 마스크 오버레이와 함께 스케일

박스를 이미지 픽셀 좌표로 `QPixmap` 위에 직접 그린 뒤, 마스크 오버레이와 함께 위젯 표시 크기로 스케일한다. 박스·마스크·클릭 좌표가 모두 동일 이미지 좌표계에 존재하므로 레터박스 오프셋 변환이 별도로 필요 없다. 위젯 좌표로 박스를 그리고 이미지 좌표 클릭을 역변환하는 방식(투 좌표계)에서 발생하는 오정합 버그를 구조적으로 회피한다.

### 5. 컷 유무 모드 자동 전환 (단일샷 무회귀)

`_DetectWorker` 완료 후 `cut_frames` 결과를 검사한다.

- `cut_frames > 0`: 컷 선택 모드로 전환한다 — 선택 패널 표시 + 박스 오버레이 활성화.
- `cut_frames == 0`: 기존 단일 클릭 모드를 유지한다 — 추가 UI 없음, 기존 동작 바이트 동일.

컷 없는 영상에서 UI 변화가 전혀 없어 단일샷 워크플로에 회귀가 없다.

## 대안

**(a) 캔버스가 직접 히트테스트 + Detection 보유**

`FrameCanvas`가 후보 `Detection` 목록을 속성으로 가지고 클릭 시 자체 판정해 `Detection`을 Signal로 방출한다.

- 거부 이유: `FrameCanvas`(ui 레이어)가 `Detection`(core 타입)을 직접 보유·조작하므로 레이어 경계를 침범한다. 캔버스가 core 함수를 import해야 하며, core 타입 변경이 캔버스 변경을 강제한다(OCP 위반). 단위 테스트 시 Qt·core를 동시에 준비해야 해 테스트 비용이 증가한다.

**(b) 단일 워커에서 검출 + 전파 직렬 처리**

하나의 워커가 `detect_cuts` → `detect_cut_candidates` → 선택 대기 → `propagate`를 순서대로 수행한다.

- 거부 이유: SAM2 전파 시작 전까지 수분간 UI가 블로킹된다. 사용자가 후보 박스를 확인하고 선택하는 인터랙션 자체가 불가능해 컷별 선택 UX 목표를 달성하지 못한다.

## 결과

### 긍정적 영향

- **UX 반응성**: `_DetectWorker` 완료(수초) 즉시 박스 UI가 뜨므로 사용자가 SAM2 대기 전에 컷별 대상을 선택할 수 있다.
- **캔버스 단순성**: 좌표 방출만 담당하므로 core 의존이 없고 Qt 단독 테스트가 가능하다.
- **테스트 용이성**: `pick_box_at`, `build_selections_from_choices` 등 핵심 판정 로직이 순수 함수로 분리되어 단위 테스트가 Qt·SAM2 없이 완전하게 가능하다(Story 4 Task 4-1 순수 함수 커밋으로 선행 확인).
- **단일샷 무회귀**: `cut_frames == 0` 경로는 기존 동작을 그대로 유지한다.
- **좌표 정합 보장**: 단일 이미지 좌표계 렌더로 레터박스 오프셋 버그 발생 경로 자체가 존재하지 않는다.

### 부정적 영향 / 트레이드오프

- **워커 2개로 코드량 증가**: `_DetectWorker`·`_TrackWorker` 각각의 스레드 관리 코드와 시그널-슬롯 연결이 단일 워커 대비 늘어난다. 결정적 `detect_cuts` 재호출(`_TrackWorker` 내부)이 중복이나 성능 영향은 무시 수준이다.
- **선택 UI 필수화**: 컷 있는 영상에서는 사용자가 컷별로 대상을 선택해야 추적이 시작된다. 자동 재매칭 단독 경로(선택 없이 바로 추적)는 군무에서 신뢰 불가하므로 의도적 제거다. 단일 인물 단순 컷에선 자동 재매칭 폴백이 여전히 동작한다(`track(selections=None)` 경로 무변경).
- **`ShotChoice` VO 추가**: UI 레이어에 별도 선택 상태 타입이 생긴다. 변환 함수(`build_selections_from_choices`)와 쌍으로 관리해야 한다.

## 연계

- [ADR 0006](0006-shot-boundary-reid.md) — 컷별 명시 선택(`CutSelection`) 전략의 데스크톱 GUI 구현. 본 ADR이 0006 보완(2026-05-29)의 UI 설계를 구체화한다(Superseded 아님).
- [ADR 0015](0015-negative-point-prompt.md) — negative point(`CutSelection.negative_points`)는 컷별 선택 완료 후 `_TrackWorker`가 `track` 호출 시 함께 전달된다. 본 ADR이 0015를 보완(Superseded 아님).
- [ADR 0012](0012-detection-backend.md) — `_DetectWorker`가 `DetectionBackend`를 통해 Grounding DINO 후보를 열거한다. 추상화 경계 무변경.
- [ADR 0010](0010-video-segmentation-backend.md) — `_TrackWorker`가 `VideoSegmentationBackend.propagate`를 호출한다. 메서드 분리 원칙 무변경.
- [ADR 0002](0002-pyside6-gui.md) — `FrameCanvas`·선택 패널이 PySide6 위젯으로 구현된다. GUI 프레임워크 결정 무변경.
