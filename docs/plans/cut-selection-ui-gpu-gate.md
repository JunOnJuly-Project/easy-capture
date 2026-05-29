# 데스크톱 컷별 선택 UI — GPU 게이트 재현 검증 체크리스트

- 문서 유형: 검증 계획 / 체크리스트
- 대상 태스크: Story 4 Task 4-6
- 작성일: 2026-05-29
- 상태: 대기 중 (실 GPU 실행 미완)

---

## 1. 목표

데스크톱 GUI(컷별 선택 UI)가 노트북(Colab)에서 이미 통과한 GPU 게이트와 **동등한 selections 입력**을 생성함을 보장한다.

### 배경 — 이미 통과한 노트북 게이트

`poc/colab/easy_capture_app_verify.ipynb` 셀 7.5 → 셀 8 경로로 다음 수치를 달성했다
(HANDOFF.md §3 GPU 블로커 해소 항목에서 인용):

| 측정 항목 | 수치 | 기준 |
|---|---|---|
| AC-01 추적 유지율 | **100% (300/300프레임)** | ≥ 80% |
| needs_correction | **0** | 0 목표 |
| AC-06 GPU 처리 속도 | **2.0 fps** | 측정값 기록 |

적용 조건: `hiera-small` 모델 + 컷별 선택(box + negative point) + 올바른 `shot_index` 키 + 멀티샷 군무 300프레임 / 컷 6개 → 4샷.

참고: `hiera-tiny`는 밀착 재합침으로 AC-01 미달 → `hiera-small` 이상이 필요하다. `largest_component` 후처리 철회로 0.7 fps → 2.0 fps 복구됨(ADR 0015).

### 이 문서의 범위

GUI 코드가 노트북 셀 7.5와 **구조적으로 동등한 selections**를 생성하는지 체크리스트로 명시한다. 실 GPU 재현 절차(Colab 주입 방법), 제약, 통과 기준을 함께 기술한다.

---

## 2. 동등성 체크리스트

GUI 경로 각 단계가 노트북 셀과 일치하는지 항목별로 확인한다.

### 2-1. detect_cut_candidates — 샷 인덱스 키 정합

**노트북 셀 7.5**

```python
shot_cands = usecase.detect_cut_candidates(frames, cut_frames)
# 반환: ShotCandidates(shot_index=0..N-1, first_frame_index, candidates)
```

**GUI 경로**

`_DetectWorker.run()`:

```python
cut_frames = usecase.detect_cuts(video_path, span) or []
shots = usecase.detect_cut_candidates(frames, cut_frames)
self.candidates_ready.emit((cut_frames, shots))
```

| 항목 | 확인 조건 | 상태 |
|---|---|---|
| `detect_cut_candidates` 호출 시그니처 | `(frames, cut_frames)` — 노트북과 동일 | ✅ 코드 확인 |
| `shot_index` 범위 | 0-기반, `n_shots` 미만 — `validate_selections`가 보장 | ✅ 코드 확인 |
| 컷 없을 때 빈 배열 처리 | `cut_frames = ... or []` — `None` → `[]` 변환 후 `detect_cut_candidates`에 전달 | ✅ 코드 확인 |
| 헤드리스 테스트 커버 | `TestDetectWorker.test_run이_cut_frames와_shots를_튜플로_방출한다` | ✅ 통과 (672 테스트 기준) |

### 2-2. build_selections_from_choices — 노트북 셀 7.5 변환 이식

**노트북 셀 7.5 핵심 변환 루프**

```python
def _box_center(box):
    return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

SELECTIONS = []
for sc in shot_cands:
    ti = SHOT_TARGETS.get(sc.shot_index)
    box  = tuple(float(v) for v in sc.candidates[ti].box)
    point = _box_center(box)
    negs  = tuple(
        _box_center(sc.candidates[ni].box)
        for ni in NEG_TARGETS.get(sc.shot_index, [])
        if ni != ti and ni < len(sc.candidates)
    )
    SELECTIONS.append(CutSelection(sc.shot_index, point, box=box, negative_points=negs))
```

**GUI 대응 함수**: `core/tracking/cut_selection.py` → `build_selections_from_choices`

| 항목 | 확인 조건 | 상태 |
|---|---|---|
| `box_center` 로직 | `int((x1+x2)/2), int((y1+y2)/2)` — 노트북 `_box_center`와 동일 절단(내림) | ✅ 코드 확인 |
| `box` 필드 | `tuple(float(v) for v in ...)` → `CutSelection.box` — float 4-튜플 동일 | ✅ 코드 확인 |
| `negative_points` 생성 | `ni != target_idx and 0 <= ni < len(boxes)` — 노트북 필터와 동일 | ✅ 코드 확인 |
| `shot_index` 오름차순 정렬 | `for shot_index in sorted(choices)` — 결정적 출력 | ✅ 코드 확인 |
| 미선택 샷 건너뜀 | `target_idx is None` → `None` 반환 → 자동 재매칭 폴백 | ✅ 코드 확인 |
| `validate_selections` 호출 | 빌드 후 범위·중복 검증 | ✅ 코드 확인 |
| 헤드리스 테스트 커버 | `test_cut_selection.py` — `build_selections_from_choices` 단위 테스트 | ✅ 통과 |

### 2-3. _candidate_boxes_from — Detection → box 경계 변환

**GUI 경로**: `video_window._candidate_boxes_from`

```python
def _candidate_boxes_from(shots):
    return [[d.box for d in shot.candidates] for shot in shots]
```

UI는 `Detection` 객체를 core에 직접 넘기지 않는다. box 좌표만 추출해 `build_selections_from_choices`에 전달한다(ADR 0016 §3 레이어 경계).

| 항목 | 확인 조건 | 상태 |
|---|---|---|
| `Detection.box` 추출 | 각 `Detection`에서 `box` 필드만 추출 | ✅ 코드 확인 |
| 2중 리스트 구조 | `list[list[tuple]]` — `build_selections_from_choices`의 `candidate_boxes` 인자 형식과 일치 | ✅ 코드 확인 |
| 헤드리스 테스트 커버 | `TestCandidateBoxes.test_샷별_box_리스트를_추출한다` | ✅ 통과 |

### 2-4. _TrackWorker — selections 전달

**핵심 연결**: `_TrackWorker.__init__`에 `selections` 인자가 명시적으로 전달되고, `run()` 내부에서 `usecase.track(selections=self._selections)`로 넘어가야 한다.

이 연결이 누락되면 GUI 흐름 전체가 노트북과 달라진다.

| 항목 | 확인 조건 | 상태 |
|---|---|---|
| `_start_track_worker(point, selections)` 호출 | `_on_track`에서 컷 모드일 때 `selections`를 빌드해 전달 | ✅ 코드 확인 |
| `_TrackWorker` 생성자 | `selections=selections` kwarg 전달 | ✅ 코드 확인 |
| `_TrackWorker.run()` | `usecase.track(..., selections=self._selections)` — kwarg 전달 | ✅ 코드 확인 |
| 단일 모드 무회귀 | `_cut_mode=False`이면 `selections=None` — 자동 경로(노트북 셀 8 `_selections=[]` 상당) | ✅ 코드 확인 |
| 헤드리스 테스트 커버 | `TestSelectionsWiring.test_컷_모드_추적은_selections를_전달한다` | ✅ 통과 |
| 헤드리스 테스트 커버 | `TestSelectionsWiring.test_TrackWorker_run이_track에_selections를_kwarg로_넘긴다` | ✅ 통과 |

### 2-5. track(selections=) → usecase 전달

**노트북 셀 8**

```python
track_result = usecase.track(
    frames, CLICK_POINT, cut_frames=cut_frames, selections=_selections
)
```

**GUI 경로**: `_TrackWorker.run()` → `usecase.track(frames, point, cut_frames=..., selections=self._selections)`

`cut_frames`는 `_TrackWorker._resolve_cut_frames()`가 `usecase.detect_cuts(video_path, span)`을 재호출해 확보한다. 이 재호출은 결정적이며(같은 영상·구간) `selections`의 `shot_index`와 정합된다(ADR 0016 §2).

| 항목 | 확인 조건 | 상태 |
|---|---|---|
| `track` 인자 일치 | `(frames, point, cut_frames=..., selections=...)` — 노트북 셀 8과 동일 | ✅ 코드 확인 |
| `cut_frames` 정합 | `_resolve_cut_frames()`가 동일 `video_path·span`으로 재실행 → `shot_index` 범위 보장 | ✅ 코드 확인 |

### 2-6. 박스 클릭 hit-test

**GUI 전용 단계** (노트북에는 없는 인터랙션):

캔버스 클릭 좌표 `(x, y)` → `pick_box_at((x, y), boxes)` → `CutSelectionPanel.set_target(idx)`

| 항목 | 확인 조건 | 상태 |
|---|---|---|
| `pick_box_at` 겹침 시 최소 넓이 선택 | core 순수 함수 — Qt 의존 없음 | ✅ 코드 확인 |
| 캔버스 좌표계 | 이미지 픽셀 좌표 → 히트테스트도 동일 좌표계 (ADR 0016 §4) | ✅ 코드 확인 |
| 헤드리스 테스트 커버 | `TestBoxClickHitTest.test_박스_내부_클릭이_해당_후보를_대상으로_지정한다` | ✅ 통과 |

### 2-7. 단일샷 무회귀 (컷 없는 영상)

| 항목 | 확인 조건 | 상태 |
|---|---|---|
| `cut_frames == 0` → 단일 모드 유지 | `_on_candidates_ready(([], []))` → `_cut_mode=False` | ✅ 통과 |
| `selections=None`으로 `track` 호출 | 기존 단일 클릭 경로 무변경 | ✅ 통과 |

---

## 3. 실 GPU 재현 절차 (Colab)

로컬 CPU에서의 SAM2 비디오 추적은 ≈ 0.10 fps로 실용적이지 않다. 검증 경로는 Colab GPU(T4 이상)다.

### 3-1. 직접 재현 방법 (권장)

아래 절차로 `easy_capture_app_verify.ipynb`의 셀 7.5 → 셀 8 경로를 GUI와 동등하게 재현한다.

1. Colab 런타임 → GPU(T4) 설정.
2. 셀 1: 저장소 클론 또는 `origin/main` 강제 갱신(셀 안의 `git reset --hard` 참조).
3. 셀 2~3: 군무 직캠 클립(300프레임 이상, 컷 4개 이상) 업로드 + 프레임 추출.
4. 셀 6: `SAM2_REPO = "facebook/sam2.1-hiera-small"` 로 변경 후 백엔드 조립.
   - `hiera-tiny`는 밀착 재합침으로 AC-01 미달함이 확인됐다(HANDOFF.md §3). `small` 이상이 필수.
5. 셀 7: 컷 감지 실행. `cut_frames` 확인.
6. 셀 7.5: 각 샷 후보 시각화 → `SHOT_TARGETS`, `NEG_TARGETS` 지정.
   - 이 단계가 GUI의 "컷 감지 → 박스 클릭 → 패널 선택" 흐름과 동등하다.
7. 셀 8: `track(selections=SELECTIONS)` 실행. AC-01, needs_correction, fps 측정.

### 3-2. GUI selections 직렬화 주입 방법 (제안)

GUI 경로에서 생성된 `selections`를 Colab에 직접 주입하면 완전한 end-to-end 동등성을 검증할 수 있다. 이 방법은 현재 구현되어 있지 않으나 아래 방식으로 추가 가능하다.

**방법 A — JSON 직렬화 (제안)**

GUI 측 `_resolve_cut_track_inputs()` 직후 `selections`를 JSON으로 덤프:

```python
# video_window.py 임시 디버그 코드 (검증 후 제거)
import json, pathlib
payload = [
    {"shot_index": s.shot_index, "point": list(s.point),
     "box": list(s.box) if s.box else None,
     "negative_points": [list(p) for p in s.negative_points]}
    for s in selections
]
pathlib.Path("debug_selections.json").write_text(json.dumps(payload, ensure_ascii=False))
```

Colab에서:

```python
import json
from easy_capture.core.tracking.cut_selection import CutSelection

raw = json.loads(open("debug_selections.json").read())
SELECTIONS = [
    CutSelection(
        shot_index=r["shot_index"],
        point=tuple(r["point"]),
        box=tuple(r["box"]) if r["box"] else None,
        negative_points=tuple(tuple(p) for p in r["negative_points"]),
    )
    for r in raw
]
```

이 방법은 GUI가 생성한 selections를 노트북 셀 8에 그대로 주입해 동등성을 직접 확인한다.

**방법 B — GPU PC 직접 실행 (최종 검증)**

GPU PC에서 `python -m easy_capture` → 비디오 모드 → 군무 클립 → 컷 감지 → 박스 선택 → 추적 실행. 결과 GIF/MP4로 추적 품질을 육안 확인. 이 경로는 GPU PC 환경이 전제된다.

### 3-3. 측정 항목

| 항목 | 측정 방법 | 목표 |
|---|---|---|
| AC-01 추적 유지율 | `n_tracked / n_frames × 100` (셀 8 콘솔 출력) | 100% (노트북 통과 수치 재현) |
| needs_correction | `sum(track_result.needs_correction)` | 0 |
| AC-06 GPU 처리 속도 | `n_frames / elapsed` (fps) | 2.0 fps 이상 (현재 달성치 유지) |

---

## 4. 제약 / 미해결

### 4-1. 코드로 검증된 항목 (헤드리스 테스트, GPU 불필요)

아래 항목은 `tests/test_video_window_cut_selection.py` 및 관련 테스트로 검증 완료다.
테스트 총 수는 HANDOFF.md 기준 672개(Story 4 Task 4-5 완료 시점).

| 테스트 클래스 | 검증 내용 |
|---|---|
| `TestDetectWorker` | `_DetectWorker`가 `(cut_frames, shots)` 튜플을 방출 |
| `TestSelectionsWiring` | 컷 모드 추적 시 `selections`가 `_TrackWorker`에 전달됨 |
| `TestSelectionsWiring` | `_TrackWorker.run()`이 `track`에 `selections`를 kwarg로 넘김 |
| `TestSelectionsWiring` | 단일 모드는 `selections=None` (무회귀) |
| `TestModeSwitch` | 컷 없으면 단일 모드, 컷 있으면 컷 모드 + 패널 표시 |
| `TestModeSwitch` | 구간 변경 시 컷 모드 해제 |
| `TestModeSwitch` | 영상 재오픈 시 컷 모드·후보·클릭점 초기화 |
| `TestCandidateBoxes` | `Detection` → `box` 경계 변환 정확성 |
| `TestBoxClickHitTest` | 캔버스 박스 클릭 hit-test → 대상 지정 |
| `TestSelectionError` | `ValueError` 발생 시 워커 미생성 + 경고 표시 |
| `TestSingleClickRegression` | 단일 클릭 추적 무회귀 |
| `test_cut_selection.py` | `build_selections_from_choices`, `pick_box_at`, `box_center` 순수 함수 단위 테스트 |

### 4-2. 실 GPU 대기 항목

| 항목 | 이유 |
|---|---|
| AC-01 100% 재현 (GUI 경로) | SAM2 비디오 추적은 로컬 CPU에서 ≈ 0.10 fps — 실검증 불가 |
| needs_correction 0 재현 (GUI 경로) | 동일 이유 |
| AC-06 fps 측정 (GUI 경로) | GPU PC 또는 Colab 필요 |
| GPU PC 수동 스모크 | GUI 인터랙션 전체 — "컷 감지" → 박스 클릭 → 패널 선택 → "추적 실행" 흐름 |

### 4-3. 알려진 제약

- **AC-06 2.0 fps**: 목표(계획서 ≥ 10 fps) 미달 상태. `hiera-small` + 군무 기준으로 현재 달성치다. fps 개선은 백로그(EdgeTAM 등 경량 백엔드 v1.1)로 이관.
- **GUI 수동 스모크 전제**: 실 GPU 검증은 GPU PC 환경 또는 Colab에 GUI selections 직렬화 주입을 전제한다. GPU PC 없이는 방법 A(직렬화 주입)로 근사 검증만 가능.
- **동일 영상 사용 권장**: 노트북 통과 시 사용한 군무 클립(300프레임, 컷 6개 → 4샷)과 동일 영상으로 재현해야 수치 비교가 유효하다.

---

## 5. 통과 기준

GPU 재현 시 아래 항목을 모두 만족하면 Task 4-6 완료 및 Story 4 GPU 게이트 통과로 판정한다.

| 항목 | 통과 조건 |
|---|---|
| AC-01 추적 유지율 | **100% (300/300프레임)** — 노트북 통과 수치 동일 재현 |
| needs_correction | **0** |
| AC-06 GPU 처리 속도 | **2.0 fps 이상** (현재 달성치 유지. 10 fps는 백로그) |
| GUI 경로 동등성 | 위 섹션 2의 체크리스트 전 항목 ✅ 확인 완료 |

게이트 통과 시 `HANDOFF.md` §3 미완료 항목에서 Task 4-6을 완료 처리하고 다음 작업(노트북 `SAM2_REPO` `hiera-small` 기본화 등)으로 진행한다.

---

## 6. 참고

- HANDOFF.md §3 — GPU 블로커 해소, 멀티샷 GPU 게이트 통과 수치
- `poc/colab/easy_capture_app_verify.ipynb` — 노트북 검증 경로 (셀 7.5, 셀 8)
- `docs/adr/0016-cut-selection-ui.md` — GUI 아키텍처 결정 (워커 분리, 레이어 경계)
- `docs/adr/0006-shot-boundary-reid.md` — 컷별 명시 선택 전략
- `docs/adr/0015-negative-point-prompt.md` — negative point 도입 배경
- `src/easy_capture/core/tracking/cut_selection.py` — 순수 변환 함수
- `src/easy_capture/ui/video_window.py` — GUI 통합 (컷 모드 흐름)
- `tests/test_video_window_cut_selection.py` — 헤드리스 검증 테스트
