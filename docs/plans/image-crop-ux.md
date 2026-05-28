# 이미지 모드 크롭 UX 확장 — 개발 계획

> 작성: 2026-05-28 · 단계: `/develop` 파이프라인 planner · 선행 슬라이스: [이미지 모드 happy path](image-mode-slice.md)(완료)
> 관련: [아키텍처](../architecture.md) · [데이터플로우](../data-flow.md) · [ADR 0007](../adr/0007-cpu-dev-strategy.md) · [ADR 0008](../adr/0008-app-usecase-layer.md)

---

## 1. 개요

이미지 모드 happy path("파일→프레임→클릭→SAM2→크롭→저장")가 완성된 상태 위에 **사용자 조정 UX**를 얹는다. 추가하는 가치는 세 가지다.

1. **마스크 오버레이 연결** — 클릭 직후 SAM2 마스크를 캔버스에 반투명으로 표시해 "무엇이 잡혔는지" 즉시 피드백한다. `FrameCanvas.set_overlay`는 이미 구현됐으나 호출 경로가 없어 데드코드 상태다.
2. **종횡비 프리셋 UI** — 1:1 / 9:16 / 16:9 를 툴바에서 선택. core `ASPECT_PRESETS`·`CropRequest.aspect`·`apply_aspect_lock`는 이미 존재하나 UI가 항상 `aspect=None`만 전달한다.
3. **크롭 크기 조정** — 슬라이더로 박스 크기를 조정. 현재 `box_size=(300, 300)`이 main_window에 하드코딩된 매직넘버다.

### 1-1. 이 슬라이스를 지배하는 단 하나의 설계 제약

> **세그멘테이션(무거움, CPU 1~3s)과 박스 계산(가벼움, 순수)을 분리한다.**

종횡비 버튼이나 크기 슬라이더를 움직일 때마다 재세그하면 매 조작마다 1~3초 UI가 멈춰 사용 불가다. 따라서:

- **세그는 클릭당 1회만** 워커 스레드에서 실행하고, 그 결과(마스크·centroid)를 main_window가 보관한다.
- **종횡비/크기 변경은 보관된 centroid만 가지고 순수 계산(`compute_box`)을 메인 스레드에서 즉시 재호출**한다. 모델을 다시 부르지 않는다.

이 분리가 이 계획서의 모든 API·UI·테스트 결정을 끌고 간다.

### 1-2. In / Out (이 슬라이스 경계)

| 구분 | 포함 (In) | 제외 (Out, 다음) |
|---|---|---|
| 오버레이 | 클릭 후 마스크 반투명 표시, 새 클릭/파일 열기 시 갱신·초기화 | 마스크 수동 편집(브러시), 다중 클릭 누적 프롬프트 |
| 종횡비 | 1:1·9:16·16:9·자유(None) 선택, 즉시 박스 갱신 | 임의 비율 입력, 비율 잠금 회전 |
| 크기 | 슬라이더로 박스 한 변 비율 조정, 즉시 갱신 | 박스 드래그 리사이즈 핸들, 위치 미세 이동 |
| 미리보기 | 확정 박스 경계선(rubber band) 표시는 선택(여력 시) | 실시간 크롭 미리보기 패널 |

> KISS: 박스 드래그 핸들·브러시 편집은 의도적으로 제외한다. 이번 목적은 "세그/계산 분리 위에 가벼운 조정 UI"를 얹는 것이다.

---

## 2. 핵심 설계 결정: usecase API 리팩터안

### 2-1. 현재 구조 (단일 무거운 메서드)

```python
# app/image_capture.py (현재)
def make_crop_box(self, frame, request: CropRequest) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    mask = self._backend.segment_image(frame, points=[request.point])   # 무거움(1~3s)
    centroid = centroid_of_mask(mask)                                   # 가벼움
    if centroid is None:
        raise EmptyMaskError("대상을 인식하지 못했어요. 다시 클릭해 주세요.")
    locked_w, locked_h = apply_aspect_lock(*request.box_size, request.aspect)  # 가벼움
    return make_crop_box(centroid, (locked_w, locked_h), (w, h))         # 가벼움
```

문제: 무거운 1줄(`segment_image`)과 가벼운 3줄이 한 메서드에 섞여 있어, 박스만 다시 계산하려 해도 세그를 다시 돌게 된다.

### 2-2. 목표 구조 (무거움 / 가벼움 2단계)

```python
# app/image_capture.py (목표)

@dataclass(frozen=True)
class SegmentResult:
    """세그멘테이션 1회 결과. 박스 재계산에 필요한 가벼운 산출물만 담는다.

    mask: bool HxW — 오버레이 표시용
    centroid: (cx, cy) — 박스 계산용. 빈 마스크는 SegmentResult로 만들지 않고
              segment()에서 EmptyMaskError를 올린다(아래 2-4).
    """
    mask: np.ndarray
    centroid: tuple[float, float]


@dataclass(frozen=True)
class BoxParams:
    """compute_box 입력 묶음 — 매개변수 3개 규칙 준수용(점진 조정 파라미터).

    box_size: 요청 크롭 (W, H)
    aspect: 종횡비 프리셋 키 or None
    frame_shape: (W, H) 프레임 크기(경계 클램프용)
    """
    box_size: tuple[int, int]
    aspect: str | None
    frame_shape: tuple[int, int]


class ImageCaptureUseCase:
    def load_frame(self) -> np.ndarray: ...   # 변경 없음

    def segment(self, frame: np.ndarray, point: tuple[int, int]) -> SegmentResult:
        """클릭 포인트로 1회 세그 → 마스크·centroid 산출(무거움).

        빈 마스크면 EmptyMaskError(정책 유지). 워커 스레드에서 호출.
        """
        mask = self._backend.segment_image(frame, points=[point])
        centroid = centroid_of_mask(mask)
        if centroid is None:
            raise EmptyMaskError("대상을 인식하지 못했어요. 다시 클릭해 주세요.")
        return SegmentResult(mask=mask, centroid=centroid)

    def compute_box(
        self, centroid: tuple[float, float], params: BoxParams
    ) -> tuple[int, int, int, int]:
        """centroid + 조정 파라미터 → 크롭 박스(순수·가벼움). 메인 스레드 즉시 호출.

        종횡비 잠금 → core.make_crop_box(짝수·경계 클램프) 조합. 모델 미호출.
        """
        locked = apply_aspect_lock(*params.box_size, params.aspect)
        return make_crop_box(centroid, locked, params.frame_shape)

    def export(self, frame, box, target) -> None: ...   # 변경 없음
```

- `segment`: 매개변수 2개(+self). 무거운 호출은 여기 한 곳에만.
- `compute_box`: 매개변수 2개(+self), `BoxParams`로 묶어 3개 규칙 충족. **순수**(모델·IO 무의존)하여 단위 테스트가 쉽고 슬라이더가 움직일 때마다 즉시 호출 가능.
- 두 dataclass는 모두 `frozen=True`로 불변. `BoxParams.frame_shape`는 `(W, H)` 순서로 통일(core `make_crop_box`의 `frame_size`와 동일).

### 2-3. 기존 `make_crop_box`(조립 메서드) 처리 — DRY 유지하며 정리

기존 `ImageCaptureUseCase.make_crop_box(frame, request)`는 `segment` + `compute_box`의 조합이다. 두 가지 선택지:

- **(A) 얇은 조합자로 유지(권장)**: `make_crop_box`를 남기되 내부를 `segment`→`compute_box` 호출로 재작성. 이러면 한 번에 박스까지 뽑는 호출부(향후 자동화·배치 등)와 기존 테스트(`test_image_capture.py`의 `TestMakeCropBox`·`TestEndToEndHappyPath`)가 **그대로 통과**한다. DRY: 로직은 `segment`/`compute_box`에 단일 소스로 존재, `make_crop_box`는 위임만.
- (B) 제거: 호출부를 전부 2단계로 바꾸고 기존 테스트도 수정. 회귀 위험·작업량 증가.

> 결정: **(A) 채택.** `make_crop_box`는 `segment`+`compute_box`를 조합하는 20줄 이내 위임 메서드로 재작성한다. `CropRequest`는 (A)에서 계속 쓰이므로 유지한다(필드 변경 없음).

```python
def make_crop_box(self, frame, request: CropRequest) -> tuple[int, int, int, int]:
    """클릭→박스 1회 조합(하위호환). 내부는 segment+compute_box 위임."""
    seg = self.segment(frame, request.point)
    params = BoxParams(request.box_size, request.aspect, _frame_wh(frame))
    return self.compute_box(seg.centroid, params)
```

(`_frame_wh(frame)`은 `frame.shape[:2]`를 `(w, h)`로 뒤집는 모듈 내부 헬퍼. 매직 인덱싱 중복 제거.)

### 2-4. EmptyMaskError 정책 — 유지

- 빈 마스크는 `segment()` 안에서 `EmptyMaskError`를 올린다(현행과 동일 위치·동일 한국어 메시지).
- `compute_box`는 항상 유효한 centroid를 받으므로 이 예외와 무관(순수성 유지).
- 워커는 `segment` 호출만 try로 감싸 `[빈마스크]` 태그로 전달(현행 `_SegWorker.run` 패턴 유지).
- `SegmentResult`는 "성공한 세그"만 표현한다(centroid None을 담지 않음) → 호출부에서 None 분기가 사라져 단순해진다(KISS).

### 2-5. 기존 호출부 영향 요약

| 호출부 | 변경 |
|---|---|
| `tests/test_image_capture.py` | `make_crop_box` 유지(A안)로 **무회귀**. 신규 `segment`/`compute_box` 단위 테스트만 추가 |
| `ui/main_window._SegWorker` | `make_crop_box` 대신 `segment`만 호출하도록 변경(박스 계산은 메인 스레드로 이동) |
| `app/router.py` | 변경 없음(usecase 생성 시그니처 동일) |
| core `crop.py` | 변경 없음(재사용) |

---

## 3. UI 변경

### 3-1. 툴바 확장 (`ui/main_window._build_toolbar`)

기존 "파일 열기 / 저장" 옆에 조정 위젯을 추가한다. SRP를 위해 빌더를 분리한다(`_build_toolbar`는 호출만, 각 그룹은 작은 메서드).

| 위젯 | 종류 | 동작 |
|---|---|---|
| 종횡비 선택 | `QComboBox`(자유/1:1/9:16/16:9) 또는 `QButtonGroup` | 변경 시 `_on_aspect_changed(key)` |
| 크기 슬라이더 | `QSlider`(가로) + 값 라벨 | 변경 시 `_on_size_changed(value)` |

- 종횡비 항목은 `ASPECT_PRESETS` 키 + "자유"(None)를 코드에서 생성(매직 문자열 중복 방지, core 상수가 단일 소스).
- 슬라이더 범위는 상수로 선언: `MIN_CROP_RATIO`/`MAX_CROP_RATIO`/`DEFAULT_CROP_RATIO`. 슬라이더 값(예: 10~100%)을 프레임 최소변 기준 픽셀로 환산하는 변환은 **순수 함수** `crop_ratio_to_size(ratio, frame_shape) -> (w, h)`로 분리(테스트 가능, §5-2).
- 파일 열기 전에는 조정 위젯 비활성(`setEnabled(False)`), 프레임 로드 후 활성.

### 3-2. main_window 상태 보관

현재 보관: `_frame`, `_crop_box`, `_worker`. 여기에 세그/조정 분리를 위해 추가한다.

```python
self._frame: np.ndarray | None = None        # 원본 프레임 (기존)
self._centroid: tuple[float, float] | None = None   # 세그 1회 결과(박스 재계산 기준)
self._mask: np.ndarray | None = None          # 세그 1회 결과(오버레이용; 캔버스에 넘기면 보관 불필요할 수도)
self._crop_box: tuple | None = None           # 현재 확정 박스 (기존)
self._aspect: str | None = DEFAULT_ASPECT     # 현재 선택 종횡비
self._size_ratio: int = DEFAULT_CROP_RATIO    # 현재 슬라이더 값
```

> `_mask`는 캔버스가 set_overlay로 들고 있으므로 main_window가 별도 보관하지 않아도 된다. 다만 "새 종횡비/크기로 박스만 바꾸되 오버레이는 유지"가 자연히 성립하려면 캔버스 오버레이는 클릭 시 한 번만 set하고 박스 갱신 시 건드리지 않는다(아래 흐름).

### 3-3. 갱신 흐름

```
[파일 열기]
  → load_frame → canvas.set_frame → canvas.set_overlay(None)
  → 상태 초기화(_centroid=None, _crop_box=None), 조정 위젯 활성, 저장 비활성

[캔버스 클릭]  ── 무거운 경로(워커 1회) ──
  → _SegWorker(usecase, frame, point).start()   # segment만 호출
  → (성공) seg_ready(mask, centroid)
       · self._centroid = centroid
       · canvas.set_overlay(mask)                 # 오버레이 연결(목표 1) — 여기 단 한 곳
       · self._recompute_box()                    # 가벼운 경로로 위임
  → (빈마스크) "대상을 인식하지 못했어요…" (현행 유지)

[종횡비 변경] / [크기 슬라이더 변경]  ── 가벼운 경로(메인 스레드) ──
  → self._aspect / self._size_ratio 갱신
  → self._recompute_box()        # 세그 재호출 없음

_recompute_box():
  if self._centroid is None: return        # 아직 클릭 전이면 무시
  size = crop_ratio_to_size(self._size_ratio, frame_shape)
  params = BoxParams(size, self._aspect, frame_shape)
  self._crop_box = usecase.compute_box(self._centroid, params)
  canvas.set_box_preview(self._crop_box)   # 선택: 경계선 표시(여력 시)
  저장 버튼 활성, 상태바 갱신
```

핵심: **`_recompute_box`는 순수·즉시**라서 슬라이더 드래그 중 연속 호출돼도 멈춤이 없다. 세그(`_SegWorker`)는 클릭 시에만.

### 3-4. 워커 시그널 변경

```python
class _SegWorker(QThread):
    seg_ready = Signal(np.ndarray, object)   # (mask, centroid) — box 대신 세그 결과
    error = Signal(str)                       # 한국어(현행 [빈마스크] 태그 유지)
    # run(): usecase.segment(frame, point) → seg_ready.emit(res.mask, res.centroid)
```

- 기존 `box_ready(tuple)` → `seg_ready(mask, centroid)`로 변경(박스는 워커 밖에서 계산).
- centroid는 `object`로 전달(튜플). numpy 마스크는 `np.ndarray` 시그널 타입.

### 3-5. box_size 매직넘버 해소

`box_size=(300, 300)` 하드코딩 제거. 대신 슬라이더 값(`_size_ratio`) → `crop_ratio_to_size`로 환산. 기본값은 상수 `DEFAULT_CROP_RATIO`. 더 이상 main_window에 픽셀 매직넘버가 없다(리뷰 백로그 해소).

---

## 4. frame_canvas 오버레이 벡터화

### 4-1. 현재 문제

`_draw_overlay`가 360×640 ≈ 23만 회 픽셀 이중 for-loop + `painter.fillRect(1,1)`. 클릭마다 수십~수백 ms 소요(파이썬 루프). 오버레이 연결(목표 1)을 켜면 체감 지연이 커진다.

### 4-2. 벡터화 방식 — numpy로 RGBA 한 번에 만들고 QImage 1회

핵심 아이디어: bool 마스크 → `(H, W, 4)` uint8 RGBA 배열을 numpy 브로드캐스트로 생성 → `QImage(Format_RGBA8888)` 한 번 → base 위에 1회 `drawImage`. 파이썬 픽셀 루프 제거.

```
def _mask_to_rgba(mask: bool HxW, color=(0,120,255), alpha=110) -> (H,W,4) uint8:   # 순수 함수
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask, 0:3] = color          # 마스크 True 위치에만 색
    rgba[mask, 3]   = alpha          # 알파(반투명). False는 알파 0(투명)
    return np.ascontiguousarray(rgba)

# _draw_overlay 내부(개정):
rgba = _mask_to_rgba(mask)
h, w = mask.shape
overlay_img = QImage(rgba.data, w, h, 4*w, QImage.Format_RGBA8888)
painter.drawImage(0, 0, overlay_img)     # base(QPixmap) 위에 합성
```

요점:
- 색·알파 결정(`_mask_to_rgba`)은 **PySide6 비의존 순수 numpy 함수**로 분리 → 단위 테스트 가능(§5-3). QImage/QPainter는 합성만.
- `Format_RGBA8888`로 알파를 데이터에 담으므로 `painter.setOpacity` 불필요(KISS).
- `rgba`는 QImage가 버퍼를 참조하므로 합성 완료 전까지 GC되지 않게 지역 변수로 잡고 `ascontiguousarray`로 메모리 연속성 보장(QImage stride 안정).
- 오버레이 좌표는 마스크가 이미 프레임 좌표계(HxW)이고 캔버스가 base+overlay를 함께 scaled하므로 **기존 좌표 변환(`coords.py`)은 그대로 유지**(변경 불필요). 마스크는 항상 base와 동일 해상도.

### 4-3. set_box_preview(선택)

여력이 있으면 확정 박스 경계선을 캔버스에 그린다(`QPainter.drawRect`, 1회). 박스 좌표는 프레임 좌표 → scaled에 함께 태워 표시. 미구현 시 상태바 텍스트(`크롭 박스 확정: …`)로 대체(현행 유지). **이번 슬라이스 필수 아님**(여력 시 추가).

---

## 5. 작업 분해 (인터페이스 → 테스트 → 구현 → 리팩터 커밋 순서)

> 브랜치: `feature/image/crop-ux`. Conventional Commits 한국어. 각 줄 = 한 논리적 변경 = 한 커밋. 의존 순서대로 나열.

### 도메인: usecase 분리 (app) — 임계 경로 시작
- [ ] **(인터페이스)** `app/image_capture.py`에 `SegmentResult`·`BoxParams` dataclass + `segment`/`compute_box` 시그니처·docstring 추가(구현 스텁 또는 본문) — 우선순위: 높음
- [ ] **(테스트)** `tests/test_image_capture.py`에 `segment`(정상→SegmentResult, 빈마스크→EmptyMaskError) + `compute_box`(순수: 종횡비/크기 변경 시 박스 재계산, 경계·짝수) 테스트 추가 — 우선순위: 높음
- [ ] **(구현)** `segment`/`compute_box` 본문 + `make_crop_box`를 두 메서드 위임으로 재작성(A안). 기존 테스트 무회귀 확인 — 우선순위: 높음

### 도메인: 크기 변환 순수 함수 (ui, 순수)
- [ ] **(인터페이스+테스트)** `crop_ratio_to_size(ratio, frame_shape)` 위치 결정(`ui/coords.py` 또는 신규 `ui/sizing.py`) + 시그니처 + 단위 테스트(최소/최대/기본 비율 → 픽셀, 짝수·하한 보장) — 우선순위: 높음
- [ ] **(구현)** `crop_ratio_to_size` 구현 → 테스트 통과 — 우선순위: 높음

### 도메인: 오버레이 벡터화 (ui)
- [ ] **(인터페이스+테스트)** `_mask_to_rgba(mask)` 순수 함수 분리(모듈 함수) + 단위 테스트(True→색·alpha, False→투명, shape (H,W,4), dtype uint8) — 우선순위: 높음
- [ ] **(구현)** `_draw_overlay`를 `_mask_to_rgba` + QImage(RGBA8888) 1회 합성으로 교체, 이중 for-loop 제거 — 우선순위: 높음

### 도메인: main_window 배선 (ui)
- [ ] **(구현)** 상수 추가(`DEFAULT_CROP_RATIO`/`MIN`/`MAX`/`DEFAULT_ASPECT`), `box_size` 매직넘버 제거 — 우선순위: 높음
- [ ] **(구현)** `_SegWorker` 시그널 `box_ready`→`seg_ready(mask, centroid)`, `run`이 `segment`만 호출 — 우선순위: 높음
- [ ] **(구현)** 툴바에 종횡비 콤보·크기 슬라이더 추가(`_build_aspect`/`_build_size` 분리), 비활성/활성 토글 — 우선순위: 중간
- [ ] **(구현)** 상태 필드(`_centroid`/`_aspect`/`_size_ratio`) + `_on_seg_ready`(오버레이 set + recompute) + `_on_aspect_changed`/`_on_size_changed`/`_recompute_box` 슬롯 배선 — 우선순위: 중간
- [ ] **(선택)** `FrameCanvas.set_box_preview` + 경계선 렌더(여력 시) — 우선순위: 낮음

### 정리(리팩터) — 리뷰 백로그
- [ ] **(리팩터)** `tests/` 독스트링의 "TDD Red 단계" 잔존 메모 제거(`test_image_capture.py`·`fakes.py` 헤더) — 우선순위: 중간
- [ ] **(문서)** `HANDOFF.md` 갱신(오버레이 연결·종횡비·크기 조정 완료, 백로그 4건 해소 표기) + `CHANGELOG.md` 사용자 영향 기록 — 우선순위: 높음

> 임계 경로: **usecase `compute_box`(순수) → main_window 배선**. 오버레이 벡터화·크기 변환 함수는 usecase와 병렬 가능(서로 독립). UI 배선은 위 세 순수 부품이 준비된 뒤.

---

## 6. 테스트 전략

> 원칙 유지(선행 슬라이스 §5): 무거운/외부 의존(SAM2·PyAV·PySide6 렌더)을 가짜·순수 분리로 격리. 좌표 변환은 기존 `test_coords.py` 그대로 유지(이번 슬라이스에서 좌표 로직 미변경).

### 6-1. usecase 분리 단위 테스트 (`test_image_capture.py` 확장)

| 테스트 | 검증 |
|---|---|
| `segment` 정상 | `FakeBackend` 주입 → `SegmentResult` 반환, `mask`는 bool HxW, `centroid`는 클릭 근방 |
| `segment` 빈마스크 | `FakeBackend(empty_mask=True)` → `EmptyMaskError`(메시지 "다시 클릭" 포함) — **정책 유지 확인** |
| `compute_box` 순수성 | **동일 centroid + 다른 BoxParams를 반복 호출** → 백엔드 `segment_image` 호출 횟수 0(재세그 없음). spy/카운터 FakeBackend로 검증 |
| `compute_box` 종횡비 변경 | centroid 고정, aspect None→9:16→16:9 → 각 박스의 w:h가 프리셋 비율 만족(짝수 오차 ±2) |
| `compute_box` 크기 변경 | centroid 고정, box_size 증감 → 박스 변 길이 단조 증가, 경계·짝수 유지 |
| `make_crop_box` 무회귀 | 기존 `TestMakeCropBox`·`TestEndToEndHappyPath` 그대로 통과(A안 위임) |

> "재세그 안 함" 검증이 이 슬라이스의 핵심 회귀 가드다. `FakeBackend`에 `segment_calls` 카운터를 추가(또는 spy)해 `compute_box` 반복 호출 시 0 증가를 단언한다.

### 6-2. 크기 변환 순수 단위 테스트

- `crop_ratio_to_size(MIN, frame)` / `(MAX, frame)` / `(DEFAULT, frame)` → 픽셀이 프레임 범위 내, 짝수, 하한(0 방지) 보장.
- 프레임 가로/세로 다른 비율에서 최소변 기준 환산이 일관적인지.

### 6-3. 오버레이 벡터화 순수 단위 테스트

- `_mask_to_rgba(mask)` → shape `(H, W, 4)`, dtype uint8.
- 마스크 True 픽셀: RGB == 지정색, A == 지정 alpha. False 픽셀: A == 0(투명).
- 빈 마스크(all False) → 전부 알파 0. 단일 픽셀 마스크 → 정확히 1픽셀만 불투명.
- **PySide6 import 없이** 통과해야 함(numpy만). QImage 합성 자체는 테스트하지 않음(렌더는 수동 스모크).

### 6-4. PySide6 위젯 — 스모크 + 수동

- 위젯 생성·시그널 배선은 offscreen(`QT_QPA_PLATFORM=offscreen`) 스모크 수준(선택). 핵심 로직은 위 순수 함수로 빠져나갔으므로 위젯 단위 테스트 부담 최소화(KISS).
- 수동 스모크(HANDOFF 절차 갱신): `python -m easy_capture` → 클릭(오버레이 표시 확인) → 종횡비 전환(박스 즉시 변경, 멈춤 없음 확인) → 슬라이더 드래그(연속 갱신, 재세그 없음 확인) → 저장.

### 6-5. 커버리지 목표

- 신규 순수 로직(`segment`/`compute_box`/`crop_ratio_to_size`/`_mask_to_rgba`) **80%+**.
- 기존 71개 테스트 무회귀. SAM2/PyAV 실모델은 자동 테스트 제외(현행 유지).

---

## 7. 리스크 및 완화책

| 리스크 | 영향 | 완화책 |
|---|---|---|
| **분리 누락 시 재세그** — 슬라이더가 실수로 `segment`를 부르면 매 조작 1~3s 멈춤 | 사용 불가 | §6-1 "재세그 안 함" 카운터 테스트로 회귀 차단. 코드 리뷰 체크포인트로 명시 |
| **슬라이더 연속 이벤트 폭주** — 드래그 중 `compute_box`가 초당 수십 회 | 순수라 가벼우나 캔버스 재렌더 비용 | `compute_box`는 즉시. 캔버스 재렌더는 가벼운 scaled 1회. 필요 시 `valueChanged` 대신 짧은 디바운스(여력 시, 기본은 즉시) |
| **QImage 버퍼 수명** — `_mask_to_rgba` 결과를 QImage가 참조하는데 조기 GC | 깨진 오버레이/크래시 | 합성이 끝날 때까지 지역 변수 유지 + `ascontiguousarray`로 연속 버퍼 보장. 합성은 동기(한 함수 내 완료) |
| **오버레이/박스 좌표 불일치** — 마스크와 박스가 다른 좌표계로 표시 | 시각 오인 | 마스크·박스 모두 프레임(HxW) 좌표, 캔버스가 함께 scaled. 좌표 변환 로직 미변경(기존 `coords.py` 유지) |
| **새 클릭 시 이전 오버레이 잔존** — centroid는 갱신됐는데 오버레이 안 바뀜 | 혼란 | `_on_seg_ready`에서 `set_overlay(mask)`로 매 세그마다 갱신. 파일 열기 시 `set_overlay(None)` 초기화(현행) |
| **종횡비/크기 조합으로 박스가 과소** — 작은 비율 + 9:16 | 너무 작은 크롭 | `crop_ratio_to_size` 하한(최소 변 보장) + `make_crop_box`의 짝수·클램프가 이미 방어 |
| **빈 마스크 후 조정** — 세그 실패인데 슬라이더 조작 | 조작 무의미 | `_recompute_box`가 `_centroid is None`이면 즉시 return(상태바 안내 유지) |

---

## 8. 방법론 가이드 (이 슬라이스 적용)

- **SRP**: 무거운 `segment`(IO·모델)와 순수 `compute_box`(계산)를 메서드 단위로 분리. UI는 표현/배선만, 계산은 usecase·순수 함수로.
- **DIP/주입**: usecase는 여전히 `SegmentationBackend` Protocol에만 의존. `FakeBackend`로 세그/박스 분리를 모델 없이 검증.
- **DRY**: 박스 산출 로직은 `compute_box` 단일 소스. `make_crop_box`는 위임만(중복 없음). 종횡비 키는 core `ASPECT_PRESETS` 단일 소스에서 UI 항목 생성.
- **KISS**: 박스 드래그 핸들·브러시 편집 제외. 오버레이는 알파를 데이터에 담아 `setOpacity` 생략. 박스 미리보기 경계선은 선택(여력 시).
- **함수 20줄·매개변수 3개**: `segment`(2)·`compute_box`(2, `BoxParams`로 묶음)·`crop_ratio_to_size`(2)·`_mask_to_rgba`(기본인자 포함). 4개 이상이 될 입력은 `SegmentResult`·`BoxParams`로 묶는다.
- **커밋 순서**: 인터페이스 → 테스트 → 구현 → 리팩터. 도메인 스코프(`feat(app):`, `feat(ui):`, `perf(ui):` 오버레이, `refactor(test):` 독스트링 정리). 커밋-문서 동기화(HANDOFF/CHANGELOG) 필수.

---

## 9. 완료 정의 (DoD)

- [ ] 클릭 후 마스크가 반투명 오버레이로 표시된다(목표 1, `set_overlay` 데드코드 해소).
- [ ] 종횡비(자유/1:1/9:16/16:9) 전환 시 박스가 **재세그 없이** 즉시 갱신된다(목표 2).
- [ ] 크기 슬라이더 조작 시 박스가 **재세그 없이** 즉시 갱신된다(목표 3, `box_size` 매직넘버 제거).
- [ ] `_draw_overlay` 픽셀 이중루프 제거, numpy RGBA + QImage 1회로 교체(리뷰 백로그 해소).
- [ ] `compute_box` 순수 단위 테스트 + "재세그 안 함" 카운터 테스트 통과. 기존 71개 무회귀.
- [ ] core 경계 유지(torch/transformers/PySide6/av 미import). 테스트 독스트링 "TDD Red" 잔존 정리.
- [ ] HANDOFF/CHANGELOG 갱신, 수동 스모크 절차로 멈춤 없는 조정 확인.
```