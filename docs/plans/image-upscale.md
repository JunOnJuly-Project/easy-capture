# 이미지 모드 업스케일(SwinIR/Swin2SR) — 개발 계획

> 작성: 2026-05-28 · 단계: `/develop` 파이프라인 planner · 선행 슬라이스: [이미지 모드 happy path](image-mode-slice.md)(완료) · [크롭 UX](image-crop-ux.md)(완료)
> 관련: [아키텍처](../architecture.md) · [ADR 0004 업스케일러](../adr/0004-upscaler-realesrgan-swinir.md) · [ADR 0007 CPU 전략·백엔드 추상화](../adr/0007-cpu-dev-strategy.md) · [ADR 0008 app/usecase 레이어](../adr/0008-app-usecase-layer.md)

---

## 1. 개요

이미지 모드("파일→클릭→SAM2→크롭→종횡비/크기 UX→저장")가 완성된 상태 위에 **저장 직전 옵션 초해상도 업스케일**을 얹는다. 사용자가 업스케일을 켜고 배율(repo)을 고르면, 크롭 결과 이미지를 저장 전에 SwinIR/Swin2SR로 N배 확대한다. 미선택 시 기존 즉시 저장 경로를 그대로 유지한다.

### 1-1. 이 슬라이스를 지배하는 단 하나의 설계 제약

> **순수 로직(crop/save)과 무거운 모델 추론(upscale)을 깨끗이 분리한다.**

crop-ux 슬라이스에서 "세그(무거움) ↔ 박스 계산(가벼움)"을 분리한 교훈을 그대로 적용한다. 업스케일은 CPU에서 수 초가 걸리는 무거운 모델 추론이므로:

- **core는 순수 유지**: `crop_array`·`save_image`는 Pillow·numpy만 의존(torch/transformers 절대 import 금지). 출력 정규화 같은 순수 산술도 core에 둔다.
- **무거운 추론은 infra + 워커 스레드**: Swin2SR 구현은 `infra/`에, 지연 로드, UI 저장 시 워커 스레드에서 실행("업스케일 중…" 표시).
- **추상화는 Protocol**: `UpscaleBackend` Protocol을 `core/upscale`에 두고(순수, torch 비의존), 구현(`Swin2srUpscaleBackend`)은 infra에. `SegmentationBackend`↔`Sam2ImageBackend`와 **완전히 동일한 패턴**(ADR 0007).

이 분리가 이 계획서의 모든 API·파이프라인·UI·테스트 결정을 끌고 간다.

### 1-2. In / Out (이 슬라이스 경계)

| 구분 | 포함 (In) | 제외 (Out, 다음) |
|---|---|---|
| 백엔드 | SwinIR/Swin2SR(`Swin2srUpscaleBackend`) **1종만 구현** | Real-ESRGAN 구현(인터페이스로 확장만 대비, 이번 미구현) |
| 배율 | repo로 고정된 배율(x2/x4) 중 선택 = "repo 선택" | 임의 배율, 타일/슬라이딩 업스케일, 가변 배율 보간 |
| 파이프라인 | crop → (옵션) upscale → save. 미선택 시 기존 즉시 저장 | 업스케일 결과 미리보기 패널, 저장 전 before/after 비교 |
| UI | 업스케일 on/off + 배율(repo) 콤보, 워커 진행 표시 | 모델 자동 추천(실사/애니 판별), 배치 업스케일 |
| 영역 | **크롭 영역만** 업스케일(전체 프레임 아님) | 원본 전체 프레임 업스케일, temporal smoothing(v1.1) |

> KISS: 타일 업스케일·미리보기·before/after는 의도적으로 제외. 이번 목적은 "순수/무거움 분리 위에 옵션 업스케일 단계를 얹는 것"이다.

---

## 2. 핵심 설계 결정 1 — `UpscaleBackend` Protocol (core/upscale, 순수)

### 2-1. 위치와 경계 불변식

`SegmentationBackend`(`core/segmentation/backend.py`)와 대칭으로 신규 패키지 `core/upscale/`에 Protocol을 둔다. **core 경계 불변식 준수**: torch·transformers·PySide6·PyAV import 금지. Protocol은 타입 계약만 선언하므로 numpy만 의존한다.

```python
# src/easy_capture/core/upscale/backend.py (신규)
"""업스케일 백엔드 인터페이스 (ADR 0004 / ADR 0007).

SwinIR/Swin2SR(기본) 또는 Real-ESRGAN(옵션, v1.x)을 교체한다.
SegmentationBackend와 대칭 패턴: Protocol은 core(순수), 구현은 infra.
경계 불변식: torch·transformers·PySide6·PyAV import 금지.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class UpscaleBackend(Protocol):
    device: str
    scale: int  # 모델이 적용하는 고정 배율(예: 2, 4). repo가 결정.

    def upscale(self, image_rgb: np.ndarray) -> np.ndarray:
        """RGB HxWx3 uint8 → (H*scale, W*scale, 3) uint8 RGB 반환(무거움)."""
        ...
```

설계 근거:
- `device`·`scale` 속성 + `upscale(image_rgb)` 단일 메서드. `SegmentationBackend.device` 패턴과 정합. `scale`을 속성으로 노출해 UI가 "결과 크기 = 크롭 × scale"을 미리 계산·고지할 수 있게 한다(배율은 모델별 고정).
- 반환 계약: **RGB uint8 (H*scale, W*scale, 3)**. bool 아님. `save_image`/`crop_array`가 받는 형식과 동일하므로 파이프라인이 추가 변환 없이 이어진다.
- `supports()` 같은 가변 능력 질의는 이번 미도입(KISS). Real-ESRGAN 확장 시 동일 Protocol을 구현하면 되므로 OCP 충족(아래 §6).

### 2-2. "업스케일 없음" 처리 방식 — **None(주입 안 함)으로 표현**

옵션 처리에 두 선택지:
- **(A) None 채택(권장)**: 업스케일 미선택 = 백엔드를 `None`으로 둔다. 파이프라인이 `backend is None`이면 crop→save 직행. NullObject(IdentityUpscaleBackend) 대비 분기 1개로 단순하고, "모델 미로드"가 None과 자연히 일치(불필요한 객체·로드 없음).
- (B) `IdentityUpscaleBackend`(NullObject, `scale=1`, 그대로 반환): 분기 제거는 깔끔하나, scale=1 객체를 항상 만들고 "업스케일 안 함"을 객체로 표현하는 게 오히려 의미 흐림. 또 미사용 시에도 import 경로가 살아있어 경계가 흐려질 수 있음.

> 결정: **(A) None 채택.** usecase `export`가 `upscaler: UpscaleBackend | None = None`을 받고, None이면 업스케일 단계를 건너뛴다. crop-ux의 `compute_box` 순수성 결정과 같은 철학 — "안 하는 경우는 안 부른다."

---

## 3. 핵심 설계 결정 2 — `Swin2srUpscaleBackend` (infra)

### 3-1. 위치와 패턴

`infra/sam2_image_backend.py`를 거울처럼 따라 `infra/swin2sr_upscale_backend.py`를 만든다. 생성자는 repo·device 보관만, 무거운 모델 로드는 첫 `upscale` 호출까지 지연(ADR 0007).

```python
# src/easy_capture/infra/swin2sr_upscale_backend.py (신규)
"""Swin2SR 초해상도 백엔드 (transformers 5.9.0, ADR 0004).

UpscaleBackend Protocol 구현체. SegmentationBackend↔Sam2ImageBackend와 동일 패턴.
생성자는 repo·device·scale 보관만, 모델 로드는 첫 upscale 호출까지 지연(지연 로드).
배율은 repo에 의해 고정(x2 모델=2배). "배율 선택"=repo 선택.
"""
from __future__ import annotations

import numpy as np

from easy_capture.core.upscale.normalize import reconstruction_to_rgb_uint8


class Swin2srUpscaleBackend:
    """Swin2SR 업스케일 백엔드. UpscaleBackend Protocol을 준수한다."""

    def __init__(self, repo: str, device: str, scale: int) -> None:
        """repo·device·scale 보관만. 모델은 아직 로드하지 않는다(지연).

        WHY: 모델 로드(수백 MB)는 '저장' 클릭 후 워커에서 1회만 수행해
             UI 응답성을 보장한다. scale은 repo가 결정하므로 주입받아 보관.
        """
        self.device = device
        self.scale = scale
        self._repo = repo
        self._model = None
        self._processor = None

    def upscale(self, image_rgb: np.ndarray) -> np.ndarray:
        """RGB HxWx3 uint8 → (H*scale, W*scale, 3) uint8 RGB(무거움)."""
        self._ensure_loaded()
        return self._run_inference(image_rgb)

    def _ensure_loaded(self) -> None:
        """모델 미로드 시 from_pretrained로 지연 로드(이중 로드 방지)."""
        if self._model is not None:
            return
        from transformers import (
            Swin2SRForImageSuperResolution,
            Swin2SRImageProcessor,
        )
        self._processor = Swin2SRImageProcessor.from_pretrained(self._repo)
        self._model = (
            Swin2SRForImageSuperResolution.from_pretrained(self._repo)
            .to(self.device)
            .eval()
        )

    def _run_inference(self, image_rgb: np.ndarray) -> np.ndarray:
        """모델 추론 → reconstruction 텐서를 순수 정규화 함수로 RGB uint8 변환."""
        import torch

        inputs = self._processor(image_rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._model(**inputs)
        recon = out.reconstruction  # (1, 3, H*scale, W*scale) float
        recon_np = recon[0].detach().cpu().float().numpy()  # (3, Hs, Ws)
        return reconstruction_to_rgb_uint8(recon_np)
```

요점:
- `_ensure_loaded`/`_run_inference` 분리는 `Sam2ImageBackend`와 동일(SRP, 20줄 규칙). torch/transformers는 메서드 내부 지연 import(생성자·모듈 상단 import 금지 — 미사용 시 무거운 import 방지, ADR 0007).
- **출력 정규화 산술은 infra가 아니라 core 순수 함수로 분리**한다(§3-2). infra는 텐서→numpy 추출(`detach().cpu().float().numpy()`)까지만 하고, 클램프·스케일·축전치·dtype 변환은 순수 함수에 위임. → torch 비의존 단위 테스트 가능.

### 3-2. 출력 정규화 순수 함수 (core/upscale, 순수)

```python
# src/easy_capture/core/upscale/normalize.py (신규)
"""Swin2SR reconstruction 텐서를 RGB uint8 이미지로 정규화(순수 numpy).

경계 불변식: torch·transformers import 금지. (3,H,W) float ndarray만 받는다.
WHY: 정규화 산술(clip·*255·축전치·uint8)을 infra에서 떼어내 순수 함수로 두면
     torch 없이 단위 테스트할 수 있고, Real-ESRGAN 등 다른 백엔드도 재사용한다(DRY).
"""
from __future__ import annotations

import numpy as np

_MAX_UINT8 = 255


def reconstruction_to_rgb_uint8(recon_chw: np.ndarray) -> np.ndarray:
    """(3, H, W) float [0,1) 근사 → (H, W, 3) uint8 RGB.

    Given: 채널 우선(CHW) float 배열(범위가 [0,1] 밖으로 약간 벗어날 수 있음)
    When:  [0,1] 클램프 → *255 반올림 → uint8 → HWC 전치
    Then:  (H, W, 3) uint8, 값 0~255
    """
    clamped = np.clip(recon_chw, 0.0, 1.0)
    scaled = np.rint(clamped * _MAX_UINT8).astype(np.uint8)  # (3, H, W)
    return np.transpose(scaled, (1, 2, 0))                   # (H, W, 3)
```

> **검증 필요(구현 단계, venv)**: Swin2SR `reconstruction`의 정확한 범위([0,1] vs [0,255])·채널 순서를 실모델로 확인한다. 본 정규화는 transformers 공식 예제 기준 [0,1] 가정. 만약 [0,255]거나 BGR이면 이 **순수 함수 1곳만** 수정한다(SRP의 이점). 입력 dtype/shape 계약(CHW float)은 유지.

### 3-3. repo → scale 매핑 (infra/device 확장)

`infra/device.py`의 `SAM2_REPO_BY_DEVICE`·`select_sam2_repo` 패턴을 따라 업스케일 repo 카탈로그를 둔다. 배율은 repo가 결정하므로 (repo, scale)을 함께 보관한다.

```python
# infra/device.py 에 추가 (또는 신규 infra/upscale_catalog.py — §3-4 결정)
@dataclass(frozen=True)
class UpscaleModel:
    """업스케일 모델 카탈로그 1항목. UI 라벨·repo·고정 배율을 묶는다."""
    label: str          # UI 표시("x2 (선명·범용)")
    repo: str           # HF repo id
    scale: int          # 고정 배율

UPSCALE_MODELS: tuple[UpscaleModel, ...] = (
    UpscaleModel("x2 (범용·선명)", "caidas/swin2SR-classical-sr-x2-64", 2),
    UpscaleModel("x4 (실사·강한 확대)", "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr", 4),
)
```

> repo 후보·정확한 출력 형식은 ADR 0004·resources.md §2 기준. **구현 단계에서 venv로 실제 출력 shape·범위 검증** 후 확정. x2/x4 외 추가는 이 튜플에만 항목을 더한다(OCP).

### 3-4. 백엔드 조립 — router(composition root)

`Sam2ImageBackend`를 router가 만들어 usecase에 주입하듯, 업스케일 백엔드도 **router에서 선택·생성해 주입**한다(DIP). 미선택 시 None. 다만 "배율(repo) 선택"이 런타임 UI 조작이므로, 백엔드 생성 시점이 SAM2와 다르다:

- SAM2: 모드 진입 시 1회 생성(파일 무관) → factory 클로저가 캡처.
- 업스케일: **선택한 repo가 바뀌면 다른 모델**이라 "저장 시점에 현재 선택 repo로 백엔드를 만든다." → router가 `repo→Swin2srUpscaleBackend` 생성 팩토리를 제공하고, main_window가 "켜짐 + 선택 repo"일 때만 호출.

```python
# app/router.py 확장 (개념)
def _build_upscaler_factory(self, device):
    """repo·scale → Swin2srUpscaleBackend 생성 팩토리(지연 로드 백엔드)."""
    from easy_capture.infra.swin2sr_upscale_backend import Swin2srUpscaleBackend

    def make(model):  # model: UpscaleModel
        return Swin2srUpscaleBackend(model.repo, device, model.scale)
    return make
```

> **(A) router가 repo별 백엔드를 캐시**(같은 repo 재선택 시 재로드 방지) vs **(B) 매 저장 시 생성**(지연 로드라 첫 추론 때만 무겁고, 객체 자체는 가벼움). 결정: **간단함 우선 (B)**, 단 같은 repo 연속 저장이 흔하면 main_window가 "마지막 생성 백엔드 + repo" 1개를 캐시(메모리 1모델). 구현 시 결정, 기본은 (B)+main_window 단일 캐시.

---

## 4. 핵심 설계 결정 3 — export 파이프라인 연결

### 4-1. 현재 export (crop → save 2단계)

```python
# app/image_capture.py (현재)
def export(self, frame, box, target: tuple[str, ExportConfig]) -> None:
    path, config = target
    cropped = crop_array(frame, box)   # 순수
    save_image(cropped, path, config)  # IO
```

### 4-2. 목표 — crop → (옵션) upscale → save 3단계

업스케일을 **usecase `export`의 선택적 매개변수**로 받는다(별도 메서드 신설 대비 호출부 단순). 매개변수 3개 규칙을 위해 기존 `target` 튜플은 유지하고 `upscaler`만 키워드로 추가:

```python
# app/image_capture.py (목표)
def export(
    self,
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    target: tuple[str, ExportConfig],
    upscaler: UpscaleBackend | None = None,
) -> None:
    """크롭 → (옵션) 업스케일 → 저장.

    upscaler가 None이면 기존 즉시 저장 경로(무회귀). 주어지면 크롭 결과만
    업스케일(전체 프레임 아님) 후 저장. 무거운 추론은 호출자(워커)가 책임진다.

    WHY: 업스케일을 별도 단계로 끼워 순수(crop/save)와 무거움(upscale)을 분리.
         upscaler 주입은 DIP — usecase는 UpscaleBackend Protocol에만 의존.
    """
    path, config = target
    cropped = crop_array(frame, box)
    image = upscaler.upscale(cropped) if upscaler is not None else cropped
    save_image(image, path, config)
```

- 매개변수 4개(self 제외 3개 + 키워드 1)인데, `target` 튜플 묶음으로 위치 인자는 3개 유지. `upscaler`는 기본값 None 키워드라 기존 호출(`export(frame, box, target)`)이 **무회귀**.
- 함수 본문 ≤ 20줄(현재 4줄 → 5줄). 분기 1줄(삼항)로 KISS.
- **무거움 책임 분리**: usecase `export`는 조립만 한다. 워커 스레드 실행 책임은 UI(`_UpscaleSaveWorker`)에 있다. usecase는 스레드를 모른다(SRP).

> **(대안) 별도 메서드 `export_upscaled`**: 분기 없는 두 메서드는 깔끔하나 DRY 위반(crop/save 중복) + 호출부가 2갈래. 옵션 1개 분기는 위 형태가 가장 단순. 채택: **export에 옵션 매개변수.**

### 4-3. ADR 필요 여부 — documenter에게 위임

> 업스케일을 **별도 usecase 메서드가 아닌 `export`의 옵션 매개변수로 통합**하는 결정, 그리고 **출력 정규화를 core 순수 함수로 분리**하는 결정은 가벼운 ADR로 기록할 가치가 있다(향후 Real-ESRGAN·배치 확장 시 근거). 본 계획서는 결정만 명시하고, **ADR 작성은 documenter 단계**에서 ADR 0009(예: "업스케일을 export 옵션으로 통합 + core 정규화 분리")로 다룬다(planner는 작성하지 않음).

### 4-4. 기존 호출부 영향 요약

| 호출부 | 변경 |
|---|---|
| `tests/test_image_capture.py` | 기존 `export` 호출 무회귀(키워드 기본 None). `upscaler` 주입 테스트만 추가 |
| `ui/main_window._on_export` | 업스케일 on/off·repo 읽어 워커로 분기(§5) |
| `app/router.py` | `_build_upscaler_factory` 추가, main_window에 팩토리+카탈로그 전달 |
| `core/export/image_export.py` | **변경 없음**(crop_array·save_image 재사용) |

---

## 5. UI 변경 (`ui/main_window.py`)

### 5-1. 툴바 확장

기존 "파일 열기 / 저장 / 종횡비 / 크기" 옆에 업스케일 위젯을 추가한다. SRP를 위해 빌더 분리(`_build_upscale_controls`).

| 위젯 | 종류 | 동작 |
|---|---|---|
| 업스케일 켜기 | `QCheckBox`("업스케일") | 체크 시 배율 콤보 활성, `_upscale_on` 갱신 |
| 배율(모델) 선택 | `QComboBox`(`UPSCALE_MODELS`의 label) | 선택 시 `_upscale_model` 갱신. 체크 꺼지면 비활성 |

- 콤보 항목은 `UPSCALE_MODELS`에서 생성(매직 문자열 중복 방지, 카탈로그가 단일 소스 — crop-ux의 `ASPECT_PRESETS` 처리와 동일).
- 파일 열기 전·크롭 박스 확정 전에는 비활성. 종횡비/크기 위젯과 동일한 활성 토글 흐름.
- 결과 크기 고지(선택): 체크 시 상태바에 "결과 예상 크기: {W×scale}×{H×scale}" 표시(scale은 선택 모델의 `.scale`). 여력 시.

### 5-2. 상태 보관 추가

```python
self._upscale_on: bool = False            # 업스케일 체크 여부
self._upscale_model = None                # 선택된 UpscaleModel (None=미선택)
self._upscaler_factory = ...              # router 주입: UpscaleModel -> UpscaleBackend
self._upscale_catalog = UPSCALE_MODELS    # router 주입: 콤보 항목 소스
self._cached_upscaler = None              # (선택) 마지막 생성 백엔드 1개 캐시
self._cached_repo: str | None = None      # 캐시 유효성 판단용
```

> router는 main_window 생성 시 `upscaler_factory`와 `upscale_catalog`를 주입한다(DIP). main_window는 transformers/torch를 직접 import하지 않는다 — 팩토리 호출만.

### 5-3. 저장 흐름 — 워커 스레드 분기

```
[저장 클릭] _on_export:
  if 박스 없음/프레임 없음: return
  경로·포맷 다이얼로그 → config 생성 (현행 유지)

  if not self._upscale_on or self._upscale_model is None:
      usecase.export(frame, box, (path, config))   # 기존 즉시 저장 경로(무회귀)
      상태바 "저장 완료" (현행)
  else:
      upscaler = self._get_or_make_upscaler(self._upscale_model)  # 캐시 or 팩토리
      self._save_btn.setEnabled(False)
      상태바 "업스케일 중… (처음 실행 시 모델 로드로 시간이 걸릴 수 있습니다)"
      self._save_worker = _UpscaleSaveWorker(usecase, frame, box, (path, config), upscaler)
      self._save_worker.done.connect(self._on_save_done)
      self._save_worker.error.connect(self._on_save_error)
      self._save_worker.start()
```

핵심: 업스케일은 **무거우므로 워커 스레드**(`_SegWorker`와 동일 패턴). 미선택 시 기존 동기 즉시 저장(가벼움 = crop+save). 단일 워커 정책(이미 실행 중이면 중복 클릭 무시·안내, `_on_canvas_click` 패턴 재사용).

### 5-4. 신규 워커 `_UpscaleSaveWorker`

```python
class _UpscaleSaveWorker(QThread):
    """크롭+업스케일+저장을 백그라운드에서 실행하는 워커.

    WHY: CPU Swin2SR가 수 초 걸려 메인 스레드 실행 시 UI가 얼기 때문에 분리.
         usecase.export(upscaler=...) 한 번만 호출하고 결과(저장 경로)를 emit.
    """
    done = Signal(str)    # 저장 완료 경로
    error = Signal(str)   # 한국어 오류 메시지

    def __init__(self, usecase, frame, box, target, upscaler) -> None:
        super().__init__()
        self._args = (usecase, frame, box, target, upscaler)   # 5개 → 튜플 묶음

    def run(self) -> None:
        usecase, frame, box, target, upscaler = self._args
        path, _ = target
        try:
            usecase.export(frame, box, target, upscaler=upscaler)
            self.done.emit(path)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"업스케일/저장 오류: {exc}")
```

- 생성자 인자 5개 → `_args` 튜플 1필드로 보관(매개변수 3개 규칙은 메서드 시그니처 기준이나, 가독성·언팩으로 관리). 또는 dataclass `UpscaleSaveRequest`로 묶어도 됨(구현 시 택1, dataclass 권장).
- `_on_save_done`: 상태바 "저장 완료: {path}", 저장 버튼 재활성. `_on_save_error`: QMessageBox + 버튼 재활성.

### 5-5. `_on_export` 20줄 규칙

현재 `_on_export`(21줄)에 분기가 추가되면 초과한다. **다이얼로그/config 생성**과 **저장 실행 분기**를 분리:
- `_on_export`: 경로·config 확보까지(또는 더 작게).
- `_save_with_optional_upscale(path, config)`: 업스케일 on/off 분기·워커 시작.
- `_get_or_make_upscaler(model)`: 캐시 조회 또는 팩토리 호출(repo 변경 시 재생성).

---

## 6. 작업 분해 (인터페이스 → 테스트 → 구현 → 리팩터 커밋 순서)

> 브랜치: `feature/image/upscale`. Conventional Commits 한국어. 각 줄 = 한 논리적 변경 = 한 커밋. 의존 순서대로 나열. 도메인 스코프: `feat(core)`·`feat(infra)`·`feat(app)`·`feat(ui)`.

### 도메인: upscale Protocol + 정규화 순수 함수 (core/upscale) — 임계 경로 시작
- [ ] **(인터페이스)** `core/upscale/__init__.py` + `core/upscale/backend.py`에 `UpscaleBackend` Protocol(`device`/`scale`/`upscale`) 정의 — 우선순위: 높음
- [ ] **(테스트)** `tests/test_upscale_normalize.py`: `reconstruction_to_rgb_uint8` 단위 테스트(shape (3,H,W)→(H,W,3), dtype uint8, [0,1]밖 클램프, 흑/백/중간값 매핑) — 우선순위: 높음
- [ ] **(구현)** `core/upscale/normalize.py`의 `reconstruction_to_rgb_uint8` 구현 → 테스트 통과(순수, torch 비의존) — 우선순위: 높음

### 도메인: FakeUpscaleBackend (tests/fixtures)
- [ ] **(테스트 인프라)** `tests/fixtures/fakes.py`에 `FakeUpscaleBackend`(Protocol 준수, torch 비의존, scale배 nearest 확대 + `upscale_call_count` 스파이) 추가 — 우선순위: 높음

### 도메인: export 파이프라인 연결 (app)
- [ ] **(인터페이스)** `app/image_capture.py` `export`에 `upscaler: UpscaleBackend | None = None` 키워드 추가 + docstring — 우선순위: 높음
- [ ] **(테스트)** `tests/test_image_capture.py`: ① `upscaler=None`이면 업스케일 미호출(기존 무회귀) ② `FakeUpscaleBackend` 주입 시 저장 이미지 크기 = 크롭×scale ③ upscale 1회만 호출(`upscale_call_count==1`) — 우선순위: 높음
- [ ] **(구현)** `export` 본문에 옵션 업스케일 단계 끼우기(crop→upscale?→save) → 테스트 통과 — 우선순위: 높음

### 도메인: Swin2SR 백엔드 구현 (infra)
- [ ] **(구현)** `infra/swin2sr_upscale_backend.py`: `Swin2srUpscaleBackend`(지연 로드, `_ensure_loaded`/`_run_inference` 분리, 정규화는 core 함수 위임) — 우선순위: 높음
- [ ] **(구현)** `infra/device.py`(또는 `infra/upscale_catalog.py`): `UpscaleModel` + `UPSCALE_MODELS`(x2/x4 repo·scale) — 우선순위: 높음
- [ ] **(수동 스모크)** venv에서 실모델 1장 업스케일 → reconstruction 범위·채널·출력 shape 확인, 필요 시 `normalize.py` 1곳 보정 — 우선순위: 높음

### 도메인: router 조립 (app)
- [ ] **(구현)** `app/router.py`: `_build_upscaler_factory(device)` + `_launch_image_mode`에서 main_window에 `upscaler_factory`·`upscale_catalog` 주입 — 우선순위: 중간

### 도메인: main_window 배선 (ui)
- [ ] **(구현)** 업스케일 체크박스·배율 콤보 추가(`_build_upscale_controls`), 활성/비활성 토글 — 우선순위: 중간
- [ ] **(구현)** `_UpscaleSaveWorker`(또는 `UpscaleSaveRequest` dataclass + 워커) + `_on_save_done`/`_on_save_error` — 우선순위: 중간
- [ ] **(구현)** `_on_export` 분리(`_save_with_optional_upscale`/`_get_or_make_upscaler`), on/off 분기·캐시 — 우선순위: 중간

### 정리(리팩터)·문서
- [ ] **(문서)** `HANDOFF.md` 갱신(업스케일 슬라이스 완료·백엔드/정규화 위치·수동 스모크 절차) + `CHANGELOG.md` 사용자 영향 기록 — 우선순위: 높음
- [ ] **(문서·위임)** ADR 0009 초안 트리거 표시(export 옵션 통합·core 정규화 분리) — documenter 단계 — 우선순위: 중간

> 임계 경로: **`UpscaleBackend` Protocol → `reconstruction_to_rgb_uint8`(순수) → `export` 연결**. Swin2SR 실구현·router·UI 배선은 그 위. FakeUpscaleBackend가 있으면 UI/usecase 배선을 실모델 없이 끝까지 테스트할 수 있어 Swin2SR 구현과 **병렬 가능**.

---

## 7. 테스트 전략

> 원칙 유지(선행 슬라이스): 무거운/외부 의존(SAM2·Swin2SR·PyAV·PySide6 렌더)을 가짜·순수 분리로 격리. Swin2SR 실모델은 자동 테스트 제외(수동 스모크).

### 7-1. 정규화 순수 단위 테스트 (`test_upscale_normalize.py`)

| 테스트 | 검증 |
|---|---|
| shape 변환 | (3, H, W) → (H, W, 3) |
| dtype | 결과 uint8 |
| 클램프 하한 | 음수 입력 → 0 |
| 클램프 상한 | 1.0 초과 입력 → 255 |
| 매핑 정확성 | 0.0→0, 1.0→255, 0.5→128(반올림) |
| 채널 보존 | R/G/B 채널 값이 전치 후에도 올바른 위치 |

- **torch import 없이** numpy만으로 통과해야 함(core 경계 검증).

### 7-2. export 파이프라인 단위 테스트 (`test_image_capture.py` 확장)

| 테스트 | 검증 |
|---|---|
| `export` 업스케일 없음 | `upscaler=None`(기본) → `FakeUpscaleBackend.upscale_call_count==0`, 저장 이미지 크기 = 크롭 크기(무회귀) |
| `export` 업스케일 적용 | `FakeUpscaleBackend(scale=2)` 주입 → 저장 PNG 재로드 크기 = 크롭W×2, 크롭H×2 |
| `export` 1회 호출 | upscale이 정확히 1회만 호출(`upscale_call_count==1`) — 중복 추론 회귀 가드 |
| 기존 export 무회귀 | `TestEndToEndHappyPath` 등 기존 export 테스트 그대로 통과 |

> `FakeUpscaleBackend`는 scale배 nearest 확대(`np.repeat`)로 결정적 출력 → 저장 크기를 정확히 예측. `upscale_call_count` 스파이로 "정확히 1회·미선택 시 0회"를 단언(crop-ux의 `segment_call_count` 패턴 재사용).

### 7-3. Swin2SR 실모델 — 수동 스모크 (자동 제외)

- venv(`.venv`)에서 `Swin2srUpscaleBackend(repo, "cpu", scale)`로 작은 크롭(예: 128×128) 1장 업스케일:
  - 반환 shape == (128×scale, 128×scale, 3), dtype uint8.
  - `reconstruction`의 실제 범위/채널 순서 확인 → `normalize.py` 가정 검증·보정.
  - 결과를 PNG로 저장해 육안 화질 확인(블러·아티팩트).
- **자동 테스트에서 transformers/torch 로드 금지**(CI·반복 속도). 수동 스모크 절차를 HANDOFF에 기록.

### 7-4. PySide6 위젯 — 스모크 + 수동

- 위젯 생성·시그널 배선은 offscreen(`QT_QPA_PLATFORM=offscreen`) 스모크 수준(선택). 핵심 로직(정규화·export)은 순수/usecase로 빠졌으므로 위젯 단위 부담 최소(KISS).
- 수동 스모크: `python -m easy_capture` → 클릭·크롭 → 업스케일 체크·x2 선택 → 저장("업스케일 중…" 표시·UI 비멈춤 확인) → 결과 크기·화질 확인 → 체크 해제 후 저장(즉시 저장 무회귀 확인).

### 7-5. 커버리지 목표

- 신규 순수/조립 로직(`reconstruction_to_rgb_uint8`·`export` 업스케일 분기) **80%+**.
- 기존 테스트 전부 무회귀. Swin2SR/SAM2/PyAV 실모델은 자동 테스트 제외(현행 유지).

---

## 8. 리스크 및 완화책

| 리스크 | 영향 | 완화책 |
|---|---|---|
| **reconstruction 형식 불일치** — 범위([0,1] vs [0,255])·채널 순서가 가정과 다름 | 색 깨짐·검은/흰 이미지 | 정규화를 **core 순수 함수 1곳**으로 분리 → venv 수동 스모크로 확인 후 그 1곳만 보정. CHW float 입력 계약은 유지(infra는 변경 불필요) |
| **CPU 업스케일 수 초 지연** | "저장 클릭 후 멈춤" 오인 | 워커 스레드(`_UpscaleSaveWorker`) + "업스케일 중…" 상태바 + 저장 버튼 비활성. 단일 워커 정책으로 중복 클릭 무시 |
| **첫 추론 시 모델 다운로드·로드(수백 MB)** | 첫 저장이 매우 느림 | 지연 로드(ADR 0007) + "처음 실행 시 모델 로드로 시간이 걸릴 수 있습니다" 안내. repo별 백엔드 1개 캐시로 재선택 시 재로드 방지 |
| **4K 크롭 업스케일 메모리** — 큰 크롭 × x4 = 수억 픽셀 텐서 | OOM·스왑 | 크롭 영역만 업스케일(전체 프레임 아님)로 입력 축소. x4 + 대형 크롭 조합 시 경고(여력 시 입력 변 상한 가드). 타일 업스케일은 v1.1 |
| **core 경계 오염** — infra·정규화에 torch가 core로 새어듦 | 순수성·테스트성 붕괴 | `core/upscale`는 numpy만 import(Protocol·정규화). torch/transformers는 infra 메서드 내부 지연 import. 경계 테스트(import 검사)로 가드 |
| **워커 결과 dtype/shape 비정상** — 모델이 예외·빈 출력 | 저장 실패·크래시 | `_UpscaleSaveWorker`가 예외를 `error` 시그널로 한국어 변환(현행 `_SegWorker` 패턴). 정규화 함수가 shape/clamp를 강제 |
| **체크 해제했는데 업스케일됨(반대도)** — 상태 동기화 누락 | 의도와 다른 저장 | `_save_with_optional_upscale`이 `_upscale_on and model is not None`을 단일 지점에서 판단. 체크 해제 시 콤보 비활성으로 시각적 정합 |
| **repo 변경 후 이전 모델 사용** — 캐시 무효화 누락 | x2 선택했는데 x4 결과 | `_get_or_make_upscaler`가 `_cached_repo != model.repo`면 재생성. 캐시 키 = repo |
| **Real-ESRGAN 확장 시 인터페이스 변경 필요** | 재작업 | `UpscaleBackend` Protocol은 백엔드 무관(device/scale/upscale). Real-ESRGAN도 동일 Protocol 구현 → export·UI·테스트 무변경(OCP). 이번엔 인터페이스만 대비 |

---

## 9. 방법론 가이드 (이 슬라이스 적용)

- **SRP**: 순수 정규화(`reconstruction_to_rgb_uint8`) / 무거운 추론(`Swin2srUpscaleBackend`) / 조립(`export`) / 스레드(`_UpscaleSaveWorker`) / 표현(툴바)을 각각 한 책임으로 분리. usecase는 스레드를 모르고, 백엔드는 UI를 모른다.
- **OCP**: Real-ESRGAN 추가 = `UpscaleBackend` 새 구현 + 카탈로그 항목 추가만. export·UI·테스트 무변경. repo 추가는 `UPSCALE_MODELS` 튜플에만.
- **DIP/주입**: usecase는 `UpscaleBackend` Protocol에만 의존. 구체 Swin2SR 조립은 router(composition root). 테스트는 `FakeUpscaleBackend`로 모델 없이 검증.
- **DRY**: 크롭/저장은 `crop_array`·`save_image` 단일 소스 재사용(변경 없음). 정규화 산술은 `reconstruction_to_rgb_uint8` 1곳 → Real-ESRGAN도 동일 함수 재사용 가능. repo·라벨·배율은 `UPSCALE_MODELS` 단일 소스에서 UI 항목 생성.
- **KISS**: "업스케일 없음"=None 분기 1개. NullObject·타일·미리보기·before/after 제외. 배율 선택=repo 선택(별도 배율 보간 없음).
- **함수 20줄·매개변수 3개**: `export`는 `target` 튜플 + `upscaler` 키워드로 위치 인자 3개 유지. `_UpscaleSaveWorker` 5개 인자는 `_args` 튜플(또는 `UpscaleSaveRequest` dataclass)로 묶음. `_on_export`는 다이얼로그/분기/캐시 헬퍼로 분할.
- **core 경계 불변식**: `core/upscale`(Protocol·정규화)는 numpy만. torch/transformers는 infra 지연 import. PySide6는 ui만.
- **커밋 순서**: 인터페이스 → 테스트 → 구현 → 리팩터. 커밋-문서 동기화(HANDOFF/CHANGELOG/ADR) 필수.

---

## 10. 완료 정의 (DoD)

- [ ] `UpscaleBackend` Protocol(core/upscale, torch 비의존) + `reconstruction_to_rgb_uint8` 순수 함수 + 단위 테스트 통과.
- [ ] `Swin2srUpscaleBackend`(infra, 지연 로드, 정규화는 core 위임) 구현 + venv 수동 스모크로 출력 형식 확인.
- [ ] `export`에 옵션 `upscaler` 연결 — None이면 즉시 저장(무회귀), 주입 시 크롭×scale 저장. `FakeUpscaleBackend`로 "1회 호출·미선택 0회" 단위 테스트 통과.
- [ ] UI: 업스케일 체크 + 배율(repo) 콤보, 저장 시 워커("업스케일 중…")로 비멈춤. 미선택 시 기존 즉시 저장.
- [ ] router가 `upscaler_factory`·`upscale_catalog` 주입(DIP). main_window는 torch/transformers 미import.
- [ ] core 경계 유지(`core/upscale` numpy만, torch/transformers/PySide6/av 미import). 기존 테스트 전부 무회귀.
- [ ] HANDOFF/CHANGELOG 갱신, ADR 0009 트리거 표시(documenter 위임), 수동 스모크 절차로 업스케일 화질·UI 응답성 확인.
```