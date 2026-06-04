# 프레임 보간(Frame Interpolation) 후보 조사 — 부드러운 슬로우모션 v1.1

> 작성일: 2026-06-04 · 상태: 조사(코드 미수정) · 관련: [ADR 0013](../adr/0013-time-remap-location.md) · [ADR 0009](../adr/0009-upscale-export-integration.md) · [ADR 0010](../adr/0010-video-segmentation-backend.md) · [ADR 0018](../adr/0018-edgetam-video-backend.md) · [ac06-fps-improvement](ac06-fps-improvement.md) · [resources.md](../resources.md)
>
> 목적: 현재 슬로우모션은 **프레임 복제(stutter)** 방식([ADR 0013] 결과 §"MVP 슬로우 품질 한계")이다. v1.1로 예약된 **부드러운 슬로우(프레임 보간)**를 위해 RIFE 및 대안 모델을 2025~2026 현황 기준으로 조사하고, 1순위 후보·통합 설계 초안·Colab PoC 절차를 제시한다. **본 문서는 조사 결과이며 코드 변경을 포함하지 않는다.** T4 실측이 없는 값은 모두 "추정"으로 표기한다.

---

## 0. 왜 보간인가 — 현재 stutter의 근거

`core/timing/timeremap.py`의 슬로우 경로는 **프레임을 정수배 복제**해 CFR 시간축을 늘린다(`schedule_to_cfr_indices` — "슬로우=같은 인덱스 정수배 복제"). 0.5x이면 각 프레임을 2번, 0.25x이면 4번 반복한다. 같은 그림이 N번 반복되므로 움직임이 끊겨 보이는 **stutter**가 발생한다([ADR 0013] 결과 §부정적 영향: "프레임 복제 방식은 stutter(끊김)이 보인다. 부드러운 슬로우(RIFE 보간)는 v1.1 후속으로 명시").

**보간의 핵심 아이디어**: 복제 대신, 인접한 두 프레임 (Iₜ, Iₜ₊₁) 사이에 **모델이 추론한 중간 프레임**을 끼워 넣는다. 0.5x 슬로우는 매 쌍 사이에 중간 프레임 1장(timestep=0.5)을, 0.25x는 3장(timestep=0.25/0.5/0.75)을 생성한다. 같은 그림 반복이 아니라 부드럽게 보간된 새 프레임이라 움직임이 자연스럽다.

---

## 1. 후보 비교표

| 후보 | 라이브러리/도입 경로 | 설치난이도 | 라이선스 | 입출력(timestep) | 품질·속도 | Colab 적합성 |
|---|---|---|---|---|---|---|
| **RIFE (Practical-RIFE v4.25/4.26)** | GitHub `hzwer/Practical-RIFE` 클론 + 모델 가중치(Google Drive). PyPI 정식 패키지 없음 | 중(가중치 수동 다운로드) | **MIT**(코드+가중치) ✅ | v4.x는 **임의 timestep** 지원(temporal encoding 입력). 2x/4x/8x·임의 ratio | 실시간급 경량. 720p T4 **추정 수십~100+fps** | **상** — 의존성 가벼움(torch만), Colab GPU 자연스러움 |
| **RIFE (ncnn-vulkan)** | PyPI `rife-ncnn-vulkan-python` (1.2.x) | 하(pip 1줄) | MIT | v4 모델만 임의 timestep | Vulkan 추론. Colab은 CUDA 우선이라 부적합 | 하 — Vulkan 의존, Colab GPU와 결 안 맞음 |
| **Google FILM** | GitHub `google-research/frame-interpolation`(TF, **2025-10-14 아카이브**) 또는 PyTorch 포트 `dajes/frame-interpolation-pytorch` + HF 가중치 `jkawamoto/frame-interpolation-pytorch` | 중(TF 원본 무겁다 / PyTorch 포트는 클론) | **Apache 2.0** ✅ | 재귀 분할(midpoint) — **2의 거듭제곱(2x/4x/8x)** 중심. 임의 timestep은 비자연 | **큰 모션 품질 최상**. 무거움(T4 추정 RIFE보다 느림) | 중 — 원본 TF 아카이브됨, PyTorch 포트는 pip 패키지 아님 |
| **EMA-VFI** | GitHub `MCG-NJU/EMA-VFI` | 중 | **Apache 2.0** ✅ | 임의 timestep 변형 존재 | 고품질. RIFE보다 무거움 | 중 |
| **AMT** | GitHub `MCG-NJU/AMT`, 가중치 HF | 중 | (확인 필요 — 추정 MIT 계열) | 고정 ratio 중심 | Vimeo90K SOTA급, 효율적 | 중 |
| **IFRNet** | GitHub | 중 | (확인 필요) | — | AMT에 0.17dB 추월당함 | 중 |

> ⚠ "추정" fps는 T4 실측 공개치가 없는 값이다. RIFE 논문 표제가 "Real-Time"이고 FILM은 품질 우선(상대적으로 무거움)이라는 일반 평가에 근거한 **상대 순위**이며, 절대 fps는 §5 노트북 측정으로 확정한다.

---

## 2. 권장 1순위 — **RIFE (Practical-RIFE v4.25/4.26 계열, PyTorch)**

### 결정 근거 (우선순위: 라이선스 안전 → Colab 쉬움 → 품질)

1. **라이선스 안전 (최우선)** — Practical-RIFE·ECCV2022-RIFE 모두 **MIT 라이선스**이며, 코드와 모델 가중치를 구분하는 별도 비상업 조항이 없다([Practical-RIFE README]: "The content of these links is under the same MIT license as this project"). 이는 프로젝트의 상업 안전 경로 정책([resources.md] §2: SAM2·DINO·SwinIR Apache 2.0 / GPL·DIV2K 회피)과 **완전히 부합**한다. FILM(Apache 2.0)·EMA-VFI(Apache 2.0)도 안전하지만, RIFE는 그중 가장 가볍다.
   - ⚠ **가중치 출처 1회 대조**([resources.md] 맨 위 정책): 가중치를 Google Drive에서 받으므로, 실제 받은 가중치 파일에 동봉된 라이선스를 구현 착수 전 1회 확인한다(README의 MIT 진술과 일치 여부).
2. **Colab 쉬움** — 의존성이 **torch만**으로 가볍다(transformers·ncnn 불필요). 이미 프로젝트가 torch+CUDA Colab 경로([ac06] §5 노트북)를 갖췄으므로 셀 추가만으로 동작한다. ncnn-vulkan 변형은 pip 1줄로 더 쉽지만 **Vulkan 의존이라 Colab CUDA 환경과 결이 안 맞아** 제외한다.
3. **품질·속도 균형** — RIFE는 "Real-Time" 표제대로 경량이라, EdgeTAM 추적(13.5fps, [ADR 0018])과 합쳐질 때 보간 비용이 **추적 대비 작게** 유지될 가능성이 높다(추정). FILM이 큰 모션에서 품질 우위지만 무겁고, 움짤(크롭된 인물 중심, 모션이 극단적이지 않음)에는 RIFE 품질로 충분하다고 판단한다. **품질이 부족하면 §3 Protocol로 FILM 백엔드를 교체**할 수 있게 설계한다(OCP).
4. **임의 timestep 지원** — v4.x 모델은 temporal encoding을 입력 채널로 받아 **0≤t≤1 임의 timestep**의 중간 프레임을 1장씩 생성할 수 있다([ECCV2022-RIFE 논문]). 0.5x(t=0.5 1장)·0.25x(t=0.25/0.5/0.75 3장)·임의 배율을 자연스럽게 매핑한다 — 이것이 [ADR 0013]의 factor 기반 segment와 직결된다(§3).

### transformers/HuggingFace 통합 여부

- **RIFE는 HF transformers에 정식 통합되어 있지 않다**(SAM2·EdgeTAM·Swin2SR과 다른 점). 따라서 `UpscaleBackend`처럼 transformers `from_pretrained` 한 줄로 끼우는 패턴은 불가하고, **GitHub 코드 + 가중치 파일을 infra 백엔드 안에 벤더링/지연 로드**하는 방식이 된다. 가중치는 HF Hub 미러를 두거나 모델 다운로드 매니저([resources.md] §7)로 받는다.
- 차선책으로 FILM은 HF 가중치(`jkawamoto/frame-interpolation-pytorch`, Apache 2.0)가 있어 가중치 배포는 더 깔끔하나, 패키지가 pip 아님은 RIFE와 동일하다.

---

## 3. 통합 설계 초안

### 3-1. `core/interpolate/` — `InterpolationBackend` Protocol (신규)

`UpscaleBackend`([ADR 0009])·`VideoSegmentationBackend`([ADR 0010])와 **대칭 패턴**: Protocol은 core(순수, torch 비의존), 구현은 infra.

```python
# core/interpolate/backend.py  (torch/transformers/av import 금지 — core 불변식)
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class InterpolationBackend(Protocol):
    """프레임 보간 백엔드 Protocol.

    device: 'cpu' | 'cuda'
    호출 계약: 인접 두 프레임 + timestep → 중간 프레임 1장.
    반환: RGB HxWx3 uint8 (입력과 동일 크기).
    """
    device: str

    def interpolate(
        self,
        frame_a: np.ndarray,   # RGB HxWx3 uint8 (Iₜ)
        frame_b: np.ndarray,   # RGB HxWx3 uint8 (Iₜ₊₁)
        timestep: float,       # 0 < t < 1
    ) -> np.ndarray:
        """Iₜ·Iₜ₊₁ 사이 timestep 위치 중간 프레임 1장 생성(무거움)."""
        ...
```

- **매개변수 3개 규칙 준수**: `interpolate(a, b, timestep)`.
- **단일 timestep 단위 인터페이스**: "중간 N장"이 아니라 "timestep 1장"을 기본 단위로 둔다. WHY — 임의 배율(0.25x=3장, 0.5x=1장)을 core 순수 함수가 **timestep 리스트로 전개**하고 백엔드는 1장씩만 책임지게 해 ISP·테스트 용이성을 높인다. (배치 최적화가 필요하면 후속 OCP 확장으로 `interpolate_batch`를 더한다.)
- **opaque 없음**: 추적과 달리 세션 상태가 없어 stateless. `init_session`류 불필요.

구현체는 `infra/rife_interpolation_backend.py`(지연 로드 — 첫 `interpolate`에서 RIFE 모델 로드, [ADR 0007] 패턴). FILM 교체 시 `infra/film_interpolation_backend.py` 추가 + router 분기(OCP).

### 3-2. core 순수 함수 — 스케줄을 보간 계획으로 전개

[ADR 0013]의 `PlaybackSchedule`은 "프레임 복제" 전제다. 보간은 **복제 대신 중간 프레임 삽입**이므로, 복제 인덱스 시퀀스가 아니라 **삽입 계획(InterpolationPlan)**이 필요하다. `core/timing` 또는 `core/interpolate`에 순수 함수를 둔다(torch 비의존, 단위 테스트 가능 — [ADR 0013] 90% 커버리지 정신 계승):

```python
# 예: core/interpolate/plan.py (순수, numpy/stdlib만)
@dataclass(frozen=True)
class InterpolationTask:
    """한 인접 쌍에 끼울 보간 작업(불변).
    index_a/index_b: 원본(또는 crops 상대) 프레임 인덱스.
    timesteps: 두 프레임 사이에 생성할 t 목록(오름차순, 0<t<1)."""
    index_a: int
    index_b: int
    timesteps: tuple[float, ...]

def build_interpolation_tasks(
    schedule: PlaybackSchedule,   # 또는 segments + n_frames
) -> tuple[InterpolationTask, ...]:
    """슬로우 구간(factor<1)을 인접 쌍별 중간 timestep 목록으로 전개(순수).

    factor=0.5 → 각 쌍에 timesteps=(0.5,)  (2x slow)
    factor=0.25 → timesteps=(0.25,0.5,0.75) (4x slow)
    factor>=1.0(등속·패스트) 구간 → 보간 작업 없음(복제/드롭 기존 경로 유지).
    """
```

- **factor → timestep 매핑**: 슬로우 배율의 역수가 "쌍당 출력 프레임 수"다. 0.5x는 2배 길이 → 쌍 사이 1장(t=0.5). 0.25x는 4배 → 3장(t=0.25/0.5/0.75). 정수배가 아닌 factor(예: 0.3x)는 **반올림 + 잔여를 복제로 보충**하거나(하이브리드) timestep을 비균등 분배한다 — PoC에서 품질 확인 후 정책 확정.
- **패스트·등속은 무영향**: factor≥1.0은 보간 없이 기존 `schedule_to_cfr_indices`(드롭)·복제 경로를 그대로 탄다(무회귀). 보간은 **슬로우 구간에만** 적용한다.

### 3-3. export 삽입 지점 — `VideoCaptureUseCase.export`

현재 흐름([app/video_capture.py] `export`):
```
build_output_indices → crop_frames → (upscale, upscaler 있을 때) → encode_frames
```

**보간 삽입 후보 = crop 직후 / upscale 직전·직후**. 권장 순서:
```
build_output_indices → crop_frames → [interpolate, interpolator 있을 때] → (upscale) → encode_frames
```

- **WHY crop 직후**: 보간은 크롭된 작은 프레임에서 추론하는 게 전프레임보다 싸다. 전 크롭이 동일 W×H([app] `_subject_fixed_size` 고정 크기)라 보간 결과도 균일 크기를 유지 → GIF/MP4 인코딩 정합이 안 깨진다(upscale과 동일 불변식).
- **upscale와의 순서**: 보간 → 업스케일 권장. 작은 해상도에서 보간(싸다) 후 확대하면 총 비용↓. 단 보간 산출 프레임 수가 늘어난 만큼 업스케일 호출 수도 증가하므로 비용 트레이드(§5 측정).
- **주입 방식**: `UpscaleBackend`와 **동일하게 메서드 옵션 주입**([ADR 0009] 결정2). `export(..., upscaler=None, interpolator: InterpolationBackend | None = None)`. None이면 기존 복제 경로 그대로(**무회귀**). router에서만 생성·주입, UI는 Protocol 미참조.

```python
crops = crop_frames(selected_frames, selected_boxes)
if interpolator is not None:
    crops, schedule = _interpolate_slow_segments(crops, config, interpolator)
    # 보간이 슬로우 구간을 실프레임으로 채웠으므로, encode 단계는 그 구간을
    # 복제하지 않도록 segments를 조정/소거한 config로 넘긴다(아래 CFR/VFR).
if upscaler is not None:
    crops = _upscale_crops(crops, upscaler)
encode_frames(crops, path, config_after_interp)
```

### 3-4. CFR(MP4)·VFR(GIF) 각각 어떻게

보간은 **실제 프레임 수를 늘리는** 방식이므로, [ADR 0013]의 두 경로를 다음과 같이 변형한다. **핵심 원칙: 보간이 이미 시간축을 늘렸으므로, encode 단계에서 같은 슬로우 구간을 또 늘리면 이중 슬로우가 된다 — 보간 적용 구간은 encode의 복제/duration 확장에서 제외한다.**

- **MP4 (CFR)**: 보간으로 슬로우 구간 프레임 수가 1/factor배로 늘어난 **확장된 crops 시퀀스**가 만들어진다. 이 시퀀스를 **균일 fps**(`config.fps`)로 그대로 인코딩하면 부드러운 슬로우가 된다. 즉 보간 경로에서는 `_resolve_mp4_frames`의 복제(`schedule_to_cfr_indices`)를 **슬로우 구간에 대해 건너뛰고**, 보간 산출 프레임을 그대로 쓴다. 패스트 구간(드롭)은 기존 경로 유지.
- **GIF (VFR)**: 두 가지 선택지.
  - (A) **보간 + 균일 duration**(권장): 슬로우 구간을 보간으로 실프레임 채우고 per-frame duration을 균일(1000/fps)로 둔다 → MP4와 동형, 가장 부드럽다. GIF 10ms 하한 가드([ADR 0013] `clamp_durations_for_gif`)는 균일 duration이라 트리거되지 않는다(슬로우는 느려지는 방향).
  - (B) 보간 + duration 미세조정: 정수배가 안 맞는 factor에서 보간 장수로 못 채운 잔여를 duration으로 보충. 복잡 → PoC 품질로 결정.
- **GIF 용량 주의**: 보간은 프레임 수를 늘리므로 GIF 파일 크기가 커진다([resources.md] §6 X 15MB 제한). `estimate_output_frame_count` 유사 헬퍼로 **보간 후 예상 프레임 수·용량을 사전 경고**한다(노트북 셀 9.5의 폭증 경고 패턴 재사용).

### 3-5. BACKGROUND 전제 계승

[ADR 0013] 결정3 구현 정합 노트의 **gap_policy=BACKGROUND 전제**(crops==span 전체)를 보간도 그대로 따른다. CUT/FREEZE에서 crops가 압축되면 보간 쌍 인덱스도 어긋날 수 있으므로, 보간 v1.1도 **BACKGROUND 전제**로 출시하고 2단계 인덱싱은 동일한 후속 과제로 묶는다.

### 3-6. ADR 후보 결정사항 (구현 착수 시 새 ADR로 승격)

1. **위치**: `InterpolationBackend` Protocol을 `core/interpolate/`에 신설(UpscaleBackend·VideoSegmentationBackend 선례). 구현은 `infra/rife_interpolation_backend.py`.
2. **인터페이스 단위**: timestep 1장 단위(`interpolate(a, b, t)`) vs 배치(`interpolate_batch`). → 1장 기본 + 배치 OCP 확장.
3. **삽입 지점**: `export`에서 crop 직후·upscale 직전, 메서드 옵션 주입(`interpolator=None` 무회귀).
4. **전개 로직 위치**: `build_interpolation_tasks` 순수 함수를 core에(90% 커버리지). factor→timestep 매핑·비정수 factor 정책.
5. **encode 이중 슬로우 방지**: 보간 적용 구간을 `schedule_to_cfr_indices` 복제·GIF duration 확장에서 제외하는 config 변환.
6. **1순위 백엔드 = RIFE(MIT)**, FILM(Apache)은 품질 미달 시 교체 후보(OCP).

---

## 4. 비용 추정 (T4, EdgeTAM 추적과 합산)

| 항목 | 추정치 | 근거 |
|---|---|---|
| RIFE 보간 1프레임 추론(크롭 720p급 T4) | **추정 ~10–50ms/frame** | RIFE "Real-Time" 표제·경량 모델. 크롭은 전프레임보다 작아 더 빠를 수 있음. **실측 필요** |
| RIFE VRAM | **추정 1–3GB** | 경량 모델(EdgeTAM·SAM2보다 작음). 실측 필요 |
| 보간 생성 프레임 수 | 슬로우 구간 길이 × (1/factor − 1) | 0.5x 30프레임 구간 → +30장, 0.25x → +90장 |
| 추적(EdgeTAM fp16) | 13.5 fps([ADR 0018] 실측) | 추적은 전 프레임 1회. 보간은 그 **이후** 슬로우 구간에만 추가 |
| 합산 총비용 | 추적 + (슬로우 구간 보간 장수 × 보간/frame) | 보간은 슬로우 구간에만 들어가므로 전체 영상 대비 부분 비용 |

**핵심**: 보간은 추적·크롭이 끝난 **슬로우 구간 프레임에만** 추가되는 비용이라, 전체 추적([ADR 0018] 13.5fps) 대비 증분이 제한적일 가능성이 높다(추정). 단 0.25x 긴 구간은 생성 장수가 급증하므로 **사전 폭증 경고**(§3-4)가 필수다. EdgeTAM·RIFE 모두 fp16 가능 → §5에서 fp16 보간도 측정.

---

## 5. Colab PoC 측정 절차 (app_verify 노트북 확장)

기존 `poc/colab/easy_capture_app_verify.ipynb`(T4)를 확장한다. 현재 셀 9.5(슬로우/트림/루프)·셀 10(GIF/MP4 export)이 이미 `core/timing` 함수를 직접 호출하므로, **그 사이에 보간 측정 셀을 추가**한다(코드 변경 전 노트북 셀만).

### 5-1. 측정 원칙 ([ac06] §5-1 계승)
1. **품질 육안 + 속도 동시**: stutter(복제) vs 보간 결과를 **나란히 GIF/MP4로 출력**해 육안 비교. fps는 보간 1프레임당 추론 시간으로.
2. **로드/추론 분리**: RIFE 모델 로드 시간과 순수 보간 시간을 따로 잰다.
3. **워밍업 1회 제외**: 첫 보간 호출(캐시/컴파일 워밍업) 버리고 2회차부터.
4. **동일 클립·동일 슬로우 구간 고정**: 셀 9.5의 `SEGMENTS=(SpeedSegment(start=30,end=60,factor=0.5),)`와 동일 구간으로 복제본 vs 보간본을 1:1 대조.

### 5-2. 셀 추가안
- **셀 9.6 (신규) — RIFE 설치·로드**: `git clone hzwer/Practical-RIFE` + 가중치 다운로드(Google Drive→Colab) + 모델 로드. transformers 통합이 없으므로 클론 경로. 로드 시간 기록.
- **셀 9.7 (신규) — 보간 PoC**:
  1. 셀 9에서 만든 crops(또는 frames)의 슬로우 구간 인접 쌍을 꺼낸다.
  2. `build_interpolation_tasks`(없으면 노트북 인라인) → timestep 목록.
  3. 각 쌍에 RIFE `interpolate(a, b, t)` → 중간 프레임. **2회차부터** 시간 측정 → 보간 fps·VRAM(`torch.cuda.max_memory_allocated`).
  4. **품질 비교 셀**: ① 복제 슬로우 GIF(기존 셀 10 경로) ② 보간 슬로우 GIF(보간 채운 crops를 균일 duration으로 encode) 둘 다 생성 → 나란히 표시해 육안 판정(끊김 해소·아티팩트·고스팅 확인).
  5. (가능 시) FILM 가중치(`jkawamoto/frame-interpolation-pytorch`)도 같은 쌍에 적용해 RIFE vs FILM 품질·속도 표 대비.
- **측정 표 산출물**: `(백엔드, dtype, 보간 fps, VRAM, 생성 프레임 수, 육안 품질 메모)` 누적. EdgeTAM 추적 fps(13.5)와 합산한 총 처리 시간 추정도 1줄.

### 5-3. 통과 기준 / 후속
- **통과**: 보간 GIF/MP4가 복제 대비 **육안으로 부드럽고**, 심한 고스팅/아티팩트가 없으며, 보간 비용이 추적 대비 수용 가능(예: 8초 클립 슬로우 구간 보간이 추가 수십 초 이내, 추정 기준).
- **반영**: 통과 시 `core/interpolate/` Protocol + `infra/rife_interpolation_backend.py` + `export` 옵션 주입을 **새 ADR로 승격**해 정식 구현 티켓화. **본 조사 단계에서는 코드 미반영.**
- 비정수 factor(0.3x 등) 품질·정책, GIF 용량 폭증 임계, fp16 보간 정확도는 PoC 결과로 확정.

---

## 6. 결론 (요약)

1. **1순위 = RIFE(Practical-RIFE v4.25/4.26, PyTorch)**. **MIT 라이선스(코드+가중치, 비상업 조항 없음)**라 프로젝트 상업 안전 경로([resources.md])에 부합하고, **torch만 의존**해 Colab이 가장 쉽다. v4.x는 **임의 timestep**을 지원해 [ADR 0013] factor 기반 슬로우와 직결된다.
2. **대안**: FILM(Apache 2.0, 큰 모션 품질 최상이나 무겁고 TF 원본 2025-10 아카이브 — PyTorch 포트 사용), EMA-VFI(Apache 2.0). 둘 다 라이선스 안전하므로 RIFE 품질 미달 시 **Protocol 교체 후보**.
3. **통합 설계**: `UpscaleBackend`·`VideoSegmentationBackend` 선례대로 `core/interpolate/InterpolationBackend` Protocol(core 순수) + `infra/rife_..._backend.py` 지연 로드. `export`에서 **crop 직후·메서드 옵션 주입**(`interpolator=None` 무회귀). 슬로우 구간을 **복제 대신 보간 실프레임으로 채우고**, encode 단계의 이중 슬로우(복제)를 그 구간에서 제외한다. CFR=균일 fps 인코딩, VFR=보간+균일 duration.
4. **비용**: 보간은 **슬로우 구간에만** 추가되는 부분 비용이라 EdgeTAM 추적(13.5fps) 대비 증분이 제한적일 가능성(추정). 단 0.25x 긴 구간은 생성 장수 급증 → **사전 폭증/용량 경고** 필수. T4 보간 fps·VRAM은 모두 **추정**, §5 노트북으로 확정.
5. **PoC**: app_verify 노트북에 셀 9.6(RIFE 설치)·9.7(보간 PoC: 복제 vs 보간 육안 비교 + 보간 fps/VRAM 측정 + RIFE vs FILM 대비)을 추가. 통과 시 새 ADR로 승격해 정식 구현.

---

## 7. 출처 (웹 검색, 2026-06 확인)

- Practical-RIFE (hzwer, MIT, v4.25/4.26): https://github.com/hzwer/Practical-RIFE — MIT 라이선스(코드+가중치 동일), `--multi` 배율, v4.26 2024-09-21, "content of these links is under the same MIT license".
- ECCV2022-RIFE (hzwer, 논문 구현): https://github.com/hzwer/ECCV2022-RIFE — MIT, 코드/모델 비상업 구분 조항 없음, "respect the commercial behavior of other developers".
- RIFE 논문 (arXiv 2011.06294): https://arxiv.org/abs/2011.06294 — temporal encoding 입력으로 **임의 timestep(0≤t≤1)** 중간 프레임 합성, optical flow 불요.
- rife-ncnn-vulkan-python (PyPI 1.2.x, MIT): https://pypi.org/project/rife-ncnn-vulkan-python/ — pip 설치 가능, Vulkan 추론, v4 모델만 custom framerate(임의 timestep).
- Google FILM (google-research, Apache 2.0, **2025-10-14 아카이브**): https://github.com/google-research/frame-interpolation — Apache 2.0, TF2 단일 네트워크, 큰 모션 특화.
- FILM PyTorch 포트 가중치 (HF, Apache 2.0): https://huggingface.co/jkawamoto/frame-interpolation-pytorch — `dajes/frame-interpolation-pytorch` 클론 + `film_net_fp16/fp32.pt`, pip 패키지 아님, CUDA/Colab 동작.
- EMA-VFI (MCG-NJU, Apache 2.0, CVPR 2023): https://github.com/MCG-NJU/EMA-VFI — Apache 2.0, inter-frame attention 효율 보간.
- AMT (CVPR 2023, 가중치 HF): https://arxiv.org/pdf/2304.09790 — All-Pairs Multi-Field Transforms, Vimeo90K에서 IFRNet-B +0.17dB, 가중치 HF 로드.

> ⚠ T4에서의 RIFE·FILM 보간 fps·VRAM 실측 공개치는 본 조사 시점에 없어 **모두 "추정"**이다. 확정은 §5 노트북 측정으로 한다.
