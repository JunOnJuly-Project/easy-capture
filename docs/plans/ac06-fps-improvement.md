# AC-06 GPU 처리 속도 개선 — 타당성 조사

> 작성일: 2026-06-04 · 상태: 조사(코드 미수정) · 관련: [poc/REPORT.md](../../poc/REPORT.md) · [ADR 0007](../adr/0007-cpu-dev-strategy.md) · [ADR 0010](../adr/0010-video-segmentation-backend.md) · [resources.md](../resources.md) · [RFP NFR-03/AC-06](../RFP.md)
>
> 목적: 백로그 항목 **"AC-06 GPU 처리 속도 개선"**(현재 2.0fps → 목표 10fps)에 대해 fp16/half · 경량 백엔드(EdgeTAM) · 프레임 서브샘플링 등 후보를 평가하고 우선순위·검증 방법을 제시한다. **본 문서는 조사 결과이며 코드 변경은 포함하지 않는다.**

---

## 1. 현황 요약 — 2.0fps의 병목은 어디인가

### 측정 사실 (poc/REPORT.md)
- **CPU**: ≈0.10 fps(프레임당 ~10s). 비실용.
- **GPU(Colab T4)**: 단일샷 200f ≈ **2.3 fps**, 멀티샷 군무 300f(컷6→4샷, `hiera-small` + 컷별 선택) **AC-01 100% · needs_correction 0 · AC-06 2.0 fps**.
- 목표: 계획서/RFP NFR-03 ≥10fps(이미 실측 기반 2fps 하한으로 재조정, 10fps는 v1.1 이관 상태).

### 병목 — 코드 근거
1. **SAM2 내부 1024×1024 인코딩**: 입력 해상도와 무관하게 프레임당 비용이 고정적으로 크다(PoC 핵심 발견 #2, resources.md §5). → 입력 다운스케일이 큰 효과를 못 내는 이유.
2. **메모리 어텐션(memory attention)이 latency 병목**: EdgeTAM 논문이 명시 — 기존 SAM 경량화는 image encoder만 압축했으나, SAM2의 진짜 병목은 **새로 도입된 memory attention 블록**이다([arXiv 2501.07256](https://arxiv.org/abs/2501.07256)). 프레임마다 과거 메모리 뱅크와 cross-attention하므로 프레임 수에 비례해 무거워진다. → 단순 인코더 교체(MobileSAM 등)로는 비디오 속도가 안 풀리는 근본 이유.
3. **dtype=float32 고정**: 현재 백엔드는 fp32로 추론한다. `src/easy_capture/infra/sam2_video_backend.py:74`
   ```python
   session = self._processor.init_video_session(
       video=frames,
       inference_device=self.device,
       dtype=torch.float32,      # ← fp16/bf16 미적용
   )
   ```
   `propagate`도 `torch.inference_mode()`만 쓰고 `autocast`가 없다(`sam2_video_backend.py:161`). PoC 노트북·`h1_track.py:54`도 동일하게 `dtype=torch.float32`. → **fp16/bf16 미적용 상태이며, 이것이 가장 먼저 회수할 수 있는 이득.**
4. **모델 로드 시간이 fps에 섞임(추정)**: PoC 200f/87.6s는 "모델 로드 포함 추정"이라 명시(REPORT.md). 즉 순수 전파 fps는 2.3보다 약간 높을 수 있다. → **측정 시 로드/전파를 분리**해야 진짜 병목이 보인다(아래 §5).

### 디바이스 기본값 불일치(부수 발견, 개선과 직접 연계)
- `infra/device.py:11`의 `SAM2_REPO_BY_DEVICE["cuda"] = "facebook/sam2.1-hiera-base-plus"`. 그러나 GPU 게이트는 `hiera-small`로 통과했고(HANDOFF §3) 노트북 기본도 small로 변경됨. **base-plus는 small보다 무겁다** → 데스크톱 기본이 base-plus면 게이트(small)보다 느릴 수 있다. fps 개선 작업 시 이 카탈로그도 함께 정합화 검토 필요(코드 변경 아님, 본 조사의 지적 사항).

---

## 2. 후보별 분석

### 방법별 비교 표

| 방법 | 기대효과 | 난이도 | 리스크 | 라이선스 | 추상화 적합성(ADR 0010) |
|---|---|---|---|---|---|
| **A. fp16/bf16 (autocast/dtype)** | T4에서 **1.3~2× 추정**(전파 대상; A100 bf16 small ≈47fps 보고치는 환경 상한) | **낮음** — `init_video_session(dtype=)` + `propagate` autocast 2~3줄 | bf16은 안정, fp16은 수치 불안정 가능(마스크 경계 미세 변화). **정확도(AC-01) 재측정 필수** | 영향 없음(가중치/코드 SAM2 Apache 2.0 그대로) | **완벽** — 백엔드 내부 dtype만 변경, Protocol·core 무변경 |
| **B. EdgeTAM 백엔드** | iPhone 16fps(22× vs SAM2 0.7fps, on-device). **T4 fps는 미측정/추정** — memory attention 자체를 경량화하므로 SAM2-small 대비 유의미한 GPU 이득 기대(추정) | **중간** — 신규 infra 백엔드 1개. API 거의 동일 | ① **transformers 5.10+ 필요**(현재 핀 `>=4.55`, 실제 SAM2 video는 5.9). ② **box prompt/negative 지원 여부 검증 필요**(군무 게이트가 box+negative 전제). ③ 군무 밀착 분리 정확도 미검증(small이 필요했던 이력) | **Apache 2.0**(코드+체크포인트) — 상업 안전 | **우수** — `VideoSegmentationBackend` Protocol 그대로 구현, opaque session 동일. router 교체만 |
| **C. 프레임 서브샘플링(N프레임 1회 추론+보간)** | N=2면 추론량 절반 → 유효 fps ~2× | **중간** — 추론은 SAM2/EdgeTAM에 위임하되 보간/마스크 채우기 로직 필요 | **메모리 뱅크 일관성 깨짐** — SAM2 video는 연속 프레임 전파가 전제라 프레임을 건너뛰면 추적 품질(AC-01) 저하 위험 큼. 빠른 군무에서 마스크 떨림/소실 | 영향 없음 | **주의** — 어디서 서브샘플할지가 관건(§아래) |
| **D-1. torch.compile** | 추정 1.1~1.3×(반복 호출 시) | 중간 | 컴파일 워밍업 지연(첫 호출 수십 초), dynamic shape 재컴파일, Colab/Windows 호환 편차 | 영향 없음 | 양호(백엔드 내부) |
| **D-2. 입력 해상도 다운스케일** | **낮음** — 내부 1024 인코딩이라 입력 축소 효과 작음(PoC #2) | 낮음 | 효과 대비 이득 적음 | 영향 없음 | 양호 |
| **D-3. video frames CPU offload** | 속도 이득 아님(메모리 절약용) | 낮음 | OOM 회피용. 속도는 오히려 약간 손해 가능 | 영향 없음 | 양호 |

> ⚠ "추정" 표기는 T4에서의 실측이 없는 값이다. 보고된 16fps는 **iPhone 15 Pro Max on-device**, 47fps는 **A100 bf16**으로, 본 프로젝트 타깃(T4)과 다르므로 직접 대입 불가.

### 각 방법 상세

#### A. fp16 / bf16 (mixed precision) — **최우선 후보**
- **근거**: SAM2 video predictor는 `torch.autocast("cuda", dtype=torch.bfloat16)` 혼합정밀 추론이 공식 권장이다([SAM2 README](https://github.com/facebookresearch/sam2/blob/main/README.md), [DeepWiki SAM2 video](https://deepwiki.com/facebookresearch/sam2/6-using-sam2-for-video-segmentation)). transformers `init_video_session`은 `dtype` 인자를 받으며(기본 float32), EdgeTAM/SAM2 video 세션 모두 dtype을 지원한다(HF 문서 §EdgeTamVideoInferenceSession `dtype` 파라미터).
- **bf16 권장**(fp16보다): bf16은 fp32와 같은 지수 범위라 overflow/underflow에 강하다([RunPod mixed precision](https://www.runpod.io/articles/guides/fp16-bf16-fp8-mixed-precision-speed-up-my-model-training)). **단 T4는 bf16 하드웨어 가속이 없다**(Ampere/A100부터) — T4에서는 fp16이 실제 가속, bf16은 정확도만 안전하고 속도 이득은 작을 수 있다(추정). → **T4=fp16, A100/최신 GPU=bf16** 분기 검토.
- **구현 위치**: `Sam2VideoBackend.init_session`의 `dtype` + `propagate`의 `torch.inference_mode()`를 `torch.autocast(...)`로 감싸기. Protocol·core 무변경.
- **검증 필수**: 멀티샷 군무 게이트(AC-01 100%·needs_correction 0)가 fp16에서 유지되는지 재측정. 마스크 경계가 미세하게 흔들리면 crop 떨림으로 이어질 수 있다.

#### B. EdgeTAM 경량 백엔드 — **차순위(중기)**
- **현황(2025~2026)**: EdgeTAM(Meta, CVPR 2025 highlight)은 **2025-09-29 HuggingFace transformers에 정식 병합**됨. 클래스 `EdgeTamVideoModel`, **프로세서는 `Sam2VideoProcessor`를 재사용**, 체크포인트 **`yonigozlan/EdgeTAM-hf`**(public·월1만+ 다운로드, 실측 사용). ⚠ 공식 문서가 예제로 쓰는 `yonigozlan/edgetam-video-1`은 **gated/미존재로 401**(2026-06 확인) — `EdgeTAM-hf`를 쓴다. API가 SAM2 video와 **동일**: `init_video_session` → `add_inputs_to_inference_session` → `propagate_in_video_iterator` → `post_process_masks`(HF [edgetam_video 문서](https://huggingface.co/docs/transformers/en/model_doc/edgetam_video), transformers v5.10.1 소스).
- **속도**: iPhone 15 Pro Max 16fps(SAM2 대비 22×, on-device). **memory attention을 2D Spatial Perceiver로 경량화**해 SAM2의 진짜 병목을 친다 → GPU에서도 이득 기대(단 **T4 fps 실측은 공개치 없음, 추정**).
- **정확도**: DAVIS 87.7 / MOSE 70.0 / SA-V val 72.3 / test 71.7 J&F로 SAM2에 "comparable". 다만 본 프로젝트는 **군무 밀착 분리에 hiera-small이 필요**했던 이력(ADR 0015 R1) → EdgeTAM(경량)이 그 난도를 통과할지는 **별도 게이트 측정 필수**.
- **추상화 적합성**: `VideoSegmentationBackend` Protocol을 그대로 구현하는 신규 infra(`infra/edgetam_video_backend.py`) 1개 추가 + router 분기. ADR 0010 opaque session·ISP 구조가 정확히 이 교체를 위해 설계됨(ADR 0010 "새 경량 이미지 백엔드 추가 시 구현 의무 없음" 정신의 비디오판). **`add_box`/negative 지원 여부 확인이 선결**(현 게이트가 box+negative 전제).
- **라이선스**: Apache 2.0(코드+가중치) — 상업 안전 경로에 부합.
- **제약**: transformers 버전. 현재 `pyproject.toml`/`requirements.txt`는 `transformers>=4.55`지만 실제로 SAM2 video는 5.9, **EdgeTAM은 5.10+** 필요. 도입 시 핀 상향(>=5.10) 검토.

#### C. 프레임 서브샘플링 — **신중(품질 리스크)**
- **아이디어**: N프레임마다 1회만 SAM2 전파, 사이 프레임은 마스크/crop box를 보간.
- **문제**: SAM2 video는 **연속 프레임 메모리 전파**가 핵심이라 프레임을 건너뛰면 memory bank 일관성이 약해진다. 정확도(AC-01) 저하 위험이 fp16보다 크다.
- **그나마 안전한 변형**: SAM2 전파는 전 프레임 수행하되, **crop box smoothing/보간은 이미 존재**(`compute_boxes`의 `smooth_window`, crop-tuning 슬라이스). 즉 "추론 서브샘플"이 아니라 "출력 프레임 데시메이션"(GIF fps 낮추기)은 이미 부분 달성. 진짜 추론 절감은 권장 보류.
- **대안 위치**: 만약 시도한다면 app 레이어(`VideoCaptureUseCase`)가 아니라, 백엔드가 N프레임 전파 후 사이를 채우는 방식 — 단 Protocol 계약(프레임별 마스크 리스트 반환)은 유지해야 함.

#### D. 기타
- **torch.compile**: `_ensure_loaded` 후 `self._model = torch.compile(model)` 가능. 워밍업·재컴파일·플랫폼 호환 리스크. fp16 검증 후 추가 레버로.
- **입력 다운스케일**: 내부 1024 고정이라 이득 작음(PoC #2). 비권장.
- **CPU offload**(`video_storage_device="cpu"`): 속도가 아닌 OOM 회피용. 긴 영상에서 메모리 부족 시 보조.
- **batched propagate / multi-object**: 동일 샷 내 다중 객체를 한 세션에서 추적하면 객체당 재호출을 줄일 수 있으나, 현재는 obj_id=1 단일 추적이라 fps 직접 이득은 제한적.

---

## 3. 권장 우선순위

| 순위 | 항목 | 근거 |
|---|---|---|
| **1** | **A. fp16/bf16 적용 + 정확도 재게이트** | 난이도 최저(2~3줄), 라이선스/추상화 영향 0, 즉시 측정 가능. **현재 fp32 고정이 명백한 미수확 이득**. T4=fp16/최신=bf16 분기. AC-01 게이트 통과만 확인하면 바로 회수. |
| **2** | **측정 인프라 정비**(로드/전파 분리, §5) | 2.0fps에 모델 로드가 섞여 있어(추정) 진짜 전파 fps와 개선폭을 정확히 못 본다. A를 측정하려면 선행 필요. |
| **3** | **device.py 카탈로그 정합화 검토** | cuda 기본이 base-plus인데 게이트는 small. 기본을 small로 맞추면 무변경으로도 게이트 대비 정합(코드 변경은 별도 결정). |
| **4** | **B. EdgeTAM 백엔드 PoC 측정** | 잠재 이득 최대(memory attention 병목 직격)이나 ① T4 fps 미검증 ② box/negative 지원 ③ 군무 정확도 미검증의 3대 불확실성. **노트북에서 측정 먼저**, 통과 시 정식 백엔드화. transformers 5.10+ 핀 상향 동반. |
| **5** | **D-1 torch.compile**(선택) | A/B로 목표 근접 시 추가 레버. 플랫폼 리스크라 마지막. |
| **보류** | **C 프레임 서브샘플링** | 정확도 리스크가 fps 이득 대비 큼. A·B로 부족할 때만 재검토. |

**핵심 논리**: fp16(A)은 *공짜에 가까운* 이득이므로 무조건 1순위. 그것으로 목표(10fps)에 못 미치면 — 거의 확실히 그렇다(2.0→~3fps 추정) — **근본 병목인 memory attention을 치는 EdgeTAM(B)**이 유일한 큰 도약 후보다. 입력 다운스케일·서브샘플은 SAM2 구조상(1024 인코딩·연속 전파) 효율/품질 트레이드가 나빠 후순위.

---

## 4. 출처 (웹 검색)

- EdgeTAM 논문(arXiv 2501.07256): https://arxiv.org/abs/2501.07256 — 16fps@iPhone15ProMax, 22× vs SAM2(0.7fps), memory attention이 병목, 2D Spatial Perceiver, Apache 2.0.
- EdgeTAM GitHub(facebookresearch): https://github.com/facebookresearch/EdgeTAM — Apache 2.0, SAM2 video API 호환(init_state/add_new_points_or_box/propagate_in_video).
- EdgeTAM transformers 문서(HF): https://huggingface.co/docs/transformers/en/model_doc/edgetam_video — `EdgeTamVideoModel` + `Sam2VideoProcessor`, 체크포인트 `yonigozlan/edgetam-video-1`·`yonigozlan/EdgeTAM-hf`, `init_video_session`/`add_inputs_to_inference_session`/`propagate_in_video_iterator` API, dtype 인자 지원. transformers v5.10.1 기준. HF 통합 병합일 **2025-09-29**(추정 — 검색 결과 요약 기준).
- SAM2 README(facebookresearch/sam2): https://github.com/facebookresearch/sam2/blob/main/README.md — `torch.autocast("cuda", dtype=torch.bfloat16)` 권장.
- SAM2 video 사용 가이드(DeepWiki): https://deepwiki.com/facebookresearch/sam2/6-using-sam2-for-video-segmentation — bf16 autocast + CPU offload 옵션.
- mixed precision(fp16/bf16) 비교(RunPod): https://www.runpod.io/articles/guides/fp16-bf16-fp8-mixed-precision-speed-up-my-model-training — bf16 수치 안정성, A100 bf16 small ≈47fps 참고치.

> ⚠ T4에서의 EdgeTAM·fp16 실측 fps는 본 조사 시점 공개 벤치마크가 없어 **모두 "추정"**이다. 확정은 §5 노트북 측정으로.

---

## 5. 검증 방법 (Colab 노트북)

기존 검증 경로(`poc/colab/easy_capture_app_verify.ipynb`, Colab T4)를 확장한다. 코드 변경 전 **측정 셀만 추가/수정**해 후보를 A/B/C로 분리 측정.

### 5-1. 측정 원칙
1. **로드/전파 분리**: 모델 로드 시간과 순수 propagate 시간을 따로 타이머로 잰다(현재 2.0fps는 로드 포함 추정). fps = 프레임수 / 순수 propagate 초.
2. **워밍업 1회 제외**: 첫 호출(컴파일/캐시 워밍업)을 버리고 2회차부터 측정(특히 torch.compile).
3. **동일 클립 고정**: 멀티샷 군무 300프레임(컷6→4샷) — 게이트와 동일 클립으로 AC-01·needs_correction·fps를 한 번에.
4. **정확도 동반 측정**: fps만 보지 말고 **AC-01 추적 유지율·needs_correction**을 매 변형마다 같이 출력(속도-정확도 트레이드 가시화).

### 5-2. 후보별 측정 절차
- **A. fp16/bf16**:
  1. 베이스라인: 현재대로 `dtype=torch.float32` → (fps, AC-01, needs_correction) 기록.
  2. `init_video_session(dtype=torch.float16)` + `propagate`를 `with torch.autocast("cuda", dtype=torch.float16):`로 감싼 사본 셀.
  3. (가능 시) bf16도 측정. T4면 bf16 가속 없음 확인.
  4. **통과 기준**: AC-01 100% · needs_correction 0 유지 + fps 향상.
- **B. EdgeTAM**:
  1. `transformers>=5.10` 설치 셀.
  2. `EdgeTamVideoModel.from_pretrained("yonigozlan/EdgeTAM-hf")` + `Sam2VideoProcessor.from_pretrained(...)`.
  3. **box prompt(`input_boxes`)·negative point 지원 여부**를 단건으로 먼저 확인(군무 게이트 전제). 미지원이면 point-only로 한정해 정확도 별도 평가.
  4. 동일 군무 클립으로 (fps, AC-01, needs_correction) 측정 → SAM2-small과 표로 대비.
  5. fp16까지 결합 측정(A×B).
- **C. 서브샘플(시도 시)**: N=2 전파 + box 보간 → AC-01 저하폭 측정. 저하가 크면 기각.
- **D-1. torch.compile**: A 통과본에 `torch.compile` 적용, 워밍업 후 2회차 fps.

### 5-3. 산출물
- 측정 표: `(변형, fps_propagate, fps_total, AC-01, needs_correction, VRAM)` 행으로 누적.
- 통과한 변형을 `infra/sam2_video_backend.py`(A/D) 또는 신규 `infra/edgetam_video_backend.py`(B)로 반영하는 후속 작업 티켓화. **본 조사 단계에서는 코드 미반영.**

---

## 6. 결론 (요약)

1. **현재 2.0fps의 병목은 SAM2의 1024 고정 인코딩 + memory attention**이며, 백엔드가 **fp32 고정**(`sam2_video_backend.py:74`)이라 mixed precision 이득을 전혀 안 쓰고 있다.
2. **1순위 = fp16/bf16**: 난이도·라이선스·추상화 영향이 사실상 0이고 즉시 측정 가능. T4=fp16, 최신 GPU=bf16. AC-01 게이트 재확인만 하면 회수.
3. **큰 도약은 EdgeTAM(B)**: 이미 transformers 5.10+에 `EdgeTamVideoModel`로 병합(Apache 2.0, SAM2 video와 동일 API, 프로세서 재사용)되어 `VideoSegmentationBackend` 추상화에 깔끔히 끼울 수 있다. 단 **T4 fps·box/negative·군무 정확도는 미검증("추정")** → 노트북 측정이 선결.
4. **프레임 서브샘플(C)은 보류**: SAM2 연속 전파 구조상 정확도 리스크가 fps 이득보다 크다. 입력 다운스케일도 1024 인코딩 때문에 이득이 작다.
5. **부수 정합화**: `device.py`의 cuda 기본이 `base-plus`인데 게이트는 `small`. fps 작업 시 기본 모델 카탈로그도 함께 검토.

---

## 7. 구현 진행 (실측 준비)

> 2026-06-04 — 1순위(A. fp16/bf16) 실측을 위한 인프라 추가. **측정 자체는 Colab T4에서 수행(대기).**

### 완료
- **백엔드 dtype 주입**(`infra/sam2_video_backend.py`): 생성자 `dtype="float32"` 인자 추가(기본 무회귀). `init_session(dtype=_resolve_torch_dtype(...))` + `propagate`를 `_autocast_context`로 감쌌다. fp32·cpu는 `nullcontext`(no-op), fp16/bf16+cuda면 `torch.autocast`. `_validate_dtype`로 생성 시 조기 검증.
- **단위 테스트**(`tests/test_sam2_video_backend_dtype.py`, 16개): dtype 검증·매핑·autocast 분기(CPU torch로 검증). 실 추론은 노트북 후행.
- **측정 셀**(`poc/colab/easy_capture_app_verify.ipynb` 셀 8.5): dtype별 백엔드 재생성 → 2회차(모델 로드 제외) propagate fps + AC-01·needs_correction 동반 측정 + fp32 대비 배속·정확도 유지 판정. T4 bf16 미지원 자동 감지.
- `router`는 **무변경**(기본 fp32). 측정으로 fp16 정확도(AC-01 100%·needs_correction 0)가 확인되면 그때 기본 상향.

### 측정 결과 (Colab T4, hiera-small, 컷별 선택, 군무 300f/컷3→4샷) — 2026-06-04

| dtype | fps | fp32 대비 | AC-01 | needs_correction |
|---|---|---|---|---|
| float32 | 1.92 | 1.00× | 100% | 0 |
| **float16** | **5.14** | **2.67×** | **100%** | **0** |
| bfloat16 | 1.84 | 0.95× | 100% | 0 |

- **fp16 채택**: 정확도 무손실(AC-01 100%·needs_correction 0) + **2.67× 가속**. → [ADR 0017](../adr/0017-fp16-mixed-precision.md).
- bf16은 T4 무가속(0.95×)이라 미채택(조사 예측 일치). bf16 가속 GPU에선 백엔드 인자로 주입 가능.

### 적용 (완료)
- `device.py`: `SAM2_DTYPE_BY_DEVICE`(cuda=fp16/cpu=fp32) + `select_sam2_dtype`. `router`가 백엔드 생성 시 주입. 3 단위테스트.
- **결과**: 데스크톱 비디오 추적 기본 cuda 경로가 fp16(2.67×)로 동작. cpu 무회귀(fp32).

### 적용 (완료) — device.py 카탈로그 정합화
- ✅ cuda 기본 모델을 `base-plus`→`hiera-small`로 변경(게이트·fp16 측정이 small로 통과). 데스크톱 cuda 경로가 검증된 small + fp16으로 동작.

### EdgeTAM(B) PoC — 측정 인프라 완료, Colab 측정 대기

fp16으로도 5.14fps라 목표 10fps 미달 → memory attention 경량화(EdgeTAM)로 추가 도약 측정(조사 §2-B). **인프라 준비 완료:**
- **`infra/edgetam_video_backend.py`**: `EdgetamVideoBackend(Sam2VideoBackend)` — `_ensure_loaded`만 `EdgeTamVideoModel`로 override(EdgeTAM이 `Sam2VideoProcessor` 재사용·SAM2 video API 호환). dtype/autocast·init_session·add_box·propagate는 부모 계승 → fp16도 그대로 적용. 5 구조테스트.
- **노트북**: 셀 1 `transformers>=5.10`(EdgeTAM 필요), 셀 8.6 EdgeTAM vs SAM2-small fp16 비교(fps·AC-01·needs_correction + box/negative 지원 확인 + 실패 시 원인 안내).
- **측정 검증 포인트**(Colab): ① EdgeTAM이 box+negative 지원하는가(track 성공 여부) ② fps가 SAM2-small fp16(≈5fps) 대비 향상되어 10fps 근접하는가 ③ 군무 AC-01 100%·needs_correction 0 유지하는가.
- **다음 결정**: 3개 모두 통과 → `router`/`device.py`에 EdgeTAM 채택(새 ADR). 정확도 미달 → SAM2 유지 + 다른 레버(torch.compile 등).
