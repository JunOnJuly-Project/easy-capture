# ADR 0017 — SAM2 비디오 추론 fp16 mixed precision 채택

- 상태: 채택 (Colab T4 실측 기반)
- 날짜: 2026-06-04
- 연계: [ADR 0010](0010-video-segmentation-backend.md) · [ADR 0007](0007-cpu-dev-strategy.md) · [docs/plans/ac06-fps-improvement.md](../plans/ac06-fps-improvement.md)

## 맥락

AC-06(GPU 처리 속도)이 `hiera-small` + 컷별 선택에서 **2.0 fps**로 목표 10 fps에 크게 미달했다([poc/REPORT.md](../../poc/REPORT.md)). [AC-06 타당성 조사](../plans/ac06-fps-improvement.md)에서 병목을 진단한 결과, 백엔드가 `dtype=torch.float32`로 **고정**되어 있어(`infra/sam2_video_backend.py`) mixed precision 이득을 전혀 사용하지 않고 있었다. SAM2 공식은 `torch.autocast`를 통한 혼합정밀 추론을 권장한다.

조사 1순위 후보(fp16/bf16)를 적용·측정하기 위해 백엔드에 `dtype` 주입을 추가하고, Colab T4에서 게이트 군무 클립(300프레임, 컷 3개→4샷)으로 실측했다.

## 측정 결과 (Colab T4, hiera-small, 컷별 선택, 동일 클립)

| dtype | fps | fp32 대비 | AC-01 | needs_correction |
|---|---|---|---|---|
| float32 | 1.92 | 1.00× | 100% | 0 |
| **float16** | **5.14** | **2.67×** | **100%** | **0** |
| bfloat16 | 1.84 | 0.95× | 100% | 0 |

- **fp16**: 정확도(AC-01 추적 유지율·needs_correction)를 **완전히 유지**하면서 **2.67배** 가속.
- **bf16**: T4는 bf16 하드웨어 가속이 없어(Ampere/A100부터 지원) 오히려 소폭 느림(0.95×). 정확도는 안전하나 T4에서 속도 이득 없음 — 조사 예측과 일치.

## 결정

**cuda 디바이스의 기본 추론 정밀도를 fp16으로 채택한다.**

- `infra/device.py`에 `SAM2_DTYPE_BY_DEVICE = {"cuda": "float16", "cpu": "float32"}` + `select_sam2_dtype(device)` 추가.
- `app/router.py`가 비디오 백엔드 생성 시 `dtype=select_sam2_dtype(device)`를 주입한다.
- `Sam2VideoBackend`는 dtype 문자열을 받아 `init_session(dtype=...)`에 적용하고, `propagate`를 `_autocast_context`로 감싼다(fp32·cpu는 `nullcontext` no-op, fp16/bf16+cuda면 `torch.autocast`). torch 의존은 infra에만 둔다(router/UI는 문자열만 전달 — 경계 유지).
- **cpu는 fp32 유지**: cpu autocast는 fp16을 지원하지 않아 무의미하다.
- **bf16은 기본 채택하지 않는다**: T4 무가속. 다만 백엔드 `dtype="bfloat16"` 인자로 주입 가능하게 남겨, bf16 가속 GPU(A100 등)에서 선택할 수 있다.

## 대안

**(a) bf16 기본 채택**
- 거부: T4에서 가속이 없어 본 프로젝트 주 검증 환경에서 이득이 없다(측정 0.95×). fp16이 동일 정확도에 2.67×로 우월.

**(b) fp32 유지 + 모델 경량화(EdgeTAM)로만 가속**
- 거부(보류): EdgeTAM은 잠재 이득이 크나 T4 fps·box/negative 지원·군무 정확도가 미검증이다(조사 §2-B). fp16은 *공짜에 가까운* 즉시 회수이므로 먼저 채택하고, EdgeTAM은 10 fps 도달을 위한 **후속**으로 둔다.

## 결과

### 긍정적 영향
- **AC-06 2.67× 개선**(1.92→5.14 fps), 정확도 무손실. 코드 변경 최소(백엔드 dtype 분기).
- **추상화 무변경**: `VideoSegmentationBackend` Protocol·core·app 시그니처 그대로. dtype은 infra 내부 디테일.
- **무회귀**: 기본 인자 fp32, cpu 폴백 fp32. 기존 호출/테스트가 깨지지 않는다.

### 부정적 영향 / 트레이드오프
- **여전히 10 fps 미달**(5.14): 목표 도달엔 EdgeTAM(B) 등 추가 작업 필요. 본 ADR은 1차 회수.
- **fp16 수치 안정성**: 일반적으로 fp16은 overflow/underflow 위험이 있으나, 본 측정에서 SAM2 추론은 T4 fp16에서 AC-01 100%를 유지했다. 다른 GPU/모델 변경 시 재측정 권장(셀 8.5 재사용).

## 후속
- EdgeTAM(Apache 2.0, transformers `EdgeTamVideoModel`) PoC 측정 — fp16과 결합 시 10 fps 도달 가능성(조사 §2-B, §5).
- `device.py` 모델 카탈로그 정합화 검토(cuda 기본 `base-plus` vs 게이트 `small`) — 별도 결정.
