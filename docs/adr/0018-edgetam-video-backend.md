# ADR 0018 — 비디오 추적 기본 백엔드 EdgeTAM 채택

- 상태: 채택 (Colab T4 실측 기반)
- 날짜: 2026-06-04
- 연계: [ADR 0010](0010-video-segmentation-backend.md) · [ADR 0017](0017-fp16-mixed-precision.md) · [docs/plans/ac06-fps-improvement.md](../plans/ac06-fps-improvement.md)

## 맥락

AC-06(GPU 처리 속도) 목표는 10 fps였다. [ADR 0017](0017-fp16-mixed-precision.md)에서 fp16을 채택해 SAM2 `hiera-small`을 1.9→**5.1 fps**(2.67×)로 끌어올렸으나 여전히 목표 절반이었다. [AC-06 조사](../plans/ac06-fps-improvement.md)는 SAM2의 진짜 병목이 **memory attention**이며, 이를 2D Spatial Perceiver로 경량화한 **EdgeTAM**(Meta, CVPR 2025)이 유일한 큰 도약 후보라고 분석했다.

EdgeTAM은 transformers에 `EdgeTamVideoModel`로 통합되었고 **`Sam2VideoProcessor`를 재사용**해 SAM2 video와 API가 동일하다. 따라서 `EdgetamVideoBackend(Sam2VideoBackend)` 상속본으로 동일 추적 경로(init_session·add_box·negative·propagate·fp16/autocast)를 그대로 태워 측정했다.

## 측정 결과 (Colab T4, 군무 300f/컷3→4샷, 컷별 선택 box+negative, fp16)

| 백엔드 | fps | 배속 | AC-01 | needs_correction |
|---|---|---|---|---|
| SAM2 hiera-small (fp32) | 1.9 | 1.0× | 100% | 0 |
| SAM2 hiera-small (fp16) | 5.1 | 2.7× | 100% | 0 |
| **EdgeTAM (fp16)** | **13.5** | **~7.1×** | **100%** | **0** |

세 통과 기준 전부 충족:
1. **box + negative 지원**: 컷별 선택(box 프롬프트 + negative point) 경로로 `track`이 성공했고 needs_correction 0 — EdgeTAM이 `input_boxes`·negative label을 처리함을 실증.
2. **10 fps 달성**: 13.5 fps(목표 초과).
3. **정확도 무손실**: 군무 AC-01 100%·needs_correction 0 유지(군무 밀착 분리에 `hiera-small`이 필요했던 우려가 EdgeTAM에서도 해소).

## 결정

**비디오 추적 기본 백엔드를 EdgeTAM로 채택한다.**

- `infra/edgetam_video_backend.py`의 `EdgetamVideoBackend`(= `Sam2VideoBackend` 상속, `_ensure_loaded`만 `EdgeTamVideoModel` 교체)를 비디오 기본으로 사용한다.
- `infra/device.py`에 `VIDEO_TRACKING_REPO = "yonigozlan/EdgeTAM-hf"`(public·transformers 호환·Apache 2.0) 추가. `app/router._build_video_usecase_factory`가 이 백엔드를 조립한다.
- dtype은 [ADR 0017](0017-fp16-mixed-precision.md)의 `select_sam2_dtype`(cuda=fp16)을 재사용한다 — 정밀도 정책은 모델과 무관.
- 의존성 `transformers>=5.10`으로 상향(EdgeTAM 필요). pyproject·requirements 반영.
- **SAM2 비디오 백엔드(`Sam2VideoBackend`)는 제거하지 않는다** — 부모 클래스이자 폴백/비교 옵션으로 유지(노트북 셀 8.5/8.6이 두 백엔드를 모두 측정).

### 체크포인트 주의

공식 transformers 문서가 예제로 안내하는 `yonigozlan/edgetam-video-1`은 **gated/미존재로 401**(2026-06 확인). 실제 사용은 public 체크포인트 **`yonigozlan/EdgeTAM-hf`**(월 1만+ 다운로드, 공식 예제에서도 사용).

## 대안

**(a) SAM2 + fp16 유지(EdgeTAM 미도입)**
- 거부: 5.1 fps로 목표 10 fps 미달. 추가 레버(torch.compile ~1.2×, 서브샘플 정확도 리스크)로는 도달이 불확실. EdgeTAM이 정확도 무손실로 목표를 단번에 초과.

**(b) SAM2를 완전히 EdgeTAM로 대체(코드 삭제)**
- 거부: SAM2 video는 EdgeTAM의 부모 구현이자 검증된 기준선이다. 비교·폴백 가치가 있어 유지한다(상속 구조라 유지 비용도 낮다).

## 결과

### 긍정적 영향
- **AC-06 목표 달성**: 1.9→13.5 fps(~7×), 정확도 무손실. 7분 영상 기준 체감 대기 시간 대폭 단축.
- **추상화 정합**: [ADR 0010](0010-video-segmentation-backend.md) `VideoSegmentationBackend` 추상화 덕분에 백엔드 교체가 router 1줄 + infra 1파일로 끝남(core·app·UI 무변경). 상속으로 코드 중복도 없음.
- **라이선스 안전**: EdgeTAM Apache 2.0(코드+가중치).

### 부정적 영향 / 트레이드오프
- **transformers 5.10+ 핀**: 의존성 하한이 올라간다(EdgeTAM 통합 버전). SAM2도 호환되어 회귀는 없으나 설치 환경 갱신 필요.
- **체크포인트 외부 의존**: `yonigozlan/EdgeTAM-hf`는 커뮤니티(transformers 통합 담당자) repo다. 향후 `facebook/EdgeTAM`에 공식 transformers 가중치가 추가되면 그쪽으로 전환 검토.
- **EdgeTAM 단독 측정 1회**: 단일 군무 클립 기준. 다양한 클립(단일 인물·빠른 군무·저조도)에서 AC-01 재확인 권장(노트북 셀 8.6 재사용).

## 후속
- 다양한 클립으로 EdgeTAM AC-01 재검증(특히 단일 인물·고밀착 군무).
- `facebook/EdgeTAM` 공식 transformers 가중치 추가 시 repo 전환 검토.
- 추가 가속 여지: torch.compile(EdgeTAM에도 적용 가능, ac06 §2-D).
