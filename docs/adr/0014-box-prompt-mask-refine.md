# ADR 0014 — SAM2 box 프롬프트 도입 및 마스크 후처리

- 상태: 채택 (노트북 GPU 게이트 검증 대기)
- 날짜: 2026-05-29

## 맥락

GPU 실검증(멀티샷 군무 직캠)에서 마스크 부정확이 드러났다(사용자 관찰):
- 추적 대상의 "배 일부 + 옆 멤버의 팔"을 마스킹(엉뚱한 영역).
- 전반적으로 마스크 과대 → `clip_crop`이 1인 클로즈업이 아니라 군무 여러 명이 담김.

원인: `VideoCaptureUseCase`가 Grounding DINO의 **전신 bbox**를 알고 있으면서도, SAM2에 `_box_center`(박스 중심 = 배/허리 지점) **점 1개(point 프롬프트)**만 전달한다. 군무 밀집에서 point 1점 세그먼트는 배 주변만 잡거나 인접 멤버로 번진다.

**핵심 — 회귀였다**: PoC 노트북(`poc/colab/easy_capture_gpu_poc.ipynb` 셀 7)은 원래 box 프롬프트(`input_boxes`)로 detect bbox를 전달해 GPU 검증을 통과했다. production `Sam2VideoBackend.add_click`으로 옮기며 point 1점으로 단순화한 것이 **box 능력 회귀**다.

## 결정

### 1. SAM2 box 프롬프트 도입 (add_box Protocol)

transformers 5.9.0 `Sam2VideoProcessor.add_inputs_to_inference_session`는 `input_boxes`를 정식 지원한다(box를 corner-points로 내부 변환). detect 전신 bbox를 box 프롬프트로 전달해 전신을 정확히 세그먼트한다.

- `VideoSegmentationBackend` Protocol에 `add_box(session, box)` 추가([ADR 0010](0010-video-segmentation-backend.md) 3-메서드 분리 연장 — ISP/OCP).
- `add_click`은 **무변경 유지**(단일샷·폴백·무회귀).
- infra `Sam2VideoBackend.add_box`: `input_boxes=[[[x1,y1,x2,y2]]]`, `frame_idx=0`, `obj_ids=1` (PoC 셀 7 패턴 그대로, `clear_old_inputs` 기본 True로 box 제약 충족).

### 2. 마스크 후처리 — 최대 연결성분 (largest_component)

box로도 인접 멤버 팔이 일부 번질 수 있어, SAM2 마스크에서 **최대 4-연결 성분만** 남긴다.

- `core/crop/mask_refine.py`의 `largest_component(mask)` 순수 함수. **numpy 순수 구현(scipy·cv2 배제 — core 경계 불변식)**. 빈/단일 마스크는 그대로 반환.
- track propagate 후 masks 확정 시 적용. `compute_boxes`는 무변경(refined 마스크 bbox가 작아지면 크롭 박스가 자동 정상화 — "마스크 과대 → 박스 과대" 연쇄를 끊는 지점).

### 3. 혼합 디스패치 정책 (box 우선, point 폴백)

- 자동 재매칭 `match.passed` → `add_box(candidates[best].box)` (중심점 변환 폐기).
- `CutSelection.box`(Optional 신규, default None) 있으면 `add_box`, 없으면 `add_click(point)`.
- box 없는 경로(첫 샷 point·검출 실패·단일샷) → `add_click` 무회귀.

## 대안

- **add_click 확장(box 인자 추가)**: 한 메서드가 box/point 분기를 모두 처리해 시그니처가 모호해진다. `add_box` 별도 메서드가 ISP에 정합. 거부.
- **scipy.ndimage.label**: core에 무거운 의존 유입(경계 불변식 위반). numpy 순수 구현으로 대체. 거부(성능 문제 시 infra `cv2.connectedComponents` 폴백을 후속으로 열어둠 — core 함수는 유지).
- **negative point(옆 사람 배제 클릭)**: UI·자동 선정 로직 복잡도 급증. box + largest_component로 충분하다고 가정. 잔여 회귀 시 후속.

## 결과

### 긍정적 영향
- 전신 정확 마스크 → **1인 클로즈업 크롭**(과대 해소). 사용자 관찰("배+옆팔") 직접 해결.
- 추적 유지율 개선 기대(point 마스크 불안정 → box 전신 안정).
- 무회귀: `add_click` 미변경, box 없으면 기존 동작. **561 passed**.
- SOLID: ISP(add_box 분리), OCP(Protocol 확장으로 기존 불변), DIP(app은 Protocol만 의존).

### 트레이드오프 / 리스크
- `largest_component` numpy 구현이 프레임당 전픽셀 순회면 느릴 수 있다(True 픽셀만 순회·visited로 완화). 성능은 **노트북 게이트에서 프레임당 처리시간 측정** — 초과 시 infra cv2 폴백.
- box가 단일 인물·애니에서 point보다 좁게/넓게 잡을 가능성 → 게이트에서 무회귀 측정. 회귀 시 단일샷은 point 유지 정책.
- infra(`sam2_video_backend`)는 CI 미실행 — **노트북 GPU 게이트 필수**.

### 검증 경로 (게이트)
GPU 의존(detect·track)이라 노트북(Colab) 우선 검증: box vs point **마스크 정확도(1인 클로즈업)** + **추적 유지율** 재측정이 데스크톱 적용 게이트.

## 연계

- [ADR 0010](0010-video-segmentation-backend.md) — `VideoSegmentationBackend`에 `add_box` 추가(3-메서드 분리 연장). 본 ADR이 0010을 확장(Superseded 아님).
- [ADR 0006](0006-shot-boundary-reid.md) — 재매칭 로직 유지, 출력(클릭점)만 box로 전달.
- [ADR 0012](0012-detection-backend.md) — `DetectionBackend`의 전신 box를 SAM2 box 프롬프트로 직접 활용.
- [ADR 0013](0013-time-remap-location.md) — `core/crop`·`core/timing`과 동일하게 `mask_refine`도 core 순수(numpy/stdlib) 경계 준수.
