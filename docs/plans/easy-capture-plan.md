# easy-capture 기획 문서 세트 제작 계획 (v2.1 — 페르소나 2차 검토 반영)

> 이 문서는 승인된 계획서의 저장소 사본이다. 원본 계획 파일과 동일 내용을 유지한다.

## Context (왜 이 작업을 하는가)

뮤직비디오·직캠 등 동영상에서 **오브젝트(인물/사물)별로 포커싱**하여 스크린샷·GIF를 쉽게 만드는 **로컬 데스크톱 프로그램**을 만든다.

- 시작 화면에서 **모드(기능)를 먼저 분리**: ① 이미지(짤) 모드 ② GIF(움짤) 모드.
- 특정 시점에서 segmentation 실행 → 오브젝트 구별 → **자동 검출(Grounding DINO) + 클릭**으로 대상 확정.
- 선택 오브젝트를 영상 전체에 걸쳐 **추적**, 그 주변의 **사용자 지정 범위(가로×세로)** 를 잘라 캡처/GIF 생성.
- GIF 는 오브젝트가 사라져도 **사용자 지정 대기 시간까지 재등장 대기 후 재추적**.
- 사진은 segmentation 후 클릭 오브젝트 주변 크롭.
- **이미지 업스케일링(초해상도)** 옵션 제공.

`easy-capture` 는 비어 있는 신규 저장소다. **이번 작업의 산출물은 코드가 아니라 기획 문서 세트**다. 코드 구현은 다음 단계(PoC → MVP).

> **개정 이력**: ① 1차 검토(영상 전문가/PM/덕후) 전원 "수정 필요" → v2 반영. ② 2차 컨펌: 영상 전문가·덕후 컨펌, PM 경미(P0 3건) → v2.1 반영. 이후 3차에서 PM 포함 **3인 전원 컨펌**으로 통과. 이 검토–수정–컨펌 루프는 이후 생성되는 모든 문서에도 동일 적용한다.

---

## MVP 범위 경계 (In / Out)

scope creep 방지를 위해 v1.0 경계를 명시한다. Out 항목도 **아키텍처는 확장 대비**하되 구현은 미룬다.

| 영역 | MVP v1.0 (In) | 백로그 v1.1+ (Out, 확장 대비) |
|---|---|---|
| 모드 | 이미지 / GIF 분리 | — |
| 대상 선택 | Grounding DINO 자동검출 + 클릭, **단일 오브젝트** 추적 | — |
| 추적 | SAM2 video predictor + occlusion 대기/재등장 | pose/appearance 기반 고급 re-ID |
| 컷 전환 | **샷 경계 감지(PySceneDetect) + 위치·클래스 기반 자동 재매칭 재추적** | 정밀 re-ID 네트워크 |
| 교정 | **수동 교정**(미리보기 중 특정 프레임 재선택 → 부분 재추적) | — |
| 크롭 | centroid 중심 + 떨림완화(기본 ON) + 종횡비 잠금(1:1/9:16/16:9/자유) | 워터마크 자동 회피 |
| 갭 채우기 | 3방식(배경/컷/프리즈, 기본=BACKGROUND) | — |
| 출력 | PNG/JPG, GIF(팔레트·디더·크기예측), MP4(짝수해상도·yuv420p·오디오 패스through) | SNS 플랫폼별 자동 압축/포맷 변환 프리셋 |
| 업스케일 | Real-ESRGAN/SwinIR 2종(2x/4x) 옵션 | temporal smoothing(플리커 저감), 직캠 왜곡보정 |
| 다중 멤버 | 세션·디코드 캐시 재사용으로 반복 비용 완화 | 같은 구간 다중 멤버 배치 일괄 처리 |
| 온보딩 | 단일 흐름 + 합리적 기본값 + 툴팁 | simple/advanced 마법사 |

---

## 결정된 기술 스택

| 영역 | 선택 |
|---|---|
| 언어 | Python 3.10+ |
| GUI | PySide6 (LGPL) |
| 세그+추적 | SAM 2.1 (`facebook/sam2.1-hiera-*`, Apache 2.0), **`transformers>=5.9.0` 단독** (별도 `sam2` 패키지 불필요 — ADR 0001 보완) |
| 클래스 검출 | Grounding DINO (`IDEA-Research/grounding-dino-tiny`, Apache 2.0) |
| 샷 경계 감지 | PySceneDetect (BSD-3) |
| 업스케일 | Real-ESRGAN(+basicsr) + SwinIR/Swin2SR (설정 선택) |
| 비디오 입력/디코드 | PyAV(PTS·VFR) + ffprobe. OpenCV 보조 |
| 색공간 | BGR/limited → RGB BT.709 full 정규화, 출력 색공간 태깅 |
| GIF | imageio/Pillow + 동적 팔레트·디더링·파일크기 예측 |
| MP4 | libx264 yuv420p, 짝수 해상도 정렬, 오디오 mux(ffmpeg) |
| 디바이스 | CUDA 자동 감지 → CPU 폴백 |

---

## 핵심 설계 결정

1. 기능 분리 = 시작 화면 모드 선택(이미지/GIF).
2. 오브젝트 식별 = 자동 검출 + 클릭. 동일 프레임 다중 인물은 후보 라벨링 후 **단일 선택**(동시 다중 추적 아님). 다른 멤버는 세션·디코드 캐시 유지 후 재선택 순차 처리(MVP), 배치는 v1.1.
3. 추적 = SAM2 video predictor 단일 오브젝트 ID 전파.
4. 샷 경계 + 재추적: PySceneDetect 컷 감지 → DINO 재검출 → 재매칭 점수 `w_pos·pos_sim + w_cls·cls_sim`(기본 0.7/0.3), 임계값 0.5 이상이면 SAM2 재초기화, 미만이면 사용자 확인. PoC 보정.
5. occlusion + 갭 채우기: 소실 → 대기 카운터 → 재등장 시 계속, 초과 시 종료. 갭 3방식(기본=BACKGROUND): 배경 계속/컷/프리즈.
6. 수동 교정: 미리보기 일시정지 → 교정 모드 → 세그 재오버레이 → 재지정 → 그 프레임부터 부분 재추적, 이전 구간 보존.
7. 크롭 = centroid 중심 W×H, 떨림완화 N-프레임 이동평균(기본 5, 약3/표준5/강10), 경계 클램프, 짝수 정렬, LANCZOS4, 종횡비 잠금.
8. 구간 지정 = 타임라인 드래그(시작/끝 핸들), 선택 구간만 스트리밍 디코드.
9. 업스케일(옵션) 2종, 2x/4x, CPU 경고, temporal smoothing 후순위.
10. 메모리 관리: 전체 디코드 금지, 구간 스트리밍, 버퍼 상한, VRAM 부족 시 자동 다운스케일.
11. 디바이스 자동 감지 → 티어/해상도 조정, 처리 예상시간 표시.
12. 입력 검증: 코덱/해상도 검사, 권장(1080p+)/최소(480p) 안내, 직캠·저화질 경고.

---

## 산출물: 문서 세트

```
README.md · HANDOFF.md
docs/
├── RFP.md · use-flow.md · wireframes.md · data-flow.md
├── architecture.md · resources.md · poc-plan.md · error-handling.md
├── plans/easy-capture-plan.md (본 문서)
└── adr/0001~0006
```

각 문서 개요·검증 방법은 RFP·아키텍처·poc-plan 등 개별 문서 참조.

---

## 리스크 & PoC 우선 검증

코드 PoC: ① SAM2 video predictor 설치·성능, ② 컷 전환 재추적 현실성(최대 리스크), ③ CPU 실사용성, ④ VFR 오디오 동기.
문서 기록: basicsr/Real-ESRGAN/FFmpeg 라이선스 → resources.md 확정. 통과 기준은 poc-plan.md 수치.

---

## 검증 방법

1. 링크/구조 검증. 2. `/handoff validate`. 3. 요구사항 추적성 매트릭스. 4. **페르소나 3인 전원 컨펌 루프**. 5. repo id·라이선스 대조. 6. docs 커밋 + HANDOFF 갱신.

---

## 다음 단계

문서 전원 컨펌 → 스캐폴딩 → PoC(poc-plan.md 기준) → `/develop` 으로 MVP 구현.
