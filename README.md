# easy-capture

> 뮤직비디오·직캠 등 동영상에서 **특정 오브젝트(인물/사물)를 중심으로 자동 크롭**하여 스크린샷(짤)과 GIF(움짤)를 쉽게 만드는 로컬 데스크톱 프로그램.

클릭 한 번으로 추적 대상을 지정하면, SAM2 기반 추적으로 그 사람/사물을 따라가며 원하는 범위를 잘라 캡처·GIF·MP4 로 내보낸다.

> **현재 상태: 구현 중 — 이미지 모드 완성 + 비디오 모드 핵심 기능 완료 + GPU 실검증 진행.**
> 이미지: 클릭→SAM2(CPU)→크롭→PNG/JPG + 종횡비·크기·업스케일(SwinIR). 비디오: 구간 추적→컷별 오브젝트 선택→크롭→GIF/MP4 + 슬로우모션·트림·루프(가변 재생속도, ADR 0013)·마스크 정제(box 프롬프트+largest_component, ADR 0014). GPU(Colab T4) 실검증 완료 — 단일샷 추적 우수(유지율 100%), 멀티샷은 컷별 명시 선택으로 정확도 확보. GPU 추적 속도는 실측 ≈2.3fps(목표 10fps 미달 — v1.1 경량 백엔드 과제).

---

## 주요 기능

(✅ 구현됨 / 🔜 후속)

- ✅ **모드 분리**: 이미지(짤) / GIF(움짤)
- ✅ **자동 검출 + 클릭 선택**: Grounding DINO 로 후보를 찾아 라벨링, 사용자가 클릭으로 대상 확정
- ✅ **추적**: SAM2 video predictor 로 후속 프레임 자동 전파, occlusion(소실) 갭 정책(컷/배경/프리즈)
- ✅ **샷 경계 재추적**: 컷이 바뀌어도 같은 인물 자동 재매칭(폴백)
- ✅ **수동 교정 = 컷별 오브젝트 선택**: 각 컷의 추적 대상을 사용자가 직접 지정해 컷별 재추적(자동 재매칭이 멀티샷에서 한계 → 명시 선택으로 정확도 확보, ADR 0006 보강)
- ✅ **마스크 정제**: SAM2 box 프롬프트(전신 bbox) + 최대 연결성분(largest_component)으로 1인 클로즈업 마스크 확보(ADR 0014)
- ✅ **슬로우모션·트림·루프**: 구간별 가변 재생속도(슬로우/패스트) + 출력 구간 트림 + GIF 루프 횟수(ADR 0013)
- ✅ **크롭**: 피사체 bbox 중심 + 떨림 완화 + 종횡비 잠금(1:1/9:16/16:9)
- ✅ **출력**: PNG/JPG · GIF(팔레트·크기예측·per-frame duration) · MP4(오디오 포함)
- ✅ **업스케일(옵션)**: SwinIR(기본) / Real-ESRGAN
- 🔜 **부드러운 슬로우(RIFE 보간)·경량 추적 백엔드(10fps 목표)**: v1.1

---

## 아키텍처 (레이어 구조)

```
ui  ──▶  app  ──▶  core  (Protocol/도메인 로직, 외부 라이브러리 비의존)
                    ▲
         infra  ────┘  (PyAV·SAM2·Pillow 구현체 주입)
```

- **core**: 순수 로직(`torch`/`PySide6`/`av`/`scenedetect` 비의존). 크롭(`crop`: 떨림완화·종횡비잠금 + `mask_refine` 마스크 정제)·내보내기(`export`)·타임리맵(`timing`: 슬로우/트림/루프)·추적(`tracking`: gap_policy·rematch·shot_split·smooth + `cut_selection` 컷별 명시 선택)·추상 Protocol(`segmentation`: SegmentationBackend·VideoSegmentationBackend(add_box 포함)·DetectionBackend / `upscale`: UpscaleBackend / `source`: FrameSource).
- **infra**: Protocol 구현체 — `Sam2ImageBackend`·`Sam2VideoBackend`·`GroundingDinoBackend`·`Swin2srUpscaleBackend`(transformers), `video_io`(PyAV·Pillow), `shot_detect`(PySceneDetect).
- **app**: `ImageCaptureUseCase`·`VideoCaptureUseCase` — 파일→추적→크롭→내보내기 오케스트레이션. `AppRouter` 조립 루트.
- **ui**: PySide6 위젯(`FrameCanvas`·`ImageMainWindow`·`VideoMainWindow`). 도메인 로직은 app 레이어에 위임.

상세: [docs/architecture.md](docs/architecture.md) · [docs/adr/0008-app-usecase-layer.md](docs/adr/0008-app-usecase-layer.md)

---

## 문서

| 문서 | 내용 |
|---|---|
| [HANDOFF.md](HANDOFF.md) | **다른 PC/세션에서 이어받기** (먼저 읽기) |
| [docs/plans/easy-capture-plan.md](docs/plans/easy-capture-plan.md) | 승인된 계획서 (v2.1) |
| [docs/RFP.md](docs/RFP.md) | 요구사항 정의서 |
| [docs/use-flow.md](docs/use-flow.md) | 유즈플로우 |
| [docs/wireframes.md](docs/wireframes.md) | 와이어프레임 |
| [docs/data-flow.md](docs/data-flow.md) | 데이터 플로우 |
| [docs/architecture.md](docs/architecture.md) | 아키텍처 |
| [docs/resources.md](docs/resources.md) | 리소스/기술스택 |
| [docs/poc-plan.md](docs/poc-plan.md) | PoC 검증 계획 |
| [docs/error-handling.md](docs/error-handling.md) | 에러/엣지케이스 |
| [docs/adr/](docs/adr/) | 기술 결정 기록(ADR) |

---

## 기술 스택

| 레이어 | 라이브러리 |
|---|---|
| UI | Python 3.10+ · PySide6 |
| 세그멘테이션 | SAM 2.1 (transformers 5.9.0, `facebook/sam2.1-hiera-tiny`) |
| 영상 디코드 | PyAV · Pillow |
| 자동 검출·샷경계 재추적 (비디오, GPU 실검증 완료 — 단일샷 우수·멀티샷 컷별 선택) | Grounding DINO · PySceneDetect |
| 업스케일 (이미지 모드 완료) | SwinIR (기본, Swin2SR) · Real-ESRGAN (옵션) |

상세·라이선스는 [docs/resources.md](docs/resources.md). 스택 결정 배경은 `docs/adr/` 참조.

---

## 설치 / 실행

```bash
git clone <repo-url>
cd easy-capture
cp .env.example .env
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"   # Windows
# source .venv/bin/activate && pip install -e ".[dev]"   # macOS/Linux
python -m easy_capture
```

### 이미지(짤) 모드 사용법
1. 시작 화면에서 **이미지** 선택
2. 이미지 파일 또는 영상 파일 열기 (영상은 첫 프레임을 추출)
3. 캔버스에서 원하는 피사체를 **클릭**
4. SAM2가 피사체를 자동 검출하고 크롭 박스를 생성 (첫 클릭은 모델 다운로드 포함, 약 1~3초)
5. **저장** 버튼으로 PNG 또는 JPG 내보내기

> 첫 클릭 시 SAM2 모델(facebook/sam2.1-hiera-tiny, 수백 MB)이 자동 다운로드된다.  
> CPU 전용 환경에서도 동작하나, SAM2 추론에 1~3초 소요된다. 워커 스레드로 실행되므로 UI는 멈추지 않는다.  
> GPU(NVIDIA CUDA 6GB+) 환경에서는 추론 속도가 크게 향상된다.

---

## 라이선스 / 저작권 고지

- 앱 라이선스: (구현 단계 확정 — 의존성 LGPL/Apache/BSD 호환 범위에서 결정)
- 의존성 라이선스: [docs/resources.md](docs/resources.md) §2
- **사용자 책임**: 본 프로그램으로 생성한 콘텐츠의 저작권·이용 책임은 사용자에게 있으며, 원본 영상의 저작권 및 배포 플랫폼 약관을 준수해야 한다.
