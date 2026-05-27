# easy-capture

> 뮤직비디오·직캠 등 동영상에서 **특정 오브젝트(인물/사물)를 중심으로 자동 크롭**하여 스크린샷(짤)과 GIF(움짤)를 쉽게 만드는 로컬 데스크톱 프로그램.

클릭 한 번으로 추적 대상을 지정하면, SAM2 기반 추적으로 그 사람/사물을 따라가며 원하는 범위를 잘라 캡처·GIF·MP4 로 내보낸다.

> **현재 상태: 기획 단계.** 코드는 아직 없으며, 본 저장소에는 기획 문서 세트가 있다. 구현은 PoC → MVP 순으로 진행한다.

---

## 주요 기능 (계획)

- **모드 분리**: 이미지(짤) / GIF(움짤)
- **자동 검출 + 클릭 선택**: Grounding DINO 로 후보를 찾아 라벨링, 사용자가 클릭으로 대상 확정
- **추적**: SAM2 video predictor 로 후속 프레임 자동 전파, occlusion(소실) 대기·재추적
- **샷 경계 재추적**: 컷이 바뀌어도 같은 인물 자동 재매칭
- **수동 교정**: 추적이 틀리면 그 지점부터 부분 재추적
- **크롭**: centroid 중심 + 떨림 완화 + 종횡비 잠금(1:1/9:16/16:9)
- **출력**: PNG/JPG · GIF(팔레트·크기예측) · MP4(오디오 포함)
- **업스케일(옵션)**: Real-ESRGAN / SwinIR

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

## 기술 스택 (계획)

Python 3.10+ · PySide6 · SAM 2.1 · Grounding DINO · PySceneDetect · PyAV/ffprobe · Real-ESRGAN/SwinIR · imageio/Pillow

상세·라이선스는 [docs/resources.md](docs/resources.md).

---

## 설치 / 실행 (예정)

> 구현 단계에서 확정. 현재는 계획만 존재.

```bash
git clone <repo-url>
cd easy-capture
cp .env.example .env
# (예정) python -m venv .venv && pip install -r requirements.txt
# (예정) python -m easy_capture
```

- GPU(NVIDIA CUDA) 권장(6GB+). CPU 도 동작하나 느림.
- 최초 실행 시 AI 모델(약 1.5~2GB) 자동 다운로드.

---

## 라이선스 / 저작권 고지

- 앱 라이선스: (구현 단계 확정 — 의존성 LGPL/Apache/BSD 호환 범위에서 결정)
- 의존성 라이선스: [docs/resources.md](docs/resources.md) §2
- **사용자 책임**: 본 프로그램으로 생성한 콘텐츠의 저작권·이용 책임은 사용자에게 있으며, 원본 영상의 저작권 및 배포 플랫폼 약관을 준수해야 한다.
