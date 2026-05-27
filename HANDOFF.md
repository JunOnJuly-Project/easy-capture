# easy-capture 핸드오프 문서

> 다른 PC / 다른 세션에서 이 프로젝트를 **끊김 없이 이어서 진행**하기 위한 안내서.
> 스키마 버전: v2
> 최종 업데이트: 2026-05-27

---

## 1. 프로젝트 개요

**easy-capture** — 동영상에서 오브젝트를 추적해 중심 크롭으로 짤/움짤을 만드는 로컬 데스크톱 프로그램.

- **대상**: 직캠·움짤 제작 팬 콘텐츠 제작자(Primary), 일반 덕후(Secondary)
- **스택**: Python 3.10+ · PySide6 · SAM 2.1 · Grounding DINO · PySceneDetect · PyAV · Real-ESRGAN/SwinIR
- **방법론**: 기획 단계는 페르소나(영상전문가/PM/덕후) 검토–수정–컨펌 루프. 구현 단계는 `/develop` 팀 파이프라인.

상세 계획서: [`docs/plans/easy-capture-plan.md`](docs/plans/easy-capture-plan.md)

---

## 2. 다른 PC에서 이어 받기

### 2-1. 필수 도구

| 도구 | 버전 | 용도 |
|---|---|---|
| Git | 2.40+ | 소스 클론 |
| Python | 3.10+ | (구현 단계) 런타임 |
| FFmpeg | 최신 LGPL 빌드 | (구현 단계) 디코드/인코드 |

### 2-2. 클론 및 시크릿

```bash
git clone <repo-url>
cd easy-capture
cp .env.example .env
```

### 2-3. 실행

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"   # PySide6 등 포함
python -m easy_capture                    # 시작 화면(모드 선택)
```

### 2-4. 동작 확인 (smoke test)

```bash
.venv\Scripts\pytest -q     # core/infra 순수 로직 테스트 (현재 20개)
```
GPU 비디오 추적 검증은 `poc/colab/` 노트북(Colab GPU).

---

## 3. 현재 진행 상태

### 현재 브랜치
`main` (초기 기획 문서 작성)

### 완료 ✅

| Phase | 항목 | 상태 |
|---|---|---|
| 기획 | 계획서 v2.1 (페르소나 3인 전원 컨펌) | ✅ |
| 기획 | 문서 세트: RFP·use-flow·wireframes·data-flow·architecture·resources·poc-plan·error-handling·ADR 0001~0006 | ✅ |
| 기획 | **문서 세트 페르소나 검토–수정–컨펌 루프 (영상전문가·PM·덕후 3인 전원 컨펌)** | ✅ |
| PoC | **`feature/poc-core` H1~H4 코드 검증 (조건부 Go)** — 상세 `poc/REPORT.md` | ✅ |
| 조사 | **라이선스 확정**(resources.md §2): 기본 업스케일러=SwinIR, Real-ESRGAN 가중치 DIV2K 리스크, libx264=GPL(상업 시 주의) | ✅ |
| 조사 | **CPU 개발 전략 확정**(ADR 0007): 이중 경로 — 이미지=CPU, 비디오=클라우드 GPU | ✅ |
| 구현 | **스캐폴딩**(`feature/app/scaffolding`): 패키지 구조(src/easy_capture, core/infra/ui), core 순수 로직(crop·gap_policy·rematch·device·backend IF), PySide6 모드선택 셸, **테스트 20개 통과** | ✅ |

### 🔴 블로커
- **GPU(CUDA) 사실상 필수**: PoC 실측상 SAM2 추적이 **CPU 에서 ≈0.10 fps**(프레임당 ~10초, 6초 클립에 ~14분). 현재 개발 PC 는 CPU 전용 → **실영상 추적·재추적 검증과 실사용에 GPU 환경 필요**. 하드웨어/클라우드 방향 결정 대기.

### 미완료 (다음 작업 순서) ⏳
1. **하드웨어 방향 결정**: GPU PC / 클라우드 GPU(Colab·런팟 등) 확보 여부.
2. **실영상 검증(GPU)**: `poc/colab/easy_capture_gpu_poc.ipynb` 를 Colab(GPU) 에서 실행, 짧은 군무 MV 클립으로 H1 추적 유지율(AC-01 ≥80%)·H2 컷 재매칭(AC-03 ≥70%, Grounding DINO 포함)·GPU fps(AC-06) 측정 → `poc/REPORT.md` 미검증 항목 채우기. (사용법: `poc/colab/README.md`)
3. **다음 슬라이스 (CPU 개발 가능)**: 이미지 모드 파이프라인 — `infra/video_io`(PyAV/ffprobe), `core/segmentation` SAM2 image 백엔드 연결, `core/crop`→`core/export`(PNG/JPG) 연결, UI 캔버스(프레임 표시·클릭). 모드선택→메인윈도 라우팅.
4. **비디오 모드 슬라이스 (GPU)**: SAM2 video 백엔드 + tracking + 샷경계 재추적 + GIF/MP4 export. 로컬 검증은 Colab.
5. 업스케일(`core/upscale`, 기본 SwinIR) 연결, 수동 교정 UI.
6. (정리) `docs/planning-set`·`feature/poc-core`·`feature/app/scaffolding` → main PR/머지.

### PoC 핵심 결과 (요약)
- SAM2(이미지+비디오)·Grounding DINO 는 **transformers 5.9.0 만으로** 사용 가능(별도 `sam2` 패키지 불필요).
- 추적 정확성 OK(합성 100%), 컷 감지·오디오 동기 OK. **병목은 오직 SAM2 추론(GPU 필요)**.

### Git / 분기 전략
- 기본: GitHub Flow. `main` 항상 배포(여기선 "문서 일관") 가능 상태 유지.
- 문서 단계: `main` 직접 커밋 허용(초기). PoC/구현부터 feature 브랜치.

### 알려진 미해결 이슈 / 주의사항
- [x] 라이선스 확정 완료 → **기본 업스케일러 SwinIR**. Real-ESRGAN 은 옵션(상업 배포 시 비활성). **libx264=GPL** → 상업 배포 시 코덱/라이선스 법무 재검토.
- [ ] SAM2 컷 재추적(ADR 0006) 현실성은 PoC H2(실영상·GPU) 가 최대 리스크 — 미검증
- [ ] EdgeTAM(CPU 비디오 추적 후보)·경량 백엔드는 v1.1 평가

---

## 4. 방법론 강제 규칙

- **기획 문서**: 영상전문가 / PM / 아이돌 덕후 3개 페르소나 검토 → 수정 → **3인 전원 컨펌 시 통과**. 모든 신규/수정 문서에 적용.
- **MVP 범위**: `docs/RFP.md` §5 및 계획서 In/Out 표 준수(scope creep 방지). Out 항목은 아키텍처만 확장 대비.
- **커밋-문서 동기화**: 코드 커밋 시 연관 문서·HANDOFF 동시 갱신(전역 지침).

---

## 5. 재개 체크리스트

새 세션에서 Claude 에게 다음을 먼저 실행하게 하라:

1. 이 문서(`HANDOFF.md`) 전체 읽기
2. `/bootstrap` 실행 (정합성 게이트)
3. `git log --oneline --all --graph | head -30` 확인
4. 블로커가 있으면 먼저 해결
5. 아니면 "미완료" 섹션의 다음 우선순위 작업 시작 (현재: 문서 검토 루프 → PoC)

---

## 6. 참고 자료

- 계획서: `docs/plans/easy-capture-plan.md`
- 전역 지침: `~/.claude/CLAUDE.md`
