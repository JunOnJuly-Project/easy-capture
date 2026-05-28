# easy-capture 핸드오프 문서

> 다른 PC / 다른 세션에서 이 프로젝트를 **끊김 없이 이어서 진행**하기 위한 안내서.
> 스키마 버전: v2
> 최종 업데이트: 2026-05-28 (이미지 모드 첫 수직 슬라이스 완료)

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
.venv\Scripts\pytest -q     # 순수 로직 단위 테스트 (현재 71개)
```

GPU 비디오 추적 검증은 `poc/colab/` 노트북(Colab GPU).

**이미지 모드 수동 스모크**

```bash
python -m easy_capture        # 모드 선택 → 이미지 선택
# 이미지 또는 영상 파일 열기 → 캔버스에 프레임 표시
# 피사체 클릭
# 첫 클릭 시 SAM2 모델 1회 다운로드(facebook/sam2.1-hiera-tiny, 수백 MB) + CPU 추론 1~3s
# 크롭 박스 생성 → 저장(PNG/JPG)
```

주의사항:
- SAM2 첫 클릭은 모델 다운로드(수백 MB) + CPU 추론 지연이 있으나, 워커 스레드로 실행되므로 UI는 멈추지 않는다.
- 빈 배경 클릭 시 "대상을 인식하지 못했어요. 다시 클릭해 주세요" 안내가 표시된다.
- 이후 클릭은 모델이 메모리에 로드된 상태이므로 추론만 수행(빠름).

---

## 3. 현재 진행 상태

### 현재 브랜치
`feature/image/capture-slice` (이미지 모드 첫 수직 슬라이스 완료, 다음 슬라이스 대기)

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
| 구현 | **이미지 모드 첫 수직 슬라이스**(`feature/image/capture-slice`): end-to-end happy path — 파일→프레임→클릭→SAM2(CPU)→크롭→PNG/JPG. 신규 모듈: `core/export`(crop_array/save_image, Pillow), `infra/video_io`(FrameSource Protocol·FrameMeta·open_source, 이미지=Pillow·영상=PyAV 첫프레임), `infra/sam2_image_backend`(Sam2ImageBackend, transformers 5.9.0 Sam2Model/Sam2Processor, 지연 로드, CPU), `app/image_capture`(ImageCaptureUseCase·CropRequest·EmptyMaskError), `app/router`(AppRouter 조립 루트), `ui/coords`(좌표 변환 순수함수), `ui/frame_canvas`, `ui/main_window`(워커 스레드 비블로킹). **테스트 71개 통과**(기존 20 포함). ADR 0008(app 유스케이스 레이어) 추가. 코드 리뷰 [중요] 4건 전원 반영. | ✅ |

### 🔴 블로커
- **GPU(CUDA) 사실상 필수**: PoC 실측상 SAM2 추적이 **CPU 에서 ≈0.10 fps**(프레임당 ~10초, 6초 클립에 ~14분). 현재 개발 PC 는 CPU 전용 → **실영상 추적·재추적 검증과 실사용에 GPU 환경 필요**. 하드웨어/클라우드 방향 결정 대기.

### 미완료 (다음 작업 순서) ⏳
1. **SAM2 실모델 CPU 수동 스모크**: 위 "2-4 동작 확인" 절차에 따라 `python -m easy_capture` 실행 → 이미지 파일 열기 → 클릭 → 모델 다운로드/추론 → PNG 저장 전 구간 직접 확인.
2. **이미지 모드 확장**: 종횡비 프리셋 UI(1:1/9:16/16:9 선택 버튼), 마스크 오버레이 연결(현재 `set_overlay` 미연결, 클릭 후 시각 피드백 없음), 사용자 크롭 크기 조정 슬라이더, 업스케일(SwinIR).
3. **비디오 모드 슬라이스 (GPU)**: SAM2 video 백엔드 + tracking + 샷경계 재추적 + GIF/MP4 export. 로컬 검증은 `poc/colab/easy_capture_gpu_poc.ipynb`(Colab GPU). PoC H1 추적 유지율(AC-01 ≥80%)·H2 컷 재매칭(AC-03 ≥70%)·GPU fps(AC-06) 측정 → `poc/REPORT.md` 미검증 항목 채우기.
4. (정리) `feature/poc-core`·`feature/app/scaffolding`·`feature/image/capture-slice` → main PR/머지.

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
- [ ] **[GPU 블로커] 비디오 모드**: SAM2 추적이 CPU 에서 ≈0.10 fps — GPU 환경(Colab 등) 없이 비디오 슬라이스 진행 불가. 하드웨어/클라우드 방향 결정 대기.

**리뷰 제안 백로그** (이미지 모드 슬라이스 코드 리뷰 [제안] 7건):
- `infra/video_io`: fps 산출에 `average_rate` 사용 — VFR 영상에서 부정확할 수 있음 (다음 슬라이스에서 `r_frame_rate` 폴백 추가 검토)
- `infra/sam2_image_backend`: `_ensure_loaded()` 스레드 락 미적용 — 현재는 UI 워커 단일 스레드로 가드되나, 다중 워커 확장 시 경합 위험
- `ui/frame_canvas`: 마스크 오버레이 픽셀 이중루프 — numpy 벡터화 또는 Qt alpha blend로 교체 시 성능 향상
- `ui/frame_canvas`: `set_overlay` 미연결 — 클릭 후 마스크 시각 피드백 없음 (이미지 모드 확장 슬라이스에서 연결 예정)
- `ui/main_window`: `box_size` 매직넘버 — 설명적 상수명(`DEFAULT_CROP_BOX_SIZE`) 으로 추출 예정
- `tests/`: 일부 테스트 독스트링에 "TDD Red" 단계 메모 잔존 — 정리 필요
- `tests/fixtures/fakes.py`: `_make_rect_mask` 매개변수 5개 — dataclass 로 묶기 검토

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
