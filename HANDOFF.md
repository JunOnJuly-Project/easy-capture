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

> 구현 전. 현재는 문서만 존재. (예정: `python -m easy_capture`)

### 2-4. 동작 확인 (smoke test)

> 구현 전. (예정: PoC 스크립트로 SAM2 추적 1건)

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

### 🔴 블로커
없음

### 미완료 (다음 작업 순서) ⏳
1. **PoC**: `feature/poc-core` 브랜치 분기 → `docs/poc-plan.md` H1~H4 코드 검증 (SAM2 설치/추적, 컷 재추적, CPU, 오디오 동기). PoC 산출물은 `poc/` 폴더, 테스트 영상은 `test-data/`(gitignore, 경로·스펙만 poc-plan 에 기록).
2. **라이선스 확정**: Real-ESRGAN/basicsr/FFmpeg 빌드 → `docs/resources.md` §2 결과 기록.
3. PoC 통과(Go/No-Go) → 스캐폴딩 → `feature/{도메인}/{기능}` 브랜치에서 `/develop` 으로 MVP 구현.
4. (선택) 원격 저장소 연결 후 푸시(현재 remote 미설정).

### Git / 분기 전략
- 기본: GitHub Flow. `main` 항상 배포(여기선 "문서 일관") 가능 상태 유지.
- 문서 단계: `main` 직접 커밋 허용(초기). PoC/구현부터 feature 브랜치.

### 알려진 미해결 이슈 / 주의사항
- [ ] Real-ESRGAN/basicsr 상업 라이선스 공식 확정 전: SwinIR 을 기본 업스케일러로 사용
- [ ] SAM2 컷 재추적(ADR 0006) 현실성은 PoC H2 가 최대 리스크
- [ ] 모델 repo id·크기·라이선스는 구현 전 HF/GitHub 로 1회 대조

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
