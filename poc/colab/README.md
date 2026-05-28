# Colab GPU PoC 검증

CPU 전용 개발 PC 에서 확인 불가했던 항목(실영상 추적 유지율·컷 재매칭·GPU 속도)을 **무료 Colab GPU** 로 검증하기 위한 노트북.

검증 항목: 추적 유지율(AC-01 ≥80%), 컷 재매칭(AC-03 ≥70%), GPU 속도(AC-06 ≥10fps), 오디오 동기 클립(AC-05).

---

## 실행 방법

### 방법 A — GitHub 에서 Colab 으로 바로 열기 (브랜치 푸시 후)
아래 URL 을 브라우저에 붙여넣으면 Colab 에서 열린다:

```
https://colab.research.google.com/github/JunOnJuly-Project/easy-capture/blob/feature/poc-core/poc/colab/easy_capture_gpu_poc.ipynb
```

### 방법 B — 직접 업로드
1. [colab.research.google.com](https://colab.research.google.com) 접속 → **업로드** → `easy_capture_gpu_poc.ipynb` 선택.

---

## 순서

1. **런타임 → 런타임 유형 변경 → 하드웨어 가속기: GPU(T4)** 설정.
2. 셀을 위에서부터 순서대로 실행(1번 셀이 의존성 설치, 수 분 소요).
3. **짧은 MV/직캠 클립(≤10초 권장)** 업로드.
4. 검출된 인물 그림을 보고 `TARGET` 인덱스 지정 → 추적 셀 실행.
5. 출력 확인:
   - 콘솔: **추적 유지율 / GPU fps / 컷 재매칭 로그**
   - `track_overlay.mp4`: 최애를 잘 따라가는지·컷 후 다른 사람으로 안 바뀌는지 **눈으로 확인**
   - `clip_audio.mp4`: 크롭 + 오디오 동기 결과

---

## 조정 포인트 (정확도 낮을 때)
- DINO `PROMPT`(기본 `"person."`), `threshold`/`text_threshold`.
- `rematch` 의 `w_pos`/`w_cls`(기본 0.7/0.3), 컷 재매칭 threshold(기본 0.5).
- `MAX_SECONDS`(처리 길이).

## 결과 반영
측정 수치를 `poc/REPORT.md` 의 "미검증" 항목에 채워 **Go/No-Go 확정**.

> 노트북은 로컬 CPU 에서 API 호환성(transformers 5.x)을 검증한 뒤 작성했으나 GPU 실행은 미검증이다. 에러가 나면 **트레이스백을 공유**해 주면 바로 수정한다.

---

## easy_capture_app_verify.ipynb — 앱 코드 검증 노트북

**기존 PoC 노트북의 크롭 버그(`crop=320:320:0:0` 좌상단 고정)를 재현하지 않도록**, 앱 패키지(`src/easy_capture`)의 공개 API를 직접 사용해 추적→크롭→GIF/MP4 파이프라인을 검증하는 노트북.

### 기존 PoC 노트북과의 차이

| 항목 | `easy_capture_gpu_poc.ipynb` | `easy_capture_app_verify.ipynb` |
|------|-----------------------------|---------------------------------|
| 크롭 방식 | `ffmpeg crop=320:320:0:0` 좌상단 고정 더미 | `VideoCaptureUseCase.compute_boxes()` — centroid 기반 추적 크롭 |
| 코드 경로 | 인라인 스크립트 | **앱 패키지 공개 API 직접 사용** |
| 주요 검증 | H1·H2·H4 PoC 검증 | **크롭 정합성 + AC-01 + AC-06** |

### Colab 에서 열기

```
https://colab.research.google.com/github/JunOnJuly-Project/easy-capture/blob/feature/video/gap-policy-ui/poc/colab/easy_capture_app_verify.ipynb
```

### 순서

1. **런타임 → 런타임 유형 변경 → GPU(T4)** 설정.
2. 셀을 위에서 아래로 순서대로 실행.
3. **짧은 MV/직캠 클립(≤10초 권장)** 업로드.
4. 검출된 인물 그림에서 `TARGET_IDX` 지정.
5. 결과 확인:
   - 콘솔: **추적 유지율(AC-01) / GPU fps(AC-06)**
   - `clip_crop.gif` / `clip_crop.mp4`: **피사체를 따라가는지** 육안 확인 (좌상단 고정이면 버그)
   - `track_overlay.mp4`: 마스크 해상도 정합 확인

### 결과 반영

수치를 `poc/REPORT.md` 의 미검증 항목에 채워 **Go/No-Go 확정**.
