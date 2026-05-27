# easy-capture — 리소스 / 기술스택 명세

> 최종 업데이트: 2026-05-27 · 관련: [아키텍처](architecture.md) · [ADR](adr/) · [RFP](RFP.md)
> ⚠ 모든 repo id·크기·라이선스는 **구현 착수 전 공식 페이지(HF/GitHub)로 1회 대조**한다.

---

## 1. 모델 카탈로그

| 모델 | repo id (예시) | 크기(대략) | VRAM(대략) | 라이선스 | 용도 |
|---|---|---|---|---|---|
| SAM 2.1 Tiny | `facebook/sam2.1-hiera-tiny` | ~155MB | 2~4GB | Apache 2.0 | 기본 세그+추적 |
| SAM 2.1 Small | `facebook/sam2.1-hiera-small` | ~185MB | 3~5GB | Apache 2.0 | 중간 품질 |
| SAM 2.1 Base+ | `facebook/sam2.1-hiera-base-plus` | ~325MB | 5~7GB | Apache 2.0 | GPU 시 |
| SAM 2.1 Large | `facebook/sam2.1-hiera-large` | ~900MB | 8GB+ | Apache 2.0 | 최고 품질 |
| Grounding DINO | `IDEA-Research/grounding-dino-tiny` | ~700MB | 2~4GB | Apache 2.0 | 클래스 자동 검출 |
| Real-ESRGAN x4plus | (realesrgan 패키지) | ~64MB | 2~4GB | BSD-3-Clause ※ | 업스케일(실사) |
| Real-ESRGAN x4plus-anime | (realesrgan 패키지) | ~18MB | 2~3GB | BSD-3-Clause ※ | 업스케일(애니/MV) |
| SwinIR / Swin2SR | `caidas/swin2SR-*` (transformers) | ~200MB | 3~5GB | Apache 2.0 | 업스케일(실사, 간결) |

> 디바이스 자동 감지 결과에 따라 SAM2 티어를 선택(CPU→Tiny, GPU→Base+/Large).

---

## 2. 라이선스 종합표 (상업 사용 안전성)

| 구성요소 | 라이선스 | 상업 사용 | 비고 |
|---|---|---|---|
| SAM 2.1 | Apache 2.0 | ✅ | — |
| Grounding DINO | Apache 2.0 | ✅ | — |
| PySceneDetect | BSD-3-Clause | ✅ | — |
| Real-ESRGAN (코드) | BSD-3-Clause | ✅(검증 필요 ※) | 사전학습 가중치 출처별 조건 재확인 |
| **basicsr** (Real-ESRGAN 의존) | Apache 2.0 | ✅(검증 필요 ※) | Real-ESRGAN 의 필수 의존성 |
| SwinIR / Swin2SR | Apache 2.0 | ✅ | transformers 내장 |
| PyAV | BSD-3-Clause | ✅ | FFmpeg 바인딩 |
| FFmpeg (런타임) | LGPL/GPL 빌드별 | ⚠ | LGPL 빌드 사용 권장, GPL 코덱 포함 빌드 주의 |
| imageio / imageio-ffmpeg | BSD-2 | ✅ | — |
| Pillow | HPND(MIT 계열) | ✅ | — |
| PySide6 | LGPLv3 | ✅ | 동적 링크 시 LGPL 준수 |
| torch / transformers | BSD / Apache 2.0 | ✅ | — |

> **※ 검증 필요 항목 (배포 전 확정, 본 표에 결과 기록)**: ① Real-ESRGAN 코드/가중치 라이선스를 공식 repo `LICENSE` 로 확정(웹 자료 간 BSD vs 비상업 혼동 존재). ② basicsr 라이선스·버전. ③ 배포할 FFmpeg 빌드(LGPL vs GPL) 및 포함 코덱.
> 결론(현 시점 판단): SAM2·DINO·SwinIR·PySceneDetect·PyAV 경로는 상업 안전. Real-ESRGAN 은 옵션 기능이므로, 라이선스 재확인 전에는 **SwinIR 을 기본 업스케일러**로 두고 Real-ESRGAN 을 사용자 선택으로 제공.

---

## 3. 의존성 (예상 requirements)

```
# AI / 추적
torch, torchvision
transformers
sam2                 # 공식 video predictor
scenedetect          # PySceneDetect

# 업스케일
realesrgan, basicsr  # 옵션 (라이선스 확정 후)
# SwinIR 은 transformers 로 사용

# 비디오 / 이미지 IO
av                   # PyAV
ffmpeg-python        # 오디오 mux
imageio, imageio-ffmpeg
pillow
numpy, opencv-python # 보조

# GUI
PySide6
```

> Python 3.10+. CUDA 빌드 torch 는 환경별로 설치(설치 가이드는 [README](../README.md)).

---

## 4. 지원 플랫폼 매트릭스

| OS | GPU(CUDA) | CPU only | 비고 |
|---|---|---|---|
| Windows 11 | ✅ 우선 지원 | ⚠ 느림 | 1차 개발/테스트 환경 |
| macOS (Apple Silicon) | MPS 부분 | ⚠ | SAM2 MPS 지원 여부 PoC 확인 |
| Linux | ✅ | ⚠ | CUDA 환경 동작 예상 |

---

## 5. 하드웨어 권장 / 최소 사양 & 예상 성능

| 시나리오 | 사양 | SAM2 추적 속도(대략) | 평가 |
|---|---|---|---|
| 권장 | NVIDIA GPU 6GB+ (RTX 3060급) | 10~15 fps | 실사용 쾌적 |
| 가능 | NVIDIA GPU 4GB | 5~10 fps (Tiny) | 사용 가능 |
| 최소 | CPU (i7급) + 16GB RAM | 1~2 fps | 짧은 구간만, 대기 길음 |
| 업스케일 | GPU 권장 | CPU 시 분 단위 | CPU 경고 표시 |

> 정확한 수치는 [poc-plan.md](poc-plan.md) 벤치마크로 확정. 처리 예상 시간은 디바이스 감지 후 UI 표시.

---

## 6. SNS 출력 제약 (참고 가이드, MVP 는 안내만)

| 플랫폼 | GIF 제한 | 영상 제한 | 권장 종횡비 |
|---|---|---|---|
| X(트위터) | 15MB | 512MB | 16:9 / 1:1 |
| 인스타 피드 | (GIF 미지원) | ~4GB | 1:1 또는 9:16 |
| 인스타 릴스/TikTok | (GIF 미지원) | 수백 MB | 9:16 |
| YouTube Shorts | — | — | 9:16 |

> MVP: 종횡비 잠금 프리셋(1:1/9:16/16:9) + GIF 파일크기 예측 + "X 는 GIF 15MB, 그 외는 MP4 권장" 안내. 자동 압축/포맷 변환은 v1.1.

---

## 7. 모델 다운로드 / 캐시

- 최초 실행 시 HuggingFace 허브에서 다운로드 → 표준 HF 캐시(`HF_HOME`/기본 경로)에 저장, 이후 오프라인 재사용.
- 총 다운로드 용량(Tiny 구성): 대략 1.5~2GB (SAM2 Tiny + DINO + 업스케일러).
- 다운로드 매니저: 진행률 콜백, 실패 재시도(기본 3회·지수 백오프), 네트워크 차단 시 오프라인 안내.
