# easy-capture — 리소스 / 기술스택 명세

> 최종 업데이트: 2026-05-28 · 관련: [아키텍처](architecture.md) · [ADR](adr/) · [RFP](RFP.md)
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

## 2. 라이선스 종합표 (상업 사용 안전성) — 공식 repo 확인 2026-05-27

| 구성요소 | 라이선스 | 상업 사용 | 근거 / 비고 |
|---|---|---|---|
| SAM 2.1 (코드+가중치) | Apache 2.0 | ✅ | 공식 facebookresearch/sam2 |
| Grounding DINO | Apache 2.0 | ✅ | IDEA-Research |
| PySceneDetect | BSD-3-Clause | ✅ | — |
| **SwinIR / Swin2SR** (코드) | Apache 2.0 | ✅ | **기본 업스케일러로 채택** |
| Real-ESRGAN (코드) | BSD-3-Clause | ✅ | — |
| **Real-ESRGAN 가중치** | BSD-3, 단 **DIV2K(학술 전용) 학습** | ⚠ | 상업 배포 시 데이터 계보 리스크 → 기본 비활성·옵션 시 경고 |
| basicsr (Real-ESRGAN 의존) | Apache 2.0 | ✅ | XPixelGroup/BasicSR |
| PyAV | BSD-3-Clause | ✅ | FFmpeg 바인딩 |
| imageio / imageio-ffmpeg | BSD-2 | ✅ | 래퍼 |
| **FFmpeg (LGPL 빌드)** | LGPL 2.1+ | ✅ | 동적 링크 + 소스 고지 |
| **libx264 (H.264 인코더)** | **GPL** | ⚠ | **GPL 코덱** → 상업 배포 시 앱 전체 GPL 강제. 개인/로컬 사용은 무방 |
| Pillow / numpy | HPND / BSD | ✅ | — |
| torch / transformers | BSD / Apache 2.0 | ✅ | — |
| PySide6 | LGPLv3 | ✅ | 동적 링크 시 준수 |

### 결론 / 정책 (확정)
- **기본 업스케일러 = SwinIR/Swin2SR(Apache 2.0)**. Real-ESRGAN 은 옵션으로만, 가중치가 DIV2K(학술 전용) 학습이라 **상업 배포 시 기본 비활성 + 경고**.
- **인코딩**: 개인·로컬 사용은 libx264(H.264, 최고 호환성)를 기본으로 둔다. **상업 배포 시에는** ① x264 상용 라이선스 구매 또는 ② libx264 회피(VP9 등) + H.264/HEVC 특허(MPEG-LA/HEVC Advance) 별도 검토 — **상업화 결정 시 법무 재검토 항목**.
- **데이터 계보 주의**: 학술 전용 데이터(DIV2K)로 학습한 SR 가중치의 상업 사용 가부는 법적으로 **미확정 영역**이며 대부분의 SR 모델이 해당된다. SwinIR 은 코드가 Apache 2.0 으로 더 투명해 **상대적 저위험**으로 채택하되, 완전한 상업 안전이 필요하면 상업 라이선스 데이터로 재학습/대체가 필요하다.
- 상업 안전 경로(코드·가중치 모두): SAM2 · Grounding DINO · SwinIR · PySceneDetect · PyAV · PySide6.

---

## 3. 의존성 (예상 requirements)

```
# AI / 추적
torch, torchvision
transformers>=5.9.0  # SAM2 이미지(Sam2Model)·비디오(Sam2VideoModel)·Grounding DINO·Swin2SR 모두 포함
# sam2               # 불필요 — transformers 5.9.0 단독으로 대체됨 (ADR 0001 보완 참조)
scenedetect          # PySceneDetect

# 업스케일
realesrgan, basicsr  # 옵션 (라이선스 확정 후)
# SwinIR/Swin2SR 은 transformers 로 사용 (별도 패키지 불필요)

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

| 시나리오 | 사양 | SAM2 추적 속도 | 평가 |
|---|---|---|---|
| **GPU (Colab T4)** | **NVIDIA T4 16GB** | **≈ 2.3 fps (실측, SAM2 video Tiny)** | 사용 가능(8초 클립 ≤~90초). 목표 10fps 미달 — SAM2 video 1024 인코딩 한계 |
| 권장 | NVIDIA GPU 6GB+ (RTX 3060급) | T4 대비 향상 기대(미측정) | 실사용 쾌적 예상 |
| **CPU (i7급, 14스레드)** | **≈ 0.10 fps (실측, PoC)** | **비실용적** — 6초 클립에 ~14분. 단발 이미지/초단편 보조용 |
| 업스케일 | GPU 권장 | CPU 시 분 단위 | CPU 경고 표시 |

> ⚠ **CPU 실측(2026-05-27, CPU 전용)**: SAM2.1-tiny 가 입력 해상도와 무관하게 내부 1024×1024 인코딩으로 **프레임당 ~10초**(0.10 fps)다. **GPU(CUDA)가 사실상 필수**이며, CPU 폴백은 강한 경고와 함께 단발/초단편으로 한정한다. 상세: [poc/REPORT.md](../poc/REPORT.md).
>
> ⚠ **GPU 실측(Colab T4)**: SAM2 video 추적은 **≈ 2.3 fps**다. 입력 해상도와 무관하게 내부 **1024×1024 인코딩**이 병목이라 당초 추정(10~15fps)에 크게 못 미친다(목표 10fps 미달). MVP 실용 하한으로는 동작하나(8초 클립 약 90초), 10fps 목표는 **fp16·경량 백엔드(EdgeTAM 등) 후속 과제**로 이관한다([RFP NFR-03·AC-06](RFP.md) 참조).

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
