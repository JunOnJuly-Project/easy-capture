# easy-capture PoC 리포트

> 실행일: 2026-05-27 · 브랜치: `feature/poc-core`
> 환경: Windows 11, Python 3.12.10, **CPU 전용**(torch 2.12.0+cpu, CUDA 미감지, 14 threads), transformers 5.9.0
> 테스트: 합성 클립 `test-data/synthetic.mp4` (640×360, 15fps, 6s/90frame, 컷@3.0s, 빨강원 소실 62~72f, 440Hz 오디오)

관련: [PoC 계획](../docs/poc-plan.md) · [리소스](../docs/resources.md)

---

## 요약 (Go/No-Go)

**조건부 Go.** 핵심 파이프라인(SAM2 추적·컷 감지·오디오 동기)은 **기술적으로 모두 동작**한다. 단, **CPU 전용 환경에서는 SAM2 추적이 비실용적으로 느려(≈0.10 fps)**, 실사용에는 **GPU(CUDA)가 사실상 필수**다. → MVP 진행은 가능하되, **타깃 실행 환경을 GPU 로 전제**하고 CPU 는 "단발 이미지/초단편" 보조로 한정 권장.

---

## 가설별 결과

| 가설 | 결과 | 측정/근거 |
|---|---|---|
| **H1** SAM2 video predictor 설치·추적 | ✅ **동작** / ⚠ CPU 속도 미달 | transformers `Sam2VideoModel`+`Sam2VideoProcessor` 로 설치(별도 sam2 패키지 불필요). 클릭 추적 **90/90 프레임(100%)** 유지(컷 통과 포함). **전파 0.10 fps(프레임당 ~9.6s), 6초 클립에 866s** |
| **H2** 컷 감지 + 재매칭 | ✅ **동작**(기계 검증) | PySceneDetect 컷 정확 감지(3.00s=frame45, 2 scenes). 재매칭 점수 near=0.747≥0.5>far=0.000. *실제 인물 identity 재매칭은 실영상+Grounding DINO 필요(미검증)* |
| **H3** CPU 실사용성 | ❌ **비실용적** | H1 의 0.10 fps. 5초·30fps(150프레임) 환산 ≈ **25분**. `resources.md` 의 "CPU 1~2fps" 추정은 **10~20배 낙관**이었음 |
| **H4** PyAV 오디오 동기 mux | ✅ **동작** | 구간 크롭(300×300 짝수·yuv420p·bt709) + 오디오 mux. **video/audio 길이차 0.0ms**. VFR 감지 로직 동작(합성은 CFR) |

---

## 핵심 발견

1. **transformers 5.9.0 만으로 SAM2(이미지+비디오)·Grounding DINO 사용 가능** → 설치 난이도·리스크 대폭 감소(공식 `sam2` 패키지 불필요). ADR 0001 보강 가능.
2. **SAM2 는 내부적으로 1024×1024 로 인코딩** → 입력 해상도와 무관하게 프레임당 비용이 크고, CPU 에서 ~10s/frame. **GPU 필수**의 결정적 근거.
3. **컷 감지·오디오 동기·크롭 인코딩은 CPU 에서도 충분히 빠름** → 무거운 건 오직 SAM2 추론.
4. 재매칭 점수식(설계 #4)·짝수정렬·색공간 태깅은 의도대로 작동.

---

## 미검증 / 다음 검증 (실영상 필요)

- **H2 실제 identity 재매칭**: 실제 군무 MV 클립 + Grounding DINO 재검출로 컷 넘어 동일인 매칭 정확도(AC-03 ≥70%) 측정.
- **H1 실영상 추적 유지율(AC-01 ≥80%)**: 군무·교차 장면 identity switch 실측.
- **GPU 벤치마크**: CUDA 환경에서 fps 재측정(AC-06 ≥10fps 검증).
- **VFR 실파일 오디오 동기**: 실제 가변 프레임 소스로 AC-08 재확인.

> 위 검증에는 **권리상 사용 가능한 짧은 MV 클립**과 가능하면 **CUDA GPU 환경**이 필요하다.

---

## 권고 (계획 반영 사항)

1. `docs/resources.md` §5 **CPU 성능 수치 정정**(0.10 fps 실측) + "GPU 사실상 필수" 명시 — **반영 완료**.
2. `docs/RFP.md` NFR-03 CPU KPI 재조정 검토(현 "1~2fps" → 측정 기반).
3. 경량화 옵션 조사(ONNX/quantization, MobileSAM/EdgeSAM 등)로 CPU 보조 경로 가능성 검토(후순위).
4. MVP 구현은 **GPU 전제**로 진행하되, 디바이스 자동 감지 시 CPU 에서는 "단발 이미지/초단편 + 강한 경고"로 UX 한정.
