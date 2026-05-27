# ADR 0004 — 업스케일러로 Real-ESRGAN + SwinIR 2종 제공

- 상태: 채택 (라이선스 단서 있음)
- 날짜: 2026-05-27

## 맥락
저화질 영상/크롭 결과의 화질을 올리는 초해상도 옵션이 필요하다. 실사·애니(MV) 특성이 다양하다.

## 결정
두 가지를 **설정에서 선택**하도록 제공한다.
- **Real-ESRGAN**: 빠르고 안정적, **애니 전용 모델(x4plus-anime)** 보유 → 뮤직비디오에 적합.
- **SwinIR/Swin2SR**: transformers 내장으로 의존성 간결, 실사 전용.

## 대안
- SUPIR/SD x4 upscaler: 고품질이나 비상업/무거움 → 제외.

## 결과 / 단서
- **라이선스**: Real-ESRGAN(코드 BSD-3)·basicsr(Apache 2.0)는 상업 안전으로 보이나 웹 자료 간 혼동 존재 → **배포 전 공식 repo 로 확정**([resources.md](../resources.md) §2). 확정 전에는 **SwinIR 을 기본**, Real-ESRGAN 은 사용자 선택.
- 프레임 단독 처리로 인한 플리커는 고지, temporal smoothing 은 v1.1([error-handling.md](../error-handling.md) §4).
- 두 모델은 `core/upscale` 공통 인터페이스로 추상화.
