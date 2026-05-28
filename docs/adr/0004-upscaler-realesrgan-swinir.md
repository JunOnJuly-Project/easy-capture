# ADR 0004 — 업스케일러로 Real-ESRGAN + SwinIR 2종 제공

- 상태: 채택 (라이선스 확정 2026-05-27 — 기본 = SwinIR)
- 날짜: 2026-05-27

## 맥락
저화질 영상/크롭 결과의 화질을 올리는 초해상도 옵션이 필요하다. 실사·애니(MV) 특성이 다양하다.

## 결정
두 가지를 제공하되 **기본값은 SwinIR/Swin2SR(Apache 2.0)** 로 한다(라이선스 확정 결과).
- **SwinIR/Swin2SR (기본)**: transformers 내장, 의존성 간결, 코드 Apache 2.0 → 상대적 저위험.
- **Real-ESRGAN (옵션)**: 애니 전용 모델(x4plus-anime) 강점이나 가중치가 DIV2K(학술 전용) 학습 → **상업 배포 시 기본 비활성 + 경고**.

## 대안
- SUPIR/SD x4 upscaler: 고품질이나 비상업/무거움 → 제외.

## 결과 (라이선스 확정)
- **확정**(공식 repo 확인, [resources.md](../resources.md) §2): SwinIR/Swin2SR·Real-ESRGAN 코드는 상업 가능(Apache/BSD-3), basicsr Apache 2.0. **Real-ESRGAN 가중치는 DIV2K(학술 전용) 학습**이라 상업 배포 리스크 → **기본 = SwinIR**, Real-ESRGAN 은 옵션·경고.
- 프레임 단독 처리로 인한 플리커는 고지, temporal smoothing 은 v1.1([error-handling.md](../error-handling.md) §4).
- 두 모델은 `core/upscale` 공통 인터페이스로 추상화([ADR 0007](0007-cpu-dev-strategy.md) 백엔드 추상화와 정합).
- **업스케일 추상화 구현**: `core/upscale` 공통 인터페이스는 [ADR 0009](0009-upscale-export-integration.md)에서 `UpscaleBackend` Protocol로 명문화·구현되었다. Swin2SR 구현체(`Swin2srUpscaleBackend`)는 `infra/swin2sr_upscale_backend.py`에 위치하며, `ImageCaptureUseCase.export`에 옵션 주입 방식으로 통합된다. 상세는 [ADR 0009](0009-upscale-export-integration.md) 참조.
