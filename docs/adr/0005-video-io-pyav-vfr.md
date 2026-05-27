# ADR 0005 — 비디오 입출력에 PyAV + ffprobe 채택 (VFR/PTS 대응)

- 상태: 채택
- 날짜: 2026-05-27

## 맥락
뮤직비디오·직캠은 가변 프레임레이트(VFR)인 경우가 있다. OpenCV `VideoCapture` 는 VFR 에서 프레임 인덱스와 실제 타임스탬프가 어긋나, 타임라인 구간 지정·오디오 동기·출력 타이밍에 오차가 생긴다.

## 결정
- **ffprobe** 로 메타데이터(코덱·해상도·fps·VFR 여부)를 먼저 추출.
- **PyAV**(FFmpeg 바인딩, BSD-3)로 **PTS 기반 정확 시크·구간 스트리밍 디코드**.
- OpenCV 는 보조(간단한 이미지 처리)로만.

## 대안
- OpenCV 단독: VFR 부정확.
- decord: 빠르나 설치 복잡 → 추후 성능 옵션으로 고려.

## 결과
- 색공간 정규화(BGR/limited→RGB BT.709)와 결합([데이터플로우](../data-flow.md) §2).
- 선택 구간만 디코드해 메모리 보호([아키텍처](../architecture.md) §4).
- 오디오 PTS 동기 검증은 [PoC](../poc-plan.md) H4.
