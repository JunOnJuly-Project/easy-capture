# ADR 0001 — 비디오 세그멘테이션·추적에 SAM 2.1 채택

- 상태: 채택
- 날짜: 2026-05-27

## 맥락
사용자가 클릭한 오브젝트를 영상 후속 프레임에 걸쳐 추적하고, 일시 소실 후 재등장 시 재추적해야 한다.

## 결정
**Meta SAM 2.1** (`facebook/sam2.1-hiera-*`)의 **공식 `sam2` 패키지 video predictor** 를 사용한다. 클릭 프롬프트→마스크→후속 프레임 자동 전파, 메모리 모듈 기반 occlusion 처리가 내장되어 요구사항에 정확히 부합한다. 라이선스 Apache 2.0 으로 상업 사용 안전.

## 대안
- YOLOv8/11-seg + DeepSORT: 빠르나 AGPL 라이선스 부담, 클릭 기반 promptable 아님.
- Mask2Former: 단일 프레임 panoptic 위주, 비디오 추적 비내장.

## 결과
- 단일 프레임 세그는 SAM2 image predictor, 비디오 추적은 video predictor 로 분기.
- 클래스 라벨이 없으므로 검출은 [ADR 0003](0003-grounding-dino-labeling.md) 조합.
- 설치 성숙도·실성능은 [PoC](../poc-plan.md) H1 에서 검증.
