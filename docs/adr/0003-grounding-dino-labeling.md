# ADR 0003 — 클래스 자동 검출·라벨링에 Grounding DINO 채택

- 상태: 채택
- 날짜: 2026-05-27

## 맥락
SAM2 는 class-agnostic 다. "인물 별로 라벨링"하려면 클래스/후보를 자동 검출해 사용자에게 보여줄 수단이 필요하다.

## 결정
**Grounding DINO**(`IDEA-Research/grounding-dino-tiny`, Apache 2.0)로 "person" 등 텍스트 프롬프트 기반 open-vocabulary 검출을 수행해 후보 bbox·라벨을 만들고, SAM2 가 마스크·추적을 담당한다(Grounded-SAM2 패턴). 사용자는 후보 중 하나를 **클릭으로 확정**.

## 대안
- 사용자 클릭만(SAM2 단독): 단순하나 자동 라벨 없음, 다중 인물 식별 보조 약함.
- YOLO detection: AGPL 부담.

## 결과
- 동일 프레임 다중 인물은 후보 라벨로 구분 후 단일 선택([RFP](../RFP.md) FR-05).
- 컷 재매칭 시 재검출에도 사용([ADR 0006](0006-shot-boundary-reid.md)).
