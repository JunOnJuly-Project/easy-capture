# ADR 0006 — 샷 경계 감지 + 컷 넘어 재추적

- 상태: 채택 (PoC 로 검증)
- 날짜: 2026-05-27

## 맥락
뮤직비디오는 1~2초마다 샷이 바뀐다. 컷이 바뀌면 SAM2 가 같은 프레임 구조에서 대상을 잃는다. 컷을 넘어 **같은 인물을 자동 재추적**하지 못하면 실사용 가치가 급감한다(아이돌 덕후 페르소나 치명적 지적).

## 결정
- **PySceneDetect**(BSD-3)로 컷 경계를 감지.
- 컷 직후 Grounding DINO 재검출 → 후보별 **재매칭 점수** 계산:
  `score = w_pos·pos_sim(prev_bbox, cand_bbox) + w_cls·cls_sim(prev_feat, cand_feat)`
  (기본 `w_pos=0.7`, `w_cls=0.3`)
- `max(score) ≥ threshold`(기본 0.5)면 SAM2 재초기화·object_id 유지, 미만이면 사용자 확인 또는 수동 교정 유도.

## 대안
- 전용 re-ID 네트워크: 정확하나 무겁고 학습/라이선스 부담 → v1.1.
- pose matching: 보조 신호로 후순위.

## 결과 / 리스크
- **가장 큰 제품 리스크** → [PoC](../poc-plan.md) H2 에서 우선 검증, 가중치·임계값 보정.
- 실패 시 수동 교정(`core/correction`)이 안전망([use-flow](../use-flow.md) §5).
- 미달 시 본 ADR 갱신(전략 재설계).
