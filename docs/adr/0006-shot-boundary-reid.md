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

---

## 보완 (2026-05-28, 구현)

샷경계 재추적 슬라이스([계획서](../plans/video-shot-retrack.md)) 구현 단계에서 아래 사항이 구체화되었다.

### 재검출 백엔드 추상화

Grounding DINO 재검출을 [ADR 0012](0012-detection-backend.md) `DetectionBackend` Protocol로 추상화한다. `VideoSegmentationBackend`([ADR 0010](0010-video-segmentation-backend.md))와 책임이 다르고(마스크 전파 vs. 후보 bbox 열거) torch/transformers 의존을 core 밖으로 격리해야 하므로 별도 Protocol을 신설했다. 구현 위치: core 추상 `core/segmentation/detection_backend.py`, infra 구현 `infra/grounding_dino_backend.py`.

### 재매칭 판정 순수 함수

재매칭 판정을 `core/tracking/rematch.py`의 순수 함수로 구현한다.

- `REMATCH_THRESHOLD = 0.5`: 재매칭 통과 임계값 상수(Colab H2 실측 후 보정).
- `RematchResult(best_index, score, passed)`: 판정 결과 frozen dataclass.
- `select_best_match(prev_box, candidates, threshold)`: 후보 리스트에서 직전 bbox와 best 매칭 후보·점수·통과여부를 반환하는 순수 함수. `rematch_score`를 내부에서 재사용한다(DRY).

torch·UI 비의존이므로 통과/미달/다중후보/빈리스트 4케이스를 단위 테스트로 완전 검증한다.

### 수동 교정 유도의 2단계 분리

원문의 "수동 교정 유도"를 구현 현실에 맞게 명시적으로 단계화한다.

- **1단계(이번 슬라이스)**: 재매칭 미달 구간을 `TrackResult.needs_correction: list[bool]` 플래그로 표시하고, UI 상태바에 "재매칭 실패 구간 수 — 교정은 추후 지원" 안내를 띄운다. 추적은 중단하지 않고 직전 위치를 hold한다(gap_policy 경로 재사용).
- **2단계(후속 슬라이스)**: `needs_correction` 구간에서 사용자가 박스 드래그·재클릭으로 대상을 재지정하고 해당 샷부터 재추적하는 교정 UI(`core/correction`)를 구현한다. `TrackResult.cut_frames`·`needs_correction` 데이터가 이미 준비된 상태에서 진입한다.

### threshold·가중치 실보정

`REMATCH_THRESHOLD=0.5`·`w_pos=0.7`·`w_cls=0.3` 기본값은 PoC 위치 기반 추정치다. Colab H2 GPU 검증에서 컷 섞인 군무 MV 클립으로 재추적 성공률·오탐률을 실측한 뒤 상수 값을 보정한다. Protocol 인터페이스 변경 없이 상수만 갱신하며, 보정 결과는 PoC `REPORT.md` H2 칸에 기록하고 본 ADR을 재갱신한다. **GPU 블로커로 인해 현 기본값은 미검증 상태임을 명시한다.**
