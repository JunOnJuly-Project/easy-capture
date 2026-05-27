# easy-capture — 데이터 플로우

> 최종 업데이트: 2026-05-27 · 관련: [아키텍처](architecture.md) · [유즈플로우](use-flow.md) · [RFP](RFP.md)

---

## 1. 전체 파이프라인 (GIF 모드 기준)

```
[영상 파일]
   │ 1. ffprobe — 메타데이터(코덱·해상도·fps·VFR 여부) 추출
   ▼
[입력 검증] ── 권장 미만 → 경고
   │ 2. PyAV 디코드 (선택 구간만 PTS 기반 스트리밍, 전체 로드 금지)
   ▼
[프레임(BGR/limited 등)]
   │ 3. 색공간 정규화 → RGB BT.709 full
   ▼
[정규화 프레임]
   │ 4. PySceneDetect — 샷 경계(컷) 인덱스 산출
   │ 5. (선택 시점) Grounding DINO 검출 → 후보 bbox·label
   ▼
[검출 후보] ──사용자 클릭──▶ [선택 객체 (object_id, label, init bbox)]
   │ 6. SAM2 image predictor — 선택 프레임 마스크
   │ 7. SAM2 video predictor — 후속 프레임 마스크 전파
   │     └─ 컷 경계에서 재매칭(§3), occlusion 시 대기(§4)
   ▼
[프레임별 mask] → centroid·bbox 산출
   │ 8. 떨림 완화 (N-프레임 이동평균)
   │ 9. 크롭 (centroid 중심 W×H, 경계 클램프, 짝수 정렬, LANCZOS4 리사이즈)
   ▼
[크롭 프레임 시퀀스]
   │ 10. (옵션) 업스케일 (Real-ESRGAN / SwinIR, 타일링)
   ▼
[출력 버퍼] ── 갭 채우기 정책 적용(§4)
   │ 11. 인코딩
   │     ├─ GIF: 8비트 인덱스(최대 256색) 동적 팔레트 양자화 + 디더링 + 크기 예측
   │     └─ MP4: libx264 yuv420p + VFR→CFR(지정 FPS) 정규화 + 원본 오디오 PTS 동기 mux(ffmpeg)
   ▼
[GIF / MP4 파일]
```

이미지 모드는 4·7·11 단계를 생략하고 단일 프레임 크롭→(옵션)업스케일→PNG/JPG 인코딩으로 끝난다.

---

## 2. 색공간 경로 (왜곡 방지)

```
디코드 색 (BGR / BT.601 또는 709 / limited range)
  → RGB BT.709 full range 정규화 (모든 모델·처리의 공통 색공간)
  → 출력 시 색공간 태깅:
       PNG/JPG : sRGB
       GIF     : 8비트 팔레트 양자화. BT.709 → sRGB 톤매핑(색역 차로 약간의 색감 손실 예상, 8비트 인덱스 한계와 함께 불가피) — v1.1 에서 광역 팔레트 옵션 검토
       MP4     : libx264 출력 스트림에 colorspace/colorprimaries/transfer = bt709 메타데이터 명시 기록(미지정 시 플레이어별 해석 차 방지)
```

### VFR → CFR 변환 (MP4)
가변 프레임레이트 소스를 MP4 로 낼 때는 **지정 FPS 의 CFR 로 정규화**한다. 각 출력 프레임은 원본 PTS 기준으로 선택·배치하고, 오디오는 동일 PTS 구간을 동기 mux 한다. 동기 어긋남이 1프레임을 초과하면 무음 폴백([error-handling.md](error-handling.md) §5). 음악 박자 정합성은 [PoC](poc-plan.md) H4 로 검증.

---

## 3. 샷 경계 재매칭 데이터 경로

```
컷 경계 프레임 도달
  → Grounding DINO 재검출 → 후보 집합 {c_i: bbox_i, label_i, feat_i}
  → 각 후보 재매칭 점수:
       score_i = w_pos · pos_sim(prev_bbox, bbox_i)
               + w_cls · cls_sim(prev_feat, feat_i)
       (기본 w_pos=0.7, w_cls=0.3)
  → max(score_i) ≥ threshold(기본 0.5)
       ├─ 예 → 해당 후보로 SAM2 재초기화, object_id 유지
       └─ 아니오 → 사용자 확인 요청 (UI)
```
가중치·임계값은 [poc-plan.md](poc-plan.md)에서 보정.

---

## 4. occlusion / 갭 채우기 데이터 경로

```
프레임별 마스크 신뢰도 검사
  ├─ 유효 → 정상 크롭·출력
  └─ 소실(빈/저신뢰 마스크) → wait_counter++
        ├─ wait_counter ≤ 대기시간 & 재등장 → 재동기화, counter 리셋
        └─ wait_counter > 대기시간 → 구간 종료

갭 구간(소실~재등장) 출력 정책:
  ① 컷       : 갭 프레임을 출력 버퍼에서 제외 (시간 점프)
  ② 배경 계속 : last_crop_box 고정, 해당 위치 배경 프레임을 그대로 출력
  ③ 프리즈   : last_crop_frame 을 갭 길이만큼 복제
```

---

## 5. 핵심 데이터 구조 (개념)

```
Session
 ├─ mode: "image" | "gif"
 ├─ source: { path, meta(fps, vfr, resolution, codec) }
 ├─ decode_cache: 구간 프레임 버퍼(상한 있음, 멤버 순차 처리 시 재사용)
 ├─ shot_boundaries: [frame_idx, ...]
 └─ tracks: [ Track ]

Track
 ├─ object_id, label
 ├─ frames: { frame_idx → { mask, bbox, centroid, confidence } }
 ├─ segments: [성공 구간 / 갭 구간]  (수동 교정 시 분기·재사용)
 └─ crop_cfg: { w, h, aspect_lock, smooth_N }

ExportConfig
 ├─ format: png|jpg|gif|mp4
 ├─ gap_policy: cut|background|freeze
 ├─ gif: { fps, palette_quality, wait_time }
 ├─ mp4: { audio: bool, yuv420p }
 └─ upscale: { enabled, model, scale }
```

---

## 6. 캐시 / 메모리 흐름

- **모델 가중치**: HuggingFace 캐시 디렉터리에 1회 다운로드 후 재사용.
- **디코드 프레임**: 선택 구간만 버퍼링(상한 초과 시 디스크 스풀 또는 재디코드). 멤버 순차 처리 시 동일 구간 버퍼 재사용.
- **VRAM 부족**: 자동 해상도 다운스케일 후 재시도 → 그래도 부족하면 CPU 폴백/타일링.
