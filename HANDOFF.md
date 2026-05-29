# easy-capture 핸드오프 문서

> 다른 PC / 다른 세션에서 이 프로젝트를 **끊김 없이 이어서 진행**하기 위한 안내서.
> 스키마 버전: v2
> 최종 업데이트: 2026-05-29 (✅ 비디오 Colab GPU 실검증 완료 — 추적·크롭·슬로우·트림·루프 전부 정상 동작. **GPU 블로커 해소**. 다음: 멀티샷 재추적 검증 / 비디오 후속)

---

## 1. 프로젝트 개요

**easy-capture** — 동영상에서 오브젝트를 추적해 중심 크롭으로 짤/움짤을 만드는 로컬 데스크톱 프로그램.

- **대상**: 직캠·움짤 제작 팬 콘텐츠 제작자(Primary), 일반 덕후(Secondary)
- **스택**: Python 3.10+ · PySide6 · SAM 2.1 · Grounding DINO · PySceneDetect · PyAV · Real-ESRGAN/SwinIR
- **방법론**: 기획 단계는 페르소나(영상전문가/PM/덕후) 검토–수정–컨펌 루프. 구현 단계는 `/develop` 팀 파이프라인.

상세 계획서: [`docs/plans/easy-capture-plan.md`](docs/plans/easy-capture-plan.md)

---

## 2. 다른 PC에서 이어 받기

### 2-1. 필수 도구

| 도구 | 버전 | 용도 |
|---|---|---|
| Git | 2.40+ | 소스 클론 |
| Python | 3.10+ | (구현 단계) 런타임 |
| FFmpeg | 최신 LGPL 빌드 | (구현 단계) 디코드/인코드 |

### 2-2. 클론 및 시크릿

```bash
git clone <repo-url>
cd easy-capture
cp .env.example .env
```

### 2-3. 실행

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"   # PySide6 등 포함
python -m easy_capture                    # 시작 화면(모드 선택)
```

### 2-4. 동작 확인 (smoke test)

```bash
.venv\Scripts\pytest -q     # 순수 로직 단위 테스트 (현재 494개)
```

GPU 비디오 추적 검증은 `poc/colab/` 노트북(Colab GPU).

**이미지 모드 수동 스모크**

```bash
python -m easy_capture        # 모드 선택 → 이미지 선택
# 이미지 또는 영상 파일 열기 → 캔버스에 프레임 표시
# 피사체 클릭
# 첫 클릭 시 SAM2 모델 1회 다운로드(facebook/sam2.1-hiera-tiny, 수백 MB) + CPU 추론 1~3s
# 크롭 박스 생성 → 저장(PNG/JPG)
```

주의사항:
- SAM2 첫 클릭은 모델 다운로드(수백 MB) + CPU 추론 지연이 있으나, 워커 스레드로 실행되므로 UI는 멈추지 않는다.
- 빈 배경 클릭 시 "대상을 인식하지 못했어요. 다시 클릭해 주세요" 안내가 표시된다.
- 이후 클릭은 모델이 메모리에 로드된 상태이므로 추론만 수행(빠름).
- **종횡비/크기 조정**은 재세그 없이 즉시 갱신된다. **업스케일** 체크 후 저장하면 워커가 Swin2SR(첫 1회 모델 다운로드)로 크롭을 확대해 저장한다. 저장 이미지 크기 = 크롭 크기 × 배율(x2/x4)임을 반드시 확인(Swin2SR processor 8배수 패딩 보정 검증).

---

## 3. 현재 진행 상태

### 현재 브랜치
`main` (전 슬라이스 + 슬로우모션 S1~S6 + **트림/루프 머지** — 494 테스트 통과. 데스크톱·노트북에서 하이라이트 트림 + 구간 슬로우/패스트 + 무한루프 GIF/MP4 생성). **새 슬라이스는 `main`에서 분기**(선형 누적 안티패턴 중단). 다음: 비디오 후속(수동 교정 UI·오디오 동기) / 멀티샷 재추적 GPU 검증. **백로그**: 🔴 CUT/FREEZE×트림·배속 좌표계 정합(ADR 0013 2단계 인덱싱 미구현 — 현재 BACKGROUND에서만 정합).

### 완료 ✅

| Phase | 항목 | 상태 |
|---|---|---|
| 기획 | 계획서 v2.1 (페르소나 3인 전원 컨펌) | ✅ |
| 기획 | 문서 세트: RFP·use-flow·wireframes·data-flow·architecture·resources·poc-plan·error-handling·ADR 0001~0006 | ✅ |
| 기획 | **문서 세트 페르소나 검토–수정–컨펌 루프 (영상전문가·PM·덕후 3인 전원 컨펌)** | ✅ |
| PoC | **`feature/poc-core` H1~H4 코드 검증 (조건부 Go)** — 상세 `poc/REPORT.md` | ✅ |
| 조사 | **라이선스 확정**(resources.md §2): 기본 업스케일러=SwinIR, Real-ESRGAN 가중치 DIV2K 리스크, libx264=GPL(상업 시 주의) | ✅ |
| 조사 | **CPU 개발 전략 확정**(ADR 0007): 이중 경로 — 이미지=CPU, 비디오=클라우드 GPU | ✅ |
| 구현 | **스캐폴딩**(`feature/app/scaffolding`): 패키지 구조(src/easy_capture, core/infra/ui), core 순수 로직(crop·gap_policy·rematch·device·backend IF), PySide6 모드선택 셸, **테스트 20개 통과** | ✅ |
| 구현 | **이미지 모드 첫 수직 슬라이스**(`feature/image/capture-slice`): end-to-end happy path — 파일→프레임→클릭→SAM2(CPU)→크롭→PNG/JPG. 신규 모듈: `core/export`(crop_array/save_image, Pillow), `infra/video_io`(FrameSource Protocol·FrameMeta·open_source, 이미지=Pillow·영상=PyAV 첫프레임), `infra/sam2_image_backend`(Sam2ImageBackend, transformers 5.9.0 Sam2Model/Sam2Processor, 지연 로드, CPU), `app/image_capture`(ImageCaptureUseCase·CropRequest·EmptyMaskError), `app/router`(AppRouter 조립 루트), `ui/coords`(좌표 변환 순수함수), `ui/frame_canvas`, `ui/main_window`(워커 스레드 비블로킹). **테스트 71개 통과**(기존 20 포함). ADR 0008(app 유스케이스 레이어) 추가. 코드 리뷰 [중요] 4건 전원 반영. | ✅ |
| 구현 | **이미지 모드 크롭 UX 확장**(`feature/image/crop-ux`): 종횡비 프리셋 UI(자유/1:1/9:16/16:9) + 크롭 크기 슬라이더 + 마스크 오버레이 표시. **핵심: 세그(무거움)/박스계산(가벼움) 분리** — `ImageCaptureUseCase.segment`(SAM2 1회, 워커)/`compute_box`(순수, 재세그 없음), `SegmentResult`·`BoxParams`. `ui/sizing`(crop_ratio_to_size), `frame_canvas.mask_to_rgba`(numpy 벡터화, 픽셀 루프 제거). **테스트 137개 통과**(재세그 카운터 회귀 가드 포함). 코드 리뷰 [중요] 2건 반영. | ✅ |
| 구현 | **이미지 모드 업스케일(SwinIR)**(`feature/image/upscale`): 크롭 결과 초해상도 옵션. `core/upscale`(UpscaleBackend Protocol + `reconstruction_to_rgb_uint8` 순수 정규화, torch 비의존), `infra/swin2sr_upscale_backend`(Swin2SR, transformers, 지연 로드, processor 8배수 패딩 보정 재크롭), `export(upscaler=None)` 옵션 주입(무회귀), router `UPSCALE_MODELS` 카탈로그, UI 업스케일 체크박스·배율 콤보 + `_UpscaleSaveWorker`(백그라운드). **테스트 166개 통과**. ADR 0009 추가. 코드 리뷰 [중요] 3건 반영. | ✅ |
| 구현 | **비디오 모드 첫 수직 슬라이스(코드)**(`feature/video/tracking-slice`): 단일 샷 구간 추적→크롭→GIF/MP4. `core/segmentation/video_backend`(VideoSegmentationBackend Protocol, ISP·opaque session, ADR 0010), `infra/sam2_video_backend`(SAM2 video, 지연 로드, PoC 패턴), `app/video_capture`(track 무거움/compute_boxes 순수 분리, 고정 box size 불변식, propagate 1회 가드), `core/export/video_export`(GIF/MP4 imageio, ADR 0011), `infra/video_io` 구간 추출, `ui/video_window`(_TrackWorker/_ExportWorker). **테스트 216개 통과**. ADR 0010·0011 추가. 코드 리뷰 [중요] 3건 반영. **🔴 SAM2 video 실추론은 Colab GPU 검증 미완(CPU 코드·Fake 테스트만 완료)**. | ⏳ |
| 구현 | **비디오 샷경계 재추적**(`feature/video/shot-retrack`): 컷 넘어 동일 인물 자동 재추적(ADR 0006). `core/crop.bbox_of_mask`, `core/tracking`(select_best_match·RematchResult·REMATCH_THRESHOLD·split_into_shots), `core/segmentation/detection_backend`(DetectionBackend Protocol stateless, ADR 0012), `infra/shot_detect`(PySceneDetect 경로 기반·**CPU 테스트 검증**), `infra/grounding_dino_backend`(재검출, 지연 로드), `app/video_capture.track(detector,cut_frames)`(샷 분할→경계 재검출→재매칭→재초기화·objid 유지/미달 needs_correction; propagate==샷수·detect==컷수 가드), router/UI 배선. **테스트 280개 통과**. ADR 0012 추가·0006 보완. 코드 리뷰 [치명적]1·[중요]3 반영(scenedetect API 2건·Grounding DINO 키워드·배선). **🔴 SAM2·Grounding DINO 실추론은 Colab GPU 검증 미완**. | ⏳ |
| 구현 | **occlusion gap 정책 UI**(`feature/video/gap-policy-ui`): 추적 끊긴(occlusion) 프레임 처리 정책(배경 유지/컷/정지)을 UI에서 선택. `video_window` 갭 콤보 + export `gap_policy` 전달(백엔드 `build_output_indices` 기존 재사용). **테스트 284개 통과**(매핑 가드). gap_policy는 순수 로직 → **CPU 검증 완료(GPU 무관)**. | ✅ |
| 구현 | **비디오 크롭 튜닝**(`feature/video/crop-tuning`, GPU 실검증 피드백): 크롭이 ① 피사체를 잘라먹고 ② 흔들리고 ③ GIF가 느리던 문제 수정. `compute_boxes`를 **마스크 bbox 최대×padding 자동 크기**(고정 320 제거 → 잘림 해소) + **bbox 중심**(centroid→자세 흔들림 완화), `_expand_to_aspect`로 종횡비 확대(잘림 방지). GIF `duration` 초→ms(imageio 2.28+, 재생속도 정상·조절). `subject_padding`·`smooth_window`·`GIF_FPS` 노출. **테스트 300개 통과**. **앱 검증 노트북**(`poc/colab/easy_capture_app_verify.ipynb`)으로 GPU 실검증 경로 확보(추적 OK 확인됨). | ✅ |
| 기획 | **슬로우모션(타임리맵) 계획**(`feature/video/speed-remap`): 구간별 가변 재생속도(슬로우/패스트, 다중 구간) 설계 — `docs/plans/video-speed-remap-plan.md`. core 타임리맵 순수 로직(`PlaybackSchedule` 이중 표현) + GIF(per-frame duration·10ms 클램프) + MP4(프레임 복제) + UI(구간 테이블·미리보기→버튼). **페르소나 3인(영상전문가·PM·덕후) 2라운드 전원 컨펌**. ADR 0013 후보. MVP=프레임 복제(보간·오디오·트림+루프는 후속). | ✅ |
| 정리 | **전 슬라이스 main baseline 머지 + 전체 정합성 정리**: 9개 feature 브랜치(선형 56커밋) → main fast-forward(충돌 0, 300 통과·push). 코드 경계(FrameSource→`core/source`·ui→infra `detect_cuts` 흡수), 문서 정합성(architecture·ADR 0001/0004/0007/0008 보완·plan·README·resources), 이미지 SAM2+업스케일 **실모델 CPU 스모크**(centroid 클릭 일치·2x 정확). 다관점(문서/코드/브랜치) 검토 + 페르소나 컨펌 기반. | ✅ |

### ✅ GPU 블로커 해소 (2026-05-29)
- SAM2 추적이 **CPU ≈0.10 fps**라 GPU 필수인 사실은 유지되나, **Colab GPU(T4)로 앱 검증 노트북 실행 → 추적·크롭·슬로우모션·트림·루프 전부 정상 동작 확인**(사용자 검증). 개발/검증 경로 = Colab GPU 확립, 실사용은 GPU 전제. → **비디오 모드 진행 블로커 해소.**
- **측정(단일샷 200f)**: ✅ AC-01 유지율 **100%**(200/200). ❌ AC-06 **2.3 fps**(목표 10 미달 — SAM2 video T4 현실 한계, fp16·경량 백엔드 개선 후속). (잔여) 컷 섞인 **멀티샷 재추적** 미검증(이번 컷 []).

### 미완료 (다음 작업 순서) ⏳
1. **✅ 슬로우모션(타임리맵) 구현 — 6 Story 전체 완료** (계획서 `docs/plans/video-speed-remap-plan.md`):
   - ✅ **ADR 0013 + Story 1**(`feature/timing/timeremap-core`): `core/timing/timeremap.py` 순수 로직 — `SpeedSegment`·`normalize_segments`·`PlaybackSchedule`(frame_indices·durations_ms tuple)·`build_playback_schedule`·`schedule_to_cfr_indices`·`clamp_durations_for_gif`. 테스트 53개(전체 355 통과). 코드 리뷰 [중요] 2 반영(CFR 잔여 가드·tuple화). segments=() 무회귀.
   - ✅ **Story 2**(`feature/export/gif-variable-duration`): `VideoExportConfig.segments` + `_encode_gif` 프레임별 duration(`build_playback_schedule`→`clamp_durations_for_gif` 연결, 10ms 클램프, loop=0 유지). 테스트 +7(전체 362). segments=() 무회귀. 리뷰 [중요] 0.
   - ✅ **Story 3**(`feature/export/mp4-frame-replication`): MP4 `_resolve_mp4_frames` — `schedule_to_cfr_indices`로 슬로우=프레임 복제·패스트=드롭(CFR). 테스트 +5(전체 367, ffmpeg 실인코딩). 단일 패스트 가드([중요1]) 실검증. segments=() 무회귀. 리뷰 [중요] 0.
   - ✅ **Story 4**(`feature/app/export-timeremap`): `estimate_output_frame_count(n_selected, segments, fps) → int` 순수 헬퍼 구현(`core/timing/timeremap.py`). `build_playback_schedule` + `schedule_to_cfr_indices` 위임으로 실제 MP4 출력 프레임 수 사전계산. 4 xfail → 379 passed. `core.timing.__init__` 공개 심볼 추가. export segments end-to-end는 기존 encode_frames 위임 구조(S2/S3)가 이미 처리. 무회귀.
   - ✅ **Story 5**(`feature/ui/speed-segment-table`): `ui/segment_table.py` — `SegmentTableWidget`(구간 테이블·배속 콤보·미리보기→구간 버튼) + 순수(`rows_to_segments`·`dynamic_fast_cap`·`PRESET_FACTORS`). video_window 통합: export에 segments 연결·GIF 클램프/폭증 경고·동적 패스트 상한 콤보. 테스트 +49(전체 430). 리뷰 [중요] 2 반영(좌표계 상대 정합·dead code 연결). `dynamic_fast_cap` 공식 `50/base_fps` 정정(계획서도).
   - ✅ **Story 6**(`feature/poc/notebook-timeremap`): 노트북에 셀 9.5(SpeedSegment 구간 + 스케줄 요약·클램프/폭증 경고) + 셀 10 segments 연결. 데스크톱과 동일 core 함수(재현성). API 정합 스모크 통과.
   - 🎉 **슬로우모션 6 Story 전체 완성** — 구간별 가변 재생속도(다중 구간) core→GIF/MP4→export→데스크톱 UI→노트북.
   - 📌 백로그: 미리보기 스크럽(prev/next — 임의 프레임 구간 지정, 페르소나 [치명적·UX] 잔여), 클램프 경고 확인 다이얼로그, segment_logic 물리 분리.
   - 📌 백로그(리뷰 [제안]): Story 3 가드 테스트 주석 "1프레임"→"1~2프레임" 정정, GIF fallback `1000/12.0` → `_DEFAULT_FPS` 상수화(GIF/MP4 일관).
   - 잔여 디테일: 미리보기 프레임 스크럽(prev/next, Story 5), 저fps GIF 패스트 경고 문구, [중요1] CFR 단일 패스트 가드는 Story 3에서 실인코딩 검증.
2. **✅ 비디오 Colab GPU 검증** (2026-05-29): 앱 검증 노트북(`poc/colab/easy_capture_app_verify.ipynb`) GPU(T4) 실행으로 **추적(SAM2 video)·크롭 정합·슬로우모션·트림·루프 전부 정상 동작 확인**(사용자). → 잔여 ②: 컷 섞인 **멀티샷 재추적**(재매칭 threshold 0.5)·AC-01(유지율)/AC-06(fps) 수치 정량화는 멀티샷 클립으로 추가 검증 권장.
3. **이미지 모드 GUI 수동 스모크** (선택): `python -m easy_capture` → 이미지 → 클릭 → 종횡비/크기 → (업스케일) 저장 실사용 확인. (실모델 코드 스모크는 완료 — API·centroid·업스케일 정합 확인됨)
4. **비디오 후속 슬라이스**: 수동 교정 UI(needs_correction → `core/correction`) → 오디오 동기(H4) → 업스케일 결합 → 트림+슬로우+루프 → 타임라인 고도화. (샷경계 재추적·occlusion gap UI 완료)

> ✅ 전 슬라이스 main fast-forward 머지 완료(2026-05-28). 이후 **모든 새 슬라이스는 main에서 분기**(끝에 선형 누적 안티패턴 중단).

### PoC 핵심 결과 (요약)
- SAM2(이미지+비디오)·Grounding DINO 는 **transformers 5.9.0 만으로** 사용 가능(별도 `sam2` 패키지 불필요).
- 추적 정확성 OK(합성 100%), 컷 감지·오디오 동기 OK. **병목은 오직 SAM2 추론(GPU 필요)**.

### Git / 분기 전략
- 기본: GitHub Flow. `main` 항상 배포(여기선 "문서 일관") 가능 상태 유지.
- 문서 단계: `main` 직접 커밋 허용(초기). PoC/구현부터 feature 브랜치.

### 알려진 미해결 이슈 / 주의사항
- [x] 라이선스 확정 완료 → **기본 업스케일러 SwinIR**. Real-ESRGAN 은 옵션(상업 배포 시 비활성). **libx264=GPL** → 상업 배포 시 코덱/라이선스 법무 재검토.
- [~] SAM2 컷 재추적(ADR 0006): 코드 + CPU 테스트(280개) 완료. **단일 샷 추적은 GPU 실검증 OK(2026-05-29)**. 컷 섞인 **멀티샷 재추적·threshold(0.5)·Grounding DINO 재검출은 GPU 멀티샷 클립으로 추가 검증 필요** — 잔여 리스크(최대 리스크에서 하향).
- [ ] EdgeTAM(CPU 비디오 추적 후보)·경량 백엔드는 v1.1 평가
- [x] **[GPU 블로커 해소] 비디오 모드**: SAM2 CPU ≈0.10 fps는 유지되나 Colab GPU(T4)로 추적·슬로우·트림·루프 실검증 완료(2026-05-29). 개발/검증은 Colab GPU, 실사용 GPU 전제로 진행.

**리뷰 제안 백로그**:
- [해소] `ui/frame_canvas` 오버레이 픽셀 이중루프 → numpy 벡터화(`mask_to_rgba`) 완료 (crop-ux)
- [해소] `ui/frame_canvas` `set_overlay` 미연결 → 클릭 후 오버레이 표시 연결 완료 (crop-ux)
- [해소] `ui/main_window` `box_size` 매직넘버 → 슬라이더/`sizing` 상수로 대체 완료 (crop-ux)
- [해소] `tests/` "TDD Red" 잔존 주석 정리 완료 (crop-ux)
- [ ] `infra/video_io`: fps 산출 `average_rate` — VFR 부정확, `r_frame_rate` 폴백 검토(비디오 모드 슬라이스에서)
- [ ] `infra/sam2_image_backend`: `_ensure_loaded()` 스레드 락 미적용 — 다중 워커 확장 시 경합 위험
- [ ] `tests/fixtures/fakes.py`: `_make_rect_mask` 매개변수 5개 — dataclass 로 묶기 검토
- [ ] (업스케일) `ui/main_window` `_UpscaleSaveWorker` 5인자 튜플 → `UpscaleSaveRequest` dataclass 권장
- [ ] (업스케일) `ui/main_window` `_on_upscale_toggled` `hasattr` 방어 — 죽은 분기 가능, 제거/주석
- [ ] (업스케일) `ui/main_window` 업스케일 워커 중 슬라이더 조작 시 저장버튼 깜빡임 — 워커 중 재계산/재활성 억제
- [ ] (업스케일) `tests/test_upscale.py` None 경로 미호출 검증이 간접적 — 직접 가드 보강
- [해소] (비디오) `core/export/video_export` GIF `duration` 초→ms 수정 완료 (imageio 2.28+, crop-tuning)
- [ ] (비디오) `infra/video_io._decode_span` step>1 시 인덱스 의미(샘플 순번) 주석 명시
- [ ] (비디오) `app/video_capture._fallback_center` 첫 프레임 None 케이스 단위 테스트 추가
- [ ] (재추적) `core/tracking.select_best_match` prev_feat 항상 None(위치 기반만) — cls_sim 확장점 docstring 명시
- [ ] (재추적) `app/video_capture` prev_box None인데 detect 호출 후 폐기 — WHY 주석(카운터 일관성)

---

## 4. 방법론 강제 규칙

- **기획 문서**: 영상전문가 / PM / 아이돌 덕후 3개 페르소나 검토 → 수정 → **3인 전원 컨펌 시 통과**. 모든 신규/수정 문서에 적용.
- **MVP 범위**: `docs/RFP.md` §5 및 계획서 In/Out 표 준수(scope creep 방지). Out 항목은 아키텍처만 확장 대비.
- **커밋-문서 동기화**: 코드 커밋 시 연관 문서·HANDOFF 동시 갱신(전역 지침).

---

## 5. 재개 체크리스트

새 세션에서 Claude 에게 다음을 먼저 실행하게 하라:

1. 이 문서(`HANDOFF.md`) 전체 읽기
2. `/bootstrap` 실행 (정합성 게이트)
3. `git log --oneline --all --graph | head -30` 확인
4. 블로커가 있으면 먼저 해결
5. 아니면 "미완료" 섹션의 다음 우선순위 작업 시작 (현재: 문서 검토 루프 → PoC)

---

## 6. 참고 자료

- 계획서: `docs/plans/easy-capture-plan.md`
- 전역 지침: `~/.claude/CLAUDE.md`
