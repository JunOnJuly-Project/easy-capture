# 변경 이력

Keep a Changelog 형식 준수. 버전은 Semantic Versioning을 따른다.

---

## [미출시]

### 수정
- **비디오 크롭 잘림·흔들림·GIF 재생속도** (`feature/video/crop-tuning`, GPU 실검증 피드백)
  - 크롭 잘림: `compute_boxes`가 고정 `box_size` 대신 구간 내 마스크 bbox 최대×`subject_padding`으로 크기를 자동 산출. `_expand_to_aspect`로 종횡비를 확대 방향 적용 → 1:1 등에서도 피사체가 안 잘림.
  - 위치 흔들림: 크롭 중심을 centroid(무게중심)→bbox 중심(`_bbox_center`)으로 변경(자세 변화에 안정). `smooth_window` 노출.
  - GIF 재생속도: `_encode_gif` duration 초→밀리초(imageio 2.28+ 단위 변경). 이전엔 무효화돼 ≈10fps로 느렸음. `GIF_FPS`로 조절 가능.
  - 앱 검증 노트북(`poc/colab/easy_capture_app_verify.ipynb`) 설치 셀 수정(editable→sys.path, Colab `ModuleNotFoundError` 해결) + 조절 파라미터 노출. 테스트 300개(신규 16).

### 추가
- **비디오 마스크 정제 — box 프롬프트 + 최대 연결성분** (`feature/video/mask-refine`, ADR 0014 신규·ADR 0010 연계)
  - GPU 실검증에서 마스크 과대(추적 대상의 배 + 옆 멤버의 팔)로 1인 클로즈업이 아니라 군무 여러 명이 담기던 문제를 해결.
  - SAM2 box 프롬프트 도입: `VideoSegmentationBackend` Protocol에 `add_box(session, box)` 추가(`add_click`은 무변경 유지·무회귀). Grounding DINO의 전신 bbox를 SAM2에 직접 전달해 점 1개(point)보다 정확한 전신 마스크를 얻는다. PoC 노트북이 통과했던 box 프롬프트를 production에 복원(회귀 수정).
  - `core/crop/mask_refine.largest_component`: SAM2 마스크에서 가장 큰 4-연결 성분만 남겨 인접 멤버 파편을 제거하는 순수 함수(numpy만, scipy·cv2 비의존). 마스크 과대 → 크롭 박스 과대 연쇄를 끊는다.
  - 혼합 디스패치(box 우선, point 폴백): `CutSelection.box`/재매칭 통과 후보 box가 있으면 `add_box`, 없으면 `add_click`.
  - 전체 561 테스트 통과. ADR 0014 신규(0010 확장·Superseded 아님), 0006·0012 연계. **GPU 노트북 게이트로 마스크 정확도·추적 유지율 재측정 필요**(infra는 CI 미실행).
- **비디오 수동 교정 = 컷별 오브젝트 명시 선택** (`feature/video/cut-selection`, ADR 0006 보강)
  - 멀티샷 군무에서 자동 재매칭(IoU 기반, appearance feature 없음)이 `needs_correction` 다발로 구조적 한계를 드러냄 → 각 컷(샷) 시작 프레임의 후보를 사용자에게 보여주고, 사용자가 추적 대상을 명시 선택하면 그 선택으로 컷별 재추적하는 방식으로 전환. 자동 재매칭은 폴백으로만 유지(혼합 정책).
  - `core/tracking/cut_selection`: `CutSelection`(샷·클릭점·전신 box 불변 값객체)·`index_selections_by_shot`·`validate_selections`(범위·중복 한국어 ValueError) 순수 함수.
  - `app/video_capture`: `detect_cut_candidates`(컷별 후보 검출)·`track(selections=)`(샷 인덱스→선택 매핑으로 컷별 SAM2 재초기화). `selections` 미지정 시 기존 자동 재매칭 경로 유지(무회귀).
  - 앱 검증 노트북에 컷별 선택 흐름 반영. ADR 0006 보강(재매칭 로직 유지·출력만 명시 선택으로 대체). GPU 실검증 결과 컷별 명시 선택 시 `needs_correction` 0건 확인.
- **비디오 슬로우모션 — 구간별 가변 재생속도** (`feature/timing/*`, ADR 0013 신규·트림+루프 보강)
  - 추적·크롭이 끝난 프레임 시퀀스에 구간별 배속(슬로우모션·패스트포워드)을 적용해 GIF/MP4로 내보낸다.
  - `core/timing/timeremap`: `build_playback_schedule`(프레임 수+구간 배속+기준 fps→재생 스케줄)·`schedule_to_cfr_indices`(MP4 CFR 프레임 복제/드롭)·`clamp_durations_for_gif`(GIF per-frame duration 10ms 하한 가드 — 빠르게 만들려다 오히려 느려지는 역전 방지) 순수 함수. numpy·stdlib만 의존(imageio·torch 비의존).
  - `PlaybackSchedule` 이중 표현(frame_indices + durations_ms): GIF는 per-frame duration 직접 적용, MP4는 프레임 복제로 시간 표현 → 두 백엔드를 한 스케줄로 대칭 지원.
  - **트림+슬로우+루프**(덕후 1순위 보강): `VideoExportConfig.trim`(출력 구간 제한)·`loop_count`(GIF 루프 횟수) 추가. 트림 먼저 슬라이스 후 segments를 트림-로컬로 평행이동해 좌표계 단일화. MP4는 루프를 조용히 무시(경고는 UI/노트북에서만).
  - `VideoExportConfig.segments = ()`(기본 빈 튜플)이면 항등 스케줄 → 기존 등속 export 무회귀. UI 구간 테이블·검증 노트북 반영.
  - ADR 0013 신규(타임리맵 순수 로직 위치·이중 표현)·트림+루프 보강. **MVP 슬로우는 프레임 복제 방식이라 끊김(stutter)이 보일 수 있음**(부드러운 슬로우 RIFE 보간은 v1.1).
- **비디오 occlusion gap 정책 UI** (`feature/video/gap-policy-ui`)
  - 추적이 끊긴(occlusion) 프레임 처리 정책을 UI에서 선택: 배경 유지(BACKGROUND)/컷(CUT)/정지(FREEZE).
  - `ui/video_window` 갭 콤보 + export 시 `VideoExportConfig.gap_policy` 전달. 백엔드 `build_output_indices`는 기존 재사용(UI가 선택값만 노출).
  - 테스트 284개 통과(`_GAP_POLICY_ITEMS` 매핑 가드, 기본 BACKGROUND 무회귀). gap_policy는 순수 로직이라 CPU 검증 완료.
- **비디오 샷경계 재추적 — 컷 넘어 동일 인물 자동 재추적** (`feature/video/shot-retrack`, ADR 0006)
  - `core/crop.bbox_of_mask`(마스크→bbox), `core/tracking`: `select_best_match`·`RematchResult`·`REMATCH_THRESHOLD=0.5`·`split_into_shots` 순수 함수.
  - `core/segmentation/detection_backend`: `Detection`·`DetectionBackend` Protocol(stateless, ADR 0012).
  - `infra/shot_detect`: PySceneDetect 컷 감지(경로 기반 `open_video`+`seek`+`detect_scenes`, CPU 테스트 검증).
  - `infra/grounding_dino_backend`: Grounding DINO 재검출(지연 로드).
  - `app/video_capture.track(detector, cut_frames)`: 샷 분할→경계 재검출→`rematch_score` 재매칭→통과 시 SAM2 재초기화·objid 유지, 미달 시 `needs_correction`. propagate==샷수·detect==컷수 회귀 가드. `detector=None` 하위호환.
  - router/UI 배선(production 재추적 활성화), `needs_correction` 안내.
  - 테스트 280개 통과. ADR 0012 추가·0006 보완. 코드 리뷰 [치명적] 1·[중요] 3 반영(scenedetect 0.7 API 2건·Grounding DINO `threshold` 키워드·프롬프트 마침표·배선).
  - **주의**: SAM2 video·Grounding DINO 실추론은 GPU 필요 → Colab 검증 후행. threshold 0.5는 H2 실보정 대기.
- **비디오(움짤) 모드 첫 수직 슬라이스 — 코드** (`feature/video/tracking-slice`)
  - 단일 샷 구간에서 클릭 대상을 추적해 크롭한 GIF/MP4 생성(척추 관통). 샷경계 재추적·Grounding DINO·오디오·업스케일 결합은 후속 슬라이스.
  - `core/segmentation/video_backend`: `VideoSegmentationBackend` Protocol(이미지와 분리·ISP, opaque session, ADR 0010).
  - `infra/sam2_video_backend`: `Sam2VideoBackend` — transformers `Sam2VideoModel`/`Sam2VideoProcessor`, 지연 로드. `post_process_masks`에 원본 해상도 전달(마스크 좌표 정합).
  - `app/video_capture`: `VideoCaptureUseCase` — `track`(전파=무거움, 워커 1회)/`compute_boxes`(순수=가벼움, 재추적 없이 즉시 갱신) 분리. 고정 box size 불변식. `propagate_call_count==1` 회귀 가드.
  - `core/export/video_export`: GIF/MP4 인코딩(imageio 지연 import, `macro_block_size=1`로 크롭 크기 보존, ADR 0011).
  - `infra/video_io`: 구간 프레임 시퀀스 추출(`FrameSpan`/`read_frames`) 확장.
  - `ui/video_window` + router 'gif' 분기: 구간 선택→클릭→추적(`_TrackWorker`)→미리보기→저장(`_ExportWorker`).
  - 테스트 216개 통과. ADR 0010·0011 추가. 코드 리뷰 [중요] 3건 반영.
  - **주의**: SAM2 video 실추론은 GPU 필요 → Colab 검증 후행(CPU 코드·Fake 테스트만 완료).
- **이미지 모드 업스케일(SwinIR)** (`feature/image/upscale`)
  - 크롭 결과를 저장 전 초해상도 업스케일(옵션, 배율 x2/x4=모델 선택).
  - `core/upscale`: `UpscaleBackend` Protocol(torch 비의존) + `reconstruction_to_rgb_uint8` 순수 정규화 함수(모델 출력 CHW float→RGB uint8, 검증 리스크 격리).
  - `infra/swin2sr_upscale_backend`: `Swin2srUpscaleBackend` — transformers `Swin2SRForImageSuperResolution`/`Swin2SRImageProcessor` 래퍼. 지연 로드. processor 8배수 패딩 보정 재크롭으로 출력 크기 = 입력×배율 보장.
  - `app/image_capture.export(upscaler=None)`: 옵션 메서드 주입. None이면 crop→save 직행(무회귀), 주입 시 crop→upscale→save.
  - `app/router`: `UPSCALE_MODELS` 카탈로그(repo·scale·라벨 단일 소스) + 팩토리 주입.
  - `ui/main_window`: 업스케일 체크박스·배율 콤보 + `_UpscaleSaveWorker`(백그라운드 crop→upscale→save).
  - 테스트 166개 통과. ADR 0009 추가. 코드 리뷰 [중요] 3건 반영(processor 패딩 보정 포함).
- **이미지 모드 크롭 UX 확장** (`feature/image/crop-ux`)
  - 종횡비 프리셋 선택(자유/1:1/9:16/16:9) + 크롭 크기 슬라이더 + 마스크 오버레이 표시.
  - `ImageCaptureUseCase`: `segment`(SAM2 1회·무거움)/`compute_box`(순수·가벼움) 2단계 분리 → 종횡비·크기 조정 시 재세그 없이 즉시 갱신. `SegmentResult`·`BoxParams` 데이터클래스.
  - `ui/sizing`: `crop_ratio_to_size` 순수 함수 + 슬라이더 상수.
  - `frame_canvas.mask_to_rgba`: numpy 벡터화 오버레이(픽셀 이중루프 제거).
  - 테스트 137개 통과(재세그 카운터 회귀 가드 포함). 코드 리뷰 [중요] 2건 반영.
- **이미지(짤) 모드 첫 수직 슬라이스 — end-to-end happy path 구현** (`feature/image/capture-slice`)
  - `core/export`: `crop_array` / `save_image` (Pillow, PNG/JPG, sRGB 태깅). `ExportConfig` 데이터클래스.
  - `infra/video_io`: `FrameSource` Protocol · `FrameMeta` · `open_source` 팩토리. 이미지=Pillow, 영상=PyAV 첫프레임(PTS 시크, RGB BT.709 정규화).
  - `infra/sam2_image_backend`: `Sam2ImageBackend` — transformers 5.9.0 `Sam2Model`/`Sam2Processor` 래퍼. 지연 로드(첫 `segment_image` 호출 시 다운로드), CPU 동작.
  - `app/image_capture`: `ImageCaptureUseCase` · `CropRequest` · `EmptyMaskError`. Protocol 주입(DIP), 워커 스레드 친화.
  - `app/router`: `AppRouter` — 모드선택 시그널 수신 → 이미지 메인윈도 생성·조립(composition root).
  - `ui/coords`: 위젯↔이미지 좌표 변환 순수 함수 (`widget_to_image`, 스케일·레터박스 보정).
  - `ui/frame_canvas`: `FrameCanvas` — RGB ndarray → QImage 표시, 클릭 시 이미지 좌표 시그널 방출, 마스크 오버레이 표시(`set_overlay`).
  - `ui/main_window`: `ImageMainWindow` — 파일 열기·캔버스·저장. `_SegWorker`(QThread) 로 SAM2 추론 비블로킹 처리.
  - `__main__`: `AppRouter.start()` 진입점으로 교체.
  - **ADR 0008**: `app/` 유스케이스 레이어 신설 결정 기록.
  - **계획서**: `docs/plans/image-mode-slice.md` (이미지 모드 슬라이스 개발 계획).
- 순수 로직 단위 테스트 71개 통과 (기존 스캐폴딩 20개 포함, 무회귀).
  - `test_image_export.py`, `test_image_capture.py`(FakeBackend·FakeFrameSource 주입), 좌표 변환 테스트.
  - `tests/fixtures/fakes.py`: `FakeBackend` · `FakeFrameSource` (torch/PyAV 비의존 가짜 구현).

### 변경
- `__main__`: `QApplication` 직접 구성 → `AppRouter.start()` 위임으로 단순화.
- README: 현재 상태를 "구현 중 — 이미지 모드 MVP 완료"로 갱신. 설치·실행·아키텍처 섹션 실제 명령어로 업데이트.

---

## [0.1.0-scaffolding] — 2026-05-28

> 브랜치: `feature/app/scaffolding`

### 추가
- 패키지 구조: `src/easy_capture/` (core / infra / ui 레이어).
- `core/crop`: `make_crop_box` · `centroid_of_mask` · `aspect_lock` 순수 함수.
- `core/gap_policy`: 추적 공백 정책 로직.
- `core/rematch`: 컷 재매칭 로직.
- `infra/device`: `detect_device` · `select_sam2_repo` 유틸.
- `core/segmentation/backend.py`: `SegmentationBackend` Protocol (추상).
- `ui/mode_select`: PySide6 모드 선택 창.
- 단위 테스트 20개 통과.
- 기획 문서 세트 (RFP · use-flow · wireframes · data-flow · architecture · resources · poc-plan · error-handling · ADR 0001~0007).
- PoC 검증: H1~H4 코드 검증 (조건부 Go), Colab GPU 노트북 (`poc/colab/`).
