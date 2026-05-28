# 변경 이력

Keep a Changelog 형식 준수. 버전은 Semantic Versioning을 따른다.

---

## [미출시]

### 추가
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
