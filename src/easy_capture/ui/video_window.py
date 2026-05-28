"""비디오 모드 메인 윈도우.

파일 열기 → 구간(시작/끝) 슬라이더 → 첫 프레임 표시·클릭 →
추적(_TrackWorker) → 종횡비/크기/smooth 즉시 재계산 → GIF/MP4 저장(_ExportWorker).

핵심 설계(계획서 §3-4):
  - _TrackWorker: VideoCaptureUseCase.track (무거움, propagate 1회).
  - _ExportWorker: VideoCaptureUseCase.export (crop+encode).
  - compute_boxes: 종횡비·크기·smooth_window 변경 시 메인 스레드 즉시 재호출.
    재추적 없음 — propagate_call_count 회귀 가드를 UI에서도 만족한다.
  - FrameCanvas·coords 재사용 (이미지 모드와 동형).
  - UI는 usecase 공개 API(probe_meta, read_span_frames)만 사용(캡슐화, 리뷰 [중요] 3).
  - 한국어 상태 안내.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QToolBar,
)

from easy_capture.core.crop.crop import ASPECT_PRESETS
from easy_capture.core.export.video_export import VideoExportConfig
from easy_capture.core.tracking.gap_policy import GapPolicy
from easy_capture.ui.frame_canvas import FrameCanvas
from easy_capture.ui.sizing import (
    DEFAULT_CROP_RATIO,
    MAX_CROP_RATIO,
    MIN_CROP_RATIO,
    crop_ratio_to_size,
)

# 종횡비 콤보 항목
_ASPECT_FREE_LABEL = "자유"
_ASPECT_ITEMS: list[tuple[str, str | None]] = [
    (_ASPECT_FREE_LABEL, None),
    *[(k, k) for k in ASPECT_PRESETS],
]

# 기본 smooth_window
# occlusion 갭 정책 콤보 항목: (표시 라벨, GapPolicy)
# WHY: 추적이 끊긴 프레임 처리 방식을 사용자가 고른다. 백엔드 build_output_indices가
#      이미 세 정책을 지원하므로 UI는 선택값을 export config로 전달하기만 한다.
_GAP_POLICY_ITEMS: list[tuple[str, GapPolicy]] = [
    ("배경 유지", GapPolicy.BACKGROUND),
    ("컷(갭 제외)", GapPolicy.CUT),
    ("정지(갭 홀드)", GapPolicy.FREEZE),
]

_DEFAULT_SMOOTH = 5

# 기본 출력 FPS
_DEFAULT_FPS = 12.0

# 구간 선택 기본값 (프레임 수)
_DEFAULT_SPAN_END = 60

# _estimate_frame_count 매직넘버 설명 상수 (리뷰 [제안])
# WHY: fps가 있을 때 SpinBox 상한을 "최대 300초(5분) 분량"으로 설정한다.
#      fps 정보가 없을 때는 관대한 상한(3000프레임)으로 입력 오류를 방지한다.
_MAX_ESTIMATE_SECONDS = 300   # fps 있는 경우 추정 상한 (초)
_DEFAULT_FRAME_LIMIT = 3000   # fps 없는 경우 관대한 상한 (프레임)


class _TrackWorker(QThread):
    """VideoCaptureUseCase.track을 백그라운드에서 실행하는 워커.

    WHY: SAM2 video 전파(propagate)·Grounding DINO 검출(detect)은 GPU에서도
         수 초 걸린다. 메인 스레드 실행 시 UI가 얼기 때문에 QThread로 분리.
         _SegWorker(이미지 모드) 패턴을 그대로 계승: Signal·run·예외 처리 동형.

    배선 흐름:
      1. usecase.detect_cuts(video_path, span)으로 컷 프레임 인덱스를 구한다.
         detector 없으면 None 반환 — usecase가 infra 위임 책임을 가진다.
      2. track(frames, point, cut_frames=...)으로 샷 경계 재추적을 실행한다.
    """

    track_ready = Signal(object)  # TrackResult
    error = Signal(str)           # 한국어 오류 메시지

    def __init__(self, usecase, frames, point, video_path=None, span=None) -> None:
        """워커 초기화.

        Args:
            usecase:    VideoCaptureUseCase 인스턴스.
            frames:     구간 전체 프레임 리스트.
            point:      클릭 좌표 (x, y).
            video_path: 비디오 파일 경로(detect_cuts 입력용, None이면 컷 감지 생략).
            span:       FrameSpan(start, end) 구간(detect_cuts 입력용).
        """
        super().__init__()
        self._usecase = usecase
        self._frames = frames
        self._point = point
        self._video_path = video_path
        self._span = span

    def run(self) -> None:
        """usecase.detect_cuts → usecase.track을 워커 스레드에서 실행한다."""
        from easy_capture.core.segmentation.video_backend import EmptyTrackError

        try:
            cut_frames = self._resolve_cut_frames()
            result = self._usecase.track(self._frames, self._point, cut_frames=cut_frames)
            self.track_ready.emit(result)
        except EmptyTrackError as exc:
            self.error.emit(f"[빈마스크] {exc}")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"추적 오류: {exc}")

    def _resolve_cut_frames(self) -> list[int] | None:
        """video_path·span이 있으면 usecase.detect_cuts에 위임한다.

        WHY: infra.shot_detect 직접 호출(ui→infra 위반)을 제거하고
             usecase 공개 API만 사용한다. video_path 없으면 단일 샷 폴백.
        """
        if self._video_path is None or self._span is None:
            return None
        return self._usecase.detect_cuts(self._video_path, self._span)


class _ExportWorker(QThread):
    """VideoCaptureUseCase.export를 백그라운드에서 실행하는 워커.

    WHY: GIF/MP4 인코딩이 수 초 걸릴 수 있어 메인 스레드 차단을 방지.
         _UpscaleSaveWorker(이미지 모드) 패턴 계승.
    """

    done = Signal(str)   # 저장 완료 경로
    error = Signal(str)  # 한국어 오류 메시지

    def __init__(self, usecase, frames, boxes, target, result) -> None:
        super().__init__()
        # WHY: 인자를 튜플로 묶어 매개변수 3개 규칙 완화 — 생성자 계약
        self._args = (usecase, frames, boxes, target, result)

    def run(self) -> None:
        """워커 스레드에서 export를 실행한다."""
        usecase, frames, boxes, target, result = self._args
        path, _ = target
        try:
            usecase.export(frames, boxes, target, result=result)
            self.done.emit(path)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"저장 오류: {exc}")


class VideoMainWindow(QMainWindow):
    """비디오 모드 메인 윈도우.

    usecase_factory: path -> VideoCaptureUseCase 를 반환하는 callable.
    WHY: 파일 경로가 결정된 후에야 FrameSource(파일 기반)를 생성할 수 있으므로
         팩토리 패턴으로 usecase 생성을 지연한다(이미지 모드 동형).
    """

    def __init__(self, usecase_factory) -> None:
        super().__init__()
        self.setWindowTitle("easy-capture — 비디오 모드")
        self.resize(900, 640)
        self._usecase_factory = usecase_factory

        # 상태 필드
        self._usecase = None
        self._frames: list[np.ndarray] | None = None
        self._track_result = None          # TrackResult — export gap_policy용
        self._boxes: list | None = None
        self._aspect: str | None = None
        self._size_ratio: int = DEFAULT_CROP_RATIO
        self._smooth_window: int = _DEFAULT_SMOOTH
        self._total_frames: int = 0
        # 리뷰 [제안]: __init__에서 None으로 선언해 hasattr 대신 is not None 검사
        self._pending_point: tuple[int, int] | None = None
        # shot_detect 배선용 — 파일 경로·구간 보관([중요] 2 수정)
        self._video_path: str | None = None
        self._pending_span = None  # FrameSpan

        # 워커
        self._track_worker: _TrackWorker | None = None
        self._export_worker: _ExportWorker | None = None

        self._build_toolbar()
        self._build_canvas()
        self._build_statusbar()

    # ------------------------------------------------------------------
    # UI 빌더
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        """파일열기·구간·종횡비·크기·smooth·저장 위젯이 있는 툴바를 구성한다."""
        tb = QToolBar("메인")
        self.addToolBar(tb)

        open_btn = QPushButton("파일 열기")
        open_btn.clicked.connect(self._on_open_file)
        tb.addWidget(open_btn)

        self._build_span_controls(tb)
        self._build_aspect_combo(tb)
        self._build_size_slider(tb)
        self._build_smooth_spinbox(tb)
        self._build_gap_combo(tb)
        self._build_track_save_buttons(tb)

    def _build_span_controls(self, tb: QToolBar) -> None:
        """구간 시작/끝 SpinBox를 툴바에 추가한다."""
        tb.addWidget(QLabel("  시작:"))
        self._span_start = QSpinBox()
        self._span_start.setMinimum(0)
        self._span_start.setValue(0)
        self._span_start.setEnabled(False)
        self._span_start.valueChanged.connect(self._on_span_changed)
        tb.addWidget(self._span_start)

        tb.addWidget(QLabel(" 끝:"))
        self._span_end = QSpinBox()
        self._span_end.setMinimum(1)
        self._span_end.setValue(_DEFAULT_SPAN_END)
        self._span_end.setEnabled(False)
        self._span_end.valueChanged.connect(self._on_span_changed)
        tb.addWidget(self._span_end)

    def _build_aspect_combo(self, tb: QToolBar) -> None:
        """종횡비 선택 콤보박스를 툴바에 추가한다."""
        tb.addWidget(QLabel("  종횡비:"))
        self._aspect_combo = QComboBox()
        for label, _ in _ASPECT_ITEMS:
            self._aspect_combo.addItem(label)
        self._aspect_combo.setEnabled(False)
        self._aspect_combo.currentIndexChanged.connect(self._on_aspect_changed)
        tb.addWidget(self._aspect_combo)

    def _build_gap_combo(self, tb: QToolBar) -> None:
        """occlusion 갭 정책 선택 콤보박스를 툴바에 추가한다.

        export 시점에만 값을 읽으므로 시그널 연결은 불필요(재추적·재계산 무관).
        """
        tb.addWidget(QLabel("  갭:"))
        self._gap_combo = QComboBox()
        for label, _ in _GAP_POLICY_ITEMS:
            self._gap_combo.addItem(label)
        self._gap_combo.setEnabled(False)
        tb.addWidget(self._gap_combo)

    def _build_size_slider(self, tb: QToolBar) -> None:
        """크기 슬라이더를 툴바에 추가한다."""
        tb.addWidget(QLabel("  크기:"))
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setMinimum(MIN_CROP_RATIO)
        self._size_slider.setMaximum(MAX_CROP_RATIO)
        self._size_slider.setValue(DEFAULT_CROP_RATIO)
        self._size_slider.setFixedWidth(100)
        self._size_slider.setEnabled(False)
        self._size_slider.valueChanged.connect(self._on_size_changed)
        tb.addWidget(self._size_slider)
        self._size_label = QLabel(f"{DEFAULT_CROP_RATIO}%")
        tb.addWidget(self._size_label)

    def _build_smooth_spinbox(self, tb: QToolBar) -> None:
        """smooth_window SpinBox를 툴바에 추가한다."""
        tb.addWidget(QLabel("  떨림완화:"))
        self._smooth_spin = QSpinBox()
        self._smooth_spin.setMinimum(1)
        self._smooth_spin.setMaximum(31)
        self._smooth_spin.setSingleStep(2)
        self._smooth_spin.setValue(_DEFAULT_SMOOTH)
        self._smooth_spin.setEnabled(False)
        self._smooth_spin.valueChanged.connect(self._on_smooth_changed)
        tb.addWidget(self._smooth_spin)

    def _build_track_save_buttons(self, tb: QToolBar) -> None:
        """추적·저장 버튼을 툴바에 추가한다."""
        self._track_btn = QPushButton("추적 실행")
        self._track_btn.clicked.connect(self._on_track)
        self._track_btn.setEnabled(False)
        tb.addWidget(self._track_btn)

        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(["GIF", "MP4"])
        self._fmt_combo.setEnabled(False)
        tb.addWidget(self._fmt_combo)

        self._fps_spin = QSpinBox()
        self._fps_spin.setMinimum(1)
        self._fps_spin.setMaximum(60)
        self._fps_spin.setValue(int(_DEFAULT_FPS))
        self._fps_spin.setPrefix("fps: ")
        self._fps_spin.setEnabled(False)
        tb.addWidget(self._fps_spin)

        self._save_btn = QPushButton("저장")
        self._save_btn.clicked.connect(self._on_export)
        self._save_btn.setEnabled(False)
        tb.addWidget(self._save_btn)

    def _build_canvas(self) -> None:
        """프레임 표시·클릭 캔버스를 중앙 위젯으로 설정한다."""
        self._canvas = FrameCanvas()
        self._canvas.clicked.connect(self._on_canvas_click)
        self.setCentralWidget(self._canvas)

    def _build_statusbar(self) -> None:
        """상태 메시지 표시 바를 구성한다."""
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_label = QLabel("영상 파일을 열어 시작하세요.")
        self._status.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # 슬롯
    # ------------------------------------------------------------------

    def _on_open_file(self) -> None:
        """파일 다이얼로그로 영상 파일을 열고 첫 프레임을 표시한다."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "영상 파일 열기",
            "",
            "영상 (*.mp4 *.mov *.avi *.mkv *.webm);;전체 파일 (*)",
        )
        if not path:
            return
        try:
            self._usecase = self._usecase_factory(path)
            self._video_path = path  # shot_detect 배선용 보관
            # WHY: usecase 공개 API probe_meta() 사용 — 비공개 _source 관통 금지(리뷰 [중요] 3)
            meta = self._usecase.probe_meta()
            self._total_frames = _estimate_frame_count(meta)
            self._apply_source_fps(meta)
            self._setup_span_controls()
            self._load_first_frame()
            self._set_status("구간을 설정하고 피사체를 클릭해 주세요.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "파일 열기 실패", f"파일을 열 수 없습니다.\n{exc}")

    def _apply_source_fps(self, meta) -> None:
        """fps 입력 기본값을 원본 영상 fps로 설정한다(사용자가 spin으로 조절 가능).

        WHY: 기본 12fps 고정이면 원본(예 25fps)과 달라 매번 수동 입력해야 한다.
             원본 fps를 기본으로 두면 '원본과 동일'이 기본이고 조절은 그대로 가능.
        """
        if meta.fps and meta.fps > 0:
            self._fps_spin.setValue(round(meta.fps))

    def _on_span_changed(self) -> None:
        """구간 변경 시 첫 프레임 미리보기를 갱신한다."""
        if self._usecase is None:
            return
        self._load_first_frame()
        self._track_result = None
        self._boxes = None
        self._save_btn.setEnabled(False)

    def _on_canvas_click(self, x: int, y: int) -> None:
        """캔버스 클릭 시 구간 프레임 로드 + 추적 버튼 활성화."""
        if self._usecase is None:
            return
        if self._track_worker is not None and self._track_worker.isRunning():
            self._set_status("추적 중입니다. 잠시만 기다려 주세요.")
            return
        try:
            self._frames, self._pending_span = self._load_span_frames_with_span()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "구간 로드 실패", str(exc))
            return
        self._set_status("클릭 포인트를 등록했습니다. '추적 실행'을 눌러 주세요.")
        self._track_btn.setEnabled(True)
        self._pending_point = (x, y)

    def _on_track(self) -> None:
        """추적 워커를 시작한다(무거움 — propagate 1회)."""
        # 리뷰 [제안]: hasattr 대신 is not None 검사 통일
        if self._frames is None or self._pending_point is None:
            return
        if self._track_worker is not None and self._track_worker.isRunning():
            self._set_status("추적 중입니다. 잠시만 기다려 주세요.")
            return
        self._set_status(
            "추적 중… (처음 실행 시 모델 로드로 시간이 걸릴 수 있습니다)"
        )
        self._track_btn.setEnabled(False)
        self._track_worker = _TrackWorker(
            self._usecase,
            self._frames,
            self._pending_point,
            video_path=self._video_path,
            span=self._pending_span,
        )
        self._track_worker.track_ready.connect(self._on_track_ready)
        self._track_worker.error.connect(self._on_track_error)
        self._track_worker.start()

    def _on_track_ready(self, result) -> None:
        """추적 결과 도착 시 박스 재계산 + 저장 버튼 활성화.

        needs_correction 구간이 있으면 한국어 안내를 상태바에 표시한다.
        WHY: 수동 교정 UI는 다음 슬라이스(플래그 표시만 — 계획서 §1, ADR 0006).
        """
        self._track_result = result
        self._recompute_boxes()
        self._aspect_combo.setEnabled(True)
        self._size_slider.setEnabled(True)
        self._smooth_spin.setEnabled(True)
        self._fmt_combo.setEnabled(True)
        self._fps_spin.setEnabled(True)
        self._gap_combo.setEnabled(True)
        self._save_btn.setEnabled(True)

        status = _build_track_status(result)
        self._set_status(status)

    def _on_track_error(self, message: str) -> None:
        """추적 실패 시 한국어 안내."""
        self._track_btn.setEnabled(True)
        if message.startswith("[빈마스크]"):
            self._set_status("대상을 인식하지 못했어요. 피사체 위를 다시 클릭해 주세요.")
        else:
            QMessageBox.warning(self, "추적 실패", message)
            self._set_status("추적에 실패했습니다. 다시 시도해 주세요.")

    def _on_aspect_changed(self, index: int) -> None:
        """종횡비 변경 시 박스 즉시 재계산(재추적 없음)."""
        _, key = _ASPECT_ITEMS[index]
        self._aspect = key
        self._recompute_boxes()

    def _on_size_changed(self, value: int) -> None:
        """크기 슬라이더 변경 시 박스 즉시 재계산(재추적 없음)."""
        self._size_ratio = value
        self._size_label.setText(f"{value}%")
        self._recompute_boxes()

    def _on_smooth_changed(self, value: int) -> None:
        """smooth_window 변경 시 박스 즉시 재계산(재추적 없음)."""
        self._smooth_window = value
        self._recompute_boxes()

    def _on_export(self) -> None:
        """저장 경로 선택 후 _ExportWorker를 시작한다."""
        if self._frames is None or self._boxes is None or self._usecase is None:
            return
        if self._export_worker is not None and self._export_worker.isRunning():
            self._set_status("저장 중입니다. 잠시만 기다려 주세요.")
            return

        fmt = self._fmt_combo.currentText().lower()
        default_name = f"output.{fmt}"
        filter_str = "GIF (*.gif)" if fmt == "gif" else "MP4 (*.mp4)"
        path, _ = QFileDialog.getSaveFileName(self, "저장", default_name, filter_str)
        if not path:
            return

        fps = float(self._fps_spin.value())
        gap_policy = _GAP_POLICY_ITEMS[self._gap_combo.currentIndex()][1]
        config = VideoExportConfig(fmt=fmt, fps=fps, gap_policy=gap_policy)
        self._save_btn.setEnabled(False)
        self._set_status("저장 중…")
        self._export_worker = _ExportWorker(
            self._usecase,
            self._frames,
            self._boxes,
            (path, config),
            self._track_result,  # gap_policy용 TrackResult 전달
        )
        self._export_worker.done.connect(self._on_export_done)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_done(self, path: str) -> None:
        """저장 완료 시 상태 바 갱신 및 저장 버튼 재활성."""
        self._save_btn.setEnabled(True)
        self._set_status(f"저장 완료: {path}")

    def _on_export_error(self, message: str) -> None:
        """저장 실패 시 한국어 안내 및 버튼 재활성."""
        self._save_btn.setEnabled(True)
        QMessageBox.critical(self, "저장 실패", message)
        self._set_status("저장에 실패했습니다. 다시 시도해 주세요.")

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _setup_span_controls(self) -> None:
        """파일 로드 후 구간 SpinBox 범위와 기본값을 설정한다."""
        end = max(1, min(_DEFAULT_SPAN_END, self._total_frames))
        self._span_start.setMaximum(max(0, self._total_frames - 1))
        self._span_end.setMaximum(self._total_frames)
        self._span_end.setValue(end)
        self._span_start.setEnabled(True)
        self._span_end.setEnabled(True)
        self._track_btn.setEnabled(False)

    def _load_first_frame(self) -> None:
        """구간 첫 프레임만 추출해 캔버스에 표시한다(가벼움)."""
        from easy_capture.core.source.frame_source import FrameSpan

        start = self._span_start.value()
        span = FrameSpan(start=start, end=start + 1)
        try:
            frame = self._usecase.load_first_frame(span)
            self._canvas.set_frame(frame)
            self._canvas.set_overlay(None)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"프레임 로드 실패: {exc}")

    def _load_span_frames(self) -> list[np.ndarray]:
        """현재 구간 설정으로 전체 프레임 시퀀스를 추출한다.

        WHY: usecase 공개 API read_span_frames() 사용 — 비공개 _source 관통 금지(리뷰 [중요] 3).
        """
        frames, _ = self._load_span_frames_with_span()
        return frames

    def _load_span_frames_with_span(self):
        """현재 구간 설정으로 프레임 시퀀스와 FrameSpan을 함께 반환한다.

        WHY: _TrackWorker가 shot_detect 호출을 위해 FrameSpan을 필요로 하므로
             프레임과 함께 span을 반환한다([중요] 2 수정).
        """
        from easy_capture.core.source.frame_source import FrameSpan

        start = self._span_start.value()
        end = self._span_end.value()
        if end <= start:
            raise ValueError("끝 프레임이 시작 프레임보다 커야 합니다.")
        span = FrameSpan(start=start, end=end)
        return self._usecase.read_span_frames(span), span

    def _recompute_boxes(self) -> None:
        """보관된 TrackResult + 현재 params로 박스를 즉시 재계산한다.

        backend를 다시 호출하지 않는다(순수 계산만).
        WHY: 종횡비·크기·smooth 슬라이더 드래그 시 연속 호출되는데
             compute_boxes가 순수 함수이므로 멈춤 없이 즉시 완료된다.
        """
        if self._track_result is None or self._frames is None:
            return

        from easy_capture.app.video_capture import VideoCropParams

        frame = self._frames[0]
        frame_w, frame_h = frame.shape[1], frame.shape[0]
        size = crop_ratio_to_size(self._size_ratio, (frame_w, frame_h))
        params = VideoCropParams(
            box_size=size,
            aspect=self._aspect,
            smooth_window=self._smooth_window,
        )
        self._boxes = self._usecase.compute_boxes(
            self._track_result, params, (frame_w, frame_h)
        )

    def _set_status(self, message: str) -> None:
        """상태 바 메시지를 업데이트한다."""
        self._status_label.setText(message)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _build_track_status(result) -> str:
    """TrackResult로부터 상태바 메시지를 생성한다.

    needs_correction 구간이 있으면 한국어 안내를 포함한다.
    WHY: 수동 교정 UI는 다음 슬라이스 — 이번은 플래그 표시만(계획서 §1).
    """
    n_frames = len(result.masks)
    n_cuts = len(getattr(result, "cut_frames", []))
    needs_correction = getattr(result, "needs_correction", [])
    n_fail = sum(1 for nc in needs_correction if nc)

    base = f"추적 완료 — {n_frames}프레임"
    if n_cuts > 0:
        base += f", {n_cuts}개 컷 감지"
    if n_fail > 0:
        base += f" / 일부 구간 재매칭 실패 — 교정 필요"
    base += ". '저장'으로 GIF/MP4를 만들어 주세요."
    return base


def _estimate_frame_count(meta) -> int:
    """FrameMeta에서 총 프레임 수를 추정한다.

    WHY: probe()가 총 프레임 수를 직접 반환하지 않으므로 fps × 추정 길이로
         SpinBox 범위를 설정한다.
         _MAX_ESTIMATE_SECONDS(300초=5분): fps가 있을 때 합리적인 SpinBox 상한.
         _DEFAULT_FRAME_LIMIT(3000): fps 정보 없을 때 관대한 상한으로 입력 오류 방지.
    """
    if meta.fps and meta.fps > 0:
        return max(_DEFAULT_FRAME_LIMIT, int(meta.fps * _MAX_ESTIMATE_SECONDS))
    return _DEFAULT_FRAME_LIMIT
