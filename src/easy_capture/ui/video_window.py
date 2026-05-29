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
    QCheckBox,
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
from easy_capture.core.timing.timeremap import (
    TrimRange,
    build_playback_schedule,
    clamp_durations_for_gif,
    estimate_output_frame_count,
    shift_segments_into_trim,
    slice_for_trim,
)
from easy_capture.core.tracking.gap_policy import GapPolicy
from easy_capture.ui.frame_canvas import FrameCanvas
from easy_capture.ui.segment_table import SegmentTableWidget
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

# 폭증 경고 임계: 예상 출력 프레임이 원본의 이 배수를 초과하면 경고
# WHY: 계획서 §7 "기존 ×3 임계" — 3배 슬로우모션 구간이 전체를 차지할 때 상한.
#      사용자에게 인코딩 시간/용량 폭증을 미리 안내한다.
_FRAME_COUNT_OVERFLOW_RATIO: float = 3.0

# 트림 SpinBox 기본값 (파일 로드 전 초기 상태)
_TRIM_SPINBOX_DEFAULT_MAX = 9999

# loop_count SpinBox 상한 — 합리적 최대 반복 횟수
_LOOP_COUNT_MAX = 999


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

        # 트림 활성 여부 — export 시 TrimRange 생성 여부를 결정
        self._trim_enabled: bool = False

        self._build_toolbar()
        self._build_canvas_with_segment_panel()
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
        self._build_frame_inject_buttons(tb)
        self._build_trim_controls(tb)
        self._build_loop_controls(tb)
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

    def _build_frame_inject_buttons(self, tb: QToolBar) -> None:
        """"현재 프레임 → 시작/끝" 버튼을 툴바에 추가한다.

        WHY: 사용자가 미리보기 프레임을 보면서 버튼으로 구간 경계를 지정한다.
             프레임 번호 암기 불필요 — UX 핵심(Task 5-2).
        """
        self._frame_to_start_btn = QPushButton("프레임→시작")
        self._frame_to_start_btn.clicked.connect(self._on_frame_to_start)
        self._frame_to_start_btn.setEnabled(False)
        tb.addWidget(self._frame_to_start_btn)

        self._frame_to_end_btn = QPushButton("프레임→끝")
        self._frame_to_end_btn.clicked.connect(self._on_frame_to_end)
        self._frame_to_end_btn.setEnabled(False)
        tb.addWidget(self._frame_to_end_btn)

    def _build_trim_controls(self, tb: QToolBar) -> None:
        """트림 체크박스 + 시작/끝 SpinBox + 미리보기→트림시작/끝 버튼을 추가한다.

        WHY 좌표계: 트림 SpinBox 값은 출력 crops 시퀀스 상대 [0, span_len) 인덱스다.
          gap_policy=BACKGROUND 전제로만 span 상대와 일치한다(ADR 0013 2단계 인덱싱
          미구현 추적). segments 버튼(_on_frame_to_start/end)과 동일 좌표계를 따른다.
        """
        tb.addWidget(QLabel("  트림:"))
        self._trim_check = QCheckBox("사용")
        self._trim_check.setChecked(False)
        self._trim_check.stateChanged.connect(self._on_trim_toggled)
        tb.addWidget(self._trim_check)

        self._trim_start_spin = QSpinBox()
        self._trim_start_spin.setMinimum(0)
        self._trim_start_spin.setMaximum(_TRIM_SPINBOX_DEFAULT_MAX)
        self._trim_start_spin.setPrefix("S:")
        self._trim_start_spin.setEnabled(False)
        tb.addWidget(self._trim_start_spin)

        self._trim_end_spin = QSpinBox()
        self._trim_end_spin.setMinimum(0)
        self._trim_end_spin.setMaximum(_TRIM_SPINBOX_DEFAULT_MAX)
        self._trim_end_spin.setPrefix("E:")
        self._trim_end_spin.setEnabled(False)
        tb.addWidget(self._trim_end_spin)

        self._trim_frame_to_start_btn = QPushButton("프레임→트림시작")
        self._trim_frame_to_start_btn.clicked.connect(self._on_frame_to_trim_start)
        self._trim_frame_to_start_btn.setEnabled(False)
        tb.addWidget(self._trim_frame_to_start_btn)

        self._trim_frame_to_end_btn = QPushButton("프레임→트림끝")
        self._trim_frame_to_end_btn.clicked.connect(self._on_frame_to_trim_end)
        self._trim_frame_to_end_btn.setEnabled(False)
        tb.addWidget(self._trim_frame_to_end_btn)

    def _build_loop_controls(self, tb: QToolBar) -> None:
        """loop_count SpinBox를 툴바에 추가한다.

        WHY 0=무한: GIF 기본 동작(기존 loop=0 계약)과 동일하다.
          MP4일 때는 loop_count가 무시되므로 fmt 콤보가 MP4이면 경고를 표시한다.
        """
        tb.addWidget(QLabel("  루프:"))
        self._loop_spin = QSpinBox()
        self._loop_spin.setMinimum(0)
        self._loop_spin.setMaximum(_LOOP_COUNT_MAX)
        self._loop_spin.setValue(0)
        self._loop_spin.setToolTip("0 = 무한 반복 (GIF만 유효, MP4는 무시)")
        self._loop_spin.setEnabled(False)
        tb.addWidget(self._loop_spin)

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

    def _build_canvas_with_segment_panel(self) -> None:
        """캔버스(프레임 표시) + 구간 테이블 패널을 중앙 위젯으로 설정한다.

        WHY: SegmentTableWidget을 별도 위젯으로 캡슐화해 배선만 담당한다.
             VideoMainWindow가 비대해지지 않도록 SRP 준수.
        """
        from PySide6.QtWidgets import QHBoxLayout, QWidget

        self._canvas = FrameCanvas()
        self._canvas.clicked.connect(self._on_canvas_click)

        self._segment_table = SegmentTableWidget()

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(self._canvas, stretch=3)
        layout.addWidget(self._segment_table, stretch=1)
        self.setCentralWidget(container)

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
        """fps 입력 기본값을 원본 영상 fps로 설정하고 동적 패스트 상한을 갱신한다.

        WHY: 기본 12fps 고정이면 원본(예 25fps)과 달라 매번 수동 입력해야 한다.
             원본 fps를 기본으로 두면 '원본과 동일'이 기본이고 조절은 그대로 가능.
             set_base_fps로 SegmentTableWidget의 패스트 상한 ComboBox 비활성화 연결.
        """
        if meta.fps and meta.fps > 0:
            self._fps_spin.setValue(round(meta.fps))
            self._segment_table.set_base_fps(meta.fps)

    def _on_span_changed(self) -> None:
        """구간 변경 시 첫 프레임 미리보기와 트림 SpinBox 상한을 갱신한다.

        WHY 트림 상한 재동기화: span_end를 줄이면 트림 상한이 옛 span_len으로 남아
          trim.end > len(crops)가 되고 validate_trim에서 실패하는 UX 불일치가 생긴다.
          span 변경 시마다 즉시 재동기화해 경고와 인코딩 단계 모두 일관성을 유지한다.
        """
        if self._usecase is None:
            return
        self._load_first_frame()
        self._track_result = None
        self._boxes = None
        self._save_btn.setEnabled(False)
        self._sync_trim_spinbox_range()

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
        self._loop_spin.setEnabled(True)
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

    def _on_frame_to_start(self) -> None:
        """현재 미리보기 구간 첫 프레임(상대 0)을 선택 행의 시작으로 주입한다.

        WHY 상대 인덱스:
            export 경로의 segments는 read_span_frames(span)로 자른 crops의
            상대 인덱스 [0, n) 기준이다(build_playback_schedule(len(crops), ...)).
            절대 인덱스를 주입하면 span_start ≠ 0일 때 구간이 완전히 빗나간다.
            현재 미리보기 = span 첫 프레임 → 상대 0이 항상 옳다.
        """
        self._segment_table.set_frame_as_start(0)

    def _on_frame_to_end(self) -> None:
        """현재 구간 길이(span_end - span_start)를 선택 행의 끝으로 주입한다.

        WHY 상대 인덱스:
            상대 끝 = span 전체 길이 = span_end - span_start (= n).
            절대 인덱스(span_end)를 주입하면 crops 배열 범위를 벗어난다.
        """
        span_len = self._span_end.value() - self._span_start.value()
        relative_end = max(span_len, 0)
        self._segment_table.set_frame_as_end(relative_end)

    def _on_trim_toggled(self, state: int) -> None:
        """트림 체크박스 on/off에 따라 SpinBox 활성·비활성을 전환한다."""
        enabled = bool(state)
        self._trim_enabled = enabled
        self._trim_start_spin.setEnabled(enabled)
        self._trim_end_spin.setEnabled(enabled)
        self._trim_frame_to_start_btn.setEnabled(enabled)
        self._trim_frame_to_end_btn.setEnabled(enabled)

    def _on_frame_to_trim_start(self) -> None:
        """현재 미리보기 구간 첫 프레임(상대 0)을 트림 시작으로 주입한다.

        WHY 상대 인덱스:
          트림 좌표는 출력 crops 시퀀스 상대 [0, span_len).
          segments 버튼(_on_frame_to_start)과 동일 좌표계 — 절대 인덱스 금지.
        """
        self._trim_start_spin.setValue(0)

    def _on_frame_to_trim_end(self) -> None:
        """현재 구간 길이(span_end - span_start)를 트림 끝으로 주입한다.

        WHY 상대 인덱스:
          상대 끝 = span 전체 길이. segments _on_frame_to_end와 동일 패턴.
        """
        span_len = self._span_end.value() - self._span_start.value()
        self._trim_end_spin.setValue(max(span_len, 0))

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

        segments = self._resolve_segments()
        if segments is None:
            return  # segments 검증 오류 — QMessageBox 이미 표시됨

        fmt = self._fmt_combo.currentText().lower()
        default_name = f"output.{fmt}"
        filter_str = "GIF (*.gif)" if fmt == "gif" else "MP4 (*.mp4)"
        path, _ = QFileDialog.getSaveFileName(self, "저장", default_name, filter_str)
        if not path:
            return

        fps = float(self._fps_spin.value())
        gap_policy = _GAP_POLICY_ITEMS[self._gap_combo.currentIndex()][1]
        try:
            trim = self._resolve_trim()
        except _TrimValidationError:
            return  # 트림 오류 — QMessageBox 이미 표시됨
        loop_count = self._loop_spin.value()
        config = VideoExportConfig(
            fmt=fmt, fps=fps, gap_policy=gap_policy,
            segments=segments, trim=trim, loop_count=loop_count,
        )

        self._warn_export_issues(fmt, fps, segments, trim, loop_count)

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
        self._frame_to_start_btn.setEnabled(True)
        self._frame_to_end_btn.setEnabled(True)
        # 트림 SpinBox 범위를 span 길이로 제한 — 로드 시점에 초기화
        self._trim_end_spin.setValue(end - self._span_start.value())
        self._sync_trim_spinbox_range()
        self._trim_check.setEnabled(True)

    def _sync_trim_spinbox_range(self) -> None:
        """현재 span 길이 기준으로 트림 SpinBox 상한을 재동기화한다(순수 UI).

        WHY: span_end를 줄인 뒤 트림 상한이 옛 span_len으로 남으면
          trim.end > len(crops) → validate_trim 실패 (UX 불일치, reviewer [중요 2]).
          span 변경 시마다 이 헬퍼를 호출해 항상 현재 span_len이 상한이 되게 한다.
        """
        span_len = self._span_end.value() - self._span_start.value()
        safe_len = max(0, span_len)
        self._trim_start_spin.setMaximum(max(0, safe_len - 1))
        self._trim_end_spin.setMaximum(safe_len)

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

    def _resolve_segments(self):
        """구간 테이블에서 SpeedSegment 튜플을 읽고 검증한다.

        검증 오류(ValueError) 시 한국어 QMessageBox를 표시하고 None 반환.
        WHY: export 버튼 클릭 시 구간 오류를 조기 감지해 저장 차단.
        """
        try:
            return self._segment_table.to_segments()
        except ValueError as exc:
            QMessageBox.warning(
                self,
                "구간 설정 오류",
                f"구간 설정에 오류가 있습니다.\n\n{exc}",
            )
            return None

    def _warn_export_issues(self, fmt, fps, segments, trim, loop_count) -> None:
        """export 직전 폭증·클램프·MP4루프 경고를 일괄 처리한다.

        WHY 추출: _on_export 20줄 규칙 준수. 세 경고 호출을 묶어 단일 책임 분리.
        WHY 트림-로컬 segments: 트림 후 실제 출력 기준으로 경고해야 오경고가 없다.
        """
        _warn_mp4_loop_if_needed(self, fmt, loop_count)
        trim_len = _trim_length(trim, self._frames)
        local_segs = shift_segments_into_trim(segments, trim)
        if local_segs:
            self._warn_frame_count_overflow_if_needed(fps, local_segs, trim_len)
        if fmt == "gif" and local_segs:
            self._warn_gif_clamp_if_needed(fps, local_segs, trim_len)

    def _resolve_trim(self) -> TrimRange | None:
        """트림 체크 상태에 따라 TrimRange 또는 None을 반환한다.

        체크 해제이면 None(항등, 무회귀).
        체크 시 start >= end이면 한국어 QMessageBox를 표시하고 _TrimValidationError.
        WHY: validate_trim 오류를 export 핸들러에서 흡수해 앱이 죽지 않게 한다.
        """
        if not self._trim_enabled:
            return None
        start = self._trim_start_spin.value()
        end = self._trim_end_spin.value()
        if start >= end:
            QMessageBox.warning(
                self, "트림 설정 오류",
                f"트림 끝({end})이 시작({start})보다 커야 합니다.",
            )
            raise _TrimValidationError
        return TrimRange(start=start, end=end)

    def _warn_frame_count_overflow_if_needed(
        self, fps: float, segments, n_frames: int | None = None
    ) -> None:
        """슬로우 구간으로 인한 출력 프레임 수 폭증 시 경고를 표시한다.

        WHY 트림 보정: n_frames에 트림 후 길이(M)를 전달하면 트림으로 출력이 줄었는데
          전체 기준으로 오경고하는 문제를 방지한다(reviewer R7).
          segments는 이미 트림-로컬로 평행이동된 값을 받는다.
        """
        if self._frames is None:
            return
        n_base = n_frames if n_frames is not None else len(self._frames)
        if n_base == 0:
            return
        n_estimated = estimate_output_frame_count(n_base, segments, fps)
        ratio = n_estimated / n_base
        if ratio >= _FRAME_COUNT_OVERFLOW_RATIO:
            self._set_status(
                f"경고: 출력 프레임 수 {n_estimated}개 (원본의 {ratio:.1f}배) — 용량/시간 폭증"
            )
            QMessageBox.warning(
                self,
                "출력 프레임 폭증 경고",
                f"슬로우모션 구간 적용 시 출력 프레임이 {n_estimated}개로\n"
                f"기준({n_base}개)의 {ratio:.1f}배가 됩니다.\n\n"
                "인코딩 시간과 파일 용량이 크게 늘어날 수 있습니다.\n"
                "구간 설정을 줄이거나 저속 배속을 높여 주세요.",
            )

    def _warn_gif_clamp_if_needed(
        self, fps: float, segments, n_frames: int | None = None
    ) -> None:
        """GIF 출력 시 20ms 클램프 대상 프레임이 있으면 경고를 표시한다.

        WHY 트림 보정: n_frames에 트림 후 길이를 전달해 오경고를 방지한다(reviewer R7).
          segments는 이미 트림-로컬로 평행이동된 값을 받는다.
        """
        if self._frames is None:
            return
        n_base = n_frames if n_frames is not None else len(self._frames)
        if n_base == 0:
            return  # WHY: _warn_frame_count_overflow와 대칭 가드(제안)
        schedule = build_playback_schedule(n_base, list(segments), fps)
        _, clamped_indices = clamp_durations_for_gif(schedule)

        if clamped_indices:
            n_clamped = len(clamped_indices)
            self._set_status(
                f"GIF 경고: {n_clamped}개 프레임이 20ms로 클램프됩니다."
            )
            QMessageBox.information(
                self,
                "GIF 속도 제한 안내",
                f"{n_clamped}개 프레임의 표시 시간이 너무 짧아 20ms로 조정됩니다.\n"
                "GIF 뷰어 호환성을 위한 자동 처리입니다.",
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


def _trim_length(trim: "TrimRange | None", frames) -> int | None:
    """트림 지정 시 트림 후 길이를, 미지정 시 None을 반환한다(순수).

    WHY: 경고 헬퍼에 트림 후 길이를 전달해 전체 기준 오경고를 방지한다(reviewer R7).
         None이면 경고 헬퍼가 전체 frames 길이를 기준으로 사용한다.
    """
    if trim is None or frames is None:
        return None
    return trim.end - trim.start


def _warn_mp4_loop_if_needed(parent, fmt: str, loop_count: int) -> None:
    """MP4 + loop_count != 0이면 한국어 경고 메시지를 표시한다(순수 조건).

    WHY: MP4 컨테이너는 루프 메타를 지원하지 않아 loop_count가 무시된다.
         사용자에게 사전 안내해 기대와 다른 결과(루프 없음)를 방지한다.
    """
    if fmt == "mp4" and loop_count != 0:
        QMessageBox.information(
            parent,
            "MP4 루프 설정 안내",
            "MP4는 루프 설정이 무시됩니다.\n"
            "루프 재생을 원하시면 GIF 형식을 사용해 주세요.",
        )


class _TrimValidationError(Exception):
    """트림 범위 검증 실패 시 _resolve_trim에서 발생하는 내부 예외.

    WHY: None 반환은 "트림 미사용(체크 해제)"과 "오류" 두 경우가 구분 안 된다.
         전용 예외로 export 핸들러가 두 경우를 명확히 구분하게 한다.
    """
