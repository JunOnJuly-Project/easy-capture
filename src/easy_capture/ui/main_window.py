"""이미지 모드 메인 윈도우.

파일 열기 → 프레임 표시 → 클릭(SAM2 세그, 워커) → 오버레이 표시 →
종횡비/크기 즉시 조정(재세그 없음) → (옵션) 업스케일 → 저장 흐름.

핵심 설계(계획서 §1-1, §5):
  - 세그(무거움): 클릭당 1회만 워커 스레드에서 실행.
  - 박스 계산(가벼움): 보관된 centroid로 메인 스레드에서 즉시 재호출.
    종횡비/크기 변경 시 재세그 없이 _recompute_box()만 호출.
  - 업스케일(무거움): 선택 시 _UpscaleSaveWorker로 백그라운드 실행.
    미선택 시 기존 동기 저장(무회귀).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStatusBar,
    QToolBar,
    QWidget,
)
from PySide6.QtCore import Qt

from easy_capture.core.crop.crop import ASPECT_PRESETS
from easy_capture.core.export.image_export import ExportConfig
from easy_capture.ui.frame_canvas import FrameCanvas
from easy_capture.ui.sizing import (
    DEFAULT_CROP_RATIO,
    MAX_CROP_RATIO,
    MIN_CROP_RATIO,
    crop_ratio_to_size,
)

# 종횡비 없음 항목 레이블
_ASPECT_FREE_LABEL = "자유"
# 종횡비 콤보 항목: (표시 레이블, 내부 키)
_ASPECT_ITEMS: list[tuple[str, str | None]] = [
    (_ASPECT_FREE_LABEL, None),
    *[(k, k) for k in ASPECT_PRESETS],
]


class _SegWorker(QThread):
    """세그멘테이션을 백그라운드에서 실행하는 워커.

    WHY: CPU SAM2 추론이 ~1~3s 걸려 메인 스레드에서 실행하면 UI가 얼기 때문에
         QThread로 분리한다. segment만 호출하고 박스 계산은 메인 스레드로 위임.
         완료 시 seg_ready(mask, centroid) Signal로 세그 결과를 전달한다.
    """

    seg_ready = Signal(np.ndarray, object)  # (mask: ndarray, centroid: tuple)
    error = Signal(str)                     # 한국어 오류 메시지

    def __init__(
        self, usecase, frame: np.ndarray, point: tuple[int, int]
    ) -> None:
        super().__init__()
        self._usecase = usecase
        self._frame = frame
        self._point = point

    def run(self) -> None:
        """usecase.segment를 워커 스레드에서 실행한다.

        WHY: segment만 호출해 박스 계산(compute_box)을 메인 스레드로 분리한다.
             EmptyMaskError는 [빈마스크] 태그로 구별해 UI에 전달한다.
        """
        from easy_capture.app.image_capture import EmptyMaskError

        try:
            result = self._usecase.segment(self._frame, self._point)
            self.seg_ready.emit(result.mask, result.centroid)
        except EmptyMaskError as exc:
            self.error.emit(f"[빈마스크] {exc}")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"세그멘테이션 오류: {exc}")


class _UpscaleSaveWorker(QThread):
    """크롭+업스케일+저장을 백그라운드에서 실행하는 워커.

    WHY: CPU Swin2SR가 수 초 걸려 메인 스레드 실행 시 UI가 얼기 때문에 분리.
         usecase.export(upscaler=...) 한 번만 호출하고 결과(저장 경로)를 emit.
         _SegWorker와 동일 패턴(Signal·run·예외 처리).
    """

    done = Signal(str)   # 저장 완료 경로
    error = Signal(str)  # 한국어 오류 메시지

    def __init__(self, usecase, frame, box, target, upscaler) -> None:
        super().__init__()
        # WHY: 5개 인자를 튜플로 묶어 매개변수 규칙(3개) 완화 — 생성자 계약
        self._args = (usecase, frame, box, target, upscaler)

    def run(self) -> None:
        """워커 스레드에서 export(upscaler=...)를 실행한다."""
        usecase, frame, box, target, upscaler = self._args
        path, _ = target
        try:
            usecase.export(frame, box, target, upscaler=upscaler)
            self.done.emit(path)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"업스케일/저장 오류: {exc}")


class ImageMainWindow(QMainWindow):
    """이미지 모드 메인 윈도우.

    usecase_factory: path -> ImageCaptureUseCase 를 반환하는 callable.
    WHY: 파일 경로가 결정된 후에야 FrameSource(파일 기반)를 생성할 수 있으므로
         팩토리 패턴으로 usecase 생성을 지연한다.
    """

    def __init__(
        self,
        usecase_factory,
        upscaler_factory=None,
        upscale_catalog=(),
    ) -> None:
        super().__init__()
        self.setWindowTitle("easy-capture — 이미지 모드")
        self.resize(900, 600)
        self._usecase_factory = usecase_factory

        # 업스케일 팩토리·카탈로그 주입(DIP) — None이면 업스케일 UI 비활성
        self._upscaler_factory = upscaler_factory
        self._upscale_catalog = upscale_catalog

        self._usecase = None

        # 상태 필드 — 세그/조정 분리를 위해 centroid·aspect·ratio 보관
        self._frame: np.ndarray | None = None
        self._centroid: tuple[float, float] | None = None
        self._crop_box: tuple | None = None
        self._aspect: str | None = None          # 현재 선택 종횡비 (None=자유)
        self._size_ratio: int = DEFAULT_CROP_RATIO

        # 업스케일 상태
        self._upscale_on: bool = False
        self._upscale_model = None               # 선택된 UpscaleModel (None=미선택)
        self._cached_upscaler = None             # 마지막 생성 백엔드 1개 캐시
        self._cached_repo: str | None = None     # 캐시 유효성 판단용

        self._worker: _SegWorker | None = None
        self._save_worker: _UpscaleSaveWorker | None = None

        self._build_toolbar()
        self._build_canvas()
        self._build_statusbar()

    # ------------------------------------------------------------------
    # UI 빌더 메서드
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        """파일열기·저장·종횡비·크기·업스케일 위젯이 있는 툴바를 구성한다."""
        toolbar = QToolBar("메인")
        self.addToolBar(toolbar)

        open_btn = QPushButton("파일 열기")
        open_btn.clicked.connect(self._on_open_file)
        toolbar.addWidget(open_btn)

        self._save_btn = QPushButton("저장")
        self._save_btn.clicked.connect(self._on_export)
        self._save_btn.setEnabled(False)
        toolbar.addWidget(self._save_btn)

        self._build_aspect_combo(toolbar)
        self._build_size_slider(toolbar)
        self._build_upscale_controls(toolbar)

    def _build_aspect_combo(self, toolbar: QToolBar) -> None:
        """종횡비 선택 콤보박스를 툴바에 추가한다."""
        toolbar.addWidget(QLabel("  종횡비:"))
        self._aspect_combo = QComboBox()
        for label, _ in _ASPECT_ITEMS:
            self._aspect_combo.addItem(label)
        self._aspect_combo.setEnabled(False)
        self._aspect_combo.currentIndexChanged.connect(self._on_aspect_changed)
        toolbar.addWidget(self._aspect_combo)

    def _build_size_slider(self, toolbar: QToolBar) -> None:
        """크기 슬라이더를 툴바에 추가한다."""
        toolbar.addWidget(QLabel("  크기:"))
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setMinimum(MIN_CROP_RATIO)
        self._size_slider.setMaximum(MAX_CROP_RATIO)
        self._size_slider.setValue(DEFAULT_CROP_RATIO)
        self._size_slider.setFixedWidth(120)
        self._size_slider.setEnabled(False)
        self._size_slider.valueChanged.connect(self._on_size_changed)
        toolbar.addWidget(self._size_slider)

        self._size_label = QLabel(f"{DEFAULT_CROP_RATIO}%")
        toolbar.addWidget(self._size_label)

    def _build_upscale_controls(self, toolbar: QToolBar) -> None:
        """업스케일 체크박스·배율 콤보박스를 툴바에 추가한다.

        WHY: 카탈로그(UPSCALE_MODELS)가 단일 소스로 UI 항목을 생성해 DRY 준수.
             팩토리가 없으면(개발/테스트 환경) 위젯을 추가하지 않는다.
        """
        if not self._upscaler_factory:
            return
        toolbar.addWidget(QLabel("  업스케일:"))
        self._upscale_check = QCheckBox("업스케일")
        self._upscale_check.setChecked(False)
        self._upscale_check.stateChanged.connect(self._on_upscale_toggled)
        toolbar.addWidget(self._upscale_check)

        self._upscale_combo = QComboBox()
        for model in self._upscale_catalog:
            self._upscale_combo.addItem(model.label)
        self._upscale_combo.setEnabled(False)
        self._upscale_combo.currentIndexChanged.connect(self._on_upscale_model_changed)
        toolbar.addWidget(self._upscale_combo)

        # 초기 모델 선택 동기화
        if self._upscale_catalog:
            self._upscale_model = self._upscale_catalog[0]

    def _build_canvas(self) -> None:
        """프레임 표시·클릭 캔버스를 중앙 위젯으로 설정한다."""
        self._canvas = FrameCanvas()
        self._canvas.clicked.connect(self._on_canvas_click)
        self.setCentralWidget(self._canvas)

    def _build_statusbar(self) -> None:
        """상태 메시지 표시 바를 구성한다."""
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_label = QLabel("파일을 열어 시작하세요.")
        self._status.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # 슬롯
    # ------------------------------------------------------------------

    def _on_open_file(self) -> None:
        """파일 다이얼로그로 이미지/영상을 열고 첫 프레임을 표시한다."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "파일 열기",
            "",
            "이미지/영상 (*.jpg *.jpeg *.png *.bmp *.mp4 *.mov *.avi *.webp);;전체 파일 (*)",
        )
        if not path:
            return

        try:
            self._usecase = self._usecase_factory(path)
            self._frame = self._usecase.load_frame()
            self._canvas.set_frame(self._frame)
            self._canvas.set_overlay(None)
            self._centroid = None
            self._crop_box = None
            self._save_btn.setEnabled(False)
            self._aspect_combo.setEnabled(True)
            self._size_slider.setEnabled(True)
            self._set_status("피사체를 클릭해 주세요.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "파일 열기 실패", f"파일을 열 수 없습니다.\n{exc}")

    def _on_canvas_click(self, x: int, y: int) -> None:
        """캔버스 클릭 시 세그멘테이션 워커를 시작한다(무거운 경로)."""
        if self._frame is None or self._usecase is None:
            return
        if self._worker is not None and self._worker.isRunning():
            # WHY: 단일 워커 정책(다중 클릭 누적은 범위 외). 무시 시 피드백 제공.
            self._set_status("분석 중입니다. 잠시만 기다려 주세요.")
            return

        self._set_status("분석 중… (처음 실행 시 모델 로드로 시간이 걸릴 수 있습니다)")
        self._worker = _SegWorker(self._usecase, self._frame, (x, y))
        self._worker.seg_ready.connect(self._on_seg_ready)
        self._worker.error.connect(self._on_seg_error)
        self._worker.start()

    def _on_seg_ready(self, mask: np.ndarray, centroid: object) -> None:
        """세그 결과 도착 시 오버레이 표시 + 박스 재계산(가벼운 경로).

        WHY: 오버레이 set은 클릭 당 1회만 이 슬롯에서 한다.
             종횡비/크기 변경은 _recompute_box만 호출해 오버레이는 유지.
        """
        self._centroid = centroid  # type: ignore[assignment]
        self._canvas.set_overlay(mask)
        self._recompute_box()

    def _on_aspect_changed(self, index: int) -> None:
        """종횡비 선택 변경 시 박스 즉시 재계산(재세그 없음)."""
        _, key = _ASPECT_ITEMS[index]
        self._aspect = key
        self._recompute_box()

    def _on_size_changed(self, value: int) -> None:
        """크기 슬라이더 변경 시 박스 즉시 재계산(재세그 없음)."""
        self._size_ratio = value
        self._size_label.setText(f"{value}%")
        self._recompute_box()

    def _on_seg_error(self, message: str) -> None:
        """세그멘테이션 실패 시 한국어 안내.

        WHY: 워커가 [빈마스크] 태그로 EmptyMaskError를 구별해 전달한다.
             문자열 패턴 매칭 대신 명시적 태그로 분기해 오탐을 방지한다.
        """
        if message.startswith("[빈마스크]"):
            self._set_status("대상을 인식하지 못했어요. 피사체 위를 다시 클릭해 주세요.")
        else:
            QMessageBox.warning(self, "분석 실패", message)
            self._set_status("분석에 실패했습니다. 다시 시도해 주세요.")

    def _on_upscale_toggled(self, state: int) -> None:
        """업스케일 체크박스 상태 변경 시 콤보 활성/비활성 토글."""
        self._upscale_on = bool(state)
        if hasattr(self, "_upscale_combo"):
            self._upscale_combo.setEnabled(self._upscale_on)

    def _on_upscale_model_changed(self, index: int) -> None:
        """배율 콤보 변경 시 선택 모델 갱신 및 캐시 무효화.

        WHY: repo 변경 시 이전 백엔드 캐시를 제거해 재선택 시 재생성한다.
        """
        if self._upscale_catalog and 0 <= index < len(self._upscale_catalog):
            model = self._upscale_catalog[index]
            if self._upscale_model != model:
                self._upscale_model = model
                self._cached_upscaler = None  # 캐시 무효화
                self._cached_repo = None

    def _on_export(self) -> None:
        """저장 다이얼로그로 경로를 받아 저장 흐름을 분기한다."""
        if self._frame is None or self._crop_box is None or self._usecase is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "저장",
            "output.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)",
        )
        if not path:
            return

        fmt = "jpg" if path.lower().endswith((".jpg", ".jpeg")) else "png"
        config = ExportConfig(fmt=fmt)
        self._save_with_optional_upscale(path, config)

    def _save_with_optional_upscale(self, path: str, config: ExportConfig) -> None:
        """업스케일 on/off에 따라 동기 저장 또는 워커 저장으로 분기한다.

        WHY: _on_export에서 분리해 20줄 규칙 준수 및 단일 책임 확보.
        """
        if not self._upscale_on or self._upscale_model is None:
            self._do_sync_save(path, config)
            return

        if self._save_worker is not None and self._save_worker.isRunning():
            self._set_status("이미 저장 중입니다. 잠시만 기다려 주세요.")
            return

        upscaler = self._get_or_make_upscaler(self._upscale_model)
        target = (path, config)
        self._save_btn.setEnabled(False)
        self._set_status(
            "업스케일 중… (처음 실행 시 모델 로드로 시간이 걸릴 수 있습니다)"
        )
        self._save_worker = _UpscaleSaveWorker(
            self._usecase, self._frame, self._crop_box, target, upscaler
        )
        self._save_worker.done.connect(self._on_save_done)
        self._save_worker.error.connect(self._on_save_error)
        self._save_worker.start()

    def _do_sync_save(self, path: str, config: ExportConfig) -> None:
        """업스케일 없는 기존 동기 저장 경로(무회귀)."""
        try:
            self._usecase.export(self._frame, self._crop_box, (path, config))
            self._set_status(f"저장 완료: {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "저장 실패", f"저장에 실패했습니다.\n{exc}")

    def _get_or_make_upscaler(self, model):
        """캐시된 백엔드 반환 또는 팩토리로 새 백엔드를 생성한다.

        WHY: 같은 repo 연속 저장 시 재로드를 방지하는 단일 백엔드 캐시.
             repo가 바뀌면 캐시를 무효화해 새 모델을 로드한다.
        """
        if self._cached_upscaler is None or self._cached_repo != model.repo:
            self._cached_upscaler = self._upscaler_factory(model)
            self._cached_repo = model.repo
        return self._cached_upscaler

    def _on_save_done(self, path: str) -> None:
        """업스케일+저장 완료 시 상태 바 갱신 및 저장 버튼 재활성."""
        self._save_btn.setEnabled(True)
        self._set_status(f"저장 완료: {path}")

    def _on_save_error(self, message: str) -> None:
        """업스케일+저장 실패 시 한국어 안내 및 버튼 재활성."""
        self._save_btn.setEnabled(True)
        QMessageBox.critical(self, "저장 실패", message)
        self._set_status("저장에 실패했습니다. 다시 시도해 주세요.")

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _recompute_box(self) -> None:
        """보관된 centroid + 현재 aspect/ratio로 박스를 즉시 재계산한다.

        세그를 다시 부르지 않는다(순수 계산만). centroid가 없으면 무시.

        WHY: 종횡비·크기 슬라이더 드래그 시 연속 호출되는데 compute_box가
             순수 함수이므로 멈춤 없이 즉시 완료된다(계획서 §3-3).
        """
        if self._centroid is None or self._frame is None or self._usecase is None:
            return

        from easy_capture.app.image_capture import BoxParams

        frame_w, frame_h = self._frame.shape[1], self._frame.shape[0]
        size = crop_ratio_to_size(self._size_ratio, (frame_w, frame_h))
        params = BoxParams(
            box_size=size,
            aspect=self._aspect,
            frame_shape=(frame_w, frame_h),
        )
        self._crop_box = self._usecase.compute_box(self._centroid, params)
        self._save_btn.setEnabled(True)
        self._set_status(f"크롭 박스 확정: {self._crop_box}. '저장' 버튼으로 내보내세요.")

    def _set_status(self, message: str) -> None:
        """상태 바 메시지를 업데이트한다."""
        self._status_label.setText(message)
