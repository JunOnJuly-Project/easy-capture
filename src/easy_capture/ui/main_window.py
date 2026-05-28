"""이미지 모드 메인 윈도우.

파일 열기 → 프레임 표시 → 클릭(SAM2 세그) → 저장 흐름.
세그멘테이션은 QThread 워커로 비블로킹 처리한다(CPU SAM2 ~1~3s).
빈 마스크·오류 시 한국어 안내.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QToolBar,
    QWidget,
)

from easy_capture.core.export.image_export import ExportConfig
from easy_capture.ui.frame_canvas import FrameCanvas


class _SegWorker(QThread):
    """세그멘테이션·크롭박스 산출을 백그라운드에서 실행하는 워커.

    WHY: CPU SAM2 추론이 ~1~3s 걸려 메인 스레드에서 실행하면 UI가 얼기 때문에
         QThread로 분리한다. 완료 시 box_ready Signal로 결과를 전달한다.
    """

    box_ready = Signal(tuple)   # (x1, y1, x2, y2)
    error = Signal(str)         # 한국어 오류 메시지

    def __init__(self, usecase, frame: np.ndarray, request) -> None:
        super().__init__()
        self._usecase = usecase
        self._frame = frame
        self._request = request

    def run(self) -> None:
        """make_crop_box를 워커 스레드에서 실행한다.

        EmptyMaskError는 한국어 안내 접두사로 구별해 UI에 전달한다.
        WHY: main_window가 EmptyMaskError를 문자열 패턴이 아닌
             의미론적으로 식별할 수 있도록 접두사 태그를 붙인다.
        """
        from easy_capture.app.image_capture import EmptyMaskError

        try:
            box = self._usecase.make_crop_box(self._frame, self._request)
            self.box_ready.emit(box)
        except EmptyMaskError as exc:
            self.error.emit(f"[빈마스크] {exc}")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"세그멘테이션 오류: {exc}")


class ImageMainWindow(QMainWindow):
    """이미지 모드 메인 윈도우.

    usecase_factory: path -> ImageCaptureUseCase 를 반환하는 callable.
    WHY: 파일 경로가 결정된 후에야 FrameSource(파일 기반)를 생성할 수 있으므로
         팩토리 패턴으로 usecase 생성을 지연한다.
    """

    def __init__(self, usecase_factory) -> None:
        super().__init__()
        self.setWindowTitle("easy-capture — 이미지 모드")
        self.resize(900, 600)
        self._usecase_factory = usecase_factory
        self._usecase = None
        self._frame: np.ndarray | None = None
        self._crop_box: tuple | None = None
        self._worker: _SegWorker | None = None

        self._build_toolbar()
        self._build_canvas()
        self._build_statusbar()

    # ------------------------------------------------------------------
    # UI 빌더 메서드
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        """파일열기·저장 버튼이 있는 툴바를 구성한다."""
        toolbar = QToolBar("메인")
        self.addToolBar(toolbar)

        open_btn = QPushButton("파일 열기")
        open_btn.clicked.connect(self._on_open_file)
        toolbar.addWidget(open_btn)

        self._save_btn = QPushButton("저장")
        self._save_btn.clicked.connect(self._on_export)
        self._save_btn.setEnabled(False)
        toolbar.addWidget(self._save_btn)

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
            self._crop_box = None
            self._save_btn.setEnabled(False)
            self._set_status("피사체를 클릭해 주세요.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "파일 열기 실패", f"파일을 열 수 없습니다.\n{exc}")

    def _on_canvas_click(self, x: int, y: int) -> None:
        """캔버스 클릭 시 세그멘테이션 워커를 시작한다."""
        if self._frame is None or self._usecase is None:
            return
        if self._worker is not None and self._worker.isRunning():
            return  # 이전 워커가 실행 중이면 무시

        from easy_capture.app.image_capture import CropRequest

        request = CropRequest(
            point=(x, y),
            box_size=(300, 300),
            aspect=None,
        )
        self._set_status("분석 중… (처음 실행 시 모델 로드로 시간이 걸릴 수 있습니다)")
        self._worker = _SegWorker(self._usecase, self._frame, request)
        self._worker.box_ready.connect(self._on_box_ready)
        self._worker.error.connect(self._on_seg_error)
        self._worker.start()

    def _on_box_ready(self, box: tuple) -> None:
        """워커에서 크롭 박스가 도착하면 오버레이 표시 및 저장 버튼 활성화."""
        self._crop_box = box
        self._save_btn.setEnabled(True)
        self._set_status(f"크롭 박스 확정: {box}. '저장' 버튼으로 내보내세요.")

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

    def _on_export(self) -> None:
        """저장 다이얼로그로 경로를 받아 크롭·저장한다."""
        if self._frame is None or self._crop_box is None or self._usecase is None:
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "저장",
            "output.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)",
        )
        if not path:
            return

        fmt = "jpg" if path.lower().endswith((".jpg", ".jpeg")) else "png"
        config = ExportConfig(fmt=fmt)
        try:
            self._usecase.export(self._frame, self._crop_box, (path, config))
            self._set_status(f"저장 완료: {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "저장 실패", f"저장에 실패했습니다.\n{exc}")

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _set_status(self, message: str) -> None:
        """상태 바 메시지를 업데이트한다."""
        self._status_label.setText(message)
