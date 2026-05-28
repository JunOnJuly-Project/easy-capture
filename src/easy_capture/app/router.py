"""모드 선택 → 메인 윈도우 라우팅 + 의존성 조립 루트(composition root).

AppRouter가 구체 의존성(Sam2ImageBackend, open_source, Swin2srUpscaleBackend)을 조립해
ImageCaptureUseCase와 ImageMainWindow에 주입한다.
UI·유스케이스·infra를 처음으로 연결하는 단일 지점.
"""
from __future__ import annotations

from PySide6.QtWidgets import QApplication, QMessageBox

from easy_capture.ui.mode_select import ModeSelectWindow


class AppRouter:
    """모드 선택 시그널을 받아 적절한 메인 윈도우를 조립·표시한다.

    WHY: composition root 패턴 — 구체 타입 조립을 진입점 한 곳에 집중시켜
         나머지 레이어가 Protocol(추상)에만 의존하도록 한다(DIP).
    """

    def __init__(self, app: QApplication) -> None:
        self._app = app
        self._mode_window: ModeSelectWindow | None = None
        self._main_window = None

    def start(self) -> None:
        """모드 선택 창을 표시하고 시그널을 연결한다."""
        self._mode_window = ModeSelectWindow()
        self._mode_window.mode_selected.connect(self._on_mode)
        self._mode_window.show()

    def _on_mode(self, mode: str) -> None:
        """선택된 모드에 맞는 메인 윈도우를 조립·표시한다."""
        if mode == "image":
            self._launch_image_mode()
        elif mode == "gif":
            QMessageBox.information(
                self._mode_window,
                "준비 중",
                "GIF(움짤) 모드는 아직 준비 중입니다.\n다음 업데이트를 기대해 주세요!",
            )
        else:
            QMessageBox.warning(
                self._mode_window,
                "알 수 없는 모드",
                f"'{mode}' 모드는 지원하지 않습니다.",
            )

    def _launch_image_mode(self) -> None:
        """이미지 모드 의존성을 조립하고 메인 윈도우를 표시한다."""
        from easy_capture.infra.device import UPSCALE_MODELS, detect_device
        from easy_capture.ui.main_window import ImageMainWindow

        device = detect_device()
        usecase_factory = self._build_usecase_factory(device)
        upscaler_factory = self._build_upscaler_factory(device)
        self._main_window = ImageMainWindow(
            usecase_factory,
            upscaler_factory=upscaler_factory,
            upscale_catalog=UPSCALE_MODELS,
        )
        self._main_window.show()
        if self._mode_window:
            self._mode_window.close()

    def _build_usecase_factory(self, device: str):
        """파일 경로를 받아 ImageCaptureUseCase를 생성하는 팩토리를 반환한다.

        WHY: 파일 경로가 결정된 후에야 FrameSource를 만들 수 있으므로
             클로저로 팩토리를 반환한다. 백엔드는 한 번만 생성해 재사용한다.
        """
        from easy_capture.app.image_capture import ImageCaptureUseCase
        from easy_capture.infra.device import select_sam2_repo
        from easy_capture.infra.sam2_image_backend import Sam2ImageBackend
        from easy_capture.infra.video_io import open_source

        repo = select_sam2_repo(device)
        # 백엔드는 지연 로드 — 생성자에서 모델 로드 안 함(ADR 0007)
        backend = Sam2ImageBackend(repo=repo, device=device)

        def factory(path: str) -> ImageCaptureUseCase:
            source = open_source(path)
            return ImageCaptureUseCase(source=source, backend=backend)

        return factory

    def _build_upscaler_factory(self, device: str):
        """UpscaleModel → Swin2srUpscaleBackend 생성 팩토리를 반환한다.

        WHY: main_window는 transformers/torch를 직접 import하지 않는다(DIP).
             router(composition root)가 구체 백엔드 생성 책임을 가진다.
             지연 로드 백엔드이므로 생성 자체는 가볍다.
        """
        from easy_capture.infra.swin2sr_upscale_backend import Swin2srUpscaleBackend

        def make(model):  # model: UpscaleModel
            return Swin2srUpscaleBackend(model.repo, device, model.scale)

        return make
