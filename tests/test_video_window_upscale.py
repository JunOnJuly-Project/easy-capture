"""VideoMainWindow 업스케일 UI 배선 테스트 (offscreen PySide6).

비디오 모드 export에 업스케일을 결합하는 UI 계약을 검증한다.
이미지 모드 ImageMainWindow 업스케일 패턴과 대칭(체크박스·배율 콤보·백엔드 캐시).

실제 Swin2SR(GPU)은 호출하지 않는다 — Fake 팩토리/백엔드로 배선·캐시·전달만 검증.
실모델 검증은 Colab 노트북 후행(ADR 0009/0018 경로).

모달 없음(_resolve_upscaler·토글은 다이얼로그 미사용) — hang 우려 없음.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="PySide6 미설치 — UI 위젯 테스트 불가")

from PySide6.QtWidgets import QApplication  # noqa: E402

from easy_capture.infra.device import UPSCALE_MODELS  # noqa: E402
from easy_capture.ui.video_window import VideoMainWindow  # noqa: E402

_app: QApplication | None = None


def _get_app() -> QApplication:
    """QApplication 싱글턴 — 위젯 생성 전 1회 보장."""
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


class _FakeUpscaler:
    """UpscaleBackend 더블 — repo 식별만(실 확대 없음)."""

    def __init__(self, repo: str) -> None:
        self.repo = repo


class _FakeUpscalerFactory:
    """model → _FakeUpscaler 생성 팩토리 + 호출 카운터."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, model):
        self.calls += 1
        return _FakeUpscaler(model.repo)


def _make_window(with_factory: bool = True) -> VideoMainWindow:
    """업스케일 팩토리 주입 여부를 골라 VideoMainWindow를 만든다."""
    _get_app()
    if not with_factory:
        return VideoMainWindow(lambda _p: None)
    return VideoMainWindow(
        lambda _p: None,
        upscaler_factory=_FakeUpscalerFactory(),
        upscale_catalog=UPSCALE_MODELS,
    )


# ===========================================================================
# 1. 위젯 존재/부재 — 팩토리 주입 여부에 따라
# ===========================================================================
class TestUpscaleWidgetPresence:
    def test_팩토리_주입_시_업스케일_체크박스가_존재한다(self):
        window = _make_window(with_factory=True)
        assert hasattr(window, "_upscale_check")
        assert hasattr(window, "_upscale_combo")

    def test_팩토리_없으면_업스케일_위젯이_없다_무회귀(self):
        window = _make_window(with_factory=False)
        assert not hasattr(window, "_upscale_check")
        assert not hasattr(window, "_upscale_combo")

    def test_콤보_항목수가_카탈로그와_일치한다(self):
        window = _make_window(with_factory=True)
        assert window._upscale_combo.count() == len(UPSCALE_MODELS)

    def test_체크박스는_추적_전_비활성이다(self):
        window = _make_window(with_factory=True)
        assert not window._upscale_check.isEnabled()


# ===========================================================================
# 2. 토글·모델 선택 동작
# ===========================================================================
class TestUpscaleToggle:
    def test_토글_on_시_콤보가_활성화된다(self):
        window = _make_window(with_factory=True)
        window._on_upscale_toggled(2)  # Qt.Checked
        assert window._upscale_on is True
        assert window._upscale_combo.isEnabled()

    def test_토글_off_시_콤보가_비활성화된다(self):
        window = _make_window(with_factory=True)
        window._on_upscale_toggled(2)
        window._on_upscale_toggled(0)  # Qt.Unchecked
        assert window._upscale_on is False
        assert not window._upscale_combo.isEnabled()

    def test_모델_변경_시_선택모델과_캐시가_갱신된다(self):
        window = _make_window(with_factory=True)
        # 첫 모델로 캐시를 채운 뒤 다른 모델로 바꾸면 캐시가 무효화돼야 한다
        window._on_upscale_toggled(2)
        first = window._resolve_upscaler()
        assert first is not None

        window._on_upscale_model_changed(1)  # 두 번째 모델
        assert window._upscale_model == UPSCALE_MODELS[1]
        assert window._cached_upscaler is None  # 캐시 무효화


# ===========================================================================
# 3. _resolve_upscaler — export에 넘길 업스케일러 결정
# ===========================================================================
class TestResolveUpscaler:
    def test_토글_off면_None을_반환한다_무회귀(self):
        window = _make_window(with_factory=True)
        assert window._resolve_upscaler() is None

    def test_토글_on이면_팩토리로_백엔드를_생성한다(self):
        window = _make_window(with_factory=True)
        window._on_upscale_toggled(2)
        upscaler = window._resolve_upscaler()
        assert isinstance(upscaler, _FakeUpscaler)
        assert upscaler.repo == UPSCALE_MODELS[0].repo

    def test_같은_모델_연속_호출_시_백엔드를_재사용한다_캐시(self):
        factory = _FakeUpscalerFactory()
        _get_app()
        window = VideoMainWindow(
            lambda _p: None,
            upscaler_factory=factory,
            upscale_catalog=UPSCALE_MODELS,
        )
        window._on_upscale_toggled(2)

        first = window._resolve_upscaler()
        second = window._resolve_upscaler()

        assert first is second        # 동일 인스턴스 재사용
        assert factory.calls == 1     # 팩토리는 1회만 호출
