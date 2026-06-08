"""VideoMainWindow 구간/트림 SpinBox mm:ss suffix 통합 테스트 (offscreen).

타임라인 가독성: 구간(절대 프레임)·트림(구간 상대 프레임) SpinBox 값 옆에
원본 fps 기준 mm:ss를 표시한다. segment_table과 time_format.mmss_suffix를 공유한다.

모달 없음(QMessageBox/QFileDialog 미사용) — hang 우려 없음.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="PySide6 미설치 — UI 위젯 테스트 불가")

from PySide6.QtWidgets import QApplication  # noqa: E402

from easy_capture.ui.time_format import mmss_suffix  # noqa: E402
from easy_capture.ui.video_window import VideoMainWindow  # noqa: E402

_app: QApplication | None = None


def _get_app() -> QApplication:
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


class _FakeMeta:
    """probe_meta 더블 — fps 속성만 제공."""

    def __init__(self, fps: float) -> None:
        self.fps = fps


def _make_window() -> VideoMainWindow:
    _get_app()
    return VideoMainWindow(lambda _p: None)


# ===========================================================================
# 1. time_format.mmss_suffix 순수 함수
# ===========================================================================
class TestMmssSuffixPure:
    def test_fps_있으면_앞공백2_mmss(self):
        assert mmss_suffix(90, 30.0) == "  00:03"

    def test_fps_None이면_빈문자열(self):
        assert mmss_suffix(90, None) == ""

    def test_fps_0이면_빈문자열(self):
        assert mmss_suffix(90, 0) == ""

    def test_분_단위_표시(self):
        # 1830/30 = 61초 → 01:01
        assert mmss_suffix(1830, 30.0) == "  01:01"


# ===========================================================================
# 2. video_window 구간/트림 SpinBox suffix
# ===========================================================================
class TestVideoWindowMmss:
    def test_base_fps_설정_전_구간_suffix가_비어있다(self):
        window = _make_window()
        window._span_start.setValue(50)
        assert window._span_start.suffix() == ""

    def test_apply_source_fps_후_구간_시작_suffix가_표시된다(self):
        window = _make_window()
        window._apply_source_fps(_FakeMeta(30.0))

        window._span_start.setValue(90)  # 3s @30fps → 00:03

        assert "00:03" in window._span_start.suffix()

    def test_apply_source_fps_후_구간_끝_suffix가_표시된다(self):
        window = _make_window()
        window._apply_source_fps(_FakeMeta(30.0))
        window._span_end.setMaximum(2000)  # _setup_span_controls 모사(기본 max 99)

        window._span_end.setValue(1830)  # 61s → 01:01

        assert "01:01" in window._span_end.suffix()

    def test_트림_SpinBox에도_mmss가_표시된다(self):
        window = _make_window()
        window._apply_source_fps(_FakeMeta(30.0))

        window._trim_end_spin.setValue(60)  # 2s → 00:02

        assert "00:02" in window._trim_end_spin.suffix()

    def test_apply_source_fps가_기존_구간값_suffix를_갱신한다(self):
        # fps 설정 전에 값을 넣고, 나중에 fps를 설정하면 기존 값도 mm:ss 표시
        window = _make_window()
        window._span_start.setMaximum(2000)  # _setup_span_controls 모사(기본 max 99)
        window._span_start.setValue(120)  # fps 미설정 → suffix ""
        assert window._span_start.suffix() == ""

        window._apply_source_fps(_FakeMeta(30.0))  # 120/30 = 4s

        assert "00:04" in window._span_start.suffix()

    def test_fps_0인_메타는_suffix를_표시하지_않는다(self):
        window = _make_window()
        window._apply_source_fps(_FakeMeta(0))

        window._span_start.setValue(90)

        assert window._span_start.suffix() == ""
