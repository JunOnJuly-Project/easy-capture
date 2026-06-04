"""SegmentTableWidget mm:ss suffix 통합 테스트 (offscreen PySide6).

타임라인 가독성: 시작/끝 SpinBox 값 옆에 원본 fps 기준 mm:ss를 표시한다.
모달 없음(QMessageBox/QFileDialog 미사용) — hang 우려 없음.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="PySide6 미설치 — UI 위젯 테스트 불가")

from PySide6.QtWidgets import QApplication  # noqa: E402

from easy_capture.ui.segment_table import (  # noqa: E402
    _COL_END,
    _COL_START,
    SegmentTableWidget,
)

_app: QApplication | None = None


def _get_app() -> QApplication:
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


def test_base_fps_설정_후_시작_프레임에_mmss_suffix가_표시된다():
    _get_app()
    widget = SegmentTableWidget()
    widget.add_row()
    widget.set_base_fps(30.0)
    widget.setCurrentCell(0, _COL_START)

    widget.set_frame_as_start(90)  # 90/30 = 3s → 00:03

    spin = widget._get_spinbox(0, _COL_START)
    assert spin is not None
    assert "00:03" in spin.suffix()


def test_끝_프레임도_mmss_suffix가_갱신된다():
    _get_app()
    widget = SegmentTableWidget()
    widget.add_row()
    widget.set_base_fps(30.0)
    widget.setCurrentCell(0, _COL_END)

    widget.set_frame_as_end(1830)  # 1830/30 = 61s → 01:01

    spin = widget._get_spinbox(0, _COL_END)
    assert spin is not None
    assert "01:01" in spin.suffix()


def test_base_fps_없으면_suffix가_비어있다():
    _get_app()
    widget = SegmentTableWidget()
    widget.add_row()

    spin = widget._get_spinbox(0, _COL_START)
    assert spin is not None
    assert spin.suffix() == ""


def test_set_base_fps가_기존_행_suffix를_갱신한다():
    # 행을 먼저 만들고(값 주입) 나중에 fps 설정 → 기존 행도 mm:ss 표시
    _get_app()
    widget = SegmentTableWidget()
    widget.add_row()
    widget.setCurrentCell(0, _COL_START)
    widget.set_frame_as_start(60)  # 2s @30fps

    widget.set_base_fps(30.0)

    spin = widget._get_spinbox(0, _COL_START)
    assert spin is not None
    assert "00:02" in spin.suffix()
