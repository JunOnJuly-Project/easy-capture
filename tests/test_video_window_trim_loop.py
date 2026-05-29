"""ui/video_window 트림+루프 컨트롤 헤드리스 단위 테스트 (Story T4).

대상 모듈:
  - easy_capture.ui.video_window.VideoMainWindow
  - easy_capture.ui.video_window._GAP_POLICY_ITEMS
  - easy_capture.ui.video_window._warn_mp4_loop_if_needed
  - easy_capture.ui.video_window._trim_length
  - easy_capture.ui.video_window._TrimValidationError

검증 계약:
  1. VideoExportConfig.trim 매핑  — 트림 체크 on → TrimRange, off → None
  2. VideoExportConfig.loop_count 매핑 — loop SpinBox 값 → config
  3. MP4 + loop_count != 0 → _warn_mp4_loop_if_needed 조건 분기
  4. _trim_length 순수 함수 — trim 지정/미지정/프레임 없음
  5. 폭증 경고 보정 — 트림 후 길이 기준으로 estimate 호출 여부
  6. 트림 SpinBox start >= end → _TrimValidationError
  7. 트림 체크 on/off → SpinBox 활성·비활성 전환
  8. _on_frame_to_trim_start/end → SpinBox 값 (span 상대 인덱스)
  9. VideoMainWindow 인스턴스화 무회귀

PySide6 의존: pytest.importorskip — 미설치 시 전부 skip.
offscreen 플랫폼: os.environ.setdefault("QT_QPA_PLATFORM", "offscreen").
"""
from __future__ import annotations

import os

# WHY: PySide6 위젯 import 전 헤드리스 플랫폼 지정 — CI 및 offscreen 환경 대응
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="PySide6 미설치 — UI 모듈 import 불가")

from PySide6.QtWidgets import QApplication  # noqa: E402

from easy_capture.core.timing.timeremap import TrimRange  # noqa: E402
from easy_capture.ui.video_window import (  # noqa: E402
    VideoMainWindow,
    _TrimValidationError,
    _trim_length,
    _warn_mp4_loop_if_needed,
)

# ---------------------------------------------------------------------------
# QApplication 싱글턴 (테스트 세션 전체에서 1회 생성)
# ---------------------------------------------------------------------------
_app: QApplication | None = None


def _get_app() -> QApplication:
    """QApplication 싱글턴을 반환한다.

    WHY: PySide6 위젯 생성 전 QApplication 인스턴스가 반드시 존재해야 한다.
         pytest fixture 대신 모듈 수준 싱글턴으로 중복 생성을 방지한다.
    """
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


# ---------------------------------------------------------------------------
# 테스트 상수 — 매직넘버 금지
# ---------------------------------------------------------------------------
_SPAN_START = 0
_SPAN_LEN = 10          # 테스트용 span 길이 (span_end - span_start)
_TRIM_START = 2
_TRIM_END = 7
_TRIM_LEN = 5           # TRIM_END - TRIM_START
_LOOP_COUNT_3 = 3
_LOOP_COUNT_INF = 0     # 무한 루프
_MSG_TRIM_NOT_IMPL = "trim/loop_count 컨트롤 미구현"


def _make_window() -> VideoMainWindow:
    """테스트용 VideoMainWindow를 생성한다(usecase_factory=None 더미).

    WHY: 헤드리스 테스트에서 실제 usecase 없이 UI 위젯 계약만 검증한다.
         None 팩토리는 파일 열기 전까지 호출되지 않으므로 안전하다.
    """
    _get_app()
    return VideoMainWindow(usecase_factory=lambda _: None)


# ---------------------------------------------------------------------------
# 1. VideoMainWindow 인스턴스화 무회귀
# ---------------------------------------------------------------------------
class TestVideoMainWindowInit:
    """VideoMainWindow가 트림/루프 컨트롤 추가 후에도 정상 생성되는지 검증."""

    def test_인스턴스화_성공(self):
        """Given: 더미 usecase_factory
        When:  VideoMainWindow 생성
        Then:  AttributeError 없이 인스턴스 생성 성공.

        WHY: 트림/루프 컨트롤 추가가 기존 초기화 흐름을 깨지 않음을 확인한다.
        """
        win = _make_window()
        assert win is not None

    def test_트림_체크박스_위젯_존재(self):
        """Given: 생성된 VideoMainWindow
        When:  _trim_check 속성 접근
        Then:  QCheckBox 인스턴스 — 트림 컨트롤이 빌드되었음을 확인한다.
        """
        from PySide6.QtWidgets import QCheckBox

        win = _make_window()
        assert hasattr(win, "_trim_check"), "_trim_check 위젯 누락"
        assert isinstance(win._trim_check, QCheckBox)

    def test_루프_SpinBox_위젯_존재(self):
        """Given: 생성된 VideoMainWindow
        When:  _loop_spin 속성 접근
        Then:  QSpinBox 인스턴스 — 루프 컨트롤이 빌드되었음을 확인한다.
        """
        from PySide6.QtWidgets import QSpinBox

        win = _make_window()
        assert hasattr(win, "_loop_spin"), "_loop_spin 위젯 누락"
        assert isinstance(win._loop_spin, QSpinBox)

    def test_트림_시작끝_SpinBox_위젯_존재(self):
        """Given: 생성된 VideoMainWindow
        When:  _trim_start_spin / _trim_end_spin 속성 접근
        Then:  두 위젯 모두 QSpinBox.
        """
        from PySide6.QtWidgets import QSpinBox

        win = _make_window()
        assert isinstance(win._trim_start_spin, QSpinBox)
        assert isinstance(win._trim_end_spin, QSpinBox)


# ---------------------------------------------------------------------------
# 2. 트림 체크 on/off → SpinBox 활성·비활성
# ---------------------------------------------------------------------------
class TestTrimCheckToggle:
    """트림 체크박스 토글에 따른 SpinBox 활성 상태 계약."""

    def test_체크_해제_시_SpinBox_비활성(self):
        """Given: 새로 생성된 윈도우(기본 체크 해제)
        When:  _trim_start_spin.isEnabled() 확인
        Then:  False — 체크 해제 상태에서 SpinBox가 비활성이어야 한다.

        WHY: 트림 미사용 시 사용자가 실수로 값을 변경하는 것을 막는다.
        """
        win = _make_window()
        assert not win._trim_start_spin.isEnabled()
        assert not win._trim_end_spin.isEnabled()

    def test_체크_on_시_SpinBox_활성화(self):
        """Given: 트림 체크 해제 상태의 윈도우
        When:  _trim_check.setChecked(True) → _on_trim_toggled 호출
        Then:  _trim_start_spin / _trim_end_spin isEnabled() == True.

        WHY: 체크 시 SpinBox가 활성화돼야 사용자가 트림 범위를 입력할 수 있다.
        """
        win = _make_window()
        win._trim_check.setChecked(True)

        assert win._trim_start_spin.isEnabled()
        assert win._trim_end_spin.isEnabled()

    def test_체크_on_후_off_시_SpinBox_다시_비활성(self):
        """Given: 트림 체크 on 상태
        When:  setChecked(False)
        Then:  SpinBox 다시 비활성.
        """
        win = _make_window()
        win._trim_check.setChecked(True)
        win._trim_check.setChecked(False)

        assert not win._trim_start_spin.isEnabled()

    def test_trim_enabled_필드가_체크_상태와_동기화된다(self):
        """Given: 트림 체크 on
        When:  _trim_enabled 필드 확인
        Then:  True — 내부 상태가 위젯 상태와 일치한다.

        WHY: _trim_enabled가 _resolve_trim의 분기 조건이므로 정확한 동기화가 필수다.
        """
        win = _make_window()
        win._trim_check.setChecked(True)
        assert win._trim_enabled is True

        win._trim_check.setChecked(False)
        assert win._trim_enabled is False


# ---------------------------------------------------------------------------
# 3. _resolve_trim → TrimRange 또는 None
# ---------------------------------------------------------------------------
class TestResolveTrim:
    """_resolve_trim 반환값 계약."""

    def test_체크_해제이면_None_반환(self):
        """Given: 트림 체크 해제(기본)
        When:  _resolve_trim()
        Then:  None — 트림 미사용 경로(무회귀).
        """
        win = _make_window()
        assert win._resolve_trim() is None

    def test_체크_on_유효_범위이면_TrimRange_반환(self):
        """Given: 트림 체크 on, start=2, end=7
        When:  _resolve_trim()
        Then:  TrimRange(start=2, end=7).

        WHY: 체크 on + 유효 범위이면 TrimRange 값객체가 생성되어 VideoExportConfig에
             주입되어야 한다.
        """
        win = _make_window()
        win._trim_check.setChecked(True)
        win._trim_start_spin.setValue(_TRIM_START)
        win._trim_end_spin.setValue(_TRIM_END)

        result = win._resolve_trim()

        assert result == TrimRange(start=_TRIM_START, end=_TRIM_END)

    def test_체크_on_start_ge_end이면_예외(self, monkeypatch):
        """Given: 트림 체크 on, start=5, end=5 (같음)
        When:  _resolve_trim()
        Then:  _TrimValidationError 발생.

        WHY: 빈 구간은 출력 프레임 0개로 인코딩 실패를 유발한다.
             조기 차단 → QMessageBox 경고 → 저장 진행 차단.
        WHY monkeypatch: _resolve_trim이 실제 QMessageBox.warning(모달)을 띄우면
             offscreen에서도 블록돼 테스트가 hang한다. 경고 호출을 무력화한다.
        """
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: None)
        win = _make_window()
        win._trim_check.setChecked(True)
        win._trim_start_spin.setValue(5)
        win._trim_end_spin.setValue(5)

        with pytest.raises(_TrimValidationError):
            win._resolve_trim()

    def test_체크_on_start_gt_end이면_예외(self, monkeypatch):
        """Given: 트림 체크 on, start=7, end=2 (역전)
        When:  _resolve_trim()
        Then:  _TrimValidationError 발생.
        """
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: None)
        win = _make_window()
        win._trim_check.setChecked(True)
        win._trim_start_spin.setValue(7)
        win._trim_end_spin.setValue(2)

        with pytest.raises(_TrimValidationError):
            win._resolve_trim()


# ---------------------------------------------------------------------------
# 4. loop_count SpinBox 기본값·범위 계약
# ---------------------------------------------------------------------------
class TestLoopCountSpinBox:
    """_loop_spin SpinBox 기본값·범위 계약."""

    def test_기본값이_0이다(self):
        """Given: 생성된 VideoMainWindow
        When:  _loop_spin.value()
        Then:  0 — 무한 루프(기존 GIF loop=0 계약과 동일, 무회귀).
        """
        win = _make_window()
        assert win._loop_spin.value() == 0

    def test_최솟값이_0이다(self):
        """Given: _loop_spin
        When:  minimum() 확인
        Then:  0 — 음수는 허용되지 않는다(_validate_loop_count와 일치).
        """
        win = _make_window()
        assert win._loop_spin.minimum() == 0

    def test_지정값이_SpinBox에_보관된다(self):
        """Given: _loop_spin.setValue(3)
        When:  value() 확인
        Then:  3.
        """
        win = _make_window()
        win._loop_spin.setValue(_LOOP_COUNT_3)
        assert win._loop_spin.value() == _LOOP_COUNT_3


# ---------------------------------------------------------------------------
# 5. _warn_mp4_loop_if_needed 순수 조건 분기
# ---------------------------------------------------------------------------
class TestWarnMp4Loop:
    """_warn_mp4_loop_if_needed — QMessageBox 표시 조건 검증.

    실제 QMessageBox를 띄우지 않도록 monkeypatch로 대체한다.
    """

    def test_mp4_plus_loop_nonzero_호출됨(self, monkeypatch):
        """Given: fmt='mp4', loop_count=3
        When:  _warn_mp4_loop_if_needed 호출
        Then:  QMessageBox.information이 1회 호출된다.

        WHY: MP4는 루프 메타 미지원 — 사용자에게 사전 안내(한국어).
        """
        from PySide6.QtWidgets import QMessageBox

        calls: list = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: calls.append(a))

        win = _make_window()
        _warn_mp4_loop_if_needed(win, "mp4", loop_count=3)

        assert len(calls) == 1

    def test_gif_plus_loop_nonzero_호출_안됨(self, monkeypatch):
        """Given: fmt='gif', loop_count=3
        When:  _warn_mp4_loop_if_needed 호출
        Then:  QMessageBox.information 미호출 — GIF는 루프 정상 지원.
        """
        from PySide6.QtWidgets import QMessageBox

        calls: list = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: calls.append(a))

        win = _make_window()
        _warn_mp4_loop_if_needed(win, "gif", loop_count=3)

        assert len(calls) == 0

    def test_mp4_plus_loop_0_호출_안됨(self, monkeypatch):
        """Given: fmt='mp4', loop_count=0
        When:  _warn_mp4_loop_if_needed 호출
        Then:  QMessageBox.information 미호출 — loop_count=0은 경고 불필요.

        WHY: loop_count=0(기본값)이면 루프를 설정하지 않은 것과 동일하다.
             기본값으로 경고가 뜨면 사용자 혼란을 유발한다.
        """
        from PySide6.QtWidgets import QMessageBox

        calls: list = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: calls.append(a))

        win = _make_window()
        _warn_mp4_loop_if_needed(win, "mp4", loop_count=0)

        assert len(calls) == 0


# ---------------------------------------------------------------------------
# 6. _trim_length 순수 함수 계약
# ---------------------------------------------------------------------------
class TestTrimLength:
    """_trim_length 순수 함수 — trim 지정/미지정/프레임 없음."""

    def test_trim_None이면_None_반환(self):
        """Given: trim=None
        When:  _trim_length 호출
        Then:  None — 경고 헬퍼가 전체 frames 기준을 사용하게 한다.
        """
        frames = list(range(10))
        assert _trim_length(None, frames) is None

    def test_trim_지정이면_길이_반환(self):
        """Given: trim=TrimRange(2,7), frames 10개
        When:  _trim_length 호출
        Then:  5 (= 7 - 2).

        WHY: 트림 후 길이를 경고 헬퍼에 전달해 오경고를 방지한다(reviewer R7).
        """
        trim = TrimRange(start=_TRIM_START, end=_TRIM_END)
        frames = list(range(10))
        assert _trim_length(trim, frames) == _TRIM_LEN

    def test_frames_None이면_None_반환(self):
        """Given: frames=None (파일 미로드 상태)
        When:  _trim_length 호출
        Then:  None — frames 없는 상태에서 안전 처리.
        """
        trim = TrimRange(start=_TRIM_START, end=_TRIM_END)
        assert _trim_length(trim, None) is None


# ---------------------------------------------------------------------------
# 7. _on_frame_to_trim_start / _on_frame_to_trim_end 좌표계 검증
# ---------------------------------------------------------------------------
class TestFrameToTrimButtons:
    """트림 시작/끝 버튼이 span 상대 인덱스를 주입하는지 검증."""

    def test_프레임_트림시작_버튼이_0을_주입한다(self):
        """Given: _on_frame_to_trim_start 호출
        When:  _trim_start_spin.value() 확인
        Then:  0 — span 첫 프레임(상대 인덱스 0).

        WHY: segments 버튼(_on_frame_to_start)과 동일하게 span 상대 0을 주입한다.
             절대 인덱스를 주입하면 span_start≠0일 때 트림이 빗나간다.
        """
        win = _make_window()
        win._on_frame_to_trim_start()
        assert win._trim_start_spin.value() == 0

    def test_프레임_트림끝_버튼이_span_len을_주입한다(self):
        """Given: span_start=0, span_end=10
        When:  _on_frame_to_trim_end 호출
        Then:  _trim_end_spin.value() == 10 (span_len).

        WHY: segments _on_frame_to_end와 동일하게 span 상대 끝(= span_len)을 주입한다.
        """
        win = _make_window()
        win._span_start.setValue(0)
        win._span_end.setValue(_SPAN_LEN)

        win._on_frame_to_trim_end()

        assert win._trim_end_spin.value() == _SPAN_LEN

    def test_span_start_nonzero일_때_상대_인덱스_정합(self):
        """Given: span_start=5, span_end=15 (span_len=10)
        When:  _on_frame_to_trim_end 호출
        Then:  _trim_end_spin.value() == 10 (상대 길이, 절대 15 아님).

        WHY: span_start가 0이 아닐 때도 상대 길이(10)가 들어가야
             trim 좌표가 crops 시퀀스 상대와 일치한다.
        """
        win = _make_window()
        win._span_start.setValue(5)
        win._span_end.setValue(15)

        win._on_frame_to_trim_end()

        assert win._trim_end_spin.value() == 10


# ---------------------------------------------------------------------------
# 8. 폭증 경고 — 트림 후 길이 기준 호출 검증
# ---------------------------------------------------------------------------
class TestOverflowWarningWithTrim:
    """트림 적용 시 폭증 경고가 트림 후 길이 기준으로 계산되는지 검증."""

    def test_trim_length_None이면_전체_frames_기준(self, monkeypatch):
        """Given: trim=None → _trim_length=None
        When:  _warn_frame_count_overflow_if_needed(fps, segs, n_frames=None)
        Then:  estimate_output_frame_count가 len(frames) 기준으로 호출된다.

        WHY: 트림 미사용 시 기존 동작(전체 frames 기준) 무회귀.
        """
        import easy_capture.ui.video_window as vw
        import easy_capture.core.timing.timeremap as tr
        from easy_capture.core.timing.timeremap import SpeedSegment

        captured: list = []

        def _fake_estimate(n, segs, fps):
            captured.append(n)
            return n  # 폭증 없는 것처럼 처리

        monkeypatch.setattr(vw, "estimate_output_frame_count", _fake_estimate)

        import numpy as np
        win = _make_window()
        win._frames = [np.zeros((10, 10, 3), dtype="uint8")] * 8  # 8프레임
        segs = (SpeedSegment(0, 4, 0.5),)

        win._warn_frame_count_overflow_if_needed(12.0, segs, n_frames=None)

        assert len(captured) == 1
        assert captured[0] == 8  # 전체 frames 수 기준

    def test_trim_length_지정이면_트림후_길이_기준(self, monkeypatch):
        """Given: trim=TrimRange(0,4) → trim_len=4
        When:  _warn_frame_count_overflow_if_needed(fps, segs, n_frames=4)
        Then:  estimate_output_frame_count가 4 기준으로 호출된다.

        WHY: 트림 적용 후에는 4프레임이 기준 — 8프레임으로 계산하면 오경고(reviewer R7).
        """
        import easy_capture.ui.video_window as vw
        from easy_capture.core.timing.timeremap import SpeedSegment

        captured: list = []

        def _fake_estimate(n, segs, fps):
            captured.append(n)
            return n

        monkeypatch.setattr(vw, "estimate_output_frame_count", _fake_estimate)

        import numpy as np
        win = _make_window()
        win._frames = [np.zeros((10, 10, 3), dtype="uint8")] * 8  # 8프레임
        segs = (SpeedSegment(0, 2, 0.5),)

        win._warn_frame_count_overflow_if_needed(12.0, segs, n_frames=4)

        assert len(captured) == 1
        assert captured[0] == 4  # 트림 후 길이 기준


# ---------------------------------------------------------------------------
# [중요 1] _on_export → VideoExportConfig 주입 end-to-end 배선 검증
# ---------------------------------------------------------------------------
class TestOnExportConfigInjection:
    """_on_export가 VideoExportConfig(trim, loop_count)를 올바르게 조립하는지 검증.

    WHY: _resolve_trim / _loop_spin.value()만 단독 검증하면 인자 순서·키워드 회귀를
         잡지 못한다. _ExportWorker 생성을 가로채 실제 전달된 config를 단언한다.
    """

    def _patch_for_export(self, monkeypatch, win):
        """_on_export 실행에 필요한 외부 의존을 패치한다.

        - QFileDialog.getSaveFileName: 고정 경로('/tmp/out.gif') 반환
        - _ExportWorker.__init__: 생성된 config를 캡처, start() 무해화
        - QMessageBox.*: 팝업 차단 (경고/안내 다이얼로그)

        Returns:
            captured_configs: config가 append되는 리스트 참조.
        """
        from easy_capture.ui import video_window as vw
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        captured: list = []

        # 저장 다이얼로그 → 고정 경로 반환
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **kw: ("/tmp/out.gif", "GIF (*.gif)")),
        )

        # _ExportWorker 생성 가로채기 — config를 캡처하고 실제 워커는 생성 안 함
        original_init = vw._ExportWorker.__init__

        def _fake_init(self_w, usecase, frames, boxes, target, result):
            _, cfg = target
            captured.append(cfg)
            # 워커 동작 없이 종료 상태로만 초기화
            super(vw._ExportWorker, self_w).__init__()

        monkeypatch.setattr(vw._ExportWorker, "__init__", _fake_init)
        monkeypatch.setattr(vw._ExportWorker, "start", lambda self_w: None)

        # 팝업 차단
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: None)
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: None)
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **kw: None)

        return captured

    def _ready_window(self):
        """export 버튼을 누를 수 있는 최소 상태로 윈도우를 준비한다.

        _frames, _boxes, _usecase 를 더미로 채워 조기 반환 조건을 통과시킨다.
        """
        import numpy as np

        win = _make_window()
        # 더미 usecase — export 핸들러가 is None 체크만 통과하면 됨
        win._usecase = object()
        win._frames = [np.zeros((10, 10, 3), dtype="uint8")] * 6
        win._boxes = [(0, 0, 10, 10)] * 6
        win._fmt_combo.setEnabled(True)
        win._fps_spin.setEnabled(True)
        win._gap_combo.setEnabled(True)
        win._loop_spin.setEnabled(True)
        return win

    def test_트림_on_시_config_trim이_TrimRange이다(self, monkeypatch):
        """Given: 트림 on, start=2, end=7, loop_count=0
        When:  _on_export 호출
        Then:  _ExportWorker에 전달된 config.trim == TrimRange(2, 7).

        WHY [중요 1]: 인자 순서·키워드 회귀 가드. VideoExportConfig 생성 시
             trim=TrimRange(...)이 올바른 위치에 주입됨을 end-to-end로 단언한다.
        """
        win = self._ready_window()
        captured = self._patch_for_export(monkeypatch, win)

        win._trim_check.setChecked(True)
        win._trim_start_spin.setValue(_TRIM_START)   # 2
        win._trim_end_spin.setValue(_TRIM_END)        # 7
        win._loop_spin.setValue(0)

        win._on_export()

        assert len(captured) == 1, "config가 캡처되지 않음 — _ExportWorker 미생성"
        assert captured[0].trim == TrimRange(start=_TRIM_START, end=_TRIM_END), (
            f"config.trim 불일치: {captured[0].trim!r}"
        )

    def test_트림_off_시_config_trim이_None이다(self, monkeypatch):
        """Given: 트림 off(기본), loop_count=0
        When:  _on_export 호출
        Then:  config.trim is None (무회귀 — 트림 미사용 시 기존 동작과 동일).

        WHY [중요 1]: 트림 off 경로에서 trim=None이 올바르게 주입되는지 확인한다.
             trim이 None이 아닌 값으로 잘못 주입되면 기존 GIF 라운드트립이 깨진다.
        """
        win = self._ready_window()
        captured = self._patch_for_export(monkeypatch, win)

        # 트림 체크 해제(기본값) 확인
        win._trim_check.setChecked(False)
        win._loop_spin.setValue(0)

        win._on_export()

        assert len(captured) == 1
        assert captured[0].trim is None, (
            f"trim 미사용인데 None이 아님: {captured[0].trim!r}"
        )

    def test_loop_count_3이_config에_주입된다(self, monkeypatch):
        """Given: 트림 off, loop_count=3
        When:  _on_export 호출
        Then:  config.loop_count == 3.

        WHY [중요 1]: loop_count SpinBox 값이 VideoExportConfig.loop_count에
             정확히 전달되는지 배선을 단언한다.
        """
        win = self._ready_window()
        captured = self._patch_for_export(monkeypatch, win)

        win._trim_check.setChecked(False)
        win._loop_spin.setValue(_LOOP_COUNT_3)  # 3

        win._on_export()

        assert len(captured) == 1
        assert captured[0].loop_count == _LOOP_COUNT_3, (
            f"config.loop_count 불일치: {captured[0].loop_count!r}"
        )

    def test_트림_on_loop_3_동시_주입된다(self, monkeypatch):
        """Given: 트림 on(2→7) + loop_count=3
        When:  _on_export 호출
        Then:  config.trim == TrimRange(2,7) AND config.loop_count == 3.

        WHY: 두 필드가 동시에 올바르게 조합되는지 확인한다(인자 순서 회귀 가드).
        """
        win = self._ready_window()
        captured = self._patch_for_export(monkeypatch, win)

        win._trim_check.setChecked(True)
        win._trim_start_spin.setValue(_TRIM_START)
        win._trim_end_spin.setValue(_TRIM_END)
        win._loop_spin.setValue(_LOOP_COUNT_3)

        win._on_export()

        assert len(captured) == 1
        cfg = captured[0]
        assert cfg.trim == TrimRange(start=_TRIM_START, end=_TRIM_END)
        assert cfg.loop_count == _LOOP_COUNT_3


# ---------------------------------------------------------------------------
# [중요 2] span 변경 시 트림 SpinBox 상한 재동기화
# ---------------------------------------------------------------------------
class TestTrimSpinboxRangeSync:
    """span 변경 시 트림 SpinBox 상한이 현재 span_len으로 재동기화되는지 검증.

    WHY [중요 2]: span_end를 줄인 뒤 트림 상한이 옛 span_len으로 남으면
      trim.end > len(crops) → validate_trim 실패 (UX 불일치).
      _on_span_changed → _sync_trim_spinbox_range 호출로 즉시 동기화해야 한다.
    """

    def test_초기_setup_시_트림_상한이_span_len과_같다(self):
        """Given: span_start=0, span_end=10
        When:  _setup_span_controls 호출 후 _trim_end_spin.maximum()
        Then:  10 (= span_len).
        """
        win = _make_window()
        # _setup_span_controls는 _usecase 없이도 SpinBox 범위만 설정함
        win._total_frames = 20
        win._setup_span_controls()

        # span_end 기본값 min(_DEFAULT_SPAN_END=60, total=20) = 20이 아니라
        # min(60,20)=20. 여기선 span_start=0 이므로 span_len=20
        span_len = win._span_end.value() - win._span_start.value()
        assert win._trim_end_spin.maximum() == span_len

    def test_sync_후_트림_상한이_현재_span_len이다(self):
        """Given: span_start=5, span_end=15 → span_len=10
        When:  _sync_trim_spinbox_range() 호출
        Then:  _trim_end_spin.maximum() == 10.
        """
        win = _make_window()
        win._span_start.setValue(5)
        win._span_end.setValue(15)

        win._sync_trim_spinbox_range()

        assert win._trim_end_spin.maximum() == 10

    def test_span_end_축소_후_트림_상한이_갱신된다(self):
        """Given: 초기 span_len=20 → span_end를 10으로 줄임
        When:  _on_span_changed 호출(usecase=None이므로 span만 재동기)
        Then:  _trim_end_spin.maximum() == 10 (옛 20 아님).

        WHY: _on_span_changed 내 _sync_trim_spinbox_range가 동작함을 확인.
             usecase=None이면 _load_first_frame는 그냥 리턴 — 범위 동기만 테스트.
        """
        win = _make_window()
        # 인위적으로 span 범위를 크게 잡은 뒤 span_end를 직접 조작
        win._span_start.setValue(0)
        win._span_end.setValue(20)
        win._sync_trim_spinbox_range()
        assert win._trim_end_spin.maximum() == 20  # 초기 상태

        win._span_end.setValue(10)
        win._sync_trim_spinbox_range()  # _on_span_changed가 이를 호출함

        assert win._trim_end_spin.maximum() == 10
