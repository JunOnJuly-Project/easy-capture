"""video_window 컷 선택 통합 테스트 (offscreen PySide6, Task 4-4·4-5).

대상:
  - _DetectWorker: detect_cuts → detect_cut_candidates (가벼움)
  - _TrackWorker.selections 전달(핵심 결함 수정)
  - VideoMainWindow 컷 유무 모드 자동 전환 + selections 배선
  - _candidate_boxes_from: Detection→box 경계 변환

검증 원칙:
  - 모달 hang 방지: QMessageBox는 monkeypatch로 치환(이 프로젝트 알려진 이슈).
  - 워커는 start()를 무해화하거나 run()을 직접 호출해 동기 검증(좀비 스레드 회피).
  - 무회귀: 단일 클릭 추적은 selections=None으로 그대로 동작.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 미설치 — UI 모듈 import 불가")

from PySide6.QtWidgets import QApplication  # noqa: E402

from easy_capture.app.video_capture import ShotCandidates  # noqa: E402
from easy_capture.core.segmentation.detection_backend import Detection  # noqa: E402
from easy_capture.core.source.frame_source import FrameSpan  # noqa: E402
from easy_capture.ui import video_window as vw  # noqa: E402

_app: QApplication | None = None


def _get_app() -> QApplication:
    """QApplication 싱글턴을 반환한다."""
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


# ---------------------------------------------------------------------------
# 테스트 상수 (매직넘버 금지)
# ---------------------------------------------------------------------------
_FRAME_H = 100
_FRAME_W = 150
_N_FRAMES = 10

_BOX_A = (10.0, 10.0, 50.0, 80.0)
_BOX_B = (60.0, 10.0, 120.0, 80.0)
_SCORE = 0.9

# _BOX_A 내부 클릭 좌표 (pick_box_at → 인덱스 0)
_CLICK_IN_A = (30, 40)

_CUT_FRAME = 5  # 단일 컷 경계


# ---------------------------------------------------------------------------
# 더블 — Fake usecase / ShotCandidates / 프레임
# ---------------------------------------------------------------------------

def _make_frames() -> list[np.ndarray]:
    """검정 프레임 시퀀스를 만든다."""
    return [np.zeros((_FRAME_H, _FRAME_W, 3), dtype=np.uint8) for _ in range(_N_FRAMES)]


def _make_shot(shot_index: int, boxes: list[tuple]) -> ShotCandidates:
    """합성 ShotCandidates(Detection 더블 포함)를 만든다."""
    candidates = [Detection(box=b, score=_SCORE) for b in boxes]
    return ShotCandidates(
        shot_index=shot_index,
        first_frame_index=shot_index,
        candidates=candidates,
    )


def _make_two_shots() -> list[ShotCandidates]:
    """후보 2개씩 2샷."""
    return [_make_shot(0, [_BOX_A, _BOX_B]), _make_shot(1, [_BOX_A, _BOX_B])]


class _FakeTrackResult:
    """track 반환 더블(이 테스트는 track 내부를 보지 않음)."""

    masks: list = []
    centroids: list = []
    needs_correction: list = []
    cut_frames: list = []


class _FakeUseCase:
    """VideoCaptureUseCase 더블 — detect/track 호출을 기록한다."""

    def __init__(self, cut_frames=None, shots=None) -> None:
        self._cut_frames = cut_frames if cut_frames is not None else [_CUT_FRAME]
        self._shots = shots if shots is not None else _make_two_shots()
        self.track_calls: list = []

    def detect_cuts(self, video_path, span):
        return self._cut_frames

    def detect_cut_candidates(self, frames, cut_frames):
        return self._shots

    def track(self, frames, point, cut_frames=None, selections=None):
        self.track_calls.append((point, cut_frames, selections))
        return _FakeTrackResult()

    def probe_meta(self):
        """_on_open_file이 호출하는 메타 더미(fps만 필요)."""
        return _FakeMeta()


class _FakeMeta:
    """probe_meta 반환 더미 — _estimate_frame_count가 fps만 참조한다."""

    fps = 30.0


def _make_window(usecase: _FakeUseCase) -> "vw.VideoMainWindow":
    """컷 추적 테스트용 VideoMainWindow를 구성한다(프레임·구간 주입)."""
    _get_app()
    win = vw.VideoMainWindow(usecase_factory=lambda _path: usecase)
    win._usecase = usecase
    win._frames = _make_frames()
    win._video_path = "fake.mp4"
    win._pending_span = FrameSpan(start=0, end=_N_FRAMES)
    return win


# ===========================================================================
# 1. _DetectWorker — detect_cuts → detect_cut_candidates
# ===========================================================================

class TestDetectWorker:
    """가벼운 검출 워커가 컷 경계와 샷 후보를 함께 방출한다."""

    def test_run이_cut_frames와_shots를_튜플로_방출한다(self):
        _get_app()
        usecase = _FakeUseCase(cut_frames=[_CUT_FRAME], shots=_make_two_shots())
        worker = vw._DetectWorker(usecase, _make_frames(), "fake.mp4", FrameSpan(0, _N_FRAMES))
        received: list = []
        worker.candidates_ready.connect(received.append)

        worker.run()

        assert len(received) == 1
        cut_frames, shots = received[0]
        assert cut_frames == [_CUT_FRAME]
        assert len(shots) == 2

    def test_detect_cuts_None이면_빈_리스트로_방출한다(self):
        _get_app()

        class _NoCutUseCase(_FakeUseCase):
            def detect_cuts(self, video_path, span):
                return None

            def detect_cut_candidates(self, frames, cut_frames):
                return []

        worker = vw._DetectWorker(_NoCutUseCase(), _make_frames(), "f.mp4", FrameSpan(0, 1))
        received: list = []
        worker.candidates_ready.connect(received.append)

        worker.run()

        cut_frames, shots = received[0]
        assert cut_frames == []


# ===========================================================================
# 2. selections 전달 (핵심 결함 수정)
# ===========================================================================

class TestSelectionsWiring:
    """_TrackWorker에 selections가 정확히 전달되는지 검증한다."""

    def test_컷_모드_추적은_selections를_전달한다(self, monkeypatch):
        monkeypatch.setattr(vw._TrackWorker, "start", lambda self: None)
        usecase = _FakeUseCase()
        win = _make_window(usecase)
        win._enter_cut_mode(_make_two_shots())
        win._cut_panel.set_target(0)  # 첫 샷 대상 선택

        win._on_track()

        selections = win._track_worker._selections
        assert selections is not None
        assert len(selections) == 1
        assert selections[0].shot_index == 0

    def test_단일_모드_추적은_selections가_None이다(self, monkeypatch):
        monkeypatch.setattr(vw._TrackWorker, "start", lambda self: None)
        usecase = _FakeUseCase()
        win = _make_window(usecase)
        win._cut_mode = False
        win._pending_point = _CLICK_IN_A

        win._on_track()

        assert win._track_worker._selections is None

    def test_TrackWorker_run이_track에_selections를_kwarg로_넘긴다(self):
        _get_app()
        usecase = _FakeUseCase()
        sentinel: list = []
        worker = vw._TrackWorker(
            usecase, _make_frames(), _CLICK_IN_A,
            video_path=None, span=None, selections=sentinel,
        )

        worker.run()

        # track 호출의 selections 인자가 그대로 전달됐는지
        assert usecase.track_calls[0][2] is sentinel


# ===========================================================================
# 3. 모드 자동 전환
# ===========================================================================

class TestModeSwitch:
    """컷 유무에 따라 단일/컷 모드가 자동 전환된다."""

    def test_컷_없으면_단일_모드_유지(self):
        usecase = _FakeUseCase()
        win = _make_window(usecase)

        win._on_candidates_ready(([], []))

        assert win._cut_mode is False
        assert win._cut_panel.isHidden()

    def test_컷_있으면_컷_모드_진입_패널_표시(self):
        usecase = _FakeUseCase()
        win = _make_window(usecase)

        win._on_candidates_ready(([_CUT_FRAME], _make_two_shots()))

        assert win._cut_mode is True
        assert not win._cut_panel.isHidden()
        assert win._track_btn.isEnabled()

    def test_구간_변경_시_컷_모드가_해제된다(self):
        usecase = _FakeUseCase()
        win = _make_window(usecase)
        win._enter_cut_mode(_make_two_shots())

        win._on_span_changed()

        assert win._cut_mode is False

    def test_영상_재오픈_시_컷_모드와_클릭점이_초기화된다(self, monkeypatch):
        """컷 모드에서 새 영상을 열면 컷 상태·후보·클릭점이 모두 해제된다.

        WHY: 잔류 컷 모드는 새 영상에서 단일 클릭을 box_clicked로 오라우팅하고
             이전 후보로 잘못된 selections를 빌드한다(reviewer [중요]).
        """
        usecase = _FakeUseCase()
        win = _make_window(usecase)
        win._enter_cut_mode(_make_two_shots())
        win._pending_point = _CLICK_IN_A
        # 파일 다이얼로그·메타·첫 프레임 로드를 더미로 우회(모달 hang 방지)
        monkeypatch.setattr(
            vw.QFileDialog, "getOpenFileName", lambda *a, **k: ("new.mp4", "")
        )
        # 모달 hang 방지 — 예외 경로의 critical 다이얼로그도 치환(방어)
        monkeypatch.setattr(vw.QMessageBox, "critical", lambda *a, **k: None)
        monkeypatch.setattr(win, "_setup_span_controls", lambda: None)
        monkeypatch.setattr(win, "_load_first_frame", lambda: None)
        monkeypatch.setattr(win, "_apply_source_fps", lambda meta: None)

        win._on_open_file()

        assert win._cut_mode is False
        assert win._shot_candidates is None
        assert win._pending_point is None
        assert win._cut_panel.isHidden()


# ===========================================================================
# 4. Detection→box 경계 변환
# ===========================================================================

class TestCandidateBoxes:
    """_candidate_boxes_from이 Detection에서 box만 추출한다."""

    def test_샷별_box_리스트를_추출한다(self):
        shots = _make_two_shots()

        boxes = vw._candidate_boxes_from(shots)

        assert boxes == [[_BOX_A, _BOX_B], [_BOX_A, _BOX_B]]


# ===========================================================================
# 5. 캔버스 박스 클릭 → hit-test → 대상 지정
# ===========================================================================

class TestBoxClickHitTest:
    """캔버스 박스 클릭이 pick_box_at으로 대상을 빠르게 지정한다."""

    def test_박스_내부_클릭이_해당_후보를_대상으로_지정한다(self):
        usecase = _FakeUseCase()
        win = _make_window(usecase)
        win._enter_cut_mode(_make_two_shots())

        win._on_box_clicked(*_CLICK_IN_A)  # _BOX_A 내부 → 인덱스 0

        assert win._cut_panel.current_choice().target_idx == 0


# ===========================================================================
# 6. ValueError 흡수 (모달 monkeypatch)
# ===========================================================================

class TestSelectionError:
    """selections 빌드 ValueError를 QMessageBox로 흡수하고 추적을 막는다."""

    def test_빌드_오류_시_워커를_시작하지_않고_경고한다(self, monkeypatch):
        usecase = _FakeUseCase()
        win = _make_window(usecase)
        win._enter_cut_mode(_make_two_shots())
        win._cut_panel.set_target(0)

        warned: list = []
        monkeypatch.setattr(
            vw.QMessageBox, "warning",
            lambda *args, **kwargs: warned.append(args),
        )

        def _raise(*_args, **_kwargs):
            raise ValueError("범위 오류")

        monkeypatch.setattr(vw, "build_selections_from_choices", _raise)
        monkeypatch.setattr(vw._TrackWorker, "start", lambda self: None)

        win._on_track()

        assert len(warned) == 1  # 경고 표시됨
        assert win._track_worker is None  # 워커 미생성


# ===========================================================================
# 7. 무회귀 — 단일 클릭 경로
# ===========================================================================

class TestSingleClickRegression:
    """단일 클릭 추적(컷 모드 아님)이 기존대로 동작한다."""

    def test_단일_클릭_후_추적_버튼_활성화(self, monkeypatch):
        usecase = _FakeUseCase()
        win = _make_window(usecase)
        win._cut_mode = False
        # _load_span_frames_with_span을 더미로 — 구간 로드 우회
        monkeypatch.setattr(
            win, "_load_span_frames_with_span",
            lambda: (win._frames, win._pending_span),
        )

        win._on_canvas_click(*_CLICK_IN_A)

        assert win._track_btn.isEnabled()
        assert win._pending_point == _CLICK_IN_A
