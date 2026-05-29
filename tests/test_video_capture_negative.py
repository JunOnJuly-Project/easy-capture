"""negative point 디스패치 — VideoCaptureUseCase.track 오케스트레이션 테스트 (RED).

대상 모듈: easy_capture.app.video_capture (track negative point 디스패치)
테스트 더블: FakeVideoBackend(add_box/add_click negatives 스파이)·
             FakeFrameSource·FakeDetectionBackend

배경:
  군무 밀착 구간에서 box+positive(point)만으론 대상+옆사람이 맞닿은 한 덩어리로
  합쳐져 마스크가 부정확하다. negative point("이 점=옆 멤버는 대상 아님",
  SAM2 label 0)로 경계를 가른다. transformers 검증상 box+positive+negative는
  SAM2 1회 호출로 조립(input_boxes+input_points+input_labels, clear_old_inputs=True)
  되므로 Protocol은 별도 메서드가 아니라 add_box/add_click에 negatives 인자를
  추가하고, track이 그 샷의 negative 좌표를 backend로 전달한다.

이 파일이 검증하는 신규 계약:
  B. 무회귀: negatives 미지정(빈)이면 기존 box/point 경로 동일.
       - add_box/add_click은 호출되되 backend.last_negatives == ().
       - add_negative_call_count == 0.
       - propagate_call_count == 샷 수(negatives가 전파 횟수를 안 바꾼다).
  E. negatives 전달:
       - selection.box + negative_points → add_box(session, box, negatives=...)로 전달.
         backend.last_negatives == 그 negative 좌표, add_negative_call_count == 1.
       - selection.box 없고 point만 + negative_points → add_click(point, negatives) 전달.
       - _SelectionPlan.negatives_for(shot_index) 조회 정확(빈/지정 구분).

설계 경계:
  실모델 호출 금지 — FakeVideoBackend·FakeDetectionBackend만 사용.
  torch·PySide6·av 미의존. UI 무관(core+app만).

구현 전 RED 상태가 정상:
  track이 아직 negatives를 add_box/add_click로 안 넘기면 last_negatives==()로 남아
  negatives 전달 단언이 실패한다. negative_points 필드/negatives_for 미구현 시
  해당 테스트만 skip(개별 집계).
"""
from __future__ import annotations

import numpy as np
import pytest

# --- video_capture 핵심 심볼 — 구현 전 try/except 격리 ---
try:
    from easy_capture.app.video_capture import (
        TrackResult,
        VideoCaptureUseCase,
    )
    _HAS_VIDEO_USECASE = True
except ModuleNotFoundError:
    TrackResult = None  # type: ignore[assignment,misc]
    VideoCaptureUseCase = None  # type: ignore[assignment,misc]
    _HAS_VIDEO_USECASE = False

# --- _SelectionPlan — 내부 계획 객체(negatives_for 조회 검증용) ---
try:
    from easy_capture.app.video_capture import _SelectionPlan
    _HAS_SELECTION_PLAN = True
except ImportError:
    _SelectionPlan = None  # type: ignore[assignment,misc]
    _HAS_SELECTION_PLAN = False

# --- Detection Protocol — 구현 전 격리 ---
try:
    from easy_capture.core.segmentation.detection_backend import Detection
    _HAS_DETECTION = True
except ImportError:
    Detection = None  # type: ignore[assignment]
    _HAS_DETECTION = False

# --- CutSelection + negative_points 필드 격리 ---
try:
    from easy_capture.core.tracking.cut_selection import CutSelection
    _HAS_CUT_SELECTION = True
except ImportError:
    CutSelection = None  # type: ignore[assignment,misc]
    _HAS_CUT_SELECTION = False

_HAS_NEGATIVES_FIELD = False
_HAS_SELECTION_BOX_FIELD = False
if _HAS_CUT_SELECTION:
    from dataclasses import fields as _dc_fields

    _field_names = {f.name for f in _dc_fields(CutSelection)}
    _HAS_NEGATIVES_FIELD = "negative_points" in _field_names
    _HAS_SELECTION_BOX_FIELD = "box" in _field_names

from tests.fixtures.fakes import (  # noqa: E402
    FakeDetectionBackend,
    FakeFrameSource,
    FakeVideoBackend,
)

# --- FakeVideoBackend negatives 스파이 존재 여부(테스트 더블 — 항상 존재 기대) ---
_HAS_FAKE_NEGATIVES = hasattr(FakeVideoBackend(), "last_negatives")

# --- track(selections=) 시그니처 존재 여부 ---
_HAS_SELECTIONS_PARAM = False
if _HAS_VIDEO_USECASE:
    import inspect

    try:
        _track_params = inspect.signature(VideoCaptureUseCase.track).parameters
        _HAS_SELECTIONS_PARAM = "selections" in _track_params
    except (ValueError, TypeError):
        _HAS_SELECTIONS_PARAM = False

# --- _SelectionPlan.negatives_for 메서드 존재 여부 ---
_HAS_NEGATIVES_FOR = (
    _HAS_SELECTION_PLAN and hasattr(_SelectionPlan, "negatives_for")
)

_MSG_NOT_IMPL = "easy_capture.app.video_capture 미구현 — RED 예상"
_MSG_NO_SELECTIONS = (
    "VideoCaptureUseCase.track에 selections 매개변수 미구현 — RED 예상"
)
_MSG_NO_CUT_SELECTION = (
    "core/tracking/cut_selection.py에 CutSelection 미구현 — RED 예상"
)
_MSG_NO_NEGATIVES_FIELD = (
    "CutSelection.negative_points 필드 미구현 — negative point RED 예상"
)
_MSG_NO_SELECTION_BOX = "CutSelection.box 필드 미구현 — box 프롬프트 RED 예상"
_MSG_NO_DETECTION = (
    "core/segmentation/detection_backend.py에 Detection 미구현 — RED 예상"
)
_MSG_NO_FAKE_NEGATIVES = (
    "FakeVideoBackend.last_negatives 스파이 미구현 — RED 예상"
)
_MSG_NO_NEGATIVES_FOR = (
    "_SelectionPlan.negatives_for 미구현 — negative point 조회 RED 예상"
)
_MSG_NO_SELECTION_PLAN = (
    "easy_capture.app.video_capture._SelectionPlan 미구현 — RED 예상"
)


# ---------------------------------------------------------------------------
# 테스트 상수 (매직넘버 금지 — test_video_capture_box.py 규약 동기화)
# ---------------------------------------------------------------------------
# 첫 샷 사용자 클릭 포인트
CLICK_X = 320
CLICK_Y = 180

# 2샷 시나리오 — 전체 20프레임, 컷 1개(프레임 10)
RETRACK_FRAME_COUNT = 20
RETRACK_CUT_FRAME = 10
RETRACK_SHOT_COUNT = 2     # 컷 1개 → 2샷

# 단일 샷(컷 없음) 시나리오 — 무회귀 가드용
SINGLE_FRAME_COUNT = 10

# 자동 재매칭 폴백용 통과 후보 박스 (test_video_capture_box.py 좌표 규약)
RETRACK_PASS_BOX = (305, 165, 344, 204)

# 사용자 선택 — point + box + negative_points
SELECT_POINT_SHOT0 = (320, 180)
SELECT_POINT_SHOT1 = (200, 150)
SELECT_BOX_SHOT0 = (300.0, 160.0, 340.0, 200.0)
SELECT_BOX_SHOT1 = (180.0, 130.0, 220.0, 170.0)

# negative 좌표 (옆 멤버, SAM2 label 0) — 샷별 서로 다른 값으로 전달 정확성 검증
NEG_SHOT0 = ((250, 200), (380, 60))
NEG_SHOT1 = ((150, 140),)

# 샷 인덱스
SHOT_0 = 0
SHOT_1 = 1


# ---------------------------------------------------------------------------
# 픽스처 헬퍼
# ---------------------------------------------------------------------------
def _make_frames(n: int) -> list[np.ndarray]:
    """n개의 고정 RGB 프레임 리스트를 반환한다(결정적)."""
    src = FakeFrameSource()
    return [src.read_frame(i) for i in range(n)]


def _make_usecase(
    detector: "FakeDetectionBackend | None",
    drift: tuple[int, int] = (0, 0),
) -> tuple["VideoCaptureUseCase", FakeVideoBackend]:
    """FakeVideoBackend·FakeDetectionBackend 주입 VideoCaptureUseCase 쌍 반환."""
    backend = FakeVideoBackend(drift=drift, empty_after=None)
    usecase = VideoCaptureUseCase(
        source=FakeFrameSource(),
        backend=backend,
        detector=detector,
    )
    return usecase, backend


# ===========================================================================
# B. 무회귀 — negatives 미지정이면 기존 box/point 경로 동일
# ===========================================================================
class TestNoNegativeNoRegression:
    """negative point 도입이 negatives 미지정 box/point 경로를 바꾸지 않는다."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_FAKE_NEGATIVES, reason=_MSG_NO_FAKE_NEGATIVES)
    def test_단일샷_point_경로는_negatives가_빈_튜플로_남는다(self):
        """Given: detector=None, 10프레임(단일 샷 point 경로)
        When:  track(frames, point) 호출
        Then:  add_click_call_count == 1, last_negatives == (),
               add_negative_call_count == 0

        WHY: negative를 지정하지 않은 기존 point 경로는 backend에 빈 negatives만
             넘기거나 안 넘겨야 한다. 어느 쪽이든 last_negatives는 ()로 남아
             기존 동작이 변하지 않음을 보장한다(무회귀).
        """
        usecase, backend = _make_usecase(detector=None)
        frames = _make_frames(SINGLE_FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y))

        assert backend.add_click_call_count == 1
        assert backend.last_negatives == (), (
            f"negatives 미지정인데 last_negatives가 비어있지 않음: "
            f"{backend.last_negatives}"
        )
        assert backend.add_negative_call_count == 0, (
            f"negatives 미지정인데 add_negative_call_count="
            f"{backend.add_negative_call_count}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_NEGATIVES, reason=_MSG_NO_FAKE_NEGATIVES)
    def test_멀티샷_자동box_경로도_negatives가_빈_튜플로_남는다(self):
        """Given: 2샷, 통과 후보(PASS_BOX), selections 없음
        When:  track(frames, point, cut_frames=[10]) 호출
        Then:  add_box_call_count == 1, last_negatives == (),
               add_negative_call_count == 0

        WHY: 자동 재매칭 통과 샷의 box 경로는 negative를 모른다 — negatives를
             빈 채로 둬야 기존 box 프롬프트 동작이 유지된다(무회귀).
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[candidate])
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        assert backend.add_box_call_count == 1
        assert backend.last_negatives == (), (
            f"자동 box 경로 negatives가 비어있지 않음: {backend.last_negatives}"
        )
        assert backend.add_negative_call_count == 0

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_NEGATIVES, reason=_MSG_NO_FAKE_NEGATIVES)
    def test_negatives_미지정이면_propagate가_샷_수와_같다(self):
        """Given: 2샷, 통과 후보(PASS_BOX), selections 없음
        When:  track 호출
        Then:  propagate_call_count == 2 (negatives 도입이 전파 횟수를 안 바꾼다)

        WHY: negative point는 add_box/add_click 입력에만 더해질 뿐 SAM2 전파
             횟수를 늘리거나 줄이면 안 된다 — 전파 카운터 불변식을 가드한다.
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[candidate])
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        assert backend.propagate_call_count == RETRACK_SHOT_COUNT, (
            f"propagate 호출 횟수 불일치: {backend.propagate_call_count} vs "
            f"{RETRACK_SHOT_COUNT}(샷 수)"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(not _HAS_SELECTION_BOX_FIELD, reason=_MSG_NO_SELECTION_BOX)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_NEGATIVES, reason=_MSG_NO_FAKE_NEGATIVES)
    def test_negative_없는_box_selection은_negatives가_빈_튜플로_남는다(self):
        """Given: 2샷, box는 있으나 negative_points는 빈 selection 2개
        When:  track(..., selections=[box 샷0, 샷1]) 호출
        Then:  add_box_call_count == 2, last_negatives == (),
               add_negative_call_count == 0

        WHY: negative_points 필드 추가가 negative 없는 기존 box 선택 동작을
             깨면 안 된다 — negatives는 비어있는 채로 전달돼야 한다(무회귀).
        """
        detector = FakeDetectionBackend(candidates_fixed=[])
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)
        selections = [
            CutSelection(
                shot_index=SHOT_0, point=SELECT_POINT_SHOT0, box=SELECT_BOX_SHOT0
            ),
            CutSelection(
                shot_index=SHOT_1, point=SELECT_POINT_SHOT1, box=SELECT_BOX_SHOT1
            ),
        ]

        usecase.track(
            frames,
            (CLICK_X, CLICK_Y),
            cut_frames=[RETRACK_CUT_FRAME],
            selections=selections,
        )

        assert backend.add_box_call_count == RETRACK_SHOT_COUNT
        assert backend.last_negatives == (), (
            f"negative 없는 box selection인데 last_negatives="
            f"{backend.last_negatives}"
        )
        assert backend.add_negative_call_count == 0


# ===========================================================================
# E. negatives 전달 — selection.box/point + negative_points → backend 전달
# ===========================================================================
class TestNegativeDispatch:
    """negative_points를 가진 selection은 그 좌표를 add_box/add_click negatives로 넘긴다."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(not _HAS_SELECTION_BOX_FIELD, reason=_MSG_NO_SELECTION_BOX)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_NEGATIVES, reason=_MSG_NO_FAKE_NEGATIVES)
    def test_box_selection의_negative_points가_add_box로_전달된다(self):
        """Given: 1샷(컷 없음), box+negative_points를 가진 샷0 selection
        When:  track(..., selections=[box+negative 샷0]) 호출
        Then:  add_box_call_count == 1, last_negatives == NEG_SHOT0,
               add_negative_call_count == 1

        WHY: 사용자가 옆 멤버(label 0)를 찍으면 그 좌표를 SAM2 box 프롬프트와
             함께(input_boxes+input_points+input_labels, 1회 조립) 넘겨야 경계가
             갈라진다. selection.negative_points가 add_box의 negatives 인자로
             정확히 전달돼야 한다(핵심 디스패치 가드). 단일 샷이라 마지막 호출
             negatives가 그 샷의 값으로 결정적이다.
        """
        detector = FakeDetectionBackend(candidates_fixed=[])
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(SINGLE_FRAME_COUNT)
        selections = [
            CutSelection(
                shot_index=SHOT_0,
                point=SELECT_POINT_SHOT0,
                box=SELECT_BOX_SHOT0,
                negative_points=NEG_SHOT0,
            ),
        ]

        usecase.track(
            frames,
            (CLICK_X, CLICK_Y),
            cut_frames=[],
            selections=selections,
        )

        assert backend.add_box_call_count == 1
        assert tuple(backend.last_negatives) == NEG_SHOT0, (
            f"add_box에 전달된 negatives 불일치: {backend.last_negatives} vs "
            f"{NEG_SHOT0} — selection.negative_points 미전달 의심"
        )
        assert backend.add_negative_call_count == 1, (
            f"add_negative_call_count 불일치: {backend.add_negative_call_count} vs 1"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_NEGATIVES, reason=_MSG_NO_FAKE_NEGATIVES)
    def test_point만인_selection의_negative_points가_add_click으로_전달된다(self):
        """Given: 1샷(컷 없음), box 없이 point+negative_points만 가진 샷0 selection
        When:  track(..., selections=[point+negative 샷0]) 호출
        Then:  add_click_call_count == 1, add_box_call_count == 0,
               last_negatives == NEG_SHOT0

        WHY: box가 없는(point만) 선택도 negative point로 옆 멤버를 가를 수 있어야
             한다. selection.box가 없으면 add_click(point, negatives) 경로로
             negative 좌표가 전달돼야 한다(box 우선, 없으면 click 폴백 정책 계승).
        """
        detector = FakeDetectionBackend(candidates_fixed=[])
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(SINGLE_FRAME_COUNT)
        selections = [
            CutSelection(
                shot_index=SHOT_0,
                point=SELECT_POINT_SHOT0,
                negative_points=NEG_SHOT0,
            ),
        ]

        usecase.track(
            frames,
            (CLICK_X, CLICK_Y),
            cut_frames=[],
            selections=selections,
        )

        assert backend.add_box_call_count == 0, (
            f"box 없는 selection인데 add_box={backend.add_box_call_count}회 호출"
        )
        assert backend.add_click_call_count == 1
        assert tuple(backend.last_negatives) == NEG_SHOT0, (
            f"add_click에 전달된 negatives 불일치: {backend.last_negatives} vs "
            f"{NEG_SHOT0}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(not _HAS_SELECTION_BOX_FIELD, reason=_MSG_NO_SELECTION_BOX)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_NEGATIVES, reason=_MSG_NO_FAKE_NEGATIVES)
    def test_negative_있는_샷만_add_negative를_올린다(self):
        """Given: 2샷, 샷0은 box+negative, 샷1은 box만(negative 없음)
        When:  track(..., selections=[샷0 negative, 샷1 negative 없음]) 호출
        Then:  add_box_call_count == 2, add_negative_call_count == 1
               (negative 가진 샷0만 카운트)

        WHY: negative를 지정한 샷에서만 negatives가 비어있지 않게 전달돼야 한다.
             샷별로 negative 유무가 다를 때 정확히 그 샷만 negative를 받는지
             가드한다(샷 단위 매핑 정확성).
        """
        detector = FakeDetectionBackend(candidates_fixed=[])
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)
        selections = [
            CutSelection(
                shot_index=SHOT_0,
                point=SELECT_POINT_SHOT0,
                box=SELECT_BOX_SHOT0,
                negative_points=NEG_SHOT0,
            ),
            CutSelection(
                shot_index=SHOT_1,
                point=SELECT_POINT_SHOT1,
                box=SELECT_BOX_SHOT1,
            ),
        ]

        usecase.track(
            frames,
            (CLICK_X, CLICK_Y),
            cut_frames=[RETRACK_CUT_FRAME],
            selections=selections,
        )

        assert backend.add_box_call_count == RETRACK_SHOT_COUNT
        assert backend.add_negative_call_count == 1, (
            f"negative 가진 샷만 카운트돼야 한다: "
            f"{backend.add_negative_call_count} vs 1"
        )


# ===========================================================================
# E. _SelectionPlan.negatives_for — 샷별 negative 조회 정확성
# ===========================================================================
class TestSelectionPlanNegativesFor:
    """_SelectionPlan.negatives_for(shot_index): 샷별 negative 좌표를 정확히 조회한다."""

    @pytest.mark.skipif(not _HAS_SELECTION_PLAN, reason=_MSG_NO_SELECTION_PLAN)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FOR, reason=_MSG_NO_NEGATIVES_FOR)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(not _HAS_SELECTION_BOX_FIELD, reason=_MSG_NO_SELECTION_BOX)
    def test_negative_지정_샷은_그_좌표를_반환한다(self):
        """Given: 2샷, 샷0=NEG_SHOT0, 샷1=NEG_SHOT1 지정 selection
        When:  _SelectionPlan.build 후 negatives_for(0), negatives_for(1)
        Then:  negatives_for(0) == NEG_SHOT0, negatives_for(1) == NEG_SHOT1

        WHY: track 오케스트레이션이 샷 인덱스로 그 샷의 negative 좌표를 O(1)
             조회해 add_box/add_click에 넘긴다. 조회가 정확해야 디스패치가 맞다.
        """
        selections = [
            CutSelection(
                shot_index=SHOT_0,
                point=SELECT_POINT_SHOT0,
                box=SELECT_BOX_SHOT0,
                negative_points=NEG_SHOT0,
            ),
            CutSelection(
                shot_index=SHOT_1,
                point=SELECT_POINT_SHOT1,
                box=SELECT_BOX_SHOT1,
                negative_points=NEG_SHOT1,
            ),
        ]
        plan = _SelectionPlan.build(
            RETRACK_FRAME_COUNT, [RETRACK_CUT_FRAME], selections
        )

        assert tuple(plan.negatives_for(SHOT_0)) == NEG_SHOT0, (
            f"샷0 negatives 조회 불일치: {plan.negatives_for(SHOT_0)} vs {NEG_SHOT0}"
        )
        assert tuple(plan.negatives_for(SHOT_1)) == NEG_SHOT1, (
            f"샷1 negatives 조회 불일치: {plan.negatives_for(SHOT_1)} vs {NEG_SHOT1}"
        )

    @pytest.mark.skipif(not _HAS_SELECTION_PLAN, reason=_MSG_NO_SELECTION_PLAN)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FOR, reason=_MSG_NO_NEGATIVES_FOR)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    @pytest.mark.skipif(not _HAS_SELECTION_BOX_FIELD, reason=_MSG_NO_SELECTION_BOX)
    def test_negative_미지정_샷은_빈_튜플을_반환한다(self):
        """Given: 2샷, 샷0만 negative 지정(샷1 미지정)
        When:  _SelectionPlan.build 후 negatives_for(1)
        Then:  negatives_for(1) == () (빈 튜플)

        WHY: negative를 지정하지 않은 샷은 빈 negatives여야 add_box/add_click가
             negative 없이 호출된다(무회귀). 미지정 샷 조회는 () 폴백이어야 한다.
        """
        selections = [
            CutSelection(
                shot_index=SHOT_0,
                point=SELECT_POINT_SHOT0,
                box=SELECT_BOX_SHOT0,
                negative_points=NEG_SHOT0,
            ),
            CutSelection(
                shot_index=SHOT_1,
                point=SELECT_POINT_SHOT1,
                box=SELECT_BOX_SHOT1,
            ),
        ]
        plan = _SelectionPlan.build(
            RETRACK_FRAME_COUNT, [RETRACK_CUT_FRAME], selections
        )

        assert tuple(plan.negatives_for(SHOT_1)) == (), (
            f"negative 미지정 샷이 빈 튜플이 아님: {plan.negatives_for(SHOT_1)}"
        )

    @pytest.mark.skipif(not _HAS_SELECTION_PLAN, reason=_MSG_NO_SELECTION_PLAN)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FOR, reason=_MSG_NO_NEGATIVES_FOR)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_NEGATIVES_FIELD, reason=_MSG_NO_NEGATIVES_FIELD)
    def test_selection_없는_샷_인덱스는_빈_튜플을_반환한다(self):
        """Given: 샷0만 선택(샷1은 selection 자체가 없음)
        When:  negatives_for(1) 조회
        Then:  () 반환 (KeyError 없이 안전한 폴백)

        WHY: 선택이 아예 없는 샷(자동 폴백 샷)을 조회해도 KeyError 없이 빈
             negatives를 돌려줘야 track 순회가 안전하다.
        """
        selections = [
            CutSelection(
                shot_index=SHOT_0,
                point=SELECT_POINT_SHOT0,
                negative_points=NEG_SHOT0,
            ),
        ]
        plan = _SelectionPlan.build(
            RETRACK_FRAME_COUNT, [RETRACK_CUT_FRAME], selections
        )

        assert tuple(plan.negatives_for(SHOT_1)) == (), (
            f"selection 없는 샷 조회가 빈 튜플이 아님: {plan.negatives_for(SHOT_1)}"
        )
