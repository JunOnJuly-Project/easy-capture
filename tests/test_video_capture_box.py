"""box 프롬프트 디스패치 — VideoCaptureUseCase.track 오케스트레이션 테스트 (RED).

대상 모듈: easy_capture.app.video_capture (track box 프롬프트 디스패치)
테스트 더블: FakeVideoBackend(add_box 스파이)·FakeFrameSource·FakeDetectionBackend

배경:
  멀티샷 군무에서 마스크가 부정확·과대(대상 배 + 옆사람 팔)했다. 원인:
  SAM2에 detect 전신 bbox를 안 주고 박스 중심점(배) 1개만(point 프롬프트)
  넘기는 방식으로 production이 회귀했다. PoC는 box 프롬프트(detect bbox→
  SAM2)를 썼다. 해결: 자동 재매칭/사용자 선택이 결정한 전신 bbox를
  add_box로 SAM2에 직접 전달한다(중심점 변환 _box_center 폐기).

이 파일이 검증하는 신규 계약:
  A-1. 자동 재매칭 통과 후속 샷: detect best box를 add_box로 전달.
       - add_box_call_count == 통과 샷 수.
       - 그 샷은 add_click 미호출(중심점 변환 폐기).
       - propagate_call_count == 샷 수(box든 click이든 샷당 1회).
  A-2. box 없는 경로(첫 샷 point만, 또는 검출 실패) → 기존 add_click 폴백.
  A-3. selection.box 우선 디스패치: selection.box 있으면 add_box, 없고 point면 add_click.
  A-4. 무회귀: 단일샷·기존 멀티샷 자동 경로가 box 디스패치로 바뀌어도
       propagate 카운터·결과 길이 불변.

설계 경계:
  실모델 호출 금지 — FakeVideoBackend·FakeDetectionBackend만 사용.
  torch·PySide6·av 미의존. UI 무관(core+app만).

구현 전 RED 상태가 정상:
  track이 아직 add_click(중심점)만 호출하면 add_box_call_count 단언이 실패한다.
  add_box 미디스패치 → box 가드(_box_dispatch_works)가 False → box 테스트만 skip/fail.
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

# --- Detection Protocol — 구현 전 격리 ---
try:
    from easy_capture.core.segmentation.detection_backend import Detection
    _HAS_DETECTION = True
except ImportError:
    Detection = None  # type: ignore[assignment]
    _HAS_DETECTION = False

# --- CutSelection — box 필드 포함 여부 격리 ---
try:
    from easy_capture.core.tracking.cut_selection import CutSelection
    _HAS_CUT_SELECTION = True
except ImportError:
    CutSelection = None  # type: ignore[assignment,misc]
    _HAS_CUT_SELECTION = False

_HAS_SELECTION_BOX_FIELD = False
if _HAS_CUT_SELECTION:
    from dataclasses import fields as _dc_fields

    _HAS_SELECTION_BOX_FIELD = any(
        f.name == "box" for f in _dc_fields(CutSelection)
    )

from tests.fixtures.fakes import (  # noqa: E402
    FakeDetectionBackend,
    FakeFrameSource,
    FakeVideoBackend,
)

# --- FakeVideoBackend.add_box 더블 존재 여부(테스트 더블이므로 항상 존재 기대) ---
_HAS_FAKE_ADD_BOX = hasattr(FakeVideoBackend, "add_box")

# --- track(selections=) 시그니처 존재 여부 ---
_HAS_SELECTIONS_PARAM = False
if _HAS_VIDEO_USECASE:
    import inspect

    try:
        _track_params = inspect.signature(VideoCaptureUseCase.track).parameters
        _HAS_SELECTIONS_PARAM = "selections" in _track_params
    except (ValueError, TypeError):
        _HAS_SELECTIONS_PARAM = False

_MSG_NOT_IMPL = "easy_capture.app.video_capture 미구현 — RED 예상"
_MSG_NO_DETECTION = (
    "core/segmentation/detection_backend.py에 Detection 미구현 — RED 예상"
)
_MSG_NO_ADD_BOX = "FakeVideoBackend.add_box 미구현 — RED 예상"
_MSG_NO_SELECTIONS = (
    "VideoCaptureUseCase.track에 selections 매개변수 미구현 — RED 예상"
)
_MSG_NO_SELECTION_BOX = "CutSelection.box 필드 미구현 — box 프롬프트 RED 예상"

# ---------------------------------------------------------------------------
# 테스트 상수 (매직넘버 금지 — test_video_capture_selection.py 규약 동기화)
# ---------------------------------------------------------------------------
# 첫 샷 사용자 클릭 포인트
CLICK_X = 320
CLICK_Y = 180

# 2샷 시나리오 — 전체 20프레임, 컷 1개(프레임 10)
RETRACK_FRAME_COUNT = 20
RETRACK_CUT_FRAME = 10
RETRACK_SHOT_COUNT = 2     # 컷 1개 → 2샷
RETRACK_CUT_COUNT = 1

# 3샷 시나리오 — 전체 30프레임, 컷 2개
RETRACK_3SHOT_FRAMES = 30
RETRACK_CUT1 = 10
RETRACK_CUT2 = 20
RETRACK_3SHOT_COUNT = 3

# 단일 샷(컷 없음) 시나리오 — 무회귀 가드용
SINGLE_FRAME_COUNT = 10

# 자동 재매칭 폴백용 후보 박스 (test_video_capture_selection.py 좌표 규약)
#   PASS_BOX: 직전 bbox와 IoU >= 0.5 예상 → 폴백 통과(box 프롬프트 디스패치).
#   FAIL_BOX: 화면 좌상단 → IoU ≈ 0 → 폴백 미달(needs_correction, box 미사용).
RETRACK_PASS_BOX = (305, 165, 344, 204)
RETRACK_FAIL_BOX = (0, 0, 50, 50)

# 사용자 선택 — point + box (Story A-3 selection.box 우선 디스패치용)
SELECT_POINT_SHOT0 = (320, 180)
SELECT_POINT_SHOT1 = (200, 150)
SELECT_BOX_SHOT0 = (300.0, 160.0, 340.0, 200.0)
SELECT_BOX_SHOT1 = (180.0, 130.0, 220.0, 170.0)

# 샷 인덱스
SHOT_0 = 0
SHOT_1 = 1
SHOT_2 = 2


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


def _all_false(flags: list[bool]) -> bool:
    """플래그 리스트가 전부 False인지 반환한다(가독성 헬퍼)."""
    return all(not f for f in flags)


# ===========================================================================
# A-1. 자동 재매칭 통과 후속 샷 — detect best box를 add_box로 전달
# ===========================================================================
class TestAutoRematchBoxDispatch:
    """자동 재매칭 통과 샷은 중심점 변환 대신 detect best box를 add_box로 넘긴다."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_ADD_BOX, reason=_MSG_NO_ADD_BOX)
    def test_재매칭_통과_후속_샷이_add_box를_통과_샷_수만큼_호출한다(self):
        """Given: 2샷(컷 1개), 후속 샷에서 통과 후보(PASS_BOX) 검출
        When:  track(frames, point, cut_frames=[10]) 호출
        Then:  backend.add_box_call_count == 1 (후속 통과 샷 1개)

        WHY: PoC가 쓰던 box 프롬프트로 복귀 — 자동 재매칭이 통과한 후속 샷은
             detect best box를 그대로 SAM2 box 프롬프트로 넘겨야 마스크가
             정확해진다(중심점 1개만 주던 회귀 해결). 첫 샷(point 클릭)은
             add_box를 쓰지 않으므로 통과 후속 샷 수(=1)만큼만 호출돼야 한다.
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[candidate])
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        # 후속 통과 샷 1개 → add_box 1회
        assert backend.add_box_call_count == RETRACK_CUT_COUNT, (
            f"add_box 호출 횟수 불일치: {backend.add_box_call_count} vs "
            f"{RETRACK_CUT_COUNT}(통과 후속 샷 수) — box 프롬프트 미디스패치 의심"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_ADD_BOX, reason=_MSG_NO_ADD_BOX)
    def test_재매칭_통과_후속_샷은_add_click을_쓰지_않는다(self):
        """Given: 2샷, 후속 샷 통과 후보(PASS_BOX)
        When:  track 호출
        Then:  add_click_call_count == 1 (첫 샷 point 클릭만, 후속 샷은 box)

        WHY: 중심점 변환(_box_center→add_click)을 폐기했으므로, 통과 후속 샷은
             add_click을 호출하면 안 된다. 첫 샷의 사용자 point 클릭 1회만 남는다.
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[candidate])
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        # 첫 샷 point 1회만 — 후속 통과 샷은 box로 갈음
        assert backend.add_click_call_count == 1, (
            f"add_click 호출 횟수 불일치: {backend.add_click_call_count} vs 1 — "
            "통과 후속 샷이 여전히 중심점(add_click)을 쓰는 것으로 보임"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_ADD_BOX, reason=_MSG_NO_ADD_BOX)
    def test_box_디스패치여도_propagate_call_count가_샷_수와_같다(self):
        """Given: 2샷, 후속 샷 통과 후보(PASS_BOX)
        When:  track 호출
        Then:  propagate_call_count == 2 (box든 click이든 샷당 1회)

        WHY: box 프롬프트로 바뀌어도 샷마다 SAM2 전파는 정확히 1회여야 한다.
             box 디스패치가 전파 횟수를 늘리거나 줄이면 비용·결과 길이가
             어긋난다 — 전파 카운터 불변식을 가드한다.
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
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_ADD_BOX, reason=_MSG_NO_ADD_BOX)
    def test_3샷_모두_재매칭_통과면_add_box를_2회_호출한다(self):
        """Given: 3샷(컷 2개), 후속 샷마다 통과 후보(PASS_BOX)
        When:  track 호출
        Then:  add_box_call_count == 2 (후속 통과 샷 2개)

        WHY: 다중 컷에서도 첫 샷을 제외한 통과 후속 샷마다 box 프롬프트를
             1회씩 써야 한다. 첫 샷은 사용자 point이므로 box는 컷 수만큼이다.
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(
            candidates_sequence=[[candidate], [candidate]]
        )
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_3SHOT_FRAMES)

        usecase.track(
            frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT1, RETRACK_CUT2]
        )

        # 후속 통과 샷 2개 → add_box 2회
        assert backend.add_box_call_count == RETRACK_3SHOT_COUNT - 1, (
            f"add_box 호출 횟수 불일치: {backend.add_box_call_count} vs "
            f"{RETRACK_3SHOT_COUNT - 1}(통과 후속 샷 수)"
        )


# ===========================================================================
# A-2. box 없는 경로 — add_click 폴백(무회귀)
# ===========================================================================
class TestNoBoxFallbackUsesClick:
    """box를 쓸 수 없는 경로는 기존 add_click 폴백을 그대로 탄다."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_FAKE_ADD_BOX, reason=_MSG_NO_ADD_BOX)
    def test_단일샷_첫샷은_point_클릭이라_add_box를_안_쓴다(self):
        """Given: detector=None, 10프레임(단일 샷)
        When:  track(frames, point) 호출
        Then:  add_box_call_count == 0, add_click_call_count == 1

        WHY: 단일 샷(첫 샷)은 사용자가 클릭한 point만 있으므로 box 프롬프트가
             없다 — 기존 add_click 경로를 그대로 써야 한다(무회귀).
        """
        usecase, backend = _make_usecase(detector=None)
        frames = _make_frames(SINGLE_FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y))

        assert backend.add_box_call_count == 0, (
            f"단일 샷에서 add_box가 {backend.add_box_call_count}회 호출됨 — "
            "point 경로는 box를 쓰면 안 된다"
        )
        assert backend.add_click_call_count == 1, (
            f"단일 샷 add_click 불일치: {backend.add_click_call_count} vs 1"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_ADD_BOX, reason=_MSG_NO_ADD_BOX)
    def test_재매칭_미달_샷은_add_box를_안_쓰고_add_click_폴백을_탄다(self):
        """Given: 2샷, 후속 샷에서 미달 후보(FAIL_BOX, IoU≈0)
        When:  track 호출
        Then:  add_box_call_count == 0 (통과 샷 없음),
               needs_correction에 True 존재(미달 폴백 동작)

        WHY: 재매칭 미달 샷은 신뢰할 box가 없으므로 box 프롬프트를 쓰면 안 된다.
             기존 hold(중앙 클릭) 폴백을 그대로 유지해야 한다(무회귀).
        """
        fail_candidate = Detection(box=RETRACK_FAIL_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[fail_candidate])
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        result = usecase.track(
            frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME]
        )

        assert backend.add_box_call_count == 0, (
            f"미달 샷에서 add_box가 {backend.add_box_call_count}회 호출됨 — "
            "신뢰할 box 없는 미달 폴백은 box를 쓰면 안 된다"
        )
        # 미달 샷 구간[10,20)에 교정 필요 플래그가 서야 한다(폴백 정상 동작)
        assert any(result.needs_correction[RETRACK_CUT_FRAME:]), (
            "미달 후속 샷 구간에 needs_correction=True가 있어야 한다"
        )


# ===========================================================================
# A-3. selection.box 우선 디스패치 — box 있으면 add_box, 없으면 add_click
# ===========================================================================
class TestSelectionBoxDispatch:
    """사용자 선택(CutSelection)에 box가 있으면 add_box로, point뿐이면 add_click."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_SELECTION_BOX_FIELD, reason=_MSG_NO_SELECTION_BOX)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_ADD_BOX, reason=_MSG_NO_ADD_BOX)
    def test_box를_가진_selection은_add_box로_디스패치된다(self):
        """Given: 2샷, 샷0·샷1 모두 box를 가진 selection
        When:  track(..., selections=[box 포함 샷0, 샷1]) 호출
        Then:  add_box_call_count == 2 (두 샷 모두 box 프롬프트),
               add_click_call_count == 0 (point 클릭 미사용)

        WHY: 사용자가 후보 박스를 선택하면 그 전신 bbox를 SAM2 box 프롬프트로
             직접 넘겨야 정확하다. selection.box가 있으면 중심점(add_click) 대신
             add_box로 디스패치해야 한다(box 우선 정책).
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

        assert backend.add_box_call_count == RETRACK_SHOT_COUNT, (
            f"box selection add_box 불일치: {backend.add_box_call_count} vs "
            f"{RETRACK_SHOT_COUNT}(box 가진 샷 수)"
        )
        assert backend.add_click_call_count == 0, (
            f"box selection인데 add_click이 {backend.add_click_call_count}회 호출됨 "
            "— box 우선 디스패치 미구현 의심"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_SELECTION_BOX_FIELD, reason=_MSG_NO_SELECTION_BOX)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_ADD_BOX, reason=_MSG_NO_ADD_BOX)
    def test_box없이_point만_가진_selection은_add_click으로_디스패치된다(self):
        """Given: 2샷, 샷0·샷1 모두 box=None(point만) selection
        When:  track(..., selections=[point만 샷0, 샷1]) 호출
        Then:  add_box_call_count == 0, add_click_call_count == 2

        WHY: box가 없는 선택(기존 point 선택)은 add_click 경로를 그대로 타야
             한다 — box 필드 추가가 기존 point 선택 동작을 깨면 안 된다(무회귀).
        """
        detector = FakeDetectionBackend(candidates_fixed=[])
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)
        selections = [
            CutSelection(shot_index=SHOT_0, point=SELECT_POINT_SHOT0),
            CutSelection(shot_index=SHOT_1, point=SELECT_POINT_SHOT1),
        ]

        usecase.track(
            frames,
            (CLICK_X, CLICK_Y),
            cut_frames=[RETRACK_CUT_FRAME],
            selections=selections,
        )

        assert backend.add_box_call_count == 0, (
            f"point만 가진 selection인데 add_box가 {backend.add_box_call_count}회 "
            "호출됨 — box 없으면 add_click 폴백이어야 한다"
        )
        assert backend.add_click_call_count == RETRACK_SHOT_COUNT, (
            f"point selection add_click 불일치: {backend.add_click_call_count} vs "
            f"{RETRACK_SHOT_COUNT}"
        )


# ===========================================================================
# A-4. 무회귀 — 단일샷·멀티샷 결과 길이·전파 카운터 불변
# ===========================================================================
class TestBoxDispatchNoRegression:
    """box 디스패치 도입이 propagate 카운터·결과 길이를 바꾸지 않는다."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_FAKE_ADD_BOX, reason=_MSG_NO_ADD_BOX)
    def test_단일샷_결과_길이와_propagate가_불변이다(self):
        """Given: detector=None, 10프레임(단일 샷)
        When:  track(frames, point) 호출
        Then:  centroids 길이 == 10, propagate_call_count == 1

        WHY: box 프롬프트 도입은 단일 샷(컷 없음) 경로를 절대 건드리면 안 된다.
        """
        usecase, backend = _make_usecase(detector=None)
        frames = _make_frames(SINGLE_FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        assert isinstance(result, TrackResult)
        assert len(result.centroids) == SINGLE_FRAME_COUNT, (
            f"단일 샷 centroids 길이 불일치: {len(result.centroids)}"
        )
        assert backend.propagate_call_count == 1, (
            f"단일 샷 propagate 불일치: {backend.propagate_call_count} ≠ 1"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_ADD_BOX, reason=_MSG_NO_ADD_BOX)
    def test_멀티샷_자동경로_결과_길이가_전체_프레임_수이다(self):
        """Given: 2샷, 통과 후보(PASS_BOX), selections 없음
        When:  track(frames, point, cut_frames=[10]) 호출
        Then:  centroids 길이 == 20 (box 디스패치로 바뀌어도 길이 불변)

        WHY: box 프롬프트로 후속 샷을 재추적해도 이어붙인 결과 길이는 전체
             프레임 수와 같아야 compute_boxes·export가 올바르게 동작한다.
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[candidate])
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        result = usecase.track(
            frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME]
        )

        assert len(result.centroids) == RETRACK_FRAME_COUNT, (
            f"멀티샷 centroids 길이 불일치: {len(result.centroids)} vs "
            f"{RETRACK_FRAME_COUNT}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    @pytest.mark.skipif(not _HAS_FAKE_ADD_BOX, reason=_MSG_NO_ADD_BOX)
    def test_멀티샷_통과_경로는_needs_correction이_전부_False이다(self):
        """Given: 2샷, 통과 후보(PASS_BOX), selections 없음
        When:  track 호출
        Then:  needs_correction 전부 False

        WHY: 자동 재매칭이 통과하면(box 프롬프트로 재추적) 교정 필요가 없어야
             한다 — box 디스패치가 통과 판정 결과를 바꾸면 안 된다(무회귀).
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[candidate])
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        result = usecase.track(
            frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME]
        )

        assert _all_false(result.needs_correction), (
            "통과 경로 needs_correction이 전부 False여야 한다 "
            f"(실제 True 개수: {sum(result.needs_correction)})"
        )
