"""수동 교정 — 컷별 오브젝트 선택 오케스트레이션 테스트 (RED).

대상 모듈: easy_capture.app.video_capture (detect_cut_candidates·track(selections=) 신규)
테스트 더블: FakeVideoBackend·FakeFrameSource·FakeDetectionBackend (tests/fixtures/fakes.py)

배경:
  멀티샷 군무에서 자동 재매칭(IoU, feat=None)이 needs_correction 82.7%로
  구조적 실패. 해결책: 각 컷(샷) 시작 프레임에서 Grounding DINO 후보 검출
  (detect_cut_candidates) → 사용자가 추적 대상 명시 선택(CutSelection) →
  그 선택으로 컷별 재추적(track(selections=)). 자동 재매칭은 폴백.

이 파일이 검증하는 신규 계약:
  A. detect_cut_candidates(frames, cut_frames) -> list[ShotCandidates]
     - 각 샷 첫 프레임에서 detect 1회 → detect_call_count == 샷 수.
     - detector=None → 빈 리스트.
     - ShotCandidates(shot_index, first_frame_index, candidates).
  B. track(frames, point, cut_frames=None, selections=None)
     - selections=None: 기존 동작 그대로(무회귀 가드).
     - 전 샷 selection: needs_correction 전부 False, propagate_call_count == 샷 수.
     - 혼합: 선택 샷은 correction False, 미선택 샷은 자동 재매칭 폴백.
     - 범위 밖 selection: 한국어 ValueError(validate_selections 전파).

설계 경계:
  실모델 호출 금지 — FakeVideoBackend·FakeDetectionBackend만 사용.
  torch·PySide6·av 미의존. UI 무관(이 슬라이스는 core+app만).

구현 전 RED 상태가 정상:
  detect_cut_candidates·ShotCandidates·track(selections=) 미구현 →
  ImportError/AttributeError/TypeError로 skip 또는 fail(개별 집계).
"""
from __future__ import annotations

import numpy as np
import pytest

# --- video_capture 핵심 심볼 — 구현 전 try/except 격리 ---
# WHY: 기존 test_video_capture.py와 동일한 격리 패턴. 미구현 시 기존 테스트를
#      차단하지 않고 신규 테스트만 skip/fail로 개별 집계되게 한다.
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

# --- ShotCandidates — 컷 후보 묶음 dataclass(신규 — 구현 전 격리) ---
try:
    from easy_capture.app.video_capture import ShotCandidates
    _HAS_SHOT_CANDIDATES = True
except ImportError:
    ShotCandidates = None  # type: ignore[assignment,misc]
    _HAS_SHOT_CANDIDATES = False

# --- CutSelection — 컷별 선택 모델(신규 — 구현 전 격리) ---
try:
    from easy_capture.core.tracking.cut_selection import CutSelection
    _HAS_CUT_SELECTION = True
except ImportError:
    CutSelection = None  # type: ignore[assignment,misc]
    _HAS_CUT_SELECTION = False

# --- Detection Protocol — 구현 전 격리 ---
try:
    from easy_capture.core.segmentation.detection_backend import Detection
    _HAS_DETECTION = True
except ImportError:
    Detection = None  # type: ignore[assignment]
    _HAS_DETECTION = False

# --- track(selections=) 시그니처 존재 여부 판별 ---
# WHY: track에 selections 키워드 매개변수가 아직 없으면 신규 selection 테스트만
#      skip 처리되어 기존 테스트가 깨지지 않는다. 시그니처 inspect로 판별.
_HAS_SELECTIONS_PARAM = False
if _HAS_VIDEO_USECASE:
    import inspect

    try:
        _track_params = inspect.signature(VideoCaptureUseCase.track).parameters
        _HAS_SELECTIONS_PARAM = "selections" in _track_params
    except (ValueError, TypeError):
        _HAS_SELECTIONS_PARAM = False

# --- detect_cut_candidates 메서드 존재 여부 판별 ---
_HAS_DETECT_CANDIDATES = (
    _HAS_VIDEO_USECASE
    and hasattr(VideoCaptureUseCase, "detect_cut_candidates")
)

from tests.fixtures.fakes import (  # noqa: E402
    FakeDetectionBackend,
    FakeFrameSource,
    FakeVideoBackend,
)

# 미구현 시 skip 이유 메시지
_MSG_NOT_IMPL = "easy_capture.app.video_capture 미구현 — RED 예상"
_MSG_NO_SHOT_CANDIDATES = (
    "easy_capture.app.video_capture.ShotCandidates 미구현 — RED 예상"
)
_MSG_NO_CUT_SELECTION = (
    "core/tracking/cut_selection.py에 CutSelection 미구현 — RED 예상"
)
_MSG_NO_DETECTION = (
    "core/segmentation/detection_backend.py에 Detection 미구현 — RED 예상"
)
_MSG_NO_SELECTIONS = (
    "VideoCaptureUseCase.track에 selections 매개변수 미구현 — RED 예상"
)
_MSG_NO_DETECT_CANDIDATES = (
    "VideoCaptureUseCase.detect_cut_candidates 미구현 — RED 예상"
)

# ---------------------------------------------------------------------------
# 테스트 상수 (매직넘버 금지 — test_video_capture.py 상수와 동기화)
# ---------------------------------------------------------------------------
FAKE_FRAME_W = 640
FAKE_FRAME_H = 360

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

# 샷별 사용자 선택 클릭점 — FakeVideoBackend가 이 점 주변 사각형 마스크를 만든다.
#   선택 경로는 detector를 거치지 않고 이 점으로 직접 SAM2 재초기화하므로
#   유효한 화면 내 좌표이면 항상 추적 성공(needs_correction False)이어야 한다.
SELECT_POINT_SHOT0 = (320, 180)
SELECT_POINT_SHOT1 = (200, 150)
SELECT_POINT_SHOT2 = (400, 200)

# 자동 재매칭 폴백용 후보 박스 (test_video_capture.py와 동일 좌표 규약)
#   PASS_BOX: 직전 bbox와 IoU >= 0.5 예상 → 폴백 통과.
#   FAIL_BOX: 화면 좌상단 → IoU ≈ 0 → 폴백 미달(needs_correction).
RETRACK_PASS_BOX = (305, 165, 344, 204)
RETRACK_FAIL_BOX = (0, 0, 50, 50)

# 샷 인덱스
SHOT_0 = 0
SHOT_1 = 1
SHOT_2 = 2

# 범위 밖 샷 인덱스 (2샷인데 shot_index=2 → 0<=i<2 위반)
SHOT_OUT_OF_RANGE_2 = 2


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
    """FakeVideoBackend·FakeDetectionBackend 주입 VideoCaptureUseCase 쌍 반환.

    WHY: 기존 test_video_capture.py의 _make_retrack_usecase 패턴 계승.
         detector=None이면 단일 샷·자동 재매칭 미사용 경로로 동작한다.
    """
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
# A. detect_cut_candidates — 컷별 후보 검출
# ===========================================================================
class TestDetectCutCandidatesDetectorNone:
    """detector=None이면 빈 리스트를 반환한다(검출 불가 폴백)."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECT_CANDIDATES, reason=_MSG_NO_DETECT_CANDIDATES)
    def test_detector_None이면_빈_리스트를_반환한다(self):
        """Given: VideoCaptureUseCase(detector=None), 2샷 프레임/컷
        When:  detect_cut_candidates(frames, cut_frames) 호출
        Then:  [] 반환

        WHY: 검출기 없이는 후보를 만들 수 없다 — 단일 샷 폴백과 동형으로
             빈 리스트를 반환해 UI가 "후보 없음"으로 처리하게 한다.
        """
        usecase, _ = _make_usecase(detector=None)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        result = usecase.detect_cut_candidates(frames, [RETRACK_CUT_FRAME])

        assert result == [], (
            f"detector=None이면 빈 리스트여야 함, 실제: {result}"
        )


class TestDetectCutCandidatesCallCount:
    """각 샷 첫 프레임에서 detect 1회 → detect_call_count == 샷 수."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECT_CANDIDATES, reason=_MSG_NO_DETECT_CANDIDATES)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_2샷이면_detect_call_count가_샷_수_2이다(self):
        """Given: 20프레임, cut_frames=[10] → 2샷
        When:  detect_cut_candidates 호출
        Then:  detector.detect_call_count == 2 (각 샷 첫 프레임 1회씩)

        WHY: 각 샷 시작 프레임에서 사용자에게 보여줄 후보를 1회만 검출한다.
             샷마다 1회 — 컷 수가 아니라 샷 수(첫 샷 포함)다.
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[candidate])
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        usecase.detect_cut_candidates(frames, [RETRACK_CUT_FRAME])

        assert detector.detect_call_count == RETRACK_SHOT_COUNT, (
            f"detect 호출 횟수 불일치: {detector.detect_call_count} vs "
            f"{RETRACK_SHOT_COUNT}(샷 수)"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECT_CANDIDATES, reason=_MSG_NO_DETECT_CANDIDATES)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_3샷이면_detect_call_count가_샷_수_3이다(self):
        """Given: 30프레임, cut_frames=[10, 20] → 3샷
        When:  detect_cut_candidates 호출
        Then:  detector.detect_call_count == 3

        WHY: 다중 컷에서도 샷마다 1회 — 샷 수 누적 가드.
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(
            candidates_sequence=[[candidate], [candidate], [candidate]]
        )
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_3SHOT_FRAMES)

        usecase.detect_cut_candidates(frames, [RETRACK_CUT1, RETRACK_CUT2])

        assert detector.detect_call_count == RETRACK_3SHOT_COUNT, (
            f"detect 호출 횟수 불일치: {detector.detect_call_count} vs "
            f"{RETRACK_3SHOT_COUNT}(샷 수)"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECT_CANDIDATES, reason=_MSG_NO_DETECT_CANDIDATES)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_detect_cut_candidates는_propagate를_호출하지_않는다(self):
        """Given: 2샷 프레임/컷, FakeVideoBackend 스파이
        When:  detect_cut_candidates 호출
        Then:  backend.propagate_call_count == 0 (검출만, 추적 없음)

        WHY: 후보 검출은 사용자 선택 UI를 위한 가벼운 단계로, SAM2 전파를
             일으키면 안 된다. track(무거움)과의 분리 불변식을 가드한다.
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[candidate])
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        usecase.detect_cut_candidates(frames, [RETRACK_CUT_FRAME])

        assert backend.propagate_call_count == 0, (
            f"detect_cut_candidates가 propagate를 {backend.propagate_call_count}회 "
            "호출함 — 검출 단계는 SAM2 전파 금지"
        )


class TestShotCandidatesShape:
    """ShotCandidates 묶음 구조 — shot_index·first_frame_index·candidates."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECT_CANDIDATES, reason=_MSG_NO_DETECT_CANDIDATES)
    @pytest.mark.skipif(not _HAS_SHOT_CANDIDATES, reason=_MSG_NO_SHOT_CANDIDATES)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_반환_원소_수가_샷_수와_같다(self):
        """Given: 20프레임, cut_frames=[10] → 2샷
        When:  detect_cut_candidates 호출
        Then:  len(result) == 2

        WHY: UI가 샷별로 후보 패널을 띄우려면 샷 수만큼의 묶음이 필요하다.
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[candidate])
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        result = usecase.detect_cut_candidates(frames, [RETRACK_CUT_FRAME])

        assert len(result) == RETRACK_SHOT_COUNT, (
            f"ShotCandidates 수 불일치: {len(result)} vs {RETRACK_SHOT_COUNT}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECT_CANDIDATES, reason=_MSG_NO_DETECT_CANDIDATES)
    @pytest.mark.skipif(not _HAS_SHOT_CANDIDATES, reason=_MSG_NO_SHOT_CANDIDATES)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_각_묶음의_shot_index가_0부터_순서대로이다(self):
        """Given: 2샷
        When:  detect_cut_candidates 호출
        Then:  result[0].shot_index == 0, result[1].shot_index == 1

        WHY: shot_index가 CutSelection.shot_index와 일치해야 선택→재추적이
             올바르게 연결된다.
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[candidate])
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        result = usecase.detect_cut_candidates(frames, [RETRACK_CUT_FRAME])

        assert result[SHOT_0].shot_index == SHOT_0
        assert result[SHOT_1].shot_index == SHOT_1

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECT_CANDIDATES, reason=_MSG_NO_DETECT_CANDIDATES)
    @pytest.mark.skipif(not _HAS_SHOT_CANDIDATES, reason=_MSG_NO_SHOT_CANDIDATES)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_각_묶음의_first_frame_index가_샷_시작_프레임이다(self):
        """Given: 20프레임, cut_frames=[10] → 샷 시작 0, 10
        When:  detect_cut_candidates 호출
        Then:  result[0].first_frame_index == 0, result[1].first_frame_index == 10

        WHY: UI가 후보 검출 프레임(=샷 첫 프레임)을 썸네일로 보여주려면
             샷별 시작 프레임 인덱스가 필요하다. split_into_shots 경계와 일치.
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[candidate])
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        result = usecase.detect_cut_candidates(frames, [RETRACK_CUT_FRAME])

        assert result[SHOT_0].first_frame_index == 0
        assert result[SHOT_1].first_frame_index == RETRACK_CUT_FRAME

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_DETECT_CANDIDATES, reason=_MSG_NO_DETECT_CANDIDATES)
    @pytest.mark.skipif(not _HAS_SHOT_CANDIDATES, reason=_MSG_NO_SHOT_CANDIDATES)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_각_묶음의_candidates가_검출_후보를_담는다(self):
        """Given: 샷마다 후보 1개를 반환하는 검출기, 2샷
        When:  detect_cut_candidates 호출
        Then:  각 묶음 candidates 길이 == 1

        WHY: 사용자가 선택할 수 있도록 검출된 후보가 묶음에 그대로 담겨야 한다.
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[candidate])
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        result = usecase.detect_cut_candidates(frames, [RETRACK_CUT_FRAME])

        for shot in result:
            assert len(shot.candidates) == 1, (
                f"샷 {shot.shot_index} 후보 수 불일치: {len(shot.candidates)}"
            )


# ===========================================================================
# B-1. track(selections=None) 무회귀 가드
# ===========================================================================
class TestTrackSelectionsNoneRegression:
    """selections=None이면 기존 동작 그대로 — 무회귀 핵심 가드.

    WHY: selections 매개변수 추가는 기존 단일/멀티샷 경로를 절대 깨면 안 된다.
         selections=None은 "사용자 선택 없음 = 기존 자동 경로" 계약이다.
    """

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    def test_selections_None_단일샷이면_기존_단일샷_동작과_같다(self):
        """Given: detector=None, 10프레임, cut_frames·selections 미제공
        When:  track(frames, point) 호출
        Then:  TrackResult 반환, centroids 길이 == 10, propagate 1회

        WHY: 첫 슬라이스 단일 샷 경로가 selections 추가로 변하면 안 된다.
        """
        usecase, backend = _make_usecase(detector=None)
        frames = _make_frames(SINGLE_FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        assert isinstance(result, TrackResult)
        assert len(result.centroids) == SINGLE_FRAME_COUNT
        assert backend.propagate_call_count == 1, (
            f"단일 샷 propagate 호출 불일치: {backend.propagate_call_count} ≠ 1"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_selections_None_멀티샷이면_자동_재매칭_경로를_탄다(self):
        """Given: 통과 후보 검출기, 20프레임, cut_frames=[10], selections=None
        When:  track(frames, point, cut_frames=[10]) 호출
        Then:  propagate_call_count == 2(샷 수), detect_call_count == 1(컷 수)

        WHY: selections=None이면 기존 자동 재매칭(컷마다 detect 1회) 경로를
             그대로 타야 한다. selection 경로(detect 미호출)와 명확히 구분된다.
        """
        candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[candidate])
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        assert backend.propagate_call_count == RETRACK_SHOT_COUNT, (
            f"자동 재매칭 propagate 불일치: {backend.propagate_call_count}"
        )
        assert detector.detect_call_count == RETRACK_CUT_COUNT, (
            f"자동 재매칭 detect 불일치: {detector.detect_call_count}"
        )


# ===========================================================================
# B-2. 전 샷 selection — needs_correction 전부 False, 사용자 선택점 재추적
# ===========================================================================
class TestTrackAllShotsSelected:
    """모든 샷에 CutSelection을 주면 전 샷이 선택점으로 재추적된다."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_전_샷_선택_시_needs_correction이_전부_False이다(self):
        """Given: 20프레임, cut_frames=[10], 샷0·샷1 모두 selection 제공
        When:  track(..., selections=[샷0, 샷1]) 호출
        Then:  result.needs_correction 전부 False

        WHY: 사용자가 직접 추적 대상을 지정했으므로 자동 재매칭 실패 개념이
             사라진다 — 교정 필요 플래그가 서면 안 된다(슬라이스 핵심 목표).
        """
        # 자동 재매칭이라면 미달할 FAIL_BOX를 검출기에 주입해도,
        # selection 경로는 detector를 무시하고 사용자 선택점으로 재추적해야 한다.
        fail_candidate = Detection(box=RETRACK_FAIL_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[fail_candidate])
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)
        selections = [
            CutSelection(shot_index=SHOT_0, point=SELECT_POINT_SHOT0),
            CutSelection(shot_index=SHOT_1, point=SELECT_POINT_SHOT1),
        ]

        result = usecase.track(
            frames,
            (CLICK_X, CLICK_Y),
            cut_frames=[RETRACK_CUT_FRAME],
            selections=selections,
        )

        assert _all_false(result.needs_correction), (
            "전 샷 selection 시 needs_correction이 전부 False여야 한다 "
            f"(실제 True 개수: {sum(result.needs_correction)})"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_전_샷_선택_시_propagate_call_count가_샷_수와_같다(self):
        """Given: 20프레임, cut_frames=[10], 전 샷 selection
        When:  track 호출
        Then:  backend.propagate_call_count == 2 (샷마다 재초기화·전파 1회)

        WHY: selection 경로도 샷마다 SAM2를 재초기화하므로 propagate 호출 수가
             샷 수와 같아야 한다. 자동 경로 카운터 가드를 그대로 계승한다.
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

        assert backend.propagate_call_count == RETRACK_SHOT_COUNT, (
            f"propagate 호출 횟수 불일치: {backend.propagate_call_count} vs "
            f"{RETRACK_SHOT_COUNT}(샷 수)"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_전_샷_선택_시_centroids_길이가_전체_프레임_수이다(self):
        """Given: 20프레임, cut_frames=[10], 전 샷 selection
        When:  track 호출
        Then:  len(result.centroids) == 20

        WHY: 샷별 재추적 결과를 이어붙인 길이가 전체 프레임 수와 같아야
             compute_boxes·export가 올바른 길이의 박스/크롭을 만든다.
        """
        detector = FakeDetectionBackend(candidates_fixed=[])
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)
        selections = [
            CutSelection(shot_index=SHOT_0, point=SELECT_POINT_SHOT0),
            CutSelection(shot_index=SHOT_1, point=SELECT_POINT_SHOT1),
        ]

        result = usecase.track(
            frames,
            (CLICK_X, CLICK_Y),
            cut_frames=[RETRACK_CUT_FRAME],
            selections=selections,
        )

        assert len(result.centroids) == RETRACK_FRAME_COUNT, (
            f"centroids 길이 불일치: {len(result.centroids)} vs {RETRACK_FRAME_COUNT}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_3샷_전_선택_시_needs_correction이_전부_False이다(self):
        """Given: 30프레임, cut_frames=[10, 20], 샷0·1·2 모두 selection
        When:  track 호출
        Then:  needs_correction 전부 False, propagate_call_count == 3

        WHY: 다중 컷(3샷)에서도 전 샷 선택 시 교정 필요가 없어야 한다.
             멀티샷 군무 시나리오의 핵심 해결 경로다.
        """
        detector = FakeDetectionBackend(candidates_fixed=[])
        usecase, backend = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_3SHOT_FRAMES)
        selections = [
            CutSelection(shot_index=SHOT_0, point=SELECT_POINT_SHOT0),
            CutSelection(shot_index=SHOT_1, point=SELECT_POINT_SHOT1),
            CutSelection(shot_index=SHOT_2, point=SELECT_POINT_SHOT2),
        ]

        result = usecase.track(
            frames,
            (CLICK_X, CLICK_Y),
            cut_frames=[RETRACK_CUT1, RETRACK_CUT2],
            selections=selections,
        )

        assert _all_false(result.needs_correction), (
            "3샷 전 선택 시 needs_correction이 전부 False여야 한다 "
            f"(실제 True 개수: {sum(result.needs_correction)})"
        )
        assert backend.propagate_call_count == RETRACK_3SHOT_COUNT, (
            f"3샷 propagate 불일치: {backend.propagate_call_count}"
        )


# ===========================================================================
# B-3. 혼합 selection — 선택 샷은 correction False, 미선택 샷은 자동 폴백
# ===========================================================================
class TestTrackMixedSelection:
    """일부 샷만 selection — 나머지는 자동 재매칭 폴백."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_미선택_샷이_자동폴백_미달이면_correction이_True가_된다(self):
        """Given: 3샷, 샷1만 selection, 검출기는 미달 후보(FAIL_BOX)
        When:  track 호출
        Then:  선택된 샷1 구간은 correction False,
               미선택 샷2 구간(자동 폴백 미달)에 correction True 존재

        WHY: 혼합 시나리오 — 선택한 샷은 안전하게 재추적, 선택 안 한 샷은
             자동 재매칭 폴백을 타고 미달 시 교정 필요로 표시한다.
             컷=[10,20] → 샷0[0,10)·샷1[10,20)·샷2[20,30).
        """
        fail_candidate = Detection(box=RETRACK_FAIL_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[fail_candidate])
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_3SHOT_FRAMES)
        # 샷1만 선택 (샷0은 첫 클릭, 샷2는 자동 폴백)
        selections = [CutSelection(shot_index=SHOT_1, point=SELECT_POINT_SHOT1)]

        result = usecase.track(
            frames,
            (CLICK_X, CLICK_Y),
            cut_frames=[RETRACK_CUT1, RETRACK_CUT2],
            selections=selections,
        )

        # 선택된 샷1 구간[10,20)은 correction False
        shot1_correction = result.needs_correction[RETRACK_CUT1:RETRACK_CUT2]
        assert _all_false(shot1_correction), (
            "선택된 샷1 구간은 needs_correction이 전부 False여야 한다"
        )
        # 미선택 샷2 구간[20,30)은 자동 폴백 미달 → correction True 존재
        shot2_correction = result.needs_correction[RETRACK_CUT2:]
        assert any(shot2_correction), (
            "미선택 샷2(자동 폴백 미달) 구간에 needs_correction=True가 있어야 한다"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_미선택_샷이_자동폴백_통과이면_correction이_False이다(self):
        """Given: 2샷, selection 없이 통과 후보(PASS_BOX) 제공
        When:  track(..., selections=[]) 호출
        Then:  needs_correction 전부 False (자동 폴백이 통과했으므로)

        WHY: 빈 selection 리스트는 selections=None과 동형으로 전 샷 자동 폴백.
             자동 폴백이 통과하면 교정 필요가 없다 — 폴백 경로 정상성 가드.
        """
        pass_candidate = Detection(box=RETRACK_PASS_BOX, score=0.9, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[pass_candidate])
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)

        result = usecase.track(
            frames,
            (CLICK_X, CLICK_Y),
            cut_frames=[RETRACK_CUT_FRAME],
            selections=[],
        )

        assert _all_false(result.needs_correction), (
            "자동 폴백이 통과하면 needs_correction이 전부 False여야 한다 "
            f"(실제 True 개수: {sum(result.needs_correction)})"
        )


# ===========================================================================
# B-4. 범위 밖 selection — validate_selections 전파 ValueError
# ===========================================================================
class TestTrackSelectionValidation:
    """범위 밖 shot_index selection → 한국어 ValueError 전파."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_샷_수_이상의_shot_index_selection이면_ValueError가_발생한다(self):
        """Given: 2샷(cut_frames=[10]), shot_index=2 selection (0<=i<2 위반)
        When:  track 호출
        Then:  ValueError 발생 (validate_selections 전파)

        WHY: 존재하지 않는 샷을 가리키는 선택은 추적 시작 전에 차단해야
             IndexError·조용한 오작동을 막는다.
        """
        detector = FakeDetectionBackend(candidates_fixed=[])
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)
        selections = [
            CutSelection(shot_index=SHOT_OUT_OF_RANGE_2, point=SELECT_POINT_SHOT0),
        ]

        with pytest.raises(ValueError):
            usecase.track(
                frames,
                (CLICK_X, CLICK_Y),
                cut_frames=[RETRACK_CUT_FRAME],
                selections=selections,
            )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_SELECTIONS_PARAM, reason=_MSG_NO_SELECTIONS)
    @pytest.mark.skipif(not _HAS_CUT_SELECTION, reason=_MSG_NO_CUT_SELECTION)
    def test_범위_밖_selection_ValueError_메시지는_한국어를_포함한다(self):
        """Given: 2샷, shot_index=2 selection
        When:  track 호출 → ValueError
        Then:  메시지에 위반 수치(2)가 포함된다

        WHY: validate_selections의 한국어·수치 메시지가 track을 통해 그대로
             사용자에게 전파되는지 확인한다(에러 메시지 한국어 정책).
        """
        detector = FakeDetectionBackend(candidates_fixed=[])
        usecase, _ = _make_usecase(detector=detector)
        frames = _make_frames(RETRACK_FRAME_COUNT)
        selections = [
            CutSelection(shot_index=SHOT_OUT_OF_RANGE_2, point=SELECT_POINT_SHOT0),
        ]

        with pytest.raises(ValueError, match=str(SHOT_OUT_OF_RANGE_2)):
            usecase.track(
                frames,
                (CLICK_X, CLICK_Y),
                cut_frames=[RETRACK_CUT_FRAME],
                selections=selections,
            )
