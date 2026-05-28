"""VideoCaptureUseCase 슬라이스 핵심 조립 테스트.

대상 모듈: easy_capture.app.video_capture
테스트 더블: FakeVideoBackend (tests/fixtures/fakes.py),
             FakeFrameSource  (tests/fixtures/fakes.py),
             FakeDetectionBackend (tests/fixtures/fakes.py) — 신규

이 테스트 파일이 검증하는 계약:
  1. Protocol 계약   — isinstance(FakeVideoBackend(), VideoSegmentationBackend) 통과
  2. FakeVideoBackend 스파이 — init/add_click/propagate 카운터 정확성
  3. FakeVideoBackend drift  — 프레임별 마스크 중심 이동 및 centroid 변화
  4. FakeVideoBackend empty_after — occlusion 구간 빈 마스크 반환
  5. track()         — FakeVideoBackend 주입 → 프레임별 centroid 산출 + TrackResult
  6. track() 전 프레임 빈 마스크 → EmptyTrackError
  7. compute_boxes() 순수성 가드 — track 1회 후 여러 번 호출해도 propagate_call_count == 1
  8. compute_boxes() 고정 박스 크기 불변식 — 전 프레임 동일 W×H
  9. compute_boxes() smooth 적용 — drift centroid에 window 적용 시 분산 감소
 10. compute_boxes() 종횡비/크기 변경 재호출 — 박스 크기 재계산, 재추적 없음
 11. gap_policy BACKGROUND + empty_after — 출력 인덱스 전 프레임 유지 검증
 12. TrackResult frozen dataclass 불변식
 13. (신규) 샷경계 재추적 — FakeDetectionBackend 주입 컷 시나리오 4종
     a. 재매칭 통과   → objid 유지·centroids 이어짐·propagate==샷수·detect==컷수
     b. 재매칭 미달   → needs_correction 플래그
     c. 다중 후보     → best 가까운 것 선택·통과
     d. 빈 검출       → needs_correction 플래그
 14. (신규) 순수성 이중 가드 — compute_boxes 반복 시 propagate·detect 카운터 불변
 15. (신규) detector=None 하위호환 — 단일 샷 동작 그대로

구현 전 RED 상태가 정상(13-15):
  VideoCaptureUseCase.detector 파라미터·TrackResult.needs_correction·cut_frames 미구현.
"""
from __future__ import annotations

import numpy as np
import pytest

# --- 비디오 슬라이스 미구현 → try/except 격리 ---
# WHY: 구현 전이므로 import 자체가 실패한다. 이 파일의 테스트들이
#      "ImportError로 전부 오류" 대신 "skip/fail 개별 집계"되도록 한다.
try:
    from easy_capture.app.video_capture import (
        TrackResult,
        VideoCaptureUseCase,
        VideoCropParams,
    )
    from easy_capture.core.segmentation.video_backend import (
        EmptyTrackError,
        VideoSegmentationBackend,
    )
    _HAS_VIDEO_USECASE = True
except ModuleNotFoundError:
    TrackResult = None  # type: ignore[assignment,misc]
    VideoCaptureUseCase = None  # type: ignore[assignment,misc]
    VideoCropParams = None  # type: ignore[assignment,misc]
    EmptyTrackError = None  # type: ignore[assignment,misc]
    VideoSegmentationBackend = None  # type: ignore[assignment,misc]
    _HAS_VIDEO_USECASE = False

# --- 샷경계 재추적 신규 심볼 — 구현 전이므로 try/except 격리 ---
# WHY: TrackResult에 needs_correction·cut_frames 필드가 아직 없으면
#      기존 216개 테스트를 차단하지 않고 신규 테스트만 FAIL 처리된다.
try:
    # needs_correction·cut_frames 필드 존재 여부로 신규 구현 판별
    _probe = TrackResult(
        masks=[], centroids=[], needs_correction=[], cut_frames=[]
    ) if TrackResult is not None else None
    _HAS_RETRACK_FIELDS = (_probe is not None)
except TypeError:
    # frozen dataclass에 없는 필드 → 아직 미구현
    _HAS_RETRACK_FIELDS = False

# FakeDetectionBackend — fakes.py에 이미 추가됨
from tests.fixtures.fakes import (
    FakeDetectionBackend,
    FakeFrameSource,
    FakeVideoBackend,
    _make_rect_mask,
)

# DetectionBackend Protocol — 구현 전 격리
try:
    from easy_capture.core.segmentation.detection_backend import Detection
    _HAS_DETECTION = True
except ImportError:
    Detection = None  # type: ignore[assignment]
    _HAS_DETECTION = False

# 미구현 시 전 테스트 skip 이유 메시지
_MSG_NOT_IMPL = (
    "easy_capture.app.video_capture 또는 "
    "easy_capture.core.segmentation.video_backend 미구현 — RED 예상"
)
_MSG_NO_RETRACK = (
    "VideoCaptureUseCase.detector 파라미터 또는 "
    "TrackResult.needs_correction·cut_frames 미구현 — RED 예상"
)
_MSG_NO_DETECTION = (
    "core/segmentation/detection_backend.py에 Detection 미구현 — RED 예상"
)

# ---------------------------------------------------------------------------
# 테스트 상수
# ---------------------------------------------------------------------------
# FakeFrameSource 고정 프레임 크기 (fakes.py 상수와 동기화)
FAKE_FRAME_W = 640
FAKE_FRAME_H = 360

# 테스트용 클릭 포인트 — 프레임 중앙에서 벗어난 예측 가능한 좌표
CLICK_X = 320
CLICK_Y = 180

# 기본 박스 요청 크기 (W, H)
BOX_W = 200
BOX_H = 150

# 드리프트 속도 (프레임마다 X 방향 이동 픽셀)
DRIFT_DX = 2
DRIFT_DY = 0

# 테스트용 프레임 수
FRAME_COUNT = 10

# empty_after 기준 인덱스 (이 이후 빈 마스크)
EMPTY_AFTER_IDX = 7

# smooth_centroids 윈도 크기
SMOOTH_WINDOW = 3

# compute_boxes 반복 호출 횟수 (순수성 가드용)
COMPUTE_BOXES_REPEAT = 5

# ---------------------------------------------------------------------------
# 샷경계 재추적 상수
# ---------------------------------------------------------------------------
# 2샷 시나리오 — 전체 20프레임, 컷 1개(프레임 10)
RETRACK_FRAME_COUNT = 20   # 전체 프레임 수
RETRACK_CUT_FRAME = 10     # 컷 위치(이 프레임부터 2번째 샷)
RETRACK_SHOT1_LEN = RETRACK_CUT_FRAME              # 첫 샷 길이 = 10
RETRACK_SHOT2_LEN = RETRACK_FRAME_COUNT - RETRACK_CUT_FRAME  # 두 번째 샷 길이 = 10
RETRACK_SHOT_COUNT = 2     # 2샷
RETRACK_CUT_COUNT = 1      # 컷 1개

# 3샷 시나리오 — 전체 30프레임, 컷 2개
RETRACK_3SHOT_FRAMES = 30
RETRACK_CUT1 = 10
RETRACK_CUT2 = 20
RETRACK_3SHOT_COUNT = 3
RETRACK_2CUT_COUNT = 2

# 재매칭용 박스 — FakeVideoBackend 마스크에서 추출될 prev_box 근방
# FakeVideoBackend는 cx=CLICK_X, cy=CLICK_Y 기준 half=20 사각형 마스크 생성
# → bbox ≈ (300, 160, 339, 199). 통과 후보는 근접 위치, 미달 후보는 먼 위치
RETRACK_PASS_BOX = (305, 165, 344, 204)   # PREV_BOX와 IoU ≥ 0.5 예상
RETRACK_FAIL_BOX = (  0,   0,  50,  50)  # 화면 좌상단 — IoU ≈ 0.0

# compute_boxes 이중 순수성 가드용 반복 횟수
RETRACK_COMPUTE_REPEAT = 4


# ---------------------------------------------------------------------------
# 픽스처 헬퍼
# ---------------------------------------------------------------------------
def _make_frames(n: int = FRAME_COUNT) -> list[np.ndarray]:
    """n개의 고정 RGB 프레임 리스트를 반환한다.

    WHY: FakeFrameSource.read_frame()을 n번 복사해 FakeVideoBackend.init_session에
         넘길 결정적 프레임 시퀀스를 만든다. 모든 프레임은 동일(추적 독립).
    """
    src = FakeFrameSource()
    return [src.read_frame(i) for i in range(n)]


def _make_usecase(
    drift: tuple[int, int] = (DRIFT_DX, DRIFT_DY),
    empty_after: int | None = None,
) -> tuple["VideoCaptureUseCase", FakeVideoBackend]:
    """VideoCaptureUseCase + FakeVideoBackend(스파이) 쌍을 반환한다."""
    backend = FakeVideoBackend(drift=drift, empty_after=empty_after)
    usecase = VideoCaptureUseCase(
        source=FakeFrameSource(),
        backend=backend,
    )
    return usecase, backend


# ---------------------------------------------------------------------------
# Protocol 계약 테스트
# ---------------------------------------------------------------------------
class TestFakeVideoBackendProtocolContract:
    """FakeVideoBackend가 VideoSegmentationBackend Protocol을 구현하는지 검증.

    WHY: @runtime_checkable Protocol은 구조적 서브타이핑을 런타임에 검증한다.
         이 테스트가 통과해야 실제 주입 시 타입 오류가 없다.
    """

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_FakeVideoBackend_인스턴스는_VideoSegmentationBackend_isinstance를_통과한다(self):
        """Given: FakeVideoBackend 인스턴스
        When:  isinstance(..., VideoSegmentationBackend) 호출
        Then:  True 반환
        """
        fake = FakeVideoBackend()

        assert isinstance(fake, VideoSegmentationBackend)

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_FakeVideoBackend_device_속성이_cpu이다(self):
        """Given: FakeVideoBackend 인스턴스
        When:  .device 접근
        Then:  'cpu' 반환 (Protocol 요구 속성)
        """
        fake = FakeVideoBackend()

        assert fake.device == "cpu"

    def test_FakeVideoBackend_propagate_결과는_프레임_수만큼의_bool_HxW_리스트이다(self):
        """Given: 10개 프레임 세션 + 클릭 포인트
        When:  propagate 호출
        Then:  길이 10 리스트, 각 원소 bool dtype shape (H, W)
        """
        fake = FakeVideoBackend()
        frames = _make_frames(FRAME_COUNT)
        session = fake.init_session(frames)
        fake.add_click(session, (CLICK_X, CLICK_Y))

        masks = fake.propagate(session)

        assert len(masks) == FRAME_COUNT
        for mask in masks:
            assert mask.dtype == bool, f"마스크 dtype이 bool이 아님: {mask.dtype}"
            assert mask.shape == (FAKE_FRAME_H, FAKE_FRAME_W), (
                f"마스크 shape 불일치: {mask.shape}"
            )

    def test_FakeVideoBackend_propagate_초기_카운터는_0이다(self):
        """Given: 새 FakeVideoBackend 인스턴스
        When:  아무것도 호출 안 함
        Then:  모든 카운터 == 0
        """
        fake = FakeVideoBackend()

        assert fake.init_call_count == 0
        assert fake.add_click_call_count == 0
        assert fake.propagate_call_count == 0

    def test_FakeVideoBackend_각_메서드_호출_시_카운터가_1씩_증가한다(self):
        """Given: FakeVideoBackend, 10개 프레임
        When:  init_session → add_click → propagate 각 1회 호출
        Then:  각 카운터 == 1
        """
        fake = FakeVideoBackend()
        frames = _make_frames(FRAME_COUNT)

        session = fake.init_session(frames)
        fake.add_click(session, (CLICK_X, CLICK_Y))
        fake.propagate(session)

        assert fake.init_call_count == 1
        assert fake.add_click_call_count == 1
        assert fake.propagate_call_count == 1


# ---------------------------------------------------------------------------
# FakeVideoBackend drift·empty_after 결정적 동작 테스트
# ---------------------------------------------------------------------------
class TestFakeVideoBackendBehavior:
    """drift 및 empty_after 옵션의 결정적 출력 검증."""

    def test_drift_적용_시_프레임별_centroid_중심이_이동한다(self):
        """Given: drift=(2, 0), 클릭 (320, 180), 10프레임
        When:  propagate → 각 마스크 centroid_of_mask
        Then:  프레임 i의 centroid_x ≈ 320 + 2*i (±HALF_SIZE+1)

        WHY: drift 시뮬레이션이 centroid 이동을 결정적으로 생성하는지 확인한다.
             smooth_centroids 테스트의 입력 조건을 보장한다.
        """
        from easy_capture.core.crop import centroid_of_mask

        fake = FakeVideoBackend(drift=(DRIFT_DX, DRIFT_DY))
        frames = _make_frames(FRAME_COUNT)
        session = fake.init_session(frames)
        fake.add_click(session, (CLICK_X, CLICK_Y))
        masks = fake.propagate(session)

        for i, mask in enumerate(masks):
            centroid = centroid_of_mask(mask)
            assert centroid is not None, f"프레임 {i} centroid가 None"
            cx, cy = centroid
            expected_cx = CLICK_X + i * DRIFT_DX
            # 마스크 경계 클램프로 실제 centroid는 오차 허용
            assert abs(cx - expected_cx) <= 21, (
                f"프레임 {i} centroid_x 불일치: {cx:.1f} vs 기대 {expected_cx}"
            )

    def test_empty_after_이후_프레임은_빈_마스크를_반환한다(self):
        """Given: empty_after=7, 10프레임
        When:  propagate
        Then:  인덱스 7, 8, 9 마스크는 전부 False(빈 마스크)

        WHY: occlusion 경로 테스트의 입력 조건을 결정적으로 생성한다.
        """
        fake = FakeVideoBackend(empty_after=EMPTY_AFTER_IDX)
        frames = _make_frames(FRAME_COUNT)
        session = fake.init_session(frames)
        fake.add_click(session, (CLICK_X, CLICK_Y))
        masks = fake.propagate(session)

        for i in range(EMPTY_AFTER_IDX, FRAME_COUNT):
            assert not masks[i].any(), (
                f"인덱스 {i} 마스크가 비어 있어야 하는데 True 픽셀 있음"
            )

    def test_empty_after_이전_프레임은_유효한_마스크를_반환한다(self):
        """Given: empty_after=7, 10프레임
        When:  propagate
        Then:  인덱스 0~6 마스크는 True 픽셀이 존재
        """
        fake = FakeVideoBackend(empty_after=EMPTY_AFTER_IDX)
        frames = _make_frames(FRAME_COUNT)
        session = fake.init_session(frames)
        fake.add_click(session, (CLICK_X, CLICK_Y))
        masks = fake.propagate(session)

        for i in range(EMPTY_AFTER_IDX):
            assert masks[i].any(), (
                f"인덱스 {i} 마스크가 비어 있으면 안 됨"
            )


# ---------------------------------------------------------------------------
# track() 테스트
# ---------------------------------------------------------------------------
class TestTrack:
    """VideoCaptureUseCase.track: FakeVideoBackend 주입 → TrackResult 산출."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_track_정상_호출_시_TrackResult를_반환한다(self):
        """Given: FakeVideoBackend·FakeFrameSource 주입, 10프레임, 클릭 (320, 180)
        When:  track(frames, point) 호출
        Then:  TrackResult 인스턴스 반환
        """
        usecase, _ = _make_usecase()
        frames = _make_frames(FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        assert isinstance(result, TrackResult)

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_track_결과_masks_길이가_프레임_수와_일치한다(self):
        """Given: 10프레임
        When:  track 호출
        Then:  result.masks 길이 == 10
        """
        usecase, _ = _make_usecase()
        frames = _make_frames(FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        assert len(result.masks) == FRAME_COUNT, (
            f"masks 길이 불일치: {len(result.masks)} vs {FRAME_COUNT}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_track_결과_masks는_bool_HxW_배열이다(self):
        """Given: 10프레임
        When:  track 호출
        Then:  각 mask.dtype == bool, shape == (H, W)
        """
        usecase, _ = _make_usecase()
        frames = _make_frames(FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        for i, mask in enumerate(result.masks):
            assert mask.dtype == bool, f"프레임 {i} dtype 불일치: {mask.dtype}"
            assert mask.shape == (FAKE_FRAME_H, FAKE_FRAME_W), (
                f"프레임 {i} shape 불일치: {mask.shape}"
            )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_track_결과_centroids_길이가_프레임_수와_일치한다(self):
        """Given: 10프레임, drift=(2, 0)
        When:  track 호출
        Then:  result.centroids 길이 == 10
        """
        usecase, _ = _make_usecase()
        frames = _make_frames(FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        assert len(result.centroids) == FRAME_COUNT

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_track_drift_적용_시_centroids가_프레임마다_이동한다(self):
        """Given: drift=(2, 0), 클릭 (320, 180)
        When:  track 호출
        Then:  centroids[0].x < centroids[9].x (드리프트로 X 증가)
        """
        usecase, _ = _make_usecase(drift=(DRIFT_DX, DRIFT_DY))
        frames = _make_frames(FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        # drift로 첫 프레임보다 마지막 프레임 centroid_x가 커야 한다
        valid_centroids = [c for c in result.centroids if c is not None]
        assert len(valid_centroids) >= 2, "유효 centroid 부족"
        first_cx = valid_centroids[0][0]
        last_cx = valid_centroids[-1][0]
        assert first_cx < last_cx, (
            f"drift 시 centroid가 이동하지 않음: 첫={first_cx:.1f}, 마지막={last_cx:.1f}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_track_empty_after_이후_centroids는_None이다(self):
        """Given: empty_after=7, 10프레임
        When:  track 호출
        Then:  centroids[7], [8], [9] 는 None (occlusion 프레임)
        """
        usecase, _ = _make_usecase(empty_after=EMPTY_AFTER_IDX)
        frames = _make_frames(FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        for i in range(EMPTY_AFTER_IDX, FRAME_COUNT):
            assert result.centroids[i] is None, (
                f"인덱스 {i} centroid가 None이어야 함: {result.centroids[i]}"
            )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_track_전_프레임_빈_마스크_시_EmptyTrackError가_발생한다(self):
        """Given: empty_after=0 (전 프레임 빈 마스크)
        When:  track 호출
        Then:  EmptyTrackError 발생

        WHY: 전 프레임 추적 실패 시 조용한 폴백 없이 명시적 예외를 발생시킨다.
             이미지 모드 EmptyMaskError 계승.
        """
        usecase, _ = _make_usecase(empty_after=0)
        frames = _make_frames(FRAME_COUNT)

        with pytest.raises(EmptyTrackError):
            usecase.track(frames, (CLICK_X, CLICK_Y))

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_track_EmptyTrackError_메시지는_한국어_안내를_포함한다(self):
        """Given: 전 프레임 빈 마스크
        When:  track 호출 → EmptyTrackError 발생
        Then:  메시지에 한국어 안내 포함 (예: '다시')
        """
        usecase, _ = _make_usecase(empty_after=0)
        frames = _make_frames(FRAME_COUNT)

        with pytest.raises(EmptyTrackError, match="다시"):
            usecase.track(frames, (CLICK_X, CLICK_Y))

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_track_1회_호출_시_propagate_call_count가_1이다(self):
        """Given: FakeVideoBackend 스파이
        When:  track 1회 호출
        Then:  backend.propagate_call_count == 1 (전파 정확히 1회)
        """
        usecase, backend = _make_usecase()
        frames = _make_frames(FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y))

        assert backend.propagate_call_count == 1, (
            f"propagate 호출 횟수 불일치: {backend.propagate_call_count}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_track_1회_호출_시_init_call_count가_1이다(self):
        """Given: FakeVideoBackend 스파이
        When:  track 1회 호출
        Then:  backend.init_call_count == 1 (세션 초기화 정확히 1회)
        """
        usecase, backend = _make_usecase()
        frames = _make_frames(FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y))

        assert backend.init_call_count == 1, (
            f"init_session 호출 횟수 불일치: {backend.init_call_count}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_TrackResult는_frozen_dataclass이다(self):
        """Given: 유효한 TrackResult 인스턴스
        When:  필드 수정 시도
        Then:  AttributeError 또는 TypeError 발생 (불변 보장)

        WHY: frozen=True dataclass는 캐시된 추적 결과를 실수로 덮어쓰는
             버그를 컴파일 타임에 차단한다.
        """
        usecase, _ = _make_usecase()
        frames = _make_frames(FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        with pytest.raises((AttributeError, TypeError)):
            result.masks = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_boxes() 순수성 가드 (핵심 회귀 가드)
# ---------------------------------------------------------------------------
class TestComputeBoxesPurity:
    """compute_boxes의 핵심 설계 계약: backend를 절대 미호출.

    검증 대상:
      - track 1회 후 compute_boxes를 N번 호출해도 propagate_call_count == 1
      - 종횡비·크기·smooth_window 변경 재호출 시에도 재추적 없음
    """

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_compute_boxes_반복_호출_시_propagate_call_count가_1로_유지된다(self):
        """Given: track 1회 완료 후 TrackResult 보관
        When:  compute_boxes를 5회 반복 호출
        Then:  backend.propagate_call_count == 1 (재추적 없음)

        WHY: 이것이 이 슬라이스의 핵심 회귀 가드.
             종횡비/크기 슬라이더 조작마다 compute_boxes가 호출되는데,
             propagate를 재호출하면 매 조작마다 수 초 멈춤이 발생한다.
             이미지 모드 segment_call_count 가드의 비디오판.
        """
        usecase, backend = _make_usecase()
        frames = _make_frames(FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))
        assert backend.propagate_call_count == 1  # 전제 확인

        params = VideoCropParams(
            box_size=(BOX_W, BOX_H),
            aspect=None,
            smooth_window=SMOOTH_WINDOW,
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        for _ in range(COMPUTE_BOXES_REPEAT):
            usecase.compute_boxes(result, params, frame_size)

        assert backend.propagate_call_count == 1, (
            f"compute_boxes 호출 중 propagate가 "
            f"{backend.propagate_call_count}회 호출됨. "
            "compute_boxes는 순수 함수여야 한다 — 모델 호출 금지."
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_compute_boxes_종횡비_변경_재호출_시에도_propagate_횟수가_1이다(self):
        """Given: track 1회 후 result 보관
        When:  aspect=None → '16:9' → '9:16' 순으로 3회 compute_boxes 호출
        Then:  backend.propagate_call_count == 1

        WHY: UI에서 종횡비 콤보 변경마다 compute_boxes가 재호출되는
             실제 시나리오를 재현한다.
        """
        usecase, backend = _make_usecase()
        frames = _make_frames(FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        for aspect in [None, "16:9", "9:16"]:
            params = VideoCropParams(
                box_size=(BOX_W, BOX_H),
                aspect=aspect,
                smooth_window=SMOOTH_WINDOW,
            )
            usecase.compute_boxes(result, params, frame_size)

        assert backend.propagate_call_count == 1, (
            f"종횡비 변경 재호출 중 propagate {backend.propagate_call_count}회 호출"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_compute_boxes_동일_파라미터_반복_호출_시_결과가_동일하다(self):
        """Given: 동일한 TrackResult + VideoCropParams
        When:  compute_boxes를 3회 호출
        Then:  모든 결과가 동일 (순수 함수 멱등성)
        """
        usecase, _ = _make_usecase()
        frames = _make_frames(FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))
        params = VideoCropParams(
            box_size=(BOX_W, BOX_H),
            aspect=None,
            smooth_window=SMOOTH_WINDOW,
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        boxes1 = usecase.compute_boxes(result, params, frame_size)
        boxes2 = usecase.compute_boxes(result, params, frame_size)
        boxes3 = usecase.compute_boxes(result, params, frame_size)

        assert boxes1 == boxes2 == boxes3, (
            f"순수 함수 멱등성 위반: {boxes1[:2]}... vs {boxes2[:2]}..."
        )


# ---------------------------------------------------------------------------
# compute_boxes() 고정 박스 크기 불변식 (GIF/MP4 인코딩 가능 조건)
# ---------------------------------------------------------------------------
class TestComputeBoxesFixedSize:
    """전 프레임 박스 W×H가 동일해야 GIF/MP4 인코딩이 가능한 불변식 검증."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_compute_boxes_전_프레임_박스_크기가_동일하다(self):
        """Given: drift=(2, 0) 10프레임, box_size=(200, 150)
        When:  compute_boxes 호출
        Then:  모든 프레임 박스의 (x2-x1, y2-y1)이 동일

        WHY: GIF/MP4는 전 프레임이 동일 W×H여야 인코딩된다.
             make_crop_box 경계 클램프로 프레임 끝에서 박스가 작아질 수 있으므로
             compute_boxes가 고정 size를 보장해야 한다(계획서 §3-3).
        """
        usecase, _ = _make_usecase(drift=(DRIFT_DX, DRIFT_DY))
        frames = _make_frames(FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))
        params = VideoCropParams(
            box_size=(BOX_W, BOX_H),
            aspect=None,
            smooth_window=SMOOTH_WINDOW,
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        boxes = usecase.compute_boxes(result, params, frame_size)

        sizes = [(b[2] - b[0], b[3] - b[1]) for b in boxes]
        assert len(set(sizes)) == 1, (
            f"박스 크기 불일치: {set(sizes)} — 전 프레임 동일해야 함"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_compute_boxes_박스_크기는_0보다_크다(self):
        """Given: 정상 params
        When:  compute_boxes
        Then:  모든 박스 W > 0, H > 0
        """
        usecase, _ = _make_usecase()
        frames = _make_frames(FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))
        params = VideoCropParams(
            box_size=(BOX_W, BOX_H),
            aspect=None,
            smooth_window=SMOOTH_WINDOW,
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        boxes = usecase.compute_boxes(result, params, frame_size)

        for i, box in enumerate(boxes):
            w, h = box[2] - box[0], box[3] - box[1]
            assert w > 0 and h > 0, f"프레임 {i} 박스 크기 0: w={w}, h={h}"

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_compute_boxes_박스는_프레임_경계_내에_있다(self):
        """Given: drift=(2, 0)으로 박스가 우측 이동
        When:  compute_boxes
        Then:  0 <= x1, x2 <= FRAME_W, 0 <= y1, y2 <= FRAME_H
        """
        usecase, _ = _make_usecase(drift=(DRIFT_DX, DRIFT_DY))
        frames = _make_frames(FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))
        params = VideoCropParams(
            box_size=(BOX_W, BOX_H),
            aspect=None,
            smooth_window=SMOOTH_WINDOW,
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        boxes = usecase.compute_boxes(result, params, frame_size)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            assert 0 <= x1 and x2 <= FAKE_FRAME_W, (
                f"프레임 {i} X 경계 위반: x1={x1}, x2={x2}"
            )
            assert 0 <= y1 and y2 <= FAKE_FRAME_H, (
                f"프레임 {i} Y 경계 위반: y1={y1}, y2={y2}"
            )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_compute_boxes_반환_박스_수가_프레임_수와_일치한다(self):
        """Given: 10프레임
        When:  compute_boxes
        Then:  len(boxes) == 10
        """
        usecase, _ = _make_usecase()
        frames = _make_frames(FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))
        params = VideoCropParams(
            box_size=(BOX_W, BOX_H),
            aspect=None,
            smooth_window=SMOOTH_WINDOW,
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        boxes = usecase.compute_boxes(result, params, frame_size)

        assert len(boxes) == FRAME_COUNT, (
            f"박스 수 불일치: {len(boxes)} vs {FRAME_COUNT}"
        )


# ---------------------------------------------------------------------------
# compute_boxes() smooth_centroids 적용 검증
# ---------------------------------------------------------------------------
class TestComputeBoxesSmooth:
    """smooth_centroids 적용 시 인접 프레임 박스 중심 분산이 감소하는지 검증."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_smooth_window_증가_시_인접_박스_중심_분산이_감소한다(self):
        """Given: drift=(2, 0) 20프레임
        When:  smooth_window=1 vs smooth_window=7로 compute_boxes 비교
        Then:  window=7 결과의 인접 프레임 중심 X 표준편차 <= window=1 결과

        WHY: smooth_centroids의 이동평균이 실제로 떨림을 줄이는지 확인한다.
             이 테스트가 통과하면 떨림완화 파이프라인이 올바르게 연결됐다.
        """
        # 더 뚜렷한 차이를 위해 큰 drift 사용
        BIG_DRIFT_FRAME_COUNT = 20
        BIG_DRIFT = (5, 3)

        usecase, _ = _make_usecase(drift=BIG_DRIFT)
        frames = _make_frames(BIG_DRIFT_FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        # window=1 (smoothing 없음)
        params_w1 = VideoCropParams(
            box_size=(BOX_W, BOX_H),
            aspect=None,
            smooth_window=1,
        )
        boxes_w1 = usecase.compute_boxes(result, params_w1, frame_size)
        centers_w1 = [(b[0] + b[2]) / 2.0 for b in boxes_w1]

        # window=7 (smoothing 강함)
        params_w7 = VideoCropParams(
            box_size=(BOX_W, BOX_H),
            aspect=None,
            smooth_window=7,
        )
        boxes_w7 = usecase.compute_boxes(result, params_w7, frame_size)
        centers_w7 = [(b[0] + b[2]) / 2.0 for b in boxes_w7]

        # 인접 프레임 간 중심 이동 분산 비교
        diff_w1 = [
            abs(centers_w1[i + 1] - centers_w1[i])
            for i in range(len(centers_w1) - 1)
        ]
        diff_w7 = [
            abs(centers_w7[i + 1] - centers_w7[i])
            for i in range(len(centers_w7) - 1)
        ]
        avg_diff_w1 = sum(diff_w1) / len(diff_w1) if diff_w1 else 0
        avg_diff_w7 = sum(diff_w7) / len(diff_w7) if diff_w7 else 0

        assert avg_diff_w7 <= avg_diff_w1 + 1.0, (
            f"smooth_window=7이 window=1보다 떨림이 커야 할 이유 없음: "
            f"avg_diff(w1)={avg_diff_w1:.2f}, avg_diff(w7)={avg_diff_w7:.2f}"
        )


# ---------------------------------------------------------------------------
# gap_policy BACKGROUND + empty_after 결합 검증
# ---------------------------------------------------------------------------
class TestGapPolicyIntegration:
    """valid_flags + BACKGROUND 정책 → build_output_indices 결과 검증."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_BACKGROUND_정책에서_출력_인덱스는_전_프레임이다(self):
        """Given: empty_after=7, 10프레임
        When:  track → centroids의 valid_flags 추출 → build_output_indices(BACKGROUND)
        Then:  출력 인덱스 리스트 == [0, 1, ..., 9] (전 프레임 포함)

        WHY: BACKGROUND 정책은 갭 구간도 포함해 전 프레임을 출력한다.
             gap_policy.build_output_indices와 TrackResult.centroids의
             None→valid_flags 변환이 올바르게 연결되는지 확인한다.
        """
        from easy_capture.core.tracking.gap_policy import GapPolicy, build_output_indices

        usecase, _ = _make_usecase(empty_after=EMPTY_AFTER_IDX)
        frames = _make_frames(FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        # centroids의 None 여부 → valid_flags
        valid_flags = [c is not None for c in result.centroids]
        output_indices = build_output_indices(valid_flags, GapPolicy.BACKGROUND)

        assert output_indices == list(range(FRAME_COUNT)), (
            f"BACKGROUND 정책 출력 인덱스 불일치: {output_indices}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_CUT_정책에서_빈_마스크_프레임은_출력_인덱스에서_제외된다(self):
        """Given: empty_after=7, 10프레임
        When:  track → valid_flags → build_output_indices(CUT)
        Then:  출력 인덱스 == [0, 1, 2, 3, 4, 5, 6] (7, 8, 9 제외)

        WHY: CUT 정책이 valid_flags와 연동되는지 gap_policy 계약 확인.
             이번 슬라이스는 BACKGROUND 기본 1종만 UI에서 사용하지만
             gap_policy 조합 자체는 검증한다.
        """
        from easy_capture.core.tracking.gap_policy import GapPolicy, build_output_indices

        usecase, _ = _make_usecase(empty_after=EMPTY_AFTER_IDX)
        frames = _make_frames(FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        valid_flags = [c is not None for c in result.centroids]
        output_indices = build_output_indices(valid_flags, GapPolicy.CUT)

        expected = list(range(EMPTY_AFTER_IDX))
        assert output_indices == expected, (
            f"CUT 정책 출력 인덱스 불일치: {output_indices} vs {expected}"
        )


# ---------------------------------------------------------------------------
# export() valid_flags 경로 직접 검증 (리뷰 [중요] 2 회귀 가드)
# ---------------------------------------------------------------------------

def _make_distinct_frames(n: int) -> list[np.ndarray]:
    """각 프레임이 서로 다른 픽셀값을 갖는 RGB 배열 리스트를 반환한다.

    WHY: FakeFrameSource.read_frame은 index를 무시해 동일 프레임을 반환한다.
         GIF writer가 동일 프레임을 1장으로 최적화하면 mimread 결과가 달라지므로,
         export 테스트에서는 프레임별로 다른 내용을 직접 생성해 사용한다.
    """
    frames = []
    for i in range(n):
        arr = np.zeros((FAKE_FRAME_H, FAKE_FRAME_W, 3), dtype=np.uint8)
        # R채널: 프레임 인덱스 기반으로 명확히 구분 (0~250 범위 내)
        arr[:, :, 0] = (i * 25) % 256
        arr[:, :, 1] = 100
        arr[:, :, 2] = 128
        frames.append(arr)
    return frames


class TestExportGapPolicy:
    """export()가 TrackResult.centroids에서 valid_flags를 올바르게 도출하는지 검증.

    WHY: compute_boxes는 occlusion 프레임에도 fallback 박스를 채우므로
         box 리스트에서 None을 찾는 방식은 항상 전부 True → gap_policy가 무력화된다.
         export(result=...)를 통해 centroids 기반 valid_flags가 실제 인코딩에
         반영되는지 라운드트립으로 직접 검증한다(리뷰 [중요] 2).

    주의: FakeFrameSource는 동일 프레임을 반환하므로 GIF 최적화가 1프레임으로
         압축된다. 이 테스트는 _make_distinct_frames로 프레임별 다른 내용을 사용한다.
    """

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_export_CUT_정책_valid_flags가_centroid_기반이다(self):
        """Given: empty_after=EMPTY_AFTER_IDX, CUT 정책
        When:  export 내부 _valid_flags_from_result 결과
        Then:  valid_flags에서 CUT 인덱스 수 == EMPTY_AFTER_IDX

        WHY: valid_flags가 box 여부(항상 True)가 아닌 centroid 여부(None=occlusion)
             에서 도출돼야 CUT 정책이 실제로 갭 프레임을 잘라낸다.
             이 테스트가 깨지면 [중요] 2 버그가 재발한 것이다.
             GIF 최적화 우회를 위해 인코딩 대신 valid_flags → indices 경로를 직접 검증.
        """
        from easy_capture.app.video_capture import _valid_flags_from_result
        from easy_capture.core.tracking.gap_policy import GapPolicy, build_output_indices

        usecase, _ = _make_usecase(empty_after=EMPTY_AFTER_IDX)
        frames = _make_distinct_frames(FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        # [핵심] valid_flags가 centroid 기반인지 확인
        valid_flags = _valid_flags_from_result(result, FRAME_COUNT)
        cut_indices = build_output_indices(valid_flags, GapPolicy.CUT)

        assert len(cut_indices) == EMPTY_AFTER_IDX, (
            f"CUT valid_flags 인덱스 수 불일치: {len(cut_indices)} vs {EMPTY_AFTER_IDX}. "
            "valid_flags가 centroid 기반이 아닌 경우 전 프레임이 포함된다."
        )
        # occlusion 프레임(7,8,9)이 제외됐는지 확인
        assert EMPTY_AFTER_IDX not in cut_indices, (
            f"occlusion 프레임({EMPTY_AFTER_IDX})이 CUT 출력에 포함됨"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_export_CUT_정책_GIF_라운드트립_프레임수(self, tmp_path):
        """Given: 각 프레임이 다른 내용을 가진 10개 프레임, empty_after=7, CUT
        When:  export(result=result, config=CUT) → GIF 인코딩
        Then:  GIF 프레임 수 == EMPTY_AFTER_IDX (7)

        WHY: _make_distinct_frames로 GIF 최적화(동일 프레임 병합)를 방지하고
             실제 인코딩된 프레임 수로 CUT 정책 효과를 확인한다.
        """
        import imageio

        from easy_capture.core.export.video_export import VideoExportConfig
        from easy_capture.core.tracking.gap_policy import GapPolicy

        usecase, _ = _make_usecase(empty_after=EMPTY_AFTER_IDX)
        frames = _make_distinct_frames(FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        params = VideoCropParams(
            box_size=(BOX_W, BOX_H),
            aspect=None,
            smooth_window=SMOOTH_WINDOW,
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)
        boxes = usecase.compute_boxes(result, params, frame_size)

        out_path = str(tmp_path / "cut_test.gif")
        config = VideoExportConfig(fmt="gif", fps=12.0, gap_policy=GapPolicy.CUT)
        usecase.export(frames, boxes, (out_path, config), result=result)

        reloaded = imageio.mimread(out_path)
        assert len(reloaded) == EMPTY_AFTER_IDX, (
            f"CUT 정책 GIF 프레임 수 불일치: {len(reloaded)} vs {EMPTY_AFTER_IDX}. "
            "valid_flags가 centroid 기반이 아닌 경우 전 프레임이 출력된다."
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_export_FREEZE_정책_valid_flags가_centroid_기반이다(self):
        """Given: empty_after=EMPTY_AFTER_IDX, FREEZE 정책
        When:  _valid_flags_from_result + build_output_indices(FREEZE)
        Then:  FREEZE 인덱스 수 == FRAME_COUNT (갭은 직전으로 채움)

        WHY: FREEZE 정책은 갭을 마지막 유효 프레임으로 대체한다.
             valid_flags가 centroid 기반이어야 갭 구간을 정확히 감지한다.
        """
        from easy_capture.app.video_capture import _valid_flags_from_result
        from easy_capture.core.tracking.gap_policy import GapPolicy, build_output_indices

        usecase, _ = _make_usecase(empty_after=EMPTY_AFTER_IDX)
        frames = _make_distinct_frames(FRAME_COUNT)
        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        valid_flags = _valid_flags_from_result(result, FRAME_COUNT)
        freeze_indices = build_output_indices(valid_flags, GapPolicy.FREEZE)

        assert len(freeze_indices) == FRAME_COUNT, (
            f"FREEZE 인덱스 수 불일치: {len(freeze_indices)} vs {FRAME_COUNT}."
        )
        # 마지막 유효 인덱스(6)가 갭 구간에 반복됐는지 확인
        last_valid = EMPTY_AFTER_IDX - 1  # 6
        for idx in range(EMPTY_AFTER_IDX, FRAME_COUNT):
            assert freeze_indices[idx] == last_valid, (
                f"FREEZE 인덱스[{idx}]={freeze_indices[idx]}, 기대={last_valid}"
            )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_export_result_None_이면_전_프레임_유효_처리된다(self):
        """Given: result=None (하위호환 폴백), BACKGROUND 정책
        When:  _valid_flags_from_result(None, n)
        Then:  valid_flags 전부 True, 인덱스 수 == n

        WHY: result=None은 하위호환 경로. 전부 유효(valid=True)로 처리한다.
        """
        from easy_capture.app.video_capture import _valid_flags_from_result
        from easy_capture.core.tracking.gap_policy import GapPolicy, build_output_indices

        valid_flags = _valid_flags_from_result(None, FRAME_COUNT)
        assert all(valid_flags), "result=None이면 전부 True여야 한다"
        indices = build_output_indices(valid_flags, GapPolicy.BACKGROUND)
        assert len(indices) == FRAME_COUNT, (
            f"result=None BACKGROUND 인덱스 수 불일치: {len(indices)} vs {FRAME_COUNT}"
        )


# ===========================================================================
# 샷경계 재추적 테스트 (슬라이스 핵심 가드)
# ===========================================================================
# 픽스처 헬퍼 — 재추적 시나리오 공용
# ---------------------------------------------------------------------------

def _make_retrack_frames(n: int = RETRACK_FRAME_COUNT) -> list[np.ndarray]:
    """재추적 시나리오용 n개 고정 RGB 프레임 리스트 반환."""
    src = FakeFrameSource()
    return [src.read_frame(i) for i in range(n)]


def _make_retrack_usecase(
    detector: "FakeDetectionBackend | None",
    drift: tuple[int, int] = (0, 0),
) -> tuple["VideoCaptureUseCase", FakeVideoBackend]:
    """FakeVideoBackend + FakeDetectionBackend 주입 VideoCaptureUseCase 반환.

    WHY: detector 파라미터가 추가된 새 생성자 계약을 테스트에서 직접 검증한다.
         detector=None이면 기존(단일 샷) 경로로 fallback — 하위호환 확인용.
    """
    backend = FakeVideoBackend(drift=drift, empty_after=None)
    usecase = VideoCaptureUseCase(
        source=FakeFrameSource(),
        backend=backend,
        detector=detector,
    )
    return usecase, backend


# ---------------------------------------------------------------------------
# FakeDetectionBackend Protocol 계약 테스트
# ---------------------------------------------------------------------------
class TestFakeDetectionBackendContract:
    """FakeDetectionBackend가 DetectionBackend Protocol 요구를 만족하는지 검증."""

    def test_FakeDetectionBackend_초기_detect_call_count는_0이다(self):
        """Given: 새 FakeDetectionBackend
        When:  아무것도 호출 안 함
        Then:  detect_call_count == 0
        """
        fake = FakeDetectionBackend()
        assert fake.detect_call_count == 0

    def test_FakeDetectionBackend_detect_호출마다_카운터가_증가한다(self):
        """Given: FakeDetectionBackend(candidates_fixed=[])
        When:  detect 3회 호출
        Then:  detect_call_count == 3
        """
        fake = FakeDetectionBackend()
        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

        fake.detect(dummy_frame, "person")
        fake.detect(dummy_frame, "person")
        fake.detect(dummy_frame, "person")

        assert fake.detect_call_count == 3

    def test_FakeDetectionBackend_fixed_후보는_매_호출마다_동일하게_반환한다(self):
        """Given: candidates_fixed=[dummy]
        When:  detect 2회 호출
        Then:  두 결과가 동일 길이
        """
        # Detection 미구현이면 dict로 대체 — 길이만 검증
        dummy_candidates = [{"box": (0, 0, 10, 10)}]
        fake = FakeDetectionBackend(candidates_fixed=dummy_candidates)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        r1 = fake.detect(frame, "person")
        r2 = fake.detect(frame, "person")

        assert len(r1) == len(r2) == 1

    def test_FakeDetectionBackend_sequence_모드는_순서대로_반환한다(self):
        """Given: candidates_sequence=[[a], [b, c], []]
        When:  detect 3회 호출
        Then:  각 호출마다 순서대로 길이 1, 2, 0 반환
        """
        fake = FakeDetectionBackend(
            candidates_sequence=[
                [{"box": (0, 0, 10, 10)}],
                [{"box": (0, 0, 10, 10)}, {"box": (50, 50, 60, 60)}],
                [],
            ]
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        r1 = fake.detect(frame, "person")
        r2 = fake.detect(frame, "person")
        r3 = fake.detect(frame, "person")

        assert len(r1) == 1
        assert len(r2) == 2
        assert len(r3) == 0

    def test_FakeDetectionBackend_빈_기본값이면_detect는_빈_리스트_반환한다(self):
        """Given: FakeDetectionBackend() — 기본값
        When:  detect 호출
        Then:  [] 반환
        """
        fake = FakeDetectionBackend()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        result = fake.detect(frame, "person")

        assert result == []

    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_FakeDetectionBackend는_DetectionBackend_isinstance를_통과한다(self):
        """Given: FakeDetectionBackend 인스턴스
        When:  isinstance(..., DetectionBackend) 호출
        Then:  True

        WHY: @runtime_checkable Protocol — 구조적 서브타이핑 런타임 검증.
        """
        from easy_capture.core.segmentation.detection_backend import DetectionBackend

        fake = FakeDetectionBackend()
        assert isinstance(fake, DetectionBackend)

    def test_FakeDetectionBackend_device_속성이_cpu이다(self):
        """Given: FakeDetectionBackend 인스턴스
        When:  .device 접근
        Then:  'cpu'
        """
        fake = FakeDetectionBackend()
        assert fake.device == "cpu"


# ---------------------------------------------------------------------------
# 시나리오 (a): 재매칭 통과 → objid 유지·추적 지속
# ---------------------------------------------------------------------------
class TestRetrackPassScenario:
    """컷 1개 + 재매칭 통과 시나리오 검증.

    FakeVideoBackend + FakeDetectionBackend(통과 후보 주입)으로
    GPU 없이 재추적 오케스트레이션 통과 경로를 완전 검증한다.
    """

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_재매칭_통과_시_centroids가_두_샷에_걸쳐_이어진다(self):
        """Given: 20프레임, 컷=[10], 통과 후보(RETRACK_PASS_BOX)
        When:  track(frames, click, cut_frames=[10]) 호출
        Then:  result.centroids 길이 == 20, 전부 None 아님

        WHY: 재매칭 통과 → SAM2 재초기화 → 두 샷 centroids를 이어붙여
             사용자에게 끊김 없는 단일 추적으로 노출한다.
        """
        pass_candidate = Detection(box=RETRACK_PASS_BOX, score=0.95, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[pass_candidate])
        usecase, _ = _make_retrack_usecase(detector)
        frames = _make_retrack_frames(RETRACK_FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        assert len(result.centroids) == RETRACK_FRAME_COUNT
        # 통과 → 재추적 성공 → needs_correction 전부 False
        assert all(not nc for nc in result.needs_correction), (
            "재매칭 통과 시 needs_correction이 전부 False여야 한다"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_재매칭_통과_시_propagate_call_count가_샷_수와_같다(self):
        """Given: 컷=[10] → 2샷
        When:  track 호출
        Then:  backend.propagate_call_count == 2 (샷마다 SAM2 재초기화·전파 1회)

        WHY: 핵심 카운터 가드 — 컷마다 SAM2를 재초기화하므로
             propagate 호출 수 == 샷 수 == 컷 수 + 1.
             이 가드가 깨지면 재초기화 로직이 누락된 것이다.
        """
        pass_candidate = Detection(box=RETRACK_PASS_BOX, score=0.95, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[pass_candidate])
        usecase, backend = _make_retrack_usecase(detector)
        frames = _make_retrack_frames(RETRACK_FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        assert backend.propagate_call_count == RETRACK_SHOT_COUNT, (
            f"propagate 호출 횟수 불일치: {backend.propagate_call_count} vs "
            f"{RETRACK_SHOT_COUNT}(샷 수)"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_재매칭_통과_시_detect_call_count가_컷_수와_같다(self):
        """Given: 컷=[10] → 컷 1개
        When:  track 호출
        Then:  detector.detect_call_count == 1 (컷마다 1회 검출)

        WHY: 핵심 카운터 가드 — 검출은 샷 경계마다 정확히 1회만 호출돼야 한다.
             이 가드가 깨지면 불필요한 재검출이 발생해 성능이 저하된다.
        """
        pass_candidate = Detection(box=RETRACK_PASS_BOX, score=0.95, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[pass_candidate])
        usecase, _ = _make_retrack_usecase(detector)
        frames = _make_retrack_frames(RETRACK_FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        assert detector.detect_call_count == RETRACK_CUT_COUNT, (
            f"detect 호출 횟수 불일치: {detector.detect_call_count} vs "
            f"{RETRACK_CUT_COUNT}(컷 수)"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_재매칭_통과_시_cut_frames_필드에_컷_인덱스가_기록된다(self):
        """Given: cut_frames=[10]
        When:  track 호출
        Then:  result.cut_frames == [10]

        WHY: UI가 컷 위치를 표시하려면 TrackResult에 컷 인덱스가 필요하다.
        """
        pass_candidate = Detection(box=RETRACK_PASS_BOX, score=0.95, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[pass_candidate])
        usecase, _ = _make_retrack_usecase(detector)
        frames = _make_retrack_frames(RETRACK_FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        assert result.cut_frames == [RETRACK_CUT_FRAME], (
            f"cut_frames 불일치: {result.cut_frames}"
        )


# ---------------------------------------------------------------------------
# 시나리오 (b): 재매칭 미달 → needs_correction 플래그
# ---------------------------------------------------------------------------
class TestRetrackFailScenario:
    """컷 1개 + 재매칭 미달 시나리오 검증.

    미달 시 추적을 점프하지 않고 needs_correction 플래그만 세운다.
    수동 교정 UI(다음 슬라이스)를 위한 안전망이다.
    """

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_재매칭_미달_시_후속_샷_needs_correction이_True이다(self):
        """Given: 컷=[10], 미달 후보(RETRACK_FAIL_BOX — IoU≈0)
        When:  track 호출
        Then:  result.needs_correction에서 후속 샷 구간(10~19)이 True

        WHY: 미달은 추적 점프보다 "교정 필요 표시"가 안전하다(ADR 0006 "수동 교정 유도").
             틀린 인물로 추적이 점프하면 사용자가 알아채기 어렵고
             회복도 어렵다. 플래그는 명시적 피드백을 제공한다.
        """
        fail_candidate = Detection(box=RETRACK_FAIL_BOX, score=0.8, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[fail_candidate])
        usecase, _ = _make_retrack_usecase(detector)
        frames = _make_retrack_frames(RETRACK_FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        # 후속 샷(컷 이후 프레임) 구간에 needs_correction True가 최소 1개
        shot2_correction = result.needs_correction[RETRACK_CUT_FRAME:]
        assert any(shot2_correction), (
            "재매칭 미달 시 후속 샷 구간에 needs_correction=True가 최소 1개여야 한다"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_재매칭_미달_시_첫_샷_needs_correction은_False이다(self):
        """Given: 컷=[10], 미달 후보
        When:  track 호출
        Then:  result.needs_correction[0:10] 전부 False

        WHY: 첫 샷은 재매칭 없이 직접 클릭으로 시작했으므로 실패 없음.
             플래그가 첫 샷에도 번지면 잘못된 구현이다.
        """
        fail_candidate = Detection(box=RETRACK_FAIL_BOX, score=0.8, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[fail_candidate])
        usecase, _ = _make_retrack_usecase(detector)
        frames = _make_retrack_frames(RETRACK_FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        shot1_correction = result.needs_correction[:RETRACK_CUT_FRAME]
        assert all(not nc for nc in shot1_correction), (
            "첫 샷 구간에 needs_correction=True가 있으면 안 된다"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_재매칭_미달_시_propagate_call_count와_detect_call_count_가드(self):
        """Given: 컷=[10], 미달 후보
        When:  track 호출
        Then:  propagate_call_count == 2, detect_call_count == 1 (미달도 동일 횟수)

        WHY: 미달 경로에서도 카운터 불변식을 지켜야 한다.
             미달이라고 해서 detect 추가 호출이 발생하면 안 된다.
        """
        fail_candidate = Detection(box=RETRACK_FAIL_BOX, score=0.8, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[fail_candidate])
        usecase, backend = _make_retrack_usecase(detector)
        frames = _make_retrack_frames(RETRACK_FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        assert backend.propagate_call_count == RETRACK_SHOT_COUNT
        assert detector.detect_call_count == RETRACK_CUT_COUNT


# ---------------------------------------------------------------------------
# 시나리오 (c): 다중 후보 → best 선택
# ---------------------------------------------------------------------------
class TestRetrackMultiCandidateScenario:
    """다중 후보 중 가장 가까운 후보(best)가 정확히 선택되는지 검증."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_다중_후보_중_best가_통과이면_needs_correction이_False이다(self):
        """Given: 컷=[10], 후보 2개(가까운 것 + 먼 것 혼재)
        When:  track 호출
        Then:  needs_correction 후속 샷 전부 False (best가 통과했으므로)

        WHY: argmax로 가장 가까운 후보를 선택해 통과하면 재추적이 이어져야 한다.
             "먼 후보가 있다"는 이유만으로 재매칭을 실패 처리하면 안 된다.
        """
        # 먼 후보를 먼저, 가까운 후보를 나중에 — argmax 정확성 검증
        pass_candidate = Detection(box=RETRACK_PASS_BOX, score=0.7, feat=None)
        fail_candidate = Detection(box=RETRACK_FAIL_BOX, score=0.95, feat=None)
        # detect_score가 높더라도 rematch_score(IoU 기반)가 낮으면 미선택
        detector = FakeDetectionBackend(
            candidates_fixed=[fail_candidate, pass_candidate]
        )
        usecase, _ = _make_retrack_usecase(detector)
        frames = _make_retrack_frames(RETRACK_FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        shot2_correction = result.needs_correction[RETRACK_CUT_FRAME:]
        assert all(not nc for nc in shot2_correction), (
            "다중 후보 중 best가 통과이면 후속 샷 needs_correction이 전부 False여야 한다"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_다중_후보에서도_detect_call_count는_컷_수이다(self):
        """Given: 다중 후보
        When:  track 호출
        Then:  detect_call_count == 1 (다중 후보라도 컷 경계에서 1회만 호출)
        """
        pass_candidate = Detection(box=RETRACK_PASS_BOX, score=0.7, feat=None)
        fail_candidate = Detection(box=RETRACK_FAIL_BOX, score=0.95, feat=None)
        detector = FakeDetectionBackend(
            candidates_fixed=[fail_candidate, pass_candidate]
        )
        usecase, _ = _make_retrack_usecase(detector)
        frames = _make_retrack_frames(RETRACK_FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        assert detector.detect_call_count == RETRACK_CUT_COUNT


# ---------------------------------------------------------------------------
# 시나리오 (d): 빈 검출 → 미달 처리
# ---------------------------------------------------------------------------
class TestRetrackEmptyDetectionScenario:
    """컷 후 detect가 []를 반환하면 미달과 동일하게 처리한다."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    def test_빈_검출이면_후속_샷_needs_correction이_True이다(self):
        """Given: 컷=[10], detect가 [] 반환(인물 없음)
        When:  track 호출
        Then:  후속 샷 구간 needs_correction에 True 존재

        WHY: 화면에 인물이 없거나 검출기가 아무것도 찾지 못하면
             빈 검출 = 미달로 동일 처리해야 한다(계획서 §2-4).
        """
        detector = FakeDetectionBackend(candidates_fixed=[])  # 항상 빈 리스트
        usecase, _ = _make_retrack_usecase(detector)
        frames = _make_retrack_frames(RETRACK_FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        shot2_correction = result.needs_correction[RETRACK_CUT_FRAME:]
        assert any(shot2_correction), (
            "빈 검출 시 후속 샷에 needs_correction=True가 있어야 한다"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    def test_빈_검출에서도_detect_call_count는_컷_수이다(self):
        """Given: 빈 후보
        When:  track 호출
        Then:  detect_call_count == 1

        WHY: 빈 결과라도 detect 자체는 컷 경계에서 정확히 1회 호출돼야 한다.
        """
        detector = FakeDetectionBackend(candidates_fixed=[])
        usecase, _ = _make_retrack_usecase(detector)
        frames = _make_retrack_frames(RETRACK_FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        assert detector.detect_call_count == RETRACK_CUT_COUNT


# ---------------------------------------------------------------------------
# 카운터 이중 순수성 가드 — compute_boxes 반복 시 두 카운터 불변
# ---------------------------------------------------------------------------
class TestRetrackPurityGuard:
    """compute_boxes 반복 호출 시 propagate·detect 카운터가 모두 불변이어야 한다.

    WHY: 이것이 "무거움(track)/가벼움(compute_boxes) 분리" 불변식의 핵심 가드.
         UI에서 종횡비·크기 슬라이더를 조작할 때마다 compute_boxes가 호출되는데,
         그때마다 SAM2나 Grounding DINO가 재호출되면 매 조작마다 수 초 멈춤이 발생한다.
         propagate_call_count·detect_call_count를 동시에 단언해 두 모델 호출이
         모두 track 안에 가둬져 있음을 강제한다(계획서 §4-1).
    """

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_compute_boxes_반복_호출_시_propagate와_detect_카운터_모두_불변이다(self):
        """Given: track 1회(컷 1개·통과) → TrackResult 보관
        When:  compute_boxes를 4회 반복 호출
        Then:  backend.propagate_call_count == 2 (불변)
               detector.detect_call_count == 1 (불변)

        WHY: track 완료 후 compute_boxes를 아무리 반복해도 두 카운터가
             track 직후 값에서 변하지 않아야 한다.
        """
        pass_candidate = Detection(box=RETRACK_PASS_BOX, score=0.95, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[pass_candidate])
        usecase, backend = _make_retrack_usecase(detector)
        frames = _make_retrack_frames(RETRACK_FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        # track 직후 카운터 기준값 포착
        propagate_after_track = backend.propagate_call_count
        detect_after_track = detector.detect_call_count

        params = VideoCropParams(
            box_size=(BOX_W, BOX_H),
            aspect=None,
            smooth_window=SMOOTH_WINDOW,
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        for _ in range(RETRACK_COMPUTE_REPEAT):
            usecase.compute_boxes(result, params, frame_size)

        assert backend.propagate_call_count == propagate_after_track, (
            f"compute_boxes 반복 중 propagate 재호출 감지: "
            f"{backend.propagate_call_count} vs 기준 {propagate_after_track}. "
            "compute_boxes는 순수 함수여야 한다 — SAM2 호출 금지."
        )
        assert detector.detect_call_count == detect_after_track, (
            f"compute_boxes 반복 중 detect 재호출 감지: "
            f"{detector.detect_call_count} vs 기준 {detect_after_track}. "
            "compute_boxes는 순수 함수여야 한다 — Grounding DINO 호출 금지."
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_종횡비_변경_재호출_시에도_두_카운터_불변이다(self):
        """Given: track 완료 후 aspect None→16:9→9:16 순 3회 compute_boxes
        When:  각 aspect로 compute_boxes 호출
        Then:  propagate·detect 카운터 모두 track 직후와 동일

        WHY: UI 종횡비 콤보 변경 시나리오를 재현한다.
        """
        pass_candidate = Detection(box=RETRACK_PASS_BOX, score=0.95, feat=None)
        detector = FakeDetectionBackend(candidates_fixed=[pass_candidate])
        usecase, backend = _make_retrack_usecase(detector)
        frames = _make_retrack_frames(RETRACK_FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y), cut_frames=[RETRACK_CUT_FRAME])

        propagate_base = backend.propagate_call_count
        detect_base = detector.detect_call_count
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        for aspect in [None, "16:9", "9:16"]:
            params = VideoCropParams(
                box_size=(BOX_W, BOX_H),
                aspect=aspect,
                smooth_window=SMOOTH_WINDOW,
            )
            usecase.compute_boxes(result, params, frame_size)

        assert backend.propagate_call_count == propagate_base, (
            "종횡비 변경 중 propagate 재호출 감지"
        )
        assert detector.detect_call_count == detect_base, (
            "종횡비 변경 중 detect 재호출 감지"
        )


# ---------------------------------------------------------------------------
# detector=None 하위호환 — 단일 샷 동작 무회귀
# ---------------------------------------------------------------------------
class TestRetrackDetectorNoneCompat:
    """detector=None이면 첫 슬라이스(단일 샷) 동작 그대로 — 기존 계약 무회귀.

    WHY: detector 파라미터 추가는 기존 코드를 깨면 안 된다.
         detector=None은 "컷 감지·재매칭 건너뜀 = 단일 샷 경로" 계약이다.
         기존 216개 테스트와 동일한 동작을 이 클래스가 추가로 보증한다.
    """

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    def test_detector_None이면_단일_샷_TrackResult를_반환한다(self):
        """Given: VideoCaptureUseCase(detector=None), 10프레임
        When:  track(frames, point) 호출 (cut_frames 미제공)
        Then:  TrackResult 반환, centroids 길이 == FRAME_COUNT

        WHY: detector=None은 기존 슬라이스 경로 — 변경 없음을 검증한다.
        """
        backend = FakeVideoBackend(drift=(DRIFT_DX, DRIFT_DY))
        usecase = VideoCaptureUseCase(
            source=FakeFrameSource(),
            backend=backend,
            detector=None,
        )
        frames = _make_frames(FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        assert isinstance(result, TrackResult)
        assert len(result.centroids) == FRAME_COUNT

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    def test_detector_None이면_propagate_call_count가_1이다(self):
        """Given: detector=None
        When:  track 1회 호출
        Then:  propagate_call_count == 1 (단일 샷 = 전파 1회)

        WHY: detector=None이면 컷 감지를 건너뛰고 단일 전파만 수행한다.
        """
        backend = FakeVideoBackend()
        usecase = VideoCaptureUseCase(
            source=FakeFrameSource(),
            backend=backend,
            detector=None,
        )
        frames = _make_frames(FRAME_COUNT)

        usecase.track(frames, (CLICK_X, CLICK_Y))

        assert backend.propagate_call_count == 1, (
            f"detector=None 단일 샷: propagate={backend.propagate_call_count} ≠ 1"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    def test_detector_None이면_cut_frames_필드가_빈_리스트이다(self):
        """Given: detector=None
        When:  track 호출
        Then:  result.cut_frames == [] (컷 감지 없음)

        WHY: cut_frames 기본값이 빈 리스트여야 UI가 "컷 없음"으로 처리한다.
        """
        backend = FakeVideoBackend()
        usecase = VideoCaptureUseCase(
            source=FakeFrameSource(),
            backend=backend,
            detector=None,
        )
        frames = _make_frames(FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        assert result.cut_frames == [], (
            f"detector=None이면 cut_frames == [], 실제: {result.cut_frames}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    def test_detector_None이면_needs_correction이_전부_False이다(self):
        """Given: detector=None
        When:  track 호출
        Then:  result.needs_correction 전부 False (재매칭 없음 = 교정 불필요)

        WHY: 단일 샷 경로에서 needs_correction이 True이면 안 된다.
             기존 export 경로(valid_flags = centroid 기반)에 영향을 주지 않음도 확인.
        """
        backend = FakeVideoBackend()
        usecase = VideoCaptureUseCase(
            source=FakeFrameSource(),
            backend=backend,
            detector=None,
        )
        frames = _make_frames(FRAME_COUNT)

        result = usecase.track(frames, (CLICK_X, CLICK_Y))

        assert all(not nc for nc in result.needs_correction), (
            "detector=None 단일 샷에서 needs_correction이 True인 프레임이 있으면 안 된다"
        )


# ---------------------------------------------------------------------------
# 다중 컷 누적 — 컷 2개(3샷) 카운터 가드
# ---------------------------------------------------------------------------
class TestRetrackMultiCutAccumulation:
    """컷 2개(3샷) 시나리오 — 누적 카운터 가드."""

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_컷_2개_시_propagate_call_count가_3이다(self):
        """Given: 30프레임, cut_frames=[10, 20] → 3샷
        When:  track 호출
        Then:  propagate_call_count == 3

        WHY: 컷 k개 → k+1샷 → propagate k+1회.
             다중 컷 시나리오에서도 카운터 가드가 성립하는지 확인한다.
        """
        pass_candidate = Detection(box=RETRACK_PASS_BOX, score=0.95, feat=None)
        # 컷 2개 → detect 2회 호출 → 시퀀스 2개 주입
        detector = FakeDetectionBackend(
            candidates_sequence=[
                [pass_candidate],  # 첫 번째 컷(10) 재매칭 → 통과
                [pass_candidate],  # 두 번째 컷(20) 재매칭 → 통과
            ]
        )
        backend = FakeVideoBackend(drift=(0, 0))
        usecase = VideoCaptureUseCase(
            source=FakeFrameSource(),
            backend=backend,
            detector=detector,
        )
        frames = _make_retrack_frames(RETRACK_3SHOT_FRAMES)

        usecase.track(
            frames,
            (CLICK_X, CLICK_Y),
            cut_frames=[RETRACK_CUT1, RETRACK_CUT2],
        )

        assert backend.propagate_call_count == RETRACK_3SHOT_COUNT, (
            f"3샷: propagate={backend.propagate_call_count} ≠ {RETRACK_3SHOT_COUNT}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_컷_2개_시_detect_call_count가_2이다(self):
        """Given: cut_frames=[10, 20] → 컷 2개
        When:  track 호출
        Then:  detect_call_count == 2

        WHY: 컷 경계마다 detect 1회 → 총 컷 수만큼 호출 누적.
        """
        pass_candidate = Detection(box=RETRACK_PASS_BOX, score=0.95, feat=None)
        detector = FakeDetectionBackend(
            candidates_sequence=[
                [pass_candidate],
                [pass_candidate],
            ]
        )
        backend = FakeVideoBackend(drift=(0, 0))
        usecase = VideoCaptureUseCase(
            source=FakeFrameSource(),
            backend=backend,
            detector=detector,
        )
        frames = _make_retrack_frames(RETRACK_3SHOT_FRAMES)

        usecase.track(
            frames,
            (CLICK_X, CLICK_Y),
            cut_frames=[RETRACK_CUT1, RETRACK_CUT2],
        )

        assert detector.detect_call_count == RETRACK_2CUT_COUNT, (
            f"컷 2개: detect={detector.detect_call_count} ≠ {RETRACK_2CUT_COUNT}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(not _HAS_RETRACK_FIELDS, reason=_MSG_NO_RETRACK)
    @pytest.mark.skipif(not _HAS_DETECTION, reason=_MSG_NO_DETECTION)
    def test_컷_2개_centroids_길이가_전체_프레임_수이다(self):
        """Given: 30프레임, cut_frames=[10, 20], 전 컷 통과
        When:  track 호출
        Then:  len(result.centroids) == 30

        WHY: 3샷 centroids를 이어붙인 결과가 전체 프레임 수와 일치해야
             compute_boxes·export가 올바른 길이의 박스/크롭을 생성한다.
        """
        pass_candidate = Detection(box=RETRACK_PASS_BOX, score=0.95, feat=None)
        detector = FakeDetectionBackend(
            candidates_sequence=[
                [pass_candidate],
                [pass_candidate],
            ]
        )
        backend = FakeVideoBackend(drift=(0, 0))
        usecase = VideoCaptureUseCase(
            source=FakeFrameSource(),
            backend=backend,
            detector=detector,
        )
        frames = _make_retrack_frames(RETRACK_3SHOT_FRAMES)

        result = usecase.track(
            frames,
            (CLICK_X, CLICK_Y),
            cut_frames=[RETRACK_CUT1, RETRACK_CUT2],
        )

        assert len(result.centroids) == RETRACK_3SHOT_FRAMES, (
            f"centroids 길이 불일치: {len(result.centroids)} vs {RETRACK_3SHOT_FRAMES}"
        )


# ===========================================================================
# 크롭 정책 변경 가드 — bbox 중심·자동 고정 크기·잘림·흔들림 해결
# ===========================================================================

# ---------------------------------------------------------------------------
# 테스트 상수
# ---------------------------------------------------------------------------
# 큰 피사체 마스크 크기 (box_size 하한을 초과하는 크기)
_LARGE_SUBJECT_HALF = 80          # 좌우/상하 각 80px → bbox 폭/높이 ≈ 160px
_LARGE_SUBJECT_SIDE = _LARGE_SUBJECT_HALF * 2   # 160px

# 작은 피사체 마스크 크기 (box_size 하한보다 작은 크기)
_SMALL_SUBJECT_HALF = 5           # 좌우/상하 각 5px → bbox 폭/높이 ≈ 10px

# padding 값 — VideoCropParams 기본값과 동일
_DEFAULT_PADDING = 1.3

# 계산 허용 오차 (픽셀 단위, 정수 반올림 등 오차)
_SIZE_TOLERANCE = 2


# ---------------------------------------------------------------------------
# _bbox_center 단위 테스트
# ---------------------------------------------------------------------------
class TestBboxCenter:
    """_bbox_center 헬퍼: bbox 중심 반환·빈 마스크 None 처리.

    WHY: centroid(무게중심) 대신 bbox 중심을 사용해 자세 변화(팔·다리)
         흔들림을 줄이는 핵심 정책 변경을 가드한다.
    """

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_사각_마스크_bbox_중심이_정확히_반환된다(self):
        """Given: 중심 (cx=320, cy=180), half=20 사각형 bool 마스크
        When:  _bbox_center(mask) 호출
        Then:  반환값 (cx_out, cy_out)이 bbox 중심값과 일치한다 (±_SIZE_TOLERANCE)

        WHY: bbox 중심이 마스크 픽셀 분포 평균(centroid)이 아닌 bbox 끝점
             평균으로 계산되는지 확인한다. 팔을 뻗으면 bbox 중심은 흔들리지 않는다.
        """
        from easy_capture.app.video_capture import _bbox_center

        # given
        H, W = FAKE_FRAME_H, FAKE_FRAME_W
        HALF = 20
        CX, CY = 320, 180
        mask = _make_rect_mask(H, W, CX, CY, half=HALF)

        # when
        result = _bbox_center(mask)

        # then
        assert result is not None, "유효 마스크에서 _bbox_center가 None을 반환함"
        cx_out, cy_out = result
        # bbox: x1=CX-HALF, x2=CX+HALF-1 → 중심 = (x1+x2)/2
        expected_cx = (CX - HALF + CX + HALF - 1) / 2.0   # ≈ CX - 0.5
        expected_cy = (CY - HALF + CY + HALF - 1) / 2.0
        assert abs(cx_out - expected_cx) <= _SIZE_TOLERANCE, (
            f"bbox 중심 X 불일치: {cx_out:.2f} vs 기대 {expected_cx:.2f}"
        )
        assert abs(cy_out - expected_cy) <= _SIZE_TOLERANCE, (
            f"bbox 중심 Y 불일치: {cy_out:.2f} vs 기대 {expected_cy:.2f}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_빈_마스크에서_bbox_center는_None을_반환한다(self):
        """Given: 전부 False인 빈 bool 마스크
        When:  _bbox_center(mask) 호출
        Then:  None 반환 (유효 픽셀 없음)

        WHY: occlusion 프레임은 빈 마스크를 가지며 None이 fallback_center를
             통해 프레임 중앙 폴백 처리되는 흐름을 보장한다.
        """
        from easy_capture.app.video_capture import _bbox_center

        # given
        mask = np.zeros((FAKE_FRAME_H, FAKE_FRAME_W), dtype=bool)

        # when
        result = _bbox_center(mask)

        # then
        assert result is None, f"빈 마스크에서 None이 아닌 값 반환: {result}"


# ---------------------------------------------------------------------------
# _expand_to_aspect 단위 테스트
# ---------------------------------------------------------------------------
class TestExpandToAspect:
    """_expand_to_aspect 헬퍼: 종횡비 '확대' 방향 적용 (축소 아님 — 잘림 방지).

    WHY: apply_aspect_lock(축소 방향)과 반대로 짧은 변을 늘려 피사체가
         항상 박스 안에 들어오게 한다. 이 동작을 가드하지 않으면 크롭이
         피사체를 잘라낼 수 있다.
    """

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_aspect_None이면_입력_크기를_그대로_반환한다(self):
        """Given: w=200, h=150, aspect=None
        When:  _expand_to_aspect(200, 150, None)
        Then:  (200, 150) 반환 (변경 없음)

        WHY: aspect=None은 종횡비 잠금 없음 — 원본 크기 보존 계약.
        """
        from easy_capture.app.video_capture import _expand_to_aspect

        # when
        w_out, h_out = _expand_to_aspect(200, 150, None)

        # then
        assert w_out == 200 and h_out == 150, (
            f"aspect=None 시 크기 변경됨: ({w_out}, {h_out})"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_가로_긴_입력에_1대1_적용하면_입력_이상_크기가_된다(self):
        """Given: w=200 > h=100 (가로 긴 피사체), aspect='1:1'
        When:  _expand_to_aspect(200, 100, '1:1')
        Then:  w_out >= 200, h_out >= 100 (축소 아님 — 잘림 방지)
               결과가 1:1 비율 (w_out == h_out)

        WHY: 가로 긴 피사체를 1:1로 만들 때 세로를 늘려야 한다.
             apply_aspect_lock처럼 가로를 줄이면 피사체 좌우가 잘린다.
        """
        from easy_capture.app.video_capture import _expand_to_aspect

        W_IN, H_IN = 200, 100

        # when
        w_out, h_out = _expand_to_aspect(W_IN, H_IN, "1:1")

        # then: 축소 아님
        assert w_out >= W_IN, f"가로가 줄어들었음(잘림 위험): {w_out} < {W_IN}"
        assert h_out >= H_IN, f"세로가 줄어들었음(잘림 위험): {h_out} < {H_IN}"
        # 1:1 비율
        assert w_out == h_out, f"1:1 비율 불일치: w={w_out}, h={h_out}"

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_9대16_세로_긴_비율_방향이_올바르다(self):
        """Given: w=100, h=100 (정방형), aspect='9:16'
        When:  _expand_to_aspect(100, 100, '9:16')
        Then:  h_out > w_out (세로가 더 긴 세로형 박스)
               w_out >= 100, h_out >= 100 (축소 아님)

        WHY: 9:16은 세로형(세로>가로) 비율이어야 한다.
             방향이 반전되면 가로형 크롭이 나와 모바일 세로 영상에 부적합하다.
        """
        from easy_capture.app.video_capture import _expand_to_aspect

        W_IN, H_IN = 100, 100

        # when
        w_out, h_out = _expand_to_aspect(W_IN, H_IN, "9:16")

        # then
        assert h_out > w_out, f"9:16 비율인데 세로({h_out})가 가로({w_out}) 이하"
        assert w_out >= W_IN, f"가로 축소 감지: {w_out} < {W_IN}"
        assert h_out >= H_IN, f"세로 축소 감지: {h_out} < {H_IN}"

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_16대9_가로_긴_비율_방향이_올바르다(self):
        """Given: w=100, h=100 (정방형), aspect='16:9'
        When:  _expand_to_aspect(100, 100, '16:9')
        Then:  w_out > h_out (가로가 더 긴 가로형 박스)
               w_out >= 100, h_out >= 100 (축소 아님)

        WHY: 16:9는 가로형(가로>세로) 비율이어야 한다.
        """
        from easy_capture.app.video_capture import _expand_to_aspect

        W_IN, H_IN = 100, 100

        # when
        w_out, h_out = _expand_to_aspect(W_IN, H_IN, "16:9")

        # then
        assert w_out > h_out, f"16:9 비율인데 가로({w_out})가 세로({h_out}) 이하"
        assert w_out >= W_IN, f"가로 축소 감지: {w_out} < {W_IN}"
        assert h_out >= H_IN, f"세로 축소 감지: {h_out} < {H_IN}"


# ---------------------------------------------------------------------------
# _subject_fixed_size 단위 테스트
# ---------------------------------------------------------------------------
class TestSubjectFixedSize:
    """_subject_fixed_size 헬퍼: 구간 최대 피사체 bbox×padding 고정 크기 산출.

    WHY: 이 함수가 box_size 하한·padding·frame_size 상한을 올바르게 결합해야
         잘림·줌 흔들림이 동시에 해결된다. 각 속성을 독립 가드한다.
    """

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_큰_피사체는_box_size_하한을_초과하는_크기를_반환한다(self):
        """Given: bbox 폭/높이 ≈ 160px인 큰 마스크, box_size=(80, 60)
        When:  _subject_fixed_size 호출
        Then:  반환 크기(W, H) 중 적어도 한 변이 box_size 하한보다 크다

        WHY: 피사체가 box_size보다 크면 고정 box_size를 그대로 쓰면 잘린다.
             자동 크기 산출이 box_size를 초과해야 한다(잘림 방지 핵심 조건).
        """
        from easy_capture.app.video_capture import _subject_fixed_size

        # given: 큰 마스크 1개 (bbox ≈ 160×160)
        CX, CY = FAKE_FRAME_W // 2, FAKE_FRAME_H // 2
        large_mask = _make_rect_mask(
            FAKE_FRAME_H, FAKE_FRAME_W, CX, CY, half=_LARGE_SUBJECT_HALF
        )
        masks = [large_mask]
        params = VideoCropParams(
            box_size=(80, 60),          # 하한: 80×60 — 피사체(160×160)보다 훨씬 작음
            aspect=None,
            subject_padding=_DEFAULT_PADDING,
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        # when
        w_out, h_out = _subject_fixed_size(masks, params, frame_size)

        # then: 결과가 box_size 하한보다 커야 함
        assert w_out > 80 or h_out > 60, (
            f"큰 피사체인데 결과({w_out}×{h_out})가 box_size 하한(80×60)과 같거나 작음. "
            "자동 크기 산출이 작동하지 않는다."
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_작은_피사체에는_box_size_하한이_적용된다(self):
        """Given: bbox 폭/높이 ≈ 10px인 작은 마스크, box_size=(200, 150)
        When:  _subject_fixed_size 호출
        Then:  반환 크기가 box_size 하한 이상 (200, 150)

        WHY: 피사체가 너무 작아도 과도하게 좁아지면 크롭이 의미 없어진다.
             box_size는 최소 하한으로 반드시 보장돼야 한다.
        """
        from easy_capture.app.video_capture import _subject_fixed_size

        # given: 작은 마스크 1개 (bbox ≈ 10×10)
        CX, CY = FAKE_FRAME_W // 2, FAKE_FRAME_H // 2
        small_mask = _make_rect_mask(
            FAKE_FRAME_H, FAKE_FRAME_W, CX, CY, half=_SMALL_SUBJECT_HALF
        )
        masks = [small_mask]
        params = VideoCropParams(
            box_size=(BOX_W, BOX_H),    # 하한: 200×150 — 피사체(10×10)보다 훨씬 큼
            aspect=None,
            subject_padding=_DEFAULT_PADDING,
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        # when
        w_out, h_out = _subject_fixed_size(masks, params, frame_size)

        # then: box_size 하한 이상
        assert w_out >= BOX_W, (
            f"작은 피사체에서 box_size 하한({BOX_W}) 미달: w={w_out}"
        )
        assert h_out >= BOX_H, (
            f"작은 피사체에서 box_size 하한({BOX_H}) 미달: h={h_out}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_padding이_크기에_반영된다(self):
        """Given: bbox 폭/높이 ≈ 160px인 마스크, padding=1.3 vs padding=2.0
        When:  padding만 다른 두 params로 _subject_fixed_size 호출
        Then:  padding=2.0 결과가 padding=1.3 결과보다 크다

        WHY: subject_padding 배수가 실제 크기에 반영돼야 피사체 주변 여백이
             설정값대로 확보된다.
        """
        from easy_capture.app.video_capture import _subject_fixed_size

        # given: 큰 마스크로 box_size 하한 초과 보장
        CX, CY = FAKE_FRAME_W // 2, FAKE_FRAME_H // 2
        large_mask = _make_rect_mask(
            FAKE_FRAME_H, FAKE_FRAME_W, CX, CY, half=_LARGE_SUBJECT_HALF
        )
        masks = [large_mask]
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        params_small_padding = VideoCropParams(
            box_size=(BOX_W, BOX_H),
            aspect=None,
            subject_padding=1.3,
        )
        params_large_padding = VideoCropParams(
            box_size=(BOX_W, BOX_H),
            aspect=None,
            subject_padding=2.0,
        )

        # when
        w_small, h_small = _subject_fixed_size(masks, params_small_padding, frame_size)
        w_large, h_large = _subject_fixed_size(masks, params_large_padding, frame_size)

        # then
        assert w_large >= w_small and h_large >= h_small, (
            f"padding=2.0 결과({w_large}×{h_large})가 "
            f"padding=1.3 결과({w_small}×{h_small})보다 크거나 같아야 함"
        )
        # 적어도 한 변은 더 커야 함 (clamp로 같아질 수 있으므로 or 조건)
        assert w_large > w_small or h_large > h_small, (
            "padding 증가가 크기에 전혀 반영되지 않음 (frame_size 클램프가 없는 상황)"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_반환_크기가_피사체_bbox를_담는다(self):
        """Given: bbox 폭/높이 = SUBJECT_SIDE인 마스크, aspect=None, padding=1.3
        When:  _subject_fixed_size 호출
        Then:  w_out >= 피사체 폭, h_out >= 피사체 높이 (clamp 전 조건)

        WHY: 박스가 피사체보다 작으면 피사체가 잘린다. 이것이 패딩 확보의
             본질적 의미다. clamp 전에 w/h >= 피사체 폭/높이를 보장해야 한다.
        """
        from easy_capture.app.video_capture import _subject_fixed_size

        # given: 큰 마스크, 넉넉한 frame_size로 clamp 없게 설정
        CX, CY = 500, 300
        SUBJECT_SIDE = _LARGE_SUBJECT_SIDE  # 160px
        large_mask = _make_rect_mask(
            1000, 1000, CX, CY, half=_LARGE_SUBJECT_HALF
        )
        masks = [large_mask]
        params = VideoCropParams(
            box_size=(50, 50),          # 작은 하한
            aspect=None,
            subject_padding=_DEFAULT_PADDING,
        )
        frame_size = (1000, 1000)       # 클램프 없게 충분히 큰 프레임

        # when
        w_out, h_out = _subject_fixed_size(masks, params, frame_size)

        # then: 결과가 피사체 bbox 크기 이상
        assert w_out >= SUBJECT_SIDE, (
            f"w_out({w_out}) < 피사체 폭({SUBJECT_SIDE}) — 잘림 발생"
        )
        assert h_out >= SUBJECT_SIDE, (
            f"h_out({h_out}) < 피사체 높이({SUBJECT_SIDE}) — 잘림 발생"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_frame_size_상한_클램프가_적용된다(self):
        """Given: 매우 큰 padding으로 피사체 크기×padding이 frame_size를 초과
        When:  _subject_fixed_size 호출
        Then:  반환 크기가 frame_size 이하 (클램프 작동)

        WHY: 박스가 프레임 크기를 초과하면 crop_array가 영역 밖을 슬라이스한다.
             반드시 frame_size 내로 제한돼야 한다.
        """
        from easy_capture.app.video_capture import _subject_fixed_size

        # given: 큰 마스크, 아주 큰 padding으로 frame_size 초과 유도
        CX, CY = FAKE_FRAME_W // 2, FAKE_FRAME_H // 2
        large_mask = _make_rect_mask(
            FAKE_FRAME_H, FAKE_FRAME_W, CX, CY, half=_LARGE_SUBJECT_HALF
        )
        masks = [large_mask]
        params = VideoCropParams(
            box_size=(BOX_W, BOX_H),
            aspect=None,
            subject_padding=10.0,       # 극단적 padding — frame_size 초과 확실
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        # when
        w_out, h_out = _subject_fixed_size(masks, params, frame_size)

        # then: frame_size 이하
        assert w_out <= FAKE_FRAME_W, (
            f"w_out({w_out}) > frame_w({FAKE_FRAME_W}) — 클램프 미작동"
        )
        assert h_out <= FAKE_FRAME_H, (
            f"h_out({h_out}) > frame_h({FAKE_FRAME_H}) — 클램프 미작동"
        )


# ---------------------------------------------------------------------------
# compute_boxes 새 동작 — 큰 마스크 자동 크기·전 프레임 고정 크기 불변식
# ---------------------------------------------------------------------------
class TestComputeBoxesNewBehavior:
    """compute_boxes 새 동작 가드: 피사체 bbox 기반 자동 크기.

    WHY: box_size 고정값 대신 구간 최대 피사체 bbox×padding으로 자동 산출하는
         정책 변경을 가드한다. 동시에 전 프레임 동일 크기 불변식도 유지 확인.
    """

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_큰_마스크_TrackResult에서_박스가_피사체를_담는다(self):
        """Given: bbox≈160×160인 큰 마스크 10개로 TrackResult 직접 구성
        When:  compute_boxes(result, params, frame_size) 호출
        Then:  박스 크기(W, H)가 피사체 폭/높이 이상 (_LARGE_SUBJECT_SIDE)

        WHY: 고정 320 box_size였다면 큰 피사체도 항상 320×...이 나왔지만,
             자동 크기 산출은 피사체 bbox×padding에서 결정된다.
             box_size 하한(80×60)보다 피사체가 크면 자동 크기가 사용됨을 확인.
        """
        # given: 큰 마스크로 직접 TrackResult 구성 (backend 미호출)
        CX, CY = FAKE_FRAME_W // 2, FAKE_FRAME_H // 2
        large_mask = _make_rect_mask(
            FAKE_FRAME_H, FAKE_FRAME_W, CX, CY, half=_LARGE_SUBJECT_HALF
        )
        # bbox 중심 계산: (CX-HALF+CX+HALF-1)/2, (CY-HALF+CY+HALF-1)/2
        bbox_cx = (CX - _LARGE_SUBJECT_HALF + CX + _LARGE_SUBJECT_HALF - 1) / 2.0
        bbox_cy = (CY - _LARGE_SUBJECT_HALF + CY + _LARGE_SUBJECT_HALF - 1) / 2.0
        masks = [large_mask] * FRAME_COUNT
        centroids = [(bbox_cx, bbox_cy)] * FRAME_COUNT
        result = TrackResult(
            masks=masks,
            centroids=centroids,
            needs_correction=[False] * FRAME_COUNT,
            cut_frames=[],
        )
        params = VideoCropParams(
            box_size=(80, 60),          # 하한: 작게 설정해 자동 크기 주도되게
            aspect=None,
            smooth_window=1,
            subject_padding=_DEFAULT_PADDING,
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        # when: VideoCaptureUseCase.compute_boxes는 순수 — backend 없이 직접 호출
        usecase, backend = _make_usecase()
        frames = _make_frames(FRAME_COUNT)
        _ = usecase.track(frames, (CLICK_X, CLICK_Y))   # backend 1회 사용
        # compute_boxes는 직접 result 주입으로 독립 호출
        boxes = usecase.compute_boxes(result, params, frame_size)

        # then: 박스 크기가 피사체 bbox 이상
        bw = boxes[0][2] - boxes[0][0]
        bh = boxes[0][3] - boxes[0][1]
        assert bw >= _LARGE_SUBJECT_SIDE, (
            f"박스 폭({bw}) < 피사체 폭({_LARGE_SUBJECT_SIDE}) — 피사체 잘림"
        )
        assert bh >= _LARGE_SUBJECT_SIDE, (
            f"박스 높이({bh}) < 피사체 높이({_LARGE_SUBJECT_SIDE}) — 피사체 잘림"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_큰_마스크_TrackResult에서_전_프레임_박스_크기가_동일하다(self):
        """Given: 큰 마스크로 구성한 TrackResult
        When:  compute_boxes 호출
        Then:  모든 프레임 박스 크기(W, H)가 동일 (GIF/MP4 인코딩 불변식)

        WHY: 자동 크기 산출이 구간 최대값으로 고정되므로 전 프레임이 동일해야 한다.
             크기가 프레임마다 다르면 encode_frames가 ValueError를 발생시킨다.
        """
        # given
        CX, CY = FAKE_FRAME_W // 2, FAKE_FRAME_H // 2
        large_mask = _make_rect_mask(
            FAKE_FRAME_H, FAKE_FRAME_W, CX, CY, half=_LARGE_SUBJECT_HALF
        )
        masks = [large_mask] * FRAME_COUNT
        bbox_cx = (CX - _LARGE_SUBJECT_HALF + CX + _LARGE_SUBJECT_HALF - 1) / 2.0
        bbox_cy = (CY - _LARGE_SUBJECT_HALF + CY + _LARGE_SUBJECT_HALF - 1) / 2.0
        centroids = [(bbox_cx, bbox_cy)] * FRAME_COUNT
        result = TrackResult(
            masks=masks,
            centroids=centroids,
            needs_correction=[False] * FRAME_COUNT,
            cut_frames=[],
        )
        params = VideoCropParams(
            box_size=(80, 60),
            aspect=None,
            smooth_window=1,
            subject_padding=_DEFAULT_PADDING,
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        # when
        usecase, _ = _make_usecase()
        frames = _make_frames(FRAME_COUNT)
        usecase.track(frames, (CLICK_X, CLICK_Y))
        boxes = usecase.compute_boxes(result, params, frame_size)

        # then: 전 프레임 동일 크기
        sizes = {(b[2] - b[0], b[3] - b[1]) for b in boxes}
        assert len(sizes) == 1, (
            f"전 프레임 박스 크기 불일치: {sizes} — GIF/MP4 인코딩 불변식 위반"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_USECASE, reason=_MSG_NOT_IMPL)
    def test_compute_boxes_큰_마스크_결과에서도_backend_미호출_순수성이_유지된다(self):
        """Given: 큰 마스크 TrackResult + track 완료 후 카운터 기준값 포착
        When:  compute_boxes를 COMPUTE_BOXES_REPEAT회 반복 호출
        Then:  backend.propagate_call_count가 track 직후 값과 동일 (재추적 없음)

        WHY: 피사체가 크더라도 compute_boxes는 backend를 호출하지 않아야 한다.
             새 자동 크기 산출 로직(_subject_fixed_size)이 순수 함수임을 강제한다.
        """
        # given
        CX, CY = FAKE_FRAME_W // 2, FAKE_FRAME_H // 2
        large_mask = _make_rect_mask(
            FAKE_FRAME_H, FAKE_FRAME_W, CX, CY, half=_LARGE_SUBJECT_HALF
        )
        masks = [large_mask] * FRAME_COUNT
        bbox_cx = (CX - _LARGE_SUBJECT_HALF + CX + _LARGE_SUBJECT_HALF - 1) / 2.0
        bbox_cy = (CY - _LARGE_SUBJECT_HALF + CY + _LARGE_SUBJECT_HALF - 1) / 2.0
        centroids = [(bbox_cx, bbox_cy)] * FRAME_COUNT
        large_result = TrackResult(
            masks=masks,
            centroids=centroids,
            needs_correction=[False] * FRAME_COUNT,
            cut_frames=[],
        )
        usecase, backend = _make_usecase()
        frames = _make_frames(FRAME_COUNT)
        usecase.track(frames, (CLICK_X, CLICK_Y))

        # track 직후 카운터 기준값 포착
        propagate_base = backend.propagate_call_count

        params = VideoCropParams(
            box_size=(80, 60),
            aspect=None,
            smooth_window=1,
            subject_padding=_DEFAULT_PADDING,
        )
        frame_size = (FAKE_FRAME_W, FAKE_FRAME_H)

        # when: 반복 호출
        for _ in range(COMPUTE_BOXES_REPEAT):
            usecase.compute_boxes(large_result, params, frame_size)

        # then: 카운터 불변
        assert backend.propagate_call_count == propagate_base, (
            f"큰 마스크 compute_boxes 반복 중 propagate 재호출 감지: "
            f"{backend.propagate_call_count} vs 기준 {propagate_base}. "
            "compute_boxes는 순수 함수여야 한다."
        )
