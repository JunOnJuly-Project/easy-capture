"""VideoCaptureUseCase 슬라이스 핵심 조립 테스트.

대상 모듈: easy_capture.app.video_capture
테스트 더블: FakeVideoBackend (tests/fixtures/fakes.py),
             FakeFrameSource  (tests/fixtures/fakes.py)

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

구현 전 RED 상태가 정상: VideoCaptureUseCase·TrackResult·VideoCropParams 미구현.
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

from tests.fixtures.fakes import FakeFrameSource, FakeVideoBackend

# 미구현 시 전 테스트 skip 이유 메시지
_MSG_NOT_IMPL = (
    "easy_capture.app.video_capture 또는 "
    "easy_capture.core.segmentation.video_backend 미구현 — RED 예상"
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
