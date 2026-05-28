"""비디오 export 슬라이스 테스트.

대상 모듈: easy_capture.core.export.video_export
테스트 전략: 합성 RGB 프레임 시퀀스만 사용. SAM2·PyAV·PySide6 의존 금지.

이 테스트 파일이 검증하는 계약:
  1. VideoExportConfig  — frozen dataclass, 기본값 검증
  2. crop_frames        — N 프레임 + N 박스 → N개 동일 크기 크롭 (crop_array 재사용)
  3. GIF 라운드트립     — encode_frames(fmt=gif) → imageio.mimread → 프레임 수·크기 일치
  4. MP4 라운드트립     — imageio-ffmpeg 미설치 시 importorskip, 설치 시 파일·크기 검증
  5. 프레임 크기 불일치 예외 — 동일 크기 요구 위반 시 ValueError
  6. gap_policy 출력 인덱스 반영 — BACKGROUND/CUT 정책별 프레임 선택 검증
  7. GIF duration ms 회귀 가드 — duration 단위 ms 검증
  8. GIF per-frame duration 통합 (Story 2) — segments 가변 duration 검증

구현 전 RED 상태가 정상: core/export/video_export.py 미구현.
"""
from __future__ import annotations

import numpy as np
import pytest

# --- imageio 미설치 시 관련 테스트 skip ---
imageio = pytest.importorskip("imageio", reason="imageio 미설치 — GIF/MP4 라운드트립 건너뜀")

# --- 비디오 export 미구현 → try/except 격리 ---
# WHY: 구현 전이므로 import 자체가 실패한다. 이 파일 로드 시 오류로
#      기존 166개 테스트를 차단하지 않도록 한다.
try:
    from easy_capture.core.export.video_export import (
        VideoExportConfig,
        crop_frames,
        encode_frames,
    )
    from easy_capture.core.tracking.gap_policy import GapPolicy
    _HAS_VIDEO_EXPORT = True
except ModuleNotFoundError:
    VideoExportConfig = None  # type: ignore[assignment,misc]
    crop_frames = None  # type: ignore[assignment,misc]
    encode_frames = None  # type: ignore[assignment,misc]
    GapPolicy = None  # type: ignore[assignment,misc]
    _HAS_VIDEO_EXPORT = False

# --- timeremap — Story 2 per-frame duration 통합 테스트용 ---
# WHY: SpeedSegment·build_playback_schedule·clamp_durations_for_gif는
#      timeremap 모듈에서 이미 구현됨. import 실패 시 Story 2 테스트만 skip.
try:
    from easy_capture.core.timing.timeremap import (
        SpeedSegment,
        build_playback_schedule,
        clamp_durations_for_gif,
    )
    _HAS_TIMEREMAP = True
except ModuleNotFoundError:
    SpeedSegment = None  # type: ignore[assignment,misc]
    build_playback_schedule = None  # type: ignore[assignment,misc]
    clamp_durations_for_gif = None  # type: ignore[assignment,misc]
    _HAS_TIMEREMAP = False

_MSG_NO_TIMEREMAP = "easy_capture.core.timing.timeremap 미설치"
_MSG_STORY2_NOT_IMPL = (
    "Story 2 미구현 — VideoExportConfig.segments 또는 timeremap 미지원"
)

_MSG_NOT_IMPL = "easy_capture.core.export.video_export 미구현 — RED 예상"

# --- 트림 슬라이스 — TrimRange 신규 심볼 격리 (TDD RED) ---
# WHY: TrimRange는 timeremap에 추가될 신규 심볼이라 구현 전 ImportError로 실패한다.
#      _HAS_TRIM 플래그로 트림/루프 테스트만 정확히 skip 처리한다.
try:
    from easy_capture.core.timing.timeremap import TrimRange
    _HAS_TRIM = True
except ImportError:
    TrimRange = None  # type: ignore[assignment,misc]
    _HAS_TRIM = False

_MSG_TRIM_NOT_IMPL = (
    "트림+루프 미구현 — VideoExportConfig.trim/loop_count 또는 "
    "timeremap.TrimRange 미지원 (RED 예상)"
)

# ---------------------------------------------------------------------------
# 테스트 상수
# ---------------------------------------------------------------------------
# 합성 프레임 크기 (GIF/MP4 인코딩 실험용 최소 크기)
SYNTH_FRAME_W = 64
SYNTH_FRAME_H = 48
SYNTH_FRAME_COUNT = 6

# 크롭 박스 크기 (GIF/MP4 출력 크기)
CROP_BOX_W = 32
CROP_BOX_H = 24

# 기본 크롭 박스 (x1, y1, x2, y2) — SYNTH 프레임 내부에 완전히 들어가도록
_CROP_BOX: tuple[int, int, int, int] = (8, 6, 8 + CROP_BOX_W, 6 + CROP_BOX_H)

# GIF 출력 fps
GIF_FPS = 12.0

# MP4 출력 fps
MP4_FPS = 24.0


# ---------------------------------------------------------------------------
# 합성 프레임 픽스처 헬퍼
# ---------------------------------------------------------------------------
def _make_synth_frames(
    n: int = SYNTH_FRAME_COUNT,
    h: int = SYNTH_FRAME_H,
    w: int = SYNTH_FRAME_W,
) -> list[np.ndarray]:
    """n개의 합성 RGB uint8 프레임 리스트를 반환한다.

    각 프레임은 인덱스 i를 R채널 오프셋으로 써서 시간적으로 구분 가능하다.
    WHY: 프레임별로 다른 색상을 갖게 해 GIF/MP4 디코드 후 프레임 순서 검증에
         활용할 수 있게 한다. 실제 내용보다 크기·개수 검증에 집중.
    """
    frames = []
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # R채널: 프레임 인덱스 기반 구분값 (0~255 내 순환)
        frame[:, :, 0] = (i * 40) % 256
        # G채널: 고정 그라디언트
        frame[:, :, 1] = (np.arange(w) * 255 // max(w - 1, 1)).astype(np.uint8)
        # B채널: 고정값
        frame[:, :, 2] = 128
        frames.append(frame)
    return frames


def _make_crop_boxes(
    n: int = SYNTH_FRAME_COUNT,
    box: tuple[int, int, int, int] = _CROP_BOX,
) -> list[tuple[int, int, int, int]]:
    """n개 동일 크롭 박스 리스트를 반환한다."""
    return [box] * n


# ---------------------------------------------------------------------------
# VideoExportConfig 테스트
# ---------------------------------------------------------------------------
class TestVideoExportConfig:
    """VideoExportConfig frozen dataclass 계약 검증."""

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_VideoExportConfig_기본값이_올바르다(self):
        """Given: 인자 없이 VideoExportConfig 생성
        When:  각 필드 접근
        Then:  fmt='gif', fps=12.0, gap_policy=GapPolicy.BACKGROUND
        """
        config = VideoExportConfig()

        assert config.fmt == "gif"
        assert config.fps == 12.0
        assert config.gap_policy == GapPolicy.BACKGROUND

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_VideoExportConfig_mp4_설정이_가능하다(self):
        """Given: fmt='mp4', fps=24.0
        When:  VideoExportConfig 생성
        Then:  필드값 그대로 반환
        """
        config = VideoExportConfig(fmt="mp4", fps=24.0)

        assert config.fmt == "mp4"
        assert config.fps == 24.0

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_VideoExportConfig는_frozen_dataclass이다(self):
        """Given: VideoExportConfig 인스턴스
        When:  필드 수정 시도
        Then:  AttributeError 또는 TypeError 발생 (불변 보장)
        """
        config = VideoExportConfig()

        with pytest.raises((AttributeError, TypeError)):
            config.fmt = "mp4"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# crop_frames 테스트
# ---------------------------------------------------------------------------
class TestCropFrames:
    """crop_frames: N 프레임 + N 박스 → N개 동일 크기 크롭."""

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_crop_frames_결과_개수가_입력_프레임_수와_일치한다(self):
        """Given: 6개 합성 프레임, 6개 동일 박스
        When:  crop_frames 호출
        Then:  결과 리스트 길이 == 6
        """
        frames = _make_synth_frames()
        boxes = _make_crop_boxes()

        crops = crop_frames(frames, boxes)

        assert len(crops) == SYNTH_FRAME_COUNT, (
            f"crop 결과 수 불일치: {len(crops)} vs {SYNTH_FRAME_COUNT}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_crop_frames_결과_각_크롭_크기가_박스_크기와_일치한다(self):
        """Given: 박스 (8, 6, 40, 30) → 32×24 크롭 기대
        When:  crop_frames 호출
        Then:  각 crop.shape == (CROP_BOX_H, CROP_BOX_W, 3)
        """
        frames = _make_synth_frames()
        boxes = _make_crop_boxes()

        crops = crop_frames(frames, boxes)

        for i, crop in enumerate(crops):
            assert crop.shape == (CROP_BOX_H, CROP_BOX_W, 3), (
                f"프레임 {i} crop shape 불일치: {crop.shape}"
            )

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_crop_frames_전_프레임_크롭_크기가_동일하다(self):
        """Given: 동일 박스 6개
        When:  crop_frames
        Then:  모든 크롭 shape 동일 (GIF/MP4 인코딩 전제)
        """
        frames = _make_synth_frames()
        boxes = _make_crop_boxes()

        crops = crop_frames(frames, boxes)

        shapes = [c.shape for c in crops]
        assert len(set(shapes)) == 1, (
            f"크롭 크기 불일치: {set(shapes)} — 인코딩 전 전 프레임 동일해야 함"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_crop_frames_크기_불일치_박스_입력_시_ValueError가_발생한다(self):
        """Given: 박스 크기가 서로 다른 boxes 리스트
        When:  crop_frames 호출
        Then:  ValueError 발생 (동일 크기 요구)

        WHY: 크기 불일치 프레임이 encode_frames에 들어가면 인코딩 실패한다.
             이를 crop_frames 단계에서 조기 감지한다.
        """
        frames = _make_synth_frames(n=2)
        # 첫 번째는 정상, 두 번째는 크기 다른 박스
        boxes = [
            (0, 0, CROP_BOX_W, CROP_BOX_H),
            (0, 0, CROP_BOX_W + 10, CROP_BOX_H),
        ]

        with pytest.raises(ValueError):
            crop_frames(frames, boxes)

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_crop_frames_crop_array_재사용으로_픽셀값이_보존된다(self):
        """Given: R채널에 고정값 200을 채운 프레임, 전체 박스
        When:  crop_frames
        Then:  크롭 배열 R채널 == 200 (crop_array 픽셀 보존 검증)

        WHY: crop_frames이 image_export.crop_array를 재사용하는지
             픽셀 수준에서 확인한다(DRY 계약).
        """
        h, w = SYNTH_FRAME_H, SYNTH_FRAME_W
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # R채널 고정

        box = (0, 0, w, h)  # 전체 프레임 박스
        crops = crop_frames([frame], [box])

        np.testing.assert_array_equal(crops[0][:, :, 0], 200)


# ---------------------------------------------------------------------------
# GIF 라운드트립 테스트
# ---------------------------------------------------------------------------
class TestGifRoundtrip:
    """합성 프레임 → encode_frames(gif) → imageio 재로드 → 프레임 수·크기 일치."""

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_GIF_encode_후_파일이_생성된다(self, tmp_path):
        """Given: 6개 32×24 합성 크롭 프레임, fps=12
        When:  encode_frames(fmt='gif')
        Then:  GIF 파일 존재
        """
        crops = crop_frames(
            _make_synth_frames(), _make_crop_boxes()
        )
        config = VideoExportConfig(fmt="gif", fps=GIF_FPS)
        output_path = str(tmp_path / "out.gif")

        encode_frames(crops, output_path, config)

        assert (tmp_path / "out.gif").exists(), "GIF 파일이 생성되지 않음"

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_GIF_라운드트립_프레임_수가_일치한다(self, tmp_path):
        """Given: 6개 합성 크롭 프레임
        When:  encode → imageio.mimread 재로드
        Then:  재로드된 프레임 수 == 6

        WHY: GIF 인코딩이 프레임을 손실 없이 보존하는지 검증한다.
             imageio.mimread는 모든 GIF 프레임을 리스트로 반환한다.
        """
        crops = crop_frames(
            _make_synth_frames(), _make_crop_boxes()
        )
        config = VideoExportConfig(fmt="gif", fps=GIF_FPS)
        output_path = str(tmp_path / "roundtrip.gif")

        encode_frames(crops, output_path, config)

        reloaded = imageio.mimread(output_path)
        assert len(reloaded) == SYNTH_FRAME_COUNT, (
            f"GIF 재로드 프레임 수 불일치: {len(reloaded)} vs {SYNTH_FRAME_COUNT}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_GIF_라운드트립_프레임_크기가_일치한다(self, tmp_path):
        """Given: 32×24 합성 크롭
        When:  encode → imageio.mimread
        Then:  각 프레임 shape[:2] == (CROP_BOX_H, CROP_BOX_W)
        """
        crops = crop_frames(
            _make_synth_frames(), _make_crop_boxes()
        )
        config = VideoExportConfig(fmt="gif", fps=GIF_FPS)
        output_path = str(tmp_path / "size_check.gif")

        encode_frames(crops, output_path, config)

        reloaded = imageio.mimread(output_path)
        for i, frame in enumerate(reloaded):
            h, w = frame.shape[:2]
            assert h == CROP_BOX_H and w == CROP_BOX_W, (
                f"GIF 프레임 {i} 크기 불일치: ({h}, {w}) vs ({CROP_BOX_H}, {CROP_BOX_W})"
            )

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_GIF_encode_전_프레임_크기_불일치_시_ValueError가_발생한다(self):
        """Given: 크기가 다른 크롭 프레임 리스트
        When:  encode_frames(fmt='gif')
        Then:  ValueError 발생 (동일 크기 요구)
        """
        frame_a = np.zeros((24, 32, 3), dtype=np.uint8)
        frame_b = np.zeros((24, 40, 3), dtype=np.uint8)  # 너비 다름
        config = VideoExportConfig(fmt="gif")

        with pytest.raises(ValueError):
            encode_frames([frame_a, frame_b], "/dev/null", config)


# ---------------------------------------------------------------------------
# MP4 라운드트립 테스트 (imageio-ffmpeg 조건부)
# ---------------------------------------------------------------------------
def _skip_if_no_ffmpeg():
    """imageio-ffmpeg 미설치 시 skip 마커를 반환한다."""
    try:
        import imageio_ffmpeg  # noqa: F401
        return False
    except ImportError:
        return True


_NO_FFMPEG = _skip_if_no_ffmpeg()
_MSG_NO_FFMPEG = "imageio-ffmpeg 미설치 — MP4 라운드트립 건너뜀"


class TestMp4Roundtrip:
    """합성 프레임 → encode_frames(mp4) → 파일 존재·크기 검증 (조건부)."""

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(_NO_FFMPEG, reason=_MSG_NO_FFMPEG)
    def test_MP4_encode_후_파일이_생성된다(self, tmp_path):
        """Given: 6개 32×24 합성 크롭, fps=24
        When:  encode_frames(fmt='mp4')
        Then:  MP4 파일 존재, 크기 > 0 (빈 파일 아님)
        """
        crops = crop_frames(
            _make_synth_frames(), _make_crop_boxes()
        )
        config = VideoExportConfig(fmt="mp4", fps=MP4_FPS)
        output_path = str(tmp_path / "out.mp4")

        encode_frames(crops, output_path, config)

        assert (tmp_path / "out.mp4").exists(), "MP4 파일이 생성되지 않음"
        assert (tmp_path / "out.mp4").stat().st_size > 0, "MP4 파일이 비어 있음"

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(_NO_FFMPEG, reason=_MSG_NO_FFMPEG)
    def test_MP4_라운드트립_프레임_크기가_일치한다(self, tmp_path):
        """Given: 32×24 합성 크롭 6개
        When:  encode → imageio.get_reader 재로드
        Then:  각 프레임 shape[:2] == (CROP_BOX_H, CROP_BOX_W) (코덱 손실 허용)

        WHY: MP4 인코딩은 손실 압축이므로 픽셀값 정확성보다 크기 일치만 검증한다.
        """
        crops = crop_frames(
            _make_synth_frames(), _make_crop_boxes()
        )
        config = VideoExportConfig(fmt="mp4", fps=MP4_FPS)
        output_path = str(tmp_path / "size_check.mp4")

        encode_frames(crops, output_path, config)

        with imageio.get_reader(output_path, format="ffmpeg") as reader:
            for i, frame in enumerate(reader):
                h, w = frame.shape[:2]
                assert h == CROP_BOX_H and w == CROP_BOX_W, (
                    f"MP4 프레임 {i} 크기 불일치: ({h}, {w}) vs "
                    f"({CROP_BOX_H}, {CROP_BOX_W})"
                )
                if i >= SYNTH_FRAME_COUNT - 1:
                    break

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(_NO_FFMPEG, reason=_MSG_NO_FFMPEG)
    def test_MP4_encode_전_프레임_크기_불일치_시_ValueError가_발생한다(self):
        """Given: 크기가 다른 크롭 프레임
        When:  encode_frames(fmt='mp4')
        Then:  ValueError 발생
        """
        frame_a = np.zeros((24, 32, 3), dtype=np.uint8)
        frame_b = np.zeros((30, 32, 3), dtype=np.uint8)  # 높이 다름
        config = VideoExportConfig(fmt="mp4")

        with pytest.raises(ValueError):
            encode_frames([frame_a, frame_b], "/dev/null", config)


# ---------------------------------------------------------------------------
# gap_policy 출력 인덱스 반영 검증
# ---------------------------------------------------------------------------
class TestGapPolicyFrameSelection:
    """gap_policy에 따른 프레임 선택이 crop_frames 입력으로 올바르게 연결되는지 검증."""

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_BACKGROUND_정책_전_인덱스_선택_후_crop_frames_개수가_전체와_일치한다(self):
        """Given: 6개 프레임, valid_flags=[T,T,T,F,F,F], BACKGROUND 정책
        When:  build_output_indices → 인덱스로 프레임 선택 → crop_frames
        Then:  crop 결과 6개 (전 프레임 선택)

        WHY: BACKGROUND 정책은 갭 구간도 포함해 전 프레임을 출력한다.
             선택된 프레임 수가 유효 프레임보다 많음을 확인한다.
        """
        from easy_capture.core.tracking.gap_policy import GapPolicy, build_output_indices

        n = SYNTH_FRAME_COUNT  # 6
        valid_flags = [True] * (n // 2) + [False] * (n - n // 2)

        output_indices = build_output_indices(valid_flags, GapPolicy.BACKGROUND)
        frames = _make_synth_frames(n)
        selected_frames = [frames[i] for i in output_indices]
        boxes = _make_crop_boxes(len(selected_frames))

        crops = crop_frames(selected_frames, boxes)

        assert len(crops) == n, (
            f"BACKGROUND 정책 crop 수 불일치: {len(crops)} vs {n}"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_CUT_정책_유효_프레임만_선택_후_crop_frames_개수가_줄어든다(self):
        """Given: 6개 프레임, valid_flags=[T,T,T,F,F,F], CUT 정책
        When:  build_output_indices → 인덱스로 프레임 선택 → crop_frames
        Then:  crop 결과 3개 (유효 프레임만)
        """
        from easy_capture.core.tracking.gap_policy import GapPolicy, build_output_indices

        n = SYNTH_FRAME_COUNT  # 6
        half = n // 2  # 3
        valid_flags = [True] * half + [False] * (n - half)

        output_indices = build_output_indices(valid_flags, GapPolicy.CUT)
        frames = _make_synth_frames(n)
        selected_frames = [frames[i] for i in output_indices]
        boxes = _make_crop_boxes(len(selected_frames))

        crops = crop_frames(selected_frames, boxes)

        assert len(crops) == half, (
            f"CUT 정책 crop 수 불일치: {len(crops)} vs {half}"
        )


# ---------------------------------------------------------------------------
# GIF duration 밀리초 회귀 가드
# ---------------------------------------------------------------------------
# GIF centisecond(10ms) 해상도 허용 오차
_GIF_DURATION_TOLERANCE_MS = 10

# 회귀 가드용 fps 값들
_GIF_REGRESSION_FPS_12 = 12.0    # duration ≈ 83ms (이전 버그: ≈0.083 초 단위 혼동)
_GIF_REGRESSION_FPS_10 = 10.0    # duration = 100ms


class TestGifDurationMs:
    """GIF 재생속도 회귀 가드: encode_frames의 duration 단위가 ms인지 검증.

    WHY: imageio 2.28+ 에서 GIF duration 단위가 '초'→'밀리초'로 변경됐다.
         이전 구현에서 1/fps(초)를 그대로 넣어 재생속도가 ≈100배 느렸다
         (≈0.083을 ms로 해석 → 거의 0ms → 뷰어 기본 100ms 적용).
         수정 후에는 1000/fps(ms)를 넣어야 하며, 이 테스트가 회귀를 방지한다.

    GIF 스펙: delay 필드는 centisecond(10ms) 단위이므로 ±10ms 허용.
    """

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_fps_12_GIF_duration이_밀리초_단위로_올바르다(self, tmp_path):
        """Given: fps=12로 GIF 인코딩
        When:  imageio.get_reader(path).get_meta_data()['duration'] 읽기
        Then:  duration ≈ 1000/12 ≈ 83ms (±10ms)
               이전 버그: ≈0.083ms — 뷰어가 기본 100ms(≈10fps)로 느리게 재생

        WHY: fps=12 기대 duration=83ms. 이전 버그(초 단위) 시에는 ≈0으로
             측정되어 뷰어가 기본값(100ms)을 쓴다. 수정 후에는 80~90ms 범위.
        """
        crops = crop_frames(
            _make_synth_frames(), _make_crop_boxes()
        )
        config = VideoExportConfig(fmt="gif", fps=_GIF_REGRESSION_FPS_12)
        output_path = str(tmp_path / "fps12.gif")

        encode_frames(crops, output_path, config)

        # GIF 메타데이터에서 duration 읽기
        with imageio.get_reader(output_path) as reader:
            meta = reader.get_meta_data()
        duration_ms = meta.get("duration", None)

        assert duration_ms is not None, "GIF 메타데이터에 duration 필드 없음"

        # duration 단위 확인: ms 단위여야 함 (80~90ms 범위)
        expected_ms = 1000.0 / _GIF_REGRESSION_FPS_12  # ≈ 83.3ms
        assert abs(duration_ms - expected_ms) <= _GIF_DURATION_TOLERANCE_MS, (
            f"GIF duration 불일치: {duration_ms}ms vs 기대 {expected_ms:.1f}ms "
            f"(±{_GIF_DURATION_TOLERANCE_MS}ms). "
            "1.0/fps(초 단위 버그) 적용 시 ≈0ms 또는 ≈100ms(뷰어 기본값)가 나온다."
        )

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_fps_10_GIF_duration이_밀리초_단위로_올바르다(self, tmp_path):
        """Given: fps=10으로 GIF 인코딩
        When:  메타데이터 duration 읽기
        Then:  duration ≈ 1000/10 = 100ms (±10ms)

        WHY: fps=10은 GIF centisecond 경계(10ms 배수)에 정확히 맞아
             허용 오차 내에서 정밀하게 검증 가능하다.
        """
        crops = crop_frames(
            _make_synth_frames(), _make_crop_boxes()
        )
        config = VideoExportConfig(fmt="gif", fps=_GIF_REGRESSION_FPS_10)
        output_path = str(tmp_path / "fps10.gif")

        encode_frames(crops, output_path, config)

        with imageio.get_reader(output_path) as reader:
            meta = reader.get_meta_data()
        duration_ms = meta.get("duration", None)

        assert duration_ms is not None, "GIF 메타데이터에 duration 필드 없음"

        expected_ms = 1000.0 / _GIF_REGRESSION_FPS_10  # = 100.0ms
        assert abs(duration_ms - expected_ms) <= _GIF_DURATION_TOLERANCE_MS, (
            f"fps=10 GIF duration 불일치: {duration_ms}ms vs 기대 {expected_ms:.1f}ms "
            f"(±{_GIF_DURATION_TOLERANCE_MS}ms)."
        )


# ---------------------------------------------------------------------------
# Story 2 — GIF per-frame duration 통합 (TDD RED)
# ---------------------------------------------------------------------------
# Story 2 테스트 상수 — 매직넘버 금지
_S2_FPS = 12.0                          # 기준 fps (1프레임 균일 duration ≈ 83ms)
_S2_N_FRAMES = 10                       # 총 프레임 수
_S2_SLOW_START = 2                      # 슬로우 구간 시작 인덱스 (포함)
_S2_SLOW_END = 5                        # 슬로우 구간 끝 인덱스 (미포함) → 프레임 2,3,4
_S2_SLOW_FACTOR = 0.5                   # 슬로우 배속 (duration × 2배)
_S2_FAST_FPS = 30.0                     # 패스트 클램프 테스트 기준 fps
_S2_FAST_FACTOR = 4.0                   # 최대 패스트 → 30fps×4x → 8.3ms → 클램프
_GIF_CENTISECOND_TOLERANCE_MS = 10      # GIF centisecond ±10ms 허용 오차
_GIF_MIN_DURATION_MS = 10.0            # 클램프 하한
_GIF_CLAMP_DURATION_MS = 20.0          # 클램프 목표값


def _read_gif_frame_durations(gif_path: str) -> list[int]:
    """PIL로 GIF를 프레임별 순회해 각 프레임의 duration(ms)을 리스트로 반환한다.

    WHY PIL 사용:
      imageio.get_reader()는 전체 메타데이터만 반환하거나 단일 duration 값을 준다.
      PIL Image.seek(i) + info['duration']는 GIF 스펙상 프레임별 delay 필드를
      각각 읽어줘 per-frame duration 검증에 유일하게 적합하다.
    """
    from PIL import Image

    durations: list[int] = []
    with Image.open(gif_path) as img:
        for i in range(img.n_frames):
            img.seek(i)
            durations.append(img.info.get("duration", 0))
    return durations


def _make_synth_frames_n(n: int = _S2_N_FRAMES) -> list[np.ndarray]:
    """Story 2용 n개 합성 크롭 프레임(32×24 RGB)을 반환한다."""
    return _make_synth_frames(n=n, h=CROP_BOX_H, w=CROP_BOX_W)


def _make_uniform_crops(n: int = _S2_N_FRAMES) -> list[np.ndarray]:
    """n개 동일 크기(32×24) 합성 크롭 리스트를 반환한다.

    WHY: Story 2 테스트는 crop_frames 과정 없이 바로 encode_frames에 넘길
         수 있도록 이미 크롭된 프레임을 준비한다.
    """
    return _make_synth_frames_n(n)


# Story 2 테스트 guard: video_export + timeremap 모두 존재해야 실행
_STORY2_SKIP = not (_HAS_VIDEO_EXPORT and _HAS_TIMEREMAP)


class TestGifVariableDuration:
    """Story 2: GIF per-frame duration 통합 테스트.

    VideoExportConfig.segments 필드 + encode_frames segments 분기 계약 검증.

    TDD RED: 구현 전이므로 아래 테스트는 모두 실패(AttributeError 또는 TypeError)
    예상됨. VideoExportConfig에 segments 필드가 없고 encode_frames가 segments를
    무시하기 때문이다.
    """

    @pytest.mark.skipif(_STORY2_SKIP, reason=_MSG_STORY2_NOT_IMPL)
    def test_VideoExportConfig_segments_기본값이_빈_튜플이다(self):
        """Given: 인자 없이 VideoExportConfig 생성
        When:  segments 필드 접근
        Then:  segments == () (빈 tuple, 기본값)

        WHY: segments=() 는 "기존 균일 fps 경로"를 의미한다.
             기본값이 빈 튜플이어야 기존 코드가 segments 지정 없이도 동작한다.
        """
        config = VideoExportConfig()

        assert hasattr(config, "segments"), (
            "VideoExportConfig에 segments 필드가 없음 — Task 2-1 미구현"
        )
        assert config.segments == (), (
            f"segments 기본값 불일치: {config.segments!r} vs () (빈 tuple 기대)"
        )

    @pytest.mark.skipif(_STORY2_SKIP, reason=_MSG_STORY2_NOT_IMPL)
    def test_VideoExportConfig_segments_frozen_불변이다(self):
        """Given: segments=(SpeedSegment(0,3,0.5),)로 생성된 VideoExportConfig
        When:  segments 필드 수정 시도
        Then:  FrozenInstanceError 또는 AttributeError 발생 (frozen dataclass)

        WHY: VideoExportConfig는 frozen=True이므로 segments 포함 모든 필드가
             불변이어야 한다. tuple 타입도 값 변경 불가를 보장한다.
        """
        seg = SpeedSegment(0, 3, _S2_SLOW_FACTOR)
        config = VideoExportConfig(segments=(seg,))

        with pytest.raises((AttributeError, TypeError)):
            config.segments = ()  # type: ignore[misc]

    @pytest.mark.skipif(_STORY2_SKIP, reason=_MSG_STORY2_NOT_IMPL)
    def test_segments_빈_튜플이면_기존_균일_duration_경로와_동일하다(self, tmp_path):
        """Given: segments=()인 VideoExportConfig, fps=12, 10개 합성 크롭
        When:  encode_frames(fmt='gif', segments=()) 호출 후 PIL로 프레임별 duration 읽기
        Then:  모든 프레임 duration ≈ 1000/12 ≈ 83ms (±10ms)

        WHY (무회귀): segments=() 경로는 기존 균일 fps 로직과 완전히 동일해야 한다.
             Story 2 구현이 기존 GIF 라운드트립 테스트를 깨면 안 된다.
        """
        crops = _make_uniform_crops(_S2_N_FRAMES)
        config = VideoExportConfig(fmt="gif", fps=_S2_FPS, segments=())
        output_path = str(tmp_path / "uniform_no_segments.gif")

        encode_frames(crops, output_path, config)

        frame_durations = _read_gif_frame_durations(output_path)
        expected_ms = 1000.0 / _S2_FPS  # ≈ 83.3ms

        assert len(frame_durations) == _S2_N_FRAMES, (
            f"GIF 프레임 수 불일치: {len(frame_durations)} vs {_S2_N_FRAMES}"
        )
        for i, dur in enumerate(frame_durations):
            assert abs(dur - expected_ms) <= _GIF_CENTISECOND_TOLERANCE_MS, (
                f"segments=() 균일 duration 불일치: 프레임 {i} → {dur}ms "
                f"vs 기대 {expected_ms:.1f}ms (±{_GIF_CENTISECOND_TOLERANCE_MS}ms). "
                "segments=() 경로가 기존 균일 fps 경로와 달라짐 — 무회귀 실패."
            )

    @pytest.mark.skipif(_STORY2_SKIP, reason=_MSG_STORY2_NOT_IMPL)
    def test_슬로우_구간_프레임_duration이_균일의_2배다(self, tmp_path):
        """Given: 10개 크롭, segments=(SpeedSegment(2, 5, 0.5),), fps=12
        When:  encode_frames(fmt='gif') 후 PIL로 프레임별 duration 읽기
        Then:  프레임 2,3,4 duration ≈ (1000/12)/0.5 ≈ 166ms (±10ms)
               프레임 0,1,5~9 duration ≈ 1000/12 ≈ 83ms (±10ms)

        WHY: SlowMotion factor=0.5 → 표시시간 = base_duration / 0.5 = base × 2.
             구간 내 프레임만 느려지고 구간 밖은 균일 속도를 유지해야 한다.
             PIL seek(i) + info['duration']으로 프레임별 검증.
        """
        crops = _make_uniform_crops(_S2_N_FRAMES)
        seg = SpeedSegment(_S2_SLOW_START, _S2_SLOW_END, _S2_SLOW_FACTOR)
        config = VideoExportConfig(
            fmt="gif", fps=_S2_FPS, segments=(seg,)
        )
        output_path = str(tmp_path / "slow_segment.gif")

        encode_frames(crops, output_path, config)

        frame_durations = _read_gif_frame_durations(output_path)
        base_ms = 1000.0 / _S2_FPS              # ≈ 83.3ms
        slow_ms = base_ms / _S2_SLOW_FACTOR     # ≈ 166.6ms (factor=0.5 → 2배)
        slow_frames = set(range(_S2_SLOW_START, _S2_SLOW_END))  # {2, 3, 4}

        assert len(frame_durations) == _S2_N_FRAMES, (
            f"슬로우 GIF 프레임 수 불일치: {len(frame_durations)} vs {_S2_N_FRAMES}"
        )
        for i, dur in enumerate(frame_durations):
            if i in slow_frames:
                assert abs(dur - slow_ms) <= _GIF_CENTISECOND_TOLERANCE_MS, (
                    f"슬로우 구간 프레임 {i} duration 불일치: {dur}ms "
                    f"vs 기대 {slow_ms:.1f}ms (±{_GIF_CENTISECOND_TOLERANCE_MS}ms). "
                    f"factor={_S2_SLOW_FACTOR} → 2배 느린 duration 기대."
                )
            else:
                assert abs(dur - base_ms) <= _GIF_CENTISECOND_TOLERANCE_MS, (
                    f"균일 구간 프레임 {i} duration 불일치: {dur}ms "
                    f"vs 기대 {base_ms:.1f}ms (±{_GIF_CENTISECOND_TOLERANCE_MS}ms). "
                    "구간 밖 프레임은 균일 속도여야 함."
                )

    @pytest.mark.skipif(_STORY2_SKIP, reason=_MSG_STORY2_NOT_IMPL)
    def test_패스트_구간_10ms_미만_클램프되어_역전_없음(self, tmp_path):
        """Given: 10개 크롭, segments=(SpeedSegment(0, 10, 4.0),), fps=30
        When:  encode_frames(fmt='gif') 후 PIL로 프레임별 duration 읽기
        Then:  모든 프레임 duration >= 20ms (10ms 미만 → 20ms 클램프)
               원래 duration = 1000/30 / 4.0 ≈ 8.3ms < 10ms → 클램프 필요

        WHY: 패스트×고fps → 8.3ms 는 GIF 10ms 하한 미만이다.
             clamp_durations_for_gif 경유 후 20ms 클램프 적용으로
             뷰어가 delay=0 해석해 ≈100ms로 느려지는 역전을 방지해야 한다.
             20ms ≥ 10ms 이므로 역전(오히려 느려짐) 없음을 보장한다.
        """
        crops = _make_uniform_crops(_S2_N_FRAMES)
        seg = SpeedSegment(0, _S2_N_FRAMES, _S2_FAST_FACTOR)
        config = VideoExportConfig(
            fmt="gif", fps=_S2_FAST_FPS, segments=(seg,)
        )
        output_path = str(tmp_path / "fast_clamp.gif")

        # 클램프 전 기대 duration 검증 (왜 클램프가 필요한지 명시)
        raw_duration_ms = (1000.0 / _S2_FAST_FPS) / _S2_FAST_FACTOR  # ≈ 8.33ms
        assert raw_duration_ms < _GIF_MIN_DURATION_MS, (
            f"테스트 전제 조건 실패: raw_duration={raw_duration_ms:.1f}ms >= "
            f"{_GIF_MIN_DURATION_MS}ms — 클램프 시나리오가 아님"
        )

        encode_frames(crops, output_path, config)

        frame_durations = _read_gif_frame_durations(output_path)
        for i, dur in enumerate(frame_durations):
            assert dur >= _GIF_CLAMP_DURATION_MS, (
                f"클램프 미적용: 프레임 {i} duration={dur}ms < "
                f"{_GIF_CLAMP_DURATION_MS}ms. "
                f"raw={raw_duration_ms:.1f}ms < 10ms → 20ms 클램프 기대."
            )

    @pytest.mark.skipif(_STORY2_SKIP, reason=_MSG_STORY2_NOT_IMPL)
    def test_segments_있어도_loop_0_무한루프_유지된다(self, tmp_path):
        """Given: segments=(SpeedSegment(2,5,0.5),)인 GIF 인코딩
        When:  PIL로 GIF 재로드
        Then:  n_frames == _S2_N_FRAMES (프레임 소실 없음)
               + imageio metadata loop == 0 또는 PIL info 루프 확인

        WHY: segments 적용 후에도 GIF 무한 루프(loop=0) 설정이 유지되어야 한다.
             encode_frames의 기존 loop=0 계약이 per-frame duration 경로에서도
             동일하게 적용됨을 검증한다. 프레임 수 보존도 함께 확인한다.
        """
        crops = _make_uniform_crops(_S2_N_FRAMES)
        seg = SpeedSegment(_S2_SLOW_START, _S2_SLOW_END, _S2_SLOW_FACTOR)
        config = VideoExportConfig(
            fmt="gif", fps=_S2_FPS, segments=(seg,)
        )
        output_path = str(tmp_path / "loop_check.gif")

        encode_frames(crops, output_path, config)

        # PIL로 프레임 수 확인
        from PIL import Image
        with Image.open(output_path) as img:
            n_frames_actual = img.n_frames
            # PIL info에서 loop 정보 확인 (loop=0 = 무한 반복)
            loop_value = img.info.get("loop", 0)

        assert n_frames_actual == _S2_N_FRAMES, (
            f"segments 있는 GIF 프레임 수 불일치: {n_frames_actual} vs {_S2_N_FRAMES}. "
            "frame_indices=range(n)이므로 프레임 복제/드롭 없이 원본 순서 유지 기대."
        )
        # loop=0은 GIF 무한 루프를 의미한다 (표준 GIF89a 스펙)
        assert loop_value == 0, (
            f"GIF loop 값 불일치: {loop_value} vs 0 (무한 루프). "
            "segments 적용 경로에서 loop=0이 유지되어야 한다."
        )

    @pytest.mark.skipif(_STORY2_SKIP, reason=_MSG_STORY2_NOT_IMPL)
    def test_segments_겹침_입력_시_ValueError가_encode_frames에_전파된다(
        self, tmp_path
    ):
        """Given: 겹치는 구간 segments=(SpeedSegment(0,5,0.5), SpeedSegment(3,8,2.0),)
        When:  encode_frames(fmt='gif') 호출
        Then:  ValueError 발생 (normalize_segments가 겹침 감지 → encode_frames 전파)

        WHY: encode_frames가 segments 유무 분기에서 build_playback_schedule을
             호출하면, normalize_segments의 겹침 ValueError가 그대로 전파되어야 한다.
             encode_frames 내부에서 예외를 흡수·은폐하면 안 된다.
        """
        crops = _make_uniform_crops(_S2_N_FRAMES)
        overlapping_segments = (
            SpeedSegment(0, 5, _S2_SLOW_FACTOR),
            SpeedSegment(3, 8, 2.0),   # 0~5와 3~8 겹침
        )
        config = VideoExportConfig(
            fmt="gif", fps=_S2_FPS, segments=overlapping_segments
        )
        output_path = str(tmp_path / "overlap_error.gif")

        with pytest.raises(ValueError):
            encode_frames(crops, output_path, config)


# ---------------------------------------------------------------------------
# Story 3 — MP4 프레임 복제/드롭 통합 (TDD RED)
# ---------------------------------------------------------------------------
# Story 3 테스트 상수 — 매직넘버 금지
_S3_FPS = 12.0                   # MP4 CFR 출력 fps
_S3_N_FRAMES = 10                # 총 프레임 수
_S3_SLOW_START = 2               # 슬로우 구간 시작 인덱스 (포함)
_S3_SLOW_END = 5                 # 슬로우 구간 끝 인덱스 (미포함) → 프레임 2,3,4 (3개)
_S3_SLOW_FACTOR = 0.5            # 슬로우 배속 → 구간 3프레임 ×2 ≈ 6프레임
_S3_FAST_START = 2               # 패스트 구간 시작 인덱스 (포함)
_S3_FAST_END = 6                 # 패스트 구간 끝 인덱스 (미포함) → 프레임 2,3,4,5 (4개)
_S3_FAST_FACTOR = 2.0            # 패스트 배속 → 구간 4프레임 ×0.5 ≈ 2프레임
_S3_FRAME_COUNT_TOLERANCE = 1    # 프레임 수 허용 오차 (Bresenham ±1)

# 패스트 가드 테스트 상수
_S3_GUARD_N_FRAMES = 3           # 패스트 가드용 짧은 총 프레임 수
_S3_GUARD_FAST_START = 0         # 전체 구간 패스트
_S3_GUARD_FAST_FACTOR = 2.0      # 드롭 적극적 — 최소 1프레임 보장 검증

# Story 3 import guard: video_export + timeremap 모두 존재해야 실행
# WHY: schedule_to_cfr_indices는 timeremap 모듈에 있으므로 별도 확인
try:
    from easy_capture.core.timing.timeremap import schedule_to_cfr_indices
    _HAS_CFR_INDICES = True
except ImportError:
    schedule_to_cfr_indices = None  # type: ignore[assignment,misc]
    _HAS_CFR_INDICES = False

_STORY3_SKIP = not (_HAS_VIDEO_EXPORT and _HAS_TIMEREMAP and _HAS_CFR_INDICES)
_MSG_STORY3_NOT_IMPL = (
    "Story 3 미구현 — encode_frames MP4 segments 분기 또는 timeremap 미지원"
)


def _make_synth_crops_s3(n: int = _S3_N_FRAMES) -> list[np.ndarray]:
    """Story 3용 n개 합성 크롭 프레임(32×24 RGB)을 반환한다.

    WHY 프레임별 색 다르게:
      R채널 값으로 프레임 인덱스를 구분해 복제/드롭 후 재배열이 올바른지
      프레임 수 단위로 검증할 수 있게 한다.
    """
    return _make_synth_frames(n=n, h=CROP_BOX_H, w=CROP_BOX_W)


def _count_mp4_frames(mp4_path: str) -> int:
    """imageio(ffmpeg)로 MP4 파일의 총 프레임 수를 반환한다.

    WHY imageio get_reader:
      imageio-ffmpeg의 get_reader("ffmpeg")는 프레임 단위 순회를 지원하며
      GIF mimread와 달리 MP4 CFR 검증에 적합하다.
    WHY 전체 순회:
      meta_data의 nframes 필드는 추정치일 수 있으므로 직접 순회로 정확히 센다.
    """
    import imageio
    count = 0
    with imageio.get_reader(mp4_path, format="ffmpeg") as reader:
        for _ in reader:
            count += 1
    return count


class TestMp4FrameReplication:
    """Story 3: MP4 CFR 프레임 복제/드롭 통합 테스트.

    encode_frames(fmt='mp4', segments=...) 계약 검증:
      - segments=() → 기존 경로(crops 그대로)와 동일 프레임 수
      - 슬로우 구간 → 프레임 복제 → MP4 재로드 시 총 프레임 수 증가
      - 패스트 구간 → 프레임 드롭 → 총 프레임 수 감소
      - 단일 패스트 가드 → MP4 프레임 수 ≥ 1 (빈 영상 금지)

    TDD RED 예상: 현재 encode_frames MP4 경로는 config.segments를 무시하고
    crops를 그대로 _encode_mp4에 전달한다(video_export.py 89번줄).
    Story 3 구현(segments 분기 + schedule_to_cfr_indices 호출) 전까지
    슬로우/패스트 프레임 수 검증 테스트가 실패한다.
    """

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(_NO_FFMPEG, reason=_MSG_NO_FFMPEG)
    def test_MP4_segments_빈_튜플이면_기존_경로와_동일_프레임_수다(self, tmp_path):
        """Given: 10개 크롭, segments=(), fps=12, fmt='mp4'
        When:  encode_frames 호출 후 MP4 재로드
        Then:  총 프레임 수 == 10 (crops 그대로 전달, 복제/드롭 없음)

        WHY (무회귀): segments=() MP4 경로는 기존 _encode_mp4(crops) 호출과
             완전히 동일해야 한다. Story 3 구현이 기존 MP4 테스트를 깨면 안 된다.
        """
        # Given
        crops = _make_synth_crops_s3(_S3_N_FRAMES)
        config = VideoExportConfig(fmt="mp4", fps=_S3_FPS, segments=())
        output_path = str(tmp_path / "mp4_no_segments.mp4")

        # When
        encode_frames(crops, output_path, config)

        # Then
        actual_count = _count_mp4_frames(output_path)
        assert actual_count == _S3_N_FRAMES, (
            f"segments=() MP4 프레임 수 불일치: {actual_count} vs {_S3_N_FRAMES}. "
            "segments=() 경로는 crops 그대로 전달해야 함(무회귀)."
        )

    @pytest.mark.skipif(_STORY3_SKIP, reason=_MSG_STORY3_NOT_IMPL)
    @pytest.mark.skipif(_NO_FFMPEG, reason=_MSG_NO_FFMPEG)
    def test_MP4_슬로우_구간_프레임이_복제되어_총_프레임_수가_증가한다(self, tmp_path):
        """Given: 10개 크롭, segments=(SpeedSegment(2, 5, 0.5),), fps=12, fmt='mp4'
        When:  encode_frames 호출 후 imageio로 MP4 재로드
        Then:  총 프레임 수 ≈ 13 (±1)
               구간 밖 7프레임(0,1,5~9) + 슬로우 구간 3프레임 × 2배 ≈ 6 = 13

        WHY 복제 검증:
          슬로우 factor=0.5 → duration이 기준의 2배 → schedule_to_cfr_indices가
          해당 프레임 인덱스를 2회 반복 삽입 → _encode_mp4에 복제 프레임 전달.
          MP4 CFR이므로 duration이 아니라 프레임 수로 속도를 표현한다.

        WHY ±1 허용:
          Bresenham 누적기 방식은 정수 반올림 오차로 ±1 차이가 발생할 수 있다.
        """
        # Given
        crops = _make_synth_crops_s3(_S3_N_FRAMES)
        slow_seg = SpeedSegment(_S3_SLOW_START, _S3_SLOW_END, _S3_SLOW_FACTOR)
        config = VideoExportConfig(fmt="mp4", fps=_S3_FPS, segments=(slow_seg,))
        output_path = str(tmp_path / "mp4_slow.mp4")

        # 기대 프레임 수 계산 (매직넘버 금지 — 상수 조합)
        # 구간 밖 프레임 수: _S3_N_FRAMES - (_S3_SLOW_END - _S3_SLOW_START) = 10 - 3 = 7
        # 슬로우 구간 프레임 수: (end - start) / factor = 3 / 0.5 = 6
        _N_OUTSIDE = _S3_N_FRAMES - (_S3_SLOW_END - _S3_SLOW_START)
        _N_SLOW_EXPECTED = round((_S3_SLOW_END - _S3_SLOW_START) / _S3_SLOW_FACTOR)
        _EXPECTED_TOTAL = _N_OUTSIDE + _N_SLOW_EXPECTED  # 7 + 6 = 13

        # When
        encode_frames(crops, output_path, config)

        # Then
        actual_count = _count_mp4_frames(output_path)
        assert abs(actual_count - _EXPECTED_TOTAL) <= _S3_FRAME_COUNT_TOLERANCE, (
            f"슬로우 MP4 프레임 수 불일치: {actual_count} vs 기대 {_EXPECTED_TOTAL} "
            f"(±{_S3_FRAME_COUNT_TOLERANCE}). "
            f"segments=({_S3_SLOW_START},{_S3_SLOW_END},{_S3_SLOW_FACTOR}) 복제 미적용 의심."
        )

    @pytest.mark.skipif(_STORY3_SKIP, reason=_MSG_STORY3_NOT_IMPL)
    @pytest.mark.skipif(_NO_FFMPEG, reason=_MSG_NO_FFMPEG)
    def test_MP4_패스트_구간_프레임이_드롭되어_총_프레임_수가_감소한다(self, tmp_path):
        """Given: 10개 크롭, segments=(SpeedSegment(2, 6, 2.0),), fps=12, fmt='mp4'
        When:  encode_frames 호출 후 MP4 재로드
        Then:  총 프레임 수 < 10 (패스트 구간 4프레임 → ≈ 2프레임으로 드롭)
               기대 ≈ 8: 구간 밖 6 + 구간 4×0.5=2 = 8

        WHY 드롭 검증:
          패스트 factor=2.0 → duration이 기준의 0.5배 → schedule_to_cfr_indices가
          ratio=0.5로 2프레임 중 1프레임만 삽입 → _encode_mp4에 드롭된 시퀀스 전달.
          전체 프레임 수가 원본 10보다 줄어들면 드롭이 적용된 것이다.
        """
        # Given
        crops = _make_synth_crops_s3(_S3_N_FRAMES)
        fast_seg = SpeedSegment(_S3_FAST_START, _S3_FAST_END, _S3_FAST_FACTOR)
        config = VideoExportConfig(fmt="mp4", fps=_S3_FPS, segments=(fast_seg,))
        output_path = str(tmp_path / "mp4_fast.mp4")

        # 기대 프레임 수 계산
        # 구간 밖: _S3_N_FRAMES - (_S3_FAST_END - _S3_FAST_START) = 10 - 4 = 6
        # 패스트 구간: (end - start) × (1/factor) = 4 × 0.5 = 2
        _N_OUTSIDE_FAST = _S3_N_FRAMES - (_S3_FAST_END - _S3_FAST_START)
        _N_FAST_EXPECTED = round((_S3_FAST_END - _S3_FAST_START) / _S3_FAST_FACTOR)
        _EXPECTED_FAST_TOTAL = _N_OUTSIDE_FAST + _N_FAST_EXPECTED  # 6 + 2 = 8

        # When
        encode_frames(crops, output_path, config)

        # Then
        actual_count = _count_mp4_frames(output_path)
        assert actual_count < _S3_N_FRAMES, (
            f"패스트 드롭 미적용: actual={actual_count} >= {_S3_N_FRAMES}(원본). "
            f"segments=({_S3_FAST_START},{_S3_FAST_END},{_S3_FAST_FACTOR}) 드롭 경로 미구현 의심."
        )
        assert abs(actual_count - _EXPECTED_FAST_TOTAL) <= _S3_FRAME_COUNT_TOLERANCE, (
            f"패스트 MP4 프레임 수 불일치: {actual_count} vs 기대 {_EXPECTED_FAST_TOTAL} "
            f"(±{_S3_FRAME_COUNT_TOLERANCE})."
        )

    @pytest.mark.skipif(_STORY3_SKIP, reason=_MSG_STORY3_NOT_IMPL)
    @pytest.mark.skipif(_NO_FFMPEG, reason=_MSG_NO_FFMPEG)
    def test_MP4_단일_패스트_가드_프레임_수가_최소_1이다(self, tmp_path):
        """Given: 3개 크롭 전체 패스트(factor=2.0), fmt='mp4'
        When:  encode_frames 호출 후 MP4 재로드
        Then:  총 프레임 수 >= 1 (빈 영상 금지)

        WHY [중요]1 가드 실검증:
          schedule_to_cfr_indices는 패스트 구간에서 ratio < 1.0이면 누적기가
          1.0에 못 미쳐 일부 프레임이 드롭될 수 있다. 모든 프레임이 드롭되어
          빈 시퀀스가 _encode_mp4에 전달되면 파일 크기=0 또는 코덱 오류가 된다.
          최소 1프레임 보장(잔여 보장)이 encode_frames MP4 경로에서 동작하는지
          실제 파일 생성으로 검증한다.

        WHY 3프레임:
          factor=2.0 이면 ratio=0.5, 3프레임×0.5=1.5 → 정수 드롭 시 1프레임.
          Bresenham 잔여 보장이 없으면 0 또는 1 경계에서 비결정적이 된다.
          3프레임은 이 경계를 명확히 테스트하는 최소 크기다.

        TDD 상태 노트:
          RED 단계에서는 segments 무시 → 3프레임 그대로 전달 → ≥1 통과.
          Story 3 구현 후(GREEN)에는 Bresenham 잔여 보장이 동작해 동일 통과.
          이 테스트는 구현 전/후 모두 PASS여야 하는 불변식 가드다.
        """
        # Given
        crops = _make_synth_crops_s3(_S3_GUARD_N_FRAMES)
        guard_seg = SpeedSegment(
            _S3_GUARD_FAST_START,
            _S3_GUARD_N_FRAMES,   # 전체 구간 패스트
            _S3_GUARD_FAST_FACTOR,
        )
        config = VideoExportConfig(fmt="mp4", fps=_S3_FPS, segments=(guard_seg,))
        output_path = str(tmp_path / "mp4_guard.mp4")

        # When
        encode_frames(crops, output_path, config)

        # Then — 파일 존재 + 프레임 수 ≥ 1
        assert (tmp_path / "mp4_guard.mp4").exists(), (
            "단일 패스트 가드: MP4 파일이 생성되지 않음 — encode_frames가 빈 시퀀스를 전달했을 수 있음."
        )
        actual_count = _count_mp4_frames(output_path)
        assert actual_count >= 1, (
            f"단일 패스트 가드 실패: MP4 프레임 수 = {actual_count} < 1. "
            "schedule_to_cfr_indices 잔여 보장([중요]1)이 encode_frames MP4 경로에서 "
            "동작하지 않음. 빈 영상 생성 위험."
        )

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    @pytest.mark.skipif(_NO_FFMPEG, reason=_MSG_NO_FFMPEG)
    def test_MP4_segments_없는_기존_테스트_무회귀(self, tmp_path):
        """Given: 6개 크롭(기존 TestMp4Roundtrip과 동일 조건), segments 미지정
        When:  encode_frames(fmt='mp4', fps=24) 호출
        Then:  파일 존재 + 크기 > 0 (기존 계약 유지)

        WHY 무회귀 명시:
          Story 3 구현 후 segments=() MP4 경로가 기존 동작을 그대로 유지하는지
          TestMp4Roundtrip과 독립적으로 재검증한다.
          TestMp4Roundtrip.test_MP4_encode_후_파일이_생성된다와 동일 시나리오를
          Story 3 클래스 내에서 반복 확인해 회귀를 이중 가드한다.
        """
        # Given
        crops = crop_frames(_make_synth_frames(), _make_crop_boxes())
        config = VideoExportConfig(fmt="mp4", fps=MP4_FPS)  # segments 미지정 = ()
        output_path = str(tmp_path / "mp4_regression.mp4")

        # When
        encode_frames(crops, output_path, config)

        # Then
        assert (tmp_path / "mp4_regression.mp4").exists(), "무회귀 MP4 파일 미생성"
        assert (tmp_path / "mp4_regression.mp4").stat().st_size > 0, (
            "무회귀 MP4 파일 크기 = 0"
        )


# ---------------------------------------------------------------------------
# Story 4 — 트림 + 루프 통합 (TDD RED)
# ---------------------------------------------------------------------------
# Story 4 테스트 상수 — 매직넘버 금지
# 좌표계: trim·segments 모두 span 상대 [0, n) 단일 (planner 설계 확정안).
# 적용 순서: 트림 먼저 → segments 트림-로컬 평행이동/클리핑 → 스케줄 빌드.
_S4_N_FRAMES = 6                 # 합성 크롭 총 프레임 수
_S4_FPS = 12.0                   # 출력 fps
_S4_TRIM_START = 2               # 트림 시작 인덱스 (포함)
_S4_TRIM_END = 5                 # 트림 끝 인덱스 (미포함) → 길이 3 (프레임 2,3,4)
_S4_TRIM_LEN = 3                 # 트림 길이 (M = end - start)
_S4_FRAME_COUNT_TOLERANCE = 1    # MP4 Bresenham ±1 허용 오차

# 트림+슬로우 결합 상수
_S4_COMBO_TRIM_START = 0         # 트림 시작 (전체 앞부분)
_S4_COMBO_TRIM_END = 4           # 트림 끝 (미포함) → 길이 4 (프레임 0,1,2,3)
_S4_COMBO_TRIM_LEN = 4
_S4_COMBO_SEG_START = 0          # 트림-로컬 슬로우 구간 시작
_S4_COMBO_SEG_END = 2            # 트림-로컬 슬로우 구간 끝 (미포함) → 로컬 0,1
_S4_COMBO_SLOW_FACTOR = 0.5      # 슬로우 → 구간 2프레임 ×2 ≈ 4프레임

# 루프 상수
_S4_LOOP_COUNT_3 = 3             # 유한 루프 3회
_S4_LOOP_COUNT_INFINITE = 0      # 무한 루프 (GIF loop=0)


def _read_gif_loop(gif_path: str) -> int:
    """PIL로 GIF의 loop 메타값을 읽어 반환한다(0=무한).

    WHY PIL info['loop']:
      GIF89a NETSCAPE2.0 확장 블록의 루프 카운트를 PIL이 info['loop']로 노출한다.
      loop=0은 무한 반복, loop=N은 N회 반복(표준 GIF 스펙).
      imageio get_reader meta_data의 'loop'보다 PIL info가 더 일관되게 노출된다.
    """
    from PIL import Image

    with Image.open(gif_path) as img:
        return img.info.get("loop", 0)


def _make_s4_crops(n: int = _S4_N_FRAMES) -> list[np.ndarray]:
    """Story 4용 n개 합성 크롭(32×24 RGB)을 반환한다."""
    return _make_synth_frames(n=n, h=CROP_BOX_H, w=CROP_BOX_W)


# Story 4 guard: video_export + TrimRange(timeremap) 모두 존재해야 실행
_STORY4_SKIP = not (_HAS_VIDEO_EXPORT and _HAS_TRIM)


class TestVideoExportConfigTrimLoop:
    """Story 4: VideoExportConfig.trim/loop_count 필드 계약 검증.

    신규 필드:
      - trim: TrimRange | None = None
      - loop_count: int = 0
    기존 필드(fmt/fps/gap_policy/segments) 순서·기본값 불변(무회귀).
    """

    @pytest.mark.skipif(_STORY4_SKIP, reason=_MSG_TRIM_NOT_IMPL)
    def test_trim_기본값이_None이다(self):
        """Given: 인자 없이 VideoExportConfig 생성
        When:  trim 필드 접근
        Then:  trim is None (기본값 = 트림 안 함).

        WHY: trim=None은 "트림 안 함" 무회귀 경로. 기본 동작 보존 핵심 조건.
        """
        config = VideoExportConfig()

        assert hasattr(config, "trim"), (
            "VideoExportConfig에 trim 필드가 없음 — Story 4 미구현"
        )
        assert config.trim is None, (
            f"trim 기본값 불일치: {config.trim!r} vs None"
        )

    @pytest.mark.skipif(_STORY4_SKIP, reason=_MSG_TRIM_NOT_IMPL)
    def test_loop_count_기본값이_0이다(self):
        """Given: 인자 없이 VideoExportConfig 생성
        When:  loop_count 필드 접근
        Then:  loop_count == 0 (기본값 = GIF 무한 루프).

        WHY: loop_count=0은 GIF 무한 반복(기존 _encode_gif loop=0 계약).
             기본값이 0이어야 기존 무회귀 동작과 동일하다.
        """
        config = VideoExportConfig()

        assert hasattr(config, "loop_count"), (
            "VideoExportConfig에 loop_count 필드가 없음 — Story 4 미구현"
        )
        assert config.loop_count == 0, (
            f"loop_count 기본값 불일치: {config.loop_count!r} vs 0"
        )

    @pytest.mark.skipif(not _HAS_VIDEO_EXPORT, reason=_MSG_NOT_IMPL)
    def test_기존_필드_기본값이_불변이다_무회귀(self):
        """Given: 인자 없이 VideoExportConfig 생성
        When:  기존 필드(fmt/fps/gap_policy/segments) 접근
        Then:  fmt='gif', fps=12.0, gap_policy=BACKGROUND, segments=() 불변.

        WHY (무회귀): trim/loop_count 추가가 기존 필드 순서·기본값을 바꾸면
             기존 300+ 테스트와 호출부가 깨진다. 신규 필드는 끝에 추가되어야 한다.
        """
        config = VideoExportConfig()

        assert config.fmt == "gif"
        assert config.fps == 12.0
        assert config.gap_policy == GapPolicy.BACKGROUND
        assert config.segments == ()

    @pytest.mark.skipif(_STORY4_SKIP, reason=_MSG_TRIM_NOT_IMPL)
    def test_trim_loop_지정값이_보관된다(self):
        """Given: trim=TrimRange(2,5), loop_count=3
        When:  VideoExportConfig 생성
        Then:  필드값 그대로 반환.
        """
        trim = TrimRange(_S4_TRIM_START, _S4_TRIM_END)
        config = VideoExportConfig(trim=trim, loop_count=_S4_LOOP_COUNT_3)

        assert config.trim == trim
        assert config.loop_count == _S4_LOOP_COUNT_3

    @pytest.mark.skipif(_STORY4_SKIP, reason=_MSG_TRIM_NOT_IMPL)
    def test_trim_loop_필드가_frozen_불변이다(self):
        """Given: trim/loop_count 지정 VideoExportConfig
        When:  loop_count 필드 수정 시도
        Then:  FrozenInstanceError(AttributeError/TypeError) 발생.

        WHY: VideoExportConfig는 frozen=True이므로 신규 필드도 불변이어야 한다.
        """
        config = VideoExportConfig(loop_count=_S4_LOOP_COUNT_3)

        with pytest.raises((AttributeError, TypeError)):
            config.loop_count = 0  # type: ignore[misc]


class TestTrimGifMp4FrameCount:
    """Story 4: 트림 지정 시 출력 프레임 수가 트림 길이 기반인지 검증.

    trim=TrimRange(2,5) + segments=() → 출력 프레임 3개(트림 길이).
    """

    @pytest.mark.skipif(_STORY4_SKIP, reason=_MSG_TRIM_NOT_IMPL)
    def test_무회귀_trim_None_loop_0이면_기존_프레임_수가_유지된다(self, tmp_path):
        """Given: 6개 크롭, trim=None, loop_count=0, segments=()
        When:  encode_frames(fmt='gif') 후 GIF 재로드
        Then:  출력 프레임 수 == 6 (기존 균일 경로와 동일).

        WHY (무회귀): trim=None and loop_count=0이면 기존 동작과 완전히 동일해야 한다.
             기존 GIF 라운드트립 테스트(6프레임)가 깨지지 않음을 명시적으로 가드한다.
        """
        crops = _make_s4_crops(_S4_N_FRAMES)
        config = VideoExportConfig(
            fmt="gif", fps=_S4_FPS, trim=None, loop_count=0
        )
        output_path = str(tmp_path / "trim_none_regression.gif")

        encode_frames(crops, output_path, config)

        reloaded = imageio.mimread(output_path)
        assert len(reloaded) == _S4_N_FRAMES, (
            f"trim=None 무회귀 프레임 수 불일치: {len(reloaded)} vs {_S4_N_FRAMES}. "
            "trim=None and loop_count=0은 기존 경로와 동일해야 함."
        )

    @pytest.mark.skipif(_STORY4_SKIP, reason=_MSG_TRIM_NOT_IMPL)
    def test_트림_GIF_출력_프레임_수가_트림_길이와_같다(self, tmp_path):
        """Given: 6개 크롭, trim=TrimRange(2,5), segments=()
        When:  encode_frames(fmt='gif') 후 GIF 재로드
        Then:  출력 프레임 수 == 3 (트림 길이, 프레임 2,3,4만).

        WHY: 트림은 crops를 [trim.start, trim.end)로 슬라이스한다.
             segments=()이므로 복제/드롭 없이 트림 길이 그대로 출력된다.
        """
        crops = _make_s4_crops(_S4_N_FRAMES)
        trim = TrimRange(_S4_TRIM_START, _S4_TRIM_END)
        config = VideoExportConfig(fmt="gif", fps=_S4_FPS, trim=trim)
        output_path = str(tmp_path / "trim.gif")

        encode_frames(crops, output_path, config)

        reloaded = imageio.mimread(output_path)
        assert len(reloaded) == _S4_TRIM_LEN, (
            f"트림 GIF 프레임 수 불일치: {len(reloaded)} vs {_S4_TRIM_LEN}. "
            f"trim=({_S4_TRIM_START},{_S4_TRIM_END}) 슬라이스 미적용 의심."
        )

    @pytest.mark.skipif(_STORY4_SKIP, reason=_MSG_TRIM_NOT_IMPL)
    @pytest.mark.skipif(_NO_FFMPEG, reason=_MSG_NO_FFMPEG)
    def test_트림_MP4_라운드트립_프레임_수가_트림_길이와_같다(self, tmp_path):
        """Given: 6개 크롭, trim=TrimRange(2,5), segments=(), fmt='mp4'
        When:  encode_frames 후 MP4 재로드
        Then:  총 프레임 수 == 3 (트림 길이).

        WHY: MP4 경로도 트림 슬라이스가 동일하게 적용되어야 한다.
             segments=()이므로 복제/드롭 없이 트림 길이 그대로다.
        """
        crops = _make_s4_crops(_S4_N_FRAMES)
        trim = TrimRange(_S4_TRIM_START, _S4_TRIM_END)
        config = VideoExportConfig(fmt="mp4", fps=_S4_FPS, trim=trim)
        output_path = str(tmp_path / "trim.mp4")

        encode_frames(crops, output_path, config)

        actual_count = _count_mp4_frames(output_path)
        assert actual_count == _S4_TRIM_LEN, (
            f"트림 MP4 프레임 수 불일치: {actual_count} vs {_S4_TRIM_LEN}. "
            f"trim=({_S4_TRIM_START},{_S4_TRIM_END}) 슬라이스 미적용 의심."
        )

    @pytest.mark.skipif(_STORY4_SKIP, reason=_MSG_TRIM_NOT_IMPL)
    @pytest.mark.skipif(_NO_FFMPEG, reason=_MSG_NO_FFMPEG)
    def test_트림_후_트림로컬_슬로우_segments가_MP4_복제로_반영된다(self, tmp_path):
        """Given: 6개 크롭, trim=TrimRange(0,4), segments=(SpeedSegment(0,2,0.5),), mp4
        When:  encode_frames 후 MP4 재로드
        Then:  총 프레임 수 ≈ 6 (±1): 트림 길이 4 → 슬로우 구간 [0,2) 2프레임 ×2=4,
               구간 밖 [2,4) 2프레임 = 2 → 4+2 = 6.

        WHY: 적용 순서 검증 — 트림 먼저(길이 4) → 트림-로컬 segments(0,2,0.5) 적용.
             segments가 트림-로컬 좌표 [0, M)로 해석되어 트림된 프레임에 복제 반영.
        """
        crops = _make_s4_crops(_S4_N_FRAMES)
        trim = TrimRange(_S4_COMBO_TRIM_START, _S4_COMBO_TRIM_END)
        seg = SpeedSegment(
            _S4_COMBO_SEG_START, _S4_COMBO_SEG_END, _S4_COMBO_SLOW_FACTOR
        )
        config = VideoExportConfig(
            fmt="mp4", fps=_S4_FPS, trim=trim, segments=(seg,)
        )
        output_path = str(tmp_path / "trim_slow.mp4")

        # 기대 프레임 수 계산 (매직넘버 금지 — 상수 조합)
        _n_slow = round(
            (_S4_COMBO_SEG_END - _S4_COMBO_SEG_START) / _S4_COMBO_SLOW_FACTOR
        )  # 2/0.5 = 4
        _n_outside = _S4_COMBO_TRIM_LEN - (_S4_COMBO_SEG_END - _S4_COMBO_SEG_START)
        _expected_total = _n_slow + _n_outside  # 4 + 2 = 6

        encode_frames(crops, output_path, config)

        actual_count = _count_mp4_frames(output_path)
        assert abs(actual_count - _expected_total) <= _S4_FRAME_COUNT_TOLERANCE, (
            f"트림+슬로우 MP4 프레임 수 불일치: {actual_count} vs 기대 "
            f"{_expected_total} (±{_S4_FRAME_COUNT_TOLERANCE}). "
            "트림 후 트림-로컬 segments 복제 미적용 의심."
        )

    @pytest.mark.skipif(_STORY4_SKIP, reason=_MSG_TRIM_NOT_IMPL)
    def test_트림_범위_오류_시_encode_frames가_ValueError를_던진다(self, tmp_path):
        """Given: 6개 크롭, trim=TrimRange(2,2) (start==end, 빈 구간)
        When:  encode_frames(fmt='gif') 호출
        Then:  ValueError 발생 (validate_trim 전파).

        WHY: encode_frames가 트림 적용 전 validate_trim을 호출하고,
             빈/범위초과 트림의 ValueError를 흡수하지 않고 그대로 전파해야 한다.
        """
        crops = _make_s4_crops(_S4_N_FRAMES)
        bad_trim = TrimRange(_S4_TRIM_START, _S4_TRIM_START)  # (2, 2) 빈 구간
        config = VideoExportConfig(fmt="gif", fps=_S4_FPS, trim=bad_trim)
        output_path = str(tmp_path / "bad_trim.gif")

        with pytest.raises(ValueError):
            encode_frames(crops, output_path, config)

    @pytest.mark.skipif(_STORY4_SKIP, reason=_MSG_TRIM_NOT_IMPL)
    def test_트림_범위_초과_시_encode_frames가_ValueError를_던진다(self, tmp_path):
        """Given: 6개 크롭, trim=TrimRange(2, 100) (end > n_frames)
        When:  encode_frames(fmt='gif') 호출
        Then:  ValueError 발생 (validate_trim 전파).

        WHY: 트림 end가 선택된 프레임 수를 초과하면 슬라이스가 잘못된 결과를 낸다.
             validate_trim이 n_frames 기준으로 검증해 ValueError를 전파해야 한다.
        """
        crops = _make_s4_crops(_S4_N_FRAMES)
        over_trim = TrimRange(_S4_TRIM_START, _S4_N_FRAMES + 100)
        config = VideoExportConfig(fmt="gif", fps=_S4_FPS, trim=over_trim)
        output_path = str(tmp_path / "over_trim.gif")

        with pytest.raises(ValueError):
            encode_frames(crops, output_path, config)


class TestGifLoopCount:
    """Story 4: GIF loop_count가 메타데이터로 라운드트립되는지 검증.

    loop_count=N → GIF loop=N, loop_count=0 → loop=0(무한).
    MP4는 loop_count 무시.
    """

    @pytest.mark.skipif(_STORY4_SKIP, reason=_MSG_TRIM_NOT_IMPL)
    def test_loop_count_3이면_GIF_메타_loop가_3이다(self, tmp_path):
        """Given: 6개 크롭, loop_count=3, fmt='gif'
        When:  encode_frames 후 PIL로 loop 메타 읽기
        Then:  loop == 3 (유한 3회 반복).

        WHY: loop_count가 GIF NETSCAPE2.0 루프 카운트로 전달되어야 한다.
             encode_frames → _encode_gif loop= 인자에 config.loop_count 반영.
        """
        crops = _make_s4_crops(_S4_N_FRAMES)
        config = VideoExportConfig(
            fmt="gif", fps=_S4_FPS, loop_count=_S4_LOOP_COUNT_3
        )
        output_path = str(tmp_path / "loop3.gif")

        encode_frames(crops, output_path, config)

        loop_value = _read_gif_loop(output_path)
        assert loop_value == _S4_LOOP_COUNT_3, (
            f"GIF loop 메타 불일치: {loop_value} vs {_S4_LOOP_COUNT_3}. "
            "loop_count가 GIF loop= 인자로 전달되지 않음."
        )

    @pytest.mark.skipif(_STORY4_SKIP, reason=_MSG_TRIM_NOT_IMPL)
    def test_loop_count_0이면_GIF_메타_loop가_0_무한이다(self, tmp_path):
        """Given: 6개 크롭, loop_count=0(기본), fmt='gif'
        When:  encode_frames 후 PIL로 loop 메타 읽기
        Then:  loop == 0 (무한 반복).

        WHY: loop_count=0은 기존 무한 루프 계약과 동일해야 한다(무회귀).
        """
        crops = _make_s4_crops(_S4_N_FRAMES)
        config = VideoExportConfig(
            fmt="gif", fps=_S4_FPS, loop_count=_S4_LOOP_COUNT_INFINITE
        )
        output_path = str(tmp_path / "loop_infinite.gif")

        encode_frames(crops, output_path, config)

        loop_value = _read_gif_loop(output_path)
        assert loop_value == _S4_LOOP_COUNT_INFINITE, (
            f"GIF loop 메타 불일치: {loop_value} vs 0(무한). "
            "loop_count=0은 무한 루프여야 함."
        )

    @pytest.mark.skipif(_STORY4_SKIP, reason=_MSG_TRIM_NOT_IMPL)
    def test_loop_count_음수이면_ValueError가_발생한다(self, tmp_path):
        """Given: 6개 크롭, loop_count=-1 (무효값)
        When:  encode_frames 호출
        Then:  ValueError 발생 — 한국어 메시지에 음수 값 포함.

        WHY: loop_count=-1은 GIF 스펙에 존재하지 않는 무효값이다.
             validate_trim과 대칭으로 인코딩 진입 전 조기 차단해
             imageio에 음수가 전달되어 undefined behavior가 발생하는 것을 막는다.
             reviewer [중요 2] 가드 요건.
        """
        crops = _make_s4_crops(_S4_N_FRAMES)
        config = VideoExportConfig(
            fmt="gif", fps=_S4_FPS, loop_count=-1
        )
        output_path = str(tmp_path / "bad_loop.gif")

        with pytest.raises(ValueError, match="반복|loop_count|-1"):
            encode_frames(crops, output_path, config)
