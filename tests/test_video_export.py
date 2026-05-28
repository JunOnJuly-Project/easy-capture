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

_MSG_NOT_IMPL = "easy_capture.core.export.video_export 미구현 — RED 예상"

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
