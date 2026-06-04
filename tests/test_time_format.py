"""ui/time_format 순수 변환 테스트 (타임라인 mm:ss)."""
from __future__ import annotations

import pytest

from easy_capture.ui.time_format import frame_to_mm_ss, mm_ss_to_frame

_FPS = 30.0


class TestFrameToMmSs:
    """프레임 인덱스 → mm:ss 표시."""

    @pytest.mark.parametrize(
        "frame, expected",
        [
            (0, "00:00"),
            (90, "00:03"),     # 90/30 = 3s
            (1830, "01:01"),   # 1830/30 = 61s = 1:01
            (29, "00:00"),     # 29/30 = 0s (내림)
        ],
    )
    def test_변환(self, frame: int, expected: str):
        assert frame_to_mm_ss(frame, _FPS) == expected

    def test_fps_0이면_폴백(self):
        assert frame_to_mm_ss(90, 0) == "00:00"

    def test_음수_프레임이면_폴백(self):
        assert frame_to_mm_ss(-5, _FPS) == "00:00"


class TestMmSsToFrame:
    """mm:ss → 프레임 인덱스."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("0:03", 90),
            ("1:30", 2700),  # 90s * 30
            ("90", 2700),    # ss 단독 = 90s
            ("00:00", 0),
        ],
    )
    def test_변환(self, text: str, expected: int):
        assert mm_ss_to_frame(text, _FPS) == expected

    @pytest.mark.parametrize("bad", ["abc", "1:2:3", "1:xx", ""])
    def test_형식_오류는_ValueError(self, bad: str):
        with pytest.raises(ValueError):
            mm_ss_to_frame(bad, _FPS)

    def test_초_60이상은_ValueError(self):
        with pytest.raises(ValueError, match="범위"):
            mm_ss_to_frame("1:90", _FPS)

    def test_fps_0이하는_ValueError(self):
        with pytest.raises(ValueError, match="fps"):
            mm_ss_to_frame("1:00", 0)


class TestRoundTrip:
    """frame → mm:ss → frame 왕복(초 해상도 경계)."""

    def test_초_경계_왕복(self):
        # 정확히 초 경계인 프레임은 왕복 보존
        frame = 60  # 2s @ 30fps
        assert mm_ss_to_frame(frame_to_mm_ss(frame, _FPS), _FPS) == frame
