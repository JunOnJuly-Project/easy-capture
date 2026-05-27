"""갭 채우기 정책 테스트."""
from easy_capture.core.tracking import GapPolicy, build_output_indices

FLAGS = [True, True, False, False, True]


def test_cut_excludes_gap():
    assert build_output_indices(FLAGS, GapPolicy.CUT) == [0, 1, 4]


def test_background_keeps_all():
    assert build_output_indices(FLAGS, GapPolicy.BACKGROUND) == [0, 1, 2, 3, 4]


def test_freeze_repeats_last_valid():
    assert build_output_indices(FLAGS, GapPolicy.FREEZE) == [0, 1, 1, 1, 4]


def test_freeze_drops_leading_gap():
    assert build_output_indices([False, True], GapPolicy.FREEZE) == [1]
