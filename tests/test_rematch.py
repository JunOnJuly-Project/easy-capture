"""샷 경계 재매칭 점수 테스트."""
from easy_capture.core.tracking import iou, rematch_score


def test_iou_identical():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_rematch_position_only():
    near = rematch_score((0, 0, 10, 10), (1, 1, 11, 11))
    far = rematch_score((0, 0, 10, 10), (50, 50, 60, 60))
    assert near > 0.5 > far


def test_rematch_with_appearance():
    score = rematch_score((0, 0, 10, 10), (0, 0, 10, 10), [1, 0], [1, 0])
    assert score > 0.9
