"""crop 기하 로직 테스트."""
import numpy as np

from easy_capture.core.crop import (apply_aspect_lock, centroid_of_mask,
                                    make_crop_box, smooth_centroids, to_even)


def test_centroid_of_mask():
    m = np.zeros((10, 10), bool)
    m[2:4, 4:6] = True
    assert centroid_of_mask(m) == (4.5, 2.5)


def test_centroid_empty_returns_none():
    assert centroid_of_mask(np.zeros((5, 5), bool)) is None


def test_smooth_reduces_jitter():
    pts = [(0, 0), (10, 0), (0, 0), (10, 0), (0, 0)]
    out = smooth_centroids(pts, window=5)
    assert 2 < out[-1][0] < 8  # 진폭(0~10)보다 작아짐


def test_smooth_holds_none_forward():
    out = smooth_centroids([(5, 5), None, None], window=1)
    assert out[1] == (5, 5) and out[2] == (5, 5)


def test_to_even():
    assert to_even(101) == 100
    assert to_even(100) == 100


def test_aspect_lock_vertical_shrinks_width():
    assert apply_aspect_lock(200, 200, "9:16") == (112, 200)


def test_aspect_lock_none_passthrough():
    assert apply_aspect_lock(123, 77, None) == (123, 77)


def test_make_crop_box_even_and_within_frame():
    x1, y1, x2, y2 = make_crop_box((1000, 1000), (200, 200), (640, 360))
    assert (x2 - x1) % 2 == 0 and (y2 - y1) % 2 == 0
    assert 0 <= x1 and x2 <= 640 and 0 <= y1 and y2 <= 360
