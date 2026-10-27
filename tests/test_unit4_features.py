import numpy as np
import pytest

from src import feature_extraction as fe
from src import utils


def test_hog_length_matches_formula(astronaut_bgr):
    gray = utils.to_gray(astronaut_bgr)[:256, :192]
    features, view = fe.compute_hog(gray, 9, 16, 2)
    assert features.size == fe.expected_hog_length(256, 192, 9, 16, 2)
    assert view.shape == gray.shape and view.dtype == np.uint8


def test_sift_descriptors_have_128_values(astronaut_bgr):
    keypoints, descriptors = fe.detect_and_describe(astronaut_bgr)
    assert len(keypoints) > 100
    assert descriptors.shape == (len(keypoints), 128)


@pytest.mark.parametrize("angle", [30, 90])
def test_matching_survives_rotation(astronaut_bgr, angle):
    features = fe.detect_and_describe(astronaut_bgr)
    rotated, transform = fe.rotate_image(astronaut_bgr, angle)
    stats, _ = fe.match_experiment("test", astronaut_bgr, features, rotated, 0.75, "homography", 5.0, transform, 3.0)
    assert stats["ratio_test_matches"] > 50
    assert stats["correct_matches_pct"] > 80


def test_resize_transform_maps_points_correctly(astronaut_bgr):
    features = fe.detect_and_describe(astronaut_bgr)
    small, transform = fe.resize_image(astronaut_bgr, 0.5)
    assert small.shape[:2] == (256, 256)
    stats, _ = fe.match_experiment("test", astronaut_bgr, features, small, 0.75, "homography", 5.0, transform, 3.0)
    assert stats["correct_matches_pct"] > 70


def test_flann_and_bf_give_similar_counts(astronaut_bgr):
    _, d1 = fe.detect_and_describe(astronaut_bgr)
    _, d2 = fe.detect_and_describe(fe.adjust_gamma(astronaut_bgr, 0.7)[0])
    bf = len(fe.ratio_test_matches(d1, d2, 0.75, "bf"))
    flann = len(fe.ratio_test_matches(d1, d2, 0.75, "flann"))
    assert bf > 0 and abs(bf - flann) / bf < 0.15


def test_empty_descriptors_return_no_matches():
    assert fe.ratio_test_matches(None, None) == []
