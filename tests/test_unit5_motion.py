import cv2
import numpy as np
import pytest

from src import config, utils
from src import motion_estimation as me


def test_validate_video_reads_properties(synthetic_video):
    info = me.validate_video(synthetic_video)
    assert info["frame_count"] == 40
    assert info["width"] == 160 and info["height"] == 120
    assert abs(info["fps"] - 10.0) < 0.5
    assert any("outside" in w for w in info["warnings"])  # 4 s is shorter than 15-30 s


def test_missing_video_raises_input_missing(tmp_path):
    with pytest.raises(utils.InputMissingError):
        me.validate_video(tmp_path / "nope.mp4")


def test_single_frame_video_is_rejected(tmp_path):
    path = tmp_path / "one_frame.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    with pytest.raises(ValueError):
        me.validate_video(path)


def test_frame_difference_marks_moving_square():
    prev = np.zeros((60, 80), dtype=np.uint8)
    cur = prev.copy()
    cur[20:40, 30:50] = 255
    _, mask = me.frame_difference(prev, cur, threshold=25, blur=3)
    assert mask[30, 40] == 255 and mask[5, 5] == 0


def test_lucas_kanade_measures_shift():
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.integers(0, 255, (120, 160), dtype=np.uint8), (7, 7), 0)
    shifted = np.roll(base, 4, axis=1)  # whole image moves 4 px to the right
    params = config.MOTION
    points = me.detect_corners(base, params)
    old, new = me.lucas_kanade(base, shifted, points, params)
    dx = np.median(new[:, 0] - old[:, 0])
    assert abs(dx - 4.0) < 0.5


def test_full_pipeline_on_synthetic_clip(synthetic_video, tmp_path):
    params = {**config.MOTION, "expected_duration": (0, 100)}
    result = me.process_video(synthetic_video, tmp_path / "out", params)
    assert result["processed_frames"] == 40
    lk = result["summary"][2]
    assert 1.5 < lk["mean_moving_speed_px_per_frame"] < 5.0  # square moves 3 px/frame
    assert result["summary"][0]["mean_foreground_pct"] > 0
    for name in ("per_frame_motion_statistics.csv", "motion_comparison.csv", "fig_motion_timeseries.png",
                 "fig_motion_snapshots.png"):
        assert (tmp_path / "out" / name).is_file()
    assert any(f.endswith((".mp4", ".avi")) for f in result["files"])
