"""End-to-end checks of main.py using temporary folders (never the real inputs/outputs)."""

import cv2

import main
from src import utils


def test_missing_inputs_do_not_crash(tmp_path):
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    inputs.mkdir()
    code = main.main(["--all", "--inputs", str(inputs), "--outputs", str(outputs), "--skip-yolo"])
    assert code == 0
    summary = utils.load_json(outputs / "run_summary.json")
    assert {t["status"] for t in summary["tasks"].values()} == {"pending"}


def test_units_3_4_5_run_on_public_sample(tmp_path, astronaut_bgr):
    """Runs the real pipelines on scikit-image's public astronaut image (no personal data)."""
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    utils.write_image(inputs / "original_photo.png", astronaut_bgr)
    code = main.main(["--unit", "3", "--unit", "4", "--unit", "5", "--inputs", str(inputs),
                      "--outputs", str(outputs), "--skip-yolo"])
    assert code == 0
    tasks = utils.load_json(outputs / "run_summary.json")["tasks"]
    assert tasks["3A"]["status"] == "completed"
    assert tasks["3B"]["status"] == "completed"
    assert tasks["4"]["status"] == "partial"  # no second photo
    assert tasks["5A"]["status"] == "partial"  # YOLO skipped, no video
    assert tasks["5B"]["status"] == "pending"
    for path in ("unit3/restoration/restoration_metrics.csv", "unit3/segmentation/fig_watershed_objects.png",
                 "unit4/fig_rotation_scale_sweep.png", "unit5/detection/fig_rotation_robustness.png"):
        assert (outputs / path).is_file(), path
    # Only a video frame present -> it is used but labelled as a video frame, never as a photograph
    utils.write_image(inputs / "second_image_video_frame.png", cv2.GaussianBlur(astronaut_bgr, (5, 5), 0))
    assert main.main(["--unit", "4", "--inputs", str(inputs), "--outputs", str(outputs)]) == 0
    results = utils.load_json(outputs / "unit4" / "unit4_results.json")
    assert results["inputs"]["second_image_kind"] == "video frame"
    assert "NOT a separately taken photograph" in results["pair_matching"][0]["pair_type"]
    assert results["status"] == "partial" and any("second photograph" in p for p in results["pending"])
    # Second photo present -> the real two-photo branch runs
    second = cv2.GaussianBlur(astronaut_bgr, (3, 3), 0)
    utils.write_image(inputs / "second_photo.png", second)
    assert main.main(["--unit", "4", "--inputs", str(inputs), "--outputs", str(outputs)]) == 0
    results = utils.load_json(outputs / "unit4" / "unit4_results.json")
    assert results["status"] == "completed"
    assert results["pair_matching"][0]["pair_type"] == "real two-photo pair"
    assert len(results["transform_matching_second_photo"]) == 5
