import numpy as np
import pytest

from src import config, utils
from src import object_detection as od


def test_haar_cascades_load():
    assert not od.load_haar_cascade(config.DETECTION["haar_face_cascade"]).empty()
    assert not od.load_haar_cascade(config.DETECTION["haar_eye_cascade"]).empty()


def test_missing_cascade_raises_clear_error():
    with pytest.raises(utils.ModelUnavailableError):
        od.load_haar_cascade("does_not_exist.xml")


def test_haar_finds_face_in_public_sample(astronaut_bgr):
    cascade = od.load_haar_cascade(config.DETECTION["haar_face_cascade"])
    faces = od.detect_haar(utils.to_gray(astronaut_bgr), cascade, 1.1, 5, 0.08)
    assert len(faces) >= 1


def test_haar_finds_nothing_in_blank_image():
    cascade = od.load_haar_cascade(config.DETECTION["haar_face_cascade"])
    assert od.detect_haar(np.full((300, 300), 128, dtype=np.uint8), cascade) == []


def test_yolo_detects_person_if_weights_present(astronaut_bgr):
    weights = config.DEFAULT_MODELS_DIR / config.DETECTION["yolo_model"]
    if not weights.is_file():
        pytest.skip("YOLO weights not downloaded yet (run python main.py --unit 5 once)")
    model = od.load_yolo(config.DETECTION["yolo_model"], config.DEFAULT_MODELS_DIR)
    detections = od.detect_yolo(model, astronaut_bgr, 0.25)
    assert any(d["label"] == "person" and d["confidence"] > 0.5 for d in detections)
