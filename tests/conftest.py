"""Shared test fixtures.

The tests never use the student's personal photo or video. They use
synthetic images, scikit-image's bundled public sample images, and a short
synthetic test clip generated in a temporary folder.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def astronaut_bgr():
    """Public-domain NASA portrait shipped with scikit-image (one person, one frontal face)."""
    from skimage import data

    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


@pytest.fixture
def smooth_image():
    """Synthetic colour image with smooth gradients and a few shapes."""
    h, w = 120, 160
    y, x = np.mgrid[0:h, 0:w]
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[..., 0] = (x * 255 / w).astype(np.uint8)
    image[..., 1] = (y * 255 / h).astype(np.uint8)
    image[..., 2] = 128
    cv2.circle(image, (50, 60), 25, (255, 255, 255), -1)
    cv2.rectangle(image, (100, 30), (140, 90), (0, 0, 0), -1)
    return image


@pytest.fixture
def synthetic_video(tmp_path):
    """SYNTHETIC test clip: a bright square moving right by 3 px/frame over a static texture."""
    rng = np.random.default_rng(0)
    background = cv2.GaussianBlur(rng.integers(0, 255, (120, 160), dtype=np.uint8), (5, 5), 0)
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    path = tmp_path / "synthetic_test_clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened(), "OpenCV could not open an mp4v writer"
    for i in range(40):
        frame = background.copy()
        x = 10 + 3 * i
        cv2.rectangle(frame, (x, 50), (x + 24, 74), (255, 255, 255), -1)
        cv2.rectangle(frame, (x + 6, 56), (x + 12, 62), (0, 0, 0), -1)  # corner texture to track
        writer.write(frame)
    writer.release()
    return path
