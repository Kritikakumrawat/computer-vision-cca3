"""Central settings for the whole project.

Every tunable number lives here, so you can change an experiment without
hunting through the code. Each unit module reads its own dictionary.
"""

import os
from pathlib import Path

# Project root = the folder that contains main.py. It can be overridden with the
# CV_PROJECT_ROOT environment variable (useful for tests or a copied project).
PROJECT_ROOT = Path(os.environ.get("CV_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

DEFAULT_INPUT_DIR = PROJECT_ROOT / "inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

# Input files: name (without extension) and the extensions we accept.
# Phones often save .jpeg/.png photos and .mov videos, so several are allowed.
INPUT_FILES = {
    "original_photo": ("original_photo", (".jpg", ".jpeg", ".png")),
    "second_photo": ("second_photo", (".jpg", ".jpeg", ".png")),
    "original_video": ("original_video", (".mp4", ".mov", ".avi", ".mkv")),
    # Optional: your own photo of several touching objects (coins, fruit, ...)
    # for watershed. If it is missing, a documented scikit-image sample is used.
    "watershed_photo": ("watershed_photo", (".jpg", ".jpeg", ".png")),
}

# One seed for every random operation, so synthetic noise is identical on every run.
RANDOM_SEED = 42

# ---------------------------------------------------------------- Unit 3
RESTORATION = {
    "gaussian_noise_sigma": 25.0,  # standard deviation on the 0-255 scale
    "salt_pepper_amount": 0.05,  # 5 % of pixels become pure black or white
    "salt_vs_pepper": 0.5,  # half salt (white), half pepper (black)
    "kernel_sizes": [3, 5, 7, 9, 11],  # filter sizes tested (odd numbers only)
    "default_kernel": 5,  # kernel used for the main comparison figures
    # Zoom window (x0, y0, x1, y1 as fractions of width/height) used to show
    # fine detail. For the supplied portrait it covers the eyes and mouth.
    "zoom_box": (0.36, 0.20, 0.64, 0.42),
}

SEGMENTATION = {
    "blur_kernel": 5,  # Gaussian blur before thresholding / Canny
    "adaptive_block_size": 31,  # neighbourhood size for adaptive thresholding
    "adaptive_c": 5,  # constant subtracted in adaptive thresholding
    "morph_kernel_size": 9,  # structuring element size for erosion/dilation
    "morph_kernel_shape": "ellipse",  # "ellipse", "rect" or "cross"
    "canny_low": 50,
    "canny_high": 150,
    "canny_auto_sigma": 0.33,  # thresholds = median * (1 -/+ sigma)
    # Watershed for touching objects
    "watershed_max_side": 800,  # large photos are shrunk first (speed)
    "watershed_min_distance_fraction": 0.04,  # peak spacing as a fraction of the short side
    "watershed_min_object_fraction": 0.0005,  # ignore blobs smaller than this share of the image
    # Uneven lighting is removed with a top-hat whose kernel is this share of the short side.
    # It only works when each object is smaller than the kernel; set to 0 to switch it off.
    "watershed_background_kernel_fraction": 0.25,
    # Marker-based watershed on the portrait (person vs background)
    "portrait_border_fraction": 0.02,  # border strip used to learn the background colour
    "portrait_bg_distance": 12.0,  # Lab colour distance treated as "surely background"
    "portrait_fg_distance": 40.0,  # Lab colour distance treated as "surely person"
    "zoom_box": (0.30, 0.02, 0.70, 0.40),  # hair/face region used to show morphology detail
}

# ---------------------------------------------------------------- Unit 4
FEATURES = {
    "hog_max_side": 512,  # HOG is computed on a resized copy so the cells are visible
    "hog_orientations": 9,
    "hog_pixels_per_cell": [8, 16, 32],  # compared in a table; the middle one is shown
    "hog_cells_per_block": 2,
    "sift_ratio": 0.75,  # Lowe's ratio test threshold
    "ransac_threshold": 5.0,  # pixels, for RANSAC geometric verification
    "correct_match_tolerance": 3.0,  # pixels, for transforms with known ground truth
    "rotation_angles": [15, 30, 45, 90, 180],
    "scale_factors": [0.25, 0.5, 0.75, 1.5, 2.0],
    "gamma_values": [0.5, 2.0],  # illumination change (dark / bright)
    "max_matches_drawn": 60,
}

# ---------------------------------------------------------------- Unit 5
DETECTION = {
    "haar_face_cascade": "haarcascade_frontalface_default.xml",
    "haar_eye_cascade": "haarcascade_eye.xml",
    "haar_scale_factor": 1.1,
    "haar_min_neighbors": 5,
    "haar_min_size_fraction": 0.08,  # smallest face = 8 % of the image's short side
    "haar_neighbor_sweep": [1, 2, 3, 5, 8],
    "yolo_model": "yolo11n.pt",  # small COCO model, downloaded once into models/
    "yolo_confidence": 0.25,
    "rotation_angles": [0, 15, 30, 45, 90],
    # Known content of the supplied portrait: one person, one face.
    "portrait_expected_faces": 1,
    "portrait_expected_persons": 1,
    "video_frame_fractions": [0.2, 0.5, 0.8],  # frames sampled for detection
}

MOTION = {
    "max_side": 640,  # frames are resized so their longer side is at most this (speed)
    "max_seconds": 60.0,  # safety limit for very long videos
    "expected_duration": (15.0, 30.0),  # assignment asks for 15-30 s
    "fallback_fps": 30.0,  # used when the file reports an invalid frame rate
    "diff_blur": 5,
    "diff_threshold": 25,  # grey-level change that counts as motion
    "min_region_fraction": 0.002,  # ignore moving blobs smaller than this share of the frame
    "mog2_history": 300,
    "mog2_var_threshold": 16,
    "mog2_detect_shadows": True,
    "mask_kernel": 5,  # clean-up kernel for both foreground masks
    "lk_max_corners": 200,
    "lk_quality_level": 0.01,
    "lk_min_distance": 7,
    "lk_block_size": 7,
    "lk_win_size": 21,
    "lk_max_level": 3,
    "lk_redetect_every": 10,  # find new corners every N frames
    "lk_moving_threshold": 1.0,  # pixels per frame that count as a moving point
    "arrow_scale": 3.0,  # motion arrows are drawn 3x longer so they are visible
    "snapshot_fractions": [0.25, 0.5, 0.75],
    "consecutive_frames": 3,
    "gif_seconds": 4.0,
    "gif_width": 480,
    "output_video_tile_width": 320,
}
