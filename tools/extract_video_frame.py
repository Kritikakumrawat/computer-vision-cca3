"""Extract a clear frame of the student from her own video as a CANDIDATE second image for Unit 4.

    python tools/extract_video_frame.py                      # uses inputs/front_camera_video.*
    python tools/extract_video_frame.py --video inputs/original_video.mp4

One frame per second is examined. Frames where the Haar cascade finds exactly one
frontal face are scored by sharpness (variance of the Laplacian inside the face)
and the sharpest one is saved, unchanged, as inputs/second_image_video_frame.jpg.
A JSON file next to it records the source video, frame number and time.

This is a frame taken FROM A VIDEO, not a separately taken photograph. Unit 4 of
the assignment asks for "a second photograph", so a frame may not satisfy that
requirement; main.py only uses it when inputs/second_photo.jpg is missing and
labels it as a video frame everywhere.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, utils  # noqa: E402
from src.object_detection import detect_eyes, detect_haar, load_haar_cascade  # noqa: E402

OUT_IMAGE = config.DEFAULT_INPUT_DIR / "second_image_video_frame.jpg"
OUT_INFO = config.DEFAULT_INPUT_DIR / "second_image_video_frame.json"


def best_frame(video: Path, step_seconds: float = 1.0):
    cascade = load_haar_cascade(config.DETECTION["haar_face_cascade"])
    eye_cascade = load_haar_cascade(config.DETECTION["haar_eye_cascade"])
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best, examined, with_face = None, 0, 0
    try:
        for index in range(0, total, max(1, int(round(fps * step_seconds)))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            examined += 1
            gray = utils.to_gray(frame)
            faces = detect_haar(gray, cascade, 1.1, 8, 0.15)  # strict: large, well-supported frontal face
            if len(faces) != 1:
                continue
            x, y, w, h = faces[0]
            # Both eyes must be found in the upper half of the face (open eyes, nothing in front of the face).
            eyes = [e for e in detect_eyes(gray, faces[0], eye_cascade, min_neighbors=8) if e[1] + e[3] / 2 < y + 0.55 * h]
            if len(eyes) != 2:
                continue
            with_face += 1
            # Sharpness of the eye region only, so hands or hair lower in the frame do not inflate the score.
            sharpness = float(cv2.Laplacian(gray[y + h // 5:y + h // 2, x:x + w], cv2.CV_64F).var())
            if best is None or sharpness > best["sharpness"]:
                best = {"frame_index": index, "time_s": round(index / fps, 2), "sharpness": round(sharpness, 1),
                        "face_box": [int(v) for v in faces[0]], "frame": frame}
    finally:
        cap.release()
    return best, examined, with_face, fps, total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, default=None, help="video to extract from")
    args = parser.parse_args(argv)
    video = args.video or utils.find_input(config.DEFAULT_INPUT_DIR, "front_camera_video") or \
        utils.find_input(config.DEFAULT_INPUT_DIR, "original_video")
    if video is None:
        sys.exit("No video found in inputs/.")
    best, examined, with_face, fps, total = best_frame(video)
    if best is None:
        sys.exit(f"No frame with exactly one clear frontal face in {video.name} ({examined} frames examined).")
    utils.write_image(OUT_IMAGE, best.pop("frame"), jpeg_quality=95)
    info = {
        "kind": "frame extracted from the student's self-recorded video - NOT a separately taken photograph",
        "source_video": video.name, "video_fps": round(fps, 3), "video_frames": total, **best,
        "selection": f"sharpest eye region of {with_face} frames (1 per second, {examined} examined) with one frontal "
                     "Haar face (minNeighbors 8, face >= 15% of the short side) and both eyes detected",
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Unit 4 asks for a second photograph; this frame is a candidate substitute only.",
    }
    utils.save_json(OUT_INFO, info)
    print(f"Saved {OUT_IMAGE.relative_to(ROOT)}: frame {info['frame_index']} at {info['time_s']} s of {video.name} "
          f"(sharpness {info['sharpness']}; {with_face}/{examined} sampled frames had one clear face)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
