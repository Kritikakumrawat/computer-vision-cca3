"""Unit 5A - Object recognition with a Viola-Jones Haar cascade and YOLO.

- Haar cascade (Viola-Jones): a face-only detector built from simple
  light/dark rectangle features, AdaBoost and a cascade of stages.
- YOLO (Ultralytics): a deep neural network trained on the 80-class COCO
  dataset; it predicts boxes, class labels and confidence scores in one pass.

Experiments:
1. Both detectors on the portrait.
2. Haar 'minNeighbors' sweep (shows false detections when it is too low).
3. Both detectors on rotated copies of the portrait (shows where they miss).
4. Both detectors on frames sampled from the original video (if supplied).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from src import config, utils
from src.feature_extraction import rotate_image

LOG = logging.getLogger("cv_assignment.unit5.detection")

HAAR_COLOR = (0, 200, 255)  # BGR: amber boxes for Haar
EYE_COLOR = (255, 160, 0)  # BGR: light blue boxes for eyes
YOLO_COLOR = (214, 120, 42)  # BGR: blue boxes for YOLO


# ------------------------------------------------------------------ Haar cascade
def load_haar_cascade(filename: str) -> cv2.CascadeClassifier:
    """Load a pretrained cascade that ships with opencv-python."""
    if not hasattr(cv2, "data") or not hasattr(cv2, "CascadeClassifier"):
        raise utils.ModelUnavailableError("This OpenCV build has no Haar cascades (use opencv-python 4.x).")
    path = Path(cv2.data.haarcascades) / filename
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise utils.ModelUnavailableError(f"Could not load Haar cascade file: {path}")
    return cascade


def detect_haar(gray: np.ndarray, cascade, scale_factor=1.1, min_neighbors=5, min_size_fraction=0.08):
    """Return a list of (x, y, w, h) boxes. Histogram equalisation evens out lighting first."""
    equalized = cv2.equalizeHist(gray)
    min_side = max(20, int(min_size_fraction * min(gray.shape[:2])))
    boxes = cascade.detectMultiScale(equalized, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                     minSize=(min_side, min_side))
    return [tuple(int(v) for v in box) for box in boxes]


def detect_eyes(gray: np.ndarray, face_box, eye_cascade, min_neighbors=5):
    """Search for eyes only inside a detected face (in image coordinates)."""
    x, y, w, h = face_box
    roi = cv2.equalizeHist(gray[y:y + h, x:x + w])
    eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=min_neighbors,
                                        minSize=(max(10, w // 10), max(10, h // 10)))
    return [(x + int(ex), y + int(ey), int(ew), int(eh)) for ex, ey, ew, eh in eyes]


# ------------------------------------------------------------------ YOLO
def load_yolo(model_name: str, models_dir: Path):
    """Load an Ultralytics YOLO model; the official weights are downloaded once into models/."""
    models_dir = utils.ensure_dir(models_dir)
    # Keep Ultralytics' settings file inside the project and switch off usage analytics.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(models_dir / "ultralytics_settings"))
    os.environ.setdefault("YOLO_VERBOSE", "False")
    try:
        from ultralytics import YOLO, settings
    except ImportError as exc:
        raise utils.ModelUnavailableError(
            "The 'ultralytics' package is not installed (pip install -r requirements.txt).") from exc
    try:
        settings.update({"sync": False})
    except Exception:  # older/newer versions may not allow this; it is only a preference
        pass
    weights = models_dir / model_name
    try:
        model = YOLO(str(weights))
    except Exception as exc:  # network failure, corrupt file, incompatible torch ...
        raise utils.ModelUnavailableError(
            f"Could not load or download YOLO weights '{model_name}' into {models_dir}: {exc}") from exc
    return model


def detect_yolo(model, image_bgr: np.ndarray, confidence: float = 0.25) -> list[dict]:
    """Run YOLO on one BGR image and return [{'label', 'confidence', 'box': (x1, y1, x2, y2)}]."""
    result = model.predict(source=image_bgr, conf=confidence, device="cpu", verbose=False)[0]
    detections = []
    for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                              result.boxes.cls.cpu().numpy()):
        detections.append({"label": model.names[int(cls)], "confidence": float(conf),
                           "box": tuple(int(round(v)) for v in box)})
    return detections


# ------------------------------------------------------------------ drawing
def draw_haar(image, faces, eyes=()):
    canvas = image.copy()
    thickness = max(2, image.shape[1] // 200)
    for x, y, w, h in faces:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), HAAR_COLOR, thickness)
        utils.draw_label(canvas, "face (Haar)", x, y, HAAR_COLOR)
    for x, y, w, h in eyes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), EYE_COLOR, max(1, thickness - 1))
    return canvas


def draw_yolo(image, detections):
    canvas = image.copy()
    thickness = max(2, image.shape[1] // 200)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), YOLO_COLOR, thickness)
        utils.draw_label(canvas, f"{det['label']} {det['confidence']:.2f}", x1, y1, YOLO_COLOR)
    return canvas


def summarize_yolo(detections) -> str:
    if not detections:
        return "no objects"
    return ", ".join(f"{d['label']} {d['confidence']:.2f}" for d in detections)


# ------------------------------------------------------------------ main task
def run(paths: utils.ProjectPaths, options: dict | None = None) -> dict:
    options = options or {}
    params = config.DETECTION
    out_dir = paths.unit_dir(5, "detection")
    files, pending, observations = [], [], []

    def keep(path):
        files.append(utils.relative_to_output(path, paths.output_dir))
        return path

    photo_path = utils.require_input(paths.input_dir, "original_photo")
    portrait = utils.read_image(photo_path)
    gray = utils.to_gray(portrait)

    face_cascade = load_haar_cascade(params["haar_face_cascade"])
    eye_cascade = load_haar_cascade(params["haar_eye_cascade"])
    haar_kwargs = dict(scale_factor=params["haar_scale_factor"], min_neighbors=params["haar_min_neighbors"],
                       min_size_fraction=params["haar_min_size_fraction"])

    # --- 1a. Haar on the portrait (one warm-up call first, as for YOLO, so timings are comparable)
    detect_haar(gray, face_cascade, **haar_kwargs)
    with utils.timer() as t_haar:
        faces = detect_haar(gray, face_cascade, **haar_kwargs)
    eyes = [e for f in faces for e in detect_eyes(gray, f, eye_cascade)]
    haar_img = draw_haar(portrait, faces, eyes)
    keep(utils.write_image(out_dir / "01_haar_faces_portrait.jpg", haar_img))
    haar_result = {"faces": len(faces), "face_boxes": faces, "eyes": len(eyes), "eye_boxes": eyes,
                   "time_ms": t_haar["ms"], "parameters": haar_kwargs}
    LOG.info("Haar: %d face(s), %d eye(s) in %.0f ms", len(faces), len(eyes), t_haar["ms"])

    # --- 1b. YOLO on the portrait
    yolo_model = None
    yolo_error = None
    yolo_result = None
    if options.get("skip_yolo"):
        yolo_error = "YOLO skipped by the --skip-yolo option."
    else:
        try:
            yolo_model = load_yolo(options.get("yolo_model") or params["yolo_model"], paths.models_dir)
            detect_yolo(yolo_model, portrait, params["yolo_confidence"])  # warm-up run (first call is slower)
            with utils.timer() as t_yolo:
                yolo_dets = detect_yolo(yolo_model, portrait, params["yolo_confidence"])
            keep(utils.write_image(out_dir / "02_yolo_portrait.jpg", draw_yolo(portrait, yolo_dets)))
            yolo_result = {"model": options.get("yolo_model") or params["yolo_model"], "detections": yolo_dets,
                           "time_ms": t_yolo["ms"], "confidence_threshold": params["yolo_confidence"]}
            LOG.info("YOLO: %s in %.0f ms", summarize_yolo(yolo_dets), t_yolo["ms"])
        except utils.ModelUnavailableError as exc:
            yolo_error = str(exc)
    if yolo_error:
        LOG.warning("YOLO not run: %s", yolo_error)
        pending.append(f"YOLO detection ({yolo_error})")

    panels = [haar_img]
    titles = [f"Haar cascade: {len(faces)} face(s), {len(eyes)} eye box(es)"]
    if yolo_result:
        panels.append(draw_yolo(portrait, yolo_result["detections"]))
        titles.append(f"YOLO ({yolo_result['model']}): {summarize_yolo(yolo_result['detections'])}")
    fig = utils.image_grid_figure(
        panels, titles, "Face detection (Haar) versus general object detection (YOLO) on the portrait",
        caption=("Haar only answers 'is there a frontal face here?'. YOLO labels any of 80 COCO classes, so it marks "
                 "the whole person rather than the face."),
        cols=len(panels), panel_width=5.0,
    )
    keep(utils.finish_figure(fig, out_dir / "fig_haar_vs_yolo_portrait.png"))

    # --- 2. Haar minNeighbors sweep
    sweep_rows = []
    sweep_imgs, sweep_titles = [], []
    for n in params["haar_neighbor_sweep"]:
        boxes = detect_haar(gray, face_cascade, params["haar_scale_factor"], n, params["haar_min_size_fraction"])
        sweep_rows.append({"min_neighbors": n, "faces_detected": len(boxes),
                           "extra_detections": max(0, len(boxes) - params["portrait_expected_faces"])})
        sweep_imgs.append(draw_haar(portrait, boxes))
        sweep_titles.append(f"minNeighbors = {n}: {len(boxes)} box(es)")
    keep(utils.save_csv(out_dir / "haar_min_neighbors_sweep.csv", sweep_rows))
    fig = utils.image_grid_figure(sweep_imgs, sweep_titles, "Haar cascade: effect of minNeighbors",
                                  caption="minNeighbors = how many overlapping candidate windows are needed to accept "
                                          "a face. Low values accept more (including false) detections.",
                                  cols=3, panel_width=3.4)
    keep(utils.finish_figure(fig, out_dir / "fig_haar_min_neighbors.png"))
    for row in sweep_rows:
        if row["extra_detections"] > 0:
            observations.append(
                f"Haar cascade with minNeighbors={row['min_neighbors']} returned {row['faces_detected']} boxes for a "
                f"portrait that contains {params['portrait_expected_faces']} face: at least {row['extra_detections']} "
                f"false detection(s).")

    # --- 3. Rotation robustness (the face is still fully visible in every rotated copy)
    rotation_rows = []
    rot_panels, rot_titles = [], []
    for angle in params["rotation_angles"]:
        rotated, _ = rotate_image(portrait, angle) if angle else (portrait, None)
        r_gray = utils.to_gray(rotated)
        r_faces = detect_haar(r_gray, face_cascade, **haar_kwargs)
        row = {"angle_deg": angle, "haar_faces": len(r_faces)}
        canvas = draw_haar(rotated, r_faces)
        if yolo_model is not None:
            dets = detect_yolo(yolo_model, rotated, params["yolo_confidence"])
            persons = [d for d in dets if d["label"] == "person"]
            others = [d for d in dets if d["label"] != "person"]
            row.update({"yolo_persons": len(persons),
                        "yolo_best_person_conf": max((d["confidence"] for d in persons), default=0.0),
                        "yolo_other_labels": ", ".join(f"{d['label']} {d['confidence']:.2f}" for d in others)})
            canvas = draw_yolo(canvas, dets)
        rotation_rows.append(row)
        rot_panels.append(canvas)
        title = f"{angle} deg: Haar {len(r_faces)} face(s)"
        if "yolo_persons" in row:
            title += f"\nYOLO {row['yolo_persons']} person(s)"
            if row["yolo_persons"]:
                title += f" (best {row['yolo_best_person_conf']:.2f})"
        rot_titles.append(title)
    keep(utils.save_csv(out_dir / "rotation_robustness.csv", rotation_rows))
    fig = utils.image_grid_figure(rot_panels, rot_titles, "Detector robustness to in-plane rotation",
                                  caption="Amber = Haar face boxes, blue = YOLO boxes. The rotated copies are made "
                                          "from the same portrait, so one face and one person are always present.",
                                  cols=3, panel_width=3.4)
    keep(utils.finish_figure(fig, out_dir / "fig_rotation_robustness.png"))

    expected_faces = params["portrait_expected_faces"]
    expected_persons = params["portrait_expected_persons"]
    for row in rotation_rows:
        if row["haar_faces"] < expected_faces:
            observations.append(f"Missed detection: the Haar cascade found {row['haar_faces']} face(s) in the portrait "
                                f"rotated by {row['angle_deg']} degrees, although the face is fully visible.")
        elif row["haar_faces"] > expected_faces:
            observations.append(f"False detection: the Haar cascade returned {row['haar_faces']} boxes in the portrait "
                                f"rotated by {row['angle_deg']} degrees (one face present).")
        if "yolo_persons" in row:
            if row["yolo_persons"] < expected_persons:
                observations.append(f"Missed detection: YOLO found no person in the portrait rotated by "
                                    f"{row['angle_deg']} degrees.")
            elif row["yolo_persons"] > expected_persons:
                observations.append(f"Duplicate/false detection: YOLO returned {row['yolo_persons']} person boxes in "
                                    f"the portrait rotated by {row['angle_deg']} degrees (one person present).")
            if row["yolo_other_labels"]:
                observations.append(f"YOLO also labelled other classes at {row['angle_deg']} degrees: "
                                    f"{row['yolo_other_labels']} (check the figure to judge whether they are correct).")
    if len(eyes) != 2 * len(faces):
        observations.append(f"The Haar eye cascade returned {len(eyes)} eye box(es) for {len(faces)} detected face(s) "
                            f"(2 eyes expected per face).")

    def detect_on_video(video_path, fractions, file_prefix, fig_name, csv_name, title, panel_width):
        """Run Haar (faces + eyes) and YOLO on frames sampled at `fractions` of a video."""
        rows, panels, titles = [], [], []
        cap = cv2.VideoCapture(str(video_path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            for fraction in fractions:
                index = max(0, min(total - 1, int(fraction * total))) if total > 0 else 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    continue
                f_gray = utils.to_gray(frame)
                f_faces = detect_haar(f_gray, face_cascade, params["haar_scale_factor"], params["haar_min_neighbors"],
                                      0.04)
                f_eyes = [e for f in f_faces for e in detect_eyes(f_gray, f, eye_cascade)]
                canvas = draw_haar(frame, f_faces, f_eyes)
                row = {"frame_index": index, "time_s": round(index / fps, 1), "haar_faces": len(f_faces),
                       "haar_eyes": len(f_eyes)}
                if yolo_model is not None:
                    dets = detect_yolo(yolo_model, frame, params["yolo_confidence"])
                    row["yolo_detections"] = summarize_yolo(dets)
                    canvas = draw_yolo(canvas, dets)
                keep(utils.write_image(out_dir / f"{file_prefix}_{index:05d}_detections.jpg", canvas))
                rows.append(row)
                panels.append(canvas)
                titles.append(f"Frame {index} ({row['time_s']} s): Haar {len(f_faces)} face(s)"
                              + (f"\nYOLO: {row['yolo_detections']}" if "yolo_detections" in row else ""))
        finally:
            cap.release()
        if panels:
            fig = utils.image_grid_figure(panels, titles, title, cols=len(panels), panel_width=panel_width)
            keep(utils.finish_figure(fig, out_dir / fig_name))
            keep(utils.save_csv(out_dir / csv_name, rows))
        return rows

    # --- 4. Frames from the self-recorded videos
    video_rows = []
    video_path = utils.find_input(paths.input_dir, "original_video")
    if video_path is None:
        pending.append("Haar/YOLO on frames of the self-recorded video (inputs/original_video.mp4 not supplied)")
    else:
        video_rows = detect_on_video(video_path, params["video_frame_fractions"], "10_video_frame",
                                     "fig_video_frame_detections.png", "video_frame_detections.csv",
                                     "Detections on frames of the original video", 4.0)
    # Optional second clip recorded with the front camera (frontal face, the case Haar is built for)
    face_rows = []
    face_video = utils.find_input(paths.input_dir, "front_camera_video")
    if face_video is not None:
        face_rows = detect_on_video(face_video, params["face_video_frame_fractions"], "11_front_camera_frame",
                                    "fig_front_camera_detections.png", "front_camera_frame_detections.csv",
                                    "Detections on frames of the front-camera video", 2.6)
        for row in face_rows:
            if row["haar_faces"] == 0:
                observations.append(f"Missed detection: the Haar cascade found no face in front-camera frame "
                                    f"{row['frame_index']} ({row['time_s']} s).")
            elif row["haar_faces"] > 1:
                observations.append(f"Extra detection: the Haar cascade returned {row['haar_faces']} face boxes in "
                                    f"front-camera frame {row['frame_index']} ({row['time_s']} s), which shows one face.")
        extra_eyes = [r for r in face_rows if r["haar_eyes"] > 2 * r["haar_faces"]]
        if extra_eyes:
            observations.append(
                f"Incorrect eye detections: in {len(extra_eyes)} of {len(face_rows)} front-camera frames the eye cascade "
                f"returned more than two eye boxes per face (up to {max(r['haar_eyes'] for r in face_rows)} in one "
                "frame); the extra boxes fall on other dark facial features such as the nostrils or mouth.")

    comparison = [
        {"aspect": "What it detects", "haar": "One object type per cascade file (here: frontal faces, eyes)",
         "yolo": "80 COCO classes (person, chair, phone, ...) in one pass"},
        {"aspect": "Output", "haar": "Boxes only (no class score)", "yolo": "Boxes + class label + confidence score"},
        {"aspect": "Portrait result", "haar": f"{len(faces)} face box(es), {len(eyes)} eye box(es)",
         "yolo": summarize_yolo(yolo_result["detections"]) if yolo_result else "not run"},
        {"aspect": "Time on this CPU (portrait)", "haar": f"{t_haar['ms']:.0f} ms (face cascade)",
         "yolo": f"{yolo_result['time_ms']:.0f} ms (after warm-up)" if yolo_result else "not run"},
        {"aspect": "Model size / training", "haar": "Small XML file shipped with OpenCV; boosted rectangle features",
         "yolo": "Neural network weights downloaded once; trained on COCO"},
    ]
    keep(utils.save_csv(out_dir / "haar_vs_yolo_comparison.csv", comparison))

    results = {
        "task": "Unit 5A - Haar cascade and YOLO detection",
        "status": "completed" if not pending else "partial",
        "input": photo_path.name,
        "haar": haar_result,
        "yolo": yolo_result,
        "yolo_error": yolo_error,
        "haar_min_neighbors_sweep": sweep_rows,
        "rotation_robustness": rotation_rows,
        "video_frames": video_rows,
        "front_camera_frames": face_rows,
        "comparison": comparison,
        "observations": observations,
        "pending": pending,
        "files": files,
    }
    utils.save_json(out_dir / "detection_results.json", results)
    return results
