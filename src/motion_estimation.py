"""Unit 5B - Video motion estimation.

Three techniques are applied to every pair of consecutive frames:

1. Frame differencing   - |frame(t) - frame(t-1)|, thresholded -> moving pixels.
2. GMM background model - OpenCV's BackgroundSubtractorMOG2 models every pixel
                          as a mixture of Gaussians; pixels that do not fit the
                          background model are foreground.
3. Lucas-Kanade flow    - tracks corner points between frames and gives a
                          motion vector (dx, dy) for each point.

Outputs: selected frames, foreground masks, motion-vector pictures, a
side-by-side output video, a short GIF, per-frame statistics and a comparison table.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src import config, utils

LOG = logging.getLogger("cv_assignment.unit5.motion")


# ------------------------------------------------------------------ video helpers
def validate_video(path: Path, fallback_fps: float = 30.0, expected_duration=(15.0, 30.0)) -> dict:
    """Open the video, read its properties and check it can actually be decoded."""
    path = Path(path)
    if not path.is_file():
        raise utils.InputMissingError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"OpenCV cannot open the video (unsupported codec or corrupt file): {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_valid = bool(fps and math.isfinite(fps) and 1.0 <= fps <= 240.0)
        if not fps_valid:
            fps = fallback_fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, first = cap.read()
        if not ok or first is None:
            raise ValueError(f"The video contains no readable frames: {path}")
        if frame_count <= 0:  # some containers do not store the count -> count by reading
            frame_count = 1
            while cap.read()[0]:
                frame_count += 1
    finally:
        cap.release()
    if frame_count < 2:
        raise ValueError("The video must contain at least 2 frames for motion estimation.")
    duration = frame_count / fps
    warnings = []
    if not fps_valid:
        warnings.append(f"Frame rate missing or invalid in the file; assumed {fallback_fps:g} fps.")
    low, high = expected_duration
    if not low <= duration <= high:
        warnings.append(f"Duration {duration:.1f} s is outside the assignment's {low:g}-{high:g} s range.")
    return {"file": path.name, "fps": float(fps), "fps_valid": fps_valid, "frame_count": frame_count,
            "width": width or first.shape[1], "height": height or first.shape[0], "duration_s": duration,
            "file_size_mb": path.stat().st_size / 1e6, "warnings": warnings}


def iterate_frames(path: Path, max_side: int, max_frames: int):
    """Yield (index, resized BGR frame). The capture is always released."""
    cap = cv2.VideoCapture(str(path))
    try:
        index = 0
        while index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame, _ = utils.resize_to_max_side(frame, max_side)
            yield index, frame
            index += 1
    finally:
        cap.release()


# ------------------------------------------------------------------ the three techniques
def frame_difference(prev_gray, gray, threshold=25, blur=5, kernel=None):
    """Return (difference image, cleaned binary motion mask)."""
    a = cv2.GaussianBlur(prev_gray, (blur, blur), 0)
    b = cv2.GaussianBlur(gray, (blur, blur), 0)
    diff = cv2.absdiff(a, b)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    if kernel is not None:
        # A small 3x3 opening removes single noisy pixels but keeps the thin bands of change
        # at the edges of a moving object; dilation then joins those bands into regions.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.dilate(mask, kernel, iterations=2)
    return diff, mask


def create_background_subtractor(history=300, var_threshold=16, detect_shadows=True):
    return cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold,
                                              detectShadows=detect_shadows)


def gmm_foreground(subtractor, frame, kernel=None):
    """Apply MOG2. Raw output: 255 = foreground, 127 = shadow, 0 = background."""
    raw = subtractor.apply(frame)
    mask = np.where(raw == 255, 255, 0).astype(np.uint8)  # drop shadows
    if kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return raw, mask


def detect_corners(gray, params: dict):
    return cv2.goodFeaturesToTrack(gray, maxCorners=params["lk_max_corners"], qualityLevel=params["lk_quality_level"],
                                   minDistance=params["lk_min_distance"], blockSize=params["lk_block_size"])


def lucas_kanade(prev_gray, gray, prev_points, params: dict):
    """Track points from prev_gray to gray. Returns (old points, new points) that were found."""
    if prev_points is None or len(prev_points) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty
    win = params["lk_win_size"]
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_points, None, winSize=(win, win), maxLevel=params["lk_max_level"],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if new_points is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty
    found = status.ravel() == 1
    return prev_points.reshape(-1, 2)[found], new_points.reshape(-1, 2)[found]


# ------------------------------------------------------------------ drawing
def moving_regions(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]


def draw_boxes(frame, boxes, color):
    canvas = frame.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
    return canvas


def draw_flow(frame, old, new, moving, arrow_scale=3.0):
    """Green arrows = moving points (drawn `arrow_scale` x longer), grey dots = static points."""
    canvas = frame.copy()
    for (x0, y0), (x1, y1), is_moving in zip(old, new, moving):
        if is_moving:
            end = (int(x0 + arrow_scale * (x1 - x0)), int(y0 + arrow_scale * (y1 - y0)))
            cv2.arrowedLine(canvas, (int(x0), int(y0)), end, (40, 220, 40), 2, cv2.LINE_AA, tipLength=0.35)
        else:
            cv2.circle(canvas, (int(x1), int(y1)), 2, (160, 160, 160), -1)
    return canvas


def tile(images, labels, tile_width):
    """2 x 2 mosaic with a caption strip on each tile (for the output video and GIF)."""
    tiles = []
    for image, label in zip(images, labels):
        image = utils.to_bgr(image)
        h, w = image.shape[:2]
        scaled = cv2.resize(image, (tile_width, max(1, int(round(h * tile_width / w)))))
        cv2.rectangle(scaled, (0, 0), (tile_width, 20), (20, 20, 20), -1)
        cv2.putText(scaled, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(scaled)
    return np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4])])


def mask_iou(a, b) -> float:
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


# ------------------------------------------------------------------ main processing
def process_video(video_path: Path, out_dir: Path, params: dict, output_dir_root: Path | None = None) -> dict:
    """Run all three techniques over the whole video and save every output."""
    out_dir = utils.ensure_dir(out_dir)
    root = output_dir_root or out_dir
    files = []

    def keep(path):
        files.append(utils.relative_to_output(path, root))
        return path

    info = validate_video(video_path, params["fallback_fps"], params["expected_duration"])
    for warning in info["warnings"]:
        LOG.warning(warning)
    fps = info["fps"]
    max_frames = min(info["frame_count"], int(params["max_seconds"] * fps))
    if max_frames < info["frame_count"]:
        LOG.warning("Only the first %.0f s (%d frames) are processed.", params["max_seconds"], max_frames)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["mask_kernel"], params["mask_kernel"]))
    subtractor = create_background_subtractor(params["mog2_history"], params["mog2_var_threshold"],
                                              params["mog2_detect_shadows"])
    snapshot_indices = sorted({max(1, min(max_frames - 1, int(f * max_frames))) for f in params["snapshot_fractions"]})
    consecutive_start = max(0, max_frames // 2 - 1)
    consecutive_set = set(range(consecutive_start, min(max_frames, consecutive_start + params["consecutive_frames"])))
    gif_start = snapshot_indices[0]
    gif_end = gif_start + int(params["gif_seconds"] * fps)

    rows, snapshots, consecutive, gif_frames = [], {}, {}, []
    peak_value, peak_index, peak_snapshot = -1.0, None, None
    writer = None
    video_out = out_dir / "motion_side_by_side.mp4"
    prev_gray = prev_points = None
    frame_size = None
    first_frame = None

    for index, frame in iterate_frames(video_path, params["max_side"], max_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_size = frame.shape[:2]
        min_area = params["min_region_fraction"] * frame.shape[0] * frame.shape[1]
        if first_frame is None:
            first_frame = frame.copy()
        if index in consecutive_set:
            consecutive[index] = frame.copy()
        with utils.timer() as t_gmm:
            raw_gmm, gmm_mask = gmm_foreground(subtractor, frame, kernel)  # MOG2 must see every frame
        if prev_gray is None:
            prev_gray = gray
            prev_points = detect_corners(gray, params)
            continue

        with utils.timer() as t_diff:
            diff, diff_mask = frame_difference(prev_gray, gray, params["diff_threshold"], params["diff_blur"], kernel)
        with utils.timer() as t_lk:
            if prev_points is None or len(prev_points) < 0.3 * params["lk_max_corners"] or \
                    index % params["lk_redetect_every"] == 0:
                prev_points = detect_corners(prev_gray, params)
            old, new = lucas_kanade(prev_gray, gray, prev_points, params)
        vectors = new - old
        magnitudes = np.linalg.norm(vectors, axis=1) if len(vectors) else np.empty(0)
        moving = magnitudes > params["lk_moving_threshold"]
        inside_gmm = [gmm_mask[min(int(y), gmm_mask.shape[0] - 1), min(int(x), gmm_mask.shape[1] - 1)] > 0
                      for x, y in new[moving]] if moving.any() else []

        diff_boxes = moving_regions(diff_mask, min_area)
        gmm_boxes = moving_regions(gmm_mask, min_area)
        rows.append({
            "frame": index, "time_s": index / fps,
            "diff_foreground_pct": float((diff_mask > 0).mean() * 100), "diff_regions": len(diff_boxes),
            "gmm_foreground_pct": float((gmm_mask > 0).mean() * 100), "gmm_regions": len(gmm_boxes),
            "mask_iou": mask_iou(diff_mask, gmm_mask),
            "lk_tracked_points": int(len(new)), "lk_moving_points": int(moving.sum()),
            "lk_mean_magnitude_px": float(magnitudes.mean()) if len(magnitudes) else 0.0,
            "lk_mean_moving_magnitude_px": float(magnitudes[moving].mean()) if moving.any() else 0.0,
            "lk_median_dx": float(np.median(vectors[:, 0])) if len(vectors) else 0.0,
            "lk_median_dy": float(np.median(vectors[:, 1])) if len(vectors) else 0.0,
            "lk_moving_in_gmm_pct": float(np.mean(inside_gmm) * 100) if len(inside_gmm) else float("nan"),
            "diff_ms": t_diff["ms"], "gmm_ms": t_gmm["ms"], "lk_ms": t_lk["ms"],
        })

        flow_view = draw_flow(frame, old, new, moving, params["arrow_scale"])
        mosaic = tile([draw_boxes(frame, gmm_boxes, (40, 200, 40)), diff_mask, gmm_mask, flow_view],
                      [f"Frame {index} ({index / fps:.1f} s)", "Frame differencing mask", "MOG2 (GMM) foreground",
                       "Lucas-Kanade motion vectors"], params["output_video_tile_width"])
        if writer is None:
            writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), min(fps, 30.0),
                                     (mosaic.shape[1], mosaic.shape[0]))
            if not writer.isOpened():
                LOG.warning("mp4v codec unavailable; writing MJPG .avi instead.")
                video_out = out_dir / "motion_side_by_side.avi"
                writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"MJPG"), min(fps, 30.0),
                                         (mosaic.shape[1], mosaic.shape[0]))
        writer.write(mosaic)
        if gif_start <= index < gif_end and index % 2 == 0:
            gif_frames.append(mosaic)
        is_peak = rows[-1]["diff_foreground_pct"] > peak_value and index not in snapshot_indices
        if index in snapshot_indices or is_peak:
            snap = {"frame": frame.copy(), "diff": diff_mask, "gmm": gmm_mask, "raw_gmm": raw_gmm,
                    "flow": flow_view, "diff_boxes": draw_boxes(frame, diff_boxes, (40, 40, 230)), "row": rows[-1]}
            if is_peak:  # frame with the most frame-differencing motion so far
                peak_value, peak_index, peak_snapshot = rows[-1]["diff_foreground_pct"], index, snap
            else:
                snapshots[index] = snap
        prev_gray = gray
        prev_points = new.reshape(-1, 1, 2).astype(np.float32) if len(new) else None

    if writer is not None:
        writer.release()
        keep(video_out)
    if not rows:
        raise ValueError("No consecutive frame pairs could be processed.")
    # Besides the evenly spaced snapshots, keep the frame with the strongest motion.
    if peak_snapshot is not None:
        snapshots[peak_index] = peak_snapshot
    snapshots = dict(sorted(snapshots.items()))

    # --- Saved frames and masks
    for index, frame in sorted(consecutive.items()):
        keep(utils.write_image(out_dir / f"01_consecutive_frame_{index:05d}.jpg", frame))
    for index, snap in snapshots.items():
        keep(utils.write_image(out_dir / f"02_frame_{index:05d}.jpg", snap["frame"]))
        keep(utils.write_image(out_dir / f"03_frame_{index:05d}_diff_mask.png", snap["diff"]))
        keep(utils.write_image(out_dir / f"04_frame_{index:05d}_mog2_mask.png", snap["gmm"]))
        keep(utils.write_image(out_dir / f"05_frame_{index:05d}_lk_vectors.jpg", snap["flow"]))

    # Consecutive frames figure with their difference images
    cons_items = sorted(consecutive.items())
    if len(cons_items) >= 2:
        images = [f for _, f in cons_items]
        titles = [f"Frame {i} ({i / fps:.2f} s)" for i, _ in cons_items]
        for (i0, f0), (i1, f1) in zip(cons_items, cons_items[1:]):
            images.append(cv2.absdiff(utils.to_gray(f0), utils.to_gray(f1)) * 4)
            titles.append(f"|frame {i1} - frame {i0}| (x4 for visibility)")
        fig = utils.image_grid_figure(images, titles, "Consecutive frames extracted from the video",
                                      cols=len(cons_items), panel_width=3.6)
        keep(utils.finish_figure(fig, out_dir / "fig_consecutive_frames.png"))

    # Snapshot comparison figure
    if snapshots:
        images, titles = [], []
        for index, snap in snapshots.items():
            r = snap["row"]
            images += [snap["diff_boxes"], snap["diff"], snap["gmm"], snap["flow"]]
            label = "peak motion" if index == peak_index else "red = diff regions"
            titles += [f"Frame {index} ({index / fps:.1f} s), {label}",
                       f"Frame differencing: {r['diff_foreground_pct']:.1f}% moving",
                       f"MOG2 foreground: {r['gmm_foreground_pct']:.1f}%",
                       f"Lucas-Kanade: {r['lk_moving_points']}/{r['lk_tracked_points']} points moving"]
        fig = utils.image_grid_figure(images, titles, "Motion estimation on selected frames", cols=4, panel_width=3.6,
                                      caption="Arrows are drawn 3x longer than the measured motion so they are visible.")
        keep(utils.finish_figure(fig, out_dir / "fig_motion_snapshots.png"))

    # Time series (three panels, one measure each - no shared dual axis)
    t = [r["time_s"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    axes[0].plot(t, [r["diff_foreground_pct"] for r in rows], color=utils.SERIES_COLORS[0], linewidth=1.5,
                 label="Frame differencing")
    axes[0].plot(t, [r["gmm_foreground_pct"] for r in rows], color=utils.SERIES_COLORS[1], linewidth=1.5,
                 label="MOG2 background subtraction")
    axes[0].set_ylabel("Foreground pixels (%)")
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_title("Share of the frame marked as moving", fontweight="bold")
    axes[1].plot(t, [r["lk_mean_moving_magnitude_px"] for r in rows], color=utils.SERIES_COLORS[2], linewidth=1.5)
    axes[1].set_ylabel("Pixels per frame")
    axes[1].set_title("Lucas-Kanade: mean speed of the moving points", fontweight="bold")
    for color, key, label in ((utils.SERIES_COLORS[0], "diff_ms", "Frame differencing"),
                              (utils.SERIES_COLORS[1], "gmm_ms", "MOG2"),
                              (utils.SERIES_COLORS[2], "lk_ms", "Lucas-Kanade")):
        axes[2].plot(t, [r[key] for r in rows], color=color, linewidth=1.2, label=label)
    axes[2].set_ylabel("Milliseconds per frame")
    axes[2].set_xlabel("Video time (s)")
    axes[2].legend(loc="upper right", ncol=3)
    axes[2].set_title("Processing time per frame", fontweight="bold")
    for ax in axes:  # leave headroom above the lines so the legends never cover data
        top = max((line.get_ydata().max() for line in ax.get_lines() if len(line.get_ydata())), default=1.0)
        ax.set_ylim(0, max(top, 1e-3) * 1.35)
    fig.tight_layout()
    keep(utils.finish_figure(fig, out_dir / "fig_motion_timeseries.png"))

    # GIF preview
    if gif_frames:
        from PIL import Image

        frames = []
        for mosaic in gif_frames:
            small, _ = utils.resize_to_max_side(mosaic, params["gif_width"])
            frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        gif_path = out_dir / "motion_preview.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(2000 / fps), loop=0)
        keep(gif_path)

    keep(utils.save_csv(out_dir / "per_frame_motion_statistics.csv", rows))

    def mean(key):
        values = [r[key] for r in rows if r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return float(np.mean(values)) if values else float("nan")

    summary = [
        {"technique": "Frame differencing", "mean_foreground_pct": mean("diff_foreground_pct"),
         "mean_regions_per_frame": mean("diff_regions"), "mean_ms_per_frame": mean("diff_ms"),
         "output": "Binary mask of pixels that changed since the previous frame"},
        {"technique": "GMM background subtraction (MOG2)", "mean_foreground_pct": mean("gmm_foreground_pct"),
         "mean_regions_per_frame": mean("gmm_regions"), "mean_ms_per_frame": mean("gmm_ms"),
         "output": "Binary mask of pixels that differ from the learned background"},
        {"technique": "Lucas-Kanade optical flow", "mean_tracked_points": mean("lk_tracked_points"),
         "mean_moving_points": mean("lk_moving_points"),
         "mean_moving_speed_px_per_frame": mean("lk_mean_moving_magnitude_px"), "mean_ms_per_frame": mean("lk_ms"),
         "output": "Sparse motion vectors (direction and speed) at corner points"},
    ]
    keep(utils.save_csv(out_dir / "motion_comparison.csv", summary))
    return {
        "video": info,
        "processed_frames": len(rows) + 1,
        "processed_size": f"{frame_size[1]}x{frame_size[0]}" if frame_size else None,
        "snapshot_frames": sorted(snapshots),
        "peak_motion_frame": peak_index,
        "summary": summary,
        "mean_mask_iou": mean("mask_iou"),
        "mean_lk_moving_in_gmm_pct": mean("lk_moving_in_gmm_pct"),
        "files": files,
    }


def run(paths: utils.ProjectPaths, options: dict | None = None) -> dict:
    out_dir = paths.unit_dir(5, "motion")
    video_path = utils.require_input(paths.input_dir, "original_video")
    LOG.info("Processing video %s", video_path.name)
    result = process_video(video_path, out_dir, config.MOTION, paths.output_dir)
    results = {"task": "Unit 5B - Video motion estimation", "status": "completed", "parameters": config.MOTION,
               **result}
    utils.save_json(out_dir / "motion_results.json", results)
    return results
