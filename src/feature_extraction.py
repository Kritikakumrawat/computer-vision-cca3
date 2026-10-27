"""Unit 4 - Feature extraction and image matching.

Steps:
1. HOG (Histogram of Oriented Gradients) features + visualisation.
2. SIFT keypoint detection and 128-value descriptors.
3. SIFT matching between the original photo and the second photo
   (brute-force matcher + Lowe's ratio test, RANSAC to count geometric inliers).
4. Matching repeated after rotating, resizing and re-lighting one image.

When a transformed copy of the SAME photo is matched, the true transform is
known, so we can also measure how many matches land at the correct position.
"""

from __future__ import annotations

import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.feature import hog

from src import config, utils

LOG = logging.getLogger("cv_assignment.unit4")


# ------------------------------------------------------------------ HOG
def compute_hog(gray: np.ndarray, orientations: int = 9, pixels_per_cell: int = 16, cells_per_block: int = 2):
    """Return (feature vector, visualisation image as uint8)."""
    features, hog_image = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    upper = float(np.percentile(hog_image, 99.5)) or 1.0
    hog_view = exposure.rescale_intensity(hog_image, in_range=(0, upper), out_range=(0, 255)).astype(np.uint8)
    return features, hog_view


def expected_hog_length(height: int, width: int, orientations: int, ppc: int, cpb: int) -> int:
    """Blocks across x blocks down x cells per block x orientations."""
    cells_y, cells_x = height // ppc, width // ppc
    blocks_y, blocks_x = cells_y - cpb + 1, cells_x - cpb + 1
    return max(0, blocks_y) * max(0, blocks_x) * cpb * cpb * orientations


# ------------------------------------------------------------------ SIFT
def create_sift():
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV build has no SIFT. Install opencv-python>=4.4 (see requirements.txt).")
    return cv2.SIFT_create()


def detect_and_describe(image: np.ndarray, sift=None):
    """Detect SIFT keypoints and compute their 128-dimensional descriptors."""
    sift = sift or create_sift()
    keypoints, descriptors = sift.detectAndCompute(utils.to_gray(image), None)
    return keypoints, descriptors


def ratio_test_matches(desc1, desc2, ratio: float = 0.75, matcher: str = "bf") -> list:
    """For each descriptor find the 2 nearest neighbours; keep the match only if the
    best one is clearly better than the second best (Lowe's ratio test)."""
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    if matcher == "bf":
        engine = cv2.BFMatcher(cv2.NORM_L2)
    elif matcher == "flann":
        engine = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))  # 1 = KD-tree
    else:
        raise ValueError(f"Unknown matcher '{matcher}' (use 'bf' or 'flann')")
    pairs = engine.knnMatch(desc1, desc2, k=2)
    return [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]


def geometric_inliers(kp1, kp2, matches, model: str = "homography", threshold: float = 5.0) -> np.ndarray:
    """RANSAC: keep matches that agree with one geometric transform.

    'homography' suits a flat scene or a pure image transform (rotation/resize).
    'fundamental' suits two photos of a 3-D scene taken from different positions.
    """
    needed = 4 if model == "homography" else 8
    if len(matches) < needed:
        return np.zeros(len(matches), dtype=bool)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(config.RANDOM_SEED)  # RANSAC samples randomly; fix it for repeatable numbers
    if model == "homography":
        _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold)
    else:
        _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, threshold, 0.99)
    if mask is None:
        return np.zeros(len(matches), dtype=bool)
    return mask.ravel().astype(bool)


def correct_match_stats(kp1, kp2, matches, true_h: np.ndarray, tolerance: float = 3.0) -> tuple[float, float]:
    """With a known transform: % of matches whose position error is within `tolerance` px."""
    if not matches:
        return 0.0, float("nan")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    projected = cv2.perspectiveTransform(pts1, true_h).reshape(-1, 2)
    errors = np.linalg.norm(projected - pts2, axis=1)
    return float((errors <= tolerance).mean() * 100.0), float(np.median(errors))


# ------------------------------------------------------------------ transforms (with ground truth)
def rotate_image(image: np.ndarray, angle: float):
    """Rotate around the centre without cropping. Returns (image, 3x3 transform)."""
    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    matrix[0, 2] += new_w / 2.0 - centre[0]
    matrix[1, 2] += new_h / 2.0 - centre[1]
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return rotated, np.vstack([matrix, [0.0, 0.0, 1.0]])


def resize_image(image: np.ndarray, scale: float):
    h, w = image.shape[:2]
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    sx, sy = new_w / w, new_h / h
    # OpenCV maps pixel centres: x' = (x + 0.5) * s - 0.5
    transform = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]])
    return resized, transform


def adjust_gamma(image: np.ndarray, gamma: float):
    """gamma < 1 brightens, gamma > 1 darkens. Geometry is unchanged (identity transform)."""
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table), np.eye(3)


def synthetic_viewpoint(image: np.ndarray):
    """PRELIMINARY substitute for a second photo: perspective tilt + lighting change + slight blur.

    This is NOT an independent photograph; it only lets the matching code be
    tested until inputs/second_photo.jpg is supplied.
    """
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0.07 * w, 0.05 * h], [0.97 * w, 0.0], [0.91 * w, 0.97 * h], [0.03 * w, 0.92 * h]])
    transform = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, transform, (w, h), borderValue=(0, 0, 0))
    relit = cv2.convertScaleAbs(warped, alpha=0.85, beta=30)
    return cv2.GaussianBlur(relit, (3, 3), 0), transform


# ------------------------------------------------------------------ drawing
def draw_keypoints(image: np.ndarray, keypoints) -> np.ndarray:
    """Circles show each keypoint's scale; the radius line shows its orientation."""
    return cv2.drawKeypoints(image, keypoints, None, color=(0, 220, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def draw_matches(img1, kp1, img2, kp2, matches, inliers, max_drawn: int = 60, max_width: int = 1800) -> np.ndarray:
    """Side-by-side picture: green lines = RANSAC inliers, red lines = rejected matches."""
    img1, img2 = utils.to_bgr(img1), utils.to_bgr(img2)
    h = max(img1.shape[0], img2.shape[0])
    canvas = np.zeros((h, img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    canvas[: img1.shape[0], : img1.shape[1]] = img1
    canvas[: img2.shape[0], img1.shape[1]:] = img2
    offset = img1.shape[1]
    thickness = max(1, canvas.shape[1] // 900)
    radius = max(3, canvas.shape[1] // 450)
    good = sorted([m for m, ok in zip(matches, inliers) if ok], key=lambda m: m.distance)[:max_drawn]
    bad = sorted([m for m, ok in zip(matches, inliers) if not ok], key=lambda m: m.distance)[: max(5, max_drawn // 4)]
    for group, color in ((bad, (40, 40, 230)), (good, (40, 200, 40))):
        for m in group:
            p1 = tuple(int(v) for v in kp1[m.queryIdx].pt)
            x2, y2 = kp2[m.trainIdx].pt
            p2 = (int(x2) + offset, int(y2))
            cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
            cv2.circle(canvas, p1, radius, color, thickness, cv2.LINE_AA)
            cv2.circle(canvas, p2, radius, color, thickness, cv2.LINE_AA)
    canvas, _ = utils.resize_to_max_side(canvas, max_width)
    return canvas


# ------------------------------------------------------------------ one matching experiment
def match_experiment(name, img1, features1, img2, ratio, model, ransac_threshold, true_transform=None,
                     tolerance=3.0, matcher="bf", sift=None):
    """Detect on img2, match against precomputed features of img1 and return statistics."""
    kp1, desc1 = features1
    with utils.timer() as t_detect:
        kp2, desc2 = detect_and_describe(img2, sift)
    with utils.timer() as t_match:
        matches = ratio_test_matches(desc1, desc2, ratio, matcher)
    inliers = geometric_inliers(kp1, kp2, matches, model, ransac_threshold)
    stats = {
        "experiment": name,
        "image2_size": f"{img2.shape[1]}x{img2.shape[0]}",
        "keypoints_1": len(kp1),
        "keypoints_2": len(kp2),
        "ratio_test_matches": len(matches),
        "match_ratio_pct": 100.0 * len(matches) / max(1, min(len(kp1), len(kp2))),
        "ransac_model": model,
        "ransac_inliers": int(inliers.sum()),
        "inlier_ratio_pct": 100.0 * inliers.sum() / max(1, len(matches)),
        "matcher": matcher,
        "detect_ms": t_detect["ms"],
        "match_ms": t_match["ms"],
    }
    if true_transform is not None:
        correct, median_error = correct_match_stats(kp1, kp2, matches, true_transform, tolerance)
        stats.update({"correct_matches_pct": correct, "median_position_error_px": median_error})
    return stats, (kp2, matches, inliers)


# ------------------------------------------------------------------ figures
def _sweep_figure(rotation_rows, scale_rows, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    for row_index, (rows, key, xlabel) in enumerate(
        [(rotation_rows, "value", "Rotation angle (degrees)"), (scale_rows, "value", "Scale factor")]
    ):
        xs = [r[key] for r in rows]
        ax = axes[row_index][0]
        ends = []
        for color, field, label in ((utils.SERIES_COLORS[0], "ratio_test_matches", "Ratio-test matches"),
                                    (utils.SERIES_COLORS[1], "ransac_inliers", "RANSAC inliers")):
            ys = [r[field] for r in rows]
            ax.plot(xs, ys, marker="o", color=color, label=label)
            ends.append((ys[-1], label.split()[0]))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of matches")
        ax.set_ylim(bottom=0)
        ax2 = axes[row_index][1]
        ys = [r["correct_matches_pct"] for r in rows]
        ax2.plot(xs, ys, marker="o", color=utils.SERIES_COLORS[2])
        ax2.set_ylim(0, 105)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Correct matches (%)")
        if rows is scale_rows:
            for a in (ax, ax2):
                a.set_xscale("log", base=2)
                a.set_xticks(xs)
                a.set_xticklabels([f"{x:g}" for x in xs])
                a.minorticks_off()
            ax.set_xlim(xs[0] / 1.3, xs[-1] * 1.9)
        else:
            ax.set_xlim(xs[0] - 10, xs[-1] + 35)
        utils.label_line_ends(ax, xs[-1], ends)
        title = "Rotation" if row_index == 0 else "Resizing"
        ax.set_title(f"{title}: how many matches survive", fontweight="bold")
        ax2.set_title(f"{title}: are the matches correct?", fontweight="bold")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("SIFT matching: original photo versus transformed copies of itself", y=1.06, fontweight="bold")
    fig.tight_layout()
    return utils.finish_figure(fig, path)


# ------------------------------------------------------------------ main task
def run(paths: utils.ProjectPaths, options: dict | None = None) -> dict:
    params = config.FEATURES
    out_dir = paths.unit_dir(4)
    files = []
    pending = []

    def keep(path):
        files.append(utils.relative_to_output(path, paths.output_dir))
        return path

    photo_path = utils.require_input(paths.input_dir, "original_photo")
    original = utils.read_image(photo_path)
    second_path = utils.find_input(paths.input_dir, "second_photo")
    second = utils.read_image(second_path) if second_path else None
    if second is None:
        pending.append("Original vs second photograph matching (inputs/second_photo.jpg not supplied)")
        LOG.warning("second_photo not found - the two-photo experiment is PENDING; "
                    "a clearly labelled synthetic view is used for a preliminary test.")

    # --- 1. HOG
    hog_rows = []
    hog_views = {}
    images_for_hog = {"Original photo": original}
    if second is not None:
        images_for_hog["Second photo"] = second
    for label, image in images_for_hog.items():
        small, _ = utils.resize_to_max_side(utils.to_gray(image), params["hog_max_side"])
        for ppc in params["hog_pixels_per_cell"]:
            with utils.timer() as t:
                features, view = compute_hog(small, params["hog_orientations"], ppc, params["hog_cells_per_block"])
            hog_views[(label, ppc)] = (small, view)
            hog_rows.append({
                "image": label, "hog_input_size": f"{small.shape[1]}x{small.shape[0]}", "pixels_per_cell": ppc,
                "cells": f"{small.shape[1] // ppc}x{small.shape[0] // ppc}", "feature_length": int(features.size),
                "expected_length": expected_hog_length(small.shape[0], small.shape[1], params["hog_orientations"], ppc,
                                                       params["hog_cells_per_block"]),
                "time_ms": t["ms"],
            })
    keep(utils.save_csv(out_dir / "hog_statistics.csv", hog_rows))
    mid_ppc = params["hog_pixels_per_cell"][len(params["hog_pixels_per_cell"]) // 2]
    small, view = hog_views[("Original photo", mid_ppc)]
    keep(utils.write_image(out_dir / f"01_hog_visualization_ppc{mid_ppc}.png", view))
    row_mid = next(r for r in hog_rows if r["image"] == "Original photo" and r["pixels_per_cell"] == mid_ppc)
    hog_images = [small] + [hog_views[("Original photo", p)][1] for p in params["hog_pixels_per_cell"]]
    hog_titles = [f"Grayscale input ({small.shape[1]}x{small.shape[0]})"] + [
        f"HOG, {p}x{p}-pixel cells\n{next(r['feature_length'] for r in hog_rows if r['image'] == 'Original photo' and r['pixels_per_cell'] == p):,} features"
        for p in params["hog_pixels_per_cell"]]
    if second is not None:
        s_small, s_view = hog_views[("Second photo", mid_ppc)]
        hog_images += [s_small, s_view]
        hog_titles += ["Second photo (grayscale)", f"Second photo HOG, {mid_ppc}x{mid_ppc} cells"]
    fig = utils.image_grid_figure(
        hog_images, hog_titles, "Histogram of Oriented Gradients (HOG)",
        caption=(f"Each cell shows its dominant gradient directions as short white lines; brighter lines mean stronger "
                 f"edges. {params['hog_orientations']} orientation bins, {params['hog_cells_per_block']}x"
                 f"{params['hog_cells_per_block']} cells per block (L2-Hys normalisation). Smaller cells keep more "
                 f"detail but give much longer feature vectors."),
        cols=2, panel_width=4.4,
    )
    keep(utils.finish_figure(fig, out_dir / "fig_hog.png"))

    # --- 2. SIFT keypoints and descriptors
    sift = create_sift()
    with utils.timer() as t_sift:
        kp_orig, desc_orig = detect_and_describe(original, sift)
    if desc_orig is None or len(kp_orig) < 10:
        raise ValueError("Too few SIFT keypoints in the original photo for matching (need at least 10).")
    keep(utils.write_image(out_dir / "02_sift_keypoints_original.jpg", draw_keypoints(original, kp_orig)))
    sift_rows = [{"image": "Original photo", "size": f"{original.shape[1]}x{original.shape[0]}",
                  "keypoints": len(kp_orig), "descriptor_shape": f"{desc_orig.shape[0]}x{desc_orig.shape[1]}",
                  "time_ms": t_sift["ms"]}]
    kp_second = desc_second = None
    if second is not None:
        with utils.timer() as t2:
            kp_second, desc_second = detect_and_describe(second, sift)
        keep(utils.write_image(out_dir / "03_sift_keypoints_second.jpg", draw_keypoints(second, kp_second)))
        sift_rows.append({"image": "Second photo", "size": f"{second.shape[1]}x{second.shape[0]}",
                          "keypoints": len(kp_second),
                          "descriptor_shape": f"{0 if desc_second is None else desc_second.shape[0]}x128",
                          "time_ms": t2["ms"]})
    keep(utils.save_csv(out_dir / "sift_keypoints.csv", sift_rows))

    strongest = int(np.argmax([kp.response for kp in kp_orig]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    axes[0].imshow(cv2.cvtColor(draw_keypoints(original, kp_orig), cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"{len(kp_orig):,} SIFT keypoints (circle = scale, line = orientation)")
    axes[0].axis("off")
    axes[1].bar(np.arange(128), desc_orig[strongest], color=utils.SERIES_COLORS[0], width=0.8)
    for boundary in range(8, 128, 8):
        axes[1].axvline(boundary - 0.5, color=utils.GRID, linewidth=0.6)
    axes[1].set_xlim(-1, 128)
    axes[1].set_xlabel("Descriptor element (16 sub-regions x 8 orientation bins)")
    axes[1].set_ylabel("Value")
    axes[1].set_title("128-value descriptor of the strongest keypoint")
    fig.suptitle("SIFT keypoints and descriptors - original photo", fontweight="bold")
    fig.tight_layout()
    keep(utils.finish_figure(fig, out_dir / "fig_sift_keypoints.png"))

    features_orig = (kp_orig, desc_orig)
    ratio, ransac = params["sift_ratio"], params["ransac_threshold"]
    tol = params["correct_match_tolerance"]
    match_rows = []
    match_images = []

    # --- 3. Matching between the two photographs (or a preliminary synthetic view)
    if second is not None:
        stats, (kp2, m, inl) = match_experiment("Original vs second photo", original, features_orig, second, ratio,
                                                "fundamental", ransac, sift=sift)
        stats["pair_type"] = "real two-photo pair"
        match_rows.append(stats)
        img = draw_matches(original, kp_orig, second, kp2, m, inl, params["max_matches_drawn"])
        keep(utils.write_image(out_dir / "04_matches_original_vs_second.jpg", img))
        match_images.append((img, f"Original vs second photo: {stats['ratio_test_matches']} matches, "
                                  f"{stats['ransac_inliers']} inliers"))
        main_pair = ("Original vs second photo", second, "fundamental", None)
    else:
        view_img, view_h = synthetic_viewpoint(original)
        keep(utils.write_image(out_dir / "04_PRELIMINARY_synthetic_view_of_original.jpg", view_img))
        stats, (kp2, m, inl) = match_experiment(
            "PRELIMINARY: original vs synthetic tilted/relit copy", original, features_orig, view_img, ratio,
            "homography", ransac, view_h, tol, sift=sift)
        stats["pair_type"] = "preliminary synthetic substitute (NOT a second photograph)"
        match_rows.append(stats)
        img = draw_matches(original, kp_orig, view_img, kp2, m, inl, params["max_matches_drawn"])
        keep(utils.write_image(out_dir / "05_PRELIMINARY_matches_synthetic_view.jpg", img))
        match_images.append((img, f"PRELIMINARY synthetic view (not a second photo): {stats['ratio_test_matches']} "
                                  f"matches, {stats['correct_matches_pct']:.1f}% correct"))
        main_pair = ("PRELIMINARY synthetic view", view_img, "homography", view_h)

    # BF versus FLANN on the main pair
    matcher_rows = []
    for matcher in ("bf", "flann"):
        s, _ = match_experiment(main_pair[0], original, features_orig, main_pair[1], ratio, main_pair[2], ransac,
                                main_pair[3], tol, matcher=matcher, sift=sift)
        s["matcher"] = "Brute force (BFMatcher, L2)" if matcher == "bf" else "FLANN (KD-tree, 5 trees, 50 checks)"
        matcher_rows.append(s)
    keep(utils.save_csv(out_dir / "matcher_comparison.csv", matcher_rows))

    # --- 4. Invariance experiments on the original (ground truth known)
    rotation_rows, scale_rows, gamma_rows = [], [], []
    showcase = {}
    for angle in params["rotation_angles"]:
        img2, h_true = rotate_image(original, angle)
        s, extra = match_experiment(f"Rotated {angle} deg", original, features_orig, img2, ratio, "homography",
                                    ransac, h_true, tol, sift=sift)
        s.update({"transform": "rotation", "value": angle})
        rotation_rows.append(s)
        if angle == 45:
            showcase["rotation"] = (img2, extra, s)
    for scale in params["scale_factors"]:
        img2, h_true = resize_image(original, scale)
        s, extra = match_experiment(f"Resized x{scale:g}", original, features_orig, img2, ratio, "homography",
                                    ransac, h_true, tol, sift=sift)
        s.update({"transform": "resize", "value": scale})
        scale_rows.append(s)
        if scale == 0.5:
            showcase["resize"] = (img2, extra, s)
    for gamma in params["gamma_values"]:
        img2, h_true = adjust_gamma(original, gamma)
        s, extra = match_experiment(f"Gamma {gamma:g} ({'brighter' if gamma < 1 else 'darker'})", original,
                                    features_orig, img2, ratio, "homography", ransac, h_true, tol, sift=sift)
        s.update({"transform": "illumination", "value": gamma})
        gamma_rows.append(s)
        if gamma == params["gamma_values"][-1]:
            showcase["illumination"] = (img2, extra, s)
    invariance_rows = rotation_rows + scale_rows + gamma_rows
    keep(utils.save_csv(out_dir / "transform_matching_original.csv", invariance_rows))
    keep(_sweep_figure(rotation_rows, scale_rows, out_dir / "fig_rotation_scale_sweep.png"))

    for key, (img2, (kp2, m, inl), s) in showcase.items():
        img = draw_matches(original, kp_orig, img2, kp2, m, inl, params["max_matches_drawn"])
        keep(utils.write_image(out_dir / f"06_matches_original_vs_{key}.jpg", img))
        match_images.append((img, f"Original vs {s['experiment'].lower()} copy: {s['ratio_test_matches']} matches, "
                                  f"{s['correct_matches_pct']:.1f}% correct"))

    # --- 5. Transformations of the second photo (only with a real second photo)
    second_rows = []
    if second is not None:
        baseline = match_rows[0]
        second_rows.append({**baseline, "transform": "none", "value": 0})
        for label, fn, value in (("rotation", rotate_image, 45), ("rotation", rotate_image, 90),
                                 ("resize", resize_image, 0.5), ("resize", resize_image, 1.5)):
            img2, _ = fn(second, value)
            s, extra = match_experiment(f"Original vs second photo {label} {value:g}", original, features_orig, img2,
                                        ratio, "fundamental", ransac, sift=sift)
            s.update({"transform": label, "value": value})
            second_rows.append(s)
        keep(utils.save_csv(out_dir / "transform_matching_second_photo.csv", second_rows))
    else:
        pending.append("Rotation/resizing of the second photograph (needs inputs/second_photo.jpg)")

    # Overview figure of the match pictures
    fig, axes = plt.subplots(len(match_images), 1, figsize=(11, 3.9 * len(match_images)))
    axes = np.atleast_1d(axes)
    for ax, (img, title) in zip(axes, match_images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("SIFT feature matching (green = RANSAC inliers, red = rejected matches)", fontweight="bold")
    fig.tight_layout()
    keep(utils.finish_figure(fig, out_dir / "fig_matching_overview.png"))

    keep(utils.save_csv(out_dir / "pair_matching.csv", match_rows))
    results = {
        "task": "Unit 4 - Feature extraction and matching",
        "status": "completed" if not pending else "partial",
        "inputs": {"original_photo": photo_path.name, "second_photo": second_path.name if second_path else None},
        "parameters": params,
        "hog": hog_rows,
        "sift": sift_rows,
        "pair_matching": match_rows,
        "matcher_comparison": matcher_rows,
        "transform_matching_original": invariance_rows,
        "transform_matching_second_photo": second_rows,
        "pending": pending,
        "files": files,
    }
    utils.save_json(out_dir / "unit4_results.json", results)
    return results
