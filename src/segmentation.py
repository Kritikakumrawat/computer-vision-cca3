"""Unit 3B - Morphological processing and segmentation.

Steps:
1. Grayscale conversion of the portrait.
2. Binary thresholding (Otsu, plus adaptive thresholding for comparison).
3. Erosion, dilation, opening and closing of the binary mask.
4. Canny edge detection (fixed and automatic thresholds, clean vs noisy input).
5. Watershed segmentation:
   a. touching/multiple objects - your own `inputs/watershed_photo.jpg` if present,
      otherwise the scikit-image "coins" sample photograph (clearly labelled);
   b. marker-based person/background watershed on the portrait itself.
"""

from __future__ import annotations

import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from src import config, utils
from src.image_restoration import add_gaussian_noise, median_filter

LOG = logging.getLogger("cv_assignment.unit3.segmentation")

KERNEL_SHAPES = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}


# ------------------------------------------------------------------ thresholding
def otsu_threshold(gray: np.ndarray, blur_kernel: int = 5, invert: bool = True) -> tuple[float, np.ndarray]:
    """Otsu picks the threshold that best separates the histogram into two classes.

    With invert=True the darker class becomes white (255) in the mask.
    """
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    value, mask = cv2.threshold(blurred, 0, 255, mode + cv2.THRESH_OTSU)
    return float(value), mask


def adaptive_threshold(gray: np.ndarray, block_size: int = 31, c: int = 5, blur_kernel: int = 5) -> np.ndarray:
    """Threshold each pixel against the weighted mean of its own neighbourhood."""
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)


# ------------------------------------------------------------------ morphology
def structuring_element(size: int, shape: str = "ellipse") -> np.ndarray:
    return cv2.getStructuringElement(KERNEL_SHAPES[shape], (size, size))


def apply_morphology(mask: np.ndarray, kernel: np.ndarray) -> dict[str, np.ndarray]:
    """Return erosion, dilation, opening (erode then dilate) and closing (dilate then erode)."""
    return {
        "Erosion": cv2.erode(mask, kernel),
        "Dilation": cv2.dilate(mask, kernel),
        "Opening": cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel),
        "Closing": cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel),
    }


def mask_statistics(mask: np.ndarray) -> dict:
    """Foreground share, number of white blobs and number of enclosed holes."""
    binary = (mask > 0).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blob_areas = stats[1:, cv2.CC_STAT_AREA]
    # Holes = black regions that do not touch the image border.
    inverse = 1 - binary
    n_inv, inv_labels = cv2.connectedComponents(inverse, connectivity=4)
    border_labels = set(np.unique(np.concatenate(
        [inv_labels[0, :], inv_labels[-1, :], inv_labels[:, 0], inv_labels[:, -1]])).tolist())
    holes = len([lab for lab in range(1, n_inv) if lab not in border_labels])
    return {
        "foreground_pct": float(binary.mean() * 100.0),
        "blobs": int(n_labels - 1),
        "small_blobs_lt_50px": int((blob_areas < 50).sum()),
        "holes": int(holes),
    }


# ------------------------------------------------------------------ edges
def canny_edges(gray: np.ndarray, low: int, high: int, blur_kernel: int = 5) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    return cv2.Canny(blurred, low, high)


def auto_canny_thresholds(gray: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    """Common heuristic: thresholds around the median grey level."""
    median = float(np.median(gray))
    return int(max(0, (1.0 - sigma) * median)), int(min(255, (1.0 + sigma) * median))


def edge_density(edges: np.ndarray) -> float:
    return float((edges > 0).mean() * 100.0)


# ------------------------------------------------------------------ watershed
def watershed_objects(image: np.ndarray, min_distance_fraction: float = 0.04,
                      min_object_fraction: float = 0.0005, background_kernel_fraction: float = 0.25) -> dict:
    """Separate touching objects with the distance-transform watershed.

    1. Background correction (top-hat) evens out uneven lighting, then Otsu
       thresholding makes the objects white.
    2. Opening + hole filling cleans the mask.
    3. Distance transform: each object pixel gets its distance to the background.
       Object centres are bright peaks.
    4. Each peak becomes a marker; watershed grows markers until they meet,
       which draws a dividing line between touching objects.
    """
    gray = utils.to_gray(image)
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Objects are assumed brighter than the background if the image is brighter than its border.
    border_mean = float(np.concatenate([blurred[0, :], blurred[-1, :], blurred[:, 0], blurred[:, -1]]).mean())
    objects_bright = float(blurred.mean()) > border_mean
    if background_kernel_fraction > 0:
        size = int(background_kernel_fraction * min(h, w)) | 1  # odd size
        operation = cv2.MORPH_TOPHAT if objects_bright else cv2.MORPH_BLACKHAT
        corrected = cv2.morphologyEx(blurred, operation, structuring_element(size, "ellipse"))
    else:
        corrected = blurred if objects_bright else cv2.bitwise_not(blurred)
    # After correction the objects are bright on a dark, even background.
    otsu_value, binary = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, structuring_element(3, "ellipse"), iterations=2)  # remove specks
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, structuring_element(5, "ellipse"), iterations=2)  # rejoin cracks
    filled = ndi.binary_fill_holes(binary > 0)
    # Drop specks smaller than the minimum object size
    min_area = max(20, int(min_object_fraction * h * w))
    labels_cc, n_cc = ndi.label(filled)
    sizes = ndi.sum(filled, labels_cc, range(1, n_cc + 1))
    keep = np.isin(labels_cc, [i + 1 for i, s in enumerate(sizes) if s >= min_area])
    clean = keep.astype(np.uint8) * 255
    labels_cc, components = ndi.label(clean > 0)

    distance = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    min_distance = max(3, int(min_distance_fraction * min(h, w)))
    peaks = peak_local_max(distance, min_distance=min_distance, labels=labels_cc, exclude_border=False)
    markers = np.zeros((h, w), dtype=np.int32)
    for index, (row, col) in enumerate(peaks, start=1):
        markers[row, col] = index
    labels = watershed(-distance, markers, mask=clean > 0)

    regions = int(len(np.unique(labels)) - (1 if 0 in labels else 0))
    areas = [int((labels == lab).sum()) for lab in np.unique(labels) if lab != 0]
    return {
        "gray": gray, "corrected": corrected, "binary": clean, "distance": distance, "markers": markers,
        "labels": labels, "objects_bright": objects_bright, "background_kernel_fraction": background_kernel_fraction,
        "otsu_value": float(otsu_value), "connected_components": int(components), "markers_found": int(len(peaks)),
        "watershed_regions": regions, "min_distance_px": min_distance,
        "region_area_min": min(areas) if areas else 0, "region_area_max": max(areas) if areas else 0,
        "region_area_median": float(np.median(areas)) if areas else 0.0,
    }


def colorize_labels(labels: np.ndarray, base_gray: np.ndarray, seed: int = config.RANDOM_SEED) -> np.ndarray:
    """Overlay each watershed region in its own colour and draw region borders in red."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(60, 255, size=(int(labels.max()) + 1, 3), dtype=np.uint8)
    palette[0] = 0
    color = palette[labels]
    base = utils.to_bgr(base_gray)
    overlay = np.where(labels[..., None] > 0, (0.45 * base + 0.55 * color).astype(np.uint8), base)
    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundaries[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    overlay[boundaries] = (0, 0, 255)
    return overlay


def watershed_person_background(image_bgr: np.ndarray, border_fraction: float = 0.02,
                                bg_distance: float = 12.0, fg_distance: float = 40.0) -> dict:
    """Marker-based watershed separating the person from a plain background.

    The background colour is learned from a thin border strip. Pixels very
    close to it (and connected to the border) are 'sure background'; pixels
    very different from it are 'sure person'. cv2.watershed floods the
    uncertain band between them along the image gradient.
    """
    h, w = image_bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(2, int(border_fraction * min(h, w)))
    border_pixels = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                                    lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg_color = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(lab - bg_color, axis=2)

    near_bg = (distance < bg_distance).astype(np.uint8)
    _, cc = cv2.connectedComponents(near_bg)
    touching = np.unique(np.concatenate([cc[0, :], cc[-1, :], cc[:, 0], cc[:, -1]]))
    sure_bg = np.isin(cc, touching[touching > 0]) & (near_bg > 0)
    sure_bg = cv2.erode(sure_bg.astype(np.uint8), structuring_element(5, "ellipse")) > 0

    sure_fg = cv2.erode((distance > fg_distance).astype(np.uint8), structuring_element(15, "ellipse")) > 0

    markers = np.zeros((h, w), dtype=np.int32)
    markers[sure_bg] = 1
    markers[sure_fg] = 2
    flooded = cv2.watershed(image_bgr.copy(), markers.copy())
    person = (flooded == 2).astype(np.uint8) * 255
    # Fill small holes left inside the person (e.g. specular highlights)
    person = (ndi.binary_fill_holes(person > 0).astype(np.uint8)) * 255

    marker_view = np.full((h, w, 3), 128, dtype=np.uint8)  # grey = unknown
    marker_view[sure_bg] = (230, 160, 60)  # blue-ish = background marker (BGR)
    marker_view[sure_fg] = (60, 120, 235)  # orange = person marker (BGR)
    overlay = image_bgr.copy()
    tint = np.zeros_like(image_bgr)
    tint[person > 0] = (60, 120, 235)
    overlay = cv2.addWeighted(overlay, 1.0, tint, 0.35, 0)
    contours, _ = cv2.findContours(person, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), max(2, w // 400))
    n_person_regions = len([c for c in contours if cv2.contourArea(c) > 0.01 * h * w])
    return {
        "distance": distance, "marker_view": marker_view, "person_mask": person, "overlay": overlay,
        "background_lab": bg_color.tolist(), "sure_bg_pct": float(sure_bg.mean() * 100),
        "sure_fg_pct": float(sure_fg.mean() * 100), "person_pct": float((person > 0).mean() * 100),
        "large_person_regions": int(n_person_regions),
    }


def synthetic_touching_discs(size: int = 300) -> np.ndarray:
    """SYNTHETIC teaching image: three overlapping white discs that form ONE connected blob."""
    image = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(image, (int(0.36 * size), int(0.42 * size)), int(0.20 * size), 255, -1)
    cv2.circle(image, (int(0.66 * size), int(0.42 * size)), int(0.17 * size), 255, -1)
    cv2.circle(image, (int(0.50 * size), int(0.72 * size)), int(0.14 * size), 255, -1)
    return image


def load_watershed_image(input_dir) -> tuple[np.ndarray, str, str]:
    """Return (image, source label, description). Prefers the student's own photo."""
    own = utils.find_input(input_dir, "watershed_photo")
    if own is not None:
        return utils.read_image(own), own.name, "Student's own photograph of multiple/touching objects."
    from skimage import data  # bundled with scikit-image, no download needed

    coins = data.coins()
    return coins, "scikit-image sample 'coins'", (
        "SUPPLEMENTARY EXAMPLE (not the student's photograph): scikit-image's bundled 'coins' image "
        "(Greek coins from Pompeii, Brooklyn Museum collection, no known copyright restrictions), "
        "used because the portrait does not contain multiple touching objects."
    )


# ------------------------------------------------------------------ main task
def run(paths: utils.ProjectPaths, options: dict | None = None) -> dict:
    params = config.SEGMENTATION
    out_dir = paths.unit_dir(3, "segmentation")
    files = []

    def keep(path):
        files.append(utils.relative_to_output(path, paths.output_dir))
        return path

    photo_path = utils.require_input(paths.input_dir, "original_photo")
    original = utils.read_image(photo_path)
    gray = utils.to_gray(original)
    keep(utils.write_image(out_dir / "01_grayscale.png", gray))

    # --- Thresholding
    otsu_value, otsu_mask = otsu_threshold(gray, params["blur_kernel"], invert=True)
    adaptive_mask = adaptive_threshold(gray, params["adaptive_block_size"], params["adaptive_c"], params["blur_kernel"])
    keep(utils.write_image(out_dir / "02_binary_otsu.png", otsu_mask))
    keep(utils.write_image(out_dir / "03_binary_adaptive.png", adaptive_mask))
    LOG.info("Otsu threshold = %.1f", otsu_value)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Grayscale portrait")
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    axes[1].fill_between(np.arange(256), hist, color=utils.SERIES_COLORS[0], alpha=0.85, linewidth=0)
    axes[1].axvline(otsu_value, color=utils.INK, linestyle="--", linewidth=1.2)
    axes[1].annotate(f"Otsu threshold = {otsu_value:.0f}", (otsu_value, hist.max() * 0.9), xytext=(6, 0),
                     textcoords="offset points", fontsize=8, color=utils.INK_SECONDARY)
    axes[1].set_xlim(0, 255)
    axes[1].set_xlabel("Grey level")
    axes[1].set_ylabel("Pixel count")
    axes[1].set_title("Grey-level histogram")
    axes[2].imshow(otsu_mask, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"Otsu binary mask (dark regions = white)")
    axes[3].imshow(adaptive_mask, cmap="gray", vmin=0, vmax=255)
    axes[3].set_title("Adaptive threshold (local)")
    for ax in (axes[0], axes[2], axes[3]):
        ax.axis("off")
    axes[1].set_box_aspect(gray.shape[0] / gray.shape[1])
    fig.suptitle("Binary thresholding of the portrait", fontweight="bold")
    fig.text(0.5, -0.02, "Otsu uses one global threshold chosen from the histogram; adaptive thresholding compares\n"
             "each pixel with its neighbourhood and therefore keeps texture and edges instead of large regions.",
             ha="center", fontsize=8.5, color=utils.INK_SECONDARY)
    fig.tight_layout()
    keep(utils.finish_figure(fig, out_dir / "fig_thresholding.png"))

    # --- Morphology on the Otsu mask
    kernel = structuring_element(params["morph_kernel_size"], params["morph_kernel_shape"])
    morph = apply_morphology(otsu_mask, kernel)
    for index, (name, result) in enumerate(morph.items(), start=4):
        keep(utils.write_image(out_dir / f"{index:02d}_{name.lower()}.png", result))
    morph_rows = [{"operation": "Binary mask (Otsu)", **mask_statistics(otsu_mask)}]
    morph_rows += [{"operation": name, **mask_statistics(result)} for name, result in morph.items()]
    keep(utils.save_csv(out_dir / "morphology_statistics.csv", morph_rows))

    ks = params["morph_kernel_size"]
    zoom = params["zoom_box"]
    images = [otsu_mask] + list(morph.values())
    titles = [f"Otsu mask\n{morph_rows[0]['blobs']} blobs, {morph_rows[0]['holes']} holes"] + [
        f"{row['operation']} ({ks}x{ks})\n{row['blobs']} blobs, {row['holes']} holes" for row in morph_rows[1:]]
    zoomed = [utils.crop_fraction(img, zoom) for img in images]
    fig = utils.image_grid_figure(
        images + zoomed, titles + [f"{t.splitlines()[0]} - zoom" for t in titles],
        "Morphological operations on the Otsu mask",
        caption=("Erosion shrinks white regions and removes thin strands; dilation grows them and closes gaps; "
                 "opening (erode then dilate) removes small white specks; closing (dilate then erode) fills small "
                 "black holes. The bottom row zooms into the hair and face region."),
        cols=5, panel_width=3.0,
    )
    keep(utils.finish_figure(fig, out_dir / "fig_morphology.png"))

    # --- Canny edge detection
    low, high = params["canny_low"], params["canny_high"]
    auto_low, auto_high = auto_canny_thresholds(gray, params["canny_auto_sigma"])
    edges_fixed = canny_edges(gray, low, high, params["blur_kernel"])
    edges_auto = canny_edges(gray, auto_low, auto_high, params["blur_kernel"])
    noisy_gray = utils.to_gray(add_gaussian_noise(original, config.RESTORATION["gaussian_noise_sigma"], config.RANDOM_SEED))
    edges_noisy = canny_edges(noisy_gray, low, high, params["blur_kernel"])
    edges_noisy_median = canny_edges(median_filter(noisy_gray, 5), low, high, params["blur_kernel"])
    keep(utils.write_image(out_dir / "08_canny_edges.png", edges_fixed))
    keep(utils.write_image(out_dir / "09_canny_auto_thresholds.png", edges_auto))
    keep(utils.write_image(out_dir / "10_canny_on_gaussian_noisy.png", edges_noisy))
    keep(utils.write_image(out_dir / "11_canny_on_median_restored.png", edges_noisy_median))
    canny_rows = [
        {"input": "Clean grayscale", "low": low, "high": high, "edge_pixels_pct": edge_density(edges_fixed)},
        {"input": "Clean grayscale (automatic thresholds)", "low": auto_low, "high": auto_high,
         "edge_pixels_pct": edge_density(edges_auto)},
        {"input": "Gaussian-noisy grayscale", "low": low, "high": high, "edge_pixels_pct": edge_density(edges_noisy)},
        {"input": "Gaussian-noisy after 5x5 median filter", "low": low, "high": high,
         "edge_pixels_pct": edge_density(edges_noisy_median)},
    ]
    keep(utils.save_csv(out_dir / "canny_statistics.csv", canny_rows))
    def printable(edges):  # thicker, dark-on-white copy for the figure only
        return cv2.bitwise_not(cv2.dilate(edges, structuring_element(3, "rect")))

    fig = utils.image_grid_figure(
        [printable(e) for e in (edges_fixed, edges_auto, edges_noisy, edges_noisy_median)],
        [f"{r['input']}\nthresholds {r['low']}/{r['high']} - {r['edge_pixels_pct']:.2f}% edge pixels" for r in canny_rows],
        "Canny edge detection",
        caption=("Canny = Gaussian smoothing, gradient magnitude, non-maximum suppression and hysteresis thresholding. "
                 "Noise creates many false edges; filtering first (median) removes most of them. Edges are drawn dark "
                 "and thickened here for printing; the saved edge maps are the raw 1-pixel output."),
        cols=2, panel_width=4.4,
    )
    keep(utils.finish_figure(fig, out_dir / "fig_canny.png"))

    # --- Watershed (a): multiple / touching objects
    ws_image, ws_source, ws_description = load_watershed_image(paths.input_dir)
    ws_image, _ = utils.resize_to_max_side(ws_image, params["watershed_max_side"])
    ws = watershed_objects(ws_image, params["watershed_min_distance_fraction"], params["watershed_min_object_fraction"],
                           params["watershed_background_kernel_fraction"])
    keep(utils.write_image(out_dir / "12_watershed_input.png", ws_image))
    ws_overlay = colorize_labels(ws["labels"], ws["gray"])
    keep(utils.write_image(out_dir / "13_watershed_objects_result.png", ws_overlay))

    def marker_picture(result, shape):
        view = utils.to_bgr(result["binary"] // 3)
        for row, col in zip(*np.nonzero(result["markers"])):
            cv2.circle(view, (int(col), int(row)), max(2, shape[1] // 120), (0, 0, 255), -1)
        return view

    distance_view = cv2.normalize(ws["distance"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fig = utils.image_grid_figure(
        [utils.to_bgr(ws["gray"]), ws["corrected"], ws["binary"], distance_view, marker_picture(ws, ws_image.shape),
         ws_overlay],
        ["Input: " + ws_source, "Background-corrected (top-hat)",
         f"Otsu mask (cleaned)\n{ws['connected_components']} connected blobs",
         "Distance transform", f"Markers (distance peaks)\n{ws['markers_found']} markers",
         f"Watershed result\n{ws['watershed_regions']} regions"],
        "Watershed segmentation of multiple objects",
        caption=ws_description + " Watershed splits a blob into several regions only when it contains more than one "
                                 "distance peak.",
        cols=3, panel_width=3.6,
    )
    keep(utils.finish_figure(fig, out_dir / "fig_watershed_objects.png"))
    LOG.info("Watershed (%s): %d blobs -> %d regions", ws_source, ws["connected_components"], ws["watershed_regions"])

    # --- Watershed (c): synthetic touching discs (teaching example, clearly labelled)
    discs = synthetic_touching_discs()
    sd = watershed_objects(discs, 0.08, params["watershed_min_object_fraction"], background_kernel_fraction=0)
    disc_overlay = colorize_labels(sd["labels"], discs // 2)
    keep(utils.write_image(out_dir / "16_SYNTHETIC_touching_discs_watershed.png", disc_overlay))
    fig = utils.image_grid_figure(
        [discs, cv2.normalize(sd["distance"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
         marker_picture(sd, discs.shape), disc_overlay],
        [f"SYNTHETIC input\n{sd['connected_components']} connected blob", "Distance transform",
         f"Markers\n{sd['markers_found']} peaks", f"Watershed\n{sd['watershed_regions']} regions"],
        "Watershed on touching objects (synthetic teaching example)",
        caption="Computer-generated image of three overlapping discs, NOT a photograph. Thresholding sees one blob; "
                "the distance-transform peaks let watershed draw a dividing line where the discs touch.",
        cols=4, panel_width=2.8,
    )
    keep(utils.finish_figure(fig, out_dir / "fig_watershed_touching_synthetic.png"))

    # --- Watershed (b): portrait person vs background
    pw = watershed_person_background(original, params["portrait_border_fraction"],
                                     params["portrait_bg_distance"], params["portrait_fg_distance"])
    keep(utils.write_image(out_dir / "14_portrait_watershed_mask.png", pw["person_mask"]))
    keep(utils.write_image(out_dir / "15_portrait_watershed_overlay.jpg", pw["overlay"]))
    dist_view = cv2.normalize(pw["distance"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fig = utils.image_grid_figure(
        [original, dist_view, pw["marker_view"], pw["person_mask"], pw["overlay"]],
        ["Portrait", "Colour distance from background", "Markers: blue = background,\norange = person, grey = unknown",
         f"Person mask\n{pw['person_pct']:.1f}% of image", "Watershed boundary (red)"],
        "Marker-based watershed on the portrait (person versus background)",
        caption=("The portrait has one person on a plain background, so watershed is used here to separate the "
                 "person from the background, not to split touching objects."),
        cols=3, panel_width=3.6,
    )
    keep(utils.finish_figure(fig, out_dir / "fig_watershed_portrait.png"))

    results = {
        "task": "Unit 3B - Morphology and segmentation",
        "status": "completed",
        "input": photo_path.name,
        "parameters": params,
        "otsu_threshold": otsu_value,
        "morphology": morph_rows,
        "canny": canny_rows,
        "watershed_objects": {
            "source": ws_source, "description": ws_description,
            "uses_student_photo": ws_source != "scikit-image sample 'coins'",
            "image_size": [int(ws_image.shape[1]), int(ws_image.shape[0])],
            **{k: v for k, v in ws.items() if not isinstance(v, np.ndarray)},
        },
        "watershed_synthetic_touching_discs": {
            "description": "Computer-generated teaching image (three overlapping discs), not a photograph.",
            **{k: v for k, v in sd.items() if not isinstance(v, np.ndarray)},
        },
        "watershed_portrait": {k: v for k, v in pw.items() if not isinstance(v, np.ndarray)},
        "files": files,
    }
    utils.save_json(out_dir / "segmentation_results.json", results)
    return results
