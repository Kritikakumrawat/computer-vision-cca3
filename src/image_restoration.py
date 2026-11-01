"""Unit 3A - Image restoration.

Steps:
1. Load the original photograph (the clean reference image).
2. Add Gaussian noise and salt-and-pepper noise (fixed random seed).
3. Restore each noisy image with a Gaussian filter and a median filter.
4. Compare every result with the original using PSNR and SSIM.

PSNR (peak signal-to-noise ratio, in dB) and SSIM (structural similarity,
0..1) both get higher when an image is closer to the reference.
"""

from __future__ import annotations

import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src import config, utils

LOG = logging.getLogger("cv_assignment.unit3.restoration")


# ------------------------------------------------------------------ noise
def add_gaussian_noise(image: np.ndarray, sigma: float = 25.0, seed: int = config.RANDOM_SEED) -> np.ndarray:
    """Add zero-mean Gaussian noise (standard deviation `sigma`) to every pixel."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(
    image: np.ndarray, amount: float = 0.05, salt_vs_pepper: float = 0.5, seed: int = config.RANDOM_SEED
) -> np.ndarray:
    """Set a random `amount` of pixels to pure white (salt) or pure black (pepper)."""
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    random_values = rng.random(image.shape[:2])  # one value per pixel location
    pepper_limit = amount * (1.0 - salt_vs_pepper)
    pepper = random_values < pepper_limit
    salt = (random_values >= pepper_limit) & (random_values < amount)
    noisy[pepper] = 0  # all colour channels of that pixel
    noisy[salt] = 255
    return noisy


# ------------------------------------------------------------------ filters
def gaussian_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Weighted average of the neighbourhood (sigma derived from the kernel size)."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def median_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Replace each pixel with the median of its ksize x ksize neighbourhood."""
    return cv2.medianBlur(image, ksize)


FILTERS = {"Gaussian filter": gaussian_filter, "Median filter": median_filter}


# ------------------------------------------------------------------ metrics
def compare_with_reference(reference: np.ndarray, test: np.ndarray) -> dict:
    """PSNR (dB) and SSIM of `test` against `reference` (same size, both BGR or both gray)."""
    if reference.shape != test.shape:
        raise ValueError(f"Images must have the same shape, got {reference.shape} and {test.shape}")
    if reference.ndim == 3:  # compare in RGB order, the usual colour representation
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
        test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    if np.array_equal(reference, test):
        return {"psnr_db": float("inf"), "ssim": 1.0}
    psnr = peak_signal_noise_ratio(reference, test, data_range=255)
    ssim = structural_similarity(
        reference, test, data_range=255, channel_axis=2 if reference.ndim == 3 else None
    )
    return {"psnr_db": float(psnr), "ssim": float(ssim)}


# ------------------------------------------------------------------ figures
def _fmt(metrics: dict) -> str:
    return f"PSNR {metrics['psnr_db']:.2f} dB | SSIM {metrics['ssim']:.3f}"


def _plot_metric_vs_kernel(rows: list[dict], kernel_sizes: list[int], path):
    """2 x 2 small multiples: PSNR and SSIM (rows) for each noise type (columns)."""
    noise_types = ["Gaussian noise", "Salt-and-pepper noise"]
    metrics = [("psnr_db", "PSNR (dB) - higher is better"), ("ssim", "SSIM - higher is better")]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
    for col, noise in enumerate(noise_types):
        for row, (key, label) in enumerate(metrics):
            ax = axes[row][col]
            baseline = next(r[key] for r in rows if r["noise"] == noise and r["method"] == "Noisy (no filter)")
            ax.axhline(baseline, color=utils.INK_MUTED, linestyle="--", linewidth=1.2)
            ax.annotate("noisy input", (kernel_sizes[0], baseline), xytext=(0, 4), textcoords="offset points",
                        fontsize=8, color=utils.INK_MUTED)
            ends = []
            for color, method in zip(utils.SERIES_COLORS, FILTERS):
                values = [next(r[key] for r in rows if r["noise"] == noise and r["method"] == method and r["kernel"] == k)
                          for k in kernel_sizes]
                ax.plot(kernel_sizes, values, marker="o", color=color, label=method)
                ends.append((values[-1], method.split()[0]))
            ax.set_xticks(kernel_sizes)
            ax.set_xlim(kernel_sizes[0] - 0.5, kernel_sizes[-1] + 2.2)
            utils.label_line_ends(ax, kernel_sizes[-1], ends)
            ax.set_xlabel("Kernel size (pixels)")
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(noise, fontweight="bold")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Restoration quality versus filter kernel size", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return utils.finish_figure(fig, path)


# ------------------------------------------------------------------ main task
def run(paths: utils.ProjectPaths, options: dict | None = None) -> dict:
    """Run every restoration experiment and save images, figures and metrics."""
    params = config.RESTORATION
    out_dir = paths.unit_dir(3, "restoration")
    photo_path = utils.require_input(paths.input_dir, "original_photo")
    original = utils.read_image(photo_path)
    h, w = original.shape[:2]
    LOG.info("Loaded original photograph %s (%d x %d)", photo_path.name, w, h)
    files = []

    def save(name, image):
        path = utils.write_image(out_dir / name, image)
        files.append(utils.relative_to_output(path, paths.output_dir))
        return path

    # 1. Display the original photograph
    fig = utils.image_grid_figure(
        [original], [f"{photo_path.name} - {w} x {h} px, 3 colour channels"],
        "Original photograph (clean reference image)",
        caption="The unmodified input photograph. All noisy and restored images are compared against it.",
        panel_width=5.5,
    )
    files.append(utils.relative_to_output(utils.finish_figure(fig, out_dir / "01_original_display.png"), paths.output_dir))

    # 2. Add noise (reproducible thanks to the fixed seed)
    sigma = params["gaussian_noise_sigma"]
    amount = params["salt_pepper_amount"]
    noisy = {
        "Gaussian noise": add_gaussian_noise(original, sigma, config.RANDOM_SEED),
        "Salt-and-pepper noise": add_salt_and_pepper_noise(original, amount, params["salt_vs_pepper"], config.RANDOM_SEED),
    }
    save(f"02_noisy_gaussian_sigma{int(sigma)}.png", noisy["Gaussian noise"])
    save(f"03_noisy_salt_pepper_{int(amount * 100)}pct.png", noisy["Salt-and-pepper noise"])

    # 3 + 4. Restore with both filters at several kernel sizes and measure quality
    rows = []
    restored = {}
    for noise_name, noisy_image in noisy.items():
        rows.append({"noise": noise_name, "method": "Noisy (no filter)", "kernel": None,
                     **compare_with_reference(original, noisy_image), "time_ms": 0.0})
        for method_name, filter_fn in FILTERS.items():
            for k in params["kernel_sizes"]:
                with utils.timer() as t:
                    result = filter_fn(noisy_image, k)
                metrics = compare_with_reference(original, result)
                rows.append({"noise": noise_name, "method": method_name, "kernel": k, **metrics, "time_ms": t["ms"]})
                restored[(noise_name, method_name, k)] = result
                LOG.debug("%-22s %-15s k=%d  %s", noise_name, method_name, k, _fmt(metrics))
    # Compact console summary: one line per noise type and filter (PSNR dB / SSIM per kernel)
    for noise_name in noisy:
        base = next(r for r in rows if r["noise"] == noise_name and r["kernel"] is None)
        LOG.info("%s - noisy input: %s", noise_name, _fmt(base))
        for method_name in FILTERS:
            parts = [f"{r['kernel']}x{r['kernel']} {r['psnr_db']:.2f}/{r['ssim']:.3f}"
                     for r in rows if r["noise"] == noise_name and r["method"] == method_name]
            LOG.info("  %-15s PSNR dB/SSIM: %s", method_name, " | ".join(parts))

    k = params["default_kernel"]
    short = {"Gaussian noise": "gaussian_noise", "Salt-and-pepper noise": "salt_pepper"}
    for index, (noise_name, method_name) in enumerate(
        [(n, m) for n in noisy for m in FILTERS], start=4
    ):
        tag = "gaussian_filter" if method_name.startswith("Gaussian") else "median_filter"
        save(f"{index:02d}_{short[noise_name]}_{tag}_k{k}.png", restored[(noise_name, method_name, k)])

    def metric(noise, method, kernel=None):
        return next(r for r in rows if r["noise"] == noise and r["method"] == method and r["kernel"] == kernel)

    # Comparison figure for each noise type: original | noisy | Gaussian | median
    for noise_name in noisy:
        images = [original, noisy[noise_name]] + [restored[(noise_name, m, k)] for m in FILTERS]
        titles = ["Original (reference)", f"Noisy\n{_fmt(metric(noise_name, 'Noisy (no filter)'))}"] + [
            f"{m} {k}x{k}\n{_fmt(metric(noise_name, m, k))}" for m in FILTERS
        ]
        noise_desc = f"sigma = {sigma:g}" if noise_name == "Gaussian noise" else f"{amount:.0%} of pixels"
        fig = utils.image_grid_figure(
            images, titles, f"{noise_name} ({noise_desc}): noisy versus restored",
            caption=(f"Each restored image is compared with the clean original. Higher PSNR and SSIM mean the "
                     f"result is closer to the original; kernel size {k}x{k} is shown here and the full sweep is "
                     f"in restoration_metrics.csv."),
            cols=2, panel_width=4.4,
        )
        name = "fig_gaussian_noise_comparison.png" if noise_name == "Gaussian noise" else "fig_salt_pepper_comparison.png"
        files.append(utils.relative_to_output(utils.finish_figure(fig, out_dir / name), paths.output_dir))

    # Zoomed comparison to judge visual quality (fine detail, edges, blur)
    zoom = params["zoom_box"]
    zoom_images = [utils.crop_fraction(original, zoom)]
    zoom_titles = ["Original"]
    for noise_name in noisy:
        zoom_images.append(utils.crop_fraction(noisy[noise_name], zoom))
        zoom_titles.append(f"{noise_name}")
        for m in FILTERS:
            zoom_images.append(utils.crop_fraction(restored[(noise_name, m, k)], zoom))
            zoom_titles.append(f"{noise_name} + {m.lower()} {k}x{k}")
    # 7 images: original + 3 per noise type; lay out as 2 rows of 4 (one empty cell)
    ordered = [zoom_images[0], zoom_images[1], zoom_images[2], zoom_images[3],
               zoom_images[0], zoom_images[4], zoom_images[5], zoom_images[6]]
    ordered_titles = [zoom_titles[0], zoom_titles[1], zoom_titles[2], zoom_titles[3],
                      zoom_titles[0], zoom_titles[4], zoom_titles[5], zoom_titles[6]]
    fig = utils.image_grid_figure(
        ordered, ordered_titles, "Zoomed detail: visual quality of the restored images",
        caption=("Crop of the face region. Top row: Gaussian noise; bottom row: salt-and-pepper noise. "
                 "Metrics alone do not show where blur or leftover noise appears, so the same crop is compared visually."),
        cols=4,
    )
    files.append(utils.relative_to_output(utils.finish_figure(fig, out_dir / "fig_zoomed_face_comparison.png"), paths.output_dir))

    # Kernel-size trade-off: more smoothing removes more noise but also more fine detail
    sizes = [params["kernel_sizes"][0], params["kernel_sizes"][len(params["kernel_sizes"]) // 2], params["kernel_sizes"][-1]]
    trade_images, trade_titles = [], []
    for noise_name, method in (("Gaussian noise", "Gaussian filter"), ("Salt-and-pepper noise", "Median filter")):
        for size in sizes:
            trade_images.append(utils.crop_fraction(restored[(noise_name, method, size)], zoom))
            trade_titles.append(f"{noise_name} + {method.lower()} {size}x{size}\n"
                                f"{_fmt(metric(noise_name, method, size))}")
    fig = utils.image_grid_figure(
        trade_images, trade_titles, "Effect of kernel size on restored detail",
        caption=("Same face crop. Larger kernels remove more noise but also soften eyelashes, hair strands and skin "
                 "texture; compare the metrics in each title with what you see."),
        cols=len(sizes),
    )
    files.append(utils.relative_to_output(utils.finish_figure(fig, out_dir / "fig_kernel_size_tradeoff.png"), paths.output_dir))

    files.append(utils.relative_to_output(
        _plot_metric_vs_kernel(rows, params["kernel_sizes"], out_dir / "fig_psnr_ssim_vs_kernel.png"), paths.output_dir))

    csv_path = utils.save_csv(out_dir / "restoration_metrics.csv", rows)
    files.append(utils.relative_to_output(csv_path, paths.output_dir))

    # Best filter per noise type (by PSNR, SSIM reported alongside)
    best = {}
    for noise_name in noisy:
        candidates = [r for r in rows if r["noise"] == noise_name and r["kernel"] is not None]
        best_psnr = max(candidates, key=lambda r: r["psnr_db"])
        best_ssim = max(candidates, key=lambda r: r["ssim"])
        best[noise_name] = {"by_psnr": best_psnr, "by_ssim": best_ssim}
        LOG.info("Best for %s: %s %dx%d (PSNR) / %s %dx%d (SSIM)", noise_name, best_psnr["method"],
                 best_psnr["kernel"], best_psnr["kernel"], best_ssim["method"], best_ssim["kernel"], best_ssim["kernel"])

    results = {
        "task": "Unit 3A - Image restoration",
        "status": "completed",
        "input": {"file": photo_path.name, "width": w, "height": h, "channels": original.shape[2]},
        "parameters": {**params, "random_seed": config.RANDOM_SEED},
        "metrics": rows,
        "best": best,
        "default_kernel": k,
        "files": files,
    }
    utils.save_json(out_dir / "restoration_results.json", results)
    return results
