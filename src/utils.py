"""Shared helpers used by every unit.

- finding and validating input files
- reading/writing images (works with any Windows path, including spaces)
- saving figures, JSON and CSV results
- a small timer and a consistent chart style
"""

from __future__ import annotations

import csv
import json
import logging
import math
import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # save figures to files; no windows pop up unless --show is used
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402

LOG = logging.getLogger("cv_assignment")


# ------------------------------------------------------------------ errors
class InputMissingError(FileNotFoundError):
    """A required input file (photo or video) is not in the inputs folder."""


class ModelUnavailableError(RuntimeError):
    """A pretrained model could not be loaded or downloaded."""


# ------------------------------------------------------------------ paths
@dataclass
class ProjectPaths:
    """Where inputs are read from and where outputs and models are written."""

    input_dir: Path = config.DEFAULT_INPUT_DIR
    output_dir: Path = config.DEFAULT_OUTPUT_DIR
    models_dir: Path = config.DEFAULT_MODELS_DIR

    def unit_dir(self, unit: int, *parts: str) -> Path:
        """Return (and create) a folder such as outputs/unit3/restoration."""
        return ensure_dir(Path(self.output_dir, f"unit{unit}", *parts))


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_input(input_dir: Path, key: str) -> Path | None:
    """Return the path of an input such as 'original_photo', or None if absent."""
    stem, extensions = config.INPUT_FILES[key]
    for ext in extensions:
        candidate = Path(input_dir) / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def require_input(input_dir: Path, key: str) -> Path:
    """Like find_input, but raises a clear InputMissingError when the file is absent."""
    path = find_input(input_dir, key)
    if path is None:
        stem, extensions = config.INPUT_FILES[key]
        names = " or ".join(f"{stem}{ext}" for ext in extensions)
        raise InputMissingError(
            f"Missing input '{key}': copy your file into {Path(input_dir)} and name it {names}."
        )
    return path


# ------------------------------------------------------------------ image I/O
def read_image(path: Path | str, grayscale: bool = False) -> np.ndarray:
    """Read an image as BGR (OpenCV order) or grayscale uint8.

    np.fromfile + cv2.imdecode is used instead of cv2.imread because imread
    cannot open some Windows paths (for example ones with non-English letters).
    """
    path = Path(path)
    if not path.is_file():
        raise InputMissingError(f"Image not found: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(data, flag)
    if image is None:
        raise ValueError(f"Could not decode image (unsupported or corrupt file): {path}")
    return image


def write_image(path: Path | str, image: np.ndarray, jpeg_quality: int = 92) -> Path:
    """Save an image. PNG is lossless; JPG is smaller (used for large previews)."""
    path = Path(path)
    ensure_dir(path.parent)
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if ext in (".jpg", ".jpeg") else []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise OSError(f"OpenCV could not encode image for {path}")
    buffer.tofile(str(path))
    return path


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale; grayscale input is returned unchanged."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert grayscale to 3-channel BGR so coloured drawings can be added."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def resize_to_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Shrink an image so its longer side is at most max_side. Returns (image, scale)."""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image, 1.0
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def crop_fraction(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Crop using fractions of width/height, e.g. (0.25, 0.25, 0.75, 0.75)."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    return image[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]


# ------------------------------------------------------------------ results
def _jsonable(value):
    """Convert numpy numbers, paths and NaN/inf into plain JSON values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def save_json(path: Path | str, data) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(_jsonable(data), indent=2), encoding="utf-8")
    return path


def load_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_csv(path: Path | str, rows: list[dict]) -> Path:
    """Write a list of dictionaries as a CSV table (columns = keys of the first row)."""
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(v) for k, v in row.items()})
    return path


def relative_to_output(path: Path, output_dir: Path) -> str:
    """Path used inside JSON results, relative to the outputs folder."""
    try:
        return Path(path).resolve().relative_to(Path(output_dir).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


@contextmanager
def timer():
    """Measure elapsed time: `with timer() as t: ...` then read t["ms"]."""
    result = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - start) * 1000.0


# ------------------------------------------------------------------ figures
# Colours: fixed order for data series, quiet greys for text and axes.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "font.family": "DejaVu Sans",
        "text.color": INK,
    }
)

_SHOW_FIGURES = False


def configure_display(show: bool) -> None:
    """Call with True to also open each figure in a window (python main.py --show)."""
    global _SHOW_FIGURES
    _SHOW_FIGURES = show
    if show:
        plt.switch_backend("TkAgg")


def finish_figure(fig, path: Path | str, dpi: int = 130) -> Path:
    """Save a matplotlib figure, optionally show it, then free its memory."""
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if _SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    return path


def image_grid_figure(
    images: list[np.ndarray],
    titles: list[str],
    suptitle: str,
    caption: str | None = None,
    cols: int | None = None,
    panel_width: float = 3.6,
    cmaps: list[str | None] | None = None,
):
    """Show several images side by side with titles and an optional caption.

    BGR colour images are converted to RGB for matplotlib. uint8 grayscale
    images are shown on a fixed 0-255 scale so masks and edges look correct.
    """
    count = len(images)
    cols = cols or count
    rows = math.ceil(count / cols)
    aspect = max(img.shape[0] / img.shape[1] for img in images)  # tallest panel decides the row height
    caption_lines = textwrap.wrap(caption, width=int(cols * panel_width * 15)) if caption else []
    title_lines = max(len(t.splitlines()) for t in titles) if titles else 1
    fig_height = rows * (panel_width * aspect + 0.22 * title_lines + 0.15) + 0.6 + 0.18 * len(caption_lines)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * panel_width, fig_height), squeeze=False)
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index >= count:
            continue
        image = images[index]
        cmap = cmaps[index] if cmaps else None
        if image.ndim == 3:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif image.dtype == np.uint8 and cmap is None:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(image, cmap=cmap or "gray")
        ax.set_title(titles[index], fontsize=9, color=INK)
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", color=INK)
    bottom = 0.0
    if caption_lines:
        bottom = min(0.25, (0.18 * len(caption_lines) + 0.15) / fig_height)
        fig.text(0.5, 0.01, "\n".join(caption_lines), ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, bottom, 1, 0.96))
    return fig


def label_line_ends(ax, x, items: list[tuple[float, str]], min_gap_px: float = 11.0) -> None:
    """Write small labels at the right end of several lines without letting them overlap.

    items = [(y value at x, label), ...]. Labels closer than `min_gap_px`
    on screen are pushed apart vertically.
    """
    ax.figure.canvas.draw()  # make the data -> pixel transform final
    to_px = ax.transData.transform
    placed = sorted(((to_px((x, y))[1], y, text) for y, text in items), key=lambda item: item[0])
    offsets = []
    last = None
    for py, y, text in placed:
        target = py if last is None else max(py, last + min_gap_px)
        offsets.append((y, text, target - py))
        last = target
    for y, text, shift in offsets:
        ax.annotate(text, (x, y), xytext=(6, shift * 72 / ax.figure.dpi), textcoords="offset points",
                    va="center", fontsize=8, color=INK_SECONDARY)


def draw_label(image: np.ndarray, text: str, x: int, y: int, color=(0, 200, 0), scale: float | None = None) -> None:
    """Draw a filled label box with text (used for detection boxes)."""
    h = image.shape[0]
    scale = scale or max(0.5, h / 800)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y_top = max(0, y - th - baseline - 4)
    cv2.rectangle(image, (x, y_top), (x + tw + 6, y_top + th + baseline + 4), color, -1)
    cv2.putText(image, text, (x + 3, y_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )
