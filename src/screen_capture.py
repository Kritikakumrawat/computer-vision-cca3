"""Windows screen-capture helpers for the execution screenshots.

`grab_window(hwnd)` copies the pixels of ONE window with the Win32 PrintWindow
API, so nothing else on the desktop is captured. Used by:
- utils.finish_figure() when main.py runs with --screenshots (live figure windows)
- tools/capture_screenshots.py (console windows running the project)
Windows only; needs pywin32 (listed in requirements.txt).
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

from PIL import Image, ImageGrab

PW_RENDERFULLCONTENT = 2  # also captures DirectX/modern window content
DWMWA_EXTENDED_FRAME_BOUNDS = 9


def require_windows() -> None:
    if sys.platform != "win32":
        raise RuntimeError("Screenshot capture is implemented for Windows only.")


def enable_dpi_awareness() -> None:
    """Work in real (physical) pixels on scaled displays such as 150 %."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def scale_factor() -> float:
    """Display scaling, e.g. 1.5 for 150 %."""
    try:
        return ctypes.windll.user32.GetDpiForSystem() / 96.0
    except Exception:
        return 1.0


def work_area() -> tuple[int, int, int, int]:
    """Desktop area without the taskbar: (left, top, right, bottom) in pixels."""
    rect = wintypes.RECT()
    ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0)  # SPI_GETWORKAREA
    return rect.left, rect.top, rect.right, rect.bottom


def _visible_bounds(hwnd: int) -> tuple[int, int, int, int]:
    """Window rectangle without the invisible resize border of Windows 10/11."""
    rect = wintypes.RECT()
    result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect))
    if result != 0:
        ctypes.windll.user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect))
    return rect.left, rect.top, rect.right, rect.bottom


def grab_window(hwnd: int) -> Image.Image:
    """Return a picture of one window (title bar included)."""
    require_windows()
    import win32gui
    import win32ui

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top
    window_dc = win32gui.GetWindowDC(hwnd)
    source_dc = win32ui.CreateDCFromHandle(window_dc)
    memory_dc = source_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(source_dc, width, height)
    memory_dc.SelectObject(bitmap)
    try:
        ok = ctypes.windll.user32.PrintWindow(hwnd, memory_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
        image = Image.frombuffer("RGB", (width, height), bitmap.GetBitmapBits(True), "raw", "BGRX", 0, 1)
    finally:
        win32gui.DeleteObject(bitmap.GetHandle())
        memory_dc.DeleteDC()
        source_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, window_dc)
    v_left, v_top, v_right, v_bottom = _visible_bounds(hwnd)
    image = image.crop((v_left - left, v_top - top, width - (right - v_right), height - (bottom - v_bottom)))
    if not ok or image.convert("L").getextrema()[1] < 10:  # PrintWindow failed -> copy from the screen
        win32gui.SetForegroundWindow(hwnd)
        image = ImageGrab.grab(bbox=(v_left, v_top, v_right, v_bottom), all_screens=True)
    return image


def save_image(image: Image.Image, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        image.convert("RGB").save(path, quality=90, optimize=True)
    else:
        image.save(path, optimize=True)
    return path
