"""Capture genuine screenshots of the project running (Windows only).

    python tools/capture_screenshots.py              # every step
    python tools/capture_screenshots.py unit3 tests  # only some steps

For each step a real console window (conhost + Windows PowerShell) is opened, the
project command runs in it, and when the command has finished exactly that window
is captured with the Win32 PrintWindow API (nothing else on the desktop).
The unit runs use `main.py --screenshots`, so the figure windows that main.py
shows while it runs are captured as well (screenshot_<figure>.jpg next to each
figure). Every screenshot is listed in outputs/screenshots_index.json, which
tools/build_report.py uses for the "Execution screenshots" section.

Windows pop up on screen for a few minutes; do not close them by hand.
"""

from __future__ import annotations

import argparse
import base64
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import screen_capture, utils  # noqa: E402

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
INDEX = ROOT / "outputs" / "screenshots_index.json"

# name: (window title, python arguments, terminal screenshot path, console columns, console lines)
STEPS = {
    "check_inputs": ("Input check", ["main.py", "--check-inputs"], "outputs/screenshot_terminal_check_inputs.png", 120, 16),
    "unit3": ("Unit 3 run", ["main.py", "--unit", "3", "--screenshots"], "outputs/unit3/screenshot_terminal_unit3.png", 132, 46),
    "unit4": ("Unit 4 run", ["main.py", "--unit", "4", "--screenshots"], "outputs/unit4/screenshot_terminal_unit4.png", 132, 28),
    "unit5": ("Unit 5 run", ["main.py", "--unit", "5", "--screenshots"], "outputs/unit5/screenshot_terminal_unit5.png", 150, 38),
    "tests": ("Automated tests", ["-m", "pytest", "-v", "-p", "no:cacheprovider"], "outputs/screenshot_terminal_pytest.png", 132, 52),
}

# Sets a smaller console font so a whole run fits on a 1080p screen.
FONT_HELPER = r"""
using System;
using System.Runtime.InteropServices;
public static class CcaConsoleFont {
    [StructLayout(LayoutKind.Sequential)] public struct COORD { public short X; public short Y; }
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct FONTINFO {
        public uint cbSize; public uint nFont; public COORD dwFontSize; public int FontFamily; public int FontWeight;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string FaceName;
    }
    [DllImport("kernel32.dll")] public static extern IntPtr GetStdHandle(int handle);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    public static extern bool SetCurrentConsoleFontEx(IntPtr handle, bool maximumWindow, ref FONTINFO font);
    public static bool Set(string face, short height) {
        FONTINFO font = new FONTINFO();
        font.cbSize = (uint)Marshal.SizeOf(typeof(FONTINFO));
        font.FaceName = face; font.dwFontSize.X = 0; font.dwFontSize.Y = height;
        font.FontFamily = 54; font.FontWeight = 400;
        return SetCurrentConsoleFontEx(GetStdHandle(-11), false, ref font);
    }
}
"""


def font_height(lines: int) -> int:
    """Largest console font (logical pixels) that fits `lines` rows on this screen."""
    _, top, _, bottom = screen_capture.work_area()
    usable = (bottom - top) / screen_capture.scale_factor() - 60  # minus title bar and margin
    return int(max(10, min(16, usable // lines)))


def build_script(title: str, args: list[str], cols: int, lines: int, status_file: Path) -> str:
    display = " ".join(args)
    quoted = " ".join(f"'{a}'" for a in args)
    return f"""
$Host.UI.RawUI.WindowTitle = 'CCA3 screenshot - {title} - running'
Add-Type -TypeDefinition @'
{FONT_HELPER}
'@
[void][CcaConsoleFont]::Set('Consolas', {font_height(lines)})
mode con: cols={cols} lines={lines}
Set-Location -LiteralPath '{ROOT}'
Write-Host ('CCA 3 Computer Vision Assignment  |  ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '  |  ' + (Get-Location).Path) -ForegroundColor Cyan
Write-Host '> .venv\\Scripts\\python.exe {display}' -ForegroundColor Yellow
& '{PYTHON}' {quoted}
$code = $LASTEXITCODE
Write-Host ('Exit code: ' + $code) -ForegroundColor Cyan
$Host.UI.RawUI.WindowTitle = 'CCA3 screenshot - {title} - finished'
Set-Content -LiteralPath '{status_file}' -Value $code
Start-Sleep -Seconds 3600
"""


def run_step(name: str, timeout_s: int = 1200) -> dict:
    import win32con
    import win32gui

    title, args, out_rel, cols, lines = STEPS[name]
    status_file = Path(tempfile.gettempdir()) / f"cca3_screenshot_{name}_{int(time.time())}.txt"
    script = build_script(title, args, cols, lines, status_file)
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    started = datetime.now()
    start_ts = time.time()
    print(f"[{name}] running: python {' '.join(args)}")
    process = subprocess.Popen(
        ["conhost.exe", "powershell.exe", "-NoLogo", "-NoProfile", "-EncodedCommand", encoded],
        creationflags=subprocess.CREATE_NEW_CONSOLE)
    try:
        while not status_file.is_file():
            if time.time() - start_ts > timeout_s:
                raise TimeoutError(f"Step {name} did not finish within {timeout_s} s")
            if process.poll() is not None:
                raise RuntimeError(f"Console for step {name} closed unexpectedly")
            time.sleep(1.0)
        exit_code = int(status_file.read_text(encoding="utf-8-sig").strip() or -1)
        time.sleep(1.5)  # let the console paint the last lines
        hwnd = win32gui.FindWindow(None, f"CCA3 screenshot - {title} - finished")
        if not hwnd:
            raise RuntimeError(f"Could not find the console window for step {name}")
        out_path = screen_capture.save_image(screen_capture.grab_window(hwnd), ROOT / out_rel)
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
    finally:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        status_file.unlink(missing_ok=True)
    windows = sorted(p for p in (ROOT / "outputs").rglob("screenshot_*.jpg") if p.stat().st_mtime >= start_ts)
    print(f"[{name}] exit code {exit_code}; terminal screenshot {out_rel}; {len(windows)} window screenshot(s)")
    return {
        "step": name, "title": title, "command": "python " + " ".join(args), "exit_code": exit_code,
        "started": started.isoformat(timespec="seconds"), "finished": datetime.now().isoformat(timespec="seconds"),
        "terminal_screenshot": out_path.relative_to(ROOT).as_posix(),
        "window_screenshots": [p.relative_to(ROOT).as_posix() for p in windows],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("steps", nargs="*", choices=list(STEPS) + [[]], help="steps to run (default: all)")
    args = parser.parse_args(argv)
    screen_capture.require_windows()
    screen_capture.enable_dpi_awareness()
    if not PYTHON.is_file():
        sys.exit(f"Virtual environment not found: {PYTHON}")
    index = utils.load_json(INDEX) if INDEX.is_file() else {"steps": {}}
    left, top, right, bottom = screen_capture.work_area()
    index["method"] = ("Real console windows (conhost + Windows PowerShell) and live matplotlib windows, each "
                       "captured with the Win32 PrintWindow API by tools/capture_screenshots.py")
    index["screen"] = f"{right - left}x{bottom - top} px work area, {screen_capture.scale_factor():.0%} scaling"
    for name in args.steps or list(STEPS):
        index["steps"][name] = run_step(name)
        utils.save_json(INDEX, index)
    print(f"Index written to {INDEX.relative_to(ROOT)}")
    return 0 if all(s["exit_code"] == 0 for s in index["steps"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
