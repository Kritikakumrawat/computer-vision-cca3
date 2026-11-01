"""Checks for the screenshot feature that do not open any window."""

import sys
from pathlib import Path

import pytest

import main
from src import config

SRC = Path(__file__).resolve().parents[1] / "src"


def test_screenshot_option_is_parsed():
    args = main.parse_args(["--unit", "3", "--screenshots"])
    assert args.screenshots is True and args.unit == [3]
    assert main.parse_args(["--all"]).screenshots is False


def test_every_screenshot_figure_is_produced_by_the_code():
    """A typo in config.SCREENSHOT_FIGURES would silently skip a screenshot."""
    code = "".join(p.read_text(encoding="utf-8") for p in SRC.glob("*.py") if p.name != "config.py")
    missing = [name for name in config.SCREENSHOT_FIGURES if f"{name}.png" not in code]
    assert missing == []


@pytest.mark.skipif(sys.platform != "win32", reason="screen capture is Windows only")
def test_screen_geometry_is_sane():
    from src import screen_capture

    left, top, right, bottom = screen_capture.work_area()
    assert right - left > 300 and bottom - top > 300
    assert 0.5 <= screen_capture.scale_factor() <= 4.0
