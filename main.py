"""CCA 3 computer vision assignment - single entry point.

Examples (run from the project folder with the virtual environment active):
    python main.py --check-inputs     # which input files are present?
    python main.py --unit 3           # image restoration + segmentation
    python main.py --unit 4           # HOG + SIFT matching
    python main.py --unit 5           # Haar/YOLO detection + video motion
    python main.py --all              # everything
    python main.py --unit 5 --skip-yolo

A missing photo or video never crashes the whole run: that task is reported
as PENDING and the other tasks continue.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

from src import config, utils

TASKS = {
    3: [("3A", "Image restoration", "src.image_restoration"),
        ("3B", "Morphology and segmentation", "src.segmentation")],
    4: [("4", "HOG and SIFT feature matching", "src.feature_extraction")],
    5: [("5A", "Haar cascade and YOLO detection", "src.object_detection"),
        ("5B", "Video motion estimation", "src.motion_estimation")],
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CCA 3 computer vision assignment (Units 3, 4 and 5)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--unit", type=int, choices=sorted(TASKS), action="append",
                       help="unit to run (can be repeated, e.g. --unit 3 --unit 4)")
    group.add_argument("--all", action="store_true", help="run every unit")
    group.add_argument("--check-inputs", action="store_true", help="only list which input files are present")
    parser.add_argument("--inputs", type=Path, default=config.DEFAULT_INPUT_DIR, help="input folder")
    parser.add_argument("--outputs", type=Path, default=config.DEFAULT_OUTPUT_DIR, help="output folder")
    parser.add_argument("--models", type=Path, default=config.DEFAULT_MODELS_DIR, help="folder for YOLO weights")
    parser.add_argument("--skip-yolo", action="store_true", help="do not load or run YOLO")
    parser.add_argument("--yolo-model", default=None, help=f"YOLO weights name (default {config.DETECTION['yolo_model']})")
    parser.add_argument("--show", action="store_true", help="also open each figure in a window")
    parser.add_argument("--screenshots", action="store_true",
                        help="Windows: show the main figures on screen and save a screenshot of each window")
    parser.add_argument("--verbose", action="store_true", help="more detailed log messages")
    return parser.parse_args(argv)


def input_status(input_dir: Path) -> dict:
    status = {}
    for key in config.INPUT_FILES:
        path = utils.find_input(input_dir, key)
        status[key] = path.name if path else None
    return status


def print_input_status(status: dict) -> None:
    print("\nInput files:")
    notes = {"watershed_photo": "(optional - a scikit-image sample is used if absent)",
             "front_camera_video": "(optional - extra face-detection frames for Unit 5A)",
             "second_photo_b": "(optional - extra same-scene pair for Unit 4)",
             "second_image_video_frame":"(optional - video frame used for Unit 4 only if second_photo is missing)"}
    for key, name in status.items():
        mark = f"found: {name}" if name else "MISSING " + notes.get(key, "")
        print(f"  {key:<16} {mark}")


def run_task(code: str, name: str, module_name: str, paths: utils.ProjectPaths, options: dict) -> dict:
    log = logging.getLogger("cv_assignment")
    log.info("=== Task %s: %s ===", code, name)
    try:
        module = importlib.import_module(module_name)
        result = module.run(paths, options)
        status = result.get("status", "completed")
        return {"task": code, "name": name, "status": status, "pending": result.get("pending", [])}
    except utils.InputMissingError as exc:
        log.warning("Task %s is PENDING: %s", code, exc)
        return {"task": code, "name": name, "status": "pending", "pending": [str(exc)]}
    except utils.ModelUnavailableError as exc:
        log.error("Task %s could not load a model: %s", code, exc)
        return {"task": code, "name": name, "status": "blocked", "pending": [str(exc)]}
    except Exception as exc:  # keep going with the other tasks, but report the failure clearly
        log.error("Task %s FAILED: %s\n%s", code, exc, traceback.format_exc())
        return {"task": code, "name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def main(argv=None) -> int:
    args = parse_args(argv)
    utils.setup_logging(args.verbose)
    utils.configure_display(args.show, args.screenshots)
    paths = utils.ProjectPaths(args.inputs, args.outputs, args.models)
    status = input_status(paths.input_dir)
    print_input_status(status)
    if args.check_inputs:
        return 0

    units = sorted(TASKS) if args.all else sorted(set(args.unit))
    options = {"skip_yolo": args.skip_yolo, "yolo_model": args.yolo_model}
    summary = []
    for unit in units:
        for code, name, module_name in TASKS[unit]:
            summary.append(run_task(code, name, module_name, paths, options))

    print("\nSummary:")
    for item in summary:
        print(f"  Task {item['task']:<3} {item['name']:<35} {item['status'].upper()}")
        for note in item.get("pending", []):
            print(f"        pending: {note}")
        if "error" in item:
            print(f"        error:   {item['error']}")

    summary_path = Path(paths.output_dir) / "run_summary.json"
    previous = utils.load_json(summary_path) if summary_path.is_file() else {}
    tasks = previous.get("tasks", {})
    for item in summary:
        tasks[item["task"]] = {**item, "run_at": datetime.now().isoformat(timespec="seconds")}
    utils.save_json(summary_path, {"inputs": status, "tasks": dict(sorted(tasks.items()))})
    print(f"\nOutputs are in {Path(paths.output_dir).resolve()}")
    return 1 if any(item["status"] == "failed" for item in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
