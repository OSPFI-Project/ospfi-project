#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run manipulation inference for one task using the project directory layout."""

import argparse
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "inference" / "source"
RESULT_ROOT = PROJECT_ROOT / "inference" / "result"
TEMPLATE_ROOT = PROJECT_ROOT / "template"
DEFAULT_CALIBRATION_PATH = (
    PROJECT_ROOT / "camera" / "camera_parameters.json"
)

RGB_FILENAME = "scene_manip_rgb.png"
DEPTH_FILENAME = "scene_manip_depth.png"
TEMPLATE_FILENAME = "manipulation_template.npz"


def resolve_task_paths(task_name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task_name):
        raise ValueError(
            "Task name may contain only letters, numbers, underscores, and hyphens."
        )

    source_dir = SOURCE_ROOT / task_name
    result_dir = RESULT_ROOT / task_name
    template_dir = TEMPLATE_ROOT / task_name
    paths = {
        "rgb": source_dir / RGB_FILENAME,
        "depth": source_dir / DEPTH_FILENAME,
        "template": template_dir / TEMPLATE_FILENAME,
        "result_dir": result_dir,
    }

    missing = [
        f"{name}: {path}"
        for name, path in paths.items()
        if name != "result_dir" and not path.is_file()
    ]
    if missing:
        available_tasks = sorted(
            path.name for path in SOURCE_ROOT.iterdir() if path.is_dir()
        ) if SOURCE_ROOT.is_dir() else []
        details = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Required input files are missing for task '{task_name}':\n"
            f"  {details}\n"
            f"Available task directories: {available_tasks}"
        )

    result_dir.mkdir(parents=True, exist_ok=True)
    return paths


def run_task(
    task_name,
    calibration_path=DEFAULT_CALIBRATION_PATH,
    depth_scale=1000.0,
    remove_object_prompt=None,
    remove_mode=1,
    num_fps_points=18,
):
    paths = resolve_task_paths(task_name)
    from manipulation_inference import ManipulationInference

    print(f"[Task] {task_name}")
    print(f"[Input] RGB: {paths['rgb']}")
    print(f"[Input] Depth: {paths['depth']}")
    print(f"[Input] Template: {paths['template']}")
    print(f"[Output] Directory: {paths['result_dir']}")

    inference = ManipulationInference(calibration_path=calibration_path)
    result = inference.run(
        rgb_path=paths["rgb"],
        depth_path=paths["depth"],
        template_path=paths["template"],
        output_dir=paths["result_dir"],
        depth_scale=depth_scale,
        remove_object_prompt=remove_object_prompt,
        remove_mode=remove_mode,
        num_fps_points=num_fps_points,
    )
    print(f"[Success] Result: {result['json_path']}")
    return result


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run local manipulation inference for a named task."
    )
    parser.add_argument(
        "task_name",
        help="Task directory name, for example: hanging, pouring, or sweeping.",
    )
    parser.add_argument(
        "--calibration-path",
        default=str(DEFAULT_CALIBRATION_PATH),
    )
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--remove-object-prompt", default=None)
    parser.add_argument("--remove-mode", type=int, choices=(1, 2), default=1)
    parser.add_argument("--num-fps-points", type=int, default=18)
    return parser


def main():
    args = build_argument_parser().parse_args()
    run_task(
        task_name=args.task_name,
        calibration_path=args.calibration_path,
        depth_scale=args.depth_scale,
        remove_object_prompt=args.remove_object_prompt,
        remove_mode=args.remove_mode,
        num_fps_points=args.num_fps_points,
    )


if __name__ == "__main__":
    main()
