#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize manipulation alignment results for one task."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "inference" / "result"
TEMPLATE_ROOT = PROJECT_ROOT / "template"
RESULT_FILENAME = "scene_manip_rgb_manipulation_result.json"
TEMPLATE_FILENAME = "manipulation_template.npz"     


def resolve_paths(task_name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task_name):
        raise ValueError(
            "Task name may contain only letters, numbers, underscores, and hyphens."
        )

    result_dir = RESULT_ROOT / task_name
    result_path = result_dir / RESULT_FILENAME
    if not result_path.is_file() and result_dir.is_dir():
        candidates = sorted(result_dir.glob("*_manipulation_result.json"))
        if len(candidates) == 1:
            result_path = candidates[0]
        elif len(candidates) > 1:
            raise RuntimeError(
                f"Multiple result files found in {result_dir}: "
                f"{[path.name for path in candidates]}"
            )

    template_path = TEMPLATE_ROOT / task_name / TEMPLATE_FILENAME
    missing = [
        str(path)
        for path in (result_path, template_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Required files are missing:\n  " + "\n  ".join(missing))
    return result_path, template_path


def load_json_points(result, key):
    if key not in result:
        raise KeyError(f"Result field is missing: {key}")
    rows = result[key]
    if not isinstance(rows, list):
        raise TypeError(f"{key} must be a list, got {type(rows).__name__}.")

    points = np.full((len(rows), 3), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        if row is None or not isinstance(row, (list, tuple)) or len(row) != 3:
            continue
        if any(value is None for value in row):
            continue
        candidate = np.asarray(row, dtype=np.float64)
        if np.isfinite(candidate).all():
            points[index] = candidate
    return points


def load_visualization_data(task_name):
    result_path, template_path = resolve_paths(task_name)
    with result_path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    target_points = load_json_points(result, "object_part_matched_points")
    ideal_tool_points = load_json_points(result, "ideal_tool_keypoints")
    actual_tool_points = load_json_points(result, "current_tool_matched_points")
    if ideal_tool_points.shape != actual_tool_points.shape:
        raise ValueError(
            "Ideal and actual tool correspondence arrays must have the same shape: "
            f"ideal={ideal_tool_points.shape}, actual={actual_tool_points.shape}."
        )

    rotation = np.asarray(result["R0"], dtype=np.float64)
    translation = np.asarray(result["T0"], dtype=np.float64).reshape(-1)
    if rotation.shape != (3, 3):
        raise ValueError(f"R0 must be 3x3, got {rotation.shape}.")
    if translation.shape != (3,):
        raise ValueError(f"T0 must have three elements, got {translation.shape}.")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("R0 or T0 contains non-finite values.")

    with np.load(template_path, allow_pickle=True) as template:
        if "tool_key_interaction_point_index" not in template:
            raise KeyError(
                "Template field is missing: tool_key_interaction_point_index"
            )
        interaction_index = int(
            np.asarray(template["tool_key_interaction_point_index"]).item()
        )
    if not 0 <= interaction_index < len(ideal_tool_points):
        raise IndexError(
            f"Interaction index {interaction_index} is outside "
            f"the tool-point range [0, {len(ideal_tool_points) - 1}]."
        )

    aligned_tool_points = np.full_like(actual_tool_points, np.nan)
    actual_valid = np.all(np.isfinite(actual_tool_points), axis=1)
    aligned_tool_points[actual_valid] = (
        actual_tool_points[actual_valid] @ rotation.T + translation
    )

    return {
        "result_path": result_path,
        "template_path": template_path,
        "result": result,
        "target_points": target_points,
        "ideal_tool_points": ideal_tool_points,
        "actual_tool_points": actual_tool_points,
        "aligned_tool_points": aligned_tool_points,
        "interaction_index": interaction_index,
    }


def finite_rows(points):
    return np.all(np.isfinite(points), axis=1)


def set_equal_axes(ax, point_groups):
    finite_groups = [
        points[finite_rows(points)]
        for points in point_groups
        if len(points) > 0 and np.any(finite_rows(points))
    ]
    if not finite_groups:
        return
    all_points = np.vstack(finite_groups)
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 0.01)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def configure_axis(ax):
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid(True, alpha=0.35)


def scatter_tool_correspondences(
    ax,
    ideal_points,
    comparison_points,
    colors,
    ideal_label,
    comparison_label,
):
    ideal_valid = finite_rows(ideal_points)
    comparison_valid = finite_rows(comparison_points)
    for index in range(len(ideal_points)):
        if ideal_valid[index]:
            ax.scatter(
                *ideal_points[index],
                color=colors[index],
                marker="o",
                s=55,
                edgecolor="black",
                linewidth=0.6,
                label=ideal_label if index == np.flatnonzero(ideal_valid)[0] else None,
            )
            ax.text(*ideal_points[index], f" I{index}", fontsize=8)
        if comparison_valid[index]:
            ax.scatter(
                *comparison_points[index],
                color=colors[index],
                marker="^",
                s=60,
                edgecolor="black",
                linewidth=0.6,
                label=(
                    comparison_label
                    if index == np.flatnonzero(comparison_valid)[0]
                    else None
                ),
            )
            ax.text(*comparison_points[index], f" A{index}", fontsize=8)


def mark_interaction_points(
    ax,
    ideal_points,
    comparison_points,
    interaction_index,
    comparison_name,
):
    ideal_point = ideal_points[interaction_index]
    comparison_point = comparison_points[interaction_index]
    if np.isfinite(ideal_point).all():
        ax.scatter(
            *ideal_point,
            color="gold",
            marker="*",
            s=260,
            edgecolor="black",
            linewidth=1.2,
            label="Ideal interaction keypoint",
            zorder=10,
        )
    if np.isfinite(comparison_point).all():
        ax.scatter(
            *comparison_point,
            color="magenta",
            marker="X",
            s=180,
            edgecolor="black",
            linewidth=1.0,
            label=f"{comparison_name} interaction keypoint",
            zorder=10,
        )
    else:
        print(
            f"[Warning] Tool interaction keypoint {interaction_index} is an outlier; "
            f"the {comparison_name.lower()} interaction point cannot be displayed."
        )


def visualize(task_name):
    data = load_visualization_data(task_name)
    target_points = data["target_points"]
    ideal_tool_points = data["ideal_tool_points"]
    actual_tool_points = data["actual_tool_points"]
    aligned_tool_points = data["aligned_tool_points"]
    interaction_index = data["interaction_index"]
    result = data["result"]

    colors = plt.get_cmap("tab10")(
        np.arange(len(ideal_tool_points)) % 10
    )

    figure_before = plt.figure(figsize=(10, 8))
    try:
        figure_before.canvas.manager.set_window_title(
            f"{task_name}: Current Scene Before Tool Alignment"
        )
    except AttributeError:
        pass
    axis_before = figure_before.add_subplot(111, projection="3d")
    target_valid = finite_rows(target_points)
    if np.any(target_valid):
        axis_before.scatter(
            *target_points[target_valid].T,
            color="dimgray",
            marker=".",
            s=45,
            label="Current target points",
        )
    scatter_tool_correspondences(
        axis_before,
        ideal_tool_points,
        actual_tool_points,
        colors,
        "Ideal tool keypoints",
        "Actual tool keypoints",
    )
    mark_interaction_points(
        axis_before,
        ideal_tool_points,
        actual_tool_points,
        interaction_index,
        "Actual",
    )
    axis_before.set_title("Current Target, Ideal Tool, and Actual Tool")
    configure_axis(axis_before)
    set_equal_axes(
        axis_before,
        [target_points, ideal_tool_points, actual_tool_points],
    )
    axis_before.legend(loc="best")
    figure_before.tight_layout()

    figure_after = plt.figure(figsize=(10, 8))
    try:
        figure_after.canvas.manager.set_window_title(
            f"{task_name}: Tool Alignment Result"
        )
    except AttributeError:
        pass
    axis_after = figure_after.add_subplot(111, projection="3d")
    if np.any(target_valid):
        axis_after.scatter(
            *target_points[target_valid].T,
            color="dimgray",
            marker=".",
            s=35,
            alpha=0.55,
            label="Current target points",
        )
    scatter_tool_correspondences(
        axis_after,
        ideal_tool_points,
        aligned_tool_points,
        colors,
        "Ideal tool keypoints",
        "Aligned tool keypoints",
    )

    paired = finite_rows(ideal_tool_points) & finite_rows(aligned_tool_points)
    for index in np.flatnonzero(paired):
        segment = np.vstack((ideal_tool_points[index], aligned_tool_points[index]))
        axis_after.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=colors[index],
            linewidth=1.0,
            alpha=0.65,
        )
    mark_interaction_points(
        axis_after,
        ideal_tool_points,
        aligned_tool_points,
        interaction_index,
        "Aligned",
    )

    if np.any(paired):
        errors = np.linalg.norm(
            aligned_tool_points[paired] - ideal_tool_points[paired], axis=1
        )
        error_text = f"mean={errors.mean():.5f}m, max={errors.max():.5f}m"
    else:
        error_text = "no valid correspondence pairs"
    alignment_name = (
        "refined alignment"
        if result.get("refinement_applied", False)
        else "coarse alignment fallback"
    )
    axis_after.set_title(
        f"Tool Keypoints After {alignment_name}\n{error_text}"
    )
    configure_axis(axis_after)
    set_equal_axes(
        axis_after,
        [target_points, ideal_tool_points, aligned_tool_points],
    )
    axis_after.legend(loc="best")
    figure_after.tight_layout()

    print(f"[Loaded] Result: {data['result_path']}")
    print(f"[Loaded] Template: {data['template_path']}")
    print(f"[Info] Interaction keypoint index: {interaction_index}")
    print(f"[Info] Alignment: {alignment_name}")
    plt.show()


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Visualize saved manipulation alignment results."
    )
    parser.add_argument(
        "task_name",
        help="Task directory name, for example: hanging, pouring, or sweeping.",
    )
    return parser


def main():
    args = build_argument_parser().parse_args()
    visualize(args.task_name)


if __name__ == "__main__":
    main()
