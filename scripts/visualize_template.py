#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize template keypoints and the tool-center execution trajectory."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK_NAME = "sweeping"


def validate_points(points, name):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}.")
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError(f"{name} must contain finite points.")
    return points


def compute_tool_center_trajectory(initial_tool_pose, relative_trajectory):
    initial_tool_pose = np.asarray(initial_tool_pose, dtype=np.float64)
    relative_trajectory = np.asarray(relative_trajectory, dtype=np.float64)
    if initial_tool_pose.shape != (4, 4):
        raise ValueError(
            "micro_trajectory_initial_pose must have shape (4, 4), got "
            f"{initial_tool_pose.shape}."
        )
    if relative_trajectory.ndim != 3 or relative_trajectory.shape[1:] != (4, 4):
        raise ValueError(
            "micro_trajectory must have shape (T-1, 4, 4), got "
            f"{relative_trajectory.shape}."
        )
    if not np.isfinite(relative_trajectory).all():
        raise ValueError("micro_trajectory contains NaN or Inf.")

    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(relative_trajectory[:, 3, :], expected_last_row, atol=1e-8):
        raise ValueError("micro_trajectory contains invalid SE(3) matrices.")

    tool_pose = initial_tool_pose.copy()
    center_trajectory = [tool_pose[:3, 3].copy()]

    for relative_transform in relative_trajectory:
        tool_pose = tool_pose @ relative_transform
        center_trajectory.append(tool_pose[:3, 3].copy())

    return np.asarray(center_trajectory, dtype=np.float64)


def set_axes_equal(ax, point_sets):
    points = np.concatenate(
        [np.asarray(points, dtype=np.float64).reshape(-1, 3) for points in point_sets],
        axis=0,
    )
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def load_template(task_name):
    template_path = (
        PROJECT_ROOT
        / "template"
        / task_name
        / "manipulation_template.npz"
    )
    if not template_path.is_file():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    with np.load(template_path, allow_pickle=True) as data:
        required_fields = {
            "tool_part_keypoints",
            "target_part_keypoints",
            "tool_key_interaction_point_index",
            "micro_trajectory",
            "micro_trajectory_initial_pose",
        }
        missing_fields = sorted(required_fields.difference(data.files))
        if missing_fields:
            raise KeyError(f"Template fields are missing: {missing_fields}")

        tool_points = validate_points(data["tool_part_keypoints"], "tool points")
        target_points = validate_points(
            data["target_part_keypoints"], "target points"
        )
        interaction_index = int(
            np.asarray(data["tool_key_interaction_point_index"]).item()
        )
        relative_trajectory = np.asarray(data["micro_trajectory"], dtype=np.float64)
        initial_tool_pose = np.asarray(
            data["micro_trajectory_initial_pose"], dtype=np.float64
        )

    if not 0 <= interaction_index < len(tool_points):
        raise IndexError(
            "tool_key_interaction_point_index is out of range: "
            f"{interaction_index} for {len(tool_points)} tool points."
        )
    return (
        template_path,
        tool_points,
        target_points,
        interaction_index,
        initial_tool_pose,
        relative_trajectory,
    )


def visualize(task_name):
    (
        template_path,
        tool_points,
        target_points,
        interaction_index,
        initial_tool_pose,
        relative_trajectory,
    ) = load_template(task_name)

    interaction_point = tool_points[interaction_index]
    regular_tool_points = np.delete(tool_points, interaction_index, axis=0)
    center_trajectory = compute_tool_center_trajectory(
        initial_tool_pose,
        relative_trajectory,
    )

    figure = plt.figure(figsize=(10, 8))
    ax = figure.add_subplot(111, projection="3d")
    ax.scatter(
        regular_tool_points[:, 0],
        regular_tool_points[:, 1],
        regular_tool_points[:, 2],
        s=42,
        color="tab:blue",
        label="Tool Part Keypoints",
        depthshade=False,
    )
    ax.scatter(
        target_points[:, 0],
        target_points[:, 1],
        target_points[:, 2],
        s=42,
        color="tab:red",
        label="Target Part Keypoints",
        depthshade=False,
    )
    ax.scatter(
        interaction_point[0],
        interaction_point[1],
        interaction_point[2],
        s=150,
        marker="*",
        color="magenta",
        edgecolors="black",
        linewidths=0.8,
        label="Functional Keypoint",
        depthshade=False,
        zorder=5,
    )
    ax.plot(
        center_trajectory[:, 0],
        center_trajectory[:, 1],
        center_trajectory[:, 2],
        color="black",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="Tool Center Trajectory",
    )
    ax.scatter(
        center_trajectory[0, 0],
        center_trajectory[0, 1],
        center_trajectory[0, 2],
        s=90,
        marker="o",
        color="limegreen",
        edgecolors="black",
        label="Trajectory Start",
        depthshade=False,
        zorder=5,
    )
    ax.scatter(
        center_trajectory[-1, 0],
        center_trajectory[-1, 1],
        center_trajectory[-1, 2],
        s=90,
        marker="s",
        color="gold",
        edgecolors="black",
        label="Trajectory End",
        depthshade=False,
        zorder=5,
    )

    ax.set_title(f"{task_name}: Template Tool-Center Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="best")
    ax.grid(True)
    set_axes_equal(ax, [tool_points, target_points, center_trajectory])
    figure.tight_layout()

    print(f"Template: {template_path}")
    print(f"Tool keypoints: {len(tool_points)}")
    print(f"Target keypoints: {len(target_points)}")
    print(f"Functional keypoint index: {interaction_index}")
    print(f"Tool-center trajectory points: {len(center_trajectory)}")
    plt.show()


def main():
    task_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK_NAME
    if len(sys.argv) > 2:
        raise SystemExit(
            "Usage: python scripts/visualize_template.py "
            "[task_name]"
        )
    visualize(task_name)


if __name__ == "__main__":
    main()
