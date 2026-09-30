#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Rigid refinement for a tool interaction keypoint.

The refined point-cloud transform is:

    p_refined = R_refined @ p_curr + t_refined

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_sci


@dataclass
class RefinementResult:
    success: bool
    s: float
    R: np.ndarray
    t: np.ndarray
    normal_curr: np.ndarray
    normal_ideal: np.ndarray
    normal_sign: float
    keypoint_error: float
    normal_alignment_abs: float
    rotation_deviation_rad: float
    translation_deviation: float
    cost: float
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "s": self.s,
            "R": self.R,
            "t": self.t,
            "normal_curr": self.normal_curr,
            "normal_ideal": self.normal_ideal,
            "normal_sign": self.normal_sign,
            "keypoint_error": self.keypoint_error,
            "normal_alignment_abs": self.normal_alignment_abs,
            "rotation_deviation_rad": self.rotation_deviation_rad,
            "translation_deviation": self.translation_deviation,
            "cost": self.cost,
            "message": self.message,
        }


def _as_vector3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{name} 必须是 3 维向量，当前 shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含 NaN/Inf: {array}")
    return array


def _as_rotation_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f"{name} 必须是 3x3 旋转矩阵，当前 shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含 NaN/Inf")

    u, _, vt = np.linalg.svd(array)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _normalize(vector: np.ndarray, name: str, eps: float) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError(f"{name} 范数过小，无法归一化: norm={norm}")
    return vector / norm


def _rotation_from_vector_to_vector(
    source: np.ndarray,
    target: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Return a rotation that maps normalized source onto normalized target."""
    dot_value = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot_value > 1.0 - eps:
        return np.eye(3, dtype=np.float64)

    if dot_value < -1.0 + eps:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(source, basis)) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _normalize(np.cross(source, basis), "opposite_normal_axis", eps=eps)
        return R_sci.from_rotvec(np.pi * axis).as_matrix()

    axis = np.cross(source, target)
    angle = np.arctan2(np.linalg.norm(axis), dot_value)
    axis = _normalize(axis, "normal_alignment_axis", eps=eps)
    return R_sci.from_rotvec(angle * axis).as_matrix()


def compute_interaction_normal(
    C_tool_curr: Any,
    C_obj_curr: Any,
    p_key_curr: Any,
    eps: float = 1e-8,
) -> np.ndarray:
    """Construct the interaction normal from the three current scene points."""
    C_tool = _as_vector3(C_tool_curr, "C_tool_curr")
    C_obj = _as_vector3(C_obj_curr, "C_obj_curr")
    p_key = _as_vector3(p_key_curr, "p_key_curr")

    a_curr = C_tool - p_key
    b_curr = C_obj - p_key
    return _normalize(np.cross(a_curr, b_curr), "n_int_curr", eps=eps)


def compute_transformed_triangle_normal(
    R: Any,
    t: Any,
    C_tool_curr: Any,
    C_obj_curr: Any,
    p_key_ideal: Any,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute the transformed tool-target-keypoint triangle normal."""
    R = _as_rotation_matrix(R, "R")
    t = _as_vector3(t, "t")
    C_tool = _as_vector3(C_tool_curr, "C_tool_curr")
    C_obj = _as_vector3(C_obj_curr, "C_obj_curr")
    p_ideal = _as_vector3(p_key_ideal, "p_key_ideal")

    C_tool_after = R @ C_tool + t
    a_after = C_tool_after - p_ideal
    b_after = C_obj - p_ideal
    return _normalize(np.cross(a_after, b_after), "n_int_after", eps=eps)


def refine_tool_alignment_by_keypoint_and_normal(
    C_tool_curr: Any,
    C_obj_curr: Any,
    p_key_curr: Any,
    p_key_ideal: Any,
    n_int_ideal: Any,
    s0: float,
    R0: Any,
    t0: Any,
    w_R: float = 1.0,
    w_t: float = 1.0,
    eps: float = 1e-8,
) -> RefinementResult:
    """
    Optimize rotation over the rigid constraint set. Translation is determined
    analytically by the keypoint constraint, while s0 is diagnostic metadata.

    Constraints:
        R @ p_key_curr + t = p_key_ideal
        cross(R @ C_tool_curr + t - p_key_ideal, C_obj_curr - p_key_ideal)
            is parallel to n_int_ideal

    Objective:
        1. Keep R close to R0.
        2. Keep t close to t0.

    Returns:
        RefinementResult containing the executable rigid transform and metrics.
    """
    C_tool = _as_vector3(C_tool_curr, "C_tool_curr")
    C_obj = _as_vector3(C_obj_curr, "C_obj_curr")
    p_key = _as_vector3(p_key_curr, "p_key_curr")
    p_ideal = _as_vector3(p_key_ideal, "p_key_ideal")
    n_ideal = _normalize(_as_vector3(n_int_ideal, "n_int_ideal"), "n_int_ideal", eps=eps)
    R0 = _as_rotation_matrix(R0, "R0")
    t0 = _as_vector3(t0, "t0")
    s0 = float(s0)

    if not np.isfinite(s0) or abs(s0) < eps:
        raise ValueError(f"s0 必须是非零有限数，当前 s0={s0}")

    tool_axis_curr = C_tool - p_key
    tool_axis_len = np.linalg.norm(tool_axis_curr)
    if tool_axis_len < eps:
        raise ValueError(f"C_tool_curr 与 p_key_curr 过近，无法构造工具轴: norm={tool_axis_len}")
    tool_axis_hat = tool_axis_curr / tool_axis_len

    obj_axis_ideal = C_obj - p_ideal
    obj_axis_hat = _normalize(obj_axis_ideal, "C_obj_curr - p_key_ideal", eps=eps)
    n_plane = n_ideal - np.dot(n_ideal, obj_axis_hat) * obj_axis_hat
    n_plane = _normalize(n_plane, "n_int_ideal projected to object-key plane", eps=eps)
    in_plane_perp = _normalize(np.cross(obj_axis_hat, n_plane), "tool-axis in-plane basis", eps=eps)

    sqrt_w_R = np.sqrt(float(w_R))
    sqrt_w_t = np.sqrt(float(w_t))

    def rotation_from_params(phi: float, psi: float) -> np.ndarray:
        tool_axis_ideal = (
            np.cos(phi) * obj_axis_hat
            + np.sin(phi) * in_plane_perp
        )
        tool_axis_ideal = _normalize(tool_axis_ideal, "tool_axis_ideal", eps=eps)
        R_align = _rotation_from_vector_to_vector(tool_axis_hat, tool_axis_ideal, eps=eps)
        R_spin = R_sci.from_rotvec(psi * tool_axis_ideal).as_matrix()
        return R_spin @ R_align

    def solve_from_initial(x0: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, Any]:
        def residual(angle_array: np.ndarray) -> np.ndarray:
            phi = float(angle_array[0])
            psi = float(angle_array[1])
            R_candidate = rotation_from_params(phi, psi)
            t_candidate = p_ideal - R_candidate @ p_key

            rot_residual = R_sci.from_matrix(R0.T @ R_candidate).as_rotvec()
            trans_residual = t_candidate - t0

            return np.concatenate([
                sqrt_w_R * rot_residual,
                sqrt_w_t * trans_residual,
            ])

        result = least_squares(
            residual,
            x0=np.asarray(x0, dtype=np.float64),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=200,
        )
        R_refined = rotation_from_params(float(result.x[0]), float(result.x[1]))
        t_refined = p_ideal - R_refined @ p_key
        return float(result.cost), R_refined, t_refined, result

    candidates = []
    phi_starts = np.linspace(-np.pi, np.pi, 9, endpoint=False)
    psi_starts = np.linspace(-np.pi, np.pi, 9, endpoint=False)
    for phi0 in phi_starts:
        for psi0 in psi_starts:
            cost, R_refined, t_refined, result = solve_from_initial(
                np.array([phi0, psi0], dtype=np.float64)
            )
            try:
                n_after_candidate = compute_transformed_triangle_normal(
                    R_refined,
                    t_refined,
                    C_tool,
                    C_obj,
                    p_ideal,
                    eps=eps,
                )
                normal_sign_candidate = 1.0 if np.dot(n_after_candidate, n_plane) >= 0 else -1.0
            except ValueError:
                normal_sign_candidate = 0.0
            candidates.append((cost, normal_sign_candidate, R_refined, t_refined, result))

    cost, normal_sign, R_refined, t_refined, opt_result = min(
        candidates,
        key=lambda item: item[0],
    )

    keypoint_after = R_refined @ p_key + t_refined
    keypoint_error = float(np.linalg.norm(keypoint_after - p_ideal))
    n_after = compute_transformed_triangle_normal(
        R_refined,
        t_refined,
        C_tool,
        C_obj,
        p_ideal,
        eps=eps,
    )
    normal_alignment_abs = float(abs(np.dot(n_after, n_plane)))
    rotation_deviation_rad = float(
        np.linalg.norm(R_sci.from_matrix(R0.T @ R_refined).as_rotvec())
    )
    translation_deviation = float(np.linalg.norm(t_refined - t0))

    return RefinementResult(
        success=bool(opt_result.success),
        s=s0,
        R=R_refined,
        t=t_refined,
        normal_curr=n_after,
        normal_ideal=n_plane,
        normal_sign=normal_sign,
        keypoint_error=keypoint_error,
        normal_alignment_abs=normal_alignment_abs,
        rotation_deviation_rad=rotation_deviation_rad,
        translation_deviation=translation_deviation,
        cost=float(cost),
        message=str(opt_result.message),
    )


def transform_points_with_refined_alignment(
    points: Any,
    s: float,
    R: Any,
    t: Any,
) -> np.ndarray:
    """Apply the refined rigid transform; s is diagnostic metadata."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points 必须为 Nx3，当前 shape={points.shape}")
    R = _as_rotation_matrix(R, "R")
    t = _as_vector3(t, "t")
    return points @ R.T + t
