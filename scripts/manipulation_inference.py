#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local manipulation-stage perception and coarse alignment."""

import argparse
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torchvision.transforms as transforms
from PIL import Image


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from dinov2 import DINOv2RGBEncoder
from matcher import Matcher
from refinement import refine_tool_alignment_by_keypoint_and_normal
from vlm_grounded_sam import VLMGuidedGroundedSAMPartSegmenter


def resolve_calibration_path(calibration_path=None):
    candidates = [
        calibration_path,
        os.environ.get("CALIBRATION_PATH"),
        PROJECT_ROOT / "camera" / "camera_parameters.json",
        PROJECT_ROOT / "camera" / "camera_parameter" / "extrinsic.pkl",
        PROJECT_ROOT / "camera_parameter" / "extrinsic.pkl",
        PROJECT_ROOT / "camera_parameter" / "Extrinsic.pkl",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())

    timestamped = sorted(
        glob.glob(str(PROJECT_ROOT / "camera_parameter" / "extrinsic*.pkl")),
        key=os.path.getmtime,
        reverse=True,
    )
    if timestamped:
        return str(Path(timestamped[0]).resolve())

    checked = [str(path) for path in candidates if path]
    raise FileNotFoundError(
        "Camera calibration was not found. Checked: " + ", ".join(checked)
    )


def load_camera_calibration(calibration_path):
    if Path(calibration_path).suffix.lower() == ".json":
        with open(calibration_path, "r", encoding="utf-8") as file:
            calibration = json.load(file)
        camera_matrix_key = "K" if "K" in calibration else "Intrinsic"
        camera_to_base_key = "T" if "T" in calibration else "Extrinsic"
    else:
        with open(calibration_path, "rb") as file:
            calibration = pickle.load(file)
        camera_matrix_key = "camera_matrix"
        camera_to_base_key = "T_cam2base"

    if not isinstance(calibration, dict):
        raise TypeError(
            f"Calibration must be a dictionary, got {type(calibration).__name__}."
        )
    if camera_matrix_key not in calibration:
        raise KeyError(f"{camera_matrix_key} is missing from {calibration_path}.")
    if camera_to_base_key not in calibration:
        raise KeyError(f"{camera_to_base_key} is missing from {calibration_path}.")

    camera_matrix = np.asarray(calibration[camera_matrix_key], dtype=np.float64)
    camera_to_base = np.asarray(calibration[camera_to_base_key], dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3, got {camera_matrix.shape}.")
    if camera_to_base.shape != (4, 4):
        raise ValueError(f"T_cam2base must be 4x4, got {camera_to_base.shape}.")
    return camera_matrix, camera_to_base


def to_json_safe(value):
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: to_json_safe(item) for key, item in value.items()}
    return value


def depth_to_cleaned_point_cloud(
    depth_img,
    camera_intrinsics,
    mask=None,
    depth_scale=1000.0,
    stat_nb_neighbors=10,
    stat_std_ratio=1.6,
    rad_nb_points=3,
    rad_radius=0.035,
):
    if depth_img is None or depth_img.size == 0:
        return np.empty((0, 3)), np.array([]), np.array([])

    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]
    height, width = depth_img.shape
    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()
    z_raw = depth_img.ravel()

    valid_depth = z_raw > 0
    u_valid = u_flat[valid_depth]
    v_valid = v_flat[valid_depth]
    z_valid = z_raw[valid_depth].astype(np.float64) / float(depth_scale)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != depth_img.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match depth shape {depth_img.shape}."
            )
        selected = mask.ravel()[valid_depth]
        u_valid = u_valid[selected]
        v_valid = v_valid[selected]
        z_valid = z_valid[selected]

    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    points = np.ascontiguousarray(
        np.column_stack((x, y, z_valid)), dtype=np.float64
    )
    if len(points) == 0:
        return points, v_valid, u_valid

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if stat_nb_neighbors > 0:
        point_cloud, indices = point_cloud.remove_statistical_outlier(
            nb_neighbors=stat_nb_neighbors,
            std_ratio=stat_std_ratio,
        )
        u_valid = u_valid[indices]
        v_valid = v_valid[indices]
    if rad_nb_points > 0 and len(point_cloud.points) > 0:
        point_cloud, indices = point_cloud.remove_radius_outlier(
            nb_points=rad_nb_points,
            radius=rad_radius,
        )
        u_valid = u_valid[indices]
        v_valid = v_valid[indices]
    return np.asarray(point_cloud.points), v_valid, u_valid



def rigid_transform_3d(source_points, target_points):
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if source_points.shape != target_points.shape:
        raise ValueError(
            f"Point sets must have the same shape, got "
            f"{source_points.shape} and {target_points.shape}."
        )
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("Point sets must have shape (N, 3).")

    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    covariance = (source_points - source_center).T @ (
        target_points - target_center
    )
    left_vectors, _, right_vectors_t = np.linalg.svd(covariance)
    rotation = right_vectors_t.T @ left_vectors.T
    if np.linalg.det(rotation) < 0:
        right_vectors_t[-1] *= -1
        rotation = right_vectors_t.T @ left_vectors.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def apply_transform_to_points(transform, points):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def read_template_text(template, key):
    if key not in template:
        raise KeyError(f"Template field is missing: {key}")
    value = template[key]
    return str(value.item() if np.asarray(value).ndim == 0 else value)


class ManipulationInference:
    def __init__(self, calibration_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.calibration_path = resolve_calibration_path(calibration_path)
        camera_matrix, self.camera_to_base = load_camera_calibration(
            self.calibration_path
        )
        self.camera_intrinsics = {
            "fx": camera_matrix[0, 0],
            "fy": camera_matrix[1, 1],
            "cx": camera_matrix[0, 2],
            "cy": camera_matrix[1, 2],
        }

        print(f"[Init] Calibration: {self.calibration_path}")
        print(f"[Init] Device: {self.device}")
        print("[Init] Loading DINOv2 encoder...")
        self.dino_encoder = DINOv2RGBEncoder(dino_size="vits14").to(self.device)
        self.dino_encoder.eval()
        self.dino_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.matcher = Matcher()
        self.segmenter = VLMGuidedGroundedSAMPartSegmenter(
            device=str(self.device),
            box_threshold=0.30,
            text_threshold=0.25,
            vlm_image_detail="high",
            vlm_timeout_seconds=120.0,
            vlm_max_retries=3,
        )

    def segment_part(
        self,
        rgb_path,
        whole_object_prompt,
        part_prompt,
        prefix,
        remove_object_prompt=None,
        remove_mode=1,
        num_fps_points=18,
        save_mask_npy=True,
        output_dir=None,
    ):
        result = self.segmenter.run(
            image_dir=str(Path(rgb_path).resolve().parent),
            whole_object_prompt=whole_object_prompt,
            part_prompt=part_prompt,
            remove_object_prompt=remove_object_prompt,
            remove_mode=remove_mode,
            num_fps_points=num_fps_points,
            image_name=Path(rgb_path).name,
            prefix=prefix,
            save_mask_npy=save_mask_npy,
            output_dir=output_dir,
        )
        mask = result.get("part_mask")
        if mask is None or not np.any(mask):
            raise RuntimeError(
                f"No mask was produced for part prompt: {part_prompt}"
            )
        return np.asarray(mask, dtype=bool)

    def extract_world_points_and_features(
        self,
        rgb_image,
        depth_image,
        mask,
        depth_scale,
        label,
    ):
        points_camera, valid_rows, valid_columns = depth_to_cleaned_point_cloud(
            depth_img=depth_image,
            camera_intrinsics=self.camera_intrinsics,
            mask=mask,
            depth_scale=depth_scale,
            stat_nb_neighbors=20,
            stat_std_ratio=1.6,
            rad_nb_points=3,
            rad_radius=0.025,
        )
        if len(points_camera) < 10:
            raise RuntimeError(
                f"{label} has too few valid depth points: {len(points_camera)}."
            )

        homogeneous = np.column_stack(
            (points_camera, np.ones(len(points_camera)))
        )
        points_world = (self.camera_to_base @ homogeneous.T).T[:, :3]
        image_crop, row_offset, column_offset = crop_local_window(rgb_image, mask)
        local_rows = np.clip(valid_rows - row_offset, 0, 517).astype(np.int64)
        local_columns = np.clip(
            valid_columns - column_offset, 0, 517
        ).astype(np.int64)
        image_tensor = self.dino_transform(
            Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dense_features = self.dino_encoder(image_tensor)
        point_features = (
            dense_features[0, :, local_rows, local_columns]
            .transpose(1, 0)
            .cpu()
            .numpy()
        )
        return points_world, point_features

    def run(
        self,
        rgb_path,
        depth_path,
        template_path,
        output_dir=None,
        depth_scale=1000.0,
        remove_object_prompt=None,
        remove_mode=1,
        num_fps_points=18,
    ):
        start_time = time.time()
        rgb_path = str(Path(rgb_path).expanduser().resolve())
        depth_path = str(Path(depth_path).expanduser().resolve())
        template_path = str(Path(template_path).expanduser().resolve())
        for input_path in (rgb_path, depth_path, template_path):
            if not Path(input_path).is_file():
                raise FileNotFoundError(f"Input file not found: {input_path}")

        output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else PROJECT_ROOT / "inference" / "result"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        with np.load(template_path, allow_pickle=True) as template:
            whole_target_object_text = read_template_text(
                template, "whole_target_object_text"
            )
            target_part_text = read_template_text(template, "target_part_text")
            whole_tool_object_text = read_template_text(
                template, "whole_tool_object_text"
            )
            tool_part_text = read_template_text(template, "tool_part_text")
            target_keypoints = np.asarray(
                template["target_part_keypoints"], dtype=np.float64
            )
            target_descriptors = np.asarray(
                template["target_part_descriptors"], dtype=np.float32
            )
            tool_keypoints = np.asarray(
                template["tool_part_keypoints"], dtype=np.float64
            )
            tool_descriptors = np.asarray(
                template["tool_part_descriptors"], dtype=np.float32
            )
            tool_key_index = int(
                np.asarray(template["tool_key_interaction_point_index"]).item()
            )
            interaction_normal_demo = np.asarray(
                template["interaction_normal_demo"], dtype=np.float64
            ).reshape(3)

        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb_image is None:
            raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
        if depth_image is None:
            raise RuntimeError(f"Failed to read depth image: {depth_path}")
        if depth_image.ndim != 2:
            raise ValueError(f"Depth image must be single-channel, got {depth_image.shape}.")
        if rgb_image.shape[:2] != depth_image.shape:
            raise ValueError(
                f"RGB shape {rgb_image.shape[:2]} does not match "
                f"depth shape {depth_image.shape}."
            )

        image_stem = Path(rgb_path).stem
        print("\n" + "=" * 72)
        print(f"[Run] RGB image: {rgb_path}")
        print(f"[Run] Depth image: {depth_path}")
        print(f"[Run] Template: {template_path}")
        print("=" * 72)

        print(
            f"[1/6] Segmenting target part: "
            f"{whole_target_object_text} / {target_part_text}"
        )
        target_mask = self.segment_part(
            rgb_path,
            whole_target_object_text,
            target_part_text,
            f"{image_stem}_current_target_part",
            remove_object_prompt,
            remove_mode,
            num_fps_points,
            save_mask_npy=False,
            output_dir=output_dir,
        )
        print("[2/6] Extracting target points and DINOv2 features...")
        target_world_points, current_target_descriptors = (
            self.extract_world_points_and_features(
                rgb_image,
                depth_image,
                target_mask,
                depth_scale,
                "Current target part",
            )
        )
        current_target_center = target_world_points.mean(axis=0)

        print("[3/6] Matching the template target part to the current scene...")
        target_template = {
            "feature": torch.from_numpy(target_descriptors).float().to(self.device),
            "position": target_keypoints,
            "num": len(target_keypoints),
        }
        target_transforms, matched_target_points = self.matcher.match(
            target_world_points,
            current_target_descriptors,
            target_template,
        )
        target_transform = target_transforms.get("global")
        if target_transform is None:
            raise RuntimeError("Target-part matching did not return a transform.")
        valid_target_matches = np.all(np.isfinite(matched_target_points), axis=1)
        if np.count_nonzero(valid_target_matches) < 3:
            raise RuntimeError("Fewer than three valid target-part matches remain.")
        ideal_tool_keypoints = apply_transform_to_points(
            target_transform, tool_keypoints
        )

        print(
            f"[4/6] Segmenting tool part: "
            f"{whole_tool_object_text} / {tool_part_text}"
        )
        tool_mask = self.segment_part(
            rgb_path,
            whole_tool_object_text,
            tool_part_text,
            f"{image_stem}_current_tool_part",
            remove_object_prompt,
            remove_mode,
            num_fps_points,
            output_dir=output_dir,
        )
        tool_mask = np.logical_and(tool_mask, np.logical_not(target_mask))
        if not np.any(tool_mask):
            raise RuntimeError(
                "The tool mask is empty after removing overlap with the target mask."
            )

        print("[5/6] Extracting tool points and DINOv2 features...")
        tool_world_points, current_tool_descriptors = (
            self.extract_world_points_and_features(
                rgb_image,
                depth_image,
                tool_mask,
                depth_scale,
                "Current tool part",
            )
        )
        print("[6/6] Matching the current tool part to its aligned template...")
        tool_template = {
            "feature": torch.from_numpy(tool_descriptors).float().to(self.device),
            "position": ideal_tool_keypoints,
            "num": len(ideal_tool_keypoints),
        }
        tool_transforms, matched_tool_points = self.matcher.match(
            tool_world_points,
            current_tool_descriptors,
            tool_template,
        )
        matched_tool_points = np.asarray(matched_tool_points, dtype=np.float64)
        if matched_tool_points.shape != ideal_tool_keypoints.shape:
            raise RuntimeError(
                "Tool correspondence output does not preserve template shape: "
                f"matched={matched_tool_points.shape}, "
                f"template={ideal_tool_keypoints.shape}."
            )

        tool_scale = float(tool_transforms.get("s", 1.0))
        valid_tool_matches = np.all(np.isfinite(matched_tool_points), axis=1)
        current_tool_valid_matches = matched_tool_points[valid_tool_matches]
        ideal_tool_matches = ideal_tool_keypoints[valid_tool_matches]
        if len(current_tool_valid_matches) < 3:
            raise RuntimeError("Fewer than three valid tool-part matches remain.")

        current_to_ideal_tool = rigid_transform_3d(
            current_tool_valid_matches, ideal_tool_matches
        )
        coarse_rotation = current_to_ideal_tool[:3, :3]
        coarse_translation = current_to_ideal_tool[:3, 3]

        rotation = coarse_rotation.copy()
        translation = coarse_translation.copy()
        refinement_applied = False
        refinement_fallback_reason = ""
        refinement_diagnostics = {
            "keypoint_error": float("nan"),
            "normal_alignment_abs": float("nan"),
            "rotation_deviation_rad": float("nan"),
            "translation_deviation": float("nan"),
        }

        if not 0 <= tool_key_index < len(matched_tool_points):
            refinement_fallback_reason = (
                "Interaction keypoint index is outside the template range: "
                f"{tool_key_index}."
            )
        elif not valid_tool_matches[tool_key_index]:
            refinement_fallback_reason = (
                "The interaction keypoint was rejected as a RANSAC outlier."
            )
        else:
            try:
                target_rotation = np.asarray(
                    target_transforms["R"], dtype=np.float64
                )
                ideal_interaction_normal = (
                    target_rotation @ interaction_normal_demo
                )
                refinement = refine_tool_alignment_by_keypoint_and_normal(
                    C_tool_curr=current_tool_valid_matches.mean(axis=0),
                    C_obj_curr=current_target_center,
                    p_key_curr=matched_tool_points[tool_key_index],
                    p_key_ideal=ideal_tool_keypoints[tool_key_index],
                    n_int_ideal=ideal_interaction_normal,
                    s0=tool_scale,
                    R0=coarse_rotation,
                    t0=coarse_translation,
                )
                if refinement.success:
                    rotation = refinement.R
                    translation = refinement.t
                    refinement_applied = True
                    refinement_diagnostics = {
                        "keypoint_error": refinement.keypoint_error,
                        "normal_alignment_abs": refinement.normal_alignment_abs,
                        "rotation_deviation_rad": refinement.rotation_deviation_rad,
                        "translation_deviation": refinement.translation_deviation,
                    }
                else:
                    refinement_fallback_reason = (
                        "Refinement did not converge: " + refinement.message
                    )
            except (KeyError, ValueError, np.linalg.LinAlgError) as error:
                refinement_fallback_reason = f"Refinement failed: {error}"

        if refinement_applied:
            print(
                "[Refinement] Applied: "
                f"keypoint_error={refinement_diagnostics['keypoint_error']:.8f}m, "
                f"normal_alignment={refinement_diagnostics['normal_alignment_abs']:.8f}, "
                f"rotation_deviation="
                f"{np.degrees(refinement_diagnostics['rotation_deviation_rad']):.4f}deg, "
                f"translation_deviation="
                f"{refinement_diagnostics['translation_deviation']:.6f}m"
            )
        else:
            print(
                "[Refinement] Skipped; returning coarse alignment. "
                f"Reason: {refinement_fallback_reason}"
            )

        elapsed = time.time() - start_time
        result_json_path = output_dir / f"{image_stem}_manipulation_result.json"
        result = {
            "status": "SUCCESS",
            "message": (
                "Manipulation-stage alignment completed with refinement."
                if refinement_applied
                else "Manipulation-stage coarse alignment completed; refinement skipped."
            ),
            "json_path": str(result_json_path),
            "s0": tool_scale,
            "R0": to_json_safe(rotation),
            "T0": to_json_safe(translation),
            "coarse_R0": to_json_safe(coarse_rotation),
            "coarse_T0": to_json_safe(coarse_translation),
            "refinement_applied": refinement_applied,
            "refinement_fallback_reason": refinement_fallback_reason,
            "refinement_keypoint_error": to_json_safe(
                refinement_diagnostics["keypoint_error"]
            ),
            "refinement_normal_alignment_abs": to_json_safe(
                refinement_diagnostics["normal_alignment_abs"]
            ),
            "refinement_rotation_deviation_rad": to_json_safe(
                refinement_diagnostics["rotation_deviation_rad"]
            ),
            "refinement_translation_deviation": to_json_safe(
                refinement_diagnostics["translation_deviation"]
            ),
            "object_part_transform": to_json_safe(target_transform),
            "ideal_tool_keypoints": to_json_safe(ideal_tool_keypoints),
            "object_part_matched_points": to_json_safe(matched_target_points),
            "current_tool_matched_points": to_json_safe(matched_tool_points),
            "object_part_scale": float(target_transforms.get("s", 1.0)),
            "processing_time_sec": elapsed,
        }
        with result_json_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        print(f"[Done] JSON result: {result_json_path}")
        print(f"[Done] Processing time: {elapsed:.3f}s")
        return result


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run local manipulation-stage perception and coarse alignment."
    )
    parser.add_argument("--rgb-path", required=True)
    parser.add_argument("--depth-path", required=True)
    parser.add_argument("--template-path", required=True)
    parser.add_argument(
        "--calibration-path",
        default=str(
            PROJECT_ROOT / "camera" / "camera_parameters.json"
        ),
    )
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--remove-object-prompt", default=None)
    parser.add_argument("--remove-mode", type=int, choices=(1, 2), default=1)
    parser.add_argument("--num-fps-points", type=int, default=18)
    return parser


def main():
    args = build_argument_parser().parse_args()
    inference = ManipulationInference(calibration_path=args.calibration_path)
    inference.run(
        rgb_path=args.rgb_path,
        depth_path=args.depth_path,
        template_path=args.template_path,
        depth_scale=args.depth_scale,
        remove_object_prompt=args.remove_object_prompt,
        remove_mode=args.remove_mode,
        num_fps_points=args.num_fps_points,
    )


if __name__ == "__main__":
    main()
