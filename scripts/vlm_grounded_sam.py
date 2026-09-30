#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import cv2
import numpy as np                                                                          
import torch

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

from VLM.call_api import (
    APIConfig,
    MultimodalPointSelectionClient,
)


CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "../GroundedSam/configs/groundingdino_swint_ogc.py"
)

DINO_CHECKPOINT = os.path.join(
    os.path.dirname(__file__),
    "../weight/groundingdino_swint_ogc.pth"
)

SAM_CHECKPOINT = os.path.join(
    os.path.dirname(__file__),
    "../weight/sam_vit_b_01ec64.pth"
)

SAM_MODEL_TYPE = "vit_b"


class VLMGuidedGroundedSAMPartSegmenter:
    """
    Segment an object part with GroundingDINO, SAM, and VLM guidance.

    Processing stages:
    1. Resolve an RGB image and the object and part prompts.
    2. Detect and segment the whole object with GroundingDINO and SAM.
    3. Sample points inside the whole-object mask with FPS and label them A-Z.
    4. Ask the VLM to identify labels that belong to the target part.
    5. Treat selected points as positive prompts and all remaining points as negative.
    6. Build a compact part box that contains every positive point and minimizes
       the number of enclosed negative points.
    7. Segment the target part with SAM using the points and part box.
    8. Save the masks and diagnostic visualizations.
    """

    def __init__(
        self,
        config_path=CONFIG_PATH,
        dino_checkpoint=DINO_CHECKPOINT,
        sam_checkpoint=SAM_CHECKPOINT,
        sam_model_type=SAM_MODEL_TYPE,
        device=None,
        box_threshold=0.20,
        text_threshold=0.15,
        vlm_image_detail="high",
        vlm_timeout_seconds=120.0,
        vlm_max_retries=3,
        vlm_system_instruction="",
    ):
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        print("[Init] Loading GroundingDINO...")
        self.grounding_dino_model = Model(
            model_config_path=config_path,
            model_checkpoint_path=dino_checkpoint,
            device=self.device
        )

        print("[Init] Loading SAM...")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam_predictor = SamPredictor(sam.to(self.device))

        print("[Init] Loading VLM API client...")
        vlm_config = APIConfig(
            model=self._get_default_vlm_model(),
            api_base=self._get_default_vlm_api_base(),
            image_detail=vlm_image_detail,
            timeout_seconds=vlm_timeout_seconds,
            max_retries=vlm_max_retries,
            max_images=2,
            system_instruction=vlm_system_instruction,
        )

        self.vlm_client = MultimodalPointSelectionClient(
            config=vlm_config,
            enable_logging=True,
        )

        print(f"[Init] Done. device = {self.device}")

    def _get_default_vlm_model(self):
        from VLM.vlm_config import VLM_MODEL
        return VLM_MODEL.strip()

    def _get_default_vlm_api_base(self):
        from VLM.vlm_config import VLM_API_BASE
        return VLM_API_BASE.strip()

    # ============================================================
    # Public API
    # ============================================================

    def run(
        self,
        image_dir,
        whole_object_prompt,
        part_prompt=None,
        remove_object_prompt=None,
        remove_mode=1,
        num_fps_points=20,
        image_name=None,
        prefix=None,
        save_mask_npy=True,
        output_dir=None,
    ):
        """
        Args:
            image_dir:
                Directory containing the input RGB image.
            whole_object_prompt:
                Whole-object name used by GroundingDINO and the VLM.
            part_prompt:
                Optional target-part name. When None, the method saves and
                returns the whole-object mask without part segmentation.
            remove_object_prompt:
                Optional name of an object to exclude before segmentation.
            remove_mode:
                1: Keep the image region to the left of the exclusion box.
                2: Segment the exclusion box with SAM and black out its mask.
            num_fps_points:
                Number of FPS points sampled inside the whole-object mask.
            image_name:
                Optional image filename. The first supported image is used
                when this argument is None.
            prefix:
                Output filename prefix. Defaults to the input image stem.
            save_mask_npy:
                Save whole-object and part masks as NPY files when true.
            output_dir:
                Optional directory for generated masks and visualizations.
        """

        image_dir = Path(image_dir).expanduser().resolve()
        image_path = self._resolve_image_path(image_dir, image_name)

        if prefix is None:
            prefix = image_path.stem

        output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else image_path.parent / "result"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("[Run] Input image:", image_path)
        print("[Run] Whole-object GroundingDINO prompt:", whole_object_prompt)
        print("[Run] Target part:", part_prompt)
        if remove_object_prompt:
            print("[Run] Object to remove beforehand:", remove_object_prompt)
            print("[Run] Pre-removal mode (remove_mode):", remove_mode)
        print("[Run] Output directory:", output_dir)
        print("=" * 80)

        if remove_mode not in (1, 2):
            raise ValueError(f"remove_mode only supports 1 or 2; current value: {remove_mode}")

        original_image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original_image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = original_image_bgr.shape[:2]

        image_bgr = original_image_bgr
        image_rgb = original_image_rgb

        # Mode 1 keeps the left image region, so the crop origin remains (0, 0).
        crop_offset_xy = np.array([0.0, 0.0], dtype=np.float32)

        crop_box_in_original = np.array(
            [0, 0, original_w - 1, original_h - 1],
            dtype=np.float32
        )

        cropped_image_path = None

        # ------------------------------------------------------------
        # Step 0: Optionally exclude a specified object.
        # ------------------------------------------------------------
        if remove_object_prompt:
            print(f"\n[Step 0] GroundingDINO detect remove object: {remove_object_prompt}")

            remove_boxes, remove_scores = self._grounding_detect(
                image_rgb=original_image_rgb,
                text_prompt=remove_object_prompt
            )

            if remove_boxes is None or len(remove_boxes) == 0:
                raise RuntimeError(f"GroundingDINO did not detect the object to remove: {remove_object_prompt}")

            remove_box = self._select_best_box_by_score(remove_boxes, remove_scores)

            if remove_mode == 1:
                crop_x = self._get_left_vertical_edge_x(
                    box_xyxy=remove_box,
                    image_width=original_w
                )

                image_bgr = original_image_bgr[:, :crop_x].copy()
                image_rgb = original_image_rgb[:, :crop_x].copy()

                crop_box_in_original = np.array(
                    [0, 0, crop_x - 1, original_h - 1],
                    dtype=np.float32
                )

                cropped_image_path = output_dir / f"{prefix}_cropped_input.png"
                cv2.imwrite(str(cropped_image_path), image_bgr)

                print("[Step 0] Candidate box for the object to remove:", remove_box)
                print("[Step 0] Crop boundary x:", crop_x)
                print(f"[Step 0] Saved cropped image: {cropped_image_path}")
            else:
                remove_mask, _ = self._sam_segment_with_box(
                    image_rgb=original_image_rgb,
                    box_xyxy=remove_box
                )
                remove_mask = self._keep_largest_component(remove_mask)

                image_bgr = original_image_bgr.copy()
                image_rgb = original_image_rgb.copy()
                image_bgr[remove_mask] = 0
                image_rgb[remove_mask] = 0

                cropped_image_path = output_dir / f"{prefix}_masked_remove_input.png"
                cv2.imwrite(str(cropped_image_path), image_bgr)

                print("[Step 0] Candidate box for the object to remove:", remove_box)
                print("[Step 0] remove mask area:", int(remove_mask.sum()))
                print(f"[Step 0] Saved image with the masked region blacked out: {cropped_image_path}")

        # ------------------------------------------------------------
        # Step 1: Segment the whole object with GroundingDINO and SAM.
        # ------------------------------------------------------------
        print(f"\n[Step 1] GroundingSAM segment whole object: {whole_object_prompt}")

        whole_boxes, whole_scores = self._grounding_detect(
            image_rgb=image_rgb,
            text_prompt=whole_object_prompt
        )

        if whole_boxes is None or len(whole_boxes) == 0:
            raise RuntimeError(f"GroundingDINO did not detect the whole object: {whole_object_prompt}")

        whole_box = self._select_best_box_by_score(whole_boxes, whole_scores)

        whole_mask, _ = self._sam_segment_with_box(
            image_rgb=image_rgb,
            box_xyxy=whole_box
        )

        whole_mask = self._keep_largest_component(whole_mask)

        whole_mask_vis_path = output_dir / f"{prefix}_whole_mask_vis.png"
        whole_mask_npy_path = (
            output_dir / f"{prefix}_whole_Mask.npy"
            if save_mask_npy
            else None
        )

        whole_mask_original = self._restore_mask_to_original_size(
            mask=whole_mask,
            original_shape=original_image_bgr.shape[:2],
            crop_box_xyxy=crop_box_in_original
        )

        whole_box_original = self._restore_box_to_original_coords(
            box_xyxy=whole_box,
            offset_xy=crop_offset_xy
        )

        self._save_mask_overlay(
            image_bgr=original_image_bgr,
            mask=whole_mask_original,
            save_path=whole_mask_vis_path,
            box=whole_box_original,
            title_text=f"whole object: {whole_object_prompt}",
            mask_color=(0, 255, 0),
            box_thickness=3
        )

        if whole_mask_npy_path is not None:
            np.save(str(whole_mask_npy_path), whole_mask_original.astype(np.uint8))

        print(f"[Step 1] Whole-object box in cropped coordinates: {whole_box}")
        print(f"[Step 1] Saved whole-object mask visualization: {whole_mask_vis_path}")
        if whole_mask_npy_path is not None:
            print(f"[Step 1] Saved whole-object mask NPY: {whole_mask_npy_path}")

        if part_prompt is None:
            print("\n" + "=" * 80)
            print("[Done] No part_prompt was provided; only whole-object mask segmentation was completed. Saved files:")
            if cropped_image_path is not None:
                print("0.", cropped_image_path)
            print("1.", whole_mask_vis_path)
            if whole_mask_npy_path is not None:
                print("2.", whole_mask_npy_path)
            print("=" * 80)

            return {
                "image_path": str(image_path),
                "cropped_image_path": str(cropped_image_path) if cropped_image_path is not None else None,
                "remove_object_prompt": remove_object_prompt,
                "whole_object_prompt": whole_object_prompt,
                "part_prompt": None,
                "whole_mask": whole_mask_original,
                "whole_mask_cropped": whole_mask,
                "whole_box": whole_box_original,
                "whole_box_cropped": whole_box,
                "whole_mask_npy_path": str(whole_mask_npy_path) if whole_mask_npy_path is not None else None,
                "part_mask": whole_mask_original,
                "part_mask_cropped": whole_mask,
                "part_mask_npy_path": str(whole_mask_npy_path) if whole_mask_npy_path is not None else None,
                "saved_images": {
                    "cropped_input": str(cropped_image_path) if cropped_image_path is not None else None,
                    "whole_mask_vis": str(whole_mask_vis_path),
                    "whole_mask_npy": str(whole_mask_npy_path) if whole_mask_npy_path is not None else None,
                    "part_mask_vis": str(whole_mask_vis_path),
                    "part_mask_npy": str(whole_mask_npy_path) if whole_mask_npy_path is not None else None,
                }
            }

        # ------------------------------------------------------------
        # Step 2: Sample FPS points in the whole-object mask and label them A-Z.
        # ------------------------------------------------------------
        print(f"\n[Step 2] FPS sampling in whole mask, num_points = {num_fps_points}")

        sampled_points = self._fps_sample_points_in_mask(
            mask=whole_mask,
            num_points=num_fps_points
        )

        indexed_points_vis_path = output_dir / f"{prefix}_fps_points_lettered.png"

        self._save_points_with_indices(
            image_bgr=image_bgr,
            mask=whole_mask,
            points_xy=sampled_points,
            save_path=indexed_points_vis_path,
            box=whole_box
        )

        print(f"[Step 2] Saved letter-labeled point image: {indexed_points_vis_path}")

        # ------------------------------------------------------------
        # Step 3: Ask the VLM to select points that belong to the target part.
        # ------------------------------------------------------------
        print("\n[Step 3] VLM selection of positive point indices...")
        reference_image_path = (
            cropped_image_path if cropped_image_path is not None else image_path
        )
        vlm_result_path = output_dir / f"{prefix}_vlm_point_selection.json"
        print(f"[Step 3] Letter-labeled image: {indexed_points_vis_path}")
        print(f"[Step 3] Reference image: {reference_image_path}")

        vlm_result = self.vlm_client.select_part_points(
            object_name=whole_object_prompt,
            part_name=part_prompt,
            image_paths=[indexed_points_vis_path, reference_image_path],
            output_path=vlm_result_path,
        )

        positive_indices = self._sanitize_indices(
            indices=vlm_result.get("selected_point_indices", []),
            num_points=len(sampled_points)
        )

        negative_indices = [
            i for i in range(len(sampled_points))
            if i not in positive_indices
        ]

        if len(positive_indices) == 0:
            raise RuntimeError(
                "The VLM returned no valid positive points, so part-mask segmentation cannot continue."
            )

        print("[Step 3] VLM selected positive indices:", positive_indices)
        print("[Step 3] Negative indices:", negative_indices)

        positive_points = sampled_points[positive_indices]
        negative_points = sampled_points[negative_indices]

        point_coords = np.concatenate(
            [positive_points, negative_points],
            axis=0
        ).astype(np.float32)

        point_labels = np.concatenate(
            [
                np.ones(len(positive_points), dtype=np.int32),
                np.zeros(len(negative_points), dtype=np.int32),
            ],
            axis=0
        )

        posneg_vis_path = output_dir / f"{prefix}_vlm_posneg_points.png"

        # ------------------------------------------------------------
        # Step 4: Build a compact part box from the whole box and point labels.
        # ------------------------------------------------------------
        print("\n[Step 4] Build part box from whole box and VLM-selected points...")

        part_box = self._build_part_box_from_points_avoid_negative(
            positive_points_xy=positive_points,
            negative_points_xy=negative_points,
            outer_box_xyxy=whole_box,
            image_shape=image_rgb.shape[:2],
            margin_candidates=(32, 64, 128, 256),
            min_box_size=20,
        )

        print("[Step 4] Using the part box constructed from positive and negative points:", part_box)

        self._save_posneg_points_visualization(
            image_bgr=image_bgr,
            mask=whole_mask,
            points_xy=sampled_points,
            positive_indices=positive_indices,
            save_path=posneg_vis_path,
            part_box=part_box
        )

        print(f"[Step 4] Saved VLM positive/negative-point image with part_box: {posneg_vis_path}")

        part_box_vis_path = output_dir / f"{prefix}_point_based_part_box_vis.png"

        self._save_box_overlay(
            image_bgr=image_bgr,
            save_path=part_box_vis_path,
            outer_box=whole_box,
            part_box=part_box,
            positive_points_xy=positive_points,
            negative_points_xy=negative_points,
            title_text="point-based part box"
        )

        print(f"[Step 4] Saved visualization of the part box constructed from positive/negative points: {part_box_vis_path}")

        # ------------------------------------------------------------
        # Step 5: Segment the part with SAM using point prompts and the part box.
        # ------------------------------------------------------------
        print("\n[Step 5] SAM segment target part with VLM-selected points and point-based box...")

        part_mask, _ = self._sam_segment_with_points_and_optional_box(
            image_rgb=image_rgb,
            point_coords=point_coords,
            point_labels=point_labels,
            box_xyxy=part_box,
            whole_object_mask=whole_mask
        )

        part_mask = np.logical_and(part_mask, whole_mask)

        part_mask = self._keep_components_containing_positive_points(
            mask=part_mask,
            positive_points_xy=positive_points
        )

        part_mask = self._keep_largest_component(part_mask)

        part_mask_vis_path = output_dir / f"{prefix}_part_mask_vis.png"
        part_mask_npy_path = (
            output_dir / f"{prefix}_Mask.npy"
            if save_mask_npy
            else None
        )

        part_mask_original = self._restore_mask_to_original_size(
            mask=part_mask,
            original_shape=original_image_bgr.shape[:2],
            crop_box_xyxy=crop_box_in_original
        )

        if part_mask_npy_path is not None:
            np.save(str(part_mask_npy_path), part_mask_original.astype(np.uint8))

        part_box_original = self._restore_box_to_original_coords(
            box_xyxy=part_box,
            offset_xy=crop_offset_xy
        )

        self._save_mask_overlay(
            image_bgr=original_image_bgr,
            mask=part_mask_original,
            save_path=part_mask_vis_path,
            box=part_box_original,
            title_text=f"final part: {part_prompt}",
            mask_color=(0, 255, 0)
        )

        # Mode 1 also saves a part-mask overlay in cropped-image coordinates.
        if remove_object_prompt and remove_mode == 1:
            cropped_part_mask_vis_path = output_dir / f"{prefix}_cropped_part_mask_vis.png"
            self._save_mask_overlay(
                image_bgr=image_bgr,
                mask=part_mask,
                save_path=cropped_part_mask_vis_path,
                box=part_box,
                title_text=f"final part: {part_prompt}",
                mask_color=(0, 255, 0)
            )
            print(f"[Step 5] Saved final part-mask visualization at cropped size: {cropped_part_mask_vis_path}")

        print(f"[Step 5] Saved final part-mask visualization: {part_mask_vis_path}")
        if part_mask_npy_path is not None:
            print(f"[Step 5] Saved final part-mask NPY: {part_mask_npy_path}")

        print("\n" + "=" * 80)
        print("[Done] Saved the following files:")
        if cropped_image_path is not None:
            print("0.", cropped_image_path)
        print("1.", whole_mask_vis_path)
        print("2.", indexed_points_vis_path)
        print("3.", posneg_vis_path)
        print("4.", part_box_vis_path)
        print("5.", part_mask_vis_path)
        if part_mask_npy_path is not None:
            print("6.", part_mask_npy_path)
        print("=" * 80)

        return {
            "image_path": str(image_path),
            "cropped_image_path": str(cropped_image_path) if cropped_image_path is not None else None,
            "remove_object_prompt": remove_object_prompt,
            "whole_object_prompt": whole_object_prompt,
            "part_prompt": part_prompt,
            "whole_mask": whole_mask_original,
            "whole_mask_cropped": whole_mask,
            "whole_box": whole_box_original,
            "whole_box_cropped": whole_box,
            "whole_mask_npy_path": str(whole_mask_npy_path) if whole_mask_npy_path is not None else None,
            "sampled_points": sampled_points,
            "positive_indices": positive_indices,
            "negative_indices": negative_indices,
            "positive_points": positive_points,
            "negative_points": negative_points,
            "part_box": part_box_original,
            "part_box_cropped": part_box,
            "part_mask": part_mask_original,
            "part_mask_cropped": part_mask,
            "part_mask_npy_path": str(part_mask_npy_path) if part_mask_npy_path is not None else None,
            "saved_images": {
                "cropped_input": str(cropped_image_path) if cropped_image_path is not None else None,
                "whole_mask_vis": str(whole_mask_vis_path),
                "whole_mask_npy": str(whole_mask_npy_path) if whole_mask_npy_path is not None else None,
                "indexed_points_vis": str(indexed_points_vis_path),
                "vlm_point_selection": str(vlm_result_path),
                "posneg_points_vis": str(posneg_vis_path),
                "point_based_part_box_vis": str(part_box_vis_path),
                "part_mask_vis": str(part_mask_vis_path),
                "part_mask_npy": str(part_mask_npy_path) if part_mask_npy_path is not None else None,
            }
        }

    # ============================================================
    # Path handling
    # ============================================================

    def _resolve_image_path(self, image_dir: Path, image_name=None) -> Path:
        if not image_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {image_dir}")

        if not image_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {image_dir}")

        if image_name is not None:
            image_path = image_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"Specified image does not exist: {image_path}")
            return image_path.resolve()

        valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]

        image_paths = []

        for ext in valid_exts:
            image_paths.extend(image_dir.glob(f"*{ext}"))

        image_paths = sorted(image_paths)

        if len(image_paths) == 0:
            raise FileNotFoundError(f"No RGB images found in directory: {image_dir}")

        if len(image_paths) > 1:
            print("[Warning] Multiple images found in the directory; using the first one by default:")
            for p in image_paths:
                print("  -", p)
            print("[Warning] Currently using:", image_paths[0])

        return image_paths[0].resolve()

    def _get_left_vertical_edge_x(self, box_xyxy, image_width):
        x1 = int(np.floor(float(box_xyxy[0])))
        x1 = int(np.clip(x1, 1, image_width))

        if x1 <= 1:
            raise RuntimeError(
                "The left boundary of the object to remove is too close to the left edge, resulting in insufficient cropped-image width."
            )

        return x1

    def _restore_mask_to_original_size(self, mask, original_shape, crop_box_xyxy):
        original_h, original_w = original_shape
        out = np.zeros((original_h, original_w), dtype=bool)

        x1, y1, x2, y2 = [int(round(float(v))) for v in crop_box_xyxy]

        x1 = int(np.clip(x1, 0, original_w - 1))
        y1 = int(np.clip(y1, 0, original_h - 1))
        x2 = int(np.clip(x2, 0, original_w - 1))
        y2 = int(np.clip(y2, 0, original_h - 1))

        crop_h = y2 - y1 + 1
        crop_w = x2 - x1 + 1

        if crop_h <= 0 or crop_w <= 0:
            raise RuntimeError("The crop region is invalid; the mask cannot be restored to the original image size.")

        if mask.shape[:2] != (crop_h, crop_w):
            raise RuntimeError(
                f"Mask size {mask.shape[:2]} does not match crop-region size {(crop_h, crop_w)}."
            )

        out[y1:y2 + 1, x1:x2 + 1] = mask.astype(bool)

        return out

    def _restore_box_to_original_coords(self, box_xyxy, offset_xy):
        if box_xyxy is None:
            return None

        offset_xy = np.asarray(offset_xy, dtype=np.float32)
        box = np.asarray(box_xyxy, dtype=np.float32).copy()

        box[[0, 2]] += offset_xy[0]
        box[[1, 3]] += offset_xy[1]

        return box

    # ============================================================
    # GroundingDINO
    # ============================================================

    def _grounding_detect(self, image_rgb, text_prompt):
        detections = self.grounding_dino_model.predict_with_caption(
            image=image_rgb,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        detection_result = detections[0] if isinstance(detections, (list, tuple)) else detections

        if detection_result is None:
            return None, None

        boxes = np.array(detection_result.xyxy, dtype=np.float32)

        if boxes.size == 0:
            return None, None

        if boxes.ndim == 1:
            boxes = boxes[None, :]

        if hasattr(detection_result, "confidence"):
            scores = np.array(detection_result.confidence, dtype=np.float32)
        else:
            scores = np.ones((len(boxes),), dtype=np.float32)

        return boxes, scores

    def _select_best_box_by_score(self, boxes_xyxy, scores):
        idx = int(np.argmax(scores))
        return boxes_xyxy[idx].astype(np.float32)

    def _select_best_part_box_with_object_constraint(self, boxes_xyxy, scores, object_mask):
        """
        Select a detection box using confidence and overlap with the object mask.
        """
        h, w = object_mask.shape
        best_score = -1e9
        best_box = None

        for box, score in zip(boxes_xyxy, scores):
            x1, y1, x2, y2 = box.astype(np.int32)

            x1 = int(np.clip(x1, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y2 = int(np.clip(y2, 0, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            box_mask = np.zeros_like(object_mask, dtype=np.uint8)
            box_mask[y1:y2 + 1, x1:x2 + 1] = 1

            overlap = np.logical_and(box_mask > 0, object_mask).sum()
            box_area = max((x2 - x1 + 1) * (y2 - y1 + 1), 1)
            overlap_ratio = overlap / float(box_area)

            combined = float(score) + 0.5 * float(overlap_ratio)

            if overlap_ratio > 0.05 and combined > best_score:
                best_score = combined
                best_box = np.array([x1, y1, x2, y2], dtype=np.float32)

        if best_box is None:
            best_box = self._select_best_box_by_score(boxes_xyxy, scores)

        return best_box

    # ============================================================
    # SAM
    # ============================================================

    def _sam_segment_with_box(self, image_rgb, box_xyxy=None):
        self.sam_predictor.set_image(image_rgb)

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_xyxy,
            multimask_output=True
        )

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(bool)
        best_score = float(scores[best_idx])

        return best_mask, best_score

    def _sam_segment_with_points_and_optional_box(
        self,
        image_rgb,
        point_coords,
        point_labels,
        box_xyxy=None,
        whole_object_mask=None,
    ):
        self.sam_predictor.set_image(image_rgb)

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_xyxy,
            multimask_output=True
        )

        best_idx = self._choose_best_sam_mask(
            masks=masks,
            scores=scores,
            point_coords=point_coords,
            point_labels=point_labels,
            whole_object_mask=whole_object_mask
        )

        best_mask = masks[best_idx].astype(bool)
        best_score = float(scores[best_idx])

        return best_mask, best_score

    def _choose_best_sam_mask(
        self,
        masks,
        scores,
        point_coords,
        point_labels,
        whole_object_mask=None
    ):
        best_value = -1e9
        best_idx = 0

        point_coords = np.asarray(point_coords, dtype=np.float32)
        point_labels = np.asarray(point_labels, dtype=np.int32)

        pos_points = point_coords[point_labels == 1]
        neg_points = point_coords[point_labels == 0]

        for i in range(len(masks)):
            mask = masks[i].astype(bool)

            if whole_object_mask is not None:
                mask = np.logical_and(mask, whole_object_mask)

            pos_hit = self._count_points_inside_mask(mask, pos_points)
            neg_hit = self._count_points_inside_mask(mask, neg_points)

            pos_hit_rate = pos_hit / max(len(pos_points), 1)

            if len(neg_points) > 0:
                neg_reject_rate = 1.0 - (neg_hit / len(neg_points))
            else:
                neg_reject_rate = 1.0

            value = (
                2.0 * pos_hit_rate +
                1.5 * neg_reject_rate +
                0.2 * float(scores[i])
            )

            if value > best_value:
                best_value = value
                best_idx = i

        return best_idx

    # ============================================================
    # Farthest point sampling
    # ============================================================

    def _fps_sample_points_in_mask(self, mask, num_points):
        ys, xs = np.where(mask)

        if len(xs) == 0:
            raise RuntimeError("The input mask is empty; FPS sampling cannot be performed.")

        coords_xy = np.stack([xs, ys], axis=1).astype(np.float32)

        if len(coords_xy) <= num_points:
            return coords_xy

        centroid = np.mean(coords_xy, axis=0, keepdims=True)
        dist_to_centroid = np.sum((coords_xy - centroid) ** 2, axis=1)
        first_idx = int(np.argmin(dist_to_centroid))

        selected_indices = [first_idx]
        min_dist = np.sum((coords_xy - coords_xy[first_idx]) ** 2, axis=1)

        for _ in range(1, num_points):
            farthest_idx = int(np.argmax(min_dist))
            selected_indices.append(farthest_idx)

            dist_to_new = np.sum(
                (coords_xy - coords_xy[farthest_idx]) ** 2,
                axis=1
            )

            min_dist = np.minimum(min_dist, dist_to_new)

        return coords_xy[selected_indices]

    # ============================================================
    # Part-box construction from positive and negative points
    # ============================================================

    def _build_part_box_from_points_avoid_negative(
        self,
        positive_points_xy,
        negative_points_xy,
        outer_box_xyxy,
        image_shape,
        margin_candidates=(4, 8, 12, 16, 24, 32, 48),
        min_box_size=20,
    ):
        """
        Build a part box from the VLM-selected positive and negative points.

        The box must remain inside outer_box_xyxy and contain every positive
        point. Candidate boxes are ranked by the number of enclosed negative
        points and then by area. Margins keep the SAM box from being too tight.
        """

        positive_points_xy = np.asarray(positive_points_xy, dtype=np.float32)
        negative_points_xy = np.asarray(negative_points_xy, dtype=np.float32)

        if positive_points_xy.ndim != 2 or positive_points_xy.shape[1] != 2:
            raise ValueError("positive_points_xy must be an array with shape [N, 2].")

        if len(positive_points_xy) == 0:
            raise RuntimeError("No positive points are available to construct the part box.")

        if negative_points_xy.size == 0:
            negative_points_xy = np.zeros((0, 2), dtype=np.float32)

        outer_box_xyxy = np.asarray(outer_box_xyxy, dtype=np.float32)

        if outer_box_xyxy.shape[0] != 4:
            raise ValueError("outer_box_xyxy must be [x1, y1, x2, y2].")

        # Minimal axis-aligned box enclosing all positive points.
        px_min = float(np.min(positive_points_xy[:, 0]))
        py_min = float(np.min(positive_points_xy[:, 1]))
        px_max = float(np.max(positive_points_xy[:, 0]))
        py_max = float(np.max(positive_points_xy[:, 1]))

        best_box = None
        best_score = None

        for margin in margin_candidates:
            candidate = np.array(
                [
                    px_min - margin,
                    py_min - margin,
                    px_max + margin,
                    py_max + margin,
                ],
                dtype=np.float32
            )

            candidate = self._ensure_min_box_size(
                box_xyxy=candidate,
                min_box_size=min_box_size,
            )

            candidate = self._clip_box_to_image_and_outer_box(
                box_xyxy=candidate,
                image_shape=image_shape,
                outer_box_xyxy=outer_box_xyxy,
            )

            if candidate is None:
                continue

            if not self._box_contains_all_points(candidate, positive_points_xy):
                continue

            neg_inside = self._count_points_inside_box(candidate, negative_points_xy)

            x1, y1, x2, y2 = candidate
            area = float(max(x2 - x1 + 1.0, 1.0) * max(y2 - y1 + 1.0, 1.0))

            # Minimize enclosed negative points first, then minimize area.
            score = neg_inside * 1000000.0 + area

            if best_score is None or score < best_score:
                best_score = score
                best_box = candidate

        if best_box is None:
            fallback = np.array(
                [
                    px_min - min_box_size,
                    py_min - min_box_size,
                    px_max + min_box_size,
                    py_max + min_box_size,
                ],
                dtype=np.float32
            )

            fallback = self._clip_box_to_image_and_outer_box(
                box_xyxy=fallback,
                image_shape=image_shape,
                outer_box_xyxy=outer_box_xyxy,
            )

            if fallback is None:
                raise RuntimeError("Unable to construct a valid part_box from the positive points.")

            best_box = fallback

        return best_box.astype(np.float32)

    def _ensure_min_box_size(self, box_xyxy, min_box_size=20):
        """
        Expand a box to the requested minimum width and height.

        Image-boundary clipping is handled separately.
        """
        box = np.asarray(box_xyxy, dtype=np.float32).copy()
        x1, y1, x2, y2 = box

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        bw = max(x2 - x1 + 1.0, float(min_box_size))
        bh = max(y2 - y1 + 1.0, float(min_box_size))

        half_w = 0.5 * bw
        half_h = 0.5 * bh

        nx1 = cx - half_w
        nx2 = cx + half_w
        ny1 = cy - half_h
        ny2 = cy + half_h

        return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

    def _clip_box_to_image_and_outer_box(
        self,
        box_xyxy,
        image_shape,
        outer_box_xyxy=None,
    ):
        """
        Clip a box to the image bounds and an optional outer box.
        """
        img_h, img_w = image_shape[:2]

        box = np.asarray(box_xyxy, dtype=np.float32).copy()
        x1, y1, x2, y2 = box

        x1 = float(np.clip(x1, 0, img_w - 1))
        x2 = float(np.clip(x2, 0, img_w - 1))
        y1 = float(np.clip(y1, 0, img_h - 1))
        y2 = float(np.clip(y2, 0, img_h - 1))

        if outer_box_xyxy is not None:
            ox1, oy1, ox2, oy2 = [float(v) for v in outer_box_xyxy]

            ox1 = float(np.clip(ox1, 0, img_w - 1))
            ox2 = float(np.clip(ox2, 0, img_w - 1))
            oy1 = float(np.clip(oy1, 0, img_h - 1))
            oy2 = float(np.clip(oy2, 0, img_h - 1))

            x1 = max(x1, ox1)
            y1 = max(y1, oy1)
            x2 = min(x2, ox2)
            y2 = min(y2, oy2)

        if x2 <= x1 or y2 <= y1:
            return None

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _box_contains_all_points(self, box_xyxy, points_xy):
        points_xy = np.asarray(points_xy, dtype=np.float32)

        if len(points_xy) == 0:
            return True

        x1, y1, x2, y2 = [float(v) for v in box_xyxy]

        xs = points_xy[:, 0]
        ys = points_xy[:, 1]

        inside = (
            (xs >= x1) &
            (xs <= x2) &
            (ys >= y1) &
            (ys <= y2)
        )

        return bool(np.all(inside))

    def _count_points_inside_box(self, box_xyxy, points_xy):
        points_xy = np.asarray(points_xy, dtype=np.float32)

        if len(points_xy) == 0:
            return 0

        x1, y1, x2, y2 = [float(v) for v in box_xyxy]

        xs = points_xy[:, 0]
        ys = points_xy[:, 1]

        inside = (
            (xs >= x1) &
            (xs <= x2) &
            (ys >= y1) &
            (ys <= y2)
        )

        return int(np.sum(inside))

    # ============================================================
    # Post-processing utilities
    # ============================================================

    def _sanitize_indices(self, indices, num_points):
        out = []

        for idx in indices:
            # Accept both numeric indices and uppercase labels from the VLM.
            if isinstance(idx, str):
                value = idx.strip().upper()
                if len(value) == 1 and "A" <= value <= "Z":
                    idx_int = ord(value) - ord("A")
                else:
                    try:
                        idx_int = int(value)
                    except Exception:
                        continue
            else:
                try:
                    idx_int = int(idx)
                except Exception:
                    continue

            if 0 <= idx_int < num_points:
                out.append(idx_int)

        return sorted(list(set(out)))

    def _count_points_inside_mask(self, mask, points_xy):
        if len(points_xy) == 0:
            return 0

        h, w = mask.shape
        count = 0

        for x, y in points_xy:
            xi = int(round(float(x)))
            yi = int(round(float(y)))

            xi = int(np.clip(xi, 0, w - 1))
            yi = int(np.clip(yi, 0, h - 1))

            if mask[yi, xi]:
                count += 1

        return count

    def _keep_largest_component(self, mask):
        mask_u8 = mask.astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8
        )

        if num_labels <= 1:
            return mask.astype(bool)

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_id = 1 + int(np.argmax(areas))

        return labels == largest_id

    def _keep_components_containing_positive_points(self, mask, positive_points_xy):
        mask_u8 = mask.astype(np.uint8)

        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8
        )

        if num_labels <= 1:
            return mask.astype(bool)

        h, w = mask.shape
        keep_labels = set()

        for x, y in positive_points_xy:
            xi = int(round(float(x)))
            yi = int(round(float(y)))

            xi = int(np.clip(xi, 0, w - 1))
            yi = int(np.clip(yi, 0, h - 1))

            lab = labels[yi, xi]

            if lab > 0:
                keep_labels.add(int(lab))

        if len(keep_labels) == 0:
            return self._keep_largest_component(mask)

        out = np.zeros_like(mask, dtype=bool)

        for lab in keep_labels:
            out |= labels == lab

        return out

    def _remove_components_containing_negative_points(self, mask, negative_points_xy):
        """
        Remove connected components that contain negative points.

        A component containing both positive and negative points is removed.
        """
        mask_u8 = mask.astype(np.uint8)

        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8
        )

        if num_labels <= 1:
            return mask.astype(bool)

        h, w = mask.shape
        remove_labels = set()

        for x, y in negative_points_xy:
            xi = int(round(float(x)))
            yi = int(round(float(y)))

            xi = int(np.clip(xi, 0, w - 1))
            yi = int(np.clip(yi, 0, h - 1))

            lab = labels[yi, xi]

            if lab > 0:
                remove_labels.add(int(lab))

        out = mask.astype(bool).copy()

        for lab in remove_labels:
            out[labels == lab] = False

        return out

    # ============================================================
    # Visualization output
    # ============================================================

    def _save_mask_overlay(
        self,
        image_bgr,
        mask,
        save_path,
        box=None,
        title_text=None,
        mask_color=(0, 255, 0),
        box_thickness=2,
    ):
        vis = image_bgr.copy()

        color = np.array(mask_color, dtype=np.uint8)

        if np.any(mask):
            vis[mask] = cv2.addWeighted(
                vis[mask],
                0.5,
                np.full_like(vis[mask], color),
                0.5,
                0
            )

        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), box_thickness)

        if title_text is not None:
            cv2.putText(
                vis,
                title_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )
            cv2.putText(
                vis,
                title_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        cv2.imwrite(str(save_path), vis)

    def _save_box_overlay(
        self,
        image_bgr,
        save_path,
        outer_box=None,
        part_box=None,
        positive_points_xy=None,
        negative_points_xy=None,
        title_text=None,
    ):
        """
        Save a diagnostic visualization of the point-based part box.

        The whole-object box is green, the part box is red, positive points
        are red circles, and negative points are blue circles.
        """
        vis = image_bgr.copy()

        if outer_box is not None:
            x1, y1, x2, y2 = [int(round(float(v))) for v in outer_box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if part_box is not None:
            x1, y1, x2, y2 = [int(round(float(v))) for v in part_box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

        if negative_points_xy is not None:
            for x, y in negative_points_xy:
                x = int(round(float(x)))
                y = int(round(float(y)))
                cv2.circle(vis, (x, y), 5, (255, 0, 0), -1)

        if positive_points_xy is not None:
            for x, y in positive_points_xy:
                x = int(round(float(x)))
                y = int(round(float(y)))
                cv2.circle(vis, (x, y), 7, (0, 0, 255), -1)
                cv2.circle(vis, (x, y), 10, (255, 255, 255), 2)

        if title_text is not None:
            cv2.putText(
                vis,
                title_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )
            cv2.putText(
                vis,
                title_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        cv2.imwrite(str(save_path), vis)

    def _save_points_with_indices(self, image_bgr, mask, points_xy, save_path, box=None):
        vis = image_bgr.copy()

        green = np.array([0, 180, 0], dtype=np.uint8)

        if np.any(mask):
            vis[mask] = cv2.addWeighted(
                vis[mask],
                0.65,
                np.full_like(vis[mask], green),
                0.35,
                0
            )

        if box is not None:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

        if len(points_xy) > 26:
            raise ValueError(
                f"The number of FPS sample points is {len(points_xy)}, exceeding the maximum of 26 representable by A-Z."
            )

        for idx, (x, y) in enumerate(points_xy):
            x = int(round(float(x)))
            y = int(round(float(y)))

            cv2.circle(vis, (x, y), 6, (0, 255, 255), -1)
            cv2.circle(vis, (x, y), 9, (0, 0, 0), 2)

            text = chr(ord("A") + idx)
            tx, ty = x-4, y+4

            cv2.putText(
                vis,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                4,
                cv2.LINE_AA
            )

            cv2.putText(
                vis,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imwrite(str(save_path), vis)

    def _save_posneg_points_visualization(
        self,
        image_bgr,
        mask,
        points_xy,
        positive_indices,
        save_path,
        part_box=None
    ):
        vis = image_bgr.copy()

        green = np.array([0, 180, 0], dtype=np.uint8)

        if np.any(mask):
            vis[mask] = cv2.addWeighted(
                vis[mask],
                0.65,
                np.full_like(vis[mask], green),
                0.35,
                0
            )

        pos_set = set(positive_indices)

        if part_box is not None:
            x1, y1, x2, y2 = [int(round(float(v))) for v in part_box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

        if len(points_xy) > 26:
            raise ValueError(
                f"The number of FPS sample points is {len(points_xy)}, exceeding the maximum of 26 representable by A-Z."
            )

        for idx, (x, y) in enumerate(points_xy):
            x = int(round(float(x)))
            y = int(round(float(y)))

            point_label = chr(ord("A") + idx)

            if idx in pos_set:
                color = (0, 0, 255)
                label_text = f"{point_label}+"
            else:
                color = (255, 0, 0)
                label_text = f"{point_label}-"

            cv2.circle(vis, (x, y), 6, color, -1)
            cv2.circle(vis, (x, y), 8, (255, 255, 255), 2)

            tx, ty = x + 4,  y - 4

            cv2.putText(
                vis,
                label_text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                vis,
                label_text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA
            )

        cv2.imwrite(str(save_path), vis)
