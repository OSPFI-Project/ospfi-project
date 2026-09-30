import torch
import numpy as np
import os
import time
import pytorch3d.ops as torch3d_ops
import matplotlib.pyplot as plt
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT))

# L2 distance threshold for normalized feature candidates.
FEATURE_DISTANCE_THRESHOLD = float(np.sqrt(2.0 - 2.0 * 0.4))

# Semantic distance weight in correspondence scoring.
LAMBDA_SEMANTIC = 0.5

MAX_CANDIDATES = 15
RANSAC_ITERATIONS = 1000
RANSAC_MIN_SAMPLES = 3
RANSAC_INLIER_THRESHOLD = 0.03

def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points)
    else:
        points = points.clone()
    
    device = points.device
    if use_cuda and torch.cuda.is_available():
        points = points.cuda()

    sampled_points, indices = torch3d_ops.sample_farthest_points(
        points=points.unsqueeze(0), K=K)
    
    sampled_points = sampled_points.squeeze(0).to(device)
    indices = indices.squeeze(0).to(device)
    
    return sampled_points, indices

def get_trans_batch(x: torch.Tensor, y: torch.Tensor, with_scale=True): 
    # Use a consistent dtype for batched SVD.
    B, n, _ = x.shape
    x = x.float()
    y = y.float()
    
    x_mean = x.mean(1, keepdim=True)
    y_mean = y.mean(1, keepdim=True)

    x1 = x - x_mean
    y1 = y - y_mean

    H = x1.transpose(1, 2) @ y1 / n
    U, S, Vt = torch.linalg.svd(H)
    
    det = torch.linalg.det(U @ Vt)
    M = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0).repeat(B, 1, 1)
    M[:, 2, 2] = det

    R = (U @ M @ Vt).transpose(1, 2)
    if with_scale:
        sx = (x1**2).sum((1, 2)) / n
        eps = torch.finfo(x.dtype).eps
        scale = (S * torch.diagonal(M, dim1=1, dim2=2)).sum(-1) / sx.clamp_min(eps)
    else:
        scale = torch.ones(B, device=x.device, dtype=x.dtype)

    T = y_mean.transpose(1, 2) - scale.reshape(-1, 1, 1) * R @ x_mean.transpose(1, 2)
    residual = (
        scale.reshape(-1, 1, 1) * R @ x.transpose(1, 2)
        + T
        - y.transpose(1, 2)
    )
    squared_geometry_error = residual.square().sum(dim=(1, 2))
    return R, T, scale, squared_geometry_error

class Matcher:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_template(self, template_path):
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found at {template_path}")

        print(f"Loading template from: {template_path}")
        
        with np.load(template_path, allow_pickle=True) as data:
            feat_key = 'descriptors' if 'descriptors' in data else 'features'
            pos_key = 'keypoints' if 'keypoints' in data else 'points'
            
            if feat_key not in data or pos_key not in data:
                raise ValueError(f"Invalid template format. Keys: {list(data.keys())}")

            template_features = torch.from_numpy(data[feat_key]).float()
            template_positions = data[pos_key] 

        return {
            'feature': template_features.to(self.device),
            'position': template_positions,
            'num': template_positions.shape[0],
        }

    def global_initialization(self, scene_cloud, scene_feats, template_data):
        """Estimate a similarity transform with feature-guided RANSAC.

        Returned target points preserve template order. Unmatched and rejected
        correspondences remain NaN.
        """
        target_feat = template_data['feature']
        target_pos_np = template_data['position']
        num_kps = template_data['num']
        num_scene_points = scene_cloud.shape[0]

        # Compute pairwise distances between normalized features.
        scene_feats_norm = torch.nn.functional.normalize(scene_feats, p=2, dim=1)
        target_feat_norm = torch.nn.functional.normalize(target_feat, p=2, dim=1)
        feature_distances = torch.cdist(
            scene_feats_norm,
            target_feat_norm,
            p=2,
        )

        # Build a bounded candidate set for each template point.
        valid_kps = []
        cands_for_valid_kps = []
        dists_for_valid_kps = []
        aligned_target_pts = np.full((num_kps, 3), np.nan)
        
        max_candidates = min(MAX_CANDIDATES, num_scene_points)
        
        for k in range(num_kps):
            distance_k = feature_distances[:, k]
            top_distances, top_indices = torch.topk(
                distance_k,
                k=max_candidates,
                largest=False,
            )
            
            # Discard candidates outside the feature-distance gate.
            mask = top_distances <= FEATURE_DISTANCE_THRESHOLD
            valid_indices = top_indices[mask]
            
            if len(valid_indices) > 0:
                valid_kps.append(k)
                cands_for_valid_kps.append(scene_cloud[valid_indices, :3])
                dists_for_valid_kps.append(top_distances[mask])

        M = len(valid_kps)
        print(f"[Global Init] Valid template points with candidates: {M} / {num_kps}")

        # A 3D similarity hypothesis needs at least three correspondences.
        if M < 3:
            print("[Global Init] Fewer than three correspondences; cannot estimate a transform.")
            T_global = torch.eye(4).to(self.device)
            R = np.eye(3)
            t = np.zeros(3)
            return T_global, 1.0, R, t, aligned_target_pts

        # Pad variable-length candidate sets for batched scoring.
        batch_size = RANSAC_ITERATIONS
        print(
            f"[Global Init] Evaluating {batch_size} RANSAC hypotheses, "
            f"inlier threshold={RANSAC_INLIER_THRESHOLD:.4f}m..."
        )

        sources_np = target_pos_np[valid_kps]
        source_points = torch.as_tensor(
            sources_np,
            dtype=torch.float32,
            device=self.device,
        )

        max_num_candidates = max(len(candidates) for candidates in cands_for_valid_kps)
        candidate_points = torch.zeros(
            (M, max_num_candidates, 3),
            dtype=torch.float32,
            device=self.device,
        )
        candidate_distances = torch.full(
            (M, max_num_candidates),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        candidate_valid = torch.zeros(
            (M, max_num_candidates),
            dtype=torch.bool,
            device=self.device,
        )

        for i in range(M):
            num_cands = len(cands_for_valid_kps[i])
            candidate_points[i, :num_cands] = cands_for_valid_kps[i]
            candidate_distances[i, :num_cands] = dists_for_valid_kps[i]
            candidate_valid[i, :num_cands] = True

        # Draw minimal three-point correspondence sets.
        random_order = torch.rand((batch_size, M), device=self.device)
        sample_template_indices = torch.topk(
            random_order,
            k=RANSAC_MIN_SAMPLES,
            dim=1,
            largest=False,
        ).indices
        sample_sources = source_points[sample_template_indices]
        sample_targets = torch.zeros_like(sample_sources)

        for sample_slot in range(RANSAC_MIN_SAMPLES):
            slot_template_indices = sample_template_indices[:, sample_slot]
            for template_idx in range(M):
                batch_rows = torch.where(slot_template_indices == template_idx)[0]
                if len(batch_rows) == 0:
                    continue
                num_cands = len(cands_for_valid_kps[template_idx])
                random_candidate_indices = torch.randint(
                    0,
                    num_cands,
                    (len(batch_rows),),
                    device=self.device,
                )
                sample_targets[batch_rows, sample_slot] = candidate_points[
                    template_idx,
                    random_candidate_indices,
                ]

        hypothesis_Rs, hypothesis_Ts, hypothesis_scales, _ = get_trans_batch(
            sample_sources,
            sample_targets,
            with_scale=True,
        )

        source_triangle_area2 = torch.linalg.norm(
            torch.cross(
                sample_sources[:, 1] - sample_sources[:, 0],
                sample_sources[:, 2] - sample_sources[:, 0],
                dim=1,
            ),
            dim=1,
        )
        target_triangle_area2 = torch.linalg.norm(
            torch.cross(
                sample_targets[:, 1] - sample_targets[:, 0],
                sample_targets[:, 2] - sample_targets[:, 0],
                dim=1,
            ),
            dim=1,
        )
        model_is_valid = (
            torch.isfinite(hypothesis_Rs).all(dim=(1, 2))
            & torch.isfinite(hypothesis_Ts).all(dim=(1, 2))
            & torch.isfinite(hypothesis_scales)
            & (hypothesis_scales > 0.0)
            & (source_triangle_area2 > 1e-8)
            & (target_triangle_area2 > 1e-8)
        )

        # Assign each template point to its lowest-cost candidate.
        expanded_sources = source_points.unsqueeze(0).expand(batch_size, -1, -1)
        predicted_points = (
            hypothesis_scales[:, None, None]
            * torch.bmm(expanded_sources, hypothesis_Rs.transpose(1, 2))
            + hypothesis_Ts.transpose(1, 2)
        )
        candidate_geometry_errors = (
            candidate_points.unsqueeze(0)
            - predicted_points[:, :, None, :]
        ).square().sum(dim=3)
        candidate_objectives = (
            candidate_geometry_errors
            + LAMBDA_SEMANTIC * candidate_distances.unsqueeze(0)
        ).masked_fill(~candidate_valid.unsqueeze(0), float("inf"))
        _, hypothesis_candidate_indices = candidate_objectives.min(dim=2)

        gather_indices = hypothesis_candidate_indices.unsqueeze(2)
        selected_geometry_errors = torch.gather(
            candidate_geometry_errors,
            dim=2,
            index=gather_indices,
        ).squeeze(2)
        selected_semantic_distances = torch.gather(
            candidate_distances.unsqueeze(0).expand(batch_size, -1, -1),
            dim=2,
            index=gather_indices,
        ).squeeze(2)
        hypothesis_inliers = (
            selected_geometry_errors <= RANSAC_INLIER_THRESHOLD ** 2
        ) & model_is_valid[:, None]
        hypothesis_inlier_counts = hypothesis_inliers.sum(dim=1)
        hypothesis_objectives = (
            selected_geometry_errors
            + LAMBDA_SEMANTIC * selected_semantic_distances
        ).masked_fill(~hypothesis_inliers, 0.0).sum(dim=1)
        hypothesis_objectives = hypothesis_objectives.masked_fill(
            hypothesis_inlier_counts < RANSAC_MIN_SAMPLES,
            float("inf"),
        )

        max_inlier_count = int(hypothesis_inlier_counts.max().item())
        if max_inlier_count < RANSAC_MIN_SAMPLES:
            print("[Global Init] RANSAC found no consensus with at least three inliers.")
            T_global = torch.eye(4, device=self.device)
            R = np.eye(3)
            t = np.zeros(3)
            return T_global, 1.0, R, t, aligned_target_pts

        # Rank hypotheses by inlier count, then by inlier cost.
        best_count_mask = hypothesis_inlier_counts == max_inlier_count
        ranked_objectives = hypothesis_objectives.masked_fill(
            ~best_count_mask,
            float("inf"),
        )
        best_idx = torch.argmin(ranked_objectives)

        selected_points = torch.gather(
            candidate_points.unsqueeze(0).expand(batch_size, -1, -1, -1),
            dim=2,
            index=hypothesis_candidate_indices[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)
        best_target_points = selected_points[best_idx].clone()
        best_semantic_distances = selected_semantic_distances[best_idx].clone()
        best_inlier_mask = hypothesis_inliers[best_idx].clone()

        def select_correspondences(R_model, T_model, scale_model):
            predicted = (
                scale_model * (source_points @ R_model.transpose(0, 1))
                + T_model.reshape(1, 3)
            )
            geometry_errors = (
                candidate_points - predicted[:, None, :]
            ).square().sum(dim=2)
            objectives = (
                geometry_errors + LAMBDA_SEMANTIC * candidate_distances
            ).masked_fill(~candidate_valid, float("inf"))
            _, candidate_indices = objectives.min(dim=1)
            rows = torch.arange(M, device=self.device)
            points = candidate_points[rows, candidate_indices]
            semantic_distances = candidate_distances[rows, candidate_indices]
            point_geometry_errors = geometry_errors[rows, candidate_indices]
            inlier_mask = point_geometry_errors <= RANSAC_INLIER_THRESHOLD ** 2
            return points, semantic_distances, point_geometry_errors, inlier_mask

        # Refit and update correspondences until stable.
        for _ in range(3):
            refined_Rs, refined_Ts, refined_scales, _ = get_trans_batch(
                source_points[best_inlier_mask].unsqueeze(0),
                best_target_points[best_inlier_mask].unsqueeze(0),
                with_scale=True,
            )
            refined_R = refined_Rs[0]
            refined_T = refined_Ts[0].squeeze(1)
            refined_scale = refined_scales[0]
            (
                updated_target_points,
                updated_semantic_distances,
                _,
                updated_inlier_mask,
            ) = select_correspondences(refined_R, refined_T, refined_scale)
            if int(updated_inlier_mask.sum().item()) < RANSAC_MIN_SAMPLES:
                break
            is_stable = (
                torch.equal(updated_inlier_mask, best_inlier_mask)
                and torch.allclose(updated_target_points, best_target_points)
            )
            best_target_points = updated_target_points
            best_semantic_distances = updated_semantic_distances
            best_inlier_mask = updated_inlier_mask
            if is_stable:
                break

        # Fit the final similarity transform from the consensus set.
        final_Rs, final_Ts, final_scales, _ = get_trans_batch(
            source_points[best_inlier_mask].unsqueeze(0),
            best_target_points[best_inlier_mask].unsqueeze(0),
            with_scale=True,
        )
        final_R = final_Rs[0]
        final_T = final_Ts[0].squeeze(1)
        final_scale = final_scales[0]

        # Recheck the consensus after refinement.
        final_predicted_points = (
            final_scale * (source_points @ final_R.transpose(0, 1))
            + final_T.reshape(1, 3)
        )
        final_geometry_errors = (
            best_target_points - final_predicted_points
        ).square().sum(dim=1)
        consistent_inlier_mask = (
            best_inlier_mask
            & (final_geometry_errors <= RANSAC_INLIER_THRESHOLD ** 2)
        )
        if int(consistent_inlier_mask.sum().item()) >= RANSAC_MIN_SAMPLES:
            best_inlier_mask = consistent_inlier_mask
            final_Rs, final_Ts, final_scales, _ = get_trans_batch(
                source_points[best_inlier_mask].unsqueeze(0),
                best_target_points[best_inlier_mask].unsqueeze(0),
                with_scale=True,
            )
            final_R = final_Rs[0]
            final_T = final_Ts[0].squeeze(1)
            final_scale = final_scales[0]

        final_predicted_points = (
            final_scale * (source_points @ final_R.transpose(0, 1))
            + final_T.reshape(1, 3)
        )
        final_geometry_errors = (
            best_target_points - final_predicted_points
        ).square().sum(dim=1)
        final_geometry_sum = final_geometry_errors[best_inlier_mask].sum()
        final_semantic_sum = best_semantic_distances[best_inlier_mask].sum()
        final_objective = final_geometry_sum + LAMBDA_SEMANTIC * final_semantic_sum
        final_inlier_count = int(best_inlier_mask.sum().item())
        final_geometry_rmse = torch.sqrt(final_geometry_sum / final_inlier_count)

        R = final_R.cpu().numpy()
        T = final_T.cpu().numpy()
        scale = final_scale.cpu().numpy()

        print(
            "[Global Init] RANSAC result: "
            f"inliers={final_inlier_count}/{M}, "
            f"geometry_sum_sq={final_geometry_sum.item():.6f}, "
            f"geometry_rmse={final_geometry_rmse.item():.6f}m, "
            f"semantic_sum={final_semantic_sum.item():.6f}, "
            f"lambda={LAMBDA_SEMANTIC:.6f}, "
            f"objective={final_objective.item():.6f}, "
            f"scale={float(scale):.6f}"
        )

        inlier_template_indices = [
            valid_kps[i]
            for i in range(M)
            if bool(best_inlier_mask[i].item())
        ]
        outlier_template_indices = [
            valid_kps[i]
            for i in range(M)
            if not bool(best_inlier_mask[i].item())
        ]
        print(f"[Global Init] RANSAC inlier template indices: {inlier_template_indices}")
        print(f"[Global Init] RANSAC outlier template indices: {outlier_template_indices}")

        # Preserve template order and leave rejected entries as NaN.
        for i, k in enumerate(valid_kps):
            if bool(best_inlier_mask[i].item()):
                aligned_target_pts[k] = best_target_points[i].cpu().numpy()

        # Build the homogeneous similarity transform.
        M_mat = np.eye(4)
        M_mat[:3, :3] = R * scale
        M_mat[:3, 3] = T
        
        T_global = torch.from_numpy(M_mat).float().to(self.device)
        
        return T_global, float(scale), R, T, aligned_target_pts

    def match(self, scene_cloud, scene_feats, template_data):
        """Match template points to a scene."""
        start_time = time.time()
        
        if not isinstance(scene_feats, torch.Tensor):
            scene_feats = torch.from_numpy(scene_feats).float()
        if not isinstance(scene_cloud, torch.Tensor):
            scene_cloud = torch.from_numpy(scene_cloud).float()
            
        scene_feats = scene_feats.to(self.device)
        scene_cloud = scene_cloud.to(self.device)

        print(">>> Starting Global Initialization (Feature-Guided RANSAC)...")
        T_global, s_g, R_g, t_g, best_target_pts = self.global_initialization(scene_cloud, scene_feats, template_data)
        
        print(f"Total time: {time.time()-start_time:.4f}s")
        # Count retained correspondences.
        valid_pt_count = np.count_nonzero(~np.isnan(best_target_pts[:, 0]))
        print(f"Matched target points: {valid_pt_count} (in template order)")
        
        T_global_np = T_global.cpu().numpy()
        final_transforms_numpy = {
            'global': T_global_np,
            's': s_g,
            'R': R_g,
            't': t_g,
        }
        return final_transforms_numpy, best_target_pts
