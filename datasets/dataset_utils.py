# Warsaw University of Technology

import numpy as np
from typing import List, Tuple, Dict
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KDTree  # (kept if you plan to use later)

from datasets.base_datasets import EvaluationTuple, TrainingDataset
from datasets.samplers import BatchSampler
from misc.utils import TrainingParams

import open3d as o3d
from scipy.spatial.transform import Rotation as R
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import torchvision.transforms as T


def make_datasets(params: TrainingParams, validation: bool = False) -> Dict[str, TrainingDataset]:
    """Create training/validation datasets."""
    datasets = {}
    datasets['train'] = TrainingDataset(
        params.dataset_folder,
        params.train_file,
        transform=None,
        set_transform=None
    )
    if validation:
        datasets['val'] = TrainingDataset(params.dataset_folder, params.val_file)
    return datasets


def make_dataloaders(params: TrainingParams, validation: bool = False) -> Dict[str, DataLoader]:
    """
    Create dataloaders with BatchSampler and custom collate function.
    Returns a dict with 'train' and optional 'val'.
    """
    datasets = make_datasets(params, validation=validation)
    dataloaders = {}

    train_sampler = BatchSampler(
        datasets['train'],
        batch_size=params.batch_size,
        batch_size_limit=params.batch_size_limit,
        batch_expansion_rate=params.batch_expansion_rate
    )
    train_collate_fn = make_collate_fn(datasets['train'], params)
    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_sampler=train_sampler,
        collate_fn=train_collate_fn,
        num_workers=params.num_workers,
        pin_memory=True
    )

    if validation and 'val' in datasets:
        val_sampler = BatchSampler(datasets['val'], batch_size=params.val_batch_size)
        val_collate_fn = make_collate_fn(datasets['val'], params)
        dataloaders['val'] = DataLoader(
            datasets['val'],
            batch_sampler=val_sampler,
            collate_fn=val_collate_fn,
            num_workers=params.num_workers,
            pin_memory=True
        )

    return dataloaders


def make_collate_fn(dataset: TrainingDataset, params: TrainingParams):
    """Factory: returns a collate_fn bound to given dataset/params."""

    # --------- Small geometry helpers ---------

    def compute_transformation_matrix(pose: np.ndarray) -> np.ndarray:
        """Pose [x,y,z,qx,qy,qz,qw] -> 4x4 matrix."""
        x, y, z, qx, qy, qz, qw = pose
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = np.array([x, y, z], dtype=np.float64)
        return T

    def backproject_range_image(range_image: np.ndarray,
                                azimuth_angles: np.ndarray,
                                elevation_angles: np.ndarray) -> np.ndarray:
        """
        Range image (H,W), azimuth (W,), elevation (H,) -> Nx3 points.
        """
        H, W = range_image.shape
        az_grid, el_grid = np.meshgrid(azimuth_angles, elevation_angles)
        cos_el = np.cos(el_grid)
        x = range_image * cos_el * np.cos(az_grid)
        y = range_image * cos_el * np.sin(az_grid)
        z = range_image * np.sin(el_grid)
        return np.stack([x, y, z], axis=-1).reshape(-1, 3)

    def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Apply 4x4 transform to Nx3 points."""
        pts_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
        out = pts_h @ T.T
        return out[:, :3]

    def project_points_to_range_image(points: np.ndarray,
                                      azimuth_angles: np.ndarray,
                                      elevation_angles: np.ndarray,
                                      image_width: int,
                                      image_height: int) -> np.ndarray:
        """
        Project Nx3 points to a range image (H,W). Uses max pooling in range bin.
        """
        H, W = image_height, image_width
        ranges = np.linalg.norm(points, axis=1)

        az = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2 * np.pi)
        el = np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1))

        # Bin indices (clamped)
        col = np.clip(np.digitize(az, azimuth_angles) - 1, 0, W - 1)
        row = np.clip(np.digitize(el, elevation_angles) - 1, 0, H - 1)

        img = np.zeros((H, W), dtype=np.float32)
        # Max pooling per (row, col)
        # Vectorized scatter-max:
        flat_idx = row * W + col
        # For repeated indices, keep max
        order = np.argsort(flat_idx)
        flat_idx_sorted = flat_idx[order]
        ranges_sorted = ranges[order]
        # Find segment boundaries and write maxima
        unique_idx, first = np.unique(flat_idx_sorted, return_index=True)
        maxima = np.maximum.reduceat(ranges_sorted, first)
        img.flat[unique_idx] = maxima.astype(np.float32)
        return img

    def project_image2_to_image1(points2: np.ndarray,
                                 points2_transformed: np.ndarray,
                                 azimuth_angles: np.ndarray,
                                 elevation_angles: np.ndarray,
                                 patch_size: int = 14,
                                 image_width: int = 1078,
                                 image_height: int = 126) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """
        Map patches between two range images by projecting original and transformed point sets
        onto the same angular bins, then majority-voting patch correspondences.
        Returns dict: (pr1, pc1) -> (pr2, pc2) in patch-grid coordinates (top-left pixel).
        """
        # Transformed
        az_t = np.mod(np.arctan2(points2_transformed[:, 1], points2_transformed[:, 0]), 2 * np.pi)
        el_t = np.arctan2(points2_transformed[:, 2], np.linalg.norm(points2_transformed[:, :2], axis=1))
        col1 = np.clip(np.digitize(az_t, azimuth_angles) - 1, 0, image_width - 1)
        row1 = np.clip(np.digitize(el_t, elevation_angles) - 1, 0, image_height - 1)

        # Original
        az_2 = np.mod(np.arctan2(points2[:, 1], points2[:, 0]), 2 * np.pi)
        el_2 = np.arctan2(points2[:, 2], np.linalg.norm(points2[:, :2], axis=1))
        col2 = np.clip(np.digitize(az_2, azimuth_angles) - 1, 0, image_width - 1)
        row2 = np.clip(np.digitize(el_2, elevation_angles) - 1, 0, image_height - 1)

        # Patch top-lefts (quantize to patch_size)
        pr1 = (row1 // patch_size) * patch_size
        pc1 = (col1 // patch_size) * patch_size
        pr2 = (row2 // patch_size) * patch_size
        pc2 = (col2 // patch_size) * patch_size

        # Keep valid inside bounds
        valid = (
            (pr1 >= 0) & (pr1 < image_height) &
            (pc1 >= 0) & (pc1 < image_width) &
            (pr2 >= 0) & (pr2 < image_height) &
            (pc2 >= 0) & (pc2 < image_width)
        )
        pr1, pc1, pr2, pc2 = pr1[valid], pc1[valid], pr2[valid], pc2[valid]

        # Majority vote per patch
        candidates = defaultdict(list)
        for a, b, c, d in zip(pr1, pc1, pr2, pc2):
            candidates[(a, b)].append((c, d))

        return {k: Counter(v).most_common(1)[0][0] for k, v in candidates.items()}

    def get_positive_pairs(image1: np.ndarray,
                           transformed_image2: np.ndarray,
                           reprojected_indices: Dict[Tuple[int, int], Tuple[int, int]],
                           patch_size: int = 14,
                           range_threshold: float = 1.0,
                           min_valid_ratio: float = 0.5) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Build positive pairs between patch grids by robustly comparing range values
        in overlapping valid pixels. Returns list of ((r1,c1),(r2,c2)) in patch-grid coords.
        """
        H, W = image1.shape
        positives = []

        for r1 in range(0, H, patch_size):
            for c1 in range(0, W, patch_size):
                key = (r1, c1)
                if key not in reprojected_indices:
                    continue
                r2, c2 = reprojected_indices[key]

                if r1 + patch_size > H or c1 + patch_size > W:
                    continue
                if r2 + patch_size > transformed_image2.shape[0] or c2 + patch_size > transformed_image2.shape[1]:
                    continue

                p1 = image1[r1:r1 + patch_size, c1:c1 + patch_size]
                p2 = transformed_image2[r2:r2 + patch_size, c2:c2 + patch_size]

                valid = (p1 > 0) & (p2 > 0)
                valid_ratio = valid.mean()
                if valid_ratio < min_valid_ratio:
                    continue

                diff = np.abs(p1[valid] - p2[valid])
                if diff.size == 0:
                    continue

                # Robust threshold via median + 1.4826 * MAD
                med = np.median(diff)
                mad = np.median(np.abs(diff - med))
                thr = med + 1.4826 * mad
                inliers = diff <= thr
                robust_mean = diff[inliers].mean() if inliers.any() else np.inf

                if robust_mean < range_threshold:
                    # Return patch indices in "patch grid" (divide by patch_size)
                    positives.append(((r1 // patch_size, c1 // patch_size),
                                      (r2 // patch_size, c2 // patch_size)))
        return positives

    def extract_unique_positive_pairs(matrix: np.ndarray) -> np.ndarray:
        """
        Get unique (i,j) with True in upper triangle (i < j) to avoid duplicates.
        matrix is boolean (N,N).
        """
        i_idx, j_idx = np.triu_indices(matrix.shape[0], k=1)
        mask = matrix[i_idx, j_idx]
        return np.column_stack((i_idx[mask], j_idx[mask]))

    def sample_positive_pairs(matrix: np.ndarray,
                              poses: List[np.ndarray],
                              sample_size: int) -> np.ndarray:
        """
        Choose up to sample_size disjoint positive pairs (no repeated row/col),
        preferring nearer pose pairs.
        """
        pairs = extract_unique_positive_pairs(matrix)
        if pairs.shape[0] == 0:
            return pairs

        # Compute Euclidean distances between positions
        dists = []
        for r, c in pairs:
            p1 = np.asarray(poses[r])[:3]
            p2 = np.asarray(poses[c])[:3]
            dists.append(((r, c), float(np.linalg.norm(p1 - p2))))
        dists.sort(key=lambda x: x[1])

        used_r, used_c = set(), set()
        chosen = []
        for (r, c), _ in dists:
            if r in used_r or c in used_c:
                continue
            chosen.append([r, c])
            used_r.add(r)
            used_c.add(c)
            if len(chosen) == sample_size:
                break
        return np.asarray(chosen, dtype=np.int64)

    def in_sorted_array(e: int, array: np.ndarray) -> bool:
        """Binary search in a sorted array; returns True if e is present."""
        pos = np.searchsorted(array, e)
        return (pos < array.size) and (array[pos] == e)

    # --------- The actual collate_fn ---------

    def collate_fn(data_list):
        """
        Batch builder. Returns:
          batch (split into chunks of params.batch_split_size),
          positives_mask (N,N),
          negatives_mask (N,N),
          sampled_pairs (K,2),
          positive_pairs (list of lists of ((r1,c1),(r2,c2)) ).
        """
        clouds_list = [e[0] for e in data_list]        # Each item is a tensor-like array
        clouds_np = np.asarray(clouds_list)
        clouds = torch.from_numpy(clouds_np)

        labels = [e[1] for e in data_list]
        poses = [e[2] for e in data_list]

        # Build positives/negatives masks (NxN)
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask, dtype=torch.bool)
        negatives_mask = torch.tensor(negatives_mask, dtype=torch.bool)

        # Sample disjoint positive pairs among batch elements
        positive_pairs = []
        sampled_pairs = sample_positive_pairs(positives_mask.numpy(), poses, params.num_positive_pairs)

        # Angle grids (ensure azimuth size=W and elevation size=H)
        # NOTE: image_H and image_W names come from your params; here we align:
        #   azimuth_angles -> width bins; elevation_angles -> height bins
        azimuth_angles = np.linspace(0, 2 * np.pi, params.image_H, endpoint=False)
        elevation_angles = np.linspace(params.uFoV * np.pi / 180.0,
                                       params.dFoV * np.pi / 180.0,
                                       params.image_W, endpoint=False)


        for i, j in sampled_pairs:
            # Channel-2 (index 1): range in [0,255], crop columns [28:-28], rescale to meters
            query_range = (clouds_np[i][1][:, 28:-28]) * params.sensor_range
            positive_range = (clouds_np[j][1][:, 28:-28]) * params.sensor_range

            # Back-project both to 3D
            pts1 = backproject_range_image(query_range, azimuth_angles, elevation_angles)
            pts2 = backproject_range_image(positive_range, azimuth_angles, elevation_angles)

            # Relative transform T2->T1 from GT poses
            T1 = compute_transformation_matrix(np.asarray(poses[i], dtype=np.float64))
            T2 = compute_transformation_matrix(np.asarray(poses[j], dtype=np.float64))
            T2_to_T1 = np.linalg.inv(T1) @ T2

            # Downsample and refine with ICP
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(pts1)
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(pts2)


            pcd1 = pcd1.voxel_down_sample(voxel_size=0.4)
            pcd2 = pcd2.voxel_down_sample(voxel_size=0.4)
            pcd2.transform(T2_to_T1)
            
            icp_threshold = 0.8
            icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
            result_icp = o3d.pipelines.registration.registration_icp(
                pcd2, pcd1, icp_threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                icp_criteria
            )
            # Free memory early
            del pcd1, pcd2

            refined = result_icp.transformation @ T2_to_T1
            pts2_refined = transform_points(pts2, refined)

            # Patch-level reprojection map
            Hq, Wq = query_range.shape
            repro = project_image2_to_image1(
                pts2, pts2_refined, azimuth_angles, elevation_angles,
                patch_size=14, image_width=Wq, image_height=Hq
            )

            # Re-render refined points to range image, then pair patches
            img2_refined = project_points_to_range_image(pts2_refined, azimuth_angles, elevation_angles, Wq, Hq)
            pos_pairs_ij = get_positive_pairs(
                query_range, img2_refined, repro,
                patch_size=14, range_threshold=0.8, min_valid_ratio=0.5
            )
            positive_pairs.append(pos_pairs_ij)

        # Split batch into chunks for multi-stage training (if enabled)
        batch = []
        bss = params.batch_split_size
        for k in range(0, len(clouds), bss):
            batch.append(clouds[k:k + bss])

        return batch, positives_mask, negatives_mask, sampled_pairs, positive_pairs

    return collate_fn
