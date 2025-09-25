import os
from typing import List, Dict, Tuple
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
from scipy.spatial.transform import Rotation as R


# =========================
# Augmentation utilities
# =========================

def random_yaw_roll(query_hw3: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """
    Roll horizontally by a random number of columns (cylindrical yaw).
    Returns (rolled_query, shift_cols, yaw_angle_rad).
    """
    W = query_hw3.shape[1]
    shift = np.random.randint(0, W)
    yaw_angle = (shift / W) * 2 * np.pi
    return np.roll(query_hw3, shift, axis=1), shift, yaw_angle


def pad_cylindrical_28(query_hw3_rolled: np.ndarray) -> np.ndarray:
    """
    Add 28 px cylindrical padding on both sides (right+query+left).
    """
    left = query_hw3_rolled[:, :28]
    right = query_hw3_rolled[:, -28:]
    return np.concatenate((right, query_hw3_rolled, left), axis=1)


def normalize_to_unit(query_chw: np.ndarray) -> np.ndarray:
    """
    Divide each channel by 255.0 (expects CHW).
    """
    query_chw = query_chw.astype(np.float32, copy=False)
    query_chw[0] /= 255.0
    query_chw[1] /= 255.0
    query_chw[2] /= 255.0
    return query_chw


def random_block_mask(height: int,
                      width: int,
                      mask_ratio: float = 0.0,
                      patch_size_min: int = 1,
                      patch_size_max: int = 14) -> np.ndarray:
    """
    Randomly zero out multiple square/rect blocks.
    """
    mask = np.ones((height, width), dtype=np.float32)
    if mask_ratio <= 0:
        return mask

    # Random patch size per call (keeps original behavior)
    patch_size = np.random.randint(patch_size_min, patch_size_max + 1)

    total_area = height * width
    patch_area = patch_size * patch_size
    total_patches_possible = max(1, total_area // patch_area)
    num_masked = int(total_patches_possible * mask_ratio)

    max_h = height - patch_size
    max_w = width - patch_size
    if max_h < 0 or max_w < 0 or num_masked <= 0:
        return mask

    h_starts = np.random.randint(0, max_h + 1, size=num_masked)
    w_starts = np.random.randint(0, max_w + 1, size=num_masked)

    for hs, ws in zip(h_starts, w_starts):
        mask[hs:hs + patch_size, ws:ws + patch_size] = 0.0
    return mask


def cylindrical_band_mask(height: int,
                          width: int,
                          max_width_ratio: float = 0.3) -> np.ndarray:
    """
    Zero out a continuous vertical band with cylindrical wrap.
    """
    mask = np.ones((height, width), dtype=np.float32)
    if max_width_ratio <= 0:
        return mask

    max_mask_width = max(1, int(width * max_width_ratio))
    band_w = np.random.randint(1, max_mask_width + 1)
    start = np.random.randint(0, width)

    if start + band_w <= width:
        mask[:, start:start + band_w] = 0.0
    else:
        first = width - start
        mask[:, start:] = 0.0
        mask[:, :band_w - first] = 0.0
    return mask


def random_line_masks(height: int,
                      width: int,
                      mask_ratio_line: float = 0.1,
                      max_v: int = 14,
                      max_h: int = 140) -> np.ndarray:
    """
    Apply several random rectangular "line" masks.
    Number of lines is proportional to mask_ratio_line.
    """
    mask = np.ones((height, width), dtype=np.float32)
    if mask_ratio_line <= 0:
        return mask

    total_pixels = height * width
    max_lines = int((total_pixels * mask_ratio_line) / max(1, width))
    if max_lines <= 0:
        return mask

    num_lines = np.random.randint(0, max_lines + 1)
    for _ in range(num_lines):
        sv = np.random.randint(0, height)
        sh = np.random.randint(0, width)
        lh = np.random.randint(1, max_v + 1)
        lw = np.random.randint(1, max_h + 1)
        ev = min(sv + lh, height)
        eh = min(sh + lw, width)
        mask[sv:ev, sh:eh] = 0.0
    return mask


def apply_masks_sequential(query_chw: np.ndarray,
                           block_ratio: float,
                           band_ratio: float,
                           line_ratio: float) -> np.ndarray:
    """
    Build and apply three masks sequentially (elementwise multiplication).
    """
    H, W = query_chw.shape[1], query_chw.shape[2]

    # 1) Random blocks
    m_block = random_block_mask(H, W, mask_ratio=block_ratio)

    # 2) Cylindrical band (band_ratio used as max width ratio)
    m_band = cylindrical_band_mask(H, W, max_width_ratio=band_ratio)

    # 3) Random lines
    m_line = random_line_masks(H, W, mask_ratio_line=line_ratio)

    # Combine
    final_mask = m_block * m_band * m_line
    return query_chw * final_mask  # broadcast over channels


# =========================
# Dataset / Tuples
# =========================

class TrainingTuple:
    def __init__(self, id: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        self.id = id
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position


class EvaluationTuple:
    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.ndarray):
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position


class TrainingDataset(Dataset):
    def __init__(self, dataset_path, query_filename, transform=None, set_transform=None):
        assert os.path.exists(dataset_path), f'Cannot access dataset path: {dataset_path}'
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(self.dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), f'Cannot access query file: {self.query_filepath}'
        with open(self.query_filepath, 'rb') as f:
            data_dict = pickle.load(f)

        self.queries: Dict[int, TrainingTuple] = {}
        for key in data_dict.keys():
            d = data_dict[key]
            self.queries[d['id']] = TrainingTuple(
                d['id'],
                d['rel_scan_filepath'],
                np.array(d['positives']),
                np.array(d['non_negatives']),
                np.array(d['position'])
            )

        print(f'{len(self.queries)} queries in the dataset')
        self.transform = transform
        self.set_transform = set_transform

    def __len__(self):
        return len(self.queries)

    @staticmethod
    def apply_yaw_to_pose(pose, yaw_angle: float):
        """
        Apply yaw rotation (radians) to pose [x,y,z,qx,qy,qz,qw].
        """
        R_pose = R.from_quat(pose[3:]).as_matrix()
        R_yaw = R.from_euler('z', yaw_angle).as_matrix()
        R_new = R_yaw @ R_pose
        q_new = R.from_matrix(R_new).as_quat()
        return [pose[0], pose[1], pose[2], *q_new]

    def __getitem__(self, index):
        # Load (H,W,3)
        file_pathname = os.path.join(self.dataset_path, self.queries[index].rel_scan_filepath)
        query = np.load(file_pathname)  # H x W x 3

        # 1) Random cylindrical yaw roll
        query, _, yaw_angle1 = random_yaw_roll(query)

        # 2) Add 28-px cylindrical padding (right + query + left)
        query = pad_cylindrical_28(query)

        # 3) HWC -> CHW
        query = query.transpose(2, 0, 1)  # (3,H,W)

        # 4) Normalize channels to [0,1]
        query = normalize_to_unit(query)

        # 5) Apply three masking stages (same as original behavior)
        #    - block_ratio ~ U[0, 0.4]
        #    - band_ratio is used as max width ratio (<= 0.3)
        #    - line_ratio ~ U[0, 0.1]
        block_ratio = float(np.random.uniform(0.0, 0.4))
        band_ratio = 0.3  # max 30% width; original code randomizes width inside function
        line_ratio = float(np.random.uniform(0.0, 0.1))
        query = apply_masks_sequential(query, block_ratio, band_ratio, line_ratio)

        # Adjust pose to undo the yaw applied to the range image
        query_range_pose = self.apply_yaw_to_pose(self.queries[index].position, (-1.0) * yaw_angle1)

        return query, index, query_range_pose

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives


class EvaluationSet:
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        query_l = [e.to_tuple() for e in self.query_set]
        map_l = [e.to_tuple() for e in self.map_set]
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))
        self.query_set = [EvaluationTuple(*e) for e in query_l]
        self.map_set = [EvaluationTuple(*e) for e in map_l]

    def get_map_positions(self):
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions
