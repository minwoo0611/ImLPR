#!/usr/bin/env python3
import struct
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import open3d as o3d  # optional; kept if you plan to add point cloud I/O later
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ----------------------------
# Utilities
# ----------------------------

def scale_log_to_255(values):
    """
    Log-scale an array to [0, 255].
    This preserves the original code's mapping logic (using 0.7..13 range).
    """
    values = np.asarray(values, dtype=np.float32)
    log_values = np.log1p(values)  # log(1 + x), safe for x=0
    scaled = 255.0 * (log_values - 0.7) / (13.0 - 0.7)
    scaled = np.clip(scaled, 0.0, 255.0)
    return scaled.astype(np.float32)


def compute_normals(points, k=4, max_neighbor_m=50.0):
    """
    Estimate a per-point 'normal intensity' via SVD shape ratio on k-NN:
      - For each point, gather k nearest neighbors (excluding itself).
      - If the farthest neighbor distance > max_neighbor_m, use a low default (0.7).
      - Otherwise, compute the singular values of centered neighbors and use (max/min) as intensity.
      - Finally, log-scale the intensities to [0, 255].

    Returns:
        (N,) float32 in [0, 255]
    """
    points = np.asarray(points, dtype=np.float32)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    out = []
    for i in range(points.shape[0]):
        neighbor_distances = distances[i][1:]          # exclude self
        neighbors = points[indices[i][1:]].copy()

        if np.max(neighbor_distances) > max_neighbor_m:
            intensity = 0.7
        else:
            neighbors -= np.mean(neighbors, axis=0)
            svals = np.linalg.svd(neighbors, full_matrices=False, compute_uv=False)
            largest = float(np.max(svals))
            smallest = float(np.min(svals) + 1e-5)
            intensity = largest / smallest

        out.append(intensity)

    return scale_log_to_255(out)  # (N,) float32 0..255


def compute_v_indices(points, ring, image_height_initial,
                      mode="ring", fov_up_deg=16.0, fov_down_deg=16.0, invert_vertical=False):
    """
    Compute vertical index v either from:
      - ring indices (mode='ring'), or
      - elevation angle (mode='angle') with provided FOV (up/down in degrees).

    Args:
        points: (N,3) xyz
        ring: (N,) ring index per point (used if mode='ring')
        image_height_initial: vertical size before any optional resizing
        mode: 'ring' or 'angle'
        fov_up_deg, fov_down_deg: only used in 'angle' mode
        invert_vertical: if True, flip vertically (equivalent to 180° rotation along v)
    """
    H = int(image_height_initial)

    if mode == "ring":
        v = np.clip(ring, 0, H - 1).astype(np.int32)

    elif mode == "angle":
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        # Elevation angle in degrees, positive = upwards
        elev_deg = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
        # Clip to FOV range, e.g., +16° (up) to -16° (down)
        elev_deg = np.clip(elev_deg, -float(fov_down_deg), float(fov_up_deg))
        total_fov = float(fov_up_deg) + float(fov_down_deg) + 1e-6  # avoid div by zero
        # Map: +fov_up -> v=0 (top), -fov_down -> v=H-1 (bottom)
        v = ((float(fov_up_deg) - elev_deg) / total_fov) * (H - 1)
        v = np.clip(v, 0, H - 1).astype(np.int32)

    else:
        raise ValueError("mode must be either 'ring' or 'angle'")

    if invert_vertical:
        v = (H - 1) - v

    return v


# ----------------------------
# Image renderers
# ----------------------------

def pointCloudToImage(points, values, ring, *,
                      image_width=1022, image_height_initial=128, image_height_final=126,
                      v_mode="ring", fov_up_deg=16.0, fov_down_deg=16.0, invert_vertical=False):
    """
    Render a panoramic image using arbitrary per-point 'values' (e.g., reflectivity or normal-intensity).
    - Horizontal axis (u): azimuth (0..2π) binned to [0, W-1]
    - Vertical axis (v): from ring or elevation angle mapping to [0, H-1]
    - Pixel collision resolved by max pooling.

    Returns:
        (image_height_final, image_width) float32 array (0..255 if values are in that range)
    """
    H0, W = int(image_height_initial), int(image_width)
    image = np.zeros((H0, W), dtype=np.float32)

    # Compute horizontal bins from azimuth
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)
    u = (azimuth / (2 * np.pi) * (W - 1)).astype(np.int32)

    # Compute vertical bins by selected mode
    v = compute_v_indices(points, ring, H0,
                          mode=v_mode, fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg,
                          invert_vertical=invert_vertical)

    # Max-pool values into pixels
    for idx in range(points.shape[0]):
        ui, vi = u[idx], v[idx]
        if 0 <= ui < W and 0 <= vi < H0:
            image[vi, ui] = max(image[vi, ui], float(values[idx]))

    # Optional vertical resize
    if image_height_final is not None and image_height_final != H0:
        image = cv2.resize(image, (W, int(image_height_final)), interpolation=cv2.INTER_AREA)

    return image


def pointCloudToRangeImage(points, ring, *,
                           image_width=1022, image_height_initial=128, image_height_final=None,
                           v_mode="ring", fov_up_deg=16.0, fov_down_deg=16.0, invert_vertical=False,
                           lidar_type="helipr", max_range_m=120.0):
    """
    Render a panoramic range image.
    - Horizontal axis (u): azimuth bins
    - Vertical axis (v): from ring or elevation
    - Pixel collision resolved by min pooling (closer range overwrites farther)

    If lidar_type == 'mulran':
        We linearly scale range [0, max_range_m] to [0, 255] at SAVE TIME.
        => Later, you only need to divide by /255.0 to get normalized range in [0,1].

    Returns:
        (image_height_final or image_height_initial, image_width) float32 array
        For MulRan: 0..255 range (float), with empty pixels = 0.
    """
    H0, W = int(image_height_initial), int(image_width)
    # Initialize with a large value so min-pooling works
    image = np.ones((H0, W), dtype=np.float32) * 255.0

    # Horizontal bins from azimuth
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)
    u = (azimuth / (2 * np.pi) * (W - 1)).astype(np.int32)

    # Vertical bins
    v = compute_v_indices(points, ring, H0,
                          mode=v_mode, fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg,
                          invert_vertical=invert_vertical)

    # Euclidean range
    range_point = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)


    # Scale to [0,255] using max_range_m
    range_scaled = np.clip(range_point, 0.0, float(max_range_m)) / float(max_range_m) * 255.0
    channel_vals = range_scaled.astype(np.float32)

    # Min-pool: keep the closest range per pixel
    for idx in range(points.shape[0]):
        ui, vi = u[idx], v[idx]
        if 0 <= ui < W and 0 <= vi < H0:
            image[vi, ui] = min(image[vi, ui], float(channel_vals[idx]))

    # Convert untouched pixels (still at init 255) to 0 for a clean background
    image = np.where(image == 255.0, 0.0, image)

    # Optional vertical resize
    if image_height_final is not None and image_height_final != H0:
        image = cv2.resize(image, (W, int(image_height_final)), interpolation=cv2.INTER_AREA)

    return image


# ----------------------------
# I/O helpers
# ----------------------------

def read_bin_file(filename, typeLiDAR):
    """
    Read a custom .bin LiDAR file.
    Supported:
      - 'helipr': 26 bytes per point:
            [x(float32), y(float32), z(float32), intensity(float32),
             t(uint32), reflectivity(uint16), ring(uint16), ambient(uint16)]
      - 'mulran': 16 bytes per point:
            [x(float32), y(float32), z(float32), reflectivity(float32)]
        ring is synthesized as (idx % 64), reflectivity scaled to 0..255.

    Returns:
        points: (N,3) float32
        reflectivity_values: (N,) float32 (0..255 for mulran)
        ring_values: (N,) int32
        ambient_values: (N,) float32 (empty for mulran)
    """
    points = []
    reflectivity_values = []
    ring_values = []
    ambient_values = []

    with open(filename, "rb") as file:
        idx = 0
        while True:
            if typeLiDAR == "helipr":
                data = file.read(26)
                if len(data) < 26:
                    break
                x, y, z, intensity = struct.unpack('ffff', data[:16])
                _t = struct.unpack('I', data[16:20])[0]
                reflectivity, ring, ambient = struct.unpack('HHH', data[20:26])

                points.append([x, y, z])
                reflectivity_values.append(float(reflectivity))  # keep raw or post-scale later as needed
                ring_values.append(int(ring))
                ambient_values.append(float(ambient))

            elif typeLiDAR == "mulran":
                data = file.read(16)
                if len(data) < 16:
                    break
                x, y, z = struct.unpack('fff', data[:12])
                refl = struct.unpack('f', data[12:16])[0]
                ring = idx % 64
                idx += 1

                # Clamp and scale MulRan reflectivity into 0..255
                if refl > 4096:
                    refl = 4096
                refl = refl / 4096.0 * 255.0

                points.append([x, y, z])
                reflectivity_values.append(float(refl))
                ring_values.append(int(ring))

            else:
                raise ValueError("Unsupported LiDAR type")

    return (np.asarray(points, dtype=np.float32),
            np.asarray(reflectivity_values, dtype=np.float32),
            np.asarray(ring_values, dtype=np.int32),
            np.asarray(ambient_values, dtype=np.float32))


def parse_gt_file(gt_path, kind):
    """
    Parse pose ground truth files.
      - kind='tum': time x y z qx qy qz qw per line (space-separated)
      - kind='kitti': csv-like with specific column order for R|t
    Returns:
        list of dicts: {'time','x','y','z','qx','qy','qz','qw'}
    """
    poses = []
    if kind == "tum":
        with open(gt_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                pose = {
                    'time': str(data[0]),
                    'x': float(data[1]),
                    'y': float(data[2]),
                    'z': float(data[3]),
                    'qx': float(data[4]),
                    'qy': float(data[5]),
                    'qz': float(data[6]),
                    'qw': float(data[7]),
                }
                poses.append(pose)

    elif kind == "kitti":
        with open(gt_path, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                # Example matrix parsing (adapt to your exact format)
                R_matrix = np.array([
                    [float(data[1]), float(data[2]),  float(data[3]) ],
                    [float(data[5]), float(data[6]),  float(data[7]) ],
                    [float(data[9]), float(data[10]), float(data[11])]
                ])
                rotation = R.from_matrix(R_matrix)
                qx, qy, qz, qw = rotation.as_quat()
                pose = {
                    'time': str(data[0]),
                    'x': float(data[4]),
                    'y': float(data[8]),
                    'z': float(data[12]),
                    'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw
                }
                poses.append(pose)
    else:
        raise ValueError("Unsupported GT kind")

    return poses


def calculate_distance(p1, p2):
    """Euclidean distance between two pose dicts with keys 'x','y','z'."""
    dx = p1['x'] - p2['x']
    dy = p1['y'] - p2['y']
    dz = p1['z'] - p2['z']
    return float(np.sqrt(dx*dx + dy*dy + dz*dz))


def select_frame_indices(poses, dist_threshold):
    """
    Return a list of indices to keep, based on Euclidean distance between
    consecutive kept poses. The first pose is always kept.
    """
    keep = []
    last_pose = None
    for i, p in enumerate(poses):
        if last_pose is None:
            keep.append(i)
            last_pose = p
        else:
            if calculate_distance(last_pose, p) > dist_threshold:
                keep.append(i)
                last_pose = p
    return keep

def process_one_frame(task):
    """
    Worker function running in a separate process.
    Reads bin, builds 3-channel pano, saves .npy, returns (out_idx, pose_line, out_name).
    """
    (out_idx, bin_file, pose, save_folder,
     W_img, H0_img, Hf_img,
     v_mode, fov_up_deg, fov_down_deg, invert_vertical,
     lidar_type, max_range_m,
     k_neighbors, max_neighbor_m) = task

    # Read LiDAR
    points, reflectivity, ring, ambient = read_bin_file(bin_file, lidar_type)

    # Compute normal intensity (0..255)
    normal_intensity = compute_normals(points, k=k_neighbors, max_neighbor_m=max_neighbor_m)

    # Reflectivity image
    img_reflect = pointCloudToImage(
        points, reflectivity, ring,
        image_width=W_img, image_height_initial=H0_img, image_height_final=Hf_img,
        v_mode=v_mode, fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg,
        invert_vertical=invert_vertical
    )

    # Range image (MulRan will be saved 0..255 -> later just /255.0)
    img_range = pointCloudToRangeImage(
        points, ring,
        image_width=W_img, image_height_initial=H0_img, image_height_final=Hf_img,
        v_mode=v_mode, fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg,
        invert_vertical=invert_vertical,
        lidar_type=lidar_type, max_range_m=max_range_m
    )

    # Normal-intensity image
    img_normal = pointCloudToImage(
        points, normal_intensity, ring,
        image_width=W_img, image_height_initial=H0_img, image_height_final=Hf_img,
        v_mode=v_mode, fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg,
        invert_vertical=invert_vertical
    )

    # Stack to 3-ch (float32)
    image = np.concatenate(
        [img_reflect[:, :, np.newaxis],
         img_range[:,   :, np.newaxis],
         img_normal[:,  :, np.newaxis]],
        axis=2
    ).astype(np.float32)

    # Save .npy using preassigned index
    out_name = f"{out_idx:06d}.npy"
    np.save(os.path.join(save_folder, out_name), image)

    # Prepare pose line
    pose_line = (f"{pose['time']} {pose['x']} {pose['y']} {pose['z']} "
                 f"{pose['qx']} {pose['qy']} {pose['qz']} {pose['qw']}\n")

    return out_idx, pose_line, out_name


# ----------------------------
# Main
# ----------------------------


# Datasets to process
datasets = ["Roundabout01"]

# Rendering config
v_mode   = "ring"   # 'ring' or 'angle' for range image

fov_up_deg     = 16.0     # used only when v_mode is 'angle'
fov_down_deg   = 16.0     # used only when v_mode is 'angle'
invert_vertical = False   # set True to vertically flip the image

# Select lidar type and GT kind for the current dataset
lidar_type = "helipr"   # 'helipr' or 'mulran'
gt_kind    = "tum"      # 'tum' or 'kitti'

# distance for sampling frames
dist_threshold = 3.0  # meters

# normal calculation params
k_neighbors = 15 # neighbors for normal estimation
max_neighbor_m = 15.0 # max neighbor distance for normal estimation

# Image sizes (you can change per-channel if desired), for mulran, H0_img=64, Hf_img=64
W_img, H0_img, Hf_img = 1022, 128, 126

# range scalining (200 for helipr, 120 for mulran)
max_range_m = 200.0


if __name__ == "__main__":
    # Optional: avoid OpenCV/BLAS oversubscription in workers
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Output
    save_folder = "/output"
    os.makedirs(save_folder, exist_ok=True)
    output_pose_file = os.path.join(save_folder, "poses.txt")

    global_idx = 0

    for dataset in datasets:
        # Paths per dataset
        bin_folder = f"HeLiPR_Dataset/{dataset}/LiDAR/Ouster/"
        gt_path    = f"HeLiPR_Dataset/{dataset}/LiDAR_GT/global_Ouster_gt.txt"

        # Load GT
        poses = parse_gt_file(gt_path, kind=gt_kind)

        # All bins
        bin_files = sorted(glob.glob(os.path.join(bin_folder, '*.bin')))
        if not bin_files:
            print(f"[WARN] No .bin files found under: {bin_folder}")
            continue

        # Keep only frame indices spaced by dist_threshold
        kept_indices = select_frame_indices(poses, dist_threshold)

        # Guard against GT/bin mismatch
        kept_indices = [i for i in kept_indices if i < len(bin_files)]
        if not kept_indices:
            print(f"[INFO] No frames selected after distance filtering in {dataset}.")
            continue

        # Build tasks with preassigned output indices (sequential file names)
        tasks = []
        for local_idx, pose_idx in enumerate(kept_indices):
            out_idx = global_idx + local_idx  # sequential numbering across this dataset
            tasks.append((
                out_idx,
                bin_files[pose_idx],
                poses[pose_idx],
                save_folder,
                W_img, H0_img, Hf_img,
                v_mode, fov_up_deg, fov_down_deg, invert_vertical,
                lidar_type, max_range_m,
                k_neighbors, max_neighbor_m
            ))

        # Parallel execution
        n_workers = max(1, multiprocessing.cpu_count() // 2)  # tune as needed
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(process_one_frame, t) for t in tasks]
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc=f"Processing {dataset} (parallel)"):
                # Collect results as they finish
                results.append(f.result())

        # Write poses.txt in correct order
        results.sort(key=lambda x: x[0])  # sort by out_idx
        with open(output_pose_file, 'a') as pose_file:
            for out_idx, pose_line, out_name in results:
                pose_file.write(pose_line)

        # Advance global index for next dataset
        global_idx += len(kept_indices)

        print(f"[OK] {dataset}: saved {len(kept_indices)} frames to {save_folder}")