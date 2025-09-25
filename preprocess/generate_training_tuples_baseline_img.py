import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from tqdm import tqdm


def construct_query_dict(df_centroids: pd.DataFrame, out_filename: str):
    """
    df_centroids columns (required):
      ['file', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    Builds:
      - KDTree on (x, y)
      - positives: neighbors within 10m (excluding self)
      - non_negatives: neighbors within 30m
      - each entry includes position: [x, y, z, qx, qy, qz, qw]
      - timestamp: derived from file stem (assumes '<number>.npy')
    """
    xy = df_centroids[['x', 'y']].values
    tree = KDTree(xy)
    ind_nn = tree.query_radius(xy, r=10.0)   # positives
    ind_r  = tree.query_radius(xy, r=30.0)   # non-negatives

    queries = {}
    for i in tqdm(range(len(df_centroids)), desc="Building training queries"):
        row = df_centroids.iloc[i]
        rel_file = row['file']
        positives = np.setdiff1d(ind_nn[i], [i]).astype(int)
        non_negatives = ind_r[i].astype(int)

        # 정렬
        positives = np.sort(positives).tolist()
        non_negatives = np.sort(non_negatives).tolist()

        # 파일명에서 timestamp 추출 (예: 000123.npy → 123)
        stem = os.path.splitext(os.path.basename(rel_file))[0]
        try:
            ts = int(stem)
        except ValueError:
            # 파일명이 숫자가 아닐 경우 timestamp=0 (또는 원하는 기본값)
            ts = 0

        position = np.array([row['x'], row['y'], row['z'], row['qx'], row['qy'], row['qz'], row['qw']],
                            dtype=np.float32)

        queries[i] = {
            "id": i,
            "timestamp": ts,
            "rel_scan_filepath": rel_file,
            "positives": positives,
            "non_negatives": non_negatives,
            "position": position
        }

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    with open(out_filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved:", out_filename)


if __name__ == "__main__":
    base_path   = "../data/training"
    runs_folder = ""     # e.g., "" or "train_subdir/"
    filename    = "poses.txt"
    pointcloud_fols = "" # if you have something like "/subdir" to insert between folder and index

    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    folders = list(all_folders)  # use all runs
    print("Number of runs:", len(folders))
    print("Runs:", folders)

    # Accumulate all rows from all runs
    df_train = []
    for folder in folders:
        poses_path = os.path.join(base_path, runs_folder, folder, filename)
        if not os.path.isfile(poses_path):
            raise FileNotFoundError(f"Missing poses file: {poses_path}")

        # timestamp x y z qx qy qz qw
        df = pd.read_csv(poses_path, sep=r"\s+", header=None,
                         names=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

        # Build relative .npy path for each row
        # -> <runs_folder><folder><pointcloud_fols>/<index>.npy
        rel_prefix = runs_folder + folder + pointcloud_fols + "/"
        files = (np.arange(len(df)).astype(int)).astype(str)
        files = np.char.zfill(files, 6)
        df['file'] = rel_prefix + pd.Series(files) + ".npy"

        df_train.append(df[['file', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']])

    df_train = pd.concat(df_train, ignore_index=True)
    print("Number of training submaps:", len(df_train))

    construct_query_dict(df_train, "../data/training_queries.pickle")
