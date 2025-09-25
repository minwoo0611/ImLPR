import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


def output_to_file(output, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved:", filename)


def construct_query_and_database_sets(BASE_PATH, RUNS_FOLDER, folders, filename):
    """
    For each folder:
      - read poses.txt (columns: timestamp x y z qx qy qz qw)
      - create dicts for database & query (same content here)
      - build a KDTree on (x, y) for neighbor assignment
      - fill, for each query item, its 'true neighbors' list for each database session i
        * radius 10m
        * if i==j (intra), only neighbors with timestamp < (query_timestamp - 60)
    """
    # Pre-read all folders into dataframes and KD-trees
    df_list = []
    trees = []
    for folder in folders:
        poses_path = os.path.join(BASE_PATH, RUNS_FOLDER, folder, filename)
        if not os.path.isfile(poses_path):
            raise FileNotFoundError(f"Missing poses file: {poses_path}")

        # timestamp x y z qx qy qz qw
        df = pd.read_csv(poses_path, sep=r"\s+", header=None,
                         names=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        # build relative .npy path: <RUNS_FOLDER><folder>/<index>.npy
        df['file'] = RUNS_FOLDER + folder + "/" + df.index.astype(str).str.zfill(6) + ".npy"
        df_list.append(df)

        trees.append(KDTree(df[['x', 'y']].values))

    # Build per-folder database & query dicts
    database_sets, query_sets = [], []
    for df in df_list:
        database = {}
        query = {}
        # keep ordering consistent with KDTree indices
        for i, row in df.iterrows():
            entry = {
                'query': row['file'],
                'northing': float(row['x']),
                'easting': float(row['y']),
                'timestamp': float(row['timestamp']),
            }
            
            database[len(database)] = entry.copy()
            query[len(query)] = entry.copy()
        database_sets.append(database)
        query_sets.append(query)

    # Assign true neighbors for each query against each database session
    radius = 10.0  # meters
    for i in range(len(folders)):        # database session index
        for j in range(len(folders)):    # query session index
            tree = trees[i]
            db = database_sets[i]
            qset = query_sets[j]

            db_xy = df_list[i][['x', 'y']].values  # not used, but kept for clarity

            for qk in range(len(qset)):
                q_xy = np.array([[qset[qk]['northing'], qset[qk]['easting']]])
                idxs = tree.query_radius(q_xy, r=radius)[0].tolist()

                if i == j:  # intra: filter by temporal constraint
                    q_ts = qset[qk]['timestamp']
                    idxs = [idx for idx in idxs if database_sets[i][idx]['timestamp'] < (q_ts - 60.0)]

                # save under key i (database session index), same as your previous convention
                qset[qk][i] = idxs

    # Output
    output_to_file(database_sets, '../data/db_img.pickle')
    output_to_file(query_sets,   '../data/query_img.pickle')   # fixed path (no leading slash)


if __name__ == "__main__":
    BASE_PATH = "../data/validation/"
    RUNS_FOLDER = ""   # e.g., "" or "validation_subdir/"
    SEQUENCE = "Town"  # folders starting with this prefix

    all_folders = sorted(os.listdir(os.path.join(BASE_PATH, RUNS_FOLDER)))
    folders = [f for f in all_folders if f.startswith(SEQUENCE)]
    print("Folders:", folders)

    construct_query_and_database_sets(BASE_PATH, RUNS_FOLDER, folders, "poses.txt")
