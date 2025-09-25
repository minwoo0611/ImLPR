# Warsaw University of Technology
# Evaluation using PointNetVLAD protocol with global dataset selectors

import os
import time
import pickle
import numpy as np
import torch
import tqdm
from sklearn.neighbors import KDTree
from misc.utils import TrainingParams
import argparse
from joblib import Parallel, delayed

# =========================
# Global selectors (edit me)
# =========================
# Example:
#   - dataset: short tag used in pickle names: db_{dataset}_{sequence}.pickle
#   - sequences: list of sequence tags matching your pickle filenames
EVAL_DATASETS = [
    {"dataset": "helipr", "sequences": ["town"]},
]

# If your eval pickles live in a subfolder inside params.dataset_folder, set it here
EVAL_SUBFOLDER = ""       # e.g., "eval_pickles" or "" to use params.dataset_folder directly
RESIZE_TO_HELIPR = False  # True if dataset is mulran

# =========================
# I/O helpers
# =========================

def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_eval_pickle_paths(params: TrainingParams):
    """
    Resolve database/query pickle paths from global EVAL_DATASETS.
    Expected file names: db_{dataset}_{sequence}.pickle, query_{dataset}_{sequence}.pickle
    Located under: os.path.join(params.dataset_folder, EVAL_SUBFOLDER)
    """
    base = os.path.join(params.dataset_folder, EVAL_SUBFOLDER) if EVAL_SUBFOLDER else params.dataset_folder
    db_paths, query_paths = [], []
    for spec in EVAL_DATASETS:
        ds = spec["dataset"]
        for seq in spec["sequences"]:
            db_paths.append(os.path.join(base, f"db_{ds}_{seq}.pickle"))
            query_paths.append(os.path.join(base, f"query_{ds}_{seq}.pickle"))
    return db_paths, query_paths


def _load_eval_sets(params: TrainingParams):
    db_paths, query_paths = _build_eval_pickle_paths(params)
    for p in db_paths + query_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Eval pickle not found: {p}")
    database_sets = [_load_pickle(p) for p in db_paths][0]
    query_sets = [_load_pickle(p) for p in query_paths][0]

    return database_sets, query_sets


# =========================
# Embedding extraction
# =========================

def _get_latent_vectors(model, data_set, device, dataset_folder, batch_size):
    """
    data_set: dict-like, index -> meta. Must contain meta["query"] as a relative path
              under {dataset_folder}/validation/.
    Loads .npy (H,W,3), pads 28 cols both sides, CHW, /255.0, forward → descriptor.
    """
    model.eval()
    n_items = len(data_set)

    embeddings = None
    batch_imgs, batch_idx = [], []

    # load data_set and in dataset[0][i]["query"], get relative path and change /PR_front_img/Town01-Ouster000000.png as /Town01-Ouster/000000.npy

    for i, elem_idx in enumerate(tqdm.tqdm(data_set, desc="Computing latent vectors")):
        rel_path = data_set[elem_idx]["query"]
        npy_path = os.path.join(dataset_folder, "validation", rel_path)
        if not os.path.isfile(npy_path):
            print(f"[warn] missing npy: {npy_path}")
            continue

        img = np.load(npy_path)  # H x W x 3
        # cylindrical pad (28 px)
        left, right = img[:, :28], img[:, -28:]
        img = np.concatenate([right, img, left], axis=1)

        if RESIZE_TO_HELIPR:
            import cv2
            H, W, _ = img.shape
            img = cv2.resize(img, (W, 126), interpolation=cv2.INTER_LINEAR)

        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        batch_imgs.append(img)
        batch_idx.append(i)

        if len(batch_imgs) == batch_size:
            embeddings = _flush_batch(model, device, batch_imgs, batch_idx, n_items, embeddings)
            batch_imgs, batch_idx = [], []

    if batch_imgs:
        embeddings = _flush_batch(model, device, batch_imgs, batch_idx, n_items, embeddings)

    if embeddings is None:  # no valid items
        embeddings = np.zeros((n_items, 0), dtype=np.float32)
    return embeddings


def _flush_batch(model, device, images, indices, total_items, embeddings):
    """Forward a batch and write into (and/or allocate) the embeddings matrix."""
    x = torch.from_numpy(np.stack(images, axis=0)).to(device)
    with torch.no_grad():
        _, desc = model(x)  # shape (B, D)

    desc_np = desc.detach().cpu().numpy()
    if embeddings is None:
        embeddings = np.zeros((total_items, desc_np.shape[1]), dtype=np.float32)
    for idx, emb in zip(indices, desc_np):
        embeddings[idx] = emb
    return embeddings


# =========================
# Metrics core
# =========================

def _process_query(
    i: int,
    query_vector: np.ndarray,
    query_meta: dict,
    kd_tree: KDTree,
    num_neighbors: int,
    threshold_k: int,
    database_output: np.ndarray,
    thresholds: np.ndarray,
    m: int,
    database_meta=None,
    is_intra: bool = False,
    init_time: float = None,
    log_file: str | None = None,
):
    """
    Single-query evaluation:
      - optional intra-session filtering (skip <90s from start; only match db entries 60s older)
      - KDTree search
      - recall@k, one-percent recall
      - threshold-based TP/FP/TN/FN using the first neighbor's distance
      - optional logging of top-20 indices and geo distance sanity check
    """
    # true neighbors for this query (stored under key m)
    true_neighbors = query_meta.get(m, [])
    if len(true_neighbors) == 0:
        return None

    # Intra-session gating
    if is_intra:
        if database_meta is None or init_time is None:
            raise ValueError("database_meta and init_time required for intra-session")
        query_ts = float(query_meta['timestamp']) / 1e9
        if query_ts - init_time < 90:
            return None

        filtered_neighbors = [
            idx for idx in true_neighbors
            if float(database_meta[idx]['timestamp']) / 1e9 < query_ts - 60
        ]
    else:
        filtered_neighbors = true_neighbors

    if len(filtered_neighbors) == 0:
        return None

    # Nearest neighbors
    distances, indices = kd_tree.query([query_vector], k=max(200, num_neighbors, threshold_k))
    indices = indices[0]
    distances = distances[0]

    # Further filter predicted indices for intra-session (only < query_ts - 60s)
    if is_intra:
        query_ts = float(query_meta['timestamp']) / 1e9
        filt_idx, filt_dist = [], []
        for j, idx in enumerate(indices):
            if float(database_meta[idx]['timestamp']) / 1e9 < query_ts - 60:
                filt_idx.append(idx)
                filt_dist.append(distances[j])
        indices, distances = filt_idx, filt_dist
        if not indices:
            return None

    # recall@k
    recall_indicator = np.zeros(num_neighbors)
    for j, idx in enumerate(indices[:num_neighbors]):
        if idx in filtered_neighbors:
            recall_indicator[j] = 1
            break

    # one-percent recall flag (top threshold_k)
    one_percent_flag = 1 if len(set(indices[:threshold_k]).intersection(filtered_neighbors)) > 0 else 0

    # threshold metrics using the first neighbor
    first_idx = indices[0]
    first_dist = distances[0]
    num_th = len(thresholds)
    tp = np.zeros(num_th)
    fp = np.zeros(num_th)
    tn = np.zeros(num_th)
    fn = np.zeros(num_th)

    for t_i, th in enumerate(thresholds):
        if first_dist < th:
            if first_idx in filtered_neighbors:
                tp[t_i] = 1
            else:
                fp[t_i] = 1
        else:
            if first_idx in filtered_neighbors:
                fn[t_i] = 1
            else:
                tn[t_i] = 1

    return recall_indicator, one_percent_flag, tp, fp, tn, fn, 1



def _get_recall_for_pair(
    m: int,
    n: int,
    database_vectors: list[np.ndarray],
    query_vectors: list[np.ndarray],
    query_sets: list[dict],
    database_sets: list[dict],
    log_dir: str | None = None,
    n_jobs: int = 20,
):
    """
    Pairwise evaluation for (m, n).
    - Runs process_query() in parallel
    - Returns (recall_cumulative, one_percent_recall)
    - No console printing; optional logs written under log_dir if provided.
    """
    database_output = database_vectors[m]
    queries_output = query_vectors[n]
    database_meta = list(database_sets[m].values())
    query_meta = list(query_sets[n].values())

    kd_tree = KDTree(database_output)
    num_neighbors = 30
    threshold_k = max(int(round(len(database_output) / 1000.0)), 1)
    thresholds = np.linspace(0, 1, 1000)

    is_intra = (m == n)
    init_time = float(min(d['timestamp'] for d in database_meta)) / 1e9 if is_intra else None

    # per-pair log file for “top-20” neighbor snapshots (optional)
    log_file = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"top20_m{m}_n{n}.txt")
        open(log_file, "w").close()  # truncate

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_query)(
            i=i,
            query_vector=queries_output[i],
            query_meta=query_meta[i],
            kd_tree=kd_tree,
            num_neighbors=num_neighbors,
            threshold_k=threshold_k,
            database_output=database_output,
            thresholds=thresholds,
            m=m,
            database_meta=database_meta,
            is_intra=is_intra,
            init_time=init_time,
            log_file=log_file
        ) for i in range(len(queries_output))
    )

    recall_sum = np.zeros(num_neighbors)
    one_percent_total = 0
    num_th = len(thresholds)
    num_tp = np.zeros(num_th)
    num_fp = np.zeros(num_th)
    num_tn = np.zeros(num_th)
    num_fn = np.zeros(num_th)
    num_eval = 0

    for res in results:
        if res is None:
            continue
        r_ind, one_flag, tp, fp, tn, fn, c = res
        recall_sum += r_ind
        one_percent_total += one_flag
        num_tp += tp
        num_fp += fp
        num_tn += tn
        num_fn += fn
        num_eval += c

    if num_eval == 0:
        recall_cumulative = np.zeros(num_neighbors)
        one_percent_recall = 0.0
    else:
        recall_cumulative = (np.cumsum(recall_sum) / num_eval) * 100.0
        one_percent_recall = (one_percent_total / num_eval) * 100.0

    # calculate maximum F1 score and AUC
    precisions = np.divide(num_tp, (num_tp + num_fp + 1e-8))
    recalls = np.divide(num_tp, (num_tp + num_fn + 1e-8))

    f1_scores = np.divide(2 * precisions * recalls, (precisions + recalls + 1e-8))
    max_f1 = np.max(f1_scores) if f1_scores.size > 0 else 0.0

    sorted_indices = np.argsort(recalls)
    sorted_recalls = recalls[sorted_indices]
    sorted_precisions = precisions[sorted_indices]
    auc = np.trapz(sorted_precisions, sorted_recalls) if sorted_recalls.size > 1 else 0.0

    r1 = recall_cumulative[0] if recall_cumulative.size > 0 else 0.0 
    print(f"[pair m={m}, n={n}] Recall@1={r1:.4f}% OPR={one_percent_recall:.4f}% MaxF1={max_f1:.4f} AUC={auc:.4f} over {num_eval} queries")

    return recall_cumulative, one_percent_recall



# =========================
# Public API
# =========================

def evaluate(model, device, params: TrainingParams, show_progress: bool = True):
    """
    Entry point you asked for:
        stats = evaluate(model, device, params, show_progress=True)

    Uses global EVAL_DATASETS to discover db/query pickle paths.
    """
    # 1) Load eval sets
    database_sets, query_sets = _load_eval_sets(params)

    # 2) Compute embeddings
    db_vecs, q_vecs = [], []

    # if dataset == mulran, resize
    print(f"[eval] computing database & query embeddings for {len(database_sets)} set(s)")
    t0 = time.time()

    for s in database_sets:
        embedding = _get_latent_vectors(model, s, device, params.dataset_folder, params.val_batch_size)
        db_vecs.append(embedding)
        q_vecs.append(embedding) 
    t1 = time.time()


    n_db = sum(v.shape[0] for v in db_vecs) or 1
    n_q = sum(v.shape[0] for v in q_vecs) or 1
    print(f"[timing] inference: {(t1 - t0) / n_db * 1000:.2f} ms/scan")

    # 3) Pairwise evaluation
    total_pairs = 0
    recall_accum = None
    opr_list = []
    for m in range(len(db_vecs)):
        for n in range(len(q_vecs)):
            recall_cum, opr = _get_recall_for_pair(m, n, db_vecs, q_vecs, query_sets, database_sets)
            recall_accum = recall_cum if recall_accum is None else (recall_accum + recall_cum)
            opr_list.append(opr)
            total_pairs += 1

    if total_pairs == 0:
        return {'ave_recall': 0.0, 'ave_one_percent_recall': 0.0}

    ave_recall_curve = recall_accum / total_pairs
    ave_one_percent = float(np.mean(opr_list)) if opr_list else 0.0

    print(f"[eval] Average Recall@1: {ave_recall_curve[0]:.4f}%")
    print(f"[eval] Average One-Percent Recall: {ave_one_percent:.4f}%")

    return {'ave_recall': float(ave_recall_curve[0]), 'ave_one_percent_recall': ave_one_percent}


def print_eval_stats(stats):
    print("=== Evaluation Summary ===")
    print(f"Average Recall@1: {stats['ave_recall']:.4f}%")
    print(f"Average One-Percent Recall: {stats['ave_one_percent_recall']:.4f}%")

def _load_model(params: TrainingParams, device: torch.device):
    from models.ImLPR import ImLPR
    model = ImLPR(params.model_params)

    # Multi-GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"[info] Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    weights = params.load_weights
    if weights:
        if os.path.isfile(weights):
            print(f"[info] loading weights: {weights}")
            state = torch.load(weights, map_location="cpu")
            if not isinstance(model, torch.nn.DataParallel):
                # strip "module." if present
                state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
            missing_unexpected = model.load_state_dict(state, strict=False)
            if hasattr(missing_unexpected, "missing_keys"):
                print("[load] missing:", missing_unexpected.missing_keys)
                print("[load] unexpected:", missing_unexpected.unexpected_keys)
        else:
            print(f"[warn] weights not found: {weights} (using random init)")
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate ImLPR")
    parser.add_argument("--config",       type=str, required=True, help="Path to training/eval config INI")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config INI")
    parser.add_argument("--cpu",          action="store_true",     help="Force CPU")
    args = parser.parse_args()

    params = TrainingParams(args.config, args.model_config, debug=False)
    params.print()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[info] device: {device}")

    model = _load_model(params, device)

    stats = evaluate(model, device, params, show_progress=True)
    print_eval_stats(stats)

if __name__ == "__main__":
    main()
