import os
import numpy as np
import torch
import pathlib


def print_global_stats(phase, stats):
    s = f"{phase}  loss: {stats['loss']:.4f}   embedding norm: {stats['avg_embedding_norm']:.3f}  "
    if 'num_triplets' in stats:
        s += (
            f"Triplets (all/active): {stats['num_triplets']:.1f}/"
            f"{stats['num_non_zero_triplets']:.1f}  "
            f"Mean dist (pos/neg): {stats['mean_pos_pair_dist']:.3f}/"
            f"{stats['mean_neg_pair_dist']:.3f}   "
        )
    if 'positives_per_query' in stats:
        s += f"#positives per query: {stats['positives_per_query']:.1f}   "
    if 'best_positive_ranking' in stats:
        s += f"best positive rank: {stats['best_positive_ranking']:.1f}   "
    if 'recall' in stats:
        s += f"Recall@1: {stats['recall'][1]:.4f}   "
    if 'ap' in stats:
        s += f"AP: {stats['ap']:.4f}   "
    print(s)


def print_stats(phase, stats):
    print_global_stats(phase, stats['global'])


def tensors_to_numbers(stats):
    return {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}


def training_step(global_iter, model, phase, device, optimizer, loss_fn, **kwargs):
    """Single-stage training/eval step."""
    assert phase in ['train', 'val']

    batch, positives_mask, negatives_mask = next(global_iter)
    batch = {e: batch[e].to(device) for e in batch}

    model.train() if phase == 'train' else model.eval()
    optimizer.zero_grad()

    with torch.set_grad_enabled(phase == 'train'):
        y = model(batch)
        stats = model.stats.copy() if hasattr(model, 'stats') else {}
        embeddings = y['global']
        loss, temp_stats = loss_fn(embeddings, positives_mask, negatives_mask)
        stats.update(tensors_to_numbers(temp_stats))
        if phase == 'train':
            loss.backward()
            optimizer.step()

    torch.cuda.empty_cache()
    return stats


def multistaged_training_step(global_iter, model, phase, device, optimizer, loss_fn, **kwargs):
    """
    Multi-staged backprop (listwise AP-style). Assumes loss_fn returns
    (loss_fn_truncated, point_infonce_loss) for stage-2 computation.
    """
    assert phase in ['train', 'val']

    batch, positives_mask, negatives_mask, sampled_pairs, positive_pairs = next(global_iter)
    model.train() if phase == 'train' else model.eval()

    # Stage 1: descriptors (no grad, keep BN in current mode)
    embeddings_l, local_feat_l = [], []
    with torch.set_grad_enabled(False):
        for minibatch in batch:
            minibatch = minibatch.to(device)
            local_feat, y = model(minibatch)
            local_feat_l.append(local_feat)
            embeddings_l.append(y)

    torch.cuda.empty_cache()

    # Stage 2: compute loss wrt embeddings
    embeddings = torch.cat(embeddings_l, dim=0)
    all_local_feats = torch.cat(local_feat_l, dim=0)
    loss_fn_truncated, point_infonce_loss = loss_fn

    with torch.set_grad_enabled(phase == 'train'):
        if phase == 'train':
            embeddings.requires_grad_(True)
            all_local_feats.requires_grad_(True)

        F0_features = all_local_feats[sampled_pairs[:, 0]]
        F1_features = all_local_feats[sampled_pairs[:, 1]]

        loss_truncated, stats = loss_fn_truncated(embeddings, positives_mask, negatives_mask)
        loss_contrastive = point_infonce_loss(F0_features, F1_features, positive_pairs)
        print(loss_truncated.item(), loss_contrastive.item())
        loss = loss_truncated + 2 * loss_contrastive
        stats = tensors_to_numbers(stats)

        if phase == 'train':
            loss.backward()
            embeddings_grad = embeddings.grad
            all_local_feats_grad = all_local_feats.grad

    embeddings_l = local_feat_l = embeddings = all_local_feats = None
    torch.cuda.empty_cache()

    # Stage 3: recompute descriptors with grad, apply cached grads
    if phase == 'train':
        optimizer.zero_grad()
        i = 0
        with torch.set_grad_enabled(True):
            for minibatch in batch:
                minibatch = minibatch.to(device)
                local_feat, y = model(minibatch)
                minibatch_size = len(y)
                y.backward(gradient=embeddings_grad[i:i + minibatch_size], retain_graph=True)
                local_feat.backward(gradient=all_local_feats_grad[i:i + minibatch_size])
                i += minibatch_size
            optimizer.step()

    torch.cuda.empty_cache()
    return stats


class CustomCosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters, min_value=0.4):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.min_value = min_value
        self.warm_lr = self._cosine(self.warmup)
        super().__init__(optimizer)

    def _cosine(self, epoch):
        return 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = self._cosine(epoch)
        if epoch <= self.warmup:
            lr_factor = self.warm_lr * (epoch + 1) / self.warmup
        elif lr_factor < self.min_value:
            lr_factor = self.min_value
        return lr_factor


def create_weights_folder():
    """Create <repo_root>/weights if missing, return its path."""
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), f'Cannot create weights folder: {weights_path}'
    return weights_path
