# Implemented in MinkLoc3Dv2

import numpy as np
import torch

from models.losses.loss_utils import sigmoid, compute_aff


class TruncatedSmoothAP:
    """Truncated Smooth-AP loss with top-k positives per query."""
    def __init__(self, tau1: float = 0.01, similarity: str = 'cosine', positives_per_query: int = 4):
        self.tau1 = tau1
        self.similarity = similarity
        self.positives_per_query = positives_per_query

    def __call__(self, embeddings, positives_mask, negatives_mask):
        device = embeddings.device
        positives_mask = positives_mask.to(device)
        negatives_mask = negatives_mask.to(device)

        # Pairwise similarity: rows = queries, cols = items
        s_qz = compute_aff(embeddings, similarity=self.similarity)

        # Select top-k positives per query
        s_positives = s_qz.detach().clone()
        s_positives.masked_fill_(~positives_mask, np.NINF)
        closest_positives_ndx = torch.topk(
            s_positives, k=self.positives_per_query, dim=1, largest=True, sorted=True
        )[1]

        n_positives = positives_mask.sum(dim=1)

        # Smooth ranks
        s_diff = s_qz.unsqueeze(1) - s_qz.gather(1, closest_positives_ndx).unsqueeze(2)
        s_sigmoid = sigmoid(s_diff, temp=self.tau1)

        # Numerator: rank among positives (exclude the exact positive position)
        pos_mask = positives_mask.unsqueeze(1)
        pos_s_sigmoid = s_sigmoid * pos_mask
        mask = torch.ones_like(pos_s_sigmoid).scatter(2, closest_positives_ndx.unsqueeze(2), 0.)
        pos_s_sigmoid = pos_s_sigmoid * mask
        r_p = torch.sum(pos_s_sigmoid, dim=2) + 1.0

        # Denominator: rank among positives + negatives
        neg_mask = negatives_mask.unsqueeze(1)
        neg_s_sigmoid = s_sigmoid * neg_mask
        r_omega = r_p + torch.sum(neg_s_sigmoid, dim=2)

        r = r_p / r_omega  # (N, k)

        # Metrics
        stats = {}
        stats['positives_per_query'] = n_positives.float().mean().item()
        temp = (s_diff.detach() > 0)
        temp = torch.logical_and(temp[:, 0], negatives_mask)
        hard_ranking = temp.sum(dim=1)
        stats['best_positive_ranking'] = hard_ranking.float().mean().item()
        stats['recall'] = {1: (hard_ranking <= 1).float().mean().item()}

        # Average precision over available positives
        valid_positives_mask = torch.gather(positives_mask, 1, closest_positives_ndx)
        masked_r = r * valid_positives_mask
        n_valid_positives = valid_positives_mask.sum(dim=1)
        valid_q_mask = n_valid_positives > 0
        masked_r = masked_r[valid_q_mask]

        ap = (masked_r.sum(dim=1) / n_valid_positives[valid_q_mask]).mean()
        loss = 1.0 - ap

        stats['loss'] = loss.item()
        stats['ap'] = ap.item()
        stats['avg_embedding_norm'] = embeddings.norm(dim=1).mean().item()
        return loss, stats
