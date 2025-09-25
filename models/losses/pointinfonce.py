import torch
import torch.nn.functional as F

class PointInfoNCELoss:
    """
    Point-to-point InfoNCE loss as a callable class.
    Call: loss = PointInfoNCELoss(...)(F0, F1, positive_pairs)
    Args in __init__: num_pos, num_hn_samples, temperature, vdist, hdist, circular_horizontal
    """
    def __init__(
        self,
        num_pos: int = 192,
        num_hn_samples: int = 64,
        temperature: float = 0.20,
        vdist: int = 3,
        hdist: int = 20,
        circular_horizontal: bool = True,
    ):
        self.num_pos = num_pos
        self.num_hn_samples = num_hn_samples
        self.temperature = temperature
        self.vdist = vdist
        self.hdist = hdist
        self.circular_horizontal = circular_horizontal

    @torch.no_grad()
    def _mine_negatives(
        self,
        F0_b: torch.Tensor,               # (H*W, C)
        F1_b: torch.Tensor,               # (H*W, C)
        pos_ind0: torch.Tensor,           # (P,)
        pos_ind1: torch.Tensor,           # (P,)
        H: int, W: int,
        device: torch.device,
    ):
        """Build 'far' negatives by masking local boxes around positives and sample up to num_hn_samples."""
        vdist, hdist = self.vdist, self.hdist
        circular = self.circular_horizontal

        N0, N1 = F0_b.shape[0], F1_b.shape[0]

        # Negatives in F1 (for anchors in F0)
        allInd1 = torch.arange(N1, device=device)
        rowAll1, colAll1 = allInd1 // W, allInd1 % W
        rowPos1, colPos1 = pos_ind1 // W, pos_ind1 % W

        mask1 = torch.ones(N1, dtype=torch.bool, device=device)
        for rA, cA in zip(rowPos1, colPos1):
            row_diff = (rowAll1 - rA).abs()
            col_diff = (colAll1 - cA).abs()
            if circular:
                col_diff = torch.minimum(col_diff, W - col_diff)
            mask1 &= ~((row_diff <= vdist) & (col_diff <= hdist))
        mask1[pos_ind1] = False

        validNeg1 = allInd1[mask1]
        if validNeg1.numel() == 0:
            validNeg1 = allInd1
        k1 = min(validNeg1.numel(), self.num_hn_samples)
        sel1 = validNeg1[torch.randperm(validNeg1.numel(), device=device)[:k1]]
        subF1 = F1_b[sel1]

        # Negatives in F0 (for anchors in F1)
        allInd0 = torch.arange(N0, device=device)
        rowAll0, colAll0 = allInd0 // W, allInd0 % W
        rowPos0, colPos0 = pos_ind0 // W, pos_ind0 % W

        mask0 = torch.ones(N0, dtype=torch.bool, device=device)
        for rB, cB in zip(rowPos0, colPos0):
            row_diff = (rowAll0 - rB).abs()
            col_diff = (colAll0 - cB).abs()
            if circular:
                col_diff = torch.minimum(col_diff, W - col_diff)
            mask0 &= ~((row_diff <= vdist) & (col_diff <= hdist))
        mask0[pos_ind0] = False

        validNeg0 = allInd0[mask0]
        if validNeg0.numel() == 0:
            validNeg0 = allInd0
        k0 = min(validNeg0.numel(), self.num_hn_samples)
        sel0 = validNeg0[torch.randperm(validNeg0.numel(), device=device)[:k0]]
        subF0 = F0_b[sel0]

        return subF0, subF1

    def __call__(self, F0: torch.Tensor, F1: torch.Tensor, positive_pairs):
        """
        F0, F1: (B, C, H, W)
        positive_pairs: list of length B; each item is [((r0,c0),(r1,c1)), ...]
        Returns: scalar loss tensor
        """
        F0 = F0[:, :, :, 2:-2]
        F1 = F1[:, :, :, 2:-2]

        device = F0.device
        B, C, H, W = F0.shape
        losses = []

        for b in range(B):
            F0_b = F0[b].permute(1, 2, 0).reshape(-1, C)
            F1_b = F1[b].permute(1, 2, 0).reshape(-1, C)

            if len(positive_pairs[b]) == 0:
                continue

            pos_pairs = torch.tensor(
                [(r0 * W + c0, r1 * W + c1) for (r0, c0), (r1, c1) in positive_pairs[b]],
                dtype=torch.long, device=device
            )
            if pos_pairs.numel() == 0:
                continue

            if pos_pairs.shape[0] > self.num_pos:
                sel = torch.randint(pos_pairs.shape[0], (self.num_pos,), device=device)
                pos_pairs = pos_pairs[sel]

            pos_ind0, pos_ind1 = pos_pairs[:, 0], pos_pairs[:, 1]

            with torch.no_grad():
                subF0, subF1 = self._mine_negatives(
                    F0_b=F0_b, F1_b=F1_b,
                    pos_ind0=pos_ind0, pos_ind1=pos_ind1,
                    H=H, W=W, device=device,
                )

            posF0 = F0_b[pos_ind0]
            posF1 = F1_b[pos_ind1]

            sim_pos   = F.cosine_similarity(posF0, posF1, dim=1)
            sim_neg_0 = F.cosine_similarity(posF0.unsqueeze(1), subF1, dim=2)
            sim_neg_1 = F.cosine_similarity(posF1.unsqueeze(1), subF0, dim=2)

            t = self.temperature
            sim_pos, sim_neg_0, sim_neg_1 = sim_pos / t, sim_neg_0 / t, sim_neg_1 / t

            logits = torch.cat([sim_pos.unsqueeze(1), sim_neg_0, sim_neg_1], dim=1)
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=device)

            losses.append(F.cross_entropy(logits, targets))

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(losses).mean()
