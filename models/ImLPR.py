import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter


def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    """Sinkhorn matrix scaling for differentiable OT. Returns log-transport matrix."""
    M = M / reg
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()
    return M + u.unsqueeze(2) + v.unsqueeze(1)


def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    """Sinkhorn with a dustbin row; returns log-assignment probs."""
    batch_size, m, n = S.size()
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)

    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm


class SALAD(nn.Module):
    """Sinkhorn Aggregation over local descriptors with a global token."""
    def __init__(self, params) -> None:
        super().__init__()
        self.num_channels = params.num_channels
        self.num_clusters = params.num_clusters
        self.cluster_dim = params.cluster_dim
        self.token_dim = params.token_dim

        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512), nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1), nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1), nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        x, t = x
        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)[:, :-1, :]

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        f = torch.cat([
            F.normalize(t, p=2, dim=-1),
            F.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)

        return F.normalize(f, p=2, dim=-1)


DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}


class MultiConvAdapter(nn.Module):
    """Parallel 1x1/3x3/5x5 conv adapter on token maps (from paper in SelaVPR++)."""
    def __init__(self, in_channels):
        super(MultiConvAdapter, self).__init__()
        out_channels = in_channels // 3
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1x1_for_3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1x1_for_5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x, H=None):
        B, C = x.size(0), x.size(2)
        x = x.view(B, C, H // 14, -1)
        x = self.relu(x)
        o1 = self.conv1x1(x)
        o3 = self.conv3x3(self.conv1x1_for_3(x))
        o5 = self.conv5x5(self.conv1x1_for_5(x))
        y = torch.cat([o1, o3, o5], dim=1) + x
        return y.view(B, -1, C)


class AdaptedDINOv2(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        model_name = params.model_name
        assert model_name in DINOV2_ARCHS, f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = params.num_trainable_blocks
        self.norm_layer = True
        self.return_token = True

        self.num_blocks = len(self.model.blocks)
        self.multi_conv_adapters = nn.ModuleList(
            [MultiConvAdapter(self.num_channels) for _ in range(self.num_blocks // params.adapter_frequency)]
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)

        output = []
        for blk in self.model.blocks[:-self.num_trainable_blocks]:
            x = blk(x)
            output.append(x[:, 1:])
        x = x.detach()

        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)
            output.append(x[:, 1:])

        for i in range(len(self.multi_conv_adapters)):
            if i == 0:
                y = self.multi_conv_adapters[i](output[0] + output[2], H=H) + output[0]
            else:
                y = self.multi_conv_adapters[i](y + output[3 * (i + 1) - 1], H=H) + y

        output_token = x[:, 0].unsqueeze(1)
        x = torch.cat([output_token, y], dim=1)
        if self.norm_layer:
            x = self.model.norm(x)

        t = x[:, 0]
        f = x[:, 1:]
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        return (f, t) if self.return_token else f


class AdaptedViT(nn.Module):
    def __init__(self, params):
        super(AdaptedViT, self).__init__()
        self.encoder = AdaptedDINOv2(params)

    def forward(self, x):
        x, t = self.encoder(x)
        return x, t


class ImLPR(nn.Module):
    def __init__(self, params):
        super(ImLPR, self).__init__()
        self.encoder = AdaptedViT(params)
        self.pooling = SALAD(params)

    def forward(self, x):
        local_feat, tokens = self.encoder(x)
        global_desc = self.pooling((local_feat, tokens))
        return local_feat, global_desc
