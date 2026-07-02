import torch
from torch import nn
import torch.nn.functional as F

from .layers import (
    CrossScaleFusion,
    GlobalStructuralPath,
    IntraScaleFusion,
    LocalDynamicPath,
    NormalizedCausalDecomposition,
    PredictionHead,
    align_trend,
    causal_average,
)


class MSDHFNet(nn.Module):
    def __init__(
        self,
        channels,
        pred_len,
        scales=(1, 2, 4),
        kernel_size=3,
        hidden_dim=64,
        dropout=0.1,
        num_heads=4,
        ffn_ratio=2,
        global_depth=2,
        local_dilations=(1, 2),
        ablation="none",
        eps=1e-5,
    ):
        super().__init__()
        self.channels = channels
        self.pred_len = pred_len
        self.scales = tuple(scales)
        self.ablation = ablation
        self.eps = eps
        self.decompositions = nn.ModuleList([NormalizedCausalDecomposition(kernel_size) for _ in self.scales])
        self.local_paths = nn.ModuleList(
            [LocalDynamicPath(hidden_dim, kernel_size, local_dilations, dropout) for _ in self.scales]
        )
        self.global_paths = nn.ModuleList(
            [
                GlobalStructuralPath(hidden_dim, global_depth, num_heads, ffn_ratio, dropout)
                for _ in self.scales
            ]
        )
        self.intra_fusions = nn.ModuleList([IntraScaleFusion(hidden_dim) for _ in self.scales])
        self.cross_fusion = CrossScaleFusion(hidden_dim, len(self.scales))
        self.head = PredictionHead(channels, hidden_dim, pred_len, dropout)

    def forward(self, x, return_aux=False):
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        normalized = (x - mean) / std
        features = []
        trends = []
        path_gates = []
        for index, scale in enumerate(self.scales):
            scaled = causal_average(normalized, scale)
            if self.ablation == "without_ncd":
                trend = scaled
                fluctuation = scaled
            else:
                trend, fluctuation = self.decompositions[index](scaled)
            local = self.local_paths[index](fluctuation)
            global_ = self.global_paths[index](trend)
            if self.ablation == "without_local":
                fused = global_
                path_gate = torch.zeros(x.size(0), global_.size(-1), device=x.device)
            elif self.ablation == "without_global":
                fused = local
                path_gate = torch.ones(x.size(0), local.size(-1), device=x.device)
            elif self.ablation == "without_igf":
                fused = 0.5 * (local + global_)
                path_gate = torch.full((x.size(0), local.size(-1)), 0.5, device=x.device)
            else:
                fused, path_gate = self.intra_fusions[index](local, global_, return_gate=True)
            features.append(fused)
            trends.append(trend)
            path_gates.append(path_gate)
        fused, scale_weights = self.cross_fusion(
            features, normalized.size(1), fixed=self.ablation == "without_ccf"
        )
        prediction = self.head(fused)
        prediction = prediction * std + mean
        consistency = (
            normalized.new_tensor(0.0)
            if self.ablation in {"without_ncd", "without_lcon"}
            else self.consistency_loss(trends)
        )
        if return_aux:
            return prediction, {
                "consistency_loss": consistency,
                "scale_weights": scale_weights,
                "path_gates": torch.stack(path_gates, dim=1),
                "trends": trends,
            }
        return prediction

    def consistency_loss(self, trends):
        losses = []
        for index in range(len(trends) - 1):
            aligned = align_trend(trends[index], trends[index + 1].size(1))
            target = trends[index + 1]
            length = min(aligned.size(1), target.size(1))
            losses.append(F.mse_loss(aligned[:, :length], target[:, :length]))
        if not losses:
            return trends[0].new_tensor(0.0)
        return torch.stack(losses).sum()
