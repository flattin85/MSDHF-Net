import math

import torch
from torch import nn
import torch.nn.functional as F


def causal_average(x, scale):
    if scale == 1:
        return x
    length = x.size(1)
    outputs = []
    for index in range(math.ceil(length / scale)):
        end = min((index + 1) * scale, length)
        start = max(0, end - scale)
        outputs.append(x[:, start:end].mean(dim=1))
    return torch.stack(outputs, dim=1)


def align_trend(x, target_length):
    aligned = causal_average(x, 2)
    if aligned.size(1) > target_length:
        aligned = aligned[:, :target_length]
    if aligned.size(1) < target_length:
        pad = aligned[:, -1:].expand(-1, target_length - aligned.size(1), -1)
        aligned = torch.cat([aligned, pad], dim=1)
    return aligned


class CausalSmoothing(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(kernel_size))

    def forward(self, x):
        weights = torch.softmax(self.logits, dim=0)
        parts = []
        first = x[:, :1]
        for index, weight in enumerate(weights):
            if index == 0:
                shifted = x
            else:
                shifted = torch.cat([first.expand(-1, index, -1), x[:, :-index]], dim=1)
            parts.append(weight * shifted)
        return torch.stack(parts, dim=0).sum(dim=0)


class NormalizedCausalDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.smoothing = CausalSmoothing(kernel_size)

    def forward(self, x):
        trend = self.smoothing(x)
        fluctuation = x - trend
        return trend, fluctuation


class LocalCausalBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size, dilation, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, dilation=dilation)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        batch, length, channels, hidden = x.shape
        residual = x
        y = x.permute(0, 2, 3, 1).reshape(batch * channels, hidden, length)
        pad = (self.kernel_size - 1) * self.dilation
        y = F.pad(y, (pad, 0), mode="replicate")
        y = self.conv(y)
        y = y.transpose(1, 2).reshape(batch, channels, length, hidden).permute(0, 2, 1, 3)
        y = self.proj(y)
        y = self.dropout(self.act(y))
        return self.norm(residual + y)


class LocalDynamicPath(nn.Module):
    def __init__(self, hidden_dim, kernel_size, dilations, dropout):
        super().__init__()
        self.embedding = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList(
            [LocalCausalBlock(hidden_dim, kernel_size, dilation, dropout) for dilation in dilations]
        )

    def forward(self, x):
        y = self.embedding(x.unsqueeze(-1))
        for block in self.blocks:
            y = block(y)
        return y


class FactorizedTemporalVariableBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_ratio, dropout):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.variable_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, channels, hidden = x.shape
        temporal_input = x.permute(0, 2, 1, 3).reshape(batch * channels, length, hidden)
        temporal, _ = self.temporal_attn(temporal_input, temporal_input, temporal_input, need_weights=False)
        temporal = temporal.reshape(batch, channels, length, hidden).permute(0, 2, 1, 3)
        variable_input = x.reshape(batch * length, channels, hidden)
        variable, _ = self.variable_attn(variable_input, variable_input, variable_input, need_weights=False)
        variable = variable.reshape(batch, length, channels, hidden)
        y = self.norm1(x + self.dropout(temporal + variable))
        return self.norm2(y + self.ffn(y))


class GlobalStructuralPath(nn.Module):
    def __init__(self, hidden_dim, depth, num_heads, ffn_ratio, dropout):
        super().__init__()
        self.embedding = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList(
            [FactorizedTemporalVariableBlock(hidden_dim, num_heads, ffn_ratio, dropout) for _ in range(depth)]
        )

    def forward(self, x):
        y = self.embedding(x.unsqueeze(-1))
        for block in self.blocks:
            y = block(y)
        return y


class IntraScaleFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.local_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, local, global_, return_gate=False):
        local_summary = local.mean(dim=(1, 2))
        global_summary = global_.mean(dim=(1, 2))
        beta = torch.sigmoid(self.gate(torch.cat([local_summary, global_summary], dim=-1)))
        beta = beta[:, None, None, :]
        y = beta * self.local_proj(local) + (1.0 - beta) * self.global_proj(global_)
        y = self.norm(y)
        if return_gate:
            return y, beta.squeeze(1).squeeze(1)
        return y


class CrossScaleFusion(nn.Module):
    def __init__(self, hidden_dim, num_scales):
        super().__init__()
        self.num_scales = num_scales
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales),
        )

    def forward(self, features, target_length, fixed=False):
        aligned = []
        descriptors = []
        for feature in features:
            if feature.size(1) != target_length:
                batch, length, channels, hidden = feature.shape
                y = feature.permute(0, 2, 3, 1).reshape(batch, channels * hidden, length)
                y = F.interpolate(y, size=target_length, mode="linear", align_corners=False)
                y = y.reshape(batch, channels, hidden, target_length).permute(0, 3, 1, 2)
            else:
                y = feature
            aligned.append(y)
            descriptors.append(y.mean(dim=(1, 2)))
        if fixed:
            alpha = descriptors[0].new_full((descriptors[0].size(0), self.num_scales), 1.0 / self.num_scales)
        else:
            alpha = torch.softmax(self.gate(torch.cat(descriptors, dim=-1)), dim=-1)
        fused = 0.0
        for index, feature in enumerate(aligned):
            fused = fused + alpha[:, index, None, None, None] * feature
        return fused, alpha


class PredictionHead(nn.Module):
    def __init__(self, channels, hidden_dim, pred_len, dropout):
        super().__init__()
        self.pred_len = pred_len
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )

    def forward(self, x):
        last = x[:, -1]
        pooled = x.mean(dim=1)
        y = self.net(torch.cat([last, pooled], dim=-1))
        return y.transpose(1, 2)
