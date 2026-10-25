"""Temporal Convolutional Network baseline, written from Bai, Kolter & Koltun (2018).

Causal dilated residual blocks: per level two weight-normalised causal convolutions with
dilation 2^i, ReLU and dropout, plus a 1x1 projection on the residual when channel counts differ.
Forecast head: the last time step's feature vector -> Linear(C -> M*H).
Receptive field = 1 + 2*(k-1)*(2^levels - 1).
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from .revin import RevIN


class CausalConv1d(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, dilation: int):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(cin, cout, k, dilation=dilation))

    def forward(self, x):                                           # [B, C, T]
        return self.conv(F.pad(x, (self.pad, 0)))                   # left zero-pad only -> causal


class TemporalBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.c1, self.c2 = CausalConv1d(cin, cout, k, dilation), CausalConv1d(cout, cout, k, dilation)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        h = self.drop(F.relu(self.c1(x)))
        h = self.drop(F.relu(self.c2(h)))
        r = x if self.down is None else self.down(x)
        return F.relu(h + r)


class TCN(nn.Module):
    def __init__(self, seq_len: int = 336, pred_len: int = 96, n_vars: int = 7, channels: int = 64,
                 levels: int = 7, kernel_size: int = 3, dropout: float = 0.1, revin: bool = True):
        super().__init__()
        self.L, self.H, self.M, self.C = seq_len, pred_len, n_vars, channels
        self.levels, self.kernel_size = levels, kernel_size
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** levels - 1)
        self.revin = RevIN(n_vars) if revin else None
        blocks, cin = [], n_vars
        for i in range(levels):
            blocks.append(TemporalBlock(cin, channels, kernel_size, 2 ** i, dropout))
            cin = channels
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, n_vars * pred_len)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, L, M] -> backbone features [B, C, L] (before the head), after RevIN."""
        if self.revin is not None:
            x, _ = self.revin.normalize(x)
        return self.net(x.permute(0, 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = None
        if self.revin is not None:
            x, stats = self.revin.normalize(x)
        h = self.net(x.permute(0, 2, 1))                            # [B, C, L]
        y = self.head(h[:, :, -1]).reshape(x.shape[0], self.M, self.H).permute(0, 2, 1)   # [B, H, M]
        if self.revin is not None:
            y = self.revin.denormalize(y, stats)
        return y
