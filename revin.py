"""Reversible instance normalisation (per-sample, per-variable, over the time axis).

Statistics are detach()-ed (as in the reference ModernTCN code) so that gradients do not
flow through the instance mean/std. The module is functional: it returns the statistics
instead of storing them, so repeated/batched/ERF calls cannot interfere with each other.
"""
from typing import Tuple

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps, self.affine = eps, affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """x: [B, L, M] -> normalised [B, L, M] and stats ([B,1,M], [B,1,M])."""
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        xn = (x - mean) / std
        if self.affine:
            xn = xn * self.weight + self.bias
        return xn, (mean, std)

    def denormalize(self, y: torch.Tensor, stats: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """y: [B, H, M] -> original scale."""
        mean, std = stats
        if self.affine:
            y = (y - self.bias) / (self.weight + self.eps * self.eps)
        return y * std + mean
