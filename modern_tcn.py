"""ModernTCN-style forecaster (independent implementation from the paper's description).

Tensor layout (B batch, M variables, D feature channels, N tokens, C = M*D):
  x [B, L, M] --RevIN--> [B, L, M] -> [B, M, L] --pad last value (P-S)--> [B, M, L+P-S]
  --stem Conv1d(1->D, kernel P, stride S) + BN--> [B*M, D, N] -> [B, C, N]   (channel = m*D + d)
  --K blocks--> [B, C, N] -> [B, M, D, N] --flatten--> [B, M, D*N] --Linear(D*N -> H)--> [B, M, H]
  -> [B, H, M] --RevIN^-1--> forecast.
Block: (DWConv_k+BN) + (DWConv_small+BN) -> BN(D) -> ConvFFN1 (groups=M) -> ConvFFN2 (groups=D) -> + residual.
The two depthwise branches are kept unmerged (structural re-parameterisation is a train/inference
equivalence that is not needed here).
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .revin import RevIN


class ModernTCNBlock(nn.Module):
    def __init__(self, n_vars: int, d_model: int, k: int, small: int, ffn_ratio: int, dropout: float):
        super().__init__()
        assert k % 2 == 1 and small % 2 == 1, "odd kernels keep the token length unchanged ('same' padding)"
        M, D = n_vars, d_model
        C, dff = M * D, ffn_ratio * D
        self.M, self.D = M, D
        # time mixing only: one filter per (variable, feature) channel
        self.dw_large = nn.Conv1d(C, C, k, padding=k // 2, groups=C, bias=False)
        self.bn_large = nn.BatchNorm1d(C)
        self.dw_small = nn.Conv1d(C, C, small, padding=small // 2, groups=C, bias=False)
        self.bn_small = nn.BatchNorm1d(C)
        self.norm = nn.BatchNorm1d(D)
        # ConvFFN1: feature mixing *within each variable* (groups = M)
        self.ffn1_a = nn.Conv1d(C, M * dff, 1, groups=M)
        self.ffn1_b = nn.Conv1d(M * dff, C, 1, groups=M)
        # ConvFFN2: variable mixing *at each feature index* (groups = D)
        self.ffn2_a = nn.Conv1d(C, M * dff, 1, groups=D)
        self.ffn2_b = nn.Conv1d(M * dff, C, 1, groups=D)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: [B, C, N]
        B, C, N = x.shape
        M, D = self.M, self.D
        h = self.bn_large(self.dw_large(x)) + self.bn_small(self.dw_small(x))
        h = self.norm(h.reshape(B * M, D, N)).reshape(B, C, N)
        h = self.drop(self.ffn1_b(self.drop(F.gelu(self.ffn1_a(h)))))
        # [B, M, D, N] -> channel index d*M + m so that groups=D groups by feature index
        h = h.reshape(B, M, D, N).permute(0, 2, 1, 3).reshape(B, D * M, N)
        h = self.drop(self.ffn2_b(self.drop(F.gelu(self.ffn2_a(h)))))
        h = h.reshape(B, D, M, N).permute(0, 2, 1, 3).reshape(B, C, N)
        return x + h


class ModernTCN(nn.Module):
    def __init__(self, seq_len: int = 336, pred_len: int = 96, n_vars: int = 7, d_model: int = 64,
                 patch_size: int = 8, patch_stride: int = 4, n_blocks: int = 1, ffn_ratio: int = 1,
                 large_kernel: int = 51, small_kernel: Optional[int] = None, dropout: float = 0.3,
                 head_dropout: float = 0.0, revin: bool = True, revin_affine: bool = False):
        super().__init__()
        P, S = patch_size, patch_stride
        assert seq_len % S == 0, "L must be divisible by the patch stride"
        self.L, self.H, self.M, self.D, self.P, self.S = seq_len, pred_len, n_vars, d_model, P, S
        self.N = seq_len // S
        self.k = large_kernel
        self.small = min(5, large_kernel) if small_kernel is None else small_kernel
        self.revin = RevIN(n_vars, affine=revin_affine) if revin else None
        self.stem = nn.Conv1d(1, d_model, P, stride=S)
        self.stem_bn = nn.BatchNorm1d(d_model)
        self.blocks = nn.ModuleList(
            [ModernTCNBlock(n_vars, d_model, large_kernel, self.small, ffn_ratio, dropout) for _ in range(n_blocks)])
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(d_model * self.N, pred_len)

    def _encode(self, x: torch.Tensor):
        """x [B, L, M] -> (features [B, M, D, N], revin stats)."""
        stats = None
        if self.revin is not None:
            x, stats = self.revin.normalize(x)
        B = x.shape[0]
        x = x.permute(0, 2, 1)                                      # [B, M, L]
        x = F.pad(x, (0, self.P - self.S), mode="replicate")        # repeat the last value
        x = x.reshape(B * self.M, 1, -1)
        x = self.stem_bn(self.stem(x))                              # [B*M, D, N]
        x = x.reshape(B, self.M * self.D, self.N)
        for blk in self.blocks:
            x = blk(x)
        return x.reshape(B, self.M, self.D, self.N), stats

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone output before the head (used for ERF Target A): [B, M, D, N]."""
        return self._encode(x)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, stats = self._encode(x)
        B = z.shape[0]
        y = self.head(self.head_drop(z.reshape(B, self.M, self.D * self.N)))   # [B, M, H]
        y = y.permute(0, 2, 1)                                      # [B, H, M]
        if self.revin is not None:
            y = self.revin.denormalize(y, stats)
        return y
