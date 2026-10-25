"""DLinear (Zeng et al., 2023): moving-average decomposition + two shared linear maps over time.

trend = MovingAvg_25(x) (replicate end-padding), seasonal = x - trend,
y = Linear_s(seasonal) + Linear_t(trend), both Linear(L -> H) shared across variables
(individual=False). No instance normalisation (as in the original model).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinear(nn.Module):
    def __init__(self, seq_len: int = 336, pred_len: int = 96, n_vars: int = 7, moving_avg: int = 25):
        super().__init__()
        assert moving_avg % 2 == 1
        self.L, self.H, self.M, self.ma = seq_len, pred_len, n_vars, moving_avg
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def decompose(self, x: torch.Tensor):
        """x [B, L, M] -> (seasonal, trend), each [B, L, M]."""
        pad = (self.ma - 1) // 2
        xt = x.permute(0, 2, 1)                                     # [B, M, L]
        xt = F.pad(xt, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(xt, kernel_size=self.ma, stride=1).permute(0, 2, 1)
        return x - trend, trend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decompose(x)
        y = self.linear_seasonal(seasonal.permute(0, 2, 1)) + self.linear_trend(trend.permute(0, 2, 1))
        return y.permute(0, 2, 1)                                   # [B, H, M]
