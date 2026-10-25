"""ETT data pipeline: chronological 12/4/4-month split, train-only z-score, leakage-safe windows.

Conventions (see research_notes/research_protocol.md Sec. 3):
  * rows used: first 14,400 (ETTh1) / 57,600 (ETTm1); the file tail is ignored (benchmark convention)
  * train rows [0, tr); val targets start at row tr; test targets start at row va
  * val/test *inputs* may reach back L rows into the previous split; targets never do
  * z-score statistics (ddof=0) are fitted on the training rows only
  * every window of a split is used for evaluation (no drop_last)
All arithmetic is re-derived here; nothing is imported from any reference loader.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import torch

FREQ_MULT = {"ETTh1": 1, "ETTm1": 4}
N_VARS = 7
FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def borders(dataset: str) -> Tuple[int, int, int]:
    """(train_end, val_end, test_end) in rows (exclusive ends)."""
    m = FREQ_MULT[dataset]
    tr = 12 * 30 * 24 * m
    va = tr + 4 * 30 * 24 * m
    te = va + 4 * 30 * 24 * m
    return tr, va, te


def n_windows(rows: int, L: int, H: int) -> int:
    return max(rows - L - H + 1, 0)


def expected_window_counts(dataset: str, L: int, H: int) -> dict:
    tr, va, te = borders(dataset)
    return {
        "train": n_windows(tr, L, H),
        "val": n_windows(va - (tr - L), L, H),
        "test": n_windows(te - (va - L), L, H),
    }


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(train_rows: np.ndarray) -> "Scaler":
        return Scaler(train_rows.mean(0), train_rows.std(0))  # ddof=0, like sklearn StandardScaler

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class WindowDataset:
    """Sliding windows over rows [start, end) of a standardised array held as one tensor.

    Window i: input rows [start+i, start+i+L), target rows [start+i+L, start+i+L+H).
    Batches are gathered by index arithmetic (no per-item Python overhead); the whole
    array lives on `device`.
    """

    def __init__(self, z: torch.Tensor, start: int, end: int, L: int, H: int, name: str = "",
                 limit: Optional[int] = None):
        assert 0 <= start < end <= z.shape[0], (start, end, z.shape)
        self.z, self.start, self.end, self.L, self.H, self.name = z, start, end, L, H, name
        n = n_windows(end - start, L, H)
        self.n = n if limit is None else min(n, int(limit))
        self._arL = torch.arange(L, device=z.device)
        self._arH = torch.arange(H, device=z.device)

    def __len__(self) -> int:
        return self.n

    def get(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = idx.to(self.z.device).long()
        base = self.start + idx
        x = self.z[base[:, None] + self._arL[None, :]]                    # [B, L, M]
        y = self.z[base[:, None] + self.L + self._arH[None, :]]           # [B, H, M]
        return x, y

    def batches(self, batch_size: int, shuffle: bool = False, seed: int = 0,
                drop_last: bool = False) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if shuffle:
            g = torch.Generator().manual_seed(int(seed))
            order = torch.randperm(self.n, generator=g)
        else:
            order = torch.arange(self.n)
        for s in range(0, self.n, batch_size):
            sel = order[s:s + batch_size]
            if drop_last and len(sel) < batch_size:
                break
            yield self.get(sel)

    # ---- helpers used by tests / audits -------------------------------------------------
    def input_rows(self, i: int) -> Tuple[int, int]:
        a = self.start + i
        return a, a + self.L

    def target_rows(self, i: int) -> Tuple[int, int]:
        a = self.start + i + self.L
        return a, a + self.H


@dataclass
class ETTSplits:
    train: WindowDataset
    val: WindowDataset
    test: WindowDataset
    scaler: Scaler
    z: torch.Tensor
    dataset: str
    L: int
    H: int


def read_ett(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    assert list(df.columns[1:]) == FEATURES, df.columns.tolist()
    arr = df[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("non-finite values in " + csv_path)
    return arr


def make_splits(data_dir: str, dataset: str, L: int = 336, H: int = 96, device: str = "cpu",
                train_limit: Optional[int] = None, val_limit: Optional[int] = None,
                test_limit: Optional[int] = None, array: Optional[np.ndarray] = None) -> ETTSplits:
    """Build train/val/test window datasets. `array` allows tests to inject synthetic data."""
    raw = array if array is not None else read_ett(os.path.join(data_dir, f"{dataset}.csv"))
    tr, va, te = borders(dataset)
    if raw.shape[0] < te:
        raise ValueError(f"{dataset}: need >= {te} rows, got {raw.shape[0]}")
    raw = raw[:te]                                    # the tail beyond test_end is never used
    scaler = Scaler.fit(raw[:tr])                     # TRAIN ROWS ONLY
    z = torch.tensor(scaler.transform(raw), dtype=torch.float32, device=device)
    return ETTSplits(
        train=WindowDataset(z, 0, tr, L, H, "train", train_limit),
        val=WindowDataset(z, tr - L, va, L, H, "val", val_limit),
        test=WindowDataset(z, va - L, te, L, H, "test", test_limit),
        scaler=scaler, z=z, dataset=dataset, L=L, H=H,
    )
