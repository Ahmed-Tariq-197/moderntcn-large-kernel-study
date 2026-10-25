"""Metrics on standardised data. One global aggregation: sum of errors / number of values,
accumulated in float64, so the result does not depend on the batch size."""
from typing import Callable, Dict

import torch

from .data import WindowDataset


@torch.no_grad()
def evaluate_predictor(predict: Callable[[torch.Tensor], torch.Tensor], ds: WindowDataset,
                       batch_size: int = 1024, detail: bool = False) -> Dict:
    se = ae = 0.0
    n_vals = 0
    se_var = torch.zeros(ds.z.shape[1], dtype=torch.float64)
    se_step = torch.zeros(ds.H, dtype=torch.float64)
    for x, y in ds.batches(batch_size, shuffle=False, drop_last=False):
        p = predict(x)
        if not torch.isfinite(p).all():
            return {"mse": float("nan"), "mae": float("nan"), "n_windows": len(ds), "finite": False}
        d = (p - y).double()
        se += float((d ** 2).sum())
        ae += float(d.abs().sum())
        n_vals += d.numel()
        if detail:
            se_var += (d ** 2).sum(dim=(0, 1)).cpu()
            se_step += (d ** 2).sum(dim=(0, 2)).cpu()
    out = {"mse": se / n_vals, "mae": ae / n_vals, "n_windows": len(ds), "n_values": n_vals, "finite": True}
    if detail:
        per_var_count = len(ds) * ds.H
        per_step_count = len(ds) * ds.z.shape[1]
        out["mse_per_variable"] = (se_var / per_var_count).tolist()
        out["mse_per_step"] = (se_step / per_step_count).tolist()
    return out


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, ds: WindowDataset, batch_size: int = 1024, detail: bool = False) -> Dict:
    was_training = model.training
    model.eval()
    try:
        return evaluate_predictor(model, ds, batch_size, detail)
    finally:
        model.train(was_training)


def predict_last_value(H: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Naive reference: repeat the last observed value of each variable."""
    return lambda x: x[:, -1:, :].expand(-1, H, -1)


def predict_train_mean(H: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Naive reference: the training mean, which is 0 in standardised space."""
    return lambda x: torch.zeros(x.shape[0], H, x.shape[2], dtype=x.dtype, device=x.device)
