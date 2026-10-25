"""Autograd-based effective receptive field (ERF) measurement - see research_protocol.md Sec. 6.

Methodological adaptation (the original Lab-4 procedure is unavailable; no compatibility claimed).
Conventions common to every model: eval() mode, fp32, RevIN statistics detached, the same 50
deterministic validation windows, lag tau = L-1-t (tau = 0 is the newest input step).

Target A (backbone, paper-style): s_A = sum ReLU(z[..., p]) for the final feature map z at position p
  (centre or last); g = ReLU(d s_A / d x); a_A(tau) = mean_samples sum_vars g[., L-1-tau].
Target B (forecast ERF): Hutchinson probes u~Rademacher^{H x M}; v = d(u . y_hat)/dx;
  E_u[v^2] = sum_{h,m'} J^2, so b(tau) = sqrt(mean_samples sum_vars E_u[v^2][., L-1-tau]).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .data import WindowDataset


def select_samples(ds: WindowDataset, n: int = 50) -> torch.Tensor:
    """n deterministic, evenly spaced windows: i_j = floor(j*(n_val-1)/(n-1)). Returns inputs [n, L, M]."""
    N = len(ds)
    idx = torch.tensor([int(np.floor(j * (N - 1) / (n - 1))) for j in range(n)])
    x, _ = ds.get(idx)
    return x


def _to_lag(per_time: torch.Tensor) -> torch.Tensor:
    """[..., L] indexed by time t -> indexed by lag tau = L-1-t."""
    return torch.flip(per_time, dims=[-1])


def _position(size: int, which: str) -> int:
    return size // 2 if which == "center" else size - 1


def erf_target_a(model: torch.nn.Module, x: torch.Tensor, position: str = "center",
                 chunk: int = 25) -> Dict:
    """Backbone ERF. x [S, L, M] standardised inputs. Returns lag profile (numpy [L]) and the target index."""
    was_training = model.training
    model.eval()
    acc, count, pos = None, 0, None
    try:
        for s in range(0, x.shape[0], chunk):
            xb = x[s:s + chunk].detach().clone().requires_grad_(True)
            z = model.forward_features(xb)
            pos = _position(z.shape[-1], position)
            score = torch.relu(z[..., pos]).sum()
            g = torch.autograd.grad(score, xb)[0]                   # [b, L, M]
            g = torch.relu(g).sum(dim=2)                            # sum over variables -> [b, L] (time order)
            acc = g.sum(0) if acc is None else acc + g.sum(0)
            count += xb.shape[0]
    finally:
        model.train(was_training)
    prof = _to_lag(acc / count).cpu().numpy()
    return {"profile": prof, "position": int(pos), "n_positions": int(z.shape[-1])}


def erf_target_b(model: torch.nn.Module, x: torch.Tensor, probes: int = 16, seed: int = 0,
                 chunk: int = 25) -> np.ndarray:
    """Forecast ERF b(tau) via Rademacher probes. x [S, L, M]. Returns numpy [L]."""
    was_training = model.training
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    energy, count = None, 0
    try:
        for s in range(0, x.shape[0], chunk):
            xb = x[s:s + chunk].detach().clone().requires_grad_(True)
            y = model(xb)                                           # [b, H, M]
            e = torch.zeros_like(xb)
            for r in range(probes):
                u = (torch.randint(0, 2, y.shape, generator=gen) * 2 - 1).to(y.dtype).to(y.device)
                v = torch.autograd.grad((u * y).sum(), xb, retain_graph=(r < probes - 1))[0]
                e += v ** 2
            e = (e / probes).sum(dim=2)                             # sum over variables -> [b, L]
            energy = e.sum(0) if energy is None else energy + e.sum(0)
            count += xb.shape[0]
    finally:
        model.train(was_training)
    return _to_lag(torch.sqrt(energy / count)).cpu().numpy()


def exact_forecast_erf(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Exact b(tau) from the full Jacobian (small models / tests / DLinear). x [S, L, M]."""
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for i in range(x.shape[0]):
            J = torch.autograd.functional.jacobian(lambda inp: model(inp[None])[0], x[i])   # [H, M, L, M]
            total = total + (J ** 2).sum(dim=(0, 1, 3))             # sum over outputs and input vars -> [L]
        total = total / x.shape[0]
    finally:
        model.train(was_training)
    return _to_lag(torch.sqrt(total)).cpu().numpy()


def erf_metrics(profile: np.ndarray, dt_hours: Optional[float] = None) -> Dict:
    """Width metrics of a non-negative lag profile (index = lag in raw steps)."""
    a = np.asarray(profile, dtype=np.float64)
    tot = a.sum()
    if tot <= 0 or not np.isfinite(tot):
        return {"r50": None, "r90": None, "r95": None, "support": 0, "mass_last24": None, "mass_last96": None,
                "max": float(a.max()) if a.size else 0.0}
    cum = np.cumsum(a) / tot
    r = lambda q: int(np.searchsorted(cum, q, side="left"))
    out = {"r50": r(0.5), "r90": r(0.9), "r95": r(0.95), "support": int((a >= 0.01 * a.max()).sum()),
           "mass_last24": float(a[:24].sum() / tot), "mass_last96": float(a[:96].sum() / tot),
           "max": float(a.max())}
    if dt_hours is not None:
        out.update({"r90_hours": out["r90"] * dt_hours, "r95_hours": out["r95"] * dt_hours})
    return out


def measure_erf(model: torch.nn.Module, x: torch.Tensor, probes: int = 16, seed: int = 0,
                has_backbone: bool = True, dt_hours: Optional[float] = None) -> Dict:
    """All ERF quantities for one trained model on the fixed validation samples."""
    res = {"B": erf_target_b(model, x, probes, seed)}
    if has_backbone and hasattr(model, "forward_features"):
        for which in ("center", "last"):
            r = erf_target_a(model, x, which)
            res[f"A_{which}"], res[f"A_{which}_pos"] = r["profile"], r["position"]
    res["metrics"] = {k: erf_metrics(v, dt_hours) for k, v in res.items() if isinstance(v, np.ndarray)}
    return res
