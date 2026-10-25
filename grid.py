"""Experiment grids (phases P1, P3, P4, P6) generated from the pre-registered protocol.
Hyperparameters for P3/P4/P6 come from configs/frozen_v1.json (written after P1)."""
from __future__ import annotations

from typing import Dict, List

from .train import default_config

SETTINGS = [("ETTh1", 96), ("ETTh1", 336), ("ETTm1", 96), ("ETTm1", 336)]
SEEDS = (2021, 2022, 2023)
HP_SEED = 2021
KERNELS = (3, 13, 31, 51, 83)
ABLATION_EXTRA = (3, 13, 31, 83)          # k=51 comes from P3
MODELS = ("tcn", "modern_tcn", "dlinear")
P1_GRID = {"modern_tcn": [{"lr": lr, "dropout": do} for lr in (1e-4, 5e-4) for do in (0.1, 0.3)],
           "tcn": [{"lr": lr} for lr in (1e-3, 3e-4)],
           "dlinear": [{"lr": lr} for lr in (5e-3, 1e-3)]}
P1_GRID_ETTM1_REDUCED = [{"lr": 1e-4, "dropout": 0.3}, {"lr": 5e-4, "dropout": 0.3}]


def hp_key(model: str, dataset: str, H: int) -> str:
    return f"{model}|{dataset}|{H}"


def reference_hp(model: str) -> Dict:
    c = default_config("ETTh1", 96, model)
    return {"lr": c["lr"], "dropout": c["dropout"]} if model != "dlinear" else {"lr": c["lr"]}


def p1_configs(ettm1_reduced: bool = False, **common) -> List[Dict]:
    out = []
    for ds, H in SETTINGS:
        for model in MODELS:
            grid = P1_GRID[model]
            if model == "modern_tcn" and ds == "ETTm1" and ettm1_reduced:
                grid = P1_GRID_ETTM1_REDUCED
            for hp in grid:
                out.append(default_config(ds, H, model, 51, HP_SEED, phase="p1", **hp, **common))
    return out


def _hp(frozen: Dict, model: str, ds: str, H: int) -> Dict:
    return dict(frozen["hparams"][hp_key(model, ds, H)])


def _keep(ds, model, only_dataset, only_model) -> bool:
    return (only_dataset is None or ds == only_dataset) and (only_model is None or model == only_model)


def p3_configs(frozen: Dict, only_dataset=None, only_model=None, **common) -> List[Dict]:
    return [default_config(ds, H, m, 51, s, phase="p3", **_hp(frozen, m, ds, H), **common)
            for ds, H in SETTINGS for m in MODELS for s in SEEDS if _keep(ds, m, only_dataset, only_model)]


def p4_configs(frozen: Dict, tier: str = "full", only_dataset=None, only_model=None, **common) -> List[Dict]:
    """tier: 'h1' (ETTh1 only), 'full' (all settings), 'minimal' (ETTh1 full + ETTm1 H=96 with k in {31, 83})."""
    out = []
    for ds, H in SETTINGS:
        if not _keep(ds, "modern_tcn", only_dataset, only_model):
            continue
        for k in ABLATION_EXTRA:
            if ds == "ETTh1" or tier == "full":
                pass
            elif tier == "minimal" and ds == "ETTm1" and H == 96 and k in (31, 83):
                pass
            else:
                continue
            for s in SEEDS:
                out.append(default_config(ds, H, "modern_tcn", k, s, phase="p4",
                                          **_hp(frozen, "modern_tcn", ds, H), **common))
    if tier == "h1":
        out = [c for c in out if c["dataset"] == "ETTh1"]
    return out


def p6_norevin(frozen: Dict, dataset: str, H: int, **common) -> List[Dict]:
    return [default_config(dataset, H, "modern_tcn", 51, s, phase="p6", revin=False,
                           **_hp(frozen, "modern_tcn", dataset, H), **common) for s in SEEDS]
