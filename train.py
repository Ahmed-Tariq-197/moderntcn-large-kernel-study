"""Resumable training for one run + validation-only bookkeeping + guarded one-shot test evaluation.

Run layout (results/runs/<run_id>/):
  config.json  train_log.csv  last.pt  best.pt  val_metrics.json  DONE  [FAILED]
Test metrics live in <run_dir>/test_metrics.json and results/test_runs.csv and can only be
written by `evaluate_test`, once per run, after configs/frozen_v1.json exists.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch

from .data import make_splits, ETTSplits
from .evaluate import evaluate_model
from .models import build_model, count_params

LEDGER_FIELDS = ["run_id", "dataset", "pred_len", "model", "large_kernel", "n_blocks", "ffn_ratio", "seed",
                 "params", "lr", "dropout", "head_dropout", "batch_size", "max_epochs", "patience", "revin",
                 "phase", "best_epoch", "epochs_run", "best_val_mse", "best_val_mae", "sec_per_epoch",
                 "peak_mem_mb", "device", "torch_version", "git_commit", "status", "cfg_hash"]
TEST_FIELDS = ["run_id", "dataset", "pred_len", "model", "large_kernel", "seed", "params", "test_mse", "test_mae",
               "n_test_windows", "best_epoch", "cfg_hash"]
NON_HASH_KEYS = {"device", "data_dir", "out_dir", "phase", "stop_after_epoch"}

ETT_DEFAULTS = {"ETTh1": {"n_blocks": 1, "ffn_ratio": 1}, "ETTm1": {"n_blocks": 3, "ffn_ratio": 8}}


# ------------------------------------------------------------------------------------------ config
def default_config(dataset: str, pred_len: int, model: str, large_kernel: int = 51, seed: int = 2021, **over) -> Dict:
    cfg = {
        "dataset": dataset, "pred_len": pred_len, "seq_len": 336, "model": model, "seed": seed,
        "d_model": 64, "large_kernel": large_kernel if model == "modern_tcn" else 0,
        "n_blocks": ETT_DEFAULTS[dataset]["n_blocks"] if model == "modern_tcn" else 0,
        "ffn_ratio": ETT_DEFAULTS[dataset]["ffn_ratio"] if model == "modern_tcn" else 0,
        "dropout": {"modern_tcn": 0.3, "tcn": 0.1, "dlinear": 0.0}[model],
        "head_dropout": (0.1 if (dataset == "ETTm1" and pred_len == 336) else 0.0) if model == "modern_tcn" else 0.0,
        "lr": {"modern_tcn": 1e-4, "tcn": 1e-3, "dlinear": 5e-3}[model],
        "lr_schedule": "type1" if model == "dlinear" else "type3",
        "batch_size": 32 if model == "dlinear" else 512,
        "max_epochs": 100, "patience": 10, "revin": model != "dlinear",
        "train_limit": None, "val_limit": None, "eval_batch_size": 1024, "micro_batch": None,
        "phase": "", "device": "cpu", "data_dir": "", "out_dir": "results",
    }
    if model == "modern_tcn" and dataset == "ETTm1":
        # K=3, r=8: ~28 MiB of saved activations per sample -> ~14 GiB at batch 512 (measured with autograd hooks);
        # gradient accumulation over micro-batches keeps the effective batch at 512 on a 15 GB GPU. Same for every k/seed.
        cfg["micro_batch"], cfg["eval_batch_size"] = 128, 512
    cfg.update(over)
    return cfg


def canonical_hash(cfg: Dict) -> str:
    d = {k: v for k, v in sorted(cfg.items()) if k not in NON_HASH_KEYS}
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_id(cfg: Dict) -> str:
    k = f"_k{cfg['large_kernel']}" if cfg["model"] == "modern_tcn" else ""
    tag = "" if cfg.get("revin", True) or cfg["model"] == "dlinear" else "_norevin"
    return f"{cfg['dataset']}_H{cfg['pred_len']}_{cfg['model']}{k}{tag}_s{cfg['seed']}_{canonical_hash(cfg)[:8]}"


def lr_at(epoch: int, base: float, mode: str) -> float:
    """1-based epoch. type1 halves every epoch; type3 keeps base for epochs 1-2 then x0.9 per epoch (reference recipes)."""
    if mode == "type1":
        return base * (0.5 ** (epoch - 1))
    if mode == "type3":
        return base if epoch < 3 else base * (0.9 ** (epoch - 3))
    if mode == "constant":
        return base
    raise ValueError(mode)


def git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------------------------------ utils
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False      # strict fp32 (protocol Sec. 8); A100 would otherwise use TF32 for convs
    torch.backends.cudnn.allow_tf32 = False


def _rng_state() -> Dict:
    s = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_state(s: Dict) -> None:
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def append_csv(path: str, fields, row: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def train_step(model, opt, x: torch.Tensor, y: torch.Tensor, micro_batch: Optional[int] = None) -> Optional[float]:
    """One optimiser step on batch (x, y); gradients are accumulated over micro-batches (exactly equal to the full-batch
    gradient for models without batch statistics; BatchNorm statistics are computed per micro-batch).
    Returns the batch-mean loss, or None if it is not finite (no step is taken)."""
    B = x.shape[0]
    mb = B if not micro_batch else min(micro_batch, B)
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for s in range(0, B, mb):
        xs, ys = x[s:s + mb], y[s:s + mb]
        loss = torch.mean((model(xs) - ys) ** 2) * (len(xs) / B)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            return None
        loss.backward()
        total += float(loss.detach())
    opt.step()
    return total


def run_dir_of(cfg: Dict) -> str:
    return os.path.join(cfg["out_dir"], "runs", run_id(cfg))


def get_splits(cfg: Dict, array: Optional[np.ndarray] = None) -> ETTSplits:
    return make_splits(cfg["data_dir"], cfg["dataset"], cfg["seq_len"], cfg["pred_len"], cfg["device"],
                       train_limit=cfg.get("train_limit"), val_limit=cfg.get("val_limit"), array=array)


# ------------------------------------------------------------------------------------------ training
def train_run(cfg: Dict, array: Optional[np.ndarray] = None, resume: bool = True, verbose: bool = True) -> Dict:
    """Train one run to completion (or until `stop_after_epoch`, used to simulate interruptions).
    Never reads the test split. Returns the final status dict."""
    rd = run_dir_of(cfg)
    os.makedirs(rd, exist_ok=True)
    if os.path.exists(os.path.join(rd, "DONE")):
        return json.load(open(os.path.join(rd, "val_metrics.json")))
    cfg_path = os.path.join(rd, "config.json")
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        if canonical_hash(old) != canonical_hash(cfg):
            raise RuntimeError(f"config hash mismatch in {rd}; refusing to mix runs")
    json.dump(cfg, open(cfg_path, "w"), indent=2, sort_keys=True)

    device = torch.device(cfg["device"])
    seed_everything(cfg["seed"])
    data = get_splits(cfg, array)
    model = build_model(cfg).to(device)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    start_epoch, best_val, best_epoch, bad, rows = 1, float("inf"), 0, 0, []
    pre_stopped = False
    last_pt = os.path.join(rd, "last.pt")
    if resume and os.path.exists(last_pt):
        ck = torch.load(last_pt, map_location="cpu", weights_only=False)   # CPU: RNG states must be CPU ByteTensors
        if ck["cfg_hash"] != canonical_hash(cfg):
            raise RuntimeError("last.pt belongs to a different config")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, best_val, best_epoch, bad = ck["epoch"] + 1, ck["best_val"], ck["best_epoch"], ck["bad"]
        _set_rng_state(ck["rng"])
        pre_stopped = bad >= cfg["patience"]            # crashed after the stopping condition was reached: do not train more
        if verbose:
            print(f"[resume] {run_id(cfg)} from epoch {start_epoch}")

    log_path = os.path.join(rd, "train_log.csv")
    if start_epoch == 1 and os.path.exists(log_path):
        os.remove(log_path)
    status, t_epochs = "running", []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stop_after = cfg.get("stop_after_epoch")
    epoch = start_epoch - 1

    for epoch in range(start_epoch, (start_epoch - 1 if pre_stopped else cfg["max_epochs"]) + 1):
        lr = lr_at(epoch, cfg["lr"], cfg["lr_schedule"])
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        model.train()
        tot, cnt = 0.0, 0
        for x, y in data.train.batches(cfg["batch_size"], shuffle=True, seed=cfg["seed"] * 100003 + epoch):
            loss = train_step(model, opt, x, y, cfg.get("micro_batch"))
            if loss is None:
                status = "failed_nonfinite_loss"
                break
            tot += loss * len(x)
            cnt += len(x)
        if status.startswith("failed"):
            break
        val = evaluate_model(model, data.val, cfg["eval_batch_size"])
        if not val["finite"]:
            status = "failed_nonfinite_val"
            break
        dt = time.time() - t0
        t_epochs.append(dt)
        improved = val["mse"] < best_val
        if improved:
            best_val, best_epoch, bad = val["mse"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, os.path.join(rd, "best.pt"))
            best_val_mae = val["mae"]
        else:
            bad += 1
        append_csv(log_path, ["epoch", "lr", "train_mse", "val_mse", "val_mae", "sec", "best"],
                   {"epoch": epoch, "lr": lr, "train_mse": tot / max(cnt, 1), "val_mse": val["mse"],
                    "val_mae": val["mae"], "sec": dt, "best": int(improved)})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best_val": best_val,
                    "best_epoch": best_epoch, "bad": bad, "rng": _rng_state(), "cfg_hash": canonical_hash(cfg)},
                   last_pt)
        if verbose:
            print(f"[{run_id(cfg)}] ep {epoch:3d} lr {lr:.2e} train {tot / max(cnt, 1):.4f} val {val['mse']:.4f} "
                  f"({dt:.1f}s){' *' if improved else ''}")
        if bad >= cfg["patience"]:
            status = "early_stopped"
            break
        if stop_after is not None and epoch >= stop_after:
            return {"status": "interrupted", "epoch": epoch}
    else:
        status = "early_stopped" if pre_stopped else "max_epochs"

    if status.startswith("failed"):
        open(os.path.join(rd, "FAILED"), "w").write(status)
        return {"status": status, "run_id": run_id(cfg)}

    # reload the best checkpoint to record its validation metrics (detailed diagnostics included)
    ck = torch.load(os.path.join(rd, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    val_detail = evaluate_model(model, data.val, cfg["eval_batch_size"], detail=True)
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    info = {"run_id": run_id(cfg), "status": status, "params": n_params, "best_epoch": best_epoch,
            "epochs_run": epoch, "best_val_mse": val_detail["mse"], "best_val_mae": val_detail["mae"],
            "val_detail": val_detail, "sec_per_epoch": float(np.mean(t_epochs)) if t_epochs else 0.0,
            "peak_mem_mb": peak, "device": str(device), "torch_version": torch.__version__, "precision": "fp32 (tf32 off)",
            "git_commit": git_commit(), "cfg_hash": canonical_hash(cfg)}
    json.dump(info, open(os.path.join(rd, "val_metrics.json"), "w"), indent=2)
    open(os.path.join(rd, "DONE"), "w").write(status)
    row = {**cfg, **info}
    row["status"] = status
    append_csv(os.path.join(cfg["out_dir"], "runs.csv"), LEDGER_FIELDS, row)
    return info


# ------------------------------------------------------------------------------------------ test (P8)
FROZEN_DEFAULT = os.path.join("configs", "frozen_v1.json")


def evaluate_test(cfg: Dict, frozen_path: str = FROZEN_DEFAULT, array: Optional[np.ndarray] = None) -> Dict:
    """One-shot test evaluation of the validation-selected checkpoint. Refuses to run before the
    protocol is frozen and refuses to overwrite an existing test record."""
    if not os.path.exists(frozen_path) or not json.load(open(frozen_path)).get("frozen", False):
        raise PermissionError(f"{frozen_path} missing or not marked frozen: test evaluation is not allowed yet")
    rd = run_dir_of(cfg)
    if not os.path.exists(os.path.join(rd, "DONE")):
        raise FileNotFoundError(f"run not finished: {rd}")
    out_path = os.path.join(rd, "test_metrics.json")
    if os.path.exists(out_path):
        raise FileExistsError(f"test metrics already recorded for {run_id(cfg)}; test is evaluated once per run")
    device = torch.device(cfg["device"])
    data = get_splits(cfg, array)
    model = build_model(cfg).to(device)
    ck = torch.load(os.path.join(rd, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    res = evaluate_model(model, data.test, cfg["eval_batch_size"], detail=True)
    res.update({"run_id": run_id(cfg), "best_epoch": ck["epoch"], "cfg_hash": canonical_hash(cfg)})
    json.dump(res, open(out_path, "w"), indent=2)
    append_csv(os.path.join(cfg["out_dir"], "test_runs.csv"), TEST_FIELDS,
               {**cfg, "run_id": run_id(cfg), "params": count_params(model), "test_mse": res["mse"],
                "test_mae": res["mae"], "n_test_windows": res["n_windows"], "best_epoch": ck["epoch"],
                "cfg_hash": canonical_hash(cfg)})
    return res
