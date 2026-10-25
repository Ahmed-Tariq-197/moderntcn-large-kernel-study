"""Gate G4: measure training step time and peak GPU memory per (setting, model); extrapolate seconds/epoch
and total grid cost. Run on the Colab GPU with the default batch (512). On CPU use a small --batch; the
per-sample time is then scaled (approximation, memory not scaled). Writes logs/runtime_<device>.json."""
import argparse, json, math, os, time
import torch
import _bootstrap  # noqa
from src.data import make_splits
from src.grid import SETTINGS
from src.models import build_model
from src.train import default_config, train_step

EPOCHS_ASSUMED = 40


def bench(cfg, data_dir, steps, warmup, batch):
    dev = torch.device(cfg["device"])
    d = make_splits(data_dir, cfg["dataset"], cfg["seq_len"], cfg["pred_len"], cfg["device"], train_limit=batch * (steps + warmup))
    model = build_model(cfg).to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    times = []
    for i, (x, y) in enumerate(d.train.batches(batch, shuffle=False)):
        if len(x) < batch:
            break
        t = time.time()
        train_step(model, opt, x, y, cfg.get("micro_batch"))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t)
    times = times[warmup:]
    n_train = len(make_splits(data_dir, cfg["dataset"], cfg["seq_len"], cfg["pred_len"], "cpu").train)
    per_sample = sum(times) / len(times) / batch
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None
    return {"sec_per_epoch_est": per_sample * n_train, "peak_mem_mb": peak, "n_train": n_train, "batch": batch}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--settings", default="all", help="all | quick (ETTh1 H=96 only)")
    a = ap.parse_args(argv)
    sets = SETTINGS if a.settings == "all" else [("ETTh1", 96)]
    out = {"device": a.device, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "runs": {}}
    for ds, H in sets:
        for m in ("modern_tcn", "tcn", "dlinear"):
            cfg = default_config(ds, H, m, 51, 2021, device=a.device)
            r = bench(cfg, a.data_dir, a.steps, a.warmup, a.batch if m != "dlinear" else min(a.batch, 32) if a.device == "cpu" else 32)
            out["runs"][f"{ds}_H{H}_{m}"] = r
            print(f"{ds} H={H} {m}: {r['sec_per_epoch_est']:.1f} s/epoch (est), peak mem {r['peak_mem_mb']}")
    sec = lambda ds, H, m: out["runs"][f"{ds}_H{H}_{m}"]["sec_per_epoch_est"]
    if a.settings == "all":
        def phase_h(runs):  # runs: list of (ds, H, model, n)
            return sum(sec(ds, H, m) * n for ds, H, m, n in runs) * EPOCHS_ASSUMED / 3600
        keys = SETTINGS
        p1 = [(d, H, "modern_tcn", 4) for d, H in keys] + [(d, H, "tcn", 2) for d, H in keys] + [(d, H, "dlinear", 2) for d, H in keys]
        p3 = [(d, H, m, 3) for d, H in keys for m in ("modern_tcn", "tcn", "dlinear")]
        p4 = lambda ks: [(d, H, "modern_tcn", 3 * len(ks)) for d, H in ks]
        out["estimate_hours"] = {"assumed_epochs_per_run": EPOCHS_ASSUMED, "P1": phase_h(p1), "P3": phase_h(p3),
                                 "P4_ETTh1": phase_h([(d, H, "modern_tcn", 12) for d, H in keys if d == "ETTh1"]),
                                 "P4_ETTm1_full": phase_h([(d, H, "modern_tcn", 12) for d, H in keys if d == "ETTm1"]),
                                 "P4_ETTm1_minimal": phase_h([("ETTm1", 96, "modern_tcn", 6)])}
        print(json.dumps(out["estimate_hours"], indent=2))
    os.makedirs("logs", exist_ok=True)
    json.dump(out, open(f"logs/runtime_{a.device}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
