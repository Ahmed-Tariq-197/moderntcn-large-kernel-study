"""Gate G3: tiny end-to-end run per model: finite outputs, finite loss, loss decreases. Writes logs/smoke_<device>.json.
Never touches the test split."""
import argparse, json, os, shutil, sys, tempfile
import pandas as pd
import _bootstrap  # noqa
from src.train import default_config, train_run


def run_smoke(data_dir, device="cpu", dataset="ETTh1", H=96, epochs=6, train_limit=512, val_limit=256, batch=64,
              models=("modern_tcn", "tcn", "dlinear")):
    out = tempfile.mkdtemp(prefix="smoke_")
    report = {"device": device, "dataset": dataset, "H": H, "models": {}}
    ok = True
    try:
        for m in models:
            lr = {"modern_tcn": 5e-4, "tcn": 1e-3, "dlinear": 5e-3}[m]
            cfg = default_config(dataset, H, m, 51, 2021, data_dir=data_dir, out_dir=out, device=device,
                                 max_epochs=epochs, patience=epochs, train_limit=train_limit, val_limit=val_limit,
                                 batch_size=batch, lr=lr, lr_schedule="constant")
            r = train_run(cfg, verbose=False)
            log = pd.read_csv(os.path.join(out, "runs", r["run_id"], "train_log.csv"))
            dec = bool(log.train_mse.iloc[-1] < log.train_mse.iloc[0])
            fin = bool(log[["train_mse", "val_mse", "val_mae"]].notna().all().all() and r["status"] in ("max_epochs", "early_stopped"))
            report["models"][m] = {"params": r["params"], "train_mse_first": float(log.train_mse.iloc[0]),
                                   "train_mse_last": float(log.train_mse.iloc[-1]), "val_mse_best": r["best_val_mse"],
                                   "sec_per_epoch": r["sec_per_epoch"], "loss_decreased": dec, "finite": fin}
            ok &= dec and fin
    finally:
        shutil.rmtree(out, ignore_errors=True)
    report["passed"] = bool(ok)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--train_limit", type=int, default=512)
    a = ap.parse_args(argv)
    rep = run_smoke(a.data_dir, a.device, a.dataset, a.H, a.epochs, a.train_limit)
    os.makedirs("logs", exist_ok=True)
    json.dump(rep, open(f"logs/smoke_{a.device}.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
