"""Read-only check for TRAIN=False mode: recompute validation and (already recorded) test metrics from the saved
best.pt files and compare with the stored records. Writes nothing to the ledgers."""
import argparse, glob, json, os, sys
import torch
import _bootstrap  # noqa
from src.data import make_splits
from src.evaluate import evaluate_model
from src.models import build_model


def verify(results: str, data_dir: str, device: str = "cpu", tol: float = 1e-4, array=None):
    bad, n = [], 0
    for rd in sorted(glob.glob(os.path.join(results, "runs", "*"))):
        if not os.path.exists(os.path.join(rd, "DONE")):
            continue
        cfg = json.load(open(os.path.join(rd, "config.json")))
        d = make_splits(data_dir, cfg["dataset"], cfg["seq_len"], cfg["pred_len"], device,
                        train_limit=cfg.get("train_limit"), val_limit=cfg.get("val_limit"), array=array)
        m = build_model(cfg).to(device)
        m.load_state_dict(torch.load(os.path.join(rd, "best.pt"), map_location=device, weights_only=False)["model"])
        checks = [("val", d.val, json.load(open(os.path.join(rd, "val_metrics.json")))["best_val_mse"])]
        tp = os.path.join(rd, "test_metrics.json")
        if os.path.exists(tp):
            checks.append(("test", d.test, json.load(open(tp))["mse"]))
        for name, ds, ref in checks:
            got = evaluate_model(m, ds, 1024)["mse"]
            n += 1
            if abs(got - ref) > tol * max(1.0, abs(ref)):
                bad.append((os.path.basename(rd), name, ref, got))
    return n, bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    n, bad = verify(a.results, a.data_dir, a.device)
    print(f"checked {n} metric records; mismatches: {len(bad)}")
    for b in bad:
        print("MISMATCH", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
