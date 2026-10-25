"""Measure ERF (Targets A and B) for finished runs on the 50 fixed validation windows; saves results/erf/<run_id>.npz."""
import argparse, glob, json, os
import numpy as np, torch
import _bootstrap  # noqa
from src.data import make_splits
from src.models import build_model
from src.receptive_field import select_samples, measure_erf

DT_HOURS = {"ETTh1": 1.0, "ETTm1": 0.25}


def erf_for_run(rd: str, data_dir: str, device: str = "cpu", out_dir: str = "results/erf", probes: int = 16):
    cfg = json.load(open(os.path.join(rd, "config.json")))
    d = make_splits(data_dir, cfg["dataset"], cfg["seq_len"], cfg["pred_len"], device)
    x = select_samples(d.val, 50)
    model = build_model(cfg).to(device)
    ck = torch.load(os.path.join(rd, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    res = measure_erf(model, x, probes=probes, has_backbone=cfg["model"] != "dlinear", dt_hours=DT_HOURS[cfg["dataset"]])
    os.makedirs(out_dir, exist_ok=True)
    rid = os.path.basename(rd)
    arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    np.savez(os.path.join(out_dir, rid + ".npz"), **arrays, metrics=json.dumps(res["metrics"]),
             meta=json.dumps({k: cfg[k] for k in ("dataset", "pred_len", "model", "large_kernel", "seed", "n_blocks")}),
             positions=json.dumps({k: int(v) for k, v in res.items() if k.endswith("_pos")}))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probes", type=int, default=16)
    a = ap.parse_args(argv)
    for rd in sorted(glob.glob(os.path.join(a.out, "runs", "*"))):
        tgt = os.path.join(a.out, "erf", os.path.basename(rd) + ".npz")
        if os.path.exists(os.path.join(rd, "DONE")) and not os.path.exists(tgt):
            print("ERF", os.path.basename(rd))
            erf_for_run(rd, a.data_dir, a.device, os.path.join(a.out, "erf"), a.probes)


if __name__ == "__main__":
    main()
