"""Select hyperparameters from the P1 runs (validation MSE only) and write configs/frozen_v1.json."""
import argparse, csv, hashlib, json, os, sys, time
import _bootstrap  # noqa
from src.grid import hp_key, reference_hp, SETTINGS, MODELS
from src.train import git_commit


def select(rows):
    """rows: ledger dicts of phase p1. Returns {key: hp} using the lowest best_val_mse; ties -> reference config."""
    best = {}
    for r in rows:
        if r.get("phase") != "p1" or str(r["status"]).startswith("failed"):
            continue
        key = hp_key(r["model"], r["dataset"], int(r["pred_len"]))
        val = float(r["best_val_mse"])
        hp = {"lr": float(r["lr"])}
        if r["model"] != "dlinear":
            hp["dropout"] = float(r["dropout"])
        ref = reference_hp(r["model"]) == hp
        cur = best.get(key)
        if cur is None or val < cur["val"] - 1e-12 or (abs(val - cur["val"]) <= 1e-12 and ref and not cur["ref"]):
            best[key] = {"val": val, "hp": hp, "ref": ref, "run_id": r["run_id"]}
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="results/runs.csv")
    ap.add_argument("--out", default="configs/frozen_v1.json")
    ap.add_argument("--protocol", default="research_notes/research_protocol.md")
    ap.add_argument("--allow_partial", action="store_true")
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(open(a.ledger)))
    best = select(rows)
    need = [hp_key(m, d, H) for d, H in SETTINGS for m in MODELS]
    missing = [k for k in need if k not in best]
    if missing and not a.allow_partial:
        raise SystemExit(f"P1 incomplete, missing: {missing}")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} already exists: the protocol is frozen once")
    doc = {"frozen": True, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "git_commit": git_commit(),
           "protocol_sha256": hashlib.sha256(open(a.protocol, "rb").read()).hexdigest() if os.path.exists(a.protocol) else None,
           "hparams": {k: v["hp"] for k, v in best.items()},
           "selected_from": {k: {"run_id": v["run_id"], "val_mse": v["val"]} for k, v in best.items()}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(doc, open(a.out, "w"), indent=2, sort_keys=True)
    print("wrote", a.out, "missing:", missing)


if __name__ == "__main__":
    main()
