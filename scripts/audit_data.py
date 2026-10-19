"""Data audit for ETTh1 / ETTm1 (no torch dependency)."""
import argparse
import json
import os
import numpy as np
import pandas as pd

L = 336
HORIZONS = (96, 336)
BATCH_DROP = 512

def borders(freq_mult: int):
    tr = 12 * 30 * 24 * freq_mult
    va = tr + 4 * 30 * 24 * freq_mult
    te = va + 4 * 30 * 24 * freq_mult
    return tr, va, te

def n_windows(n_rows_in_segment: int, L: int, H: int) -> int:
    return n_rows_in_segment - L - H + 1

def audit(path: str, mult: int):
    df = pd.read_csv(path)
    out = {"file": os.path.basename(path), "rows": int(len(df)), "columns": list(df.columns)}
    feats = df.columns[1:]
    out["n_features"] = int(len(feats))
    out["nan_count"] = int(df[feats].isna().sum().sum())
    ts = pd.to_datetime(df["date"])
    out["first_ts"], out["last_ts"] = str(ts.iloc[0]), str(ts.iloc[-1])
    out["monotonic_increasing"] = bool(ts.is_monotonic_increasing)
    out["duplicate_ts"] = int(ts.duplicated().sum())
    d = ts.diff().dropna()
    out["sampling_intervals"] = {str(k): int(v) for k, v in d.value_counts().items()}
    tr, va, te = borders(mult)
    out["border_rows"] = {"train_end": tr, "val_end": va, "test_end": te}
    out["rows_used_by_benchmark"] = te
    out["rows_unused_tail"] = int(len(df) - te)
    out["frac_used"] = round(te / len(df), 4)
    out["ts_at_borders"] = {
        "train_start": str(ts.iloc[0]), "train_last": str(ts.iloc[tr - 1]),
        "val_first_target": str(ts.iloc[tr]), "val_last": str(ts.iloc[va - 1]),
        "test_first_target": str(ts.iloc[va]), "test_last": str(ts.iloc[te - 1]),
    }
    wins = {}
    for H in HORIZONS:
        ntr = n_windows(tr, L, H)
        nva = n_windows(va - (tr - L), L, H)
        nte = n_windows(te - (va - L), L, H)
        dropped = nte % BATCH_DROP
        wins[str(H)] = {
            "train": ntr, "val": nva, "test": nte,
            "test_dropped_by_drop_last_bs512": dropped,
            "test_dropped_pct": round(100 * dropped / nte, 2),
        }
    out["windows"] = wins
    X = df[feats].to_numpy(dtype=np.float64)
    mu, sd = X[:tr].mean(0), X[:tr].std(0)
    Z = (X - mu) / sd
    out["train_mean"] = dict(zip(feats, np.round(mu, 4).tolist()))
    out["train_std"] = dict(zip(feats, np.round(sd, 4).tolist()))
    out["shift_val_mean_in_train_std"] = dict(zip(feats, np.round(Z[tr:va].mean(0), 3).tolist()))
    out["shift_test_mean_in_train_std"] = dict(zip(feats, np.round(Z[va:te].mean(0), 3).tolist()))
    out["val_std_ratio_to_train"] = dict(zip(feats, np.round(Z[tr:va].std(0), 3).tolist()))
    out["test_std_ratio_to_train"] = dict(zip(feats, np.round(Z[va:te].std(0), 3).tolist()))
    out["mse_zero_forecast"] = {"train": float((Z[:tr] ** 2).mean()), "val": float((Z[tr:va] ** 2).mean()), "test": float((Z[va:te] ** 2).mean())}
    return out, df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="logs/data_audit.json")
    a = ap.parse_args()
    res = {}
    h, dfh = audit(os.path.join(a.data_dir, "ETTh1.csv"), 1)
    m, dfm = audit(os.path.join(a.data_dir, "ETTm1.csv"), 4)
    res["ETTh1"], res["ETTm1"] = h, m
    sub = dfm.iloc[::4].reset_index(drop=True)
    n = min(len(sub), len(dfh))
    same_ts = bool((pd.to_datetime(sub["date"][:n]).values == pd.to_datetime(dfh["date"][:n]).values).all())
    cols = dfh.columns[1:]
    maxdiff = float(np.abs(sub[cols][:n].to_numpy() - dfh[cols][:n].to_numpy()).max())
    res["h_vs_m"] = {"ETTh1_rows": len(dfh), "ETTm1_every4th_rows": len(sub), "timestamps_identical": same_ts, "max_abs_value_diff": maxdiff}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
