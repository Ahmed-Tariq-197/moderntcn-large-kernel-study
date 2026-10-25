"""Data audit for ETTh1 / ETTm1 (no torch dependency).

Verifies, from the raw CSV files only:
  * file shape, column layout, NaN / duplicate / ordering / sampling-interval checks
  * the standard 12/4/4-month chronological split borders used by benchmark loaders
  * number of sliding windows per split for the (L=336, H in {96, 336}) setting
  * the effect of the "drop-last" test-loader convention on the test set
  * train-only standardisation statistics and train -> val/test distribution shift
  * relationship between the hourly (ETTh1) and 15-minute (ETTm1) files

Usage:
    python scripts/audit_data.py --data_dir /path/to/ETT-small --out logs/data_audit.json

The border formulas are re-derived here from the benchmark convention
(12*30*24 / 4*30*24 / 4*30*24 steps, x4 for 15-minute data); they are NOT
imported from any reference loader.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

L = 336
HORIZONS = (96, 336)
BATCH_DROP = 512  # batch size used by the reference ModernTCN forecasting scripts


def borders(freq_mult: int):
    """(train_end, val_end, test_end) in rows; freq_mult=1 hourly, 4 for 15-min."""
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

    # windows per split (inputs for val/test reach back L rows into the previous split)
    wins = {}
    for H in HORIZONS:
        ntr = n_windows(tr, L, H)                 # rows [0, tr)
        nva = n_windows(va - (tr - L), L, H)      # rows [tr-L, va)
        nte = n_windows(te - (va - L), L, H)      # rows [va-L, te)
        dropped = nte % BATCH_DROP
        wins[str(H)] = {
            "train": ntr, "val": nva, "test": nte,
            "test_dropped_by_drop_last_bs512": dropped,
            "test_dropped_pct": round(100 * dropped / nte, 2),
        }
    out["windows"] = wins

    # train-only standardisation + shift diagnostics (in train-std units)
    X = df[feats].to_numpy(dtype=np.float64)
    mu, sd = X[:tr].mean(0), X[:tr].std(0)  # ddof=0, as sklearn StandardScaler
    Z = (X - mu) / sd
    out["train_mean"] = dict(zip(feats, np.round(mu, 4).tolist()))
    out["train_std"] = dict(zip(feats, np.round(sd, 4).tolist()))
    out["shift_val_mean_in_train_std"] = dict(zip(feats, np.round(Z[tr:va].mean(0), 3).tolist()))
    out["shift_test_mean_in_train_std"] = dict(zip(feats, np.round(Z[va:te].mean(0), 3).tolist()))
    out["val_std_ratio_to_train"] = dict(zip(feats, np.round(Z[tr:va].std(0), 3).tolist()))
    out["test_std_ratio_to_train"] = dict(zip(feats, np.round(Z[va:te].std(0), 3).tolist()))
    # MSE of the trivial "predict the train mean (= 0 after scaling)" forecaster per split
    out["mse_zero_forecast"] = {"train": float((Z[:tr] ** 2).mean()),
                                "val": float((Z[tr:va] ** 2).mean()),
                                "test": float((Z[va:te] ** 2).mean())}
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
    # is ETTh1 the hourly subsample of ETTm1?
    sub = dfm.iloc[::4].reset_index(drop=True)
    n = min(len(sub), len(dfh))
    same_ts = bool((pd.to_datetime(sub["date"][:n]).values == pd.to_datetime(dfh["date"][:n]).values).all())
    cols = dfh.columns[1:]
    maxdiff = float(np.abs(sub[cols][:n].to_numpy() - dfh[cols][:n].to_numpy()).max())
    res["h_vs_m"] = {"ETTh1_rows": len(dfh), "ETTm1_every4th_rows": len(sub),
                     "timestamps_identical": same_ts, "max_abs_value_diff": maxdiff}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
