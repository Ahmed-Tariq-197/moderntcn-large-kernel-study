"""Aggregate the run ledgers into mean +/- sd (ddof=1) tables and evaluate the pre-registered claim C1.
Reads results/runs.csv (validation bookkeeping) and results/test_runs.csv (append-only test records)."""
import argparse, json, os
import numpy as np, pandas as pd
import _bootstrap  # noqa

GROUP = ["dataset", "pred_len", "model", "large_kernel", "revin"]
MAIN_PHASES = ("p3", "p4", "p6")


def load_ledger(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates("run_id", keep="last")
    if "status" in df:
        df = df[~df["status"].astype(str).str.startswith("failed")]
    return df


def select_main(runs: pd.DataFrame, frozen_path: str = None) -> pd.DataFrame:
    """Runs belonging to the reported matrix: phases p3/p4/p6, plus P1 runs that are *the same run* as a P3 seed-2021
    run (run ids ignore `phase`, so the selected hyperparameters' P1 run is re-used rather than retrained)."""
    keep = runs.phase.isin(MAIN_PHASES)
    if frozen_path and os.path.exists(frozen_path):
        hp = json.load(open(frozen_path)).get("hparams", {})
        def is_selected(r):
            h = hp.get(f"{r['model']}|{r['dataset']}|{int(r['pred_len'])}")
            if h is None or r["phase"] != "p1" or int(r["seed"]) != 2021:
                return False
            if r["model"] == "modern_tcn" and int(r["large_kernel"]) != 51:
                return False
            return bool(np.isclose(float(r["lr"]), h["lr"]) and (r["model"] == "dlinear" or np.isclose(float(r["dropout"]), h["dropout"])))
        keep = keep | runs.apply(is_selected, axis=1)
    return runs[keep]


def summarize(df: pd.DataFrame, cols, group=GROUP) -> pd.DataFrame:
    """mean, sample sd (ddof=1) and n per group for each column in `cols`; plus parameter count and seed list."""
    g = df.groupby(group, dropna=False)
    parts = []
    for c in cols:
        parts.append(g[c].mean().rename(f"{c}_mean"))
        parts.append(g[c].std(ddof=1).rename(f"{c}_sd"))
        parts.append(g[c].count().rename(f"{c}_n"))
    out = pd.concat(parts, axis=1)
    out["params"] = g["params"].first()
    out["seeds"] = g["seed"].apply(lambda s: ",".join(str(int(v)) for v in sorted(s)))
    return out.reset_index()


def claim_c1(test: pd.DataFrame, thresh: float = 0.005) -> dict:
    """Pre-registered rule (research_protocol.md Sec. 7). `test`: ETTh1 modern_tcn test rows with revin=True."""
    t = test[(test.dataset == "ETTh1") & (test.model == "modern_tcn") & (test.revin.astype(bool))]
    res = {"comparisons": {}, "threshold": thresh}
    passes, complete = 0, True
    def per_seed(H, k):
        s = t[(t.pred_len == H) & (t.large_kernel == k)].set_index("seed")["test_mse"]
        return s
    for H in (96, 336):
        base = per_seed(H, 3)
        for k in (31, 51):
            cur = per_seed(H, k)
            seeds = sorted(set(base.index) & set(cur.index))
            if len(seeds) < 3:
                complete = False
                res["comparisons"][f"H{H}_k{k}"] = {"status": "missing_seeds", "seeds": seeds}
                continue
            d = np.array([base[s] - cur[s] for s in seeds])
            ok = bool((d > 0).all() and d.mean() >= thresh)
            passes += ok
            res["comparisons"][f"H{H}_k{k}"] = {"d_per_seed": d.tolist(), "mean_d": float(d.mean()), "pass": ok}
    c1b = {}
    for H in (96, 336):
        a, b = per_seed(H, 31), per_seed(H, 51)
        seeds = sorted(set(a.index) & set(b.index))
        if len(seeds) < 3:
            complete = False
            continue
        diff = float(np.mean([a[s] - b[s] for s in seeds]))
        c1b[f"H{H}"] = {"mean_mse31_minus_mse51": diff, "no_further_gain": abs(diff) < thresh}
    res["C1b"] = c1b
    res["passes"] = int(passes)
    res["C1a_verdict"] = ("inconclusive_missing_data" if not complete else
                          "supported" if passes == 4 else "partially_supported" if passes >= 2 else "not_supported")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--frozen", default="configs/frozen_v1.json")
    a = ap.parse_args(argv)
    runs = load_ledger(os.path.join(a.results, "runs.csv"))
    runs = select_main(runs, a.frozen)
    val = summarize(runs, ["best_val_mse", "best_val_mae"])
    val.to_csv(os.path.join(a.results, "summary_val.csv"), index=False)
    print(val.to_string(index=False))
    tp = os.path.join(a.results, "test_runs.csv")
    if os.path.exists(tp):
        test = pd.read_csv(tp).drop_duplicates("run_id", keep="last")
        meta = runs[["run_id", "revin", "phase"]]
        test = test.merge(meta, on="run_id", how="inner")                 # inner join: only runs of the reported matrix
        ts = summarize(test, ["test_mse", "test_mae"])
        ts.to_csv(os.path.join(a.results, "summary_test.csv"), index=False)
        c1 = claim_c1(test)
        json.dump(c1, open(os.path.join(a.results, "claim_c1.json"), "w"), indent=2)
        print(ts.to_string(index=False))
        print(json.dumps(c1, indent=2))


if __name__ == "__main__":
    main()
