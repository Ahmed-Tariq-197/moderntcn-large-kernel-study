"""Figures generated only from results/erf/*.npz and results/summary_*.csv (no hand-entered numbers)."""
import argparse, glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _bootstrap  # noqa

KEY = ("dataset", "pred_len", "model", "large_kernel")


def load_erf(erf_dir: str):
    items = []
    for p in sorted(glob.glob(os.path.join(erf_dir, "*.npz"))):
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        items.append({"meta": meta, "arrays": {k: z[k] for k in z.files if k in ("B", "A_center", "A_last")},
                      "metrics": json.loads(str(z["metrics"])), "path": p})
    return items


def mean_profiles(items):
    g = {}
    for it in items:
        m = it["meta"]
        g.setdefault(tuple(m[k] for k in KEY), []).append(it)
    out = {}
    for key, lst in g.items():
        out[key] = {t: np.mean([i["arrays"][t] for i in lst if t in i["arrays"]], axis=0)
                    for t in ("B", "A_center", "A_last") if any(t in i["arrays"] for i in lst)}
        out[key]["metrics"] = {t: {m: np.mean([i["metrics"][t][m] for i in lst if i["metrics"].get(t, {}).get(m) is not None])
                                   for m in ("r50", "r90", "r95", "support")} for t in out[key] if t != "metrics"}
        out[key]["n"] = len(lst)
    return out


def plot_models(items, out_dir):
    prof = mean_profiles(items)
    paths = []
    for ds, H in sorted({(k[0], k[1]) for k in prof}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for key, p in prof.items():
            if key[0] != ds or key[1] != H or (key[2] == "modern_tcn" and key[3] != 51):
                continue
            label = {"tcn": "TCN", "modern_tcn": "ModernTCN-style (k=51)", "dlinear": "DLinear"}[key[2]]
            for ax, t in zip(axes, ("B", "A_last")):
                if t in p:
                    ax.plot(np.arange(len(p[t])), p[t] / p[t].max(), label=label)
        axes[0].set_title(f"{ds} H={H}: forecast ERF (Target B)")
        axes[1].set_title("backbone ERF at last position (Target A)")
        for ax in axes:
            ax.set_xlabel("lag (raw steps, 0 = newest)"); ax.set_ylabel("sensitivity / max"); ax.set_yscale("log"); ax.legend() if ax.get_legend_handles_labels()[0] else None
        fig.tight_layout(); pth = os.path.join(out_dir, f"erf_models_{ds}_H{H}.png"); fig.savefig(pth, dpi=130); plt.close(fig)
        paths.append(pth)
    return paths


def plot_kernels(items, summary_csv, out_dir):
    prof = mean_profiles(items)
    sm = pd.read_csv(summary_csv) if summary_csv and os.path.exists(summary_csv) else None
    paths = []
    for ds, H in sorted({(k[0], k[1]) for k in prof if k[2] == "modern_tcn"}):
        ks = sorted(k[3] for k in prof if k[:2] == (ds, H) and k[2] == "modern_tcn")
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        for k in ks:
            p = prof[(ds, H, "modern_tcn", k)]
            axes[0].plot(p["B"] / p["B"].max(), label=f"k={k}")
            if "A_center" in p:
                axes[1].plot(p["A_center"] / p["A_center"].max(), label=f"k={k}")
        axes[0].set_title("Target B (forecast ERF)"); axes[1].set_title("Target A, centre token")
        for ax in axes[:2]:
            ax.set_yscale("log"); ax.set_xlabel("lag (raw steps)"); ax.legend()
        r90B = [prof[(ds, H, "modern_tcn", k)]["metrics"]["B"]["r90"] for k in ks]
        axes[2].plot(ks, r90B, "o-", label="r90 of B (lags)")
        if all("A_center" in prof[(ds, H, "modern_tcn", k)] for k in ks):
            axes[2].plot(ks, [prof[(ds, H, "modern_tcn", k)]["metrics"]["A_center"]["r90"] for k in ks], "s-", label="r90 of A centre")
        if sm is not None:
            ax2 = axes[2].twinx()
            s = sm[(sm.dataset == ds) & (sm.pred_len == H) & (sm.model == "modern_tcn")].sort_values("large_kernel")
            col = "test_mse" if "test_mse_mean" in s else "best_val_mse"
            mcol, scol = f"{col}_mean", f"{col}_sd"
            if len(s):
                ax2.errorbar(s.large_kernel, s[mcol], yerr=s[scol], fmt="k^--", capsize=3, label=col)
                ax2.set_ylabel(col)
        axes[2].set_xlabel("large kernel (tokens)"); axes[2].set_ylabel("ERF width (lags)"); axes[2].legend(loc="upper left")
        fig.suptitle(f"{ds} H={H}: kernel ablation"); fig.tight_layout()
        pth = os.path.join(out_dir, f"kernel_ablation_{ds}_H{H}.png"); fig.savefig(pth, dpi=130); plt.close(fig)
        paths.append(pth)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--summary", default=None, help="summary_test.csv (default) or summary_val.csv")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    items = load_erf(os.path.join(a.results, "erf"))
    summ = a.summary or (os.path.join(a.results, "summary_test.csv") if os.path.exists(os.path.join(a.results, "summary_test.csv"))
                         else os.path.join(a.results, "summary_val.csv"))
    print(plot_models(items, a.out) + plot_kernels(items, summ, a.out))


if __name__ == "__main__":
    main()
