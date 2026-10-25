"""Analytic parameter / MAC counts (pure Python, NO torch).

STATUS: analytical hand-derived formulas. They are NOT yet verified against
instantiated nn.Module objects. tests/test_param_counts.py (to be written with the
models) must assert that these numbers equal sum(p.numel()) of the real modules
before any training run is launched.

Architecture specification being counted (ModernTCN-style, forecasting, single stage):
  RevIN (affine=False -> 0 params) -> stem Conv1d(1,D,P,stride S)+BN(D)
  -> K x block[ DWConv(M*D ch, groups=M*D, k, no bias)+BN(M*D)
                + parallel DWConv(small=min(5,k))+BN(M*D)        (unmerged re-param branch)
                -> BN(D)  (applied per variable over the D feature channels)
                -> ConvFFN1 (grouped 1x1, groups=M, hidden dff=r*D, with bias)
                -> ConvFFN2 (grouped 1x1, groups=D, hidden dff=r*D, with bias) ; residual ]
  -> flatten(D*N) -> Linear(D*N, H)   (shared across variables)
Counted: trainable parameters only (BatchNorm running stats are buffers).
"""
import json
import math
import os

M, L, P, S = 7, 336, 8, 4
N = L // S  # 84 tokens after padding (P - S) last values and stride-S patching
D = 64


def modern_tcn(k: int, K: int, r: int, H: int, small: int | None = None, D: int = D):
    small = min(5, k) if small is None else small
    C = M * D
    dff = r * D
    stem = (D * P + D) + 2 * D
    dw_large = C * k + 2 * C
    dw_small = C * small + 2 * C
    norm = 2 * D
    # ConvFFN1: Conv1d(C -> M*dff, groups=M), Conv1d(M*dff -> C, groups=M)
    ffn1 = (M * dff * D + M * dff) + (C * dff + C)
    # ConvFFN2: Conv1d(C -> M*dff, groups=D), Conv1d(M*dff -> C, groups=D)
    ffn2 = (M * dff * M + M * dff) + (C * (M * dff // D) + C)
    block = dw_large + dw_small + norm + ffn1 + ffn2
    head = D * N * H + H
    total = stem + K * block + head
    # MACs per sample (forward)
    macs_stem = M * N * D * P
    macs_block = (C * N * k) + (C * N * small) + (M * dff * D * N) + (C * dff * N) \
        + (M * dff * M * N) + (C * (M * dff // D) * N)
    macs = macs_stem + K * macs_block + M * (D * N) * H
    return {"total": total, "stem": stem, "per_block": block, "dw_large": dw_large,
            "dw_small": dw_small, "ffn1": ffn1, "ffn2": ffn2, "head": head, "macs_per_sample": macs}


def dlinear(H: int):
    # two shared Linear(L -> H) layers (trend / seasonal); individual=False
    total = 2 * (L * H + H)
    return {"total": total, "macs_per_sample": 2 * L * H * M}


def tcn(H: int, ch: int = 64, levels: int = 7, k: int = 3):
    """Bai et al. TemporalConvNet: 2 weight-normed causal convs per level, 1x1 downsample
    on the residual when channel counts differ; head = Linear(ch -> M*H) on the last step.
    weight_norm adds one scale g per output channel to each conv."""
    total, macs = 0, 0
    cin = M
    for _ in range(levels):
        c1 = ch * cin * k + ch + ch          # v + g + bias
        c2 = ch * ch * k + ch + ch
        down = (cin * ch + ch) if cin != ch else 0
        total += c1 + c2 + down
        macs += (ch * cin * k + ch * ch * k) * L + (cin * ch * L if cin != ch else 0)
        cin = ch
    head = ch * (M * H) + M * H
    total += head
    macs += ch * M * H
    rf = 1 + 2 * (k - 1) * (2 ** levels - 1)
    return {"total": total, "head": head, "receptive_field_steps": rf, "macs_per_sample": macs}


# dataset/horizon configs follow the official ETT scripts (K, r); windows from logs/data_audit.json
CFG = {"ETTh1": dict(K=1, r=1), "ETTm1": dict(K=3, r=8)}
KERNELS = [3, 13, 31, 51, 83]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    audit = json.load(open(os.path.join(here, "..", "logs", "data_audit.json")))
    out = {"assumptions": {"M": M, "L": L, "P": P, "S": S, "N": N, "D": D,
                           "small_kernel_rule": "min(5, k)", "revin_params": 0},
           "ablation": {}, "baselines": {}}
    for ds, c in CFG.items():
        for H in (96, 336):
            key = f"{ds}_H{H}"
            rows = {k: modern_tcn(k, c["K"], c["r"], H) for k in KERNELS}
            tot = {k: rows[k]["total"] for k in KERNELS}
            spread = max(tot.values()) / min(tot.values()) - 1
            ntr = audit[ds]["windows"][str(H)]["train"]
            out["ablation"][key] = {
                "K": c["K"], "r": c["r"],
                "totals": tot, "max_over_min_minus_1": round(spread, 4),
                "within_10pct": spread <= 0.10,
                "macs_per_sample_k51": rows[51]["macs_per_sample"],
                "train_windows": ntr,
                "train_gflop_per_epoch_k51": round(6 * rows[51]["macs_per_sample"] * ntr / 1e9, 1),
                "detail_k51": rows[51],
            }
            t = tcn(H); d = dlinear(H)
            out["baselines"][key] = {
                "TCN": t, "DLinear": d,
                "tcn_train_gflop_per_epoch": round(6 * t["macs_per_sample"] * ntr / 1e9, 1),
                "dlinear_train_gflop_per_epoch": round(6 * d["macs_per_sample"] * ntr / 1e9, 1),
            }
    # raw-time-step reach of the DW kernel in the stem-token axis
    out["kernel_reach_raw_steps_single_block"] = {k: S * (k - 1) + P for k in KERNELS}
    out["note"] = ("train_gflop_per_epoch = 3 (fwd+bwd) * 2 FLOP/MAC * MACs * #train windows; "
                   "it is an arithmetic count, NOT a measured runtime.")
    path = os.path.join(here, "..", "logs", "param_counts_analytic.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
