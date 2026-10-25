# Experiment Matrix

Status: **v0.2 (2026-10-20). No model has been trained for results.** Parameter counts in Sec. 2 are analytical *and have since been verified* against instantiated modules (gate G1). FLOP/hour figures in Sec. 4 remain **analytical**; no runtime or GPU memory has been measured on a GPU (the authoring sandbox has none) - gate G4 is open and must be completed on Colab before the grid.

## 1. Settings and models

| Axis | Values |
|---|---|
| Dataset | ETTh1, ETTm1 |
| Horizon H | 96, 336 (input L = 336) -> 4 *settings* |
| Models | TCN (independent, Bai et al.), ModernTCN-style (kernel `k`), DLinear; naive references (repeat-last, train-mean) |
| ModernTCN structure | ETTh1: K=1, r=1; ETTm1: K=3, r=8; D=64, P=8, S=4 |
| Kernel sizes (tokens) | **{3, 13, 31, 51, 83}**; "large" = {31, 51, 83}; 3 and 13 are the small/moderate controls; 51 = reference default |
| Seeds | **2021, 2022, 2023** (pre-registered) |

Why these kernels (derived from N = 84 tokens, not from raw steps): with `P=8, S=4` one DW kernel of `k` tokens spans `4(k-1)+8` raw steps:
k=3 -> 16, 13 -> 56, 31 -> 128, 51 -> 208, **83 -> 336 = the entire input window**. Kernels larger than `2N-1 = 167` add only zero-padding taps. k=3/31/51 coincide with the paper's Table 16, so claim C1 needs no extra runs.

## 2. Kernel candidates and parameter counts (analytical; **verified by instantiation**, gate G1)

`small = min(5, k)` in all variants (so the small branch contributes 3*448 params at k=3 and 5*448 elsewhere); width, depth, FFN ratio and head are identical.

| Setting | Config | k=3 | k=13 | k=31 | k=51 | k=83 | max/min - 1 |
|---|---|---|---|---|---|---|---|
| ETTh1 H=96 | K=1, r=1 | 586,912 | 592,288 | 600,352 | 609,312 | 623,648 | **6.26 %** |
| ETTh1 H=336 | K=1, r=1 | 1,877,392 | 1,882,768 | 1,890,832 | 1,899,792 | 1,914,128 | 1.96 % |
| ETTm1 H=96 | K=3, r=8 | 2,081,696 | 2,097,824 | 2,122,016 | 2,148,896 | 2,191,904 | 5.29 % |
| ETTm1 H=336 | K=3, r=8 | 3,372,176 | 3,388,304 | 3,412,496 | 3,439,376 | 3,482,384 | 3.27 % |

All four settings satisfy the 10 % limit **analytically** with the full 5-kernel set, so no width/depth/FFN compensation is needed. The widest spread (ETTh1 H=96, 6.26 %)
is still 3.7 points under the limit; k up to ~101 would also stay under 10 % there, but k > 83 is not useful given N = 84. The unit test `tests/test_models.py::test_modern_tcn_param_counts_equal_analytic_and_spread_within_10pct` asserts that
`sum(p.numel())` of the real modules equals these numbers and that the spread is <= 10 % for every setting (passed).
Composition at k=51, ETTh1 H=96 (analytic): stem 704, per-block 92,416 (DW-large 23,744 incl. BN, DW-small 3,136, norm 128, ConvFFN1 58,240, ConvFFN2 7,168), head 516,192 -> **609,312** (RevIN has no parameters with `affine=False`).
The linear head holds 85 % of ETTh1 H=96 parameters and 95 % at H=336.

Baselines (analytical): TCN (64 ch, 7 levels, k=3) 207,072 (H=96) / 316,272 (H=336), theoretical RF 509; DLinear 64,704 / 226,464.
Parameter counts differ strongly between model families; the 10 % rule applies to the **kernel ablation**, not to TCN-vs-ModernTCN-vs-DLinear (report counts next to every result).

## 3. Run grid by phase

| Phase | Purpose | Runs | Details | Split used for decisions |
|---|---|---|---|---|
| P0 | Unit tests, shape/NaN/loss-decrease smoke test, runtime & memory measurement | tests only | gating, see Sec. 6 | - |
| P1 | Hyperparameter selection | up to 32 | seed 2021, 4 settings: ModernTCN k=51 grid `lr{1e-4,5e-4} x dropout{0.1,0.3}` (16); TCN `lr{1e-3,3e-4}` (8); DLinear `lr{5e-3,1e-3}` (8) | **validation only** |
| P2 | Freeze configuration | - | write `configs/frozen_v1.json`, commit hash recorded | - |
| P3 | Main results | 36 | {TCN, ModernTCN k=51, DLinear} x 4 settings x 3 seeds | validation for checkpointing; test later |
| P4 | Kernel ablation | 48 (full) | ModernTCN k in {3, 13, 31, 83} x 4 settings x 3 seeds (k=51 comes from P3) | validation for checkpointing; test later |
| P5 | ERF measurement | 0 training | Target A (center, last) and B on every ModernTCN/TCN checkpoint (50 val samples; B uses R=16 probes); DLinear B exact | validation windows |
| P6 | Failure diagnostics | ~3-6 | e.g. ModernTCN without RevIN on the failing cell (3 seeds); extra variants only if the failure analysis calls for them | validation |
| P7 | Freeze interpretations | - | `reports/failure_hypotheses_frozen.md` + `configs/frozen_v1.json` hash | - |
| P8 | **Final test evaluation (once)** | all trained runs | `evaluate_test` writes one record per run ID, refuses to overwrite | **test** |
| P9 | Aggregation, figures, report | - | `scripts/aggregate.py`, `scripts/make_figures.py` read the results ledger only | - |

Run-count upper bound without re-use: 32 + 36 + 48 + 6 = **122** trainings. If the selected P1 configuration's seed-2021 run is re-used as the P3 seed-2021 run (same config, same seed, same code), 12 runs are saved; any re-use is recorded in the ledger.

Priority tiers (execute in this order; each tier is useful on its own):
1. **Tier 1 - ETTh1 complete:** P1 for ETTh1 (16 runs), P3 ETTh1 (18), P4 ETTh1 (24), P5 for those. Answers Q1-Q3 and the claim C1 on the paper's own dataset.
2. **Tier 2 - ETTm1 main:** P1 for ETTm1 (16), P3 ETTm1 (18).
3. **Tier 3 - ETTm1 ablation:** *minimal* = k in {31, 83} at H=96 (6 runs; with k=51 gives three large kernels); *full* = {3, 13, 31, 83} x 2 horizons (24 runs). The tier is chosen after measuring cost (Sec. 6, gate G4).
4. **Tier 4 - diagnostics** for the failure case.

## 4. Analytical compute estimate (NOT a measurement)

Per-epoch training cost (GFLOP, `3 x 2 x MACs x #train windows`):

| Setting | ModernTCN k=51 | TCN | DLinear |
|---|---|---|---|
| ETTh1 H=96 | 559.8 | 2,675.4 | 22.2 |
| ETTh1 H=336 | 975.2 | 2,602.4 | 75.6 |
| ETTm1 H=96 | 28,358.3 | 11,123.1 | 92.5 |
| ETTm1 H=336 | 29,995.3 | 11,066.8 | 321.4 |

Totals under two **assumptions** that must be replaced by measurements: (i) mean 40 epochs per run (early stopping; max 100), (ii) sustained throughput of 1-4 TFLOP/s on a free Colab-class GPU (grouped/depthwise convolutions typically run well below peak):

| Phase | PFLOP | hours @ 1 TFLOP/s | hours @ 4 TFLOP/s |
|---|---|---|---|
| P1 (32 runs) | 11.8 | 3.3 | 0.8 |
| P3 (36 runs) | 10.5 | 2.9 | 0.7 |
| P4 ETTh1 (24 runs) | 0.74 | 0.2 | 0.1 |
| P4 ETTm1 full (24 runs) | 28.0 | 7.8 | 1.9 |
| P4 ETTm1 minimal (6 runs) | 6.8 | 1.9 | 0.5 |

Observations: (1) ETTm1 ModernTCN (K=3, r=8) dominates cost (about 28 TFLOP per epoch; ConvFFN width `M*r*D = 3,584`); ETTh1 is ~50x cheaper. (2) P1 for ETTm1 ModernTCN alone is about 80 % of the P1 cost (4 configs x 2 horizons, ~9.3 of 11.8 PFLOP); if the measured cost is high, the ETTm1 grid is cut to the 2 configurations `{reference, lr 5e-4}`. (3) The task's "notebook runs in 30 minutes on a free Colab GPU" cannot hold for the full grid; the notebook will therefore expose `TRAIN = False` and load checkpoints + the results ledger (checkpoints hosted outside Git; link to be added to the README). This is a documented conflict between two requirements, not a hidden one.

## 5. Validation decisions (all made on validation data only)

- Hyperparameters per (model, dataset, horizon): lowest validation MSE among the P1 grid at seed 2021; ties -> reference configuration; frozen for all seeds and kernels.
- Checkpoint inside a run: epoch with lowest validation MSE (ties -> earlier epoch).
- Failure setting and its interpretation: identified from validation results (protocol Sec. 10), frozen at P7.
- Kernel "best" is never selected on test; the ablation reports all kernels.
- No decision may use test numbers; the test ledger is append-only (one record per run ID).

## 6. Gates before launching the grid

Status after the implementation phase (authoring sandbox: CPU only, 1 core; **no GPU measurement exists yet**). Evidence = commands that were actually run.

| Gate | Condition | Status | Evidence |
|---|---|---|---|
| G0 | Dataset loading and chronological borders; train-only scaler; no target leakage; batch-invariant metrics | **passed** | `tests/test_data.py` (20 tests, real ETT files + synthetic) |
| G1 | Real-module parameter counts equal the analytic table; ablation spread <= 10 % in all 4 settings | **passed** | `tests/test_models.py`: 20 ModernTCN cells + TCN + DLinear equal `logs/param_counts_analytic.json`; max spread 6.26 % |
| G2 | Shapes `[B,336,7] -> [B,H,7]`; forward/backward finite; all parameters receive gradients; grouping/channel-order/permutation tests; causality and RF 509; shift-equivariance | **passed** | `tests/test_models.py` (37 tests) |
| G3 | Smoke test: no NaN/Inf, loss decreases | **passed on CPU** (real data) | `logs/smoke_cpu_ETTh1_H336.json`, `logs/smoke_cpu_ETTm1_H336.json`; plus `tests/test_train.py` |
| G4 | Measure s/epoch and peak GPU memory per (model, setting) on a Colab GPU; recompute Sec. 4; choose Tier 3 | **NOT DONE - requires GPU** | `scripts/benchmark_runtime.py` ready; CPU run only exercised the script (`logs/runtime_cpu.json`, not representative) |
| G5 | Resume test: interrupt mid-run, restart, identical continuation | **passed** | `tests/test_train.py::test_resume_after_interruption_reproduces_the_uninterrupted_run` (log and best weights identical), plus crash-at-stop and skip-done tests |
| G6 | ERF: probe estimator equals exact Jacobian (DLinear, tiny ModernTCN, tiny TCN); convergence R=16 vs 256; lag orientation; architectural support; determinism | **passed** | `tests/test_erf.py` (12 tests) |

Whole suite: **102 tests pass** (`python -m pytest tests -q`, ~45-70 s on CPU). Test-suite quality: `scripts/mutation_check.py` injects 20 realistic bugs (wrong grouping, missing permutation, non-causal TCN, scaler on val rows, dropped last batch, unflipped lag axis, removed test guard, ...); **20/20 are detected** (`logs/mutation_report.json`).

Saved-for-backward activation memory (measured with autograd hooks, `logs/activation_memory.json`; lower bound for training memory, gradients/workspace come on top):

| Setting | ModernTCN k=83 per sample | at batch 512 | with micro-batch 128 | TCN at batch 512 |
|---|---|---|---|---|
| ETTh1 H=96 / 336 | 2.9 / 4.1 MiB | 1.45 / 2.07 GiB | - | 2.2 / 2.3 GiB |
| ETTm1 H=96 / 336 | 26.5 / 27.9 MiB | **13.25 / 13.94 GiB** | 3.3 / 3.5 GiB | 2.2 / 2.3 GiB |

Hence amendment A1.3 (micro-batch 128 for ETTm1 ModernTCN). Whether even that fits with workspace on the Colab GPU is to be confirmed in G4.

## 7. Run identifiers, resume and records

- Run ID: `{dataset}_H{H}_{model}[_k{k}]_s{seed}_{cfg8}` where `cfg8` = first 8 hex chars of the SHA-256 of the canonical JSON config.
- Layout: `results/runs/<run_id>/{config.json, train_log.csv, last.pt, best.pt, val_metrics.json, DONE}`; `test_metrics.json` appears only in P8.
- Resume: a run with `DONE` is skipped; otherwise training resumes from `last.pt` (model, optimiser, scheduler, RNG state, epoch, best-val bookkeeping) after verifying the config hash matches. Failed runs are never deleted; they are logged with status `failed` and a reason.
- Ledger `results/runs.csv` (one row per *finished* run; training/validation only): run_id, dataset, pred_len, model, large_kernel, n_blocks, ffn_ratio, seed, params, lr, dropout, head_dropout, batch_size, max_epochs, patience, revin, phase, best_epoch, epochs_run, best_val_mse, best_val_mae, sec_per_epoch, peak_mem_mb, device, torch_version, git_commit, status, cfg_hash. **Test records live only in `results/test_runs.csv`** (append-only, written in P8). Failed runs leave a `FAILED` marker file and no ledger row.
- `scripts/aggregate.py` computes mean +/- sd (ddof=1) over seeds from the ledger; `scripts/make_figures.py` reads only the ledger and ERF `.npz` files.
- Hardware, precision, interrupted/failed runs and their handling are recorded in the ledger and `reports/Discussion.md`.
