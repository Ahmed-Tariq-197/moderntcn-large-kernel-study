# Research Protocol (frozen before any training run)

Protocol version: **v0.1**, written 2026-10-20, **before any model has been trained or evaluated**.
Anything below marked *(pre-registered)* must not be changed after results are seen; changes require
a dated amendment at the end of this file with the reason.

## 1. Research question

When do large depthwise temporal kernels give better forecasting accuracy, or a different effective
receptive field (ERF), than dilation-based temporal convolutions - and does the measured ERF explain accuracy?

| Sub-question | Evidence that answers it |
|---|---|
| Q1. Does a ModernTCN-style model improve accuracy over a dilated TCN and over DLinear? | 3-seed MSE/MAE, 2 datasets x 2 horizons |
| Q2. Does changing the temporal kernel size change the measured ERF? | ERF profiles + width metrics for k in {3,13,31,51,83} |
| Q3. Does a larger measured ERF correspond to better accuracy? | Per-setting scatter/table of ERF width vs. validation/test MSE |

Nothing is assumed in advance: larger kernels, larger ERF, or more complexity are hypotheses, not premises.

## 2. Variables

- **Independent (ablation):** large depthwise kernel size `k` in {3, 13, 31, 51, 83} (units: **patch tokens**, not raw time steps). Only `k` changes
  (the small parallel kernel follows the fixed rule `small = min(5, k)`; see Sec. 11, D9).
- **Held fixed in the ablation:** D=64, number of blocks K, FFN ratio r, patch P=8/S=4, dropout, optimiser, learning rate, batch size, schedule, epochs, patience, seeds, data, selection rule.
- **Dependent:** MSE, MAE (standardised space); ERF profile and width metrics; parameter count.
- **Between-model comparison:** TCN vs. ModernTCN-style (k=51) vs. DLinear.

## 3. Dataset protocol (verified; see `logs/data_audit.json`, `research_notes/sources.md` Sec. 5)

- Files: `ETTh1.csv` (17,420 rows, 1 h), `ETTm1.csv` (69,680 rows, 15 min); 7 variables (`HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`), all used (multivariate-to-multivariate, `features=M`).
- Chronological 12/4/4-month benchmark split: train rows `[0, 8640)`, val `[8640, 11520)`, test `[11520, 14400)` for ETTh1; all x4 for ETTm1. Rows beyond 14,400 / 57,600 are **unused** (benchmark convention, kept for comparability).
- Window construction: input length **L = 336**, horizons **H in {96, 336}**, stride 1. Target windows lie strictly inside their own split: first validation target is row 8,640, first test target row 11,520.
- **Context policy (exact):** validation/test *inputs* may reach back L rows into the preceding split (val inputs may include the last 336 train rows; test inputs the last 336 validation rows). Targets never do. This follows the benchmark loaders (S2, S6); it uses no information from the future of the forecast origin.
- **Normalisation:** per-variable z-score with mean/std (ddof=0) fitted on the **training rows only** (`[0, 8640)` / `[0, 34560)`), applied unchanged to val/test. **All metrics are computed in this standardised space.**
- **Evaluation set:** *all* windows of the split (`drop_last=False`; batch size must not change the metric). Metric = total squared (absolute) error divided by the total number of predicted values (windows x H x 7).
- Window counts (from audit): ETTh1 H=96 train/val/test = 8,209/2,785/2,785; H=336 = 7,969/2,545/2,545. ETTm1 H=96 = 34,129/11,425/11,425; H=336 = 33,889/11,185/11,185.
- Model-internal instance normalisation (RevIN) is a model component and is separate from the dataset z-scoring above.

## 4. ModernTCN-style architecture (to be implemented from the paper; reference code is read-only)

Notation: B batch, M=7 variables, L=336, D=64 feature channels, P=8, S=4, N=84 tokens, K blocks, r FFN ratio, H horizon, C=M*D=448.

| Stage | Operation | Output shape |
|---|---|---|
| input | window from loader | `[B, L, M]` |
| RevIN (norm) | per-instance, per-variable mean/std over L; statistics `detach()`ed; `affine=False` | `[B, L, M]` -> permute `[B, M, L]` |
| pad | repeat last value `P - S = 4` times at the end (paper App. B.1) | `[B, M, 340]` |
| stem | reshape `[B*M, 1, 340]`; `Conv1d(1, D, kernel=P, stride=S)`; BatchNorm1d(D) | `[B*M, D, 84]` -> `[B, M*D, 84]` (channel = m*D + d) |
| block (x K) | see below | `[B, M*D, 84]` |
| flatten + head | `[B, M, D*N]` -> dropout -> `Linear(D*N, H)` shared over variables | `[B, M, H]` |
| RevIN (denorm) | invert with stored statistics | `[B, M, H]` -> `[B, H, M]` |

Block (residual around the whole block, `Z_{i+1} = Block(Z_i) + Z_i`, paper Eq. 3):
1. **Large-kernel DWConv:** `Conv1d(C, C, k, padding=k//2, groups=C, bias=False)` + BatchNorm1d(C); in parallel **small DWConv** `Conv1d(C, C, s, padding=s//2, groups=C, bias=False)` + BatchNorm1d(C); the two branch outputs are summed (structural re-parameterisation, branches kept unmerged). Shape `[B, C, N]`. Mixes **time only**, independently per (variable, feature).
2. BatchNorm1d(D) applied on `[B*M, D, N]`, restored to `[B, C, N]`.
3. **ConvFFN1 (within each variable):** `Conv1d(C, M*rD, 1, groups=M)` -> GELU -> Dropout -> `Conv1d(M*rD, C, 1, groups=M)` -> Dropout. Mixes features within one variable.
4. **ConvFFN2 (across variables):** reshape `[B, M, D, N]` -> permute to `[B, D*M, N]` (channel = d*M + m); `Conv1d(DM, M*rD, 1, groups=D)` -> GELU -> Dropout -> `Conv1d(M*rD, DM, 1, groups=D)` -> Dropout; permute back. Mixes variables at each feature index.
5. Residual add.

Configurations (from the paper's recommendations, App. C.1/Tables 12-13 and the ETT scripts): **ETTh1: K=1, r=1**; **ETTm1: K=3, r=8**; D=64, P=8, S=4.
Kernel size is in token units: a single block's DW kernel `k` spans `S*(k-1) + P` raw steps: k=3: 16, 13: 56, 31: 128, 51: 208, 83: 336 (= the whole input window).
Deviations from the paper are listed in Sec. 11.

## 5. Baselines

**TCN (independent implementation grounded in Bai, Kolter & Koltun 2018; structure verified against `locuslab/TCN/tcn.py`).**
`[B, M, L]` (after the same RevIN wrapper as ModernTCN, so the backbone is the only difference) -> 7 levels x `TemporalBlock`
(2 x [weight-normed causal `Conv1d(k=3, dilation=2^i)` + Chomp + ReLU + Dropout], 1x1 residual projection when channels differ), 64 channels -> **last time step** `[B, 64]` -> `Linear(64, M*H)` -> `[B, M, H]` -> RevIN denorm.
Theoretical receptive field `1 + 2(k-1)(2^n - 1) = 509 >= 336`. Dropout 0.1 fixed; learning rate chosen on validation.
*Adaptation note:* the Lab 4 TCN is unavailable; this baseline is therefore a documented re-implementation, not "the Lab 4 TCN". Its single-vector head (64 -> M*H) is an architectural choice that may limit capacity; this is an explicit hypothesis for any TCN underperformance, not a conclusion.

**DLinear (independent implementation from the official code structure).** Moving-average decomposition (kernel 25, replicate end-padding) -> two shared `Linear(L, H)` (trend, seasonal), summed; `individual=False`; no RevIN (as the original). Parameter counts 64,704 (H=96), 226,464 (H=336).
*Confound to report:* ModernTCN/TCN use RevIN, DLinear does not.

**Naive references (no training, for failure analysis):** repeat-last-value; train-mean (zero in standardised space). Both evaluated on val and (at the end) test.

## 6. ERF measurement protocol (methodological adaptation - pre-registered)

**Why adapted:** the Lab 4 autograd procedure is unavailable. The official ModernTCN repository publishes no ERF code. The procedure below is built from (i) ModernTCN Sec. 5.2 ("sample 50 length-336 input time series from the validation set in ETTh1 ... how many points in the input series can make contribution to the middle point of the final feature map"), and
(ii) the audit repository's `visualize_erf` routine (RepLKNet-derived; center feature -> ReLU -> sum -> `autograd.grad` -> ReLU -> sum over batch and variable axes). It makes **no claim of equivalence to Lab 4**.

**Common conventions (identical for all models).** Models in `eval()` mode (BatchNorm running statistics, dropout off), fp32, RevIN with detached statistics (as in the reference). Input `x in R^{M x L}` is a standardised validation window. Lag `tau = L - 1 - t`, `tau = 0` is the most recent input step; unit = raw time steps (1 h for ETTh1, 15 min for ETTm1; both are reported in steps and in hours). Samples: `S = 50` validation windows at deterministic, evenly spaced indices `i_j = floor(j * (n_val - 1) / 49)`, `j = 0..49` (the same windows for every model, seed and kernel). Measurement is done per trained seed checkpoint; results are reported as mean +/- sd over seeds.

**Target A - backbone ERF (paper-style).** Let `z(x)` be the final backbone feature map before the head (`[M, D, N]` for ModernTCN, `[C, L]` for TCN) and `p` a position index.
`s_A(x; p) = sum_{m,d} ReLU(z_{m,d,p}(x))`, `g_A = ReLU(d s_A / d x) in R^{M x L}`, `a_A(tau) = sum_m g_A[m, L-1-tau]`, averaged over the 50 samples.
Positions: `p_center` (ModernTCN token `N//2 = 42`, raw patch `[168, 175]`; TCN step `L//2 = 168`) and `p_last` (token 83 / step 335). The center target mirrors the paper's figure; the last-position target is the one the forecast actually uses. For `p_center` the profile is also plotted the **signed offset** `t - t_target` because a causal TCN can only see the past while the (non-causal) ModernTCN sees both sides.

**Target B - forecast ERF (added by this study; the quantity that matters for accuracy).** For the full prediction `y_hat in R^{M x H}`, the Jacobian `J = d y_hat / d x` has `(M*H) x (M*L)` entries. Its column energy is estimated by Hutchinson-style probes:
`u ~ Rademacher^{M x H}`, `v = d (u . y_hat) / d x in R^{M x L}`; since `E_u[u u^T] = I`, `E_u[v_{m,t}^2] = sum_{m',h} J_{(m',h),(m,t)}^2`.
`b(tau) = sqrt( (1/S) sum_samples sum_m E_u[ v_{m, L-1-tau}^2 ] )` with `R = 16` probes per sample (a convergence check against `R = 64` on 5 samples is part of the test suite; B is also computed exactly for DLinear, where `J` is the weight matrix).
B is sign-agnostic (no ReLU), A follows the reference's ReLU convention - the difference is intentional and documented.

**Normalisation & summary metrics (same for A and B):** report the profile normalised by its maximum, `a(tau)/max a`. Cumulative mass `C(tau) = sum_{tau' <= tau} a(tau') / sum a`. Width metrics: `r50, r90, r95` = smallest lag with `C >= 0.5/0.9/0.95`; `support = #{tau : a(tau) >= 0.01 max a}`; also total mass in the last 24 / 96 steps. Aggregation across variables: sum (A) / energy-sum (B); across samples: mean; across seeds: mean +/- sd.

**Pre-registered expectations (to be checked, not assumed):**
- E1: Target A at `p_center` grows with `k` and is bounded by the window (raw reach of one block: 16/56/128/208/336 steps; K=3 blocks in ETTm1 can saturate the window sooner).
- E2: Target B is **much flatter in `k`** than A, because the linear head reads all 84 tokens, so every input step can reach the output regardless of `k`; `k` changes how features are mixed, not whether the forecast can see old inputs. If B turns out to scale with `k`, E2 is falsified.
- E3: For the TCN, B is concentrated at small lags, far below the theoretical receptive field 509 (theoretical RF vs. effective RF gap).
- E4 (Q3): accuracy improves from `k=3` to `k~31` and then saturates (paper's Table 16 pattern) while A keeps growing; therefore ERF width is not expected to predict accuracy beyond the small-kernel regime.

## 7. Quantitative paper claim (selected before any results are available) - pre-registered

**Claim C1 (primary)** - ModernTCN, Appendix F.2.1 "Impact of large kernel size" and Table 16:
> "Results on Table 16 show that increasing the kernel size leads to performance improvement."

Paper setting: ETTh1, `D=64, K=1, r=1`, kernel sizes 3/31/51 with structural re-parameterisation, horizons 96/192/336/720, metrics averaged over 5 seeds, NVIDIA A100. Reported ETTh1 MSE (96/336): k=3: 0.381/0.403; k=31: 0.367/0.389; k=51: 0.368/0.391. The paper does not state, for this table, the small-kernel size paired with k=3 nor the input length.
Note that the paper's own numbers are **not monotone** from 31 to 51 (+0.001, +0.002), so the sentence is only true between small and large kernels in the paper's data.

Test design (smallest defensible): ETTh1, **L=336, H in {96, 336}, k in {3, 31, 51}** (all inside the ablation grid, no extra training), 3 seeds, same frozen hyperparameters for every k, test MSE/MAE at the validation-selected checkpoint. Extension (not covered by the paper's table): ETTm1, and k in {13, 83}.

**Decision rule (pre-registered).** Let `d_i(k) = MSE_test(k=3, seed i) - MSE_test(k, seed i)` (paired by seed).
- **C1a "large beats small"** is *supported* if, for `k in {31, 51}` and both horizons (4 comparisons), `d_i > 0` for all 3 seeds and `mean(d) >= 0.005` (about 1.3 % of MSE; paper's effect is 0.014). *Partially supported* if 2-3 of the 4 comparisons pass. *Not supported* otherwise.
- **C1b "no further gain 31 -> 51"** (the paper's own numbers) is supported if `|mean(MSE(31) - MSE(51))| < 0.005`.
- With 3 seeds the test has little power; verdicts are reported with the paired differences and per-seed signs, never as significance claims.
Comparability limits: the available budget (patience 10, 3 seeds), fixed L=336, k=3 uses small kernel 3 (assumption), single-stage model without multi-scale, ETTm1/k extensions are outside the paper's table.

**Claim C2 (secondary, ERF)** - same section: "directly enlarging the kernel size ... can effectively increase ERF". Tested through Target A/B metrics vs `k` (Sec. 6, E1/E2).

## 8. Training, selection and seeds

- Loss: MSE on standardised targets. Optimiser: Adam. Precision: fp32 (TF32 matmul allowed on GPU only if recorded).
- ModernTCN: reference-style recipe: batch 512, `dropout 0.3`, `head_dropout 0.0` (ETTh1) / `0.1` (ETTm1 H=336), lr schedule "keep 3 epochs then x0.9 per epoch", max 100 epochs, **early-stopping patience 10 on validation MSE** (reference uses 20; reduced for compute, deviation D3).
- TCN: Adam, batch 512 (same schedule, patience 10). DLinear: Adam, batch 32, lr schedule as the reference baseline, max 100 epochs, patience 10.
- **Hyperparameter selection (validation only, seed 2021, ModernTCN at k=51):** grid `lr in {1e-4, 5e-4} x dropout in {0.1, 0.3}`; TCN `lr in {1e-3, 3e-4}`; DLinear `lr in {5e-3, 1e-3}`. One configuration is chosen per (model, dataset, horizon) by lowest validation MSE and then **frozen for all seeds and all kernel sizes** (no per-kernel retuning). Selection ties are broken in favour of the reference configuration.
- Checkpoint selection inside a run: epoch with the lowest validation MSE.
- **Seeds (pre-registered): 2021, 2022, 2023** for every reported configuration (same seeds across kernels/models). Reported: mean +/- sample standard deviation (ddof=1) over the 3 seeds.
- Test discipline: training/selection code never touches the test split. A separate `evaluate_test` entry point writes a test record **once per run ID** and refuses to overwrite it. The protocol hash (git commit + config files) is recorded before the test phase.

## 9. Metrics

MSE and MAE in standardised space (Sec. 3). Additionally logged (diagnostics): per-variable MSE, per-forecast-step MSE, validation-to-test gap, parameter count, wall-clock per epoch, peak GPU memory.

## 10. Failure-analysis plan (pre-registered procedure)

1. **Definition.** A *clearly failing setting* is a (model, dataset, horizon[, kernel]) cell where, on **validation** and over all 3 seeds, the model is worse than a reference by more than the seed spread: (a) worse than the repeat-last-value or the train-mean baseline, or (b) worse than DLinear by more than 2 pooled standard deviations in every seed, or (c) training diverges / produces NaN in any seed. Failures are reported as they occur; no model is degraded on purpose and no seed is excluded.
2. **Candidate hypotheses** (to be tested, not asserted): H-a distribution shift between train and val/test (test OT mean `-1.34` train-sd, sd ratio `0.34`); H-b context/horizon mismatch for H=336; H-c over-fitting of the large linear head (head = 516k-1.8M of 0.6M-1.9M parameters for ETTh1); H-d limited capacity of the single-vector TCN head; H-e overly large kernel wasting capacity in the K=1 regime; H-f resolution (ETTh1 vs ETTm1 are the same series at two sampling rates, so they are not independent).
3. **Diagnostics** (computed on validation): per-variable and per-step error, learning curves (train vs val), RevIN-off variant of ModernTCN for one failing cell (3 seeds), ERF profiles of the failing model, comparison with naive baselines.
4. **Freeze:** the written interpretation and the predictions it makes for the test split are stored in `reports/failure_hypotheses_frozen.md` (with the git commit) **before** the test phase. The test result may then confirm or contradict it; contradictions are reported.
5. The report must contain: the observed failure, the strongest evidence, alternative explanations, and the experiment that would discriminate between them.

## 11. Methodological adaptations and deviations (must appear in the final report)

- **D1** Single-stage ModernTCN: no multi-scale downsampling, time-feature embeddings unused (the reference also ignores them for ETT).
- **D2** Re-parameterisation branches are trained unmerged; fusing is not needed because evaluation uses the unmerged equivalent form.
- **D3** Early-stopping patience 10 instead of 20; 3 seeds instead of 5 (compute and the task's requirement).
- **D4** Fixed L = 336 for every model; the paper re-ran baselines with the best input length, so absolute numbers are not directly comparable.
- **D5** TCN baseline re-implemented from Bai et al.; Lab 4 TCN unavailable. Not claimed identical to Lab 4.
- **D6** ERF procedure adapted from ModernTCN Sec. 5.2 / the audit repo; Target B added; Lab 4 ERF procedure unavailable; no compatibility with Lab 4 is claimed.
- **D7** All test windows are evaluated (`drop_last=False`), unlike the reference ModernTCN scripts.
- **D8** RevIN `affine=False`, statistics detached (reference default); gradients therefore exclude dependence through the instance statistics.
- **D9** For `k=3` the small kernel is 3 (rule `small=min(5,k)`); the paper does not say what it used. A `small=1` variant is a possible sensitivity check only if compute allows.
- **D10** ETTm1 ablation may be reduced if the measured cost is too high (see `experiment_matrix.md`, tiers); any reduction is recorded as such.

## Amendments

### A1 - 2026-10-20, implementation phase (made before any training run; no model result has been seen)
1. **LR schedule, exact form.** "type3" (ModernTCN recipe) = base lr for epochs 1-2, then `base * 0.9^(epoch-3)` (1-based epochs); DLinear uses "type1" = `base * 0.5^(epoch-1)`, as in the respective reference trainers (re-read). The earlier wording "keeps lr for 3 epochs" in Sec. 8 / `sources.md` was imprecise.
2. **DLinear initialisation:** PyTorch default `nn.Linear` init. The reference code's constant `1/L` initialisation is commented out in the version read.
3. **D11 - micro-batching for ETTm1 ModernTCN (K=3, r=8).** Measured saved-for-backward activations are 26.4-27.9 MiB/sample (`scripts/measure_activation_memory.py`), i.e. 13.2-13.9 GiB at batch 512 before gradient buffers and cuDNN workspace, which does not fit a free 15 GB GPU. The effective batch stays 512 via 4 micro-batches of 128 (gradient accumulation) for **every kernel size and seed** of this model; the accumulated gradient equals the full-batch gradient (unit-tested) except that BatchNorm batch statistics are computed over 128 instead of 512 samples. `micro_batch` is part of the config hash. Evaluation batch 512 (evaluation is batch-size invariant, unit-tested).
4. **Strict fp32.** TF32 disabled for matmul and cuDNN; `cudnn.deterministic=True`, `benchmark=False`; `use_deterministic_algorithms` is *not* enabled (may raise for depthwise-conv backward on CUDA), so small run-to-run GPU nondeterminism is possible and is covered by the seed spread. Recorded as `precision` in `val_metrics.json`.
5. **Ledgers.** Training/validation bookkeeping: `results/runs.csv`. Test records: only `results/test_runs.csv` (append-only) and `<run>/test_metrics.json`, written by `evaluate_test`, which refuses to run unless `configs/frozen_v1.json` exists with `"frozen": true`, refuses unfinished runs and refuses to overwrite. (The matrix previously described test columns inside `runs.csv`.)
6. **Run re-use.** `phase` is not part of the config hash, so the P1 run with the selected hyperparameters (seed 2021, k=51) *is* the P3 seed-2021 run; `scripts/aggregate.py::select_main` counts a P1 run in the reported matrix only if it matches the frozen hyperparameters (and seed 2021, k=51).
7. **Early stopping:** a run stops after `patience` consecutive epochs without a *strict* improvement of validation MSE; a crash at the stopping point is resumed without further training.
8. **Non-finite training:** a non-finite loss aborts the run as `failed_nonfinite_loss` (marker file `FAILED`, no ledger row, no `DONE`); the grid continues.
9. **Target A and the residual path.** The backbone feature at a token includes the identity residual path, so Target A is expected to be sharply localised at the target token plus the kernel reach; unit tests verify exact zeros outside `S*(pos -/+ blocks*(k//2))` raw steps. E1 should be read in that sense.
