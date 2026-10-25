# Risk Register

## Status update after the implementation phase (2026-10-20)

| Risk | Status |
|---|---|
| R1.1-R1.6 leakage | Mitigated by tests: train-only scaler, no target overlap, context policy, batch-composition invariance, NaN-poison canary on test rows/tail, guarded one-shot test evaluation. Residual: process discipline (a human can still run code that reads test rows). |
| R2.1, R2.5 | `drop_last=False` everywhere; single global metric aggregation; batch-size invariance unit-tested. |
| R3.1-R3.3 | Closed by controlled-perturbation tests (depthwise = time only; ConvFFN1 = within variable; ConvFFN2 = across variables per feature; permutation hooks; padding arithmetic). |
| R4.1, R4.2 | **Closed**: real parameter counts equal the analytic table in all 20 (setting, k) cells; only `dw_large` (and `dw_small` at k=3) differ across kernels. |
| R5.2-R5.8 | Probe estimator verified against exact Jacobians; lag orientation, support and determinism tested. Open: how Target A/B behave on *trained* models (scientific, not a code risk). |
| R6.1 | **Open until G4 on a GPU.** New finding: ETTm1 ModernTCN needs ~14 GiB of saved activations at batch 512 -> micro-batching (A1.3). |
| R6.5 | torch now imports and runs on the authoring CPU; no GPU/CUDA code path has been executed anywhere. Colab-specific hazards found by review and fixed: CUDA-resident RNG state on resume, TF32 defaults. Residual: unexecuted CUDA paths. |
| R3.5, R7.x | Unchanged (process/interpretation risks). |

Format: **risk** - evidence (verified / hypothesis) - mitigation - residual risk. Status as of 2026-10-20 (pre-implementation).
"Verified" means established from retrieved sources or from `logs/data_audit.json`; "Hypothesis" means not yet tested.

## R1. Data leakage

| ID | Risk | Evidence | Mitigation | Residual |
|---|---|---|---|---|
| R1.1 | Normalisation statistics computed on val/test | Reference loaders fit the scaler on train rows only (verified in S2, S6) | `fit` on `[0, train_end)` only; unit test compares scaler stats with a recomputation on the train slice and asserts val/test never enter | Low |
| R1.2 | Target overlap across splits (a validation target appearing in a training window or test target in validation) | Border formulas verified; first val/test target rows are 8,640 / 11,520 (ETTh1) | Test: for every window, `target_start >= split_start`; no target index of one split is inside another split's target range | Low |
| R1.3 | Historical context at boundaries | Val/test inputs reach back 336 rows into the previous split (benchmark convention, verified) | Policy documented (protocol Sec. 3); contexts only contain information available at forecast origin; test asserts input windows end before the first target step | Context from the preceding split is a convention, not leakage |
| R1.4 | Test split inspected during development | Reference trainer evaluates test every epoch (verified in S2); audit calls out test-set misuse in other tasks | Training code has no test code path; `evaluate_test` is a separate entry point, one record per run ID, append-only; protocol frozen (config hash) before P8 | Process discipline; mitigated by the ledger, not impossible to bypass manually |
| R1.5 | Hyperparameters tuned on test via repeated "final" runs | - | Selection only from validation (P1); hyperparameters frozen once; no per-kernel retuning | Low |
| R1.6 | RevIN or batch statistics leaking across samples | RevIN uses per-instance statistics of the *input* window only; BatchNorm uses running statistics in eval | Eval-mode tests: output for a window is identical whether computed alone or inside a batch (tolerance 1e-5) | Low |
| R1.7 | ETTh1 and ETTm1 share the same underlying series (ETTh1 is the every-4th-row subsample; verified) | `logs/data_audit.json`: identical timestamps, max abs diff 0.0 | Never treat the two datasets as independent evidence; state this in the report; no cross-dataset checkpoint selection | Conclusions are about **one transformer, one test winter (2017-10-24..2018-02-20)** |

## R2. Benchmark inconsistency

| ID | Risk | Evidence | Mitigation | Residual |
|---|---|---|---|---|
| R2.1 | `drop_last=True` on the test loader inflates/deflates scores | Verified: ETTh1 loses 8.08 % (H=96) / 19.53 % (H=336) of test windows at batch 512, ETTm1 1.41 % / 3.87 %; the audit repo's fix is `drop_last=False` | All windows are evaluated; metric = sum of errors / number of values, independent of batch size; test asserts metric invariance to batch size | Reported numbers are not comparable to published tables that used `drop_last=True` |
| R2.2 | Published numbers use different input lengths per model | Paper Table 27 caption: input length "searched to the best" for each model | L = 336 for every model; no direct comparison to published tables; the paper's values are used only for sign/size of effects | Absolute values will differ; documented (D4) |
| R2.3 | The "6:2:2" statement vs. the rows actually used | Verified: 14,400 of 17,420 (82.7 %) / 57,600 of 69,680 rows used | Document; keep the benchmark convention for comparability; the unused tail is not touched | Possible distribution difference to the unused tail is out of scope |
| R2.4 | Different loaders disagree historically (borders, scalers, drop_last) | Current TSL and the reference ModernTCN loader agree on borders/scaler (verified); only drop_last differs | Own loader re-derives borders from the 12/4/4-month rule and is tested against the audit counts | Older library versions not reviewed |
| R2.5 | Metric aggregation differences (per-batch mean vs. global mean) | Hypothesis | Single global aggregation implemented once in `src/evaluate.py` and unit-tested | Low |
| R2.6 | Seed count differs from the paper (5) and the reference scripts (`--itr 1`) | Verified | 3 predefined seeds; report mean +/- sd and per-seed values; no significance claims | Low statistical power (see R7.2) |

## R3. Implementation fidelity

| ID | Risk | Evidence | Mitigation | Residual |
|---|---|---|---|---|
| R3.1 | Grouped-conv channel ordering wrong (ConvFFN1 groups = M must group by variable; ConvFFN2 groups = D must group by feature index) | Paper App. G.2 / Table 19 (verified) | Unit tests with *controlled perturbations*: changing variable `m` at the input of ConvFFN1 must not change other variables' outputs; for ConvFFN2, changing feature `d` of all variables must only affect that feature index; DW conv must not mix variables or features | Medium until tests exist |
| R3.2 | DW conv does not act purely in time | Paper: DWConv is "both variable and feature independent" (verified) | Test: output channel `c` depends only on input channel `c` (Jacobian block structure) | Low |
| R3.3 | Patch/pad arithmetic (N = 84) wrong; last-value padding forgotten | Paper App. B.1 (verified) | Shape tests: padded length 340, N = 84 for L=336, P=8, S=4 | Low |
| R3.4 | Unfaithful simplifications (single stage, no time features, unmerged re-param) | Deviations D1-D2 | Documented; none changes the question asked (kernel size effect at stage 1) | The result applies to this simplified model, not to the full multi-stage ModernTCN |
| R3.5 | Accidental copying from the reference | Rules require independent implementation | Reference code read-only in a scratch dir outside the repo; model written from paper equations/figures; files read are listed (sources.md Sec. 10); reviewer-explainable code | Provenance is by process, not by tooling |
| R3.6 | TCN baseline is not "the" Lab 4 TCN | Lab 4 unavailable (stated project constraint) | Independent Bai-et-al. implementation; named "independent TCN baseline"; adaptation recorded (D5) | Comparison to a different TCN could change magnitude of Q1 conclusions |
| R3.7 | TCN design choices (single last-step vector head; 64 channels) may handicap it | Hypothesis | Report as a limitation; optionally add a flatten-head TCN variant only if compute allows (not required) | Q1 conclusion is conditional on the chosen TCN design |
| R3.8 | DLinear without RevIN vs. others with RevIN | Verified from code structure | Report as confound; optionally include ModernTCN-no-RevIN diagnostic | Medium for Q1 comparisons |

## R4. Parameter-count constraint

| ID | Risk | Evidence | Mitigation | Residual |
|---|---|---|---|---|
| R4.1 | Real modules differ from the analytic formulas (BN, bias, grouping) | Analytic only (not instantiated) | Gate G1: test equality with `sum(p.numel())` and re-derive the spread; **do not train before it passes** | **Open** |
| R4.2 | Kernel change alters other parameters (e.g. head) | Head depends on `D*N`, not on `k` | Test that every parameter outside `dw_large` is identical across kernels (names + shapes) | Low |
| R4.3 | `small = min(5,k)` couples a second knob to `k` at k=3 | Deviation D9 | Documented; difference is 2*448 params per block (<0.2 %) | Slight confound at k=3 |
| R4.4 | The 10 % rule is satisfied only for the *ablation*, not across model families | Verified (analytic) | Report parameter counts in every table | Not an error, but readers may over-interpret model-family comparisons |
| R4.5 | Head dominates parameters (85-95 %); "kernel matters" competes with head capacity/overfitting | Verified (analytic) | Interpret kernel effects relative to head; failure hypothesis H-c | Interpretation risk |

## R5. ERF measurement

| ID | Risk | Evidence | Mitigation | Residual |
|---|---|---|---|---|
| R5.1 | Lab 4 ERF procedure unavailable; method not identical | Stated project constraint | Adapted protocol with explicit math (protocol Sec. 6); no claim of compatibility | Cannot be compared with Lab 4 plots |
| R5.2 | Backbone ERF (A) at the center token does not measure what the forecast uses | The head is linear over all tokens (verified in paper App. B.2) | Add Target B (forecast ERF) and `p_last`; report both and their disagreement (pre-registered E2) | Interpretation risk; mitigated by reporting both |
| R5.3 | ReLU on gradients (reference convention) discards sign information and can inflate "support" | Verified in S4 code | B is sign-agnostic; report A as reference-style only; compare shapes, not absolute magnitudes | Moderate |
| R5.4 | Gradient-based ERF depends on trained weights, input samples and normalisation | Hypothesis | Fixed deterministic 50-sample set; mean +/- sd over seeds; max-normalised profiles plus cumulative-mass widths | Moderate |
| R5.5 | Detached RevIN statistics remove an input path (flat component) | Verified (RevIN `.detach()`) | Same convention in all models; documented (D8) | Low |
| R5.6 | Zero-padding / boundary effects: central-token ERF truncated at window edges; last-value padding injects extra gradient on the last raw step | Hypothesis from architecture | Plot offsets and lags; mark the window edges; test synthetic impulses to see the padding effect | Low-moderate |
| R5.7 | Probe estimator (Rademacher, R=16) noise | Hypothesis | Convergence check R=16 vs. R=64; exact DLinear Jacobian as a ground-truth test | Low |
| R5.8 | Audit routine counts DataLoader iterations, not necessarily 50 samples | Verified code loop; batch size not verified | The implemented routine takes exactly 50 windows, one at a time | Low |
| R5.9 | Theoretical RF != empirical ERF (TCN RF 509 vs. 336 window) | Expectation E3 | Report both; do not equate RF with ERF | By design |

## R6. Compute and engineering

| ID | Risk | Evidence | Mitigation | Residual |
|---|---|---|---|---|
| R6.1 | ETTm1 ModernTCN (K=3, r=8) is ~28 TFLOP/epoch (50x ETTh1); full grid does not fit a free Colab session | Analytic (Sec. 4 of the matrix) | Measure first (G4); tiered execution; reduce Tier 3 or the P1 grid for ETTm1 rather than seeds | **Open until measured** |
| R6.2 | Session disconnects lose progress | - | Per-epoch `last.pt`, resumable run IDs, DONE markers, ledger append-only; resume test (G5) | Low |
| R6.3 | Nondeterminism (cuDNN grouped/depthwise conv) breaks bitwise reproducibility | Hypothesis | Seed all RNGs, record `torch.backends.cudnn` flags; accept small run-to-run noise and report seed spread; do not claim bitwise reproducibility | Low |
| R6.4 | The 30-minute notebook requirement conflicts with the grid size | Conflict identified | `TRAIN=False` loads checkpoints from an external location; evaluation/plots reproduce from checkpoints and ledger | Needs hosting for checkpoints |
| R6.5 | Authoring sandbox has no GPU and a partly installed CUDA stack; tests so far are pure Python | Verified | Gate G1-G3 on Colab or after finishing the install (needs `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`; ~1.9 GB disk free) | Nothing neural has been executed yet |
| R6.6 | Repository README previously claimed `tcn.py`/`modern_tcn.py` were implemented while only the audit scripts exist | Verified by cloning the GitHub repo (2026-10-20) | README rewritten to state only verified status; the status block is updated at each milestone | Keep the README in sync with `PROJECT_STATE.md` |

## R7. Interpretation of negative or null results

| ID | Risk | Mitigation |
|---|---|---|
| R7.1 | Treating "claim not supported" as "paper wrong" | Report differences in setup (D1-D10) first; separate *replication under the protocol of this study* from *the paper's result* |
| R7.2 | Three seeds give little power; sd estimates are noisy | Show paired per-seed differences; verdict categories (supported / partial / not supported / inconclusive); no p-values |
| R7.3 | Explaining a failure with a plausible but untested story | Hypotheses labelled H-a..H-f until tested by a diagnostic; separate "observed" from "proposed mechanism"; state the discriminating experiment |
| R7.4 | Post-hoc selection of the failure setting or of the claim | Both fixed in the protocol before results (Sec. 7, 10); failure identification uses validation only and is frozen before the test phase |
| R7.5 | Accuracy-vs-ERF correlation over few points (5 kernels x 4 settings) read as causation | Report per-setting tables, rank agreement only descriptively, discuss confounders (head capacity, optimisation, seed noise); the ablation is the only causal manipulation, ERF is a measured covariate |
| R7.6 | Expecting ERF A and B to agree | Pre-registered E2 predicts they differ; a surprise either way is reported |
| R7.7 | Over-generalising from one transformer / one winter test period | State scope in the report's limitations (R1.7) |
| R7.8 | Unverified secondhand numbers from the audit paper leaking into the report | Effect sizes from Revisited are marked **not re-verified** in `sources.md`; re-check before quoting |

## R8. Documentation and provenance

| ID | Risk | Mitigation |
|---|---|---|
| R8.1 | Quoting papers from memory or from unretrieved bodies | Only retrieved text is quoted (ModernTCN full text, audit submission version); DLinear/TCN paper bodies are explicitly **not** used for claims |
| R8.2 | Camera-ready audit paper differs from the submission version read | Declared in `sources.md`; re-check only if a specific statement is quoted in the report |
| R8.3 | Handoff loss between sessions | `PROJECT_STATE.md`, `HANDOFF.md`, `CHANGELOG.md`, and `deliverables/latest_checkpoint.zip` updated at every milestone |
