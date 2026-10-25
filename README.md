<div align="center">

# ModernTCN-Style Large Kernels vs. Dilation

**An independent, reproducible PyTorch study of large-kernel temporal convolutions for multivariate long-horizon forecasting**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Stage](https://img.shields.io/badge/stage-code%20locked%20%7C%20awaiting%20Colab%20run-blue.svg)](#project-status)
[![Tests](https://img.shields.io/badge/tests-102%20passing-brightgreen.svg)](tests/)
[![Datasets](https://img.shields.io/badge/data-ETTh1%20%7C%20ETTm1-lightgrey.svg)](https://github.com/zhouhaoyi/ETDataset)

*Does a larger measured effective receptive field actually buy forecasting accuracy?*

</div>

---

## Project status

> **Read this first.** All **pre-training code is written and locally verified** (102 unit/integration tests pass; 20/20 injected bugs are detected by the suite).
> **No result exists yet:** nothing has been trained for reporting, no GPU has been used, and no test-split number has been computed. Every number below is either a property of the data, an analytic/measured *cost* figure, or a test outcome.

| Component | State | Evidence |
|---|---|---|
| Source audit, data audit, protocol (v0.1 + amendment A1), experiment matrix, risk register | done | [`research_notes/`](research_notes/), [`logs/data_audit.json`](logs/data_audit.json) |
| Data pipeline (chronological split, train-only z-score, leakage-safe windows) | done, tested | [`src/data.py`](src/data.py), `tests/test_data.py` (gate G0) |
| ModernTCN-style model, independent TCN, DLinear | done, tested | [`src/models/`](src/models/); real parameter counts equal the analytic table (G1), shape/grouping/causality tests (G2) |
| Training loop (resumable, ledger, micro-batching), guarded one-shot test evaluation | done, tested | [`src/train.py`](src/train.py); smoke on real data (G3), resume equivalence (G5) |
| ERF measurement (Target A backbone, Target B forecast Jacobian probes) | done, tested | [`src/receptive_field.py`](src/receptive_field.py); probe estimator equals exact Jacobian (G6) |
| Experiment grid, freeze, test-eval, ERF runner, aggregation + claim rule, figures, Colab notebook | done, tested end-to-end on real ETTh1 (DLinear, 1 epoch) | [`scripts/`](scripts/), [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb) |
| **GPU runtime / memory measurement (G4)** | **not done - needs Colab** | `scripts/benchmark_runtime.py` |
| Hyperparameter selection, 3-seed runs, ablation, ERF on trained models, claim verdict, failure analysis, `reports/Discussion.md` | *not started* | - |

Live state and exact next steps: [`PROJECT_STATE.md`](PROJECT_STATE.md), [`HANDOFF.md`](HANDOFF.md).

---

## Scientific question

> When do **large depthwise temporal kernels** give better forecasting accuracy, or a different **effective receptive field (ERF)**, than **dilation-based** temporal convolutions — and does the measured ERF explain accuracy?

Three separate sub-questions are kept apart on purpose:

1. Does the ModernTCN-style architecture improve accuracy over a dilated TCN and over DLinear?
2. Does changing the temporal kernel size change the *empirical* ERF?
3. Does a larger measured ERF correspond to better accuracy in the tested settings?

Nothing is assumed in advance: larger kernels, larger ERF or more complexity are hypotheses, not premises. A negative result is an acceptable outcome.

## Fixed experimental specification

| | |
|---|---|
| Datasets | ETTh1 (1 h), ETTm1 (15 min), 7 variables, multivariate-to-multivariate |
| Input window | $L = 336$ |
| Horizons | $H \in \{96, 336\}$ |
| Metrics | MSE and MAE on **train-statistics z-scored data** |
| Models | independent **TCN** (Bai et al., 2018), independent **ModernTCN-style** model (Luo & Wang, ICLR 2024), independent **DLinear** (Zeng et al., 2023) |
| Seeds | 2021, 2022, 2023 — mean ± sd (ddof = 1) |
| Ablation | large depthwise kernel $k \in \{3, 13, 31, 51, 83\}$ (patch-token units), parameter spread ≤ 10 % |
| Selection | validation split only; test split evaluated **once**, after the protocol is frozen |

## What the audit has already established (verified from the files)

| Fact | Value |
|---|---|
| Files | ETTh1: 17,420 rows · ETTm1: 69,680 rows · 7 numeric columns · no NaN · no duplicate timestamps |
| Benchmark split | 12 / 4 / 4 months → ETTh1 rows `[0,8640)`, `[8640,11520)`, `[11520,14400)`; ×4 for ETTm1 |
| Rows actually used | 14,400 (82.7 %) / 57,600 — the benchmark ignores the file tail |
| Test period | 2017-10-24 → 2018-02-20 |
| **ETTh1 ⊂ ETTm1** | ETTh1 is *exactly* every 4th row of ETTm1 → the two datasets are **not independent evidence** |
| Windows ($L=336$) | ETTh1 H=96: 8,209 / 2,785 / 2,785 · H=336: 7,969 / 2,545 / 2,545 (train/val/test) |
| ETTm1 windows | H=96: 34,129 / 11,425 / 11,425 · H=336: 33,889 / 11,185 / 11,185 |
| **`drop_last` effect** | A test loader with batch 512 and `drop_last=True` silently discards 8.08 % (H=96) / 19.53 % (H=336) of ETTh1 test windows and 1.41 % / 3.87 % for ETTm1. This study evaluates **all** windows. |
| Distribution shift | In train-σ units the ETTh1 test `OT` mean is −1.34 and its σ is 0.34× the train σ (descriptive only; a *hypothesis* for later failure analysis) |

Reproduce: `python scripts/audit_data.py --data_dir <ETT-small dir> --out logs/data_audit.json`

## Design at a glance

**ModernTCN-style block** (tensor shapes, $B$ batch, $M=7$ variables, $D=64$, $N=84$ tokens, $C=MD=448$):

```
[B, 336, 7] → RevIN → [B, 7, 336] → pad last value ×4 → [B, 7, 340]
  → stem Conv1d(1→D, kernel 8, stride 4) + BN → [B·M, D, 84] → [B, C, 84]
  → K × { DWConv(k, groups=C) ‖ DWConv(small, groups=C)  (+BN, summed)   mixes TIME only
          → BN(D)
          → ConvFFN1 (grouped 1×1, groups=M)                              mixes FEATURES within a variable
          → ConvFFN2 (grouped 1×1, groups=D, after permute)               mixes VARIABLES per feature
          → residual }
  → flatten [B, 7, D·N] → Linear(D·N → H) → RevIN⁻¹ → [B, H, 7]
```

Kernel size is interpreted **in token space**: one depthwise kernel of $k$ tokens spans $4(k-1)+8$ raw steps — 16, 56, 128, 208 and **336** (the whole window) for $k = 3, 13, 31, 51, 83$.

**Parameter counts for the kernel ablation** (analytic formula; **equal to `sum(p.numel())` of the instantiated modules** in all 20 cells — gate G1):

| Setting | k=3 | k=13 | k=31 | k=51 | k=83 | spread |
|---|---|---|---|---|---|---|
| ETTh1 H=96 (K=1, r=1) | 586,912 | 592,288 | 600,352 | 609,312 | 623,648 | 6.26 % |
| ETTh1 H=336 | 1,877,392 | 1,882,768 | 1,890,832 | 1,899,792 | 1,914,128 | 1.96 % |
| ETTm1 H=96 (K=3, r=8) | 2,081,696 | 2,097,824 | 2,122,016 | 2,148,896 | 2,191,904 | 5.29 % |
| ETTm1 H=336 | 3,372,176 | 3,388,304 | 3,412,496 | 3,439,376 | 3,482,384 | 3.27 % |

Width, depth, FFN ratio and head are identical across kernels, so **kernel size is the only intentional change**. The linear head holds 85–95 % of the ETTh1 parameters, which is a confound the report must discuss.

## ERF measurement (methodological adaptation)

The original Lab-4-style autograd procedure was not available, and the official ModernTCN repository publishes no ERF code. The procedure used here is therefore an **explicit adaptation**, built from ModernTCN §5.2 and the audit repository's routine, and makes no claim of equivalence to anything unseen:

- **Target A (backbone, paper-style):** gradient of $\sum \mathrm{ReLU}(z_{\cdot,\cdot,p})$ at the centre (and last) position of the final feature map w.r.t. the input, ReLU-ed, summed over variables, averaged over 50 fixed validation windows.
- **Target B (forecast ERF, added by this study):** root-mean-square Jacobian magnitude $b(\tau)$ of the *forecast* w.r.t. the input at lag $\tau$, estimated with Rademacher probes ($E_u[v^2]=\sum J^2$); exact for DLinear.
- Same samples, lag axis ($\tau=0$ = newest step), normalisation (max-normalised profile, cumulative-mass widths $r_{50}, r_{90}, r_{95}$) and eval-mode conventions for every model.
- **Pre-registered expectation:** because the head reads all 84 tokens, B should be much flatter in $k$ than A. If not, that expectation is falsified and reported as such.

Full mathematics: [`research_notes/research_protocol.md`](research_notes/research_protocol.md) §6.

## Quantitative claim under test (selected before any results)

> *"Results on Table 16 show that increasing the kernel size leads to performance improvement."* — ModernTCN, Appendix F.2.1

Tested on ETTh1 with $k \in \{3, 31, 51\}$ (plus the extensions $13, 83$ and ETTm1) using a paired-by-seed decision rule fixed in advance (protocol §7). The paper's own numbers are not monotone between $k=31$ and $k=51$; that is part of what is examined.

## Repository layout

```text
src/
  data.py               chronological split, train-only scaler, window datasets
  models/               modern_tcn.py · tcn.py · dlinear.py · revin.py
  train.py              resumable training, run ledger, guarded one-shot test evaluation
  evaluate.py           global-sum MSE/MAE (batch-size invariant), naive references
  receptive_field.py    ERF Targets A/B, width metrics
  grid.py               phases P1/P3/P4/P6 generated from the protocol
scripts/                run_grid · freeze_protocol · run_test_eval · run_erf · aggregate · make_figures
                        smoke_test · benchmark_runtime · measure_activation_memory · verify_checkpoints
                        audit_data · param_count_analytic · mutation_check · make_notebook
tests/                  102 tests: data, models, ERF, training, scripts, real-data end-to-end
notebooks/              colab_runner.ipynb (TRAIN flag; TRAIN=False verifies saved checkpoints)
research_notes/         sources · research_protocol (v0.1 + A1) · experiment_matrix · risks
logs/                   data_audit · param_counts_analytic · activation_memory · smoke_cpu_* · mutation_report
configs/  results/  reports/   (filled when the experiments run; frozen_v1.json is written once after P1)
```

Raw ETT CSVs, checkpoints and caches are not committed (see `.gitignore`).

## Getting started

```bash
git clone <repository-url> moderntcn-large-kernel-study
cd moderntcn-large-kernel-study
pip install -r requirements.txt
git clone --depth 1 https://github.com/zhouhaoyi/ETDataset.git ../ETDataset      # data (not versioned)

export ETT_DIR=$PWD/../ETDataset/ETT-small
python -m pytest -q                                      # gates G0-G6 (102 tests; real-data tests need ETT_DIR)
python scripts/audit_data.py --data_dir $ETT_DIR         # data facts -> logs/data_audit.json
python scripts/smoke_test.py --data_dir $ETT_DIR --device cpu        # G3
python scripts/mutation_check.py                         # optional: proves the tests can fail (~4 min)
```

**On a GPU (Colab):** open `notebooks/colab_runner.ipynb`, or run the phases by hand:

```bash
python scripts/benchmark_runtime.py --data_dir $ETT_DIR --device cuda      # G4: s/epoch, peak memory
python scripts/run_grid.py --phase p1 --data_dir $ETT_DIR --device cuda    # validation-only hyperparameter selection
python scripts/freeze_protocol.py                                           # writes configs/frozen_v1.json (once)
python scripts/run_grid.py --phase p3 --data_dir $ETT_DIR --device cuda    # 3 models x 4 settings x 3 seeds
python scripts/run_grid.py --phase p4 --tier h1 --data_dir $ETT_DIR --device cuda   # kernel ablation (tier per G4)
python scripts/run_erf.py --data_dir $ETT_DIR --device cuda
python scripts/run_test_eval.py --data_dir $ETT_DIR --device cuda          # test split, once per run, only after freezing
python scripts/aggregate.py && python scripts/make_figures.py
```

Interrupted runs resume from `last.pt`; finished runs are skipped; failed runs never stop the grid.

## Roadmap

- [x] Source retrieval and audit; data audit; protocol, matrix, risks
- [x] Data pipeline + tests (borders, scaler, windows, no target overlap, batch-size-invariant metrics)
- [x] ModernTCN-style model + tests (grouping/channel order, shapes, parameter counts = analytic table)
- [x] Independent TCN and DLinear + tests
- [x] Training/evaluation with resume, ledger, one-shot test guard; ERF module; grid/aggregate/figure scripts; Colab notebook
- [x] Local verification: 102 tests, real-data smoke runs, end-to-end pipeline, mutation check (20/20 detected)
- [ ] **G4 on a Colab GPU** (runtime, memory) → pick the ETTm1 ablation tier
- [ ] Hyperparameter selection on validation → freeze configuration
- [ ] Main 3-seed results; kernel ablation; ERF on trained models; claim verdict
- [ ] Failure analysis (validation first, interpretation frozen before test) → single test evaluation
- [ ] `reports/Discussion.md`, figures, checkpoints link

## Limitations to keep in mind

- Results will concern **one transformer and one test winter**; ETTh1 and ETTm1 are the same series at two resolutions.
- Three seeds give little statistical power; per-seed values are shown and no significance claims are made.
- The model is a **single-stage, simplified** ModernTCN-style network; the TCN baseline is an independent re-implementation, not a reproduction of any earlier lab code.
- Published numbers used input lengths searched per model and sometimes `drop_last=True` test loaders; direct comparison with them is not valid.
- Compute: the ETTm1 ModernTCN is about 50× costlier per epoch than ETTh1 (analytic) and needs ~14 GiB of saved activations at batch 512 (measured); it is trained with 4×128 micro-batches. The full grid will not fit a 30-minute free-GPU session, so the notebook has a `TRAIN=False` checkpoint-verification mode.
- No CUDA code path has been executed in the authoring environment (CPU only); the first Colab run is the first GPU execution.

## Provenance and transparency

- The authors' implementation was **read, not copied**; the files read are listed in [`research_notes/sources.md`](research_notes/sources.md) §10. Models are written from the paper's equations and figures.
- Retrieval limits are stated, not hidden: the audit paper was read in its *submission* version; the TCN and DLinear paper bodies were not retrieved (code and abstracts only).

## References

1. D. Luo, X. Wang. *ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis.* ICLR 2024. [OpenReview](https://openreview.net/forum?id=vpJMJerXHU) · [code](https://github.com/luodhhh/ModernTCN)
2. *ModernTCN Revisited: A Critical Look at the Experimental Setup in General Time Series Analysis.* TMLR 2025. [OpenReview](https://openreview.net/forum?id=R20kKdWmVZ) · [code](https://github.com/onderakacik/RE-ModernTCN)
3. A. Zeng et al. *Are Transformers Effective for Time Series Forecasting?* AAAI 2023. [arXiv:2205.13504](https://arxiv.org/abs/2205.13504)
4. S. Bai, J. Z. Kolter, V. Koltun. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* [arXiv:1803.01271](https://arxiv.org/abs/1803.01271), 2018.
5. H. Zhou et al. *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.* AAAI 2021. [arXiv:2012.07436](https://arxiv.org/abs/2012.07436) · [ETDataset](https://github.com/zhouhaoyi/ETDataset)
6. W. Luo et al. *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks.* NeurIPS 2016 (cited by ModernTCN §5.2; not independently retrieved here).
