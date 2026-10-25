# PROJECT_STATE

Last updated: 2026-10-20 (checkpoint 2: **pre-training code complete and locally verified**).
Stage: code locked for the first Colab GPU run. **No result has been produced**: nothing trained for reporting, no GPU used, no test-split number computed.

## 1. Completed and verified

| Item | How it was verified (actually run) | Result |
|---|---|---|
| Source audit, data audit, protocol v0.1 + amendment A1, experiment matrix v0.2, risk register | see `research_notes/` | done |
| Data pipeline `src/data.py` | `tests/test_data.py`: borders, window counts == `logs/data_audit.json`, train-only scaler (perturbing val/test/tail rows leaves train data unchanged), no target overlap, context policy, batch-size-invariant metrics | 20 pass |
| Models `src/models/*` | `tests/test_models.py`: **real parameter counts equal the analytic table in all 20 ModernTCN cells + TCN + DLinear; ablation spread <= 6.26 %**; only `dw_large` (and `dw_small` at k=3) differ across kernels; grouping/permutation/depthwise tests; padding arithmetic; TCN causality and RF=509; RevIN shift-equivariance; batch invariance | 37 pass |
| ERF `src/receptive_field.py` | `tests/test_erf.py`: probe estimator == exact Jacobian (DLinear, tiny ModernTCN, tiny TCN), convergence R=16 -> 256, lag orientation, exact-zero support outside the architectural reach, TCN causality, determinism, mode restoration | 12 pass |
| Training/eval `src/train.py`, `src/evaluate.py` | `tests/test_train.py`: loss decreases (3 models), resume reproduces the uninterrupted run bit-for-bit (log + best weights), skip-done, tampered config refused, early stopping, NaN failure handled, ledger content, **NaN-poisoned test rows/tail do not affect training/validation**, test evaluation guarded/one-shot/uses best checkpoint, micro-batch == full-batch gradient, CPU-resident RNG state | 20+ pass |
| Scripts/grids/notebook | `tests/test_scripts.py`: grid sizes (32/36/48/24/30), frozen-hparam injection, freeze selection + tie rule + write-once, aggregation (ddof=1), claim C1 rule, figures, notebook JSON + cell syntax, **real-data end-to-end pipeline** (P1 -> freeze -> P3 -> ERF -> guarded test -> aggregate -> verify checkpoints -> figures) | pass |
| Whole suite | `python -m pytest tests -q` | **102 passed** |
| Test-suite strength | `scripts/mutation_check.py` (20 injected bugs) -> `logs/mutation_report.json` | **20/20 detected** |
| G3 smoke on real data (CPU) | `logs/smoke_cpu_ETTh1_H336.json`, `logs/smoke_cpu_ETTm1_H336.json` (all 3 models: finite, loss decreases; ETTm1 H=336 ModernTCN 3,439,376 params) | passed |
| Memory | `logs/activation_memory.json`: ETTm1 ModernTCN saves 26.4-27.9 MiB/sample = **13.2-13.9 GiB at batch 512** -> micro-batch 128 (amendment A1.3) | measured (lower bound) |

## 2. Bugs found and fixed during verification (so they are not re-introduced)

1. `aggregate.summarize` mislabelled the sd column (lambda names) -> rewritten with explicit reductions.
2. Re-used P1 run (same run id as P3 seed 2021) was dropped by the phase filter, leaving 2 seeds -> `select_main` uses the frozen hparams.
3. Resume loaded `last.pt` with `map_location=device`; on CUDA the RNG state would be a CUDA tensor and `torch.set_rng_state` would raise -> load on CPU (tested).
4. Crash exactly at the stopping point trained one extra epoch on resume -> `pre_stopped`.
5. TF32 defaults (cuDNN) would silently change numerics on A100 -> disabled.
6. Test-suite defects found on review (unasserted RevIN detach check; wrong perturbation index in the TCN reach test; meaningless `pytest.raises` tail) -> fixed; mutation check added.

## 3. Repository inventory

```
README.md PROJECT_STATE.md HANDOFF.md CHANGELOG.md requirements.txt .gitignore
src/{data.py,evaluate.py,train.py,grid.py,receptive_field.py,models/{__init__,modern_tcn,tcn,dlinear,revin}.py}
scripts/{run_grid,freeze_protocol,run_test_eval,run_erf,aggregate,make_figures,smoke_test,benchmark_runtime,
         measure_activation_memory,verify_checkpoints,audit_data,param_count_analytic,mutation_check,make_notebook,_bootstrap}.py
tests/{conftest,test_data,test_models,test_erf,test_train,test_scripts}.py
notebooks/colab_runner.ipynb
research_notes/{sources,research_protocol,experiment_matrix,risks}.md
logs/{data_audit,param_counts_analytic,activation_memory,runtime_cpu,mutation_report,smoke_cpu*}.json
configs/ reports/ results/ (.gitkeep)    deliverables/latest_checkpoint.zip
```
The remote repository is **not updated from this environment** (no credentials): the checkpoint contents are to be copied over it.

## 4. Experiment status

No experiment has been run. No interrupted or failed runs exist. `configs/frozen_v1.json` does not exist (test evaluation is therefore impossible by construction).

## 5. Decisions (see `research_protocol.md` Sec. 11 and Amendment A1)

Independent TCN (no Lab 4); adapted ERF protocol (Targets A/B); kernels {3,13,31,51,83}; claim C1 with paired-by-seed rule (0.005); all test windows evaluated; seeds 2021-2023; patience 10; ETTm1 ModernTCN micro-batch 128 (D11); strict fp32; test split touched only by `evaluate_test` after freezing; P1 run re-use for P3 seed 2021.

## 6. Known limitations and open items

- **G4 not done**: no GPU runtime/memory measurement; all hour estimates in the matrix are analytic. `logs/runtime_cpu.json` (1 CPU core) only proves the script works.
- No CUDA code path has been executed in the authoring environment. First Colab run = first GPU execution; the notebook runs the whole test suite and the smoke test first.
- Even with micro-batch 128, ETTm1 ModernTCN GPU memory is unconfirmed (workspace/gradient buffers on top of ~3.5 GiB).
- ModernTCN Revisited: only the submission version was read; its numeric effect sizes are not to be quoted without re-checking. TCN/DLinear paper bodies not retrieved.
- Scientific scope: one transformer, one test winter; ETTh1 ⊂ ETTm1; 3 seeds; simplified single-stage model.
- Colab-only code path not covered: Drive mounting in the notebook.

## 7. Next actions (in order)

1. Copy this checkpoint into the GitHub repo (see HANDOFF) and open `notebooks/colab_runner.ipynb` on a GPU runtime.
2. Run notebook cells: setup -> `pytest` (expect 102 passed) -> smoke test -> **G4 benchmark**; save `logs/runtime_cuda.json`; update `experiment_matrix.md` Sec. 4 with measured numbers and choose `P4_TIER`. If ETTm1 ModernTCN OOMs at micro-batch 128, lower `micro_batch` in `src/train.py::default_config` (keep it identical for all kernels/seeds) and record it as an amendment.
3. P1 -> inspect validation table -> `freeze_protocol.py` -> P3 -> P4 -> `run_erf.py` -> `aggregate.py` (validation summaries).
4. Failure analysis on validation only; write `reports/failure_hypotheses_frozen.md`, commit; then `run_test_eval.py` (once) -> aggregate -> figures.
5. Write `reports/Discussion.md` from `results/summary_*.csv`, `claim_c1.json`, ERF figures.

Do **not** redo: source/data audit, protocol, model/test implementation (re-open only via a dated amendment).
