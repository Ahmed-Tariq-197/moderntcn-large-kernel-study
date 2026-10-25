# Changelog

All notable changes to this project. Dates are UTC. Only verified changes are listed.

## [Unreleased]

## 2026-10-20 - Pre-training code complete and locally verified (checkpoint 2)

### Added
- `src/`: `data.py` (split/scaler/windows), `models/{modern_tcn,tcn,dlinear,revin}.py`, `train.py` (resumable training, ledger, guarded one-shot test evaluation, micro-batching), `evaluate.py`, `receptive_field.py` (ERF Targets A/B, width metrics), `grid.py`.
- `scripts/`: `run_grid`, `freeze_protocol`, `run_test_eval`, `run_erf`, `aggregate` (mean/sd ddof=1, claim C1 rule), `make_figures`, `smoke_test`, `benchmark_runtime`, `measure_activation_memory`, `verify_checkpoints`, `mutation_check`, `make_notebook`.
- `tests/` (102 tests) and `notebooks/colab_runner.ipynb`.
- `logs/`: `activation_memory.json`, `smoke_cpu_*.json`, `runtime_cpu.json`, `mutation_report.json`.
- Protocol amendment A1; experiment matrix v0.2 with gate statuses; risk-register status block; README rewritten for the verified state.

### Verified
- Real parameter counts == analytic table (20 ModernTCN cells, TCN, DLinear); max kernel-ablation spread 6.26 %.
- 102 tests pass; 20/20 injected bugs detected; real-data CPU smoke tests pass for ETTh1 H=336 and ETTm1 H=336; real-data end-to-end pipeline (DLinear, ETTh1) passes.

### Fixed during verification
- `aggregate` sd column mislabelled; re-used P1 run dropped from P3 aggregation; CUDA-RNG-state resume crash (load `last.pt` on CPU); extra epoch after crash at stopping point; TF32 defaults disabled; several weak/defective tests corrected.

### Decided
- ETTm1 ModernTCN trained with 4x128 micro-batches (effective batch 512) because ~13.9 GiB of activations at batch 512 do not fit a 15 GB GPU (amendment A1.3).

### Known issues / open
- G4 (GPU runtime/memory) not measured; no CUDA path executed yet; ETTm1 ModernTCN memory with micro-batch unconfirmed.

## 2026-10-20 - Research foundation (v0.1, protocol frozen, no models yet)

### Added
- `research_notes/sources.md`: source log with retrieval status, verified facts, limitations, and the list of authors' files read.
- `research_notes/research_protocol.md` (v0.1): research question, variables, dataset protocol, architecture + tensor shapes, baselines, explicit ERF method (Targets A/B),
  pre-registered claim C1 and decision rule, seed policy, failure-analysis procedure, deviations D1-D10.
- `research_notes/experiment_matrix.md`: settings, kernel candidates {3,13,31,51,83}, analytic parameter counts, phase grid, tiers, analytic compute, gates G0-G6, run-ID/resume/ledger design.
- `research_notes/risks.md`: risk register (leakage, benchmark consistency, fidelity, parameter counts, ERF, compute, interpretation, provenance).
- `scripts/audit_data.py` (pandas/numpy only) and its output `logs/data_audit.json`.
- `scripts/param_count_analytic.py` and its output `logs/param_counts_analytic.json` (analytic; not yet checked against real modules).
- `PROJECT_STATE.md`, `HANDOFF.md`, `CHANGELOG.md`, `requirements.txt`, `.gitignore`, rewritten `README.md`.

### Decided
- No Lab 4 materials are available: TCN baseline is an independent Bai-et-al. implementation; ERF uses an explicit adapted autograd protocol; compatibility with Lab 4 is **not** claimed.
- Claim C1 = ModernTCN Appendix F.2.1 / Table 16 (kernel size), with a pre-registered paired-by-seed decision rule.
- Evaluate all test windows (`drop_last=False`); ETTh1/ETTm1 not treated as independent datasets.

### Known issues
- torch is not importable in the authoring sandbox (CUDA runtime libraries partly installed; needs `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`); no neural code has been executed anywhere yet.
- Camera-ready ModernTCN-Revisited PDF, DLinear/TCN paper bodies not retrieved (see `sources.md`).
- The GitHub repository previously contained condensed copies of the two scripts and a README claiming model files that did not exist; both are superseded by this checkpoint.
