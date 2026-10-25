# HANDOFF (read with PROJECT_STATE.md)

**State:** all pre-training code written and locally verified (102 tests pass; mutation check 20/20; real-data smoke + end-to-end pipeline pass). **No experiment has been run; no result exists.**

**To resume**
1. Unzip `deliverables/latest_checkpoint.zip` into the repository root, commit and push.
2. Colab (GPU runtime): open `notebooks/colab_runner.ipynb`, set `REPO_URL`, run setup, pytest, smoke, **G4 benchmark** (the only gate not yet done).
3. Follow `PROJECT_STATE.md` Sec. 7. Never run `run_test_eval.py` before `configs/frozen_v1.json` exists (the code refuses).

**Watch for on the first GPU run:** OOM for ETTm1 ModernTCN (measured 13.9 GiB saved activations at batch 512; micro-batch 128 is configured); any CUDA-only exception (none of the CUDA paths has run yet); runtime far above the analytic estimate -> choose `P4_TIER` / `--ettm1_reduced` after G4.

**Do not change after P1 is frozen:** hyperparameters, seeds, kernel set, `micro_batch`, checkpoint-selection rule, evaluation code.
