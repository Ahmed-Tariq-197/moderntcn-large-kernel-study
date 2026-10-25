"""Generates notebooks/colab_runner.ipynb (plain nbformat-4 JSON, no extra dependency)."""
import json, os

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}

cells = [
 md("""# ModernTCN-style large-kernel study - Colab runner
Runs gates G0-G6, measures runtime (G4), and executes the experiment phases with resumable runs.
**Set `TRAIN`**: `True` trains (checkpoints and ledgers go to `RESULTS_DIR`, which should live on Google Drive so a disconnect never loses runs);
`False` downloads published checkpoints + ledgers and only re-verifies/aggregates/plots (no training, test split is not re-evaluated into the ledger).
Runtime: select a **GPU** runtime (Runtime > Change runtime type)."""),
 code("""REPO_URL = "<repository-url>"   # set to the HTTPS clone URL of this repository
TRAIN = True                      # False -> load checkpoints from CHECKPOINT_URL
CHECKPOINT_URL = ""               # zip of results/ (runs, runs.csv, test_runs.csv, erf/) used when TRAIN=False
USE_DRIVE = True                  # keep results on Drive so interrupted sessions resume
ETTM1_REDUCED_P1 = False          # set True if the measured ETTm1 cost is too high (protocol: 2-config grid)
P4_TIER = "full"                  # 'h1' | 'minimal' | 'full' (decide after the G4 runtime measurement)"""),
 code("""import os, subprocess, sys, torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "Select a GPU runtime"
assert "<repository-url>" not in REPO_URL, "set REPO_URL to the repository clone URL"
if not os.path.exists("moderntcn-large-kernel-study"):
    subprocess.run(["git", "clone", REPO_URL], check=True)
os.chdir("moderntcn-large-kernel-study")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest", "matplotlib", "pandas"], check=True)
if not os.path.exists("/content/ETT-small"):
    subprocess.run("git clone --depth 1 https://github.com/zhouhaoyi/ETDataset.git /content/ETDataset && cp -r /content/ETDataset/ETT-small /content/ETT-small", shell=True, check=True)
ETT = "/content/ETT-small"
RESULTS = "results"
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS = "/content/drive/MyDrive/moderntcn_results"
    os.makedirs(RESULTS, exist_ok=True)
print("data:", ETT, "| results:", RESULTS)"""),
 md("## Gates G0-G6 (unit tests) and G3 smoke test"),
 code("""!ETT_DIR=/content/ETT-small python -m pytest -q tests"""),
 code("""!python scripts/smoke_test.py --data_dir /content/ETT-small --device cuda"""),
 md("## G4 - runtime and GPU memory (replace the analytic estimates in `research_notes/experiment_matrix.md` with these)"),
 code("""!python scripts/benchmark_runtime.py --data_dir /content/ETT-small --device cuda --batch 512"""),
 md("## TRAIN = True: phases P1 -> freeze -> P3 -> P4 -> (P6) -> ERF -> test (once) -> aggregate -> figures"),
 code("""if TRAIN:
    flag = "--ettm1_reduced" if ETTM1_REDUCED_P1 else ""
    !python scripts/run_grid.py --phase p1 --data_dir $ETT --out $RESULTS --device cuda $flag"""),
 code("""if TRAIN:
    # Inspect validation results of P1 first. Freezing is irreversible (the script refuses to overwrite).
    !python scripts/freeze_protocol.py --ledger $RESULTS/runs.csv --out configs/frozen_v1.json
    !cat configs/frozen_v1.json"""),
 code("""if TRAIN:
    !python scripts/run_grid.py --phase p3 --data_dir $ETT --out $RESULTS --device cuda
    !python scripts/run_grid.py --phase p4 --tier $P4_TIER --data_dir $ETT --out $RESULTS --device cuda"""),
 code("""if TRAIN:
    !python scripts/run_erf.py --data_dir $ETT --out $RESULTS --device cuda
    !python scripts/aggregate.py --results $RESULTS   # validation summaries only (no test file yet)"""),
 md("""### Failure analysis (validation only) - then freeze interpretations
Write `reports/failure_hypotheses_frozen.md` and commit it BEFORE running the next cell. Optional P6 diagnostic:
`!python scripts/run_grid.py --phase p6 --p6_dataset ETTh1 --p6_H 336 --data_dir $ETT --out $RESULTS --device cuda`"""),
 code("""if TRAIN:
    # P8: test split, once per run. Refuses to run unless configs/frozen_v1.json exists.
    !python scripts/run_test_eval.py --data_dir $ETT --out $RESULTS --device cuda
    !python scripts/aggregate.py --results $RESULTS
    !python scripts/make_figures.py --results $RESULTS --out reports/figures"""),
 md("## TRAIN = False: reproduce from saved checkpoints (read-only verification)"),
 code("""if not TRAIN:
    assert CHECKPOINT_URL, "set CHECKPOINT_URL"
    !wget -q "$CHECKPOINT_URL" -O /content/ckpt.zip && unzip -q -o /content/ckpt.zip -d $RESULTS
    !python scripts/verify_checkpoints.py --results $RESULTS --data_dir $ETT --device cuda
    !python scripts/aggregate.py --results $RESULTS
    !python scripts/make_figures.py --results $RESULTS --out reports/figures"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "colab_runner.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
