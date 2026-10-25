"""Grid generation, freezing, aggregation, claim rule, figures, notebook validity and a real-data end-to-end pipeline."""
import csv, glob, json, os, sys

import numpy as np
import pandas as pd
import pytest

import aggregate, freeze_protocol, make_figures, run_grid, run_erf, run_test_eval, verify_checkpoints, smoke_test
from src import grid
from src.train import default_config, run_id, canonical_hash
from conftest import ROOT


def fake_frozen():
    return {"frozen": True, "hparams": {grid.hp_key(m, d, H): ({"lr": 1e-4, "dropout": 0.3} if m != "dlinear" else {"lr": 5e-3})
                                       for m in grid.MODELS for d, H in grid.SETTINGS}}


# ----------------------------------------------------------------------------- grids
def test_grid_sizes_match_the_experiment_matrix():
    fz = fake_frozen()
    p1, p3 = grid.p1_configs(), grid.p3_configs(fz)
    assert len(p1) == 32 and len(grid.p1_configs(ettm1_reduced=True)) == 28 and len(p3) == 36
    assert len(grid.p4_configs(fz, "full")) == 48 and len(grid.p4_configs(fz, "h1")) == 24 and len(grid.p4_configs(fz, "minimal")) == 30
    assert len(grid.p6_norevin(fz, "ETTh1", 336)) == 3
    for cfgs in (p1, p3, grid.p4_configs(fz, "full")):
        assert len({run_id(c) for c in cfgs}) == len(cfgs)                       # unique run ids
    assert {c["seed"] for c in p3} == {2021, 2022, 2023} and {c["seed"] for c in p1} == {2021}
    p4 = grid.p4_configs(fz, "full")
    assert {c["large_kernel"] for c in p4} == {3, 13, 31, 83} and all(c["model"] == "modern_tcn" for c in p4)
    minimal = [c for c in grid.p4_configs(fz, "minimal") if c["dataset"] == "ETTm1"]
    assert {(c["pred_len"], c["large_kernel"]) for c in minimal} == {(96, 31), (96, 83)}
    # kernel is the only architectural difference inside one setting: everything else in the config is identical
    ref = [c for c in p4 if c["dataset"] == "ETTh1" and c["pred_len"] == 96 and c["seed"] == 2021]
    keys = [k for k in ref[0] if k not in ("large_kernel",)]
    assert all(all(c[k] == ref[0][k] for k in keys) for c in ref)
    assert all(c["phase"] != "test" for c in p1 + p3 + p4)


def test_grid_uses_frozen_hyperparameters_and_refuses_missing_keys():
    fz = fake_frozen()
    fz["hparams"][grid.hp_key("tcn", "ETTm1", 336)] = {"lr": 3e-4}
    c = [c for c in grid.p3_configs(fz) if c["model"] == "tcn" and c["dataset"] == "ETTm1" and c["pred_len"] == 336][0]
    assert c["lr"] == 3e-4
    with pytest.raises(KeyError):
        grid.p3_configs({"hparams": {}})
    assert len(grid.p3_configs({"hparams": fz["hparams"]}, only_dataset="ETTh1", only_model="dlinear")) == 6


def test_run_grid_dry_run_lists_all_and_skips_done(tmp_path, capsys):
    run_grid.main(["--phase", "p1", "--data_dir", "x", "--out", str(tmp_path), "--dry_run"])
    out = capsys.readouterr().out
    assert "32 configs, 32 to run" in out and out.count("TODO ") == 32


# ----------------------------------------------------------------------------- freeze
def ledger_row(model, ds, H, lr, dropout, val, status="max_epochs", phase="p1"):
    return {"run_id": f"{model}_{ds}_{H}_{lr}_{dropout}", "model": model, "dataset": ds, "pred_len": H, "lr": lr,
            "dropout": dropout, "best_val_mse": val, "status": status, "phase": phase}


def test_freeze_selects_lowest_validation_and_breaks_ties_with_reference():
    rows = [ledger_row("modern_tcn", "ETTh1", 96, 1e-4, 0.3, 0.50), ledger_row("modern_tcn", "ETTh1", 96, 5e-4, 0.1, 0.45),
            ledger_row("modern_tcn", "ETTh1", 96, 5e-4, 0.3, 0.40, status="failed_nonfinite_loss"),
            ledger_row("modern_tcn", "ETTh1", 336, 5e-4, 0.1, 0.60), ledger_row("modern_tcn", "ETTh1", 336, 1e-4, 0.3, 0.60),
            ledger_row("dlinear", "ETTh1", 96, 5e-3, 0.0, 0.7), ledger_row("dlinear", "ETTh1", 96, 1e-3, 0.0, 0.6),
            ledger_row("tcn", "ETTh1", 96, 1e-3, 0.1, 0.1, phase="p3")]
    best = freeze_protocol.select(rows)
    assert best["modern_tcn|ETTh1|96"]["hp"] == {"lr": 5e-4, "dropout": 0.1}          # failed run ignored
    assert best["modern_tcn|ETTh1|336"]["hp"] == {"lr": 1e-4, "dropout": 0.3}         # tie -> reference config
    assert best["dlinear|ETTh1|96"]["hp"] == {"lr": 1e-3}
    assert "tcn|ETTh1|96" not in best                                                  # only phase p1 is eligible


def test_freeze_cli_refuses_incomplete_overwrite_and_is_written_once(tmp_path):
    led = tmp_path / "runs.csv"
    pd.DataFrame([ledger_row("dlinear", "ETTh1", 96, 5e-3, 0.0, 0.7)]).to_csv(led, index=False)
    out = tmp_path / "frozen.json"
    proto = os.path.join(ROOT, "research_notes", "research_protocol.md")
    with pytest.raises(SystemExit):
        freeze_protocol.main(["--ledger", str(led), "--out", str(out), "--protocol", proto])
    freeze_protocol.main(["--ledger", str(led), "--out", str(out), "--protocol", proto, "--allow_partial"])
    d = json.load(open(out))
    assert d["frozen"] is True and d["hparams"] == {"dlinear|ETTh1|96": {"lr": 5e-3}} and len(d["protocol_sha256"]) == 64
    with pytest.raises(SystemExit):
        freeze_protocol.main(["--ledger", str(led), "--out", str(out), "--protocol", proto, "--allow_partial"])


# ----------------------------------------------------------------------------- aggregation + claim
def test_summarize_mean_and_sample_sd_over_seeds():
    df = pd.DataFrame({"dataset": "ETTh1", "pred_len": 96, "model": "modern_tcn", "large_kernel": 51, "revin": True,
                       "seed": [2021, 2022, 2023], "params": 609312, "best_val_mse": [0.40, 0.42, 0.44], "best_val_mae": [0.5, 0.5, 0.5]})
    s = aggregate.summarize(df, ["best_val_mse", "best_val_mae"])
    assert s.best_val_mse_mean[0] == pytest.approx(0.42) and s.best_val_mse_sd[0] == pytest.approx(0.02)   # ddof=1
    assert s.best_val_mse_n[0] == 3 and s.seeds[0] == "2021,2022,2023" and s.params[0] == 609312


def claim_df(d31, d51, k3=0.40, H=(96, 336)):
    rows = []
    for h in H:
        for seed, base in zip((2021, 2022, 2023), (k3, k3 + 0.002, k3 - 0.002)):
            for k, dec in ((3, 0.0), (31, d31), (51, d51)):
                rows.append({"dataset": "ETTh1", "model": "modern_tcn", "revin": True, "pred_len": h, "large_kernel": k,
                             "seed": seed, "test_mse": base - dec})
    return pd.DataFrame(rows)


def test_claim_c1_decision_rule():
    sup = aggregate.claim_c1(claim_df(0.014, 0.013))
    assert sup["C1a_verdict"] == "supported" and sup["passes"] == 4 and sup["C1b"]["H96"]["no_further_gain"]
    assert aggregate.claim_c1(claim_df(0.001, 0.001))["C1a_verdict"] == "not_supported"                 # below 0.005 threshold
    part = claim_df(0.014, 0.003)                                                                       # k=51 fails -> 2 of 4
    assert aggregate.claim_c1(part)["C1a_verdict"] == "partially_supported"
    df = claim_df(0.014, 0.014)
    inc = aggregate.claim_c1(df[~((df.large_kernel == 51) & (df.seed == 2023))])
    assert inc["C1a_verdict"] == "inconclusive_missing_data"
    # sign consistency: a single seed with a negative difference breaks the pass even when the mean is large
    mixed = claim_df(0.014, 0.014)
    mixed.loc[(mixed.large_kernel == 31) & (mixed.seed == 2022) & (mixed.pred_len == 96), "test_mse"] += 0.05
    assert aggregate.claim_c1(mixed)["comparisons"]["H96_k31"]["pass"] is False
    assert aggregate.claim_c1(claim_df(0.014, 0.0, H=(96, 336)))["C1b"]["H96"]["no_further_gain"] is False


# ----------------------------------------------------------------------------- figures
def test_make_figures_from_synthetic_erf_and_summary(tmp_path):
    res = tmp_path / "results"; (res / "erf").mkdir(parents=True)
    lag = np.arange(336)
    for model, k in (("tcn", 0), ("dlinear", 0), ("modern_tcn", 3), ("modern_tcn", 51), ("modern_tcn", 83)):
        for seed in (2021, 2022):
            arrays = {"B": np.exp(-lag / (20 + k)) + 1e-3}
            if model != "dlinear":
                arrays.update({"A_center": np.exp(-lag / (10 + k)) + 1e-4, "A_last": np.exp(-lag / (15 + k)) + 1e-4})
            met = {t: {"r50": 5, "r90": 30 + k, "r95": 60, "support": 100} for t in arrays}
            np.savez(res / "erf" / f"{model}_{k}_{seed}.npz", **arrays, metrics=json.dumps(met),
                     meta=json.dumps({"dataset": "ETTh1", "pred_len": 96, "model": model, "large_kernel": k, "seed": seed, "n_blocks": 1}))
    pd.DataFrame([{"dataset": "ETTh1", "pred_len": 96, "model": "modern_tcn", "large_kernel": k, "best_val_mse_mean": 0.4 - k / 1000,
                   "best_val_mse_sd": 0.002} for k in (3, 51, 83)]).to_csv(res / "summary_val.csv", index=False)
    out = tmp_path / "figs"
    make_figures.main(["--results", str(res), "--out", str(out)])
    names = sorted(os.listdir(out))
    assert names == ["erf_models_ETTh1_H96.png", "kernel_ablation_ETTh1_H96.png"]
    assert all(os.path.getsize(out / n) > 5000 for n in names)


# ----------------------------------------------------------------------------- notebook
def test_notebook_is_valid_and_code_cells_compile():
    nb = json.load(open(os.path.join(ROOT, "notebooks", "colab_runner.ipynb")))
    assert nb["nbformat"] == 4 and len(nb["cells"]) > 10
    src_all = ""
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        lines = []
        for ln in "".join(c["source"]).splitlines():
            lines.append(ln[: len(ln) - len(ln.lstrip())] + "pass" if ln.lstrip().startswith(("!", "%")) else ln)
        code = "\n".join(lines)
        compile(code, "<cell>", "exec")
        src_all += "".join(c["source"])
    assert "TRAIN" in src_all and "CHECKPOINT_URL" in src_all and "pytest" in src_all and "freeze_protocol" in src_all
    for script in ("run_grid", "freeze_protocol", "run_erf", "run_test_eval", "aggregate", "make_figures", "smoke_test",
                   "benchmark_runtime", "verify_checkpoints"):
        assert os.path.exists(os.path.join(ROOT, "scripts", script + ".py")) and script in src_all


# ----------------------------------------------------------------------------- real-data end-to-end
def test_end_to_end_pipeline_on_real_etth1(tmp_path, ett_dir):
    """P1 -> freeze -> P3 -> ERF -> test (guarded, once) -> aggregate -> verify -> figures, DLinear only, 1 epoch."""
    res, frozen = str(tmp_path / "res"), str(tmp_path / "frozen.json")
    proto = os.path.join(ROOT, "research_notes", "research_protocol.md")
    base = ["--data_dir", ett_dir, "--out", res, "--device", "cpu", "--only_dataset", "ETTh1", "--only_model", "dlinear", "--max_epochs", "1"]
    assert run_grid.main(["--phase", "p1"] + base) == 0
    # test evaluation is impossible before freezing
    with pytest.raises(PermissionError):
        run_test_eval.main(["--data_dir", ett_dir, "--out", res, "--device", "cpu", "--frozen", frozen])
    freeze_protocol.main(["--ledger", f"{res}/runs.csv", "--out", frozen, "--protocol", proto, "--allow_partial"])
    assert run_grid.main(["--phase", "p3", "--frozen", frozen] + base) == 0
    run_erf.main(["--out", res, "--data_dir", ett_dir, "--device", "cpu"])
    assert len(glob.glob(f"{res}/erf/*.npz")) == 8      # 4 (P1) + 6 (P3) configs, 2 of them the same run (selected P1 run re-used)
    run_test_eval.main(["--data_dir", ett_dir, "--out", res, "--device", "cpu", "--frozen", frozen])
    tr = pd.read_csv(f"{res}/test_runs.csv")
    assert len(tr) == 8 and tr.run_id.is_unique and np.isfinite(tr.test_mse).all()
    run_test_eval.main(["--data_dir", ett_dir, "--out", res, "--device", "cpu", "--frozen", frozen])   # idempotent: nothing re-evaluated
    assert len(pd.read_csv(f"{res}/test_runs.csv")) == 8
    aggregate.main(["--results", res, "--frozen", frozen])
    vs, ts = pd.read_csv(f"{res}/summary_val.csv"), pd.read_csv(f"{res}/summary_test.csv")
    assert len(vs) == 2 and (vs.best_val_mse_n == 3).all() and (ts.test_mse_n == 3).all()   # seed 2021 comes from the re-used P1 run
    assert (vs.seeds == "2021,2022,2023").all()
    assert (ts.test_mse_mean > 0).all() and np.isfinite(ts.test_mse_sd).all()
    n, bad = verify_checkpoints.verify(res, ett_dir, "cpu")
    assert n == 16 and bad == []                                                   # 8 val + 8 test records reproduced from checkpoints
    make_figures.main(["--results", res, "--out", str(tmp_path / "figs")])


def test_smoke_script_runs_on_real_data_cpu(ett_dir):
    rep = smoke_test.run_smoke(ett_dir, "cpu", "ETTh1", 96, epochs=5, train_limit=192, val_limit=64, batch=32, models=("dlinear", "modern_tcn"))
    assert rep["passed"] and all(v["loss_decreased"] and v["finite"] for v in rep["models"].values())


def test_select_main_counts_reused_p1_run_only_when_it_matches_frozen_hparams(tmp_path):
    rows = pd.DataFrame([
        {"run_id": "a", "model": "dlinear", "dataset": "ETTh1", "pred_len": 96, "lr": 1e-3, "dropout": 0.0, "seed": 2021, "large_kernel": 0, "phase": "p1"},
        {"run_id": "b", "model": "dlinear", "dataset": "ETTh1", "pred_len": 96, "lr": 5e-3, "dropout": 0.0, "seed": 2021, "large_kernel": 0, "phase": "p1"},
        {"run_id": "c", "model": "dlinear", "dataset": "ETTh1", "pred_len": 96, "lr": 1e-3, "dropout": 0.0, "seed": 2022, "large_kernel": 0, "phase": "p3"}])
    fz = tmp_path / "f.json"; json.dump({"frozen": True, "hparams": {"dlinear|ETTh1|96": {"lr": 1e-3}}}, open(fz, "w"))
    assert sorted(aggregate.select_main(rows, str(fz)).run_id) == ["a", "c"]          # 'b' (non-selected P1 config) excluded
    assert sorted(aggregate.select_main(rows, None).run_id) == ["c"]                   # no frozen file -> only phase p3/p4/p6


def test_benchmark_runtime_script_cpu_and_estimate_arithmetic(tmp_path, monkeypatch, ett_dir):
    import benchmark_runtime as br
    monkeypatch.chdir(tmp_path)
    br.main(["--data_dir", ett_dir, "--device", "cpu", "--batch", "4", "--steps", "2", "--warmup", "1", "--settings", "quick"])
    out = json.load(open(tmp_path / "logs" / "runtime_cpu.json"))
    assert set(out["runs"]) == {"ETTh1_H96_modern_tcn", "ETTh1_H96_tcn", "ETTh1_H96_dlinear"}
    assert all(r["sec_per_epoch_est"] > 0 and r["n_train"] == 8209 for r in out["runs"].values())
    # full-grid estimate arithmetic with a stubbed measurement of exactly 10 s/epoch for every run
    monkeypatch.setattr(br, "bench", lambda *a, **k: {"sec_per_epoch_est": 10.0, "peak_mem_mb": None, "n_train": 1, "batch": 1})
    br.main(["--data_dir", ett_dir, "--device", "cpu", "--settings", "all"])
    est = json.load(open(tmp_path / "logs" / "runtime_cpu.json"))["estimate_hours"]
    assert est["P1"] == pytest.approx(32 * 10 * 40 / 3600) and est["P3"] == pytest.approx(36 * 10 * 40 / 3600)
    assert est["P4_ETTm1_full"] == pytest.approx(24 * 10 * 40 / 3600) and est["P4_ETTm1_minimal"] == pytest.approx(6 * 10 * 40 / 3600)
    assert est["P4_ETTh1"] == pytest.approx(24 * 10 * 40 / 3600)
