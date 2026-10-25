"""Gates G3 (smoke), G5 (resume), failure handling, ledger, one-shot test evaluation and test-split isolation."""
import csv, json, os, shutil

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import borders
from src.train import (default_config, canonical_hash, run_id, lr_at, train_run, evaluate_test, run_dir_of, LEDGER_FIELDS)
from conftest import synthetic_array

TINY = dict(d_model=8, tcn_channels=8, tcn_levels=3, train_limit=96, val_limit=48, batch_size=32,
            eval_batch_size=64, max_epochs=4, patience=10)


def cfg_for(tmp_path, model="modern_tcn", **over):
    c = default_config("ETTh1", 96, model, 11, 7, out_dir=str(tmp_path), lr=1e-3, lr_schedule="constant", **TINY)
    c.update(over)
    return c


def test_run_id_and_hash_properties():
    a = default_config("ETTh1", 96, "modern_tcn", 51, 2021)
    assert run_id(a) == run_id(dict(a)) and "k51" in run_id(a) and "s2021" in run_id(a)
    b = dict(a, device="cuda", data_dir="/x", out_dir="/y", phase="p9")
    assert canonical_hash(a) == canonical_hash(b)                     # paths/device are not part of the hash
    assert canonical_hash(a) != canonical_hash(dict(a, lr=2e-4)) and canonical_hash(a) != canonical_hash(dict(a, seed=2022))
    assert run_id(default_config("ETTh1", 96, "modern_tcn", 51, 2021, revin=False)).endswith(
        run_id(default_config("ETTh1", 96, "modern_tcn", 51, 2021, revin=False))[-9:]) and "_norevin" in run_id(
        default_config("ETTh1", 96, "modern_tcn", 51, 2021, revin=False))


def test_lr_schedules_follow_the_reference_recipes():
    assert [lr_at(e, 1.0, "type3") for e in (1, 2, 3, 4, 5)] == pytest.approx([1, 1, 1, 0.9, 0.81])
    assert [lr_at(e, 1.0, "type1") for e in (1, 2, 3)] == pytest.approx([1, 0.5, 0.25])
    assert lr_at(7, 0.1, "constant") == 0.1
    with pytest.raises(ValueError):
        lr_at(1, 1.0, "bad")


@pytest.mark.parametrize("model", ["modern_tcn", "tcn", "dlinear"])
def test_smoke_loss_decreases_and_everything_is_finite(model, tmp_path, synth_h1):
    lr = {"modern_tcn": 2e-3, "tcn": 3e-3, "dlinear": 1e-2}[model]
    c = cfg_for(tmp_path, model, lr=lr, max_epochs=6, train_limit=256, val_limit=64)
    r = train_run(c, array=synth_h1, verbose=False)
    log = pd.read_csv(os.path.join(run_dir_of(c), "train_log.csv"))
    assert r["status"] in ("max_epochs", "early_stopped") and len(log) == r["epochs_run"]
    assert log[["train_mse", "val_mse", "val_mae"]].notna().all().all() and np.isfinite(log.val_mse).all()
    assert log.train_mse.iloc[-1] < log.train_mse.iloc[0]
    for f in ("config.json", "last.pt", "best.pt", "val_metrics.json", "DONE"):
        assert os.path.exists(os.path.join(run_dir_of(c), f)), f
    assert not os.path.exists(os.path.join(run_dir_of(c), "test_metrics.json"))   # training never writes test results


def test_ledger_row_has_all_fields_and_matches_val_metrics(tmp_path, synth_h1):
    c = cfg_for(tmp_path, "dlinear", lr=5e-3, batch_size=32)
    r = train_run(c, array=synth_h1, verbose=False)
    rows = list(csv.DictReader(open(tmp_path / "runs.csv")))
    assert len(rows) == 1 and list(rows[0].keys()) == LEDGER_FIELDS
    assert rows[0]["run_id"] == run_id(c) and float(rows[0]["best_val_mse"]) == pytest.approx(r["best_val_mse"])
    assert int(rows[0]["params"]) == r["params"] == 2 * (336 * 96 + 96)
    # best epoch recorded in the log is the epoch with the lowest val mse
    log = pd.read_csv(os.path.join(run_dir_of(c), "train_log.csv"))
    assert int(log.val_mse.idxmin()) + 1 == r["best_epoch"]
    assert r["best_val_mse"] == pytest.approx(float(log.val_mse.min()), rel=1e-6)


def test_resume_after_interruption_reproduces_the_uninterrupted_run(tmp_path, synth_h1):
    """Gate G5: kill after epoch 2, restart, identical continuation (model, log, best checkpoint)."""
    A, B = tmp_path / "A", tmp_path / "B"
    cA = cfg_for(A, "modern_tcn", max_epochs=5, dropout=0.3)
    cB = cfg_for(B, "modern_tcn", max_epochs=5, dropout=0.3)
    ra = train_run(cA, array=synth_h1, verbose=False)
    part = train_run(dict(cB, stop_after_epoch=2), array=synth_h1, verbose=False)
    assert part["status"] == "interrupted" and not os.path.exists(os.path.join(run_dir_of(cB), "DONE"))
    rb = train_run(cB, array=synth_h1, verbose=False)                     # resume
    la = pd.read_csv(os.path.join(run_dir_of(cA), "train_log.csv"))
    lb = pd.read_csv(os.path.join(run_dir_of(cB), "train_log.csv"))
    np.testing.assert_allclose(la[["train_mse", "val_mse", "val_mae"]].values, lb[["train_mse", "val_mse", "val_mae"]].values, rtol=0, atol=1e-9)
    assert ra["best_epoch"] == rb["best_epoch"] and ra["epochs_run"] == rb["epochs_run"]
    ca = torch.load(os.path.join(run_dir_of(cA), "best.pt"), weights_only=False)["model"]
    cb = torch.load(os.path.join(run_dir_of(cB), "best.pt"), weights_only=False)["model"]
    for k in ca:
        assert torch.equal(ca[k], cb[k]), k


def test_completed_run_is_skipped_not_retrained(tmp_path, synth_h1):
    c = cfg_for(tmp_path, "dlinear")
    r1 = train_run(c, array=synth_h1, verbose=False)
    m1 = os.path.getmtime(os.path.join(run_dir_of(c), "last.pt"))
    r2 = train_run(c, array=synth_h1, verbose=False)
    assert os.path.getmtime(os.path.join(run_dir_of(c), "last.pt")) == m1 and r2["best_val_mse"] == r1["best_val_mse"]
    assert len(list(csv.DictReader(open(tmp_path / "runs.csv")))) == 1       # no duplicate ledger row


def test_tampered_config_is_refused(tmp_path, synth_h1):
    c = cfg_for(tmp_path, "dlinear")
    train_run(dict(c, stop_after_epoch=1), array=synth_h1, verbose=False)
    p = os.path.join(run_dir_of(c), "config.json")
    d = json.load(open(p)); d["lr"] = 0.5; json.dump(d, open(p, "w"))
    with pytest.raises(RuntimeError):
        train_run(c, array=synth_h1, verbose=False)


def test_early_stopping_uses_strict_improvement_and_patience(tmp_path, synth_h1):
    c = cfg_for(tmp_path, "dlinear", lr=0.0, max_epochs=20, patience=2)      # lr=0 -> validation never improves after epoch 1
    r = train_run(c, array=synth_h1, verbose=False)
    assert r["status"] == "early_stopped" and r["best_epoch"] == 1 and r["epochs_run"] == 3


def test_nonfinite_training_fails_cleanly_without_ledger_or_done(tmp_path, synth_h1):
    bad = cfg_for(tmp_path, "dlinear", lr=1e30, batch_size=32)
    good = cfg_for(tmp_path, "dlinear", lr=5e-3, seed=8)
    r = train_run(bad, array=synth_h1, verbose=False)
    assert r["status"].startswith("failed") and os.path.exists(os.path.join(run_dir_of(bad), "FAILED"))
    assert not os.path.exists(os.path.join(run_dir_of(bad), "DONE"))
    assert not (tmp_path / "runs.csv").exists()
    train_run(good, array=synth_h1, verbose=False)                              # other runs unaffected
    assert [r["run_id"] for r in csv.DictReader(open(tmp_path / "runs.csv"))] == [run_id(good)]


def test_training_and_validation_never_read_the_test_rows(tmp_path):
    """Canary: poison the test rows and the unused tail with NaN. Training/validation must be unaffected."""
    raw = synthetic_array("ETTh1")
    tr, va, te = borders("ETTh1")
    clean = raw.copy()
    poisoned = raw.copy(); poisoned[va:] = np.nan
    c1, c2 = cfg_for(tmp_path / "clean", "modern_tcn"), cfg_for(tmp_path / "poison", "modern_tcn")
    r1 = train_run(c1, array=clean, verbose=False)
    r2 = train_run(c2, array=poisoned, verbose=False)
    assert np.isfinite(r2["best_val_mse"]) and r1["best_val_mse"] == pytest.approx(r2["best_val_mse"], abs=1e-12)


def test_test_evaluation_is_guarded_one_shot_and_uses_the_best_checkpoint(tmp_path, synth_h1):
    c = cfg_for(tmp_path, "dlinear", lr=5e-3)
    train_run(c, array=synth_h1, verbose=False)
    frozen = str(tmp_path / "frozen.json")
    with pytest.raises(PermissionError):
        evaluate_test(c, frozen_path=frozen, array=synth_h1)                       # not frozen yet
    json.dump({"frozen": False}, open(frozen, "w"))
    with pytest.raises(PermissionError):
        evaluate_test(c, frozen_path=frozen, array=synth_h1)
    json.dump({"frozen": True}, open(frozen, "w"))
    res = evaluate_test(c, frozen_path=frozen, array=synth_h1)
    assert np.isfinite(res["mse"]) and res["n_windows"] > 0
    # independent recomputation with the saved best checkpoint
    from src.data import make_splits
    from src.models import build_model
    from src.evaluate import evaluate_model
    s = make_splits("", "ETTh1", 336, 96, array=synth_h1, train_limit=c["train_limit"], val_limit=c["val_limit"])
    m = build_model(c); m.load_state_dict(torch.load(os.path.join(run_dir_of(c), "best.pt"), weights_only=False)["model"])
    assert evaluate_model(m, s.test, 77)["mse"] == pytest.approx(res["mse"], rel=1e-9)
    with pytest.raises(FileExistsError):
        evaluate_test(c, frozen_path=frozen, array=synth_h1)                       # once per run
    rows = list(csv.DictReader(open(tmp_path / "test_runs.csv")))
    assert len(rows) == 1 and float(rows[0]["test_mse"]) == pytest.approx(res["mse"])


def test_test_evaluation_requires_a_finished_run(tmp_path, synth_h1):
    c = cfg_for(tmp_path, "dlinear")
    train_run(dict(c, stop_after_epoch=1), array=synth_h1, verbose=False)
    frozen = str(tmp_path / "f.json"); json.dump({"frozen": True}, open(frozen, "w"))
    with pytest.raises(FileNotFoundError):
        evaluate_test(c, frozen_path=frozen, array=synth_h1)


@pytest.mark.parametrize("model", ["dlinear", "tcn"])
def test_micro_batch_accumulation_equals_full_batch_for_models_without_batch_statistics(model):
    from src.train import train_step
    from src.models import build_model
    torch.manual_seed(0)
    c = default_config("ETTh1", 96, model, tcn_channels=8, tcn_levels=3, dropout=0.0)
    x, y = torch.randn(24, 336, 7), torch.randn(24, 96, 7)
    outs = []
    for mb in (None, 8, 5):
        torch.manual_seed(1)
        m = build_model(c); opt = torch.optim.SGD(m.parameters(), lr=0.1)
        loss = train_step(m, opt, x, y, mb)
        outs.append((loss, torch.cat([p.detach().flatten() for p in m.parameters()])))
    for loss, w in outs[1:]:
        assert loss == pytest.approx(outs[0][0], rel=1e-5)
        assert torch.allclose(w, outs[0][1], atol=1e-6)


def test_train_step_skips_the_update_on_nonfinite_loss():
    from src.train import train_step
    from src.models import DLinear
    m = DLinear(); opt = torch.optim.SGD(m.parameters(), lr=0.1)
    before = [p.detach().clone() for p in m.parameters()]
    assert train_step(m, opt, torch.full((4, 336, 7), float("inf")), torch.zeros(4, 96, 7), 2) is None
    assert all(torch.equal(a, p) for a, p in zip(before, m.parameters()))


def test_ettm1_modern_tcn_uses_micro_batches_for_every_kernel_and_trains(tmp_path):
    for k in (3, 51, 83):
        c = default_config("ETTm1", 336, "modern_tcn", k)
        assert c["micro_batch"] == 128 and c["batch_size"] == 512
    assert default_config("ETTh1", 336, "modern_tcn")["micro_batch"] is None and default_config("ETTm1", 96, "tcn")["micro_batch"] is None


def test_resume_after_crash_at_the_stopping_point_does_not_train_further(tmp_path, synth_h1):
    c = cfg_for(tmp_path, "dlinear", lr=0.0, max_epochs=20, patience=2)
    train_run(dict(c, stop_after_epoch=2), array=synth_h1, verbose=False)             # epoch 1 best, epoch 2 -> bad=1
    p = os.path.join(run_dir_of(c), "last.pt")
    ck = torch.load(p, weights_only=False); ck["bad"] = 2; torch.save(ck, p)           # simulate: stop condition reached, then crash
    r = train_run(c, array=synth_h1, verbose=False)
    assert r["status"] == "early_stopped" and r["epochs_run"] == 2
    assert len(pd.read_csv(os.path.join(run_dir_of(c), "train_log.csv"))) == 2


def test_checkpoint_rng_state_is_loaded_on_cpu_and_strict_fp32(tmp_path, synth_h1):
    c = cfg_for(tmp_path, "dlinear")
    train_run(dict(c, stop_after_epoch=1), array=synth_h1, verbose=False)
    ck = torch.load(os.path.join(run_dir_of(c), "last.pt"), map_location="cpu", weights_only=False)
    assert ck["rng"]["torch"].device.type == "cpu" and ck["rng"]["torch"].dtype == torch.uint8
    torch.set_rng_state(ck["rng"]["torch"])                                            # the call that fails on a CUDA tensor
    assert torch.backends.cudnn.allow_tf32 is False and torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.deterministic is True
    r = train_run(c, array=synth_h1, verbose=False)
    assert json.load(open(os.path.join(run_dir_of(c), "val_metrics.json")))["precision"] == "fp32 (tf32 off)"
