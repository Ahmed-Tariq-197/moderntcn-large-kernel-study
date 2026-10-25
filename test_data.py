"""Gate G0: dataset loading, chronological borders, normalisation, window construction, leakage."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data import (borders, expected_window_counts, make_splits, read_ett, Scaler, FEATURES, WindowDataset)
from src.evaluate import evaluate_predictor, predict_last_value, predict_train_mean
from conftest import synthetic_array

L = 336


def test_borders():
    assert borders("ETTh1") == (8640, 11520, 14400)
    assert borders("ETTm1") == (34560, 46080, 57600)


@pytest.mark.parametrize("ds", ["ETTh1", "ETTm1"])
@pytest.mark.parametrize("H", [96, 336])
def test_window_counts_match_audit(ds, H, audit):
    exp = expected_window_counts(ds, L, H)
    a = audit[ds]["windows"][str(H)]
    assert exp == {"train": a["train"], "val": a["val"], "test": a["test"]}


@pytest.mark.parametrize("ds", ["ETTh1", "ETTm1"])
@pytest.mark.parametrize("H", [96, 336])
def test_real_data_counts_and_columns(ds, H, ett_dir):
    s = make_splits(ett_dir, ds, L, H)
    exp = expected_window_counts(ds, L, H)
    assert (len(s.train), len(s.val), len(s.test)) == (exp["train"], exp["val"], exp["test"])
    assert list(pd.read_csv(f"{ett_dir}/{ds}.csv").columns[1:]) == FEATURES


def test_real_boundary_timestamps(ett_dir, audit):
    df = pd.read_csv(f"{ett_dir}/ETTh1.csv")
    tr, va, te = borders("ETTh1")
    got = audit["ETTh1"]["ts_at_borders"]
    assert df.date[tr] == got["val_first_target"] and df.date[va] == got["test_first_target"]
    assert df.date[te - 1] == got["test_last"]


def test_scaler_fit_on_train_rows_only(ett_dir):
    raw = read_ett(f"{ett_dir}/ETTh1.csv")
    s = make_splits(ett_dir, "ETTh1", L, 96)
    tr = borders("ETTh1")[0]
    np.testing.assert_allclose(s.scaler.mean, raw[:tr].mean(0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(s.scaler.std, raw[:tr].std(0), rtol=0, atol=1e-12)
    # train rows are exactly standardised
    z = s.z[:tr].double().numpy()
    assert np.abs(z.mean(0)).max() < 1e-5 and np.abs(z.std(0) - 1).max() < 1e-5


def test_scaler_and_train_windows_independent_of_val_test_and_tail():
    raw = synthetic_array("ETTh1")
    tr, va, te = borders("ETTh1")
    base = make_splits("", "ETTh1", L, 96, array=raw)
    mod = raw.copy()
    mod[tr:] += 1000.0                                    # destroy everything after the train rows
    alt = make_splits("", "ETTh1", L, 96, array=mod)
    np.testing.assert_array_equal(base.scaler.mean, alt.scaler.mean)
    np.testing.assert_array_equal(base.scaler.std, alt.scaler.std)
    xb, yb = base.train.get(torch.arange(len(base.train)))
    xa, ya = alt.train.get(torch.arange(len(alt.train)))
    assert torch.equal(xb, xa) and torch.equal(yb, ya)
    tail = raw.copy()
    tail[te:] *= -50.0                                    # rows beyond test_end must never matter
    t2 = make_splits("", "ETTh1", L, 96, array=tail)
    assert torch.equal(base.z, t2.z)


@pytest.mark.parametrize("H", [96, 336])
def test_no_target_leakage_between_splits(H):
    raw = synthetic_array("ETTh1")
    tr, va, te = borders("ETTh1")
    s = make_splits("", "ETTh1", L, H, array=raw)
    for ds, lo, hi in ((s.train, 0, tr), (s.val, tr, va), (s.test, va, te)):
        a0, _ = ds.target_rows(0)
        _, b1 = ds.target_rows(len(ds) - 1)
        assert a0 >= lo and b1 == hi, (ds.name, a0, b1)           # targets lie exactly inside the own split
    # inputs of train stay inside train; val/test inputs reach back at most L rows
    assert s.train.input_rows(0)[0] == 0 and s.train.target_rows(len(s.train) - 1)[1] == tr
    assert s.val.input_rows(0)[0] == tr - L and s.test.input_rows(0)[0] == va - L
    # first val target is row tr, first test target row va
    assert s.val.target_rows(0)[0] == tr and s.test.target_rows(0)[0] == va
    # train targets never overlap val/test targets
    assert s.train.target_rows(len(s.train) - 1)[1] <= s.val.target_rows(0)[0]
    assert s.val.target_rows(len(s.val) - 1)[1] <= s.test.target_rows(0)[0]


def test_window_content_matches_raw_rows():
    raw = synthetic_array("ETTh1")
    s = make_splits("", "ETTh1", L, 96, array=raw)
    z = s.scaler.transform(raw)
    for ds, off in ((s.val, 0), (s.test, 0)):
        for i in (0, 1, len(ds) // 2, len(ds) - 1):
            x, y = ds.get(torch.tensor([i]))
            a, b = ds.input_rows(i)
            c, d = ds.target_rows(i)
            np.testing.assert_allclose(x[0].numpy(), z[a:b], atol=1e-6)
            np.testing.assert_allclose(y[0].numpy(), z[c:d], atol=1e-6)
            assert b == c


def test_batches_cover_every_window_once_and_are_batch_size_invariant():
    s = make_splits("", "ETTh1", L, 96, array=synthetic_array("ETTh1"))
    def collect(bs):
        return torch.cat([x for x, _ in s.test.batches(bs, shuffle=False, drop_last=False)])
    a, b = collect(7), collect(1000)
    assert a.shape[0] == len(s.test) == b.shape[0] and torch.equal(a, b)


def test_shuffle_is_seeded_and_a_permutation():
    s = make_splits("", "ETTh1", L, 96, array=synthetic_array("ETTh1"), train_limit=300)
    f = lambda seed: torch.cat([y[:, 0, 0] for _, y in s.train.batches(64, shuffle=True, seed=seed)])
    assert torch.equal(f(1), f(1)) and not torch.equal(f(1), f(2))
    assert sorted(f(1).tolist()) == sorted(torch.cat([y[:, 0, 0] for _, y in s.train.batches(64)]).tolist())


def test_metrics_match_numpy_and_are_batch_invariant():
    s = make_splits("", "ETTh1", L, 96, array=synthetic_array("ETTh1"))
    ref = lambda x: x[:, -1:, :].expand(-1, 96, -1)
    r1 = evaluate_predictor(predict_last_value(96), s.val, batch_size=13, detail=True)
    r2 = evaluate_predictor(predict_last_value(96), s.val, batch_size=4096, detail=True)
    assert r1["mse"] == pytest.approx(r2["mse"], rel=1e-12) and r1["mae"] == pytest.approx(r2["mae"], rel=1e-12)
    X, Y = s.val.get(torch.arange(len(s.val)))
    d = (ref(X) - Y).double()
    assert r1["mse"] == pytest.approx(float((d ** 2).mean()), rel=1e-9)
    assert r1["mae"] == pytest.approx(float(d.abs().mean()), rel=1e-9)
    assert np.mean(r1["mse_per_variable"]) == pytest.approx(r1["mse"], rel=1e-9)
    assert np.mean(r1["mse_per_step"]) == pytest.approx(r1["mse"], rel=1e-9)
    z = evaluate_predictor(predict_train_mean(96), s.val)
    assert z["mse"] == pytest.approx(float((Y.double() ** 2).mean()), rel=1e-9)


def test_nonfinite_prediction_is_flagged():
    s = make_splits("", "ETTh1", L, 96, array=synthetic_array("ETTh1"), val_limit=20)
    r = evaluate_predictor(lambda x: torch.full((x.shape[0], 96, 7), float("nan")), s.val)
    assert r["finite"] is False


def test_limits_and_synthetic_guard():
    s = make_splits("", "ETTh1", L, 96, array=synthetic_array("ETTh1"), train_limit=50, val_limit=10, test_limit=5)
    assert (len(s.train), len(s.val), len(s.test)) == (50, 10, 5)
    with pytest.raises(ValueError):
        make_splits("", "ETTh1", L, 96, array=np.zeros((100, 7)))
