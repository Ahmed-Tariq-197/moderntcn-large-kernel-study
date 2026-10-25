"""Gate G6: ERF measurement correctness (probe estimator vs exact Jacobian, orientation, support, determinism)."""
import numpy as np
import pytest
import torch

from src.data import make_splits
from src.models import ModernTCN, TCN, DLinear
from src.receptive_field import (select_samples, erf_target_a, erf_target_b, exact_forecast_erf, erf_metrics, measure_erf)
from conftest import synthetic_array


def tiny_modern(L=32, H=4, M=3, k=5, blocks=1, seed=0):
    torch.manual_seed(seed)
    return ModernTCN(seq_len=L, pred_len=H, n_vars=M, d_model=4, n_blocks=blocks, ffn_ratio=1, large_kernel=k,
                     dropout=0.0).eval()


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_probe_estimator_matches_exact_jacobian_dlinear():
    torch.manual_seed(0)
    m = DLinear(seq_len=48, pred_len=8).eval()
    x = torch.randn(4, 48, 7)
    est = erf_target_b(m, x, probes=1500, seed=1)
    ex = exact_forecast_erf(m, x)
    assert rel_err(est, ex) < 0.05


def test_probe_estimator_matches_exact_jacobian_modern_tcn():
    m = tiny_modern()
    x = torch.randn(3, 32, 3)
    assert rel_err(erf_target_b(m, x, probes=1500, seed=2), exact_forecast_erf(m, x)) < 0.06


def test_probe_estimator_matches_exact_jacobian_tcn():
    torch.manual_seed(0)
    m = TCN(seq_len=32, pred_len=4, n_vars=3, channels=8, levels=3, kernel_size=3, dropout=0.0).eval()
    x = torch.randn(3, 32, 3)
    assert rel_err(erf_target_b(m, x, probes=1500, seed=3), exact_forecast_erf(m, x)) < 0.06


def test_probe_estimator_converges_with_more_probes():
    m = tiny_modern()
    x = torch.randn(3, 32, 3)
    ex = exact_forecast_erf(m, x)
    e16 = np.mean([rel_err(erf_target_b(m, x, probes=16, seed=s), ex) for s in range(6)])
    e256 = np.mean([rel_err(erf_target_b(m, x, probes=256, seed=s), ex) for s in range(6)])
    assert e256 < e16 and e256 < 0.12


def test_lag_orientation_lag_zero_is_the_newest_step():
    m = DLinear(seq_len=48, pred_len=4).eval()
    with torch.no_grad():
        m.linear_seasonal.weight.zero_(); m.linear_trend.weight.zero_()
        m.linear_seasonal.weight[:, -1] = 1.0                      # forecast reads only the newest seasonal value
    x = torch.randn(2, 48, 7)
    b = exact_forecast_erf(m, x)
    assert int(np.argmax(b)) == 0 and b[0] > 0
    assert np.all(b[13:] == 0)                                    # newest value only touches the last 12 steps via the trend
    est = erf_target_b(m, x, probes=64)
    assert int(np.argmax(est)) == 0 and np.all(est[13:] == 0)


def test_target_a_is_zero_outside_the_architectural_support_modern_tcn():
    L, S, P, k = 64, 4, 8, 3
    m = tiny_modern(L=L, k=k, blocks=1, M=3)
    x = torch.randn(4, L, 3)
    r = erf_target_a(m, x, "center")
    pos, prof = r["position"], r["profile"]
    assert pos == (L // S) // 2
    lo, hi = S * (pos - (k // 2)), S * (pos + (k // 2)) + P - 1      # tokens pos +/- k//2, patches of P raw steps
    time_prof = prof[::-1]                                          # back to time order
    assert np.all(time_prof[:lo] == 0) and np.all(time_prof[hi + 1:] == 0)
    assert time_prof[lo:hi + 1].sum() > 0
    # two blocks double the reach
    m2 = tiny_modern(L=L, k=k, blocks=2, M=3)
    t2 = erf_target_a(m2, x, "center")["profile"][::-1]
    assert np.all(t2[:S * (pos - 2 * (k // 2))] == 0) and (t2 != 0).sum() >= (time_prof != 0).sum()


def test_target_a_tcn_is_strictly_causal():
    torch.manual_seed(0)
    L = 64
    m = TCN(seq_len=L, pred_len=4, n_vars=3, channels=8, levels=3, kernel_size=3, dropout=0.0).eval()
    x = torch.randn(4, L, 3)
    r = erf_target_a(m, x, "center")
    t = r["position"]
    prof = r["profile"][::-1]
    assert t == L // 2 and np.all(prof[t + 1:] == 0) and prof[:t + 1].sum() > 0
    last = erf_target_a(m, x, "last")
    assert last["position"] == L - 1 and last["profile"].sum() > 0


def test_batched_equals_per_sample_and_restores_mode():
    m = tiny_modern()
    m.train()
    x = torch.randn(5, 32, 3)
    a1 = erf_target_a(m, x, "last", chunk=1)["profile"]
    a5 = erf_target_a(m, x, "last", chunk=5)["profile"]
    np.testing.assert_allclose(a1, a5, rtol=1e-5, atol=1e-8)
    assert m.training is True                                      # mode restored
    b1, b2 = erf_target_b(m, x, 8, seed=4, chunk=1), erf_target_b(m, x, 8, seed=4, chunk=5)
    assert b1.shape == b2.shape == (32,) and m.training is True
    np.testing.assert_allclose(erf_target_a(m, x, "center")["profile"], erf_target_a(m, x, "center")["profile"])   # deterministic


def test_profiles_are_nonnegative_and_finite():
    m = tiny_modern()
    x = torch.randn(4, 32, 3)
    r = measure_erf(m, x, probes=8)
    for k in ("B", "A_center", "A_last"):
        assert np.isfinite(r[k]).all() and (r[k] >= 0).all()
    assert set(r["metrics"]) == {"B", "A_center", "A_last"}


def test_dlinear_has_no_backbone_profile():
    r = measure_erf(DLinear(seq_len=48, pred_len=4).eval(), torch.randn(2, 48, 7), probes=4, has_backbone=False)
    assert set(r["metrics"]) == {"B"}


def test_erf_metrics_widths():
    m = erf_metrics(np.ones(1000))
    assert m["r50"] == 499 and abs(m["r90"] - 899) <= 1 and abs(m["r95"] - 949) <= 1 and m["support"] == 1000
    spike = np.zeros(100); spike[0] = 1.0
    ms = erf_metrics(spike)
    assert ms["r50"] == ms["r90"] == ms["r95"] == 0 and ms["support"] == 1 and ms["mass_last24"] == 1.0
    assert erf_metrics(np.zeros(10))["r90"] is None
    assert erf_metrics(np.ones(10), dt_hours=0.25)["r90_hours"] == pytest.approx(erf_metrics(np.ones(10))["r90"] * 0.25)


def test_select_samples_is_deterministic_evenly_spaced_and_validation_only():
    raw = synthetic_array("ETTh1")
    s = make_splits("", "ETTh1", 336, 96, array=raw)
    x1, x2 = select_samples(s.val, 50), select_samples(s.val, 50)
    assert x1.shape == (50, 336, 7) and torch.equal(x1, x2)
    n = len(s.val)
    idx = [int(np.floor(j * (n - 1) / 49)) for j in range(50)]
    ref, _ = s.val.get(torch.tensor(idx))
    assert torch.equal(x1, ref) and idx[0] == 0 and idx[-1] == n - 1
    # every sampled input lies inside the validation context range [tr-L, va): no test rows are ever used
    from src.data import borders
    tr, va, _ = borders("ETTh1")
    for i in idx:
        a, b = s.val.input_rows(i)
        c, d = s.val.target_rows(i)
        assert a >= tr - 336 and d <= va
