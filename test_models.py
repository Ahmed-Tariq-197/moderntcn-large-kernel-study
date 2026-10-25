"""Gates G1 (parameter counts) and G2 (shapes, grouping, causality, invariances, forward/backward)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import param_count_analytic as pa
from src.models import ModernTCN, TCN, DLinear, build_model, count_params
from src.models.modern_tcn import ModernTCNBlock
from src.models.revin import RevIN
from src.train import default_config

KERNELS = (3, 13, 31, 51, 83)
CONFIGS = [("ETTh1", 96, 1, 1), ("ETTh1", 336, 1, 1), ("ETTm1", 96, 3, 8), ("ETTm1", 336, 3, 8)]


# ---------------------------------------------------------------- G1: parameter counts
@pytest.mark.parametrize("ds,H,K,r", CONFIGS)
def test_modern_tcn_param_counts_equal_analytic_and_spread_within_10pct(ds, H, K, r):
    totals = {}
    for k in KERNELS:
        m = ModernTCN(pred_len=H, n_blocks=K, ffn_ratio=r, large_kernel=k)
        totals[k] = count_params(m)
        assert totals[k] == pa.modern_tcn(k, K, r, H)["total"], (ds, H, k)
    spread = max(totals.values()) / min(totals.values()) - 1
    assert spread <= 0.10, spread


@pytest.mark.parametrize("H", [96, 336])
def test_baseline_param_counts_equal_analytic(H):
    assert count_params(TCN(pred_len=H)) == pa.tcn(H)["total"]
    assert count_params(DLinear(pred_len=H)) == pa.dlinear(H)["total"]


@pytest.mark.parametrize("ds,H,K,r", CONFIGS)
def test_only_the_large_kernel_weights_change_across_kernels(ds, H, K, r):
    ref = {n: tuple(p.shape) for n, p in ModernTCN(pred_len=H, n_blocks=K, ffn_ratio=r, large_kernel=51).named_parameters()}
    for k in (3, 13, 31, 83):
        cur = {n: tuple(p.shape) for n, p in ModernTCN(pred_len=H, n_blocks=K, ffn_ratio=r, large_kernel=k).named_parameters()}
        assert cur.keys() == ref.keys()
        diff = {n for n in ref if ref[n] != cur[n]}
        # large-kernel conv weights always differ in shape; the small branch only changes at k=3 (rule small=min(5,k))
        allowed = {n for n in ref if n.endswith("dw_large.weight")} | ({n for n in ref if n.endswith("dw_small.weight")} if k == 3 else set())
        assert diff == allowed, (k, diff - allowed)


# ---------------------------------------------------------------- G2: shapes, forward/backward
@pytest.mark.parametrize("ds,H,K,r", CONFIGS)
@pytest.mark.parametrize("k", [3, 83])
def test_modern_tcn_output_shape_and_features(ds, H, K, r, k):
    m = ModernTCN(pred_len=H, n_blocks=K, ffn_ratio=r, large_kernel=k).eval()
    x = torch.randn(2, 336, 7)
    assert m(x).shape == (2, H, 7)
    assert m.forward_features(x).shape == (2, 7, 64, 84)
    assert torch.isfinite(m(x)).all()


def test_patch_arithmetic_and_last_value_padding():
    m = ModernTCN(pred_len=96).eval()
    cap = {}
    m.stem.register_forward_hook(lambda mod, i, o: cap.update(inp=i[0], out=o))
    x = torch.randn(3, 336, 7)
    m(x)
    assert cap["inp"].shape == (3 * 7, 1, 340) and cap["out"].shape == (3 * 7, 64, 84)
    xi = cap["inp"].reshape(3, 7, 340)
    assert torch.equal(xi[:, :, 336:], xi[:, :, 335:336].expand(-1, -1, 4))        # replicated last value


@pytest.mark.parametrize("name", ["modern", "tcn", "dlinear"])
def test_forward_backward_finite_and_all_params_get_gradients(name):
    torch.manual_seed(0)
    m = {"modern": lambda: ModernTCN(pred_len=96, n_blocks=2, ffn_ratio=2, large_kernel=13),
         "tcn": lambda: TCN(pred_len=96), "dlinear": lambda: DLinear(pred_len=96)}[name]().train()
    x, y = torch.randn(4, 336, 7), torch.randn(4, 96, 7)
    loss = ((m(x) - y) ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert sum(float(p.grad.abs().sum()) for p in m.parameters()) > 0


@pytest.mark.parametrize("name", ["modern", "tcn", "dlinear"])
def test_eval_output_independent_of_batch_composition(name):
    torch.manual_seed(1)
    m = {"modern": lambda: ModernTCN(pred_len=96, large_kernel=13), "tcn": lambda: TCN(pred_len=96),
         "dlinear": lambda: DLinear(pred_len=96)}[name]().eval()
    x = torch.randn(6, 336, 7)
    with torch.no_grad():
        full = m(x)
        single = torch.cat([m(x[i:i + 1]) for i in range(6)])
    assert torch.allclose(full, single, atol=1e-5)


def test_build_model_dispatch_and_default_configs():
    for model in ("modern_tcn", "tcn", "dlinear"):
        for ds in ("ETTh1", "ETTm1"):
            cfg = default_config(ds, 96, model)
            m = build_model(cfg)
            assert m(torch.randn(1, 336, 7)).shape == (1, 96, 7)
    c = default_config("ETTm1", 336, "modern_tcn")
    assert (c["n_blocks"], c["ffn_ratio"], c["head_dropout"]) == (3, 8, 0.1)
    c = default_config("ETTh1", 336, "modern_tcn")
    assert (c["n_blocks"], c["ffn_ratio"], c["head_dropout"]) == (1, 1, 0.0)
    with pytest.raises(ValueError):
        build_model({"model": "nope", "pred_len": 96})


# ---------------------------------------------------------------- grouping / channel-order (risks R3.1-R3.3)
def make_block(M=3, D=4, N=10, k=5, r=2):
    torch.manual_seed(0)
    b = ModernTCNBlock(M, D, k, min(5, k), r, dropout=0.0).eval()
    for bn in (b.bn_large, b.bn_small, b.norm):                   # non-trivial BN statistics
        bn.running_mean.normal_(); bn.running_var.uniform_(0.5, 2); bn.weight.data.uniform_(0.5, 1.5); bn.bias.data.normal_()
    return b


def test_depthwise_convs_mix_time_only():
    M, D, N = 3, 4, 10
    b = make_block(M, D, N)
    x = torch.randn(1, M * D, N)
    for conv, bn in ((b.dw_large, b.bn_large), (b.dw_small, b.bn_small)):
        base = bn(conv(x))
        for c in range(M * D):
            x2 = x.clone(); x2[0, c] += 1.0
            ch = (bn(conv(x2)) - base).abs().sum(dim=(0, 2)) > 0
            assert ch.nonzero().flatten().tolist() == [c], (c, ch)


def test_convffn1_mixes_features_within_a_variable_only():
    M, D, N = 3, 4, 10
    b = make_block(M, D, N)
    dff = b.ffn1_a.out_channels // M
    f = lambda h: b.ffn1_b(F.gelu(b.ffn1_a(h)))
    x = torch.randn(1, M * D, N)
    base = f(x)
    for m in range(M):
        for d in range(D):
            x2 = x.clone(); x2[0, m * D + d] += 1.0
            changed = ((f(x2) - base).abs().sum(dim=(0, 2)) > 0).nonzero().flatten().tolist()
            assert changed == list(range(m * D, (m + 1) * D)), (m, d, changed)       # exactly variable m's D channels


def test_convffn2_mixes_variables_at_the_same_feature_index_only():
    M, D, N = 3, 4, 10
    b = make_block(M, D, N)
    f = lambda h: b.ffn2_b(F.gelu(b.ffn2_a(h)))
    x = torch.randn(1, D * M, N)                                   # d-major layout: channel = d*M + m
    base = f(x)
    for d in range(D):
        for m in range(M):
            x2 = x.clone(); x2[0, d * M + m] += 1.0
            changed = ((f(x2) - base).abs().sum(dim=(0, 2)) > 0).nonzero().flatten().tolist()
            assert changed == list(range(d * M, (d + 1) * M)), (d, m, changed)       # all variables of feature d


def test_block_permutations_connect_the_two_ffns_correctly():
    M, D, N = 3, 4, 10
    b = make_block(M, D, N)
    cap = {}
    b.ffn1_b.register_forward_hook(lambda mod, i, o: cap.update(ffn1_out=o))
    b.ffn2_a.register_forward_hook(lambda mod, i, o: cap.update(ffn2_in=i[0]))
    b.ffn2_b.register_forward_hook(lambda mod, i, o: cap.update(ffn2_out=o))
    x = torch.randn(2, M * D, N)
    y = b(x)
    for m in range(M):
        for d in range(D):
            assert torch.equal(cap["ffn2_in"][:, d * M + m], cap["ffn1_out"][:, m * D + d])      # m-major -> d-major
            assert torch.allclose(y[:, m * D + d] - x[:, m * D + d], cap["ffn2_out"][:, d * M + m], atol=1e-6)   # back + residual


def test_variables_are_processed_independently_before_the_cross_variable_ffn():
    """Stem + DW + BN + FFN1 never mix variables; only ConvFFN2 may. Disable ConvFFN2 and check independence."""
    torch.manual_seed(0)
    m = ModernTCN(seq_len=64, pred_len=8, n_vars=3, d_model=4, n_blocks=1, ffn_ratio=1, large_kernel=5, dropout=0.0, revin=False).eval()
    for p in m.blocks[0].ffn2_b.parameters():
        torch.nn.init.zeros_(p)                                    # ConvFFN2 contributes nothing -> pure residual path
    for p in m.blocks[0].ffn2_a.parameters():
        torch.nn.init.zeros_(p)
    x = torch.randn(1, 64, 3)
    f = m.forward_features(x)
    x2 = x.clone(); x2[:, :, 1] += 1.0                             # perturb variable 1 only
    f2 = m.forward_features(x2)
    diff = (f2 - f).abs().sum(dim=(0, 2, 3))
    assert diff[0] == 0 and diff[2] == 0 and diff[1] > 0
    m2 = ModernTCN(seq_len=64, pred_len=8, n_vars=3, d_model=4, n_blocks=1, ffn_ratio=1, large_kernel=5, dropout=0.0, revin=False).eval()
    d2 = (m2.forward_features(x2) - m2.forward_features(x)).abs().sum(dim=(0, 2, 3))
    assert d2[0] > 0 and d2[2] > 0                                  # with ConvFFN2 active, variables do interact


# ---------------------------------------------------------------- RevIN
def test_revin_roundtrip_and_stats():
    r = RevIN(7)
    x = torch.randn(5, 336, 7) * 3 + 2
    xn, st = r.normalize(x)
    assert torch.allclose(xn.mean(1), torch.zeros(5, 7), atol=1e-5) and torch.allclose(xn.std(1, unbiased=False), torch.ones(5, 7), atol=1e-3)
    assert torch.allclose(r.denormalize(xn, st), x, atol=1e-4)
    xg = x.clone().requires_grad_(True)
    _, (mu, sd) = r.normalize(xg)
    assert mu.requires_grad is False and sd.requires_grad is False   # statistics are detached


@pytest.mark.parametrize("name", ["modern", "tcn"])
def test_revin_models_are_shift_equivariant(name):
    torch.manual_seed(2)
    m = (ModernTCN(pred_len=96, large_kernel=13) if name == "modern" else TCN(pred_len=96)).eval()
    x = torch.randn(3, 336, 7)
    with torch.no_grad():
        assert torch.allclose(m(x + 5.0), m(x) + 5.0, atol=2e-4)


# ---------------------------------------------------------------- TCN
def test_tcn_is_causal_and_has_the_documented_receptive_field():
    assert TCN().receptive_field == 509
    torch.manual_seed(0)
    m = TCN(seq_len=600, pred_len=4, revin=False).eval()
    x = torch.randn(1, 600, 7)
    f0 = m.forward_features(x)
    t = 300
    x2 = x.clone(); x2[:, t, :] += 1.0
    f1 = m.forward_features(x2)
    assert torch.equal(f0[:, :, :t], f1[:, :, :t])                  # strictly causal
    # output at the last step: reaches lag 508 but not lag 509
    def reach(lag):                                                  # change |output| when input at lag `lag` is perturbed
        x3 = x.clone(); x3[:, 599 - lag, :] += 1.0
        with torch.no_grad():
            return (m(x3) - m(x)).abs().max().item()
    assert reach(509) == 0.0                                         # outside the receptive field: exactly unchanged
    assert min(reach(l) for l in (508, 507, 505, 500, 0)) > 0.0      # inside: every probed lag has influence


# ---------------------------------------------------------------- DLinear
def test_dlinear_decomposition_matches_numpy():
    m = DLinear(seq_len=60, pred_len=5, moving_avg=25)
    x = torch.randn(2, 60, 7)
    seas, trend = m.decompose(x)
    assert torch.allclose(seas + trend, x, atol=1e-6)
    xn = x[0].numpy()
    padded = np.concatenate([np.repeat(xn[:1], 12, 0), xn, np.repeat(xn[-1:], 12, 0)])
    manual = np.stack([padded[i:i + 25].mean(0) for i in range(60)])
    np.testing.assert_allclose(trend[0].numpy(), manual, atol=1e-5)
    c = torch.full((1, 60, 7), 3.0)
    assert torch.allclose(m.decompose(c)[1], c, atol=1e-6)            # constant series: trend == series


def test_dlinear_forward_matches_manual_linear_maps():
    m = DLinear(seq_len=48, pred_len=6)
    x = torch.randn(3, 48, 7)
    s, t = m.decompose(x)
    exp = (s.permute(0, 2, 1) @ m.linear_seasonal.weight.T + m.linear_seasonal.bias
           + t.permute(0, 2, 1) @ m.linear_trend.weight.T + m.linear_trend.bias).permute(0, 2, 1)
    assert torch.allclose(m(x), exp, atol=1e-5)
