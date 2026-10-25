from .dlinear import DLinear
from .modern_tcn import ModernTCN
from .tcn import TCN


def build_model(cfg: dict):
    """Build a model from a run-config dict (see src/train.py::default_config)."""
    m = cfg["model"]
    L, H = cfg.get("seq_len", 336), cfg["pred_len"]
    if m == "modern_tcn":
        return ModernTCN(seq_len=L, pred_len=H, d_model=cfg.get("d_model", 64), n_blocks=cfg["n_blocks"],
                         ffn_ratio=cfg["ffn_ratio"], large_kernel=cfg["large_kernel"],
                         small_kernel=cfg.get("small_kernel"), dropout=cfg["dropout"],
                         head_dropout=cfg.get("head_dropout", 0.0), revin=cfg.get("revin", True))
    if m == "tcn":
        return TCN(seq_len=L, pred_len=H, channels=cfg.get("tcn_channels", 64), levels=cfg.get("tcn_levels", 7),
                   kernel_size=cfg.get("tcn_kernel", 3), dropout=cfg["dropout"], revin=cfg.get("revin", True))
    if m == "dlinear":
        return DLinear(seq_len=L, pred_len=H)
    raise ValueError(f"unknown model {m!r}")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
