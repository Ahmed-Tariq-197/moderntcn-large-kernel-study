"""Measures bytes saved for backward per training sample (autograd saved-tensor hooks, deduplicated by storage) and
scales linearly to a batch size. This is a device-independent LOWER bound for training memory: gradients of
intermediates, cuDNN workspaces and allocator fragmentation come on top. Writes logs/activation_memory.json."""
import argparse, json, os
import torch
import _bootstrap  # noqa
from src.models import build_model, count_params
from src.train import default_config
from src.grid import SETTINGS


def saved_bytes_per_sample(model, B: int = 4, L: int = 336, M: int = 7) -> float:
    seen, tot = set(), [0]
    def pack(t):
        st = t.untyped_storage()
        key = (st.data_ptr(), st.nbytes())
        if key not in seen:
            seen.add(key); tot[0] += st.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model.train()
        loss = model(torch.randn(B, L, M)).pow(2).mean()
    loss.backward()
    return tot[0] / B


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args(argv)
    out = {}
    for ds, H in SETTINGS:
        for m in ("modern_tcn", "tcn", "dlinear"):
            for k in ((3, 83) if m == "modern_tcn" else (0,)):
                cfg = default_config(ds, H, m, k or 51)
                model = build_model(cfg)
                per = saved_bytes_per_sample(model)
                mb = cfg.get("micro_batch") or a.batch
                out[f"{ds}_H{H}_{m}" + (f"_k{k}" if k else "")] = {
                    "mib_per_sample": per / 2**20, "gib_at_batch": per * a.batch / 2**30,
                    "gib_at_micro_batch": per * min(mb, a.batch) / 2**30, "micro_batch": cfg.get("micro_batch"),
                    "params": count_params(model), "param_grad_adam_gib": count_params(model) * 16 / 2**30}
    os.makedirs("logs", exist_ok=True)
    json.dump({"batch": a.batch, "note": "saved-for-backward bytes only (lower bound)", "runs": out}, open("logs/activation_memory.json", "w"), indent=2)
    for k, v in out.items():
        print(f"{k:32s} {v['mib_per_sample']:7.1f} MiB/sample  batch{a.batch}: {v['gib_at_batch']:6.2f} GiB  micro-batch: {v['gib_at_micro_batch']:6.2f} GiB")


if __name__ == "__main__":
    main()
