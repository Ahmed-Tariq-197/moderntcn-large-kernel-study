"""Run (or dry-run) an experiment phase. Finished runs are skipped; a failed run never stops the grid.
Example: python scripts/run_grid.py --phase p1 --data_dir /content/ETT-small --device cuda"""
import argparse, json, os, sys, traceback
import _bootstrap  # noqa
from src.grid import p1_configs, p3_configs, p4_configs, p6_norevin
from src.train import train_run, run_dir_of, run_id


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["p1", "p3", "p4", "p6"])
    ap.add_argument("--tier", default="full", help="p4 tier: h1 | minimal | full")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frozen", default="configs/frozen_v1.json")
    ap.add_argument("--only_dataset", default=None)
    ap.add_argument("--only_model", default=None)
    ap.add_argument("--p6_dataset", default="ETTh1")
    ap.add_argument("--p6_H", type=int, default=336)
    ap.add_argument("--ettm1_reduced", action="store_true")
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)
    common = {"data_dir": a.data_dir, "out_dir": a.out, "device": a.device}
    if a.max_epochs is not None:
        common["max_epochs"] = a.max_epochs
    if a.phase == "p1":
        cfgs = p1_configs(ettm1_reduced=a.ettm1_reduced, **common)
    else:
        frozen = json.load(open(a.frozen))
        if not frozen.get("hparams"):
            raise SystemExit("frozen config has no hparams")
        if a.phase == "p3":
            cfgs = p3_configs(frozen, a.only_dataset, a.only_model, **common)
        elif a.phase == "p4":
            cfgs = p4_configs(frozen, a.tier, a.only_dataset, a.only_model, **common)
        else:
            cfgs = p6_norevin(frozen, a.p6_dataset, a.p6_H, **common)
    if a.only_dataset:
        cfgs = [c for c in cfgs if c["dataset"] == a.only_dataset]
    if a.only_model:
        cfgs = [c for c in cfgs if c["model"] == a.only_model]
    todo = [c for c in cfgs if not os.path.exists(os.path.join(run_dir_of(c), "DONE"))]
    print(f"{a.phase}: {len(cfgs)} configs, {len(todo)} to run")
    if a.dry_run:
        for c in cfgs:
            print(("TODO " if c in todo else "DONE ") + run_id(c))
        return 0
    failed = 0
    for i, c in enumerate(todo, 1):
        print(f"--- [{i}/{len(todo)}] {run_id(c)}")
        try:
            r = train_run(c)
            if str(r.get("status", "")).startswith("failed"):
                failed += 1
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f"finished; failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
