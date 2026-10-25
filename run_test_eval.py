"""P8: evaluate every finished run once on the test split (requires configs/frozen_v1.json)."""
import argparse, glob, json, os, sys
import _bootstrap  # noqa
from src.train import evaluate_test


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frozen", default="configs/frozen_v1.json")
    a = ap.parse_args(argv)
    n = skipped = 0
    for rd in sorted(glob.glob(os.path.join(a.out, "runs", "*"))):
        if not os.path.exists(os.path.join(rd, "DONE")):
            continue
        if os.path.exists(os.path.join(rd, "test_metrics.json")):
            skipped += 1
            continue
        cfg = json.load(open(os.path.join(rd, "config.json")))
        cfg.update({"data_dir": a.data_dir, "device": a.device, "out_dir": a.out})
        r = evaluate_test(cfg, frozen_path=a.frozen)
        print(os.path.basename(rd), f"test mse {r['mse']:.4f} mae {r['mae']:.4f}")
        n += 1
    print(f"evaluated {n}, already had test records: {skipped}")


if __name__ == "__main__":
    main()
