"""Mutation check of the test suite: inject one realistic bug at a time into a COPY of the repo and require that the
relevant tests fail. A surviving mutant means a test gap. Writes logs/mutation_report.json."""
import json, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (name, file, old, new, test files that must fail)
MUTANTS = [
 ("ffn2 grouped by variable instead of feature", "src/models/modern_tcn.py", "self.ffn2_a = nn.Conv1d(C, M * dff, 1, groups=D)", "self.ffn2_a = nn.Conv1d(C, M * dff, 1, groups=M)", ["tests/test_models.py"]),
 ("missing permutation before ConvFFN2", "src/models/modern_tcn.py", "h = h.reshape(B, M, D, N).permute(0, 2, 1, 3).reshape(B, D * M, N)", "h = h.reshape(B, M, D, N).reshape(B, D * M, N)", ["tests/test_models.py"]),
 ("depthwise conv not depthwise", "src/models/modern_tcn.py", "self.dw_large = nn.Conv1d(C, C, k, padding=k // 2, groups=C, bias=False)", "self.dw_large = nn.Conv1d(C, C, k, padding=k // 2, groups=1, bias=False)", ["tests/test_models.py"]),
 ("zero padding instead of last-value padding", "src/models/modern_tcn.py", 'mode="replicate")', 'mode="constant")', ["tests/test_models.py"]),
 ("TCN not causal", "src/models/tcn.py", "return self.conv(F.pad(x, (self.pad, 0)))", "return self.conv(F.pad(x, (self.pad // 2, self.pad - self.pad // 2)))", ["tests/test_models.py", "tests/test_erf.py"]),
 ("RevIN statistics not detached", "src/models/revin.py", "mean = x.mean(dim=1, keepdim=True).detach()", "mean = x.mean(dim=1, keepdim=True)", ["tests/test_models.py"]),
 ("scaler fitted on train+val rows", "src/data.py", "scaler = Scaler.fit(raw[:tr])", "scaler = Scaler.fit(raw[:va])", ["tests/test_data.py", "tests/test_train.py"]),
 ("validation windows without context rows", "src/data.py", "val=WindowDataset(z, tr - L, va, L, H, \"val\", val_limit)", "val=WindowDataset(z, tr, va, L, H, \"val\", val_limit)", ["tests/test_data.py"]),
 ("evaluation drops the last partial batch", "src/evaluate.py", "ds.batches(batch_size, shuffle=False, drop_last=False)", "ds.batches(batch_size, shuffle=False, drop_last=True)", ["tests/test_data.py"]),
 ("ERF lag axis not flipped", "src/receptive_field.py", "return torch.flip(per_time, dims=[-1])", "return per_time", ["tests/test_erf.py"]),
 ("ERF probe estimate without sqrt", "src/receptive_field.py", "return _to_lag(torch.sqrt(energy / count)).cpu().numpy()", "return _to_lag(energy / count).cpu().numpy()", ["tests/test_erf.py"]),
 ("ERF gradient not ReLU-ed (Target A)", "src/receptive_field.py", "g = torch.relu(g).sum(dim=2)", "g = g.sum(dim=2)", ["tests/test_erf.py"]),
 ("test guard removed", "src/train.py", 'if not os.path.exists(frozen_path) or not json.load(open(frozen_path)).get("frozen", False):', "if False:", ["tests/test_train.py"]),
 ("test overwrite allowed", "src/train.py", "if os.path.exists(out_path):\n        raise FileExistsError", "if False:\n        raise FileExistsError", ["tests/test_train.py"]),
 ("RNG state not restored on resume", "src/train.py", "        _set_rng_state(ck[\"rng\"])\n", "", ["tests/test_train.py"]),
 ("early stopping on non-strict improvement", "src/train.py", "improved = val[\"mse\"] < best_val", "improved = val[\"mse\"] <= best_val", ["tests/test_train.py"]),
 ("sd with ddof=0", "scripts/aggregate.py", "g[c].std(ddof=1)", "g[c].std(ddof=0)", ["tests/test_scripts.py"]),
 ("ablation grid includes k=51 twice", "src/grid.py", "ABLATION_EXTRA = (3, 13, 31, 83)", "ABLATION_EXTRA = (3, 13, 31, 51, 83)", ["tests/test_scripts.py"]),
 ("lr schedule type3 off by one", "src/train.py", "return base if epoch < 3 else base * (0.9 ** (epoch - 3))", "return base if epoch < 3 else base * (0.9 ** (epoch - 2))", ["tests/test_train.py"]),
 ("micro-batch loss not rescaled", "src/train.py", "loss = torch.mean((model(xs) - ys) ** 2) * (len(xs) / B)", "loss = torch.mean((model(xs) - ys) ** 2)", ["tests/test_train.py"]),
]


def main():
    only = sys.argv[1:] 
    report = []
    for i, (name, f, old, new, tests) in enumerate(MUTANTS):
        if only and not any(o in name for o in only):
            continue
        tmp = tempfile.mkdtemp(prefix="mut_")
        dst = os.path.join(tmp, "repo")
        shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "results", "deliverables", ".git"))
        p = os.path.join(dst, f)
        src = open(p).read()
        if old not in src:
            report.append({"mutant": name, "status": "PATTERN_NOT_FOUND"}); shutil.rmtree(tmp); continue
        open(p, "w").write(src.replace(old, new, 1))
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests], cwd=dst,
                           capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        killed = r.returncode != 0
        first = next((l for l in r.stdout.splitlines() if l.startswith("FAILED")), "")
        report.append({"mutant": name, "file": f, "status": "KILLED" if killed else "SURVIVED", "first_failure": first[:140], "sec": round(time.time() - t0, 1)})
        print(f"[{'KILLED  ' if killed else 'SURVIVED'}] {name}  <- {first[:90]}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    if not only:
        json.dump(report, open(os.path.join(ROOT, "logs", "mutation_report.json"), "w"), indent=2)
    surv = [r for r in report if r["status"] != "KILLED"]
    print(f"{len(report) - len(surv)}/{len(report)} mutants killed; not killed: {[r['mutant'] for r in surv]}")
    return 1 if surv else 0


if __name__ == "__main__":
    sys.exit(main())
