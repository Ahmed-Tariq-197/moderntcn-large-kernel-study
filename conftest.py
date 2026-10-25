import json, os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

CANDIDATES = [os.environ.get("ETT_DIR", ""), os.path.join(ROOT, "data", "ETT-small"),
              os.path.join(ROOT, "..", "ETDataset", "ETT-small"), "/content/ETT-small"]


def find_ett_dir():
    for c in CANDIDATES:
        if c and os.path.exists(os.path.join(c, "ETTh1.csv")) and os.path.exists(os.path.join(c, "ETTm1.csv")):
            return c
    return None


@pytest.fixture(scope="session")
def ett_dir():
    d = find_ett_dir()
    if d is None:
        pytest.skip("real ETT CSVs not found (set ETT_DIR)")
    return d


@pytest.fixture(scope="session")
def audit():
    return json.load(open(os.path.join(ROOT, "logs", "data_audit.json")))


def synthetic_array(dataset, seed=0):
    """Smooth multivariate series with trend + daily cycle + noise, long enough for the benchmark borders."""
    from src.data import borders
    n = borders(dataset)[2] + 500
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None]
    period = 24 if dataset == "ETTh1" else 96
    base = np.sin(2 * np.pi * t / period + rng.uniform(0, 6, 7)) * rng.uniform(0.5, 2, 7)
    walk = np.cumsum(rng.normal(0, 0.02, (n, 7)), axis=0)
    return (base + walk + rng.normal(0, 0.1, (n, 7)) + rng.uniform(-3, 3, 7)).astype(np.float64)


@pytest.fixture(scope="session")
def synth_h1():
    return synthetic_array("ETTh1")
