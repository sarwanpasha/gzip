"""Shared helpers: config, paths, logging, timing, memory."""
import json
import os
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.json"
    with open(path) as f:
        cfg = json.load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def p(cfg, *parts):
    """Path inside the project working directory (cfg['workdir'], default ROOT)."""
    base = Path(cfg.get("workdir") or ROOT)
    out = base.joinpath(*parts)
    return out


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def log(*a):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *a, flush=True)
    sys.stdout.flush()


def peak_rss_mb():
    """Peak resident set size of this process (and reaped children) in MB."""
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(self_kb, child_kb) / 1024.0


@contextmanager
def timer(store, key):
    t0 = time.perf_counter()
    c0 = time.process_time()
    yield
    store[key + "_wall_s"] = time.perf_counter() - t0
    store[key + "_cpu_s"] = time.process_time() - c0


def write_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def read_json(path):
    with open(path) as f:
        return json.load(f)
