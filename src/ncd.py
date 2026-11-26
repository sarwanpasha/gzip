"""Normalized Compression Distance (NCD) over all ordered pairs.

NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y)), C = compressed length in bytes of the
UTF-8 encoding. The matrix is symmetrized as D' = (NCD + NCD^T) / 2.

NCD(x, y) depends only on the two sequences and has no fitted parameters, so computing it for all
pairs once does not use any label or split information. Split-dependent steps (kernel bandwidth,
kernel centering, kernel PCA) are fitted on training rows/columns only (see features.py).

Subcommands
  cx     compressed length of every single sequence
  rows   C(s_i + s_j) for a block of rows i (--chunk selects the block)
  merge  assemble blocks -> D.npy (symmetrized NCD), timing.json
"""
import argparse
import bz2
import gzip
import json
import lzma
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import ensure_dir, load_config, log, p, peak_rss_mb, write_json

COMPRESSORS = {
    "gzip": lambda b: len(gzip.compress(b, compresslevel=9, mtime=0)),
    "bz2": lambda b: len(bz2.compress(b, compresslevel=9)),
    "lzma": lambda b: len(lzma.compress(b)),  # xz container, preset 6 (8 MiB dictionary)
}
COMPRESSOR_SETTINGS = {
    "gzip": "Python gzip.compress, DEFLATE level 9, 32 KiB window",
    "bz2": "Python bz2.compress, level 9 (900 kB blocks)",
    "lzma": "Python lzma.compress, FORMAT_XZ preset 6 (8 MiB dictionary)",
}

_SEQS = None
_FN = None


def load_sequences(cfg, ds, variant="original"):
    df = pd.read_csv(p(cfg, "data", "processed", ds, "sequences.csv"))
    seqs = df["sequence"].astype(str).tolist()
    if variant == "shuffled":
        # composition-preserving control: residues shuffled within each sequence
        out = []
        for i, s in enumerate(seqs):
            rng = np.random.default_rng(7919 + i)
            a = np.array(list(s))
            rng.shuffle(a)
            out.append("".join(a))
        seqs = out
    return seqs


def out_dir(cfg, ds, comp, variant):
    name = comp if variant == "original" else f"{comp}_{variant}"
    return ensure_dir(p(cfg, "data", "ncd", ds, name))


def _init(seqs, comp):
    global _SEQS, _FN
    _SEQS, _FN = seqs, COMPRESSORS[comp]


def _row(i):
    t = time.process_time()
    si = _SEQS[i]
    row = np.fromiter((_FN((si + sj).encode("utf-8")) for sj in _SEQS), dtype=np.int32, count=len(_SEQS))
    return i, row, time.process_time() - t


def _single(i):
    return _FN(_SEQS[i].encode("utf-8"))


def n_chunks(cfg, ds):
    n = len(pd.read_csv(p(cfg, "data", "processed", ds, "sequences.csv"), usecols=["idx"]))
    return int(np.ceil(n / cfg["ncd_row_chunk"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cx", "rows", "merge", "nchunks"])
    ap.add_argument("--config")
    ap.add_argument("--ds", required=True)
    ap.add_argument("--comp", choices=list(COMPRESSORS))
    ap.add_argument("--variant", default="original", choices=["original", "shuffled"])
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--procs", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)

    if a.cmd == "nchunks":
        print(n_chunks(cfg, a.ds))
        return

    seqs = load_sequences(cfg, a.ds, a.variant)
    n = len(seqs)
    od = out_dir(cfg, a.ds, a.comp, a.variant)
    procs = a.procs or cfg["ncd_processes"]

    if a.cmd == "cx":
        t0, c0 = time.perf_counter(), time.process_time()
        _init(seqs, a.comp)
        cx = np.array([_single(i) for i in range(n)], dtype=np.int32)
        np.save(od / "Cx.npy", cx)
        write_json({"wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0, "n_calls": n},
                   od / "timing_cx.json")
        log(a.ds, a.comp, a.variant, "Cx done")

    elif a.cmd == "rows":
        cs = cfg["ncd_row_chunk"]
        i0, i1 = a.chunk * cs, min(n, (a.chunk + 1) * cs)
        if i0 >= n:
            return
        fn = od / f"Cxy_{i0:06d}_{i1:06d}.npy"
        if fn.exists():
            log("exists", fn)
            return
        t0 = time.perf_counter()
        block = np.zeros((i1 - i0, n), dtype=np.int32)
        cpu = 0.0
        with Pool(procs, initializer=_init, initargs=(seqs, a.comp)) as pool:
            for i, row, c in pool.imap_unordered(_row, range(i0, i1)):
                block[i - i0] = row
                cpu += c
        np.save(fn, block)
        write_json({"rows": [i0, i1], "wall_s": time.perf_counter() - t0, "worker_cpu_s": cpu,
                    "n_calls": int((i1 - i0) * n), "procs": procs, "peak_rss_mb": peak_rss_mb(),
                    "host": os.uname().nodename},
                   od / f"timing_rows_{i0:06d}_{i1:06d}.json")
        log(a.ds, a.comp, a.variant, f"rows {i0}-{i1} done in {time.perf_counter() - t0:.1f}s")

    elif a.cmd == "merge":
        cx = np.load(od / "Cx.npy").astype(np.float64)
        cxy = np.zeros((n, n), dtype=np.float64)
        files = sorted(od.glob("Cxy_*.npy"))
        covered = np.zeros(n, dtype=bool)
        for f in files:
            i0, i1 = map(int, f.stem.split("_")[1:3])
            cxy[i0:i1] = np.load(f)
            covered[i0:i1] = True
        if not covered.all():
            raise SystemExit(f"missing rows: {np.flatnonzero(~covered)[:10]} ...")
        mn = np.minimum.outer(cx, cx)
        mx = np.maximum.outer(cx, cx)
        ncd = (cxy - mn) / mx
        D = ((ncd + ncd.T) / 2).astype(np.float32)
        np.save(od / "D.npy", D)
        tim = [json.load(open(t)) for t in sorted(od.glob("timing_rows_*.json"))]
        cxt = json.load(open(od / "timing_cx.json"))
        write_json({
            "compressor": a.comp, "settings": COMPRESSOR_SETTINGS[a.comp], "variant": a.variant,
            "n_sequences": n, "n_ordered_pair_calls": int(n * n), "n_single_calls": n,
            "total_compression_cpu_s": float(sum(t["worker_cpu_s"] for t in tim) + cxt["cpu_s"]),
            "mean_cpu_s_per_pair_call": float(sum(t["worker_cpu_s"] for t in tim) / (n * n)),
            "max_peak_rss_mb_per_job": float(max(t["peak_rss_mb"] for t in tim)),
            "hosts": sorted({t["host"] for t in tim}),
            "D_bytes": int(D.nbytes),
            "ncd_range": [float(ncd.min()), float(ncd.max())],
        }, od / "timing.json")
        for f in files:
            f.unlink()
        log(a.ds, a.comp, a.variant, "merged; D shape", D.shape)


if __name__ == "__main__":
    main()
