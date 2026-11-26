"""CPU cost of embedding NEW sequences with the NCD pipeline (run on the same node type as the
foundation-model CPU timing). For each query: compress the query, compress query+reference and
reference+query for every training (reference) sequence, then project with the fitted kernel PCA.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import features as F
from common import ensure_dir, load_config, log, p, peak_rss_mb, read_json, write_json
from ncd import COMPRESSORS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ds", required=True)
    ap.add_argument("--n-queries", type=int, default=20)
    a = ap.parse_args()
    cfg = load_config(a.config)
    df = pd.read_csv(p(cfg, "data", "processed", a.ds, "sequences.csv"))
    seqs = df["sequence"].astype(str).tolist()
    sp = read_json(p(cfg, "data", "splits", a.ds, "random", "seed0.json"))
    meta = read_json(p(cfg, "results", a.ds, "random", "seed0_meta.json"))
    tr = np.array(sp["train"])
    q = np.random.default_rng(0).choice(sp["test"], size=min(a.n_queries, len(sp["test"])), replace=False)
    out = {"host": os.uname().nodename, "n_reference": int(len(tr)), "n_queries": int(len(q)),
           "note": "single CPU core; NCD for a query is embarrassingly parallel over references"}
    for comp, fn in COMPRESSORS.items():
        D = p(cfg, "data", "ncd", a.ds, comp, "D.npy")
        if not D.exists() or comp not in meta["compressors"]:
            continue
        D = np.load(D, mmap_mode="r")
        cx = np.load(p(cfg, "data", "ncd", a.ds, comp, "Cx.npy")).astype(float)
        t = time.perf_counter()
        prof = []
        for i in q:
            s = seqs[i].encode()
            ci = fn(s)
            row = []
            for j in tr:
                r = seqs[j]
                a1 = fn((seqs[i] + r).encode())
                a2 = fn((r + seqs[i]).encode())
                mn, mx = min(ci, cx[j]), max(ci, cx[j])
                row.append(((a1 - mn) / mx + (a2 - mn) / mx) / 2)
            prof.append(row)
        ncd_s = (time.perf_counter() - t) / len(q)
        m = meta["compressors"][comp]
        rk = F.RowKernelPCA(m["sigma2_mult"], m["k"]).fit(F.profiles(D, tr, tr))
        t = time.perf_counter()
        rk.transform(np.array(prof))
        proj_s = (time.perf_counter() - t) / len(q)
        out[comp] = {"ncd_s_per_query_1core": ncd_s, "projection_s_per_query": proj_s,
                     "total_s_per_query_1core": ncd_s + proj_s,
                     "max_abs_diff_vs_precomputed": float(np.abs(np.array(prof) - D[np.ix_(q, tr)]).max())}
        log(a.ds, comp, out[comp])
    out["peak_rss_mb"] = peak_rss_mb()
    write_json(out, ensure_dir(p(cfg, "results", "timing")) / f"{a.ds}_ncd_query.json")


if __name__ == "__main__":
    main()
