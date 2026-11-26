"""One task = (dataset, split scheme, seed). Builds every representation inductively from the
training partition and evaluates all downstream classifiers.

Writes
  results/<ds>/<scheme>/seed<k>.csv            one row per (representation, classifier)
  results/<ds>/<scheme>/seed<k>_meta.json      selected sigma^2/k, PSD checks, feature timings
  results/<ds>/<scheme>/seed<k>_kernel_grid.csv validation scores over the sigma^2 x k grid
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import classify as C
import features as F
from common import ensure_dir, load_config, log, p, peak_rss_mb, read_json, write_json


def load_task(cfg, ds, scheme, seed):
    df = pd.read_csv(p(cfg, "data", "processed", ds, "sequences.csv"))
    sp = read_json(p(cfg, "data", "splits", ds, scheme, f"seed{seed}.json"))
    return df, {k: np.array(sp[k], dtype=int) for k in ("train", "val", "test")}


def select_kernel(D, sp, y, cfg, seed):
    """Choose (sigma^2 multiplier, k) on the VALIDATION partition with a fast classifier."""
    phi_tr = F.profiles(D, sp["train"], sp["train"])
    phi_va = F.profiles(D, sp["val"], sp["train"])
    ks = sorted(cfg["kernel"]["n_components"])
    rows = []
    for m in cfg["kernel"]["sigma2_multipliers"]:
        rk = F.RowKernelPCA(m, max(ks)).fit(phi_tr)
        Ztr, Zva = rk.Z_tr, rk.transform(phi_va)
        for k in ks:
            if k > Ztr.shape[1]:
                continue
            model = C.make(cfg["kernel"]["selection_classifier"], {"C": 1.0}, seed)
            model.fit(Ztr[:, :k], y[sp["train"]])
            pr = model.predict(Zva[:, :k])
            met = C.metrics(y[sp["val"]], pr)
            rows.append({"sigma2_mult": m, "k": k, "val_acc": met["acc"], "val_f1_m": met["f1_m"]})
    g = pd.DataFrame(rows)
    key = "val_f1_m" if cfg["kernel"]["selection_metric"] == "macro_f1" else "val_acc"
    best = g.sort_values([key, "k"], ascending=[False, True]).iloc[0]
    return float(best["sigma2_mult"]), int(best["k"]), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)))
    ap.add_argument("--ds")
    ap.add_argument("--scheme")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--classifiers", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.task >= 0:
        tasks = [l.split() for l in open(p(cfg, "results", "tasks.txt")).read().split("\n") if l.strip()]
        ds, scheme, seed = tasks[a.task][0], tasks[a.task][1], int(tasks[a.task][2])
    else:
        ds, scheme, seed = a.ds, a.scheme, a.seed
    clfs = a.classifiers or cfg["classifiers"]
    alphabet = cfg["datasets"][ds]["alphabet"]
    df, sp = load_task(cfg, ds, scheme, seed)
    y = df["label"].astype(str).to_numpy()
    seqs = df["sequence"].astype(str).tolist()
    od = ensure_dir(p(cfg, "results", ds, scheme))
    out_csv = od / f"seed{seed}.csv"
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["representation"], prev["classifier"]))
    meta = {"dataset": ds, "scheme": scheme, "seed": seed,
            "n_train": int(len(sp["train"])), "n_val": int(len(sp["val"])), "n_test": int(len(sp["test"])),
            "compressors": {}}
    reps = {}  # name -> (Xtr, Xva, Xte, family)
    special_nn = {}
    grids = []
    rng = np.random.default_rng(10_000 + seed)

    for comp in cfg["compressors"]:
        dpath = p(cfg, "data", "ncd", ds, comp, "D.npy")
        if not dpath.exists():
            log("missing", dpath)
            continue
        D = np.load(dpath, mmap_mode="r")
        m, k, g = select_kernel(D, sp, y, cfg, seed)
        g.insert(0, "compressor", comp)
        grids.append(g)
        phi = {part: F.profiles(D, sp[part], sp["train"]) for part in ("train", "val", "test")}
        t = time.perf_counter()
        rk = F.RowKernelPCA(m, k).fit(phi["train"])
        fit_s = time.perf_counter() - t
        t = time.perf_counter()
        Zva, Zte = rk.transform(phi["val"]), rk.transform(phi["test"])
        proj_s = (time.perf_counter() - t) / (len(sp["val"]) + len(sp["test"]))
        cm = {"sigma2_mult": m, "k": k, "sigma2": rk.sigma2, "kpca_fit_s": fit_s,
              "projection_s_per_seq_excluding_ncd": proj_s,
              "row_kernel_min_eig_centered": F.min_eig(rk.K_tr, center=True)}
        reps[f"{comp}+KPCA"] = (rk.Z_tr, Zva, Zte, "ours")
        # ablations
        reps[f"{comp}+dist"] = (phi["train"], phi["val"], phi["test"], "ablation")
        Ptr, (Pva, Pte) = F.linear_pca(phi["train"], [phi["val"], phi["test"]], k)
        reps[f"{comp}+PCA"] = (Ptr, Pva, Pte, "ablation")
        dk = F.DirectKernelPCA(m, k).fit(F.profiles(D, sp["train"], sp["train"]))
        reps[f"{comp}+KPCA-direct"] = (dk.Z_tr, dk.transform(phi["val"]), dk.transform(phi["test"]), "ablation")
        cm["direct_kernel_min_eig_centered"] = dk.min_eig_centered
        cm["direct_kernel_min_eig_raw"] = dk.min_eig_raw
        special_nn[f"{comp}+kNN-NCD"] = (phi["test"], phi["val"])
        # controls
        perm = rng.permutation(len(sp["train"]))
        reps[f"{comp}+KPCA [shuffled labels]"] = (rk.Z_tr, Zva, Zte, "control_labels", perm)
        dsh = p(cfg, "data", "ncd", ds, f"{comp}_shuffled", "D.npy")
        if dsh.exists():
            Ds = np.load(dsh, mmap_mode="r")
            ph = {part: F.profiles(Ds, sp[part], sp["train"]) for part in ("train", "val", "test")}
            rs = F.RowKernelPCA(m, k).fit(ph["train"])
            reps[f"{comp}+KPCA [shuffled residues]"] = (rs.Z_tr, rs.transform(ph["val"]), rs.transform(ph["test"]), "control_residues")
        meta["compressors"][comp] = cm
        del D

    kk = max([v["k"] for v in meta["compressors"].values()] or [64])
    G = rng.standard_normal((len(df), kk))
    reps["Random Gaussian features"] = (G[sp["train"]], G[sp["val"]], G[sp["test"]], "control_random")

    # simple baselines
    for name, X in (("Length", F.length_features(seqs)),
                    ("Composition", F.composition_features(seqs, alphabet)),
                    ("3-mer spectrum", F.kmer_features(seqs, alphabet, 3))):
        reps[name] = (X[sp["train"]], X[sp["val"]], X[sp["test"]], "simple")
    cx = {c: np.load(p(cfg, "data", "ncd", ds, c, "Cx.npy")) for c in cfg["compressors"]
          if p(cfg, "data", "ncd", ds, c, "Cx.npy").exists()}
    if cx:
        X = F.compressed_length_features(cx, seqs)
        reps["Compressed length"] = (X[sp["train"]], X[sp["val"]], X[sp["test"]], "simple")

    # frozen foundation-model embeddings
    for fm in cfg["foundation_models"][alphabet]:
        e = p(cfg, "data", "emb", ds, f"{fm}.npy")
        if e.exists():
            E = np.load(e)
            reps[fm] = (E[sp["train"]], E[sp["val"]], E[sp["test"]], "foundation")

    rows = []

    def flush():
        if rows:
            new = pd.DataFrame(rows)
            if out_csv.exists():
                new = pd.concat([pd.read_csv(out_csv), new], ignore_index=True)
            new.to_csv(out_csv, index=False)
            rows.clear()

    ytr, yva, yte = y[sp["train"]], y[sp["val"]], y[sp["test"]]
    for name, (phi_te, phi_va) in special_nn.items():
        if (name, "kNN-NCD") in done:
            continue
        r = C.nn_on_distances(phi_te, phi_va, ytr, yva, yte)
        rows.append({"representation": name, "family": "ablation", **r})
    flush()
    for name, v in reps.items():
        Xtr, Xva, Xte, fam = v[:4]
        ytr_use, yva_use = ytr, yva
        if fam == "control_labels":
            ytr_use = ytr[v[4]]
            yva_use = yva[np.random.default_rng(20_000 + seed).permutation(len(yva))]
        for c in clfs:
            if (name, c) in done:
                continue
            t = time.perf_counter()
            r, _ = C.evaluate(c, Xtr, ytr_use, Xva, yva_use, Xte, yte, seed)
            rows.append({"representation": name, "family": fam, "dim": Xtr.shape[1], **r})
            log(ds, scheme, seed, name, c, f"acc={r['acc']:.3f} f1m={r['f1_m']:.3f}",
                f"({time.perf_counter() - t:.1f}s)")
        flush()

    if grids:
        pd.concat(grids).to_csv(od / f"seed{seed}_kernel_grid.csv", index=False)
    meta["peak_rss_mb"] = peak_rss_mb()
    write_json(meta, od / f"seed{seed}_meta.json")
    log("done", ds, scheme, seed)


if __name__ == "__main__":
    main()
