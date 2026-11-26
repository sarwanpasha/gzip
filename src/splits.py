"""Random stratified splits and MMseqs2 cluster-disjoint splits, with full statistics.

Outputs:
  data/splits/<ds>/random/seed<k>.json
  data/splits/<ds>/cluster_id<t>/seed<k>.json
  data/splits/<ds>/cluster_id<t>/clusters.tsv        (representative, member) from MMseqs2
  data/splits/<ds>/cluster_id<t>/cluster_stats.json  counts, size distribution, feasibility
  data/splits/<ds>/cluster_id<t>/cross_split_identity_seed0.json
  data/splits/<ds>/mmseqs_commands.txt               exact commands + MMseqs2 version
"""
import argparse
import shutil
import subprocess
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from common import ensure_dir, load_config, log, p, read_json, write_json


def stratified_split(labels, fracs, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    tr, va, te = [], [], []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        rng.shuffle(idx)
        n = len(idx)
        n_te = int(round(fracs[2] * n))
        n_va = int(round(fracs[1] * n))
        if n >= 3:
            n_te, n_va = max(n_te, 1), max(n_va, 1)
        n_tr = n - n_te - n_va
        if n_tr < 1:  # very rare class: keep at least one training example
            n_tr, n_va, n_te = 1, 0, n - 1
        tr += idx[:n_tr].tolist()
        va += idx[n_tr:n_tr + n_va].tolist()
        te += idx[n_tr + n_va:].tolist()
    return {"train": sorted(tr), "val": sorted(va), "test": sorted(te)}


def run(cmd, log_file):
    with open(log_file, "a") as f:
        f.write(" ".join(cmd) + "\n")
    log("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def mmseqs_cluster(fasta, out_prefix, tmp, min_id, cov, log_file):
    cmd = ["mmseqs", "easy-cluster", str(fasta), str(out_prefix), str(tmp),
           "--min-seq-id", str(min_id), "-c", str(cov), "--cov-mode", "0", "--threads", "8"]
    run(cmd, log_file)
    tsv = pd.read_csv(f"{out_prefix}_cluster.tsv", sep="\t", header=None, names=["rep", "member"])
    return tsv


def group_split(groups, labels, fracs, seed):
    """Assign whole clusters to train/val/test, aiming at the target fractions.
    Clusters are processed largest first; each goes to the split that is furthest
    below its target size (random tie-breaking)."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.lexsort((rng.random(len(uniq)), -counts))
    n = len(groups)
    target = np.array(fracs) * n
    fill = np.zeros(3)
    assign = {}
    for gi in order:
        deficit = (target - fill) / np.maximum(target, 1)
        s = int(np.argmax(deficit + 1e-9 * rng.random(3)))
        assign[uniq[gi]] = s
        fill[s] += counts[gi]
    part = np.array([assign[g] for g in groups])
    return {"train": np.flatnonzero(part == 0).tolist(),
            "val": np.flatnonzero(part == 1).tolist(),
            "test": np.flatnonzero(part == 2).tolist()}


def split_stats(split, labels, fracs):
    labels = np.asarray(labels)
    n = len(labels)
    out = {}
    tr_classes = set(labels[split["train"]]) if split["train"] else set()
    for k in ("train", "val", "test"):
        idx = split[k]
        out[k] = {"n": len(idx), "fraction": len(idx) / n,
                  "class_counts": {str(c): int(v) for c, v in Counter(labels[idx]).items()}}
    te = labels[split["test"]] if split["test"] else np.array([])
    out["test_samples_with_class_unseen_in_train"] = int(sum(1 for c in te if c not in tr_classes))
    out["n_classes_train"] = len(tr_classes)
    out["n_classes_test"] = len(set(te))
    dev = max(abs(out[k]["fraction"] - f) for k, f in zip(("train", "val", "test"), fracs))
    out["max_fraction_deviation"] = dev
    out["feasible"] = bool(len(split["train"]) > 0 and len(split["test"]) > 0 and dev <= 0.10
                           and out["test_samples_with_class_unseen_in_train"] < 0.5 * max(1, len(te)))
    return out


def cross_split_identity(fasta, split, workdir, alphabet, log_file, threshold):
    """Highest identity of each test sequence to any training sequence (MMseqs2 easy-search hits)."""
    seqs = {}
    name = None
    for line in open(fasta):
        line = line.strip()
        if line.startswith(">"):
            name = int(line[1:])
        else:
            seqs[name] = line
    w = ensure_dir(workdir)
    for part in ("train", "test"):
        with open(w / f"{part}.fasta", "w") as f:
            for i in split[part]:
                f.write(f">{i}\n{seqs[i]}\n")
    cmd = ["mmseqs", "easy-search", str(w / "test.fasta"), str(w / "train.fasta"), str(w / "hits.m8"), str(w / "tmp"),
           "--threads", "8", "-s", "7.5", "--max-seqs", "50"]
    if alphabet == "dna":
        cmd += ["--search-type", "3"]
    run(cmd, log_file)
    try:
        hits = pd.read_csv(w / "hits.m8", sep="\t", header=None)
        best = hits.groupby(0)[2].max()
    except pd.errors.EmptyDataError:
        best = pd.Series(dtype=float)
    pid = np.array([best.get(i, 0.0) for i in split["test"]], dtype=float)
    shutil.rmtree(w / "tmp", ignore_errors=True)
    return {"n_test": len(pid), "n_test_without_any_hit": int((pid == 0).sum()),
            "max_identity_quantiles": {q: float(np.quantile(pid, q)) for q in (0.5, 0.9, 0.99, 1.0)},
            "identity_scale": "fident (0-1), MMseqs2 default output",
            "threshold": threshold,
            "fraction_test_above_threshold": float((pid > threshold).mean()) if len(pid) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--no-mmseqs", action="store_true", help="only random splits (for local testing)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    fr = cfg["split_fractions"]
    for ds in a.datasets or cfg["datasets"]:
        df = pd.read_csv(p(cfg, "data", "processed", ds, "sequences.csv"))
        labels = df["label"].astype(str).to_numpy()
        base = ensure_dir(p(cfg, "data", "splits", ds))
        rdir = ensure_dir(base / "random")
        for s in range(cfg["n_seeds"]):
            sp = stratified_split(labels, fr, seed=1000 + s)
            write_json({**sp, "stats": split_stats(sp, labels, fr)}, rdir / f"seed{s}.json")
        log(ds, "random splits written")
        if a.no_mmseqs or not cfg["cluster_thresholds"][cfg["datasets"][ds]["alphabet"]]:
            continue
        alphabet = cfg["datasets"][ds]["alphabet"]
        cmd_log = base / "mmseqs_commands.txt"
        ver = subprocess.run(["mmseqs", "version"], capture_output=True, text=True).stdout.strip()
        with open(cmd_log, "w") as f:
            f.write(f"# MMseqs2 version: {ver}\n")
        fasta = p(cfg, "data", "processed", ds, "sequences.fasta")
        for t in cfg["cluster_thresholds"][alphabet]:
            cdir = ensure_dir(base / f"cluster_id{t}")
            tsv = mmseqs_cluster(fasta, cdir / "mm", cdir / "tmp", t, cfg["cluster_coverage"], cmd_log)
            shutil.rmtree(cdir / "tmp", ignore_errors=True)
            tsv.to_csv(cdir / "clusters.tsv", sep="\t", index=False)
            rep_of = dict(zip(tsv["member"].astype(int), tsv["rep"].astype(int)))
            groups = np.array([rep_of[i] for i in df["idx"]])
            sizes = np.array(sorted(Counter(groups).values(), reverse=True))
            pure = defaultdict(set)
            for g, l in zip(groups, labels):
                pure[g].add(l)
            stats = {"min_seq_id": t, "coverage": cfg["cluster_coverage"], "n_sequences": int(len(df)),
                     "n_clusters": int(len(sizes)), "largest_cluster": int(sizes[0]),
                     "largest_cluster_fraction": float(sizes[0] / len(df)),
                     "n_singletons": int((sizes == 1).sum()),
                     "size_quantiles": {q: float(np.quantile(sizes, q)) for q in (0.5, 0.9, 0.99)},
                     "top10_sizes": sizes[:10].tolist(),
                     "n_clusters_with_mixed_labels": int(sum(len(v) > 1 for v in pure.values()))}
            feas = []
            for s in range(cfg["n_seeds"]):
                sp = group_split(groups, labels, fr, seed=2000 + s)
                st = split_stats(sp, labels, fr)
                feas.append(st["feasible"])
                write_json({**sp, "stats": st}, cdir / f"seed{s}.json")
            stats["feasible_all_seeds"] = bool(all(feas))
            stats["n_feasible_seeds"] = int(sum(feas))
            if all(feas):
                sp0 = {k: v for k, v in read_json(cdir / "seed0.json").items() if k != "stats"}
                xs = cross_split_identity(fasta, sp0, cdir / "xsearch", alphabet, cmd_log, t)
                write_json(xs, cdir / "cross_split_identity_seed0.json")
            write_json(stats, cdir / "cluster_stats.json")
            log(ds, f"id={t}", "clusters=", stats["n_clusters"], "largest=", stats["largest_cluster"],
                "feasible seeds=", stats["n_feasible_seeds"])


if __name__ == "__main__":
    main()
