"""Load raw datasets into a common format and write a dataset report.

Output per dataset (data/processed/<ds>/):
  sequences.csv   columns: idx,id,label,sequence   (idx = row index used everywhere)
  sequences.fasta headers ">idx" (used by MMseqs2)
  report.json     dataset statistics
"""
import argparse
import collections
import csv
import hashlib

import numpy as np
import pandas as pd

from common import ensure_dir, load_config, log, p, write_json

GZIP_WINDOW = 32768
BZ2_BLOCK = 900_000


def _read_fasta(path):
    recs, hdr, buf = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if hdr is not None:
                    recs.append((hdr, "".join(buf)))
                hdr, buf = line[1:].strip(), []
            elif line.strip():
                buf.append(line.strip())
    if hdr is not None:
        recs.append((hdr, "".join(buf)))
    return recs


def load_raw(cfg, ds):
    d = cfg["datasets"][ds]
    path = p(cfg, d["raw"])
    fmt = d["format"]
    rows = []
    if fmt == "tsv_seq_class":  # Kaggle human_data.txt: "sequence<TAB>class" with header
        df = pd.read_csv(path, sep="\t")
        names = d.get("label_names", {})
        for i, (s, c) in enumerate(zip(df.iloc[:, 0], df.iloc[:, 1])):
            rows.append((f"seq{i}", names.get(str(c), str(c)), str(s)))
    elif fmt == "fasta_label_last_field":
        sep = d.get("header_sep", "|")
        for i, (h, s) in enumerate(_read_fasta(path)):
            parts = h.split(sep)
            rows.append((parts[0].strip() or f"seq{i}", parts[-1].strip(), s))
    elif fmt == "csv":  # columns: id (optional), sequence, label
        df = pd.read_csv(path)
        for i, r in df.iterrows():
            rows.append((str(r.get("id", f"seq{i}")), str(r["label"]), str(r["sequence"])))
    else:
        raise ValueError(f"unknown format {fmt}")
    out = pd.DataFrame(rows, columns=["id", "label", "sequence"])
    out["sequence"] = out["sequence"].str.strip().str.upper()
    out.insert(0, "idx", np.arange(len(out)))
    return out


def window_stats(lengths):
    """Concatenation sizes (1 byte per residue) for distinct pairs and self-pairs."""
    L = np.sort(np.asarray(lengths))[::-1]
    n = len(L)
    max_distinct = int(L[0] + L[1]) if n > 1 else int(L[0])
    max_self = int(2 * L[0])
    asc = L[::-1]
    # count distinct unordered pairs with sum > window (two pointers on sorted array)
    def count_over(th):
        lo, hi, c = 0, n - 1, 0
        while lo < hi:
            if asc[lo] + asc[hi] > th:
                c += hi - lo
                hi -= 1
            else:
                lo += 1
        return c
    over_distinct = count_over(GZIP_WINDOW)
    over_self = int((2 * asc > GZIP_WINDOW).sum())
    return {
        "max_concat_distinct_pair_bytes": max_distinct,
        "max_concat_self_pair_bytes": max_self,
        "n_distinct_pairs": n * (n - 1) // 2,
        "n_distinct_pairs_over_32KB": int(over_distinct),
        "n_self_pairs_over_32KB": over_self,
        "n_distinct_pairs_over_bz2_block": int(count_over(BZ2_BLOCK)),
    }


def report(df, alphabet):
    lens = df["sequence"].str.len().to_numpy()
    comp = collections.Counter("".join(df["sequence"]))
    std = set("ACGT") if alphabet == "dna" else set("ACDEFGHIKLMNPQRSTVWY")
    nonstd = {k: v for k, v in comp.items() if k not in std}
    seq_hash = df["sequence"].map(lambda s: hashlib.md5(s.encode()).hexdigest())
    dup_groups = df.groupby(seq_hash)["label"].agg(list)
    dup_groups = dup_groups[dup_groups.map(len) > 1]
    per_class = {}
    for lab, g in df.groupby("label"):
        l = g["sequence"].str.len()
        per_class[lab] = {"n": int(len(g)), "len_min": int(l.min()), "len_median": float(l.median()),
                          "len_mean": float(l.mean()), "len_max": int(l.max())}
    shortest = df.assign(length=lens).nsmallest(20, "length")[["idx", "id", "label", "length", "sequence"]]
    return {
        "n_sequences": int(len(df)),
        "n_classes": int(df["label"].nunique()),
        "class_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
        "length": {"min": int(lens.min()), "p1": float(np.percentile(lens, 1)), "median": float(np.median(lens)),
                   "mean": float(lens.mean()), "p99": float(np.percentile(lens, 99)), "max": int(lens.max()),
                   "all_same_length": bool(lens.min() == lens.max())},
        "length_per_class": per_class,
        "n_shorter_than_50": int((lens < 50).sum()),
        "shortest_20": shortest.to_dict(orient="records"),
        "non_standard_characters": {k: int(v) for k, v in sorted(nonstd.items(), key=lambda x: -x[1])},
        "fraction_non_standard": float(sum(nonstd.values()) / max(1, sum(comp.values()))),
        "n_exact_duplicate_groups": int(len(dup_groups)),
        "n_sequences_in_duplicate_groups": int(dup_groups.map(len).sum()) if len(dup_groups) else 0,
        "n_duplicate_groups_with_conflicting_labels": int(dup_groups.map(lambda l: len(set(l)) > 1).sum()) if len(dup_groups) else 0,
        "window": window_stats(lens),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--datasets", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config)
    for ds in a.datasets or cfg["datasets"]:
        log("loading", ds)
        df = load_raw(cfg, ds)
        out = ensure_dir(p(cfg, "data", "processed", ds))
        df.to_csv(out / "sequences.csv", index=False, quoting=csv.QUOTE_MINIMAL)
        with open(out / "sequences.fasta", "w") as f:
            for i, s in zip(df["idx"], df["sequence"]):
                f.write(f">{i}\n{s}\n")
        rep = report(df, cfg["datasets"][ds]["alphabet"])
        write_json(rep, out / "report.json")
        log(ds, "n=", rep["n_sequences"], "classes=", rep["n_classes"], "window=", rep["window"])


if __name__ == "__main__":
    main()
