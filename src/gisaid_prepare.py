"""Build the GISAID spike benchmark (lineage classification with a date-based split).

Inputs (GISAID data cannot be redistributed; download them yourself from GISAID):
  --fasta   spike protein FASTA from GISAID ("spikeprot" download). Header fields separated by '|':
            Spike | virus name | collection date | EPI_ISL accession | ... | host | ...
  --labels  CSV without header: accession, Pango lineage, ... (one row per sequence)

Steps
  1. keep human-host records with a complete collection date (YYYY-MM-DD) and a lineage label
  2. drop a trailing '*'; keep sequences of length --min-len..--max-len with only the 20
     standard amino acids (no X, no internal stop)
  3. remove exact duplicates: each distinct sequence is kept once, with its earliest date;
     sequences that occur under more than one lineage are removed
  4. keep the --n-lineages lineages with the most distinct sequences, and draw --pool distinct
     sequences per lineage at random
  5. for every seed, draw --per-class of the pool sequences per lineage and split them
       temporal: per lineage, the earliest 60% by collection date -> train, next 10% -> validation,
                 latest 30% -> test (every test sequence was collected no earlier than any
                 training sequence of its lineage)
       random:   the same sequences, split 60/10/30 at random within each lineage
     All methods use identical splits, so comparisons are paired by seed.

Outputs
  data/raw/<ds>.csv                    id, label, sequence (input for data.py)
  data/raw/<ds>_dates.csv              idx, id, label, date
  data/raw/<ds>_prepare_report.json    counts after every filter, chosen lineages, date ranges
  data/splits/<ds>/{temporal,random}/seed<k>.json
"""
import argparse
import csv
import datetime
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from common import ensure_dir, load_config, log, p, write_json
from splits import split_stats

AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def valid_date(d):
    """Complete, real calendar date (GISAID uses e.g. 2021-03 or 2021-00-00 for partial dates)."""
    if not DATE.match(d):
        return False
    try:
        datetime.date.fromisoformat(d)
        return True
    except ValueError:
        return False


def read_fasta(path):
    hdr, buf = None, []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(buf)
                hdr, buf = line[1:].rstrip("\n\r"), []
            else:
                buf.append(line.strip())
    if hdr is not None:
        yield hdr, "".join(buf)


def split_one(idx, dates, rng, fr, temporal):
    idx = np.asarray(idx)
    if temporal:
        order = np.lexsort((rng.random(len(idx)), dates))  # by date, random tie-break
    else:
        order = rng.permutation(len(idx))
    idx = idx[order]
    n = len(idx)
    n_tr, n_va = int(round(fr[0] * n)), int(round(fr[1] * n))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ds", default="gisaid_spike")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--min-len", type=int, default=1260)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--n-lineages", type=int, default=20)
    ap.add_argument("--pool", type=int, default=750)
    ap.add_argument("--per-class", type=int, default=500)
    a = ap.parse_args()
    cfg = load_config(a.config)
    fr = cfg["split_fractions"]
    rep = {"filters": {}}

    lab = pd.read_csv(a.labels, header=None, usecols=[0, 1], names=["acc", "lineage"], dtype=str)
    lab = lab.dropna()
    lab = lab[(lab.lineage != "None") & (lab.lineage != "") & (lab.lineage != "Unassigned")]
    lineage_of = dict(zip(lab.acc.str.strip(), lab.lineage.str.strip()))
    rep["filters"]["label_rows_with_lineage"] = len(lineage_of)
    del lab
    log("labels", len(lineage_of))

    cnt = defaultdict(int)
    first = {}             # sequence -> (date, accession, lineage)
    lineages_of_seq = defaultdict(set)
    for h, s in read_fasta(a.fasta):
        cnt["fasta_records"] += 1
        f = h.split("|")
        if len(f) < 7:
            cnt["bad_header"] += 1
            continue
        date, acc, host = f[2].strip(), f[3].strip(), f[6].strip()
        lin = lineage_of.get(acc)
        if lin is None:
            cnt["no_lineage"] += 1
            continue
        if host.lower() != "human":
            cnt["non_human_host"] += 1
            continue
        if not valid_date(date):
            cnt["incomplete_date"] += 1
            continue
        s = s.upper().rstrip("*")
        if not (a.min_len <= len(s) <= a.max_len):
            cnt["length_out_of_range"] += 1
            continue
        if not AA.match(s):
            cnt["non_standard_residues"] += 1
            continue
        cnt["passed"] += 1
        lineages_of_seq[s].add(lin)
        prev = first.get(s)
        if prev is None or date < prev[0]:
            first[s] = (date, acc, lin)
        if cnt["fasta_records"] % 500_000 == 0:
            log("read", cnt["fasta_records"], "passed", cnt["passed"], "distinct", len(first))
    rep["filters"].update(dict(cnt))
    rep["filters"]["distinct_sequences"] = len(first)
    conflict = {s for s, l in lineages_of_seq.items() if len(l) > 1}
    rep["filters"]["distinct_sequences_with_conflicting_lineages_removed"] = len(conflict)
    del lineages_of_seq
    by_lin = defaultdict(list)
    for s, (d, acc, lin) in first.items():
        if s not in conflict:
            by_lin[lin].append((d, acc, s))
    del first
    sizes = sorted(((len(v), k) for k, v in by_lin.items()), reverse=True)
    chosen = [k for n, k in sizes[:a.n_lineages] if n >= a.pool]
    rep["distinct_sequences_per_lineage_top40"] = {k: n for n, k in sizes[:40]}
    rep["chosen_lineages"] = chosen
    log("chosen", len(chosen), [(k, len(by_lin[k])) for k in chosen])
    log("top lineages by distinct sequences", sizes[:30])

    rng = np.random.default_rng(0)
    rows = []
    for lin in chosen:
        v = by_lin[lin]
        pick = rng.choice(len(v), size=a.pool, replace=False)
        for j in sorted(pick):
            d, acc, s = v[j]
            rows.append((acc, lin, s, d))
    df = pd.DataFrame(rows, columns=["id", "label", "sequence", "date"])
    raw = ensure_dir(p(cfg, "data", "raw")) / f"{a.ds}.csv"
    df[["id", "label", "sequence"]].to_csv(raw, index=False, quoting=csv.QUOTE_MINIMAL)
    dd = df[["id", "label", "date"]].copy()
    dd.insert(0, "idx", np.arange(len(dd)))
    dd.to_csv(p(cfg, "data", "raw", f"{a.ds}_dates.csv"), index=False)
    rep["n_sequences"] = int(len(df))
    rep["date_range_per_lineage"] = {lin: [g.date.min(), g.date.max()] for lin, g in df.groupby("label")}

    labels = df["label"].to_numpy()
    dates = pd.to_datetime(df["date"]).to_numpy().astype("datetime64[D]").astype(np.int64)
    for scheme in ("temporal", "random"):
        sd = ensure_dir(p(cfg, "data", "splits", a.ds, scheme))
        for k in range(cfg["n_seeds"]):
            r_sub = np.random.default_rng(3000 + k)
            r_split = np.random.default_rng(4000 + k)
            tr, va, te = [], [], []
            for lin in chosen:
                idx = np.flatnonzero(labels == lin)
                sub = np.sort(r_sub.choice(idx, size=a.per_class, replace=False))
                a_, b_, c_ = split_one(sub, dates[sub], r_split, fr, scheme == "temporal")
                tr += a_.tolist(); va += b_.tolist(); te += c_.tolist()
            sp = {"train": sorted(tr), "val": sorted(va), "test": sorted(te)}
            used = np.array(sp["train"] + sp["val"] + sp["test"])  # stats relative to the seed's subsample
            pos = {int(i): j for j, i in enumerate(used)}
            st = split_stats({kk: [pos[i] for i in sp[kk]] for kk in sp}, labels[used], fr)
            later = [float((dates[np.intersect1d(sp["test"], np.flatnonzero(labels == lin))].min()
                            >= dates[np.intersect1d(sp["train"], np.flatnonzero(labels == lin))].max()))
                     for lin in chosen]
            st["fraction_lineages_with_all_test_after_train"] = float(np.mean(later))
            st["median_date_train"] = str(np.datetime64(int(np.median(dates[sp["train"]])), "D"))
            st["median_date_test"] = str(np.datetime64(int(np.median(dates[sp["test"]])), "D"))
            write_json({**sp, "stats": st}, sd / f"seed{k}.json")
        log(a.ds, scheme, "splits written")
    write_json(rep, p(cfg, "data", "raw", f"{a.ds}_prepare_report.json"))
    log("done", a.ds, len(df))


if __name__ == "__main__":
    main()
