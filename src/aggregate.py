"""Aggregate per-seed results: means, SDs, 95% CIs, validation-based best configurations, paired
statistical comparisons with Holm correction, kernel-grid sensitivity, timing and PSD summaries.

Selection rule (no test information): for every representation, the classifier with the highest
mean VALIDATION macro-F1 is its 'best' configuration. 'Ours' = best compression-based KPCA
representation by validation macro-F1. Each run (seed) uses a different data split and a different
model random seed; comparisons are paired by seed (identical splits for all methods).
"""
import argparse
import warnings
warnings.filterwarnings("ignore")
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import ensure_dir, load_config, p, read_json

METRICS = ["acc", "prec_w", "rec_w", "f1_w", "prec_m", "rec_m", "f1_m", "auc", "train_s", "predict_s_per_seq",
           "val_acc", "val_f1_m"]


def holm(pvals):
    pv = np.asarray(pvals, dtype=float)
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * pv[i]))
        adj[i] = running
    return adj


def load_all(cfg):
    frames = []
    for f in p(cfg, "results").glob("*/*/seed*.csv"):
        if f.name.endswith("_kernel_grid.csv"):
            continue
        d = pd.read_csv(f)
        d["dataset"], d["scheme"] = f.parent.parent.name, f.parent.name
        d["seed"] = int(f.stem.replace("seed", ""))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def summarize(R):
    g = R.groupby(["dataset", "scheme", "family", "representation", "classifier"])
    rows = []
    for key, d in g:
        r = dict(zip(["dataset", "scheme", "family", "representation", "classifier"], key))
        r["n_runs"] = len(d)
        for m in METRICS:
            v = d[m].astype(float).to_numpy()
            r[f"{m}_mean"] = np.nanmean(v)
            r[f"{m}_sd"] = np.nanstd(v, ddof=1) if len(v) > 1 else np.nan
            half = stats.t.ppf(0.975, len(v) - 1) * r[f"{m}_sd"] / np.sqrt(len(v)) if len(v) > 1 else np.nan
            r[f"{m}_ci95"] = half
        rows.append(r)
    return pd.DataFrame(rows)


def best_configs(S):
    idx = S.groupby(["dataset", "scheme", "representation"])["val_f1_m_mean"].idxmax()
    return S.loc[idx].reset_index(drop=True)


def paired(R, B, metric_list=("f1_m", "acc", "auc")):
    out = []
    for (ds, sc), b in B.groupby(["dataset", "scheme"]):
        ours = b[b["family"] == "ours"]
        if ours.empty:
            continue
        o = ours.loc[ours["val_f1_m_mean"].idxmax()]
        ro = R[(R.dataset == ds) & (R.scheme == sc) & (R.representation == o.representation) & (R.classifier == o.classifier)]
        for _, other in b.iterrows():
            if other.representation == o.representation:
                continue
            rb = R[(R.dataset == ds) & (R.scheme == sc) & (R.representation == other.representation) & (R.classifier == other.classifier)]
            for metric in metric_list:
                m = pd.merge(ro[["seed", metric]], rb[["seed", metric]], on="seed", suffixes=("_ours", "_other")).dropna()
                if len(m) < 3:
                    continue
                d = m[f"{metric}_ours"] - m[f"{metric}_other"]
                sd = d.std(ddof=1)
                tt = stats.ttest_rel(m[f"{metric}_ours"], m[f"{metric}_other"])
                try:
                    wp = stats.wilcoxon(d).pvalue
                except ValueError:
                    wp = np.nan
                half = stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d))
                out.append({"dataset": ds, "scheme": sc, "metric": metric,
                            "ours": f"{o.representation} / {o.classifier}",
                            "other": f"{other.representation} / {other.classifier}", "other_family": other.family,
                            "n_paired_runs": len(d), "mean_ours": m[f"{metric}_ours"].mean(),
                            "mean_other": m[f"{metric}_other"].mean(), "mean_diff": d.mean(),
                            "ci95_low": d.mean() - half, "ci95_high": d.mean() + half,
                            "cohens_dz": d.mean() / sd if sd > 0 else np.nan,
                            "t_stat": tt.statistic, "p_ttest": tt.pvalue, "p_wilcoxon": wp})
    P = pd.DataFrame(out)
    if not P.empty:
        P["p_holm"] = np.nan
        for _, idx in P.groupby(["dataset", "scheme", "metric"]).groups.items():
            P.loc[idx, "p_holm"] = holm(P.loc[idx, "p_ttest"].fillna(1.0))
    return P


def kernel_grid(cfg):
    frames = []
    for f in p(cfg, "results").glob("*/random/seed*_kernel_grid.csv"):
        d = pd.read_csv(f)
        d["dataset"] = f.parent.parent.name
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    G = pd.concat(frames)
    return G.groupby(["dataset", "compressor", "sigma2_mult", "k"]).agg(
        val_acc_mean=("val_acc", "mean"), val_acc_sd=("val_acc", "std"),
        val_f1_m_mean=("val_f1_m", "mean"), n=("val_acc", "size")).reset_index()


def meta_summary(cfg):
    rows = []
    for f in p(cfg, "results").glob("*/*/seed*_meta.json"):
        m = read_json(f)
        for comp, c in m["compressors"].items():
            rows.append({"dataset": m["dataset"], "scheme": m["scheme"], "seed": m["seed"], "compressor": comp,
                         "n_train": m["n_train"], **c, "peak_rss_mb": m.get("peak_rss_mb")})
    return pd.DataFrame(rows)


def timing_summary(cfg):
    rows = []
    for ds in cfg["datasets"]:
        for comp in cfg["compressors"]:
            f = p(cfg, "data", "ncd", ds, comp, "timing.json")
            if f.exists():
                t = read_json(f)
                rows.append({"dataset": ds, "method": comp, "stage": "all-pairs NCD (CPU)",
                             "cpu_s": t["total_compression_cpu_s"], "peak_rss_mb": t["max_peak_rss_mb_per_job"],
                             "n": t["n_sequences"], "per_pair_call_s": t["mean_cpu_s_per_pair_call"]})
        for f in p(cfg, "data", "emb", ds).glob("*_timing_*.json") if p(cfg, "data", "emb", ds).exists() else []:
            t = read_json(f)
            rows.append({"dataset": ds, "method": t["model"], "stage": f"embedding ({t['device']})",
                         "wall_s": t["embed_wall_s"], "per_seq_s": t["embed_s_per_seq"], "n": t["n_sequences"],
                         "peak_rss_mb": t["peak_rss_mb"], "gpu": t.get("gpu"), "gpu_peak_mem_mb": t.get("gpu_peak_mem_mb"),
                         "cpu_threads": t.get("cpu_threads"), "host": t.get("host")})
        f = p(cfg, "results", "timing", f"{ds}_ncd_query.json")
        if f.exists():
            t = read_json(f)
            for comp in cfg["compressors"]:
                if comp in t:
                    rows.append({"dataset": ds, "method": comp, "stage": "new-sequence embedding (1 CPU core)",
                                 "per_seq_s": t[comp]["total_s_per_query_1core"], "n_reference": t["n_reference"],
                                 "host": t["host"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = ensure_dir(p(cfg, "results", "summary"))
    R = load_all(cfg)
    R.to_csv(out / "all_runs.csv", index=False)
    S = summarize(R)
    S.to_csv(out / "summary.csv", index=False)
    B = best_configs(S)
    B.to_csv(out / "best_by_validation.csv", index=False)
    paired(R, B).to_csv(out / "paired_tests.csv", index=False)
    kernel_grid(cfg).to_csv(out / "kernel_grid_validation.csv", index=False)
    meta_summary(cfg).to_csv(out / "kernel_meta.csv", index=False)
    timing_summary(cfg).to_csv(out / "timing.csv", index=False)
    cl = []
    for f in p(cfg, "data", "splits").glob("*/cluster_id*/cluster_stats.json"):
        s = read_json(f)
        s["dataset"] = f.parent.parent.name
        x = f.parent / "cross_split_identity_seed0.json"
        if x.exists():
            xs = read_json(x)
            s["test_max_identity_median"] = xs["max_identity_quantiles"]["0.5"]
            s["test_max_identity_p99"] = xs["max_identity_quantiles"]["0.99"]
            s["test_fraction_above_threshold"] = xs["fraction_test_above_threshold"]
        cl.append({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in s.items()})
    pd.DataFrame(cl).to_csv(out / "cluster_stats.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
