"""Generate LaTeX tables directly from the aggregated CSV/JSON outputs (no manual transcription).
Output: results/tables/*.tex (booktabs; \\usepackage{booktabs} required)."""
import argparse
import json

import numpy as np
import pandas as pd

from common import ensure_dir, load_config, p, read_json

NICE = {"human_dna": "Human DNA", "coronavirus_host": "Coronavirus Host", "spike7k": "Spike7k"}
FAM_ORDER = {"ours": 0, "foundation": 1, "simple": 2, "ablation": 3, "control_labels": 4,
             "control_residues": 5, "control_random": 6, "transductive": 7}
CLF_ORDER = ["SVM", "NB", "MLP", "KNN", "RF", "LR", "DT", "kNN-NCD"]


def esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def ms(r, m, nd=3):
    mu, sd = r[f"{m}_mean"], r[f"{m}_sd"]
    if pd.isna(mu):
        return "N/A"
    return f"{mu:.{nd}f}" if pd.isna(sd) else f"{mu:.{nd}f} $\\pm$ {sd:.{nd}f}"


def scheme_name(s):
    return "random split" if s == "random" else f"cluster-disjoint, {float(s.replace('cluster_id', '')) * 100:.0f}\\% identity"


def write(path, body):
    with open(path, "w") as f:
        f.write(body)


def main_tables(S, od):
    keep = ["ours", "foundation", "simple"]
    for (ds, sc), d in S.groupby(["dataset", "scheme"]):
        d = d[d.family.isin(keep)].copy()
        d["fo"] = d.family.map(FAM_ORDER)
        d["co"] = d.classifier.map({c: i for i, c in enumerate(CLF_ORDER)})
        d = d.sort_values(["fo", "representation", "co"])
        best = {m: d[f"{m}_mean"].max() for m in ("acc", "f1_w", "f1_m", "auc")}
        lines = [r"\begin{table}[!ht]", r"\centering", r"\resizebox{\textwidth}{!}{%",
                 r"\begin{tabular}{llcccccccc}", r"\toprule",
                 r"Representation & Classifier & Acc. & Prec. (w) & Rec. (w) & F1 (w) & F1 (macro) & ROC-AUC & Train (s) & Predict (ms/seq) \\",
                 r"\midrule"]
        prev = None
        for _, r in d.iterrows():
            rep = esc(r.representation) if r.representation != prev else ""
            if prev is not None and r.representation != prev:
                lines.append(r"\cmidrule{1-10}")
            prev = r.representation
            cells = []
            for m in ("acc", "prec_w", "rec_w", "f1_w", "f1_m", "auc"):
                c = ms(r, m)
                if m in best and np.isclose(r[f"{m}_mean"], best[m]):
                    c = r"\textbf{" + c + "}"
                cells.append(c)
            lines.append(f"{rep} & {r.classifier} & " + " & ".join(cells) +
                         f" & {r.train_s_mean:.2f} & {1000 * r.predict_s_per_seq_mean:.3f} \\\\")
        n = int(d.n_runs.max())
        lines += [r"\bottomrule", r"\end{tabular}}",
                  rf"\caption{{Classification results on the \textbf{{{NICE.get(ds, ds)}}} dataset ({scheme_name(sc)}; mean $\pm$ SD over {n} runs, each with a different split and model seed). "
                  r"Embeddings are fitted on the training partition only; classifier hyperparameters are selected on the validation partition. "
                  r"Prec./Rec./F1 (w): support-weighted averages; F1 (macro): unweighted class average; ROC-AUC: one-vs-rest, macro-averaged, from predicted probabilities. "
                  r"Train: time to fit the selected classifier. Best value per column in bold.}",
                  rf"\label{{tbl_main_{ds}_{sc}}}", r"\end{table}", ""]
        write(od / f"main_{ds}_{sc}.tex", "\n".join(lines))


def best_table(B, od, families, fname, caption, label):
    d = B[B.family.isin(families)].copy()
    d["fo"] = d.family.map(FAM_ORDER)
    d = d.sort_values(["dataset", "scheme", "fo", "representation"])
    lines = [r"\begin{table}[!ht]", r"\centering", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lllcccc}", r"\toprule",
             r"Dataset & Split & Representation (best classifier) & Acc. & F1 (macro) & ROC-AUC & Runs \\", r"\midrule"]
    for (ds, sc), g in d.groupby(["dataset", "scheme"], sort=False):
        first = True
        for _, r in g.iterrows():
            lines.append(f"{NICE.get(ds, ds) if first else ''} & {scheme_name(sc) if first else ''} & {esc(r.representation)} ({r.classifier}) & "
                         f"{ms(r, 'acc')} & {ms(r, 'f1_m')} & {ms(r, 'auc')} & {int(r.n_runs)} \\\\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""]
    write(od / fname, "\n".join(lines))


def cluster_stats_table(cfg, od):
    rows = []
    for ds in cfg["datasets"]:
        for f in sorted(p(cfg, "data", "splits", ds).glob("cluster_id*/cluster_stats.json")):
            s = read_json(f)
            x = f.parent / "cross_split_identity_seed0.json"
            xs = read_json(x) if x.exists() else None
            rows.append(f"{NICE.get(ds, ds)} & {100 * s['min_seq_id']:.0f}\\% & {s['n_clusters']} & {s['largest_cluster']} "
                        f"({100 * s['largest_cluster_fraction']:.1f}\\%) & {s['n_singletons']} & "
                        f"{'yes' if s['feasible_all_seeds'] else 'no'} & "
                        f"{(str(round(xs['max_identity_quantiles']['0.5'], 3)) + ' / ' + str(round(xs['max_identity_quantiles']['0.99'], 3))) if xs else 'N/A'} \\\\")
    body = [r"\begin{table}[!ht]", r"\centering", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccc}", r"\toprule",
            r"Dataset & Identity & Clusters & Largest cluster & Singletons & Usable split & Test-to-train max identity (median / 99th pct.) \\",
            r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}}",
            r"\caption{MMseqs2 \textit{easy-cluster} statistics (coverage 0.8, \texttt{--cov-mode 0}). A split is usable when whole clusters can be assigned to train/validation/test within 10 percentage points of 60/10/30 and most test classes occur in training. Identity: highest MMseqs2 alignment identity of each test sequence to any training sequence (seed 0).}",
            r"\label{tbl_cluster_stats}", r"\end{table}", ""]
    write(od / "cluster_stats.tex", "\n".join(body))


def paired_table(P, od):
    d = P[(P.metric == "f1_m") & (P.other_family.isin(["foundation", "simple"]))].copy()
    lines = [r"\begin{table}[!ht]", r"\centering", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lllcccc}", r"\toprule",
             r"Dataset & Split & Ours vs. & $\Delta$ F1 (macro) [95\% CI] & Cohen's $d_z$ & $p$ (paired $t$) & $p$ (Holm) \\", r"\midrule"]
    for _, r in d.sort_values(["dataset", "scheme", "other"]).iterrows():
        lines.append(f"{NICE.get(r.dataset, r.dataset)} & {scheme_name(r.scheme)} & {esc(r.other)} & "
                     f"{r.mean_diff:+.3f} [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}] & {r.cohens_dz:.2f} & {r.p_ttest:.3g} & {r.p_holm:.3g} \\\\")
    first = d.iloc[0] if len(d) else None
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{Paired comparison of our best compression-based configuration (selected on validation) against each baseline's best configuration, on identical splits"
              + (f" ({int(first.n_paired_runs)} runs)" if first is not None else "") +
              r". $\Delta>0$ favors the compression-based representation. Holm correction within each dataset and split scheme.}",
              r"\label{tbl_paired}", r"\end{table}", ""]
    write(od / "paired_tests.tex", "\n".join(lines))


def sensitivity_table(G, B, od):
    for ds, g in G.groupby("dataset"):
        b = B[(B.dataset == ds) & (B.scheme == "random") & (B.family == "ours")]
        if b.empty:
            continue
        comp = b.loc[b.val_f1_m_mean.idxmax()].representation.split("+")[0]
        g = g[g.compressor == comp]
        piv = g.pivot(index="sigma2_mult", columns="k", values="val_acc_mean")
        best = piv.max().max()
        lines = [r"\begin{table}[!ht]", r"\centering", r"\begin{tabular}{l" + "c" * piv.shape[1] + "}", r"\toprule",
                 r"$\sigma^2$ multiplier $\backslash$ $k$ & " + " & ".join(str(c) for c in piv.columns) + r" \\", r"\midrule"]
        for m, row in piv.iterrows():
            cells = [(r"\textbf{%.3f}" % v) if np.isclose(v, best) else f"{v:.3f}" for v in row.values]
            lines.append(f"{m:g} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}",
                  rf"\caption{{Validation accuracy on {NICE.get(ds, ds)} ({comp}) over the kernel bandwidth $\sigma^2$ (multiple of the median squared distance between training profiles) and the number of kernel-PCA components $k$ (mean over random-split runs; selection classifier: logistic regression). The test partition is not used for this selection.}}",
                  rf"\label{{tbl_sensitivity_{ds}}}", r"\end{table}", ""]
        write(od / f"sensitivity_{ds}.tex", "\n".join(lines))


def controls_table(cfg, S, od):
    lines = [r"\begin{table}[!ht]", r"\centering", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Dataset & Observed F1 (macro) & Shuffled labels & Shuffled residues & Random features & Permutation null mean (95th pct.) & $p$ \\", r"\midrule"]
    for ds in cfg["datasets"]:
        f = p(cfg, "results", "perm_null", f"{ds}.json")
        if not f.exists():
            continue
        n = read_json(f)
        comp = n["representation"].split("+")[0]
        s = S[(S.dataset == ds) & (S.scheme == "random") & (S.classifier == n["classifier"])]

        def get(rep):
            r = s[s.representation == rep]
            return ms(r.iloc[0], "f1_m") if len(r) else "N/A"
        lines.append(f"{NICE.get(ds, ds)} & {get(n['representation'])} & {get(n['representation'] + ' [shuffled labels]')} & "
                     f"{get(comp + '+KPCA [shuffled residues]')} & {get('Random Gaussian features')} & "
                     f"{n['null_f1_m_mean']:.3f} ({n['null_f1_m_p95']:.3f}) & {n['p_f1_m']:.3g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{Control experiments (macro-F1, test partition, random split) for the best compression-based configuration of each dataset. Shuffled residues: each sequence is randomly permuted (composition preserved) before computing NCD. The permutation null is obtained by refitting the selected classifier on permuted training labels; $p = (1 + \#\{\mathrm{null} \geq \mathrm{observed}\}) / (1 + N)$.}",
              r"\label{tbl_controls}", r"\end{table}", ""]
    write(od / "controls.tex", "\n".join(lines))


def timing_table(T, od):
    if T.empty:
        return
    lines = [r"\begin{table}[!ht]", r"\centering", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lllccc}", r"\toprule",
             r"Dataset & Method & Stage & Total time & Per sequence (s) & Peak memory (MB) \\", r"\midrule"]
    for _, r in T.iterrows():
        tot = r.get("cpu_s") if not pd.isna(r.get("cpu_s", np.nan)) else r.get("wall_s", np.nan)
        tot = "N/A" if pd.isna(tot) else (f"{tot / 3600:.2f} CPU-h" if not pd.isna(r.get("cpu_s", np.nan)) else f"{tot:.0f} s")
        per = r.get("per_seq_s", np.nan)
        lines.append(f"{NICE.get(r.dataset, r.dataset)} & {esc(r.method)} & {esc(r.stage)} & {tot} & "
                     f"{'N/A' if pd.isna(per) else f'{per:.3f}'} & {'N/A' if pd.isna(r.get('peak_rss_mb', np.nan)) else f'{r.peak_rss_mb:.0f}'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{Computational cost. All-pairs NCD: total CPU time of all compression calls. Foundation models: embedding time on GPU and on CPU (CPU timing on a random subset, same node type as the NCD jobs). New-sequence embedding: NCD of one query against the full training reference set plus kernel-PCA projection, on a single CPU core.}",
              r"\label{tbl_timing}", r"\end{table}", ""]
    write(od / "timing.tex", "\n".join(lines))


def dataset_table(cfg, od):
    lines = [r"\begin{table}[!ht]", r"\centering", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"Dataset & Seq. & Classes & Length min / median / max & $<$50 res. & Duplicate groups (conflicting labels) & Non-standard symbols & Max distinct pair (bytes) & Max self pair (bytes) \\",
             r"\midrule"]
    for ds in cfg["datasets"]:
        f = p(cfg, "data", "processed", ds, "report.json")
        if not f.exists():
            continue
        r = read_json(f)
        L, w = r["length"], r["window"]
        lines.append(f"{NICE.get(ds, ds)} & {r['n_sequences']} & {r['n_classes']} & {L['min']} / {L['median']:.0f} / {L['max']} & "
                     f"{r['n_shorter_than_50']} & {r['n_exact_duplicate_groups']} ({r['n_duplicate_groups_with_conflicting_labels']}) & "
                     f"{100 * r['fraction_non_standard']:.2f}\\% & {w['max_concat_distinct_pair_bytes']:,} & {w['max_concat_self_pair_bytes']:,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{Dataset statistics. Pair sizes assume one byte per residue (UTF-8). Gzip's DEFLATE window is 32{,}768 bytes.}",
              r"\label{tbl_data_statistics_full}", r"\end{table}", ""]
    write(od / "dataset_stats.tex", "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    sd = p(cfg, "results", "summary")
    od = ensure_dir(p(cfg, "results", "tables"))
    S = pd.read_csv(sd / "summary.csv")
    B = pd.read_csv(sd / "best_by_validation.csv")
    main_tables(S, od)
    best_table(B, od, ["ours", "ablation", "transductive"], "ablations.tex",
               "Ablation of the representation pipeline (best classifier per row selected on validation). dist: NCD distance profile to the training sequences used directly; PCA: linear PCA of the profile; KPCA: Gaussian kernel on profiles + kernel PCA (proposed); KPCA-direct: Gaussian applied elementwise to NCD distances; kNN-NCD: nearest neighbours on NCD; KPCA-elementwise500: elementwise kernel with fixed bandwidth and 500 components, fitted on training sequences; KPCA-transductive: the same but fitted on all sequences before splitting.",
               "tbl_ablation")
    best_table(B[B.scheme != "random"], od, ["ours", "foundation", "simple"], "cluster_results.tex",
               "Results under cluster-disjoint splits for all representations (best classifier per row selected on validation).",
               "tbl_cluster_split")
    cluster_stats_table(cfg, od)
    P = pd.read_csv(sd / "paired_tests.csv") if (sd / "paired_tests.csv").stat().st_size > 5 else pd.DataFrame()
    if not P.empty:
        paired_table(P, od)
    G = pd.read_csv(sd / "kernel_grid_validation.csv") if (sd / "kernel_grid_validation.csv").stat().st_size > 5 else pd.DataFrame()
    if not G.empty:
        sensitivity_table(G, B, od)
    controls_table(cfg, S, od)
    T = pd.read_csv(sd / "timing.csv") if (sd / "timing.csv").stat().st_size > 5 else pd.DataFrame()
    timing_table(T, od)
    dataset_table(cfg, od)
    print("tables in", od)


if __name__ == "__main__":
    main()
