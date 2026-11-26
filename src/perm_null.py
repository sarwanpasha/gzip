"""Empirical null distribution by label permutation (random split, seed 0).

For the best compression-based configuration of a dataset (chosen on validation), the classifier is
refit with the selected hyperparameters on N random permutations of the training labels, and the
test metrics are recorded. The empirical null mean / 95th percentile replaces the 1/C heuristic,
and p = (1 + #{null >= observed}) / (1 + N).
"""
import argparse
import ast

import numpy as np
import pandas as pd

import classify as C
import features as F
from common import ensure_dir, load_config, log, p, read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ds", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    B = pd.read_csv(p(cfg, "results", "summary", "best_by_validation.csv"))
    b = B[(B.dataset == a.ds) & (B.scheme == "random") & (B.family == "ours")]
    best = b.loc[b["val_f1_m_mean"].idxmax()]
    comp = best.representation.split("+")[0]
    R = pd.read_csv(p(cfg, "results", a.ds, "random", "seed0.csv"))
    row = R[(R.representation == best.representation) & (R.classifier == best.classifier)].iloc[0]
    params = ast.literal_eval(row.params)
    meta = read_json(p(cfg, "results", a.ds, "random", "seed0_meta.json"))["compressors"][comp]
    df = pd.read_csv(p(cfg, "data", "processed", a.ds, "sequences.csv"))
    y = df["label"].astype(str).to_numpy()
    sp = {k: np.array(v) for k, v in read_json(p(cfg, "data", "splits", a.ds, "random", "seed0.json")).items() if k != "stats"}
    D = np.load(p(cfg, "data", "ncd", a.ds, comp, "D.npy"), mmap_mode="r")
    rk = F.RowKernelPCA(meta["sigma2_mult"], meta["k"]).fit(F.profiles(D, sp["train"], sp["train"]))
    Zte = rk.transform(F.profiles(D, sp["test"], sp["train"]))
    ytr, yte = y[sp["train"]], y[sp["test"]]

    def run(labels, seed):
        m = C.make(best.classifier, params, seed).fit(rk.Z_tr, labels)
        pr = m.predict(Zte)
        return C.metrics(yte, pr, m.predict_proba(Zte), m.classes_)

    obs = run(ytr, 0)
    rng = np.random.default_rng(123)
    null = []
    for i in range(cfg["permutation_null_n"]):
        null.append(run(ytr[rng.permutation(len(ytr))], i))
        if i % 20 == 0:
            log(a.ds, "perm", i)
    N = pd.DataFrame(null)
    out = {"dataset": a.ds, "representation": best.representation, "classifier": best.classifier,
           "params": row.params, "n_permutations": len(N), "observed": obs}
    for m in ("acc", "f1_m", "auc"):
        v = N[m].to_numpy()
        out[f"null_{m}_mean"] = float(np.nanmean(v))
        out[f"null_{m}_p95"] = float(np.nanpercentile(v, 95))
        out[f"null_{m}_max"] = float(np.nanmax(v))
        out[f"p_{m}"] = float((1 + np.sum(v >= obs[m])) / (1 + len(v)))
    out["one_over_C"] = 1.0 / len(np.unique(ytr))
    od = ensure_dir(p(cfg, "results", "perm_null"))
    N.to_csv(od / f"{a.ds}_null.csv", index=False)
    write_json(out, od / f"{a.ds}.json")
    log("done", out)


if __name__ == "__main__":
    main()
