"""Downstream classifiers with validation-based hyperparameter selection, and metric definitions.

Metrics (all computed on the held-out test partition):
  accuracy
  precision/recall/F1 'weighted' : per-class scores averaged with class-support weights
                                   (weighted recall equals accuracy by definition)
  precision/recall/F1 'macro'    : unweighted mean over classes present in y_true or y_pred
  ROC-AUC                        : one-vs-rest, macro-averaged over classes that occur in the test
                                   set and in training, computed from predicted class probabilities
Model selection: for every classifier, the hyperparameter setting with the best validation
macro-F1 is chosen; that model (trained on the training partition only) is evaluated on test.
"""
import itertools
import os
import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

GRIDS = {
    "SVM": {"C": [0.1, 1.0, 10.0]},
    "NB": {"var_smoothing": [1e-9, 1e-6, 1e-3]},
    "MLP": {"hidden_layer_sizes": [(128,), (256, 128)], "alpha": [1e-4, 1e-2]},
    "KNN": {"n_neighbors": [1, 3, 5, 11]},
    "RF": {"max_depth": [None, 30], "max_features": ["sqrt", 0.2]},
    "LR": {"C": [0.1, 1.0, 10.0]},
    "DT": {"max_depth": [None, 20], "min_samples_leaf": [1, 5]},
}


def make(name, params, seed):
    if name == "SVM":
        m = SVC(kernel="rbf", probability=True, random_state=seed, **params)
    elif name == "NB":
        return GaussianNB(**params)
    elif name == "MLP":
        m = MLPClassifier(max_iter=500, early_stopping=True, random_state=seed, **params)
    elif name == "KNN":
        m = KNeighborsClassifier(**params)
    elif name == "RF":
        return RandomForestClassifier(n_estimators=300, n_jobs=int(os.environ.get("GZ_NJOBS", "4")), random_state=seed, **params)
    elif name == "LR":
        m = LogisticRegression(max_iter=3000, random_state=seed, **params)
    elif name == "DT":
        return DecisionTreeClassifier(random_state=seed, **params)
    else:
        raise ValueError(name)
    return make_pipeline(StandardScaler(), m)


def auc_ovr_macro(y_true, proba, classes):
    y_true = np.asarray(y_true)
    scores = []
    for j, c in enumerate(classes):
        pos = y_true == c
        if pos.any() and (~pos).any():
            scores.append(roc_auc_score(pos.astype(int), proba[:, j]))
    return float(np.mean(scores)) if scores else float("nan")


def metrics(y_true, y_pred, proba=None, classes=None):
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "prec_w": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "rec_w": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_w": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "prec_m": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec_m": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_m": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    out["auc"] = auc_ovr_macro(y_true, proba, classes) if proba is not None else float("nan")
    return out


def fit_select(name, Xtr, ytr, Xva, yva, seed):
    grid = GRIDS[name]
    keys = list(grid)
    best = None
    for vals in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, vals))
        if name == "KNN" and params["n_neighbors"] > len(ytr):
            continue
        m = make(name, params, seed)
        t = time.perf_counter()
        m.fit(Xtr, ytr)
        tt = time.perf_counter() - t
        score = f1_score(yva, m.predict(Xva), average="macro", zero_division=0) if len(yva) else 0.0
        if best is None or score > best[0]:
            best = (score, params, m, tt)
    return best


def evaluate(name, Xtr, ytr, Xva, yva, Xte, yte, seed):
    val_f1, params, model, train_s = fit_select(name, Xtr, ytr, Xva, yva, seed)
    t = time.perf_counter()
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None
    pred_s = time.perf_counter() - t
    res = metrics(yte, pred, proba, model.classes_)
    val_acc = accuracy_score(yva, model.predict(Xva)) if len(yva) else float("nan")
    res.update({"classifier": name, "params": str(params), "val_f1_m": val_f1, "val_acc": val_acc,
                "train_s": train_s, "predict_s_per_seq": pred_s / max(1, len(yte))})
    return res, model


def nn_on_distances(D_te_tr, D_va_tr, ytr, yva, yte):
    """k-NN directly on NCD distances (k chosen on validation)."""
    ytr = np.asarray(ytr)
    classes = np.unique(ytr)
    best = None
    for k in (1, 3, 5, 11):
        if k > len(ytr):
            continue
        idx = np.argsort(D_va_tr, axis=1)[:, :k]
        pred = np.array([_vote(ytr[r]) for r in idx])
        s = f1_score(yva, pred, average="macro", zero_division=0) if len(yva) else 0
        if best is None or s > best[0]:
            best = (s, k)
    k = best[1]
    t = time.perf_counter()
    idx = np.argsort(D_te_tr, axis=1)[:, :k]
    pred = np.array([_vote(ytr[r]) for r in idx])
    proba = np.stack([(ytr[idx] == c).mean(axis=1) for c in classes], axis=1)
    pred_s = time.perf_counter() - t
    res = metrics(yte, pred, proba, classes)
    res.update({"classifier": "kNN-NCD", "params": f"{{'k': {k}}}", "val_f1_m": best[0], "val_acc": float("nan"),
                "train_s": 0.0, "predict_s_per_seq": pred_s / max(1, len(yte))})
    return res


def _vote(labels):
    v, c = np.unique(labels, return_counts=True)
    return v[np.argmax(c)]
