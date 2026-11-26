"""Inductive feature construction from the symmetrized NCD matrix D'.

All split-dependent quantities are fitted on TRAINING sequences only:
  * reference set R      = training sequences
  * distance profile     phi(s) = [D'(s, r) for r in R]           (fixed-length vector per sequence)
  * kernel ("row")       K(s, t) = exp(-||phi(s) - phi(t)||^2 / sigma^2)
                         a Gaussian (RBF) kernel on Euclidean vectors, hence positive semi-definite
  * bandwidth            sigma^2 = multiplier * median squared distance between training profiles
  * kernel PCA           fitted on K(train, train) (with centering learned on training data);
                         validation/test sequences are projected out of sample with K(new, train)

Ablations (same reference set and bandwidth rule):
  * "dist"         phi(s) used directly as features
  * "pca"          linear PCA of phi(s) fitted on training profiles
  * "kpca_direct"  kernel applied elementwise to the distances, K = exp(-D'^2 / sigma^2),
                   which is not guaranteed to be PSD (we report its smallest eigenvalue)
  * "nn"           nearest-neighbour classification directly on D'(test, train)
"""
import time

import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics.pairwise import euclidean_distances


def _median_offdiag(M):
    n = M.shape[0]
    iu = np.triu_indices(n, 1)
    return float(np.median(M[iu]))


def profiles(D, idx_rows, idx_ref):
    return D[np.ix_(idx_rows, idx_ref)].astype(np.float64)


def min_eig(K, center=True):
    K = K.copy()
    if center:
        n = K.shape[0]
        one = np.full((n, n), 1.0 / n)
        K = K - one @ K - K @ one + one @ K @ one
    return float(eigh(K, eigvals_only=True, subset_by_index=[0, 0])[0])


class RowKernelPCA:
    """Kernel PCA on Gaussian kernel of distance profiles; fit on train, project new samples."""

    def __init__(self, sigma2_mult=1.0, n_components=128):
        self.m, self.k = sigma2_mult, n_components

    def fit(self, phi_tr):
        t = time.perf_counter()
        sq = euclidean_distances(phi_tr, squared=True)
        self.sigma2 = self.m * _median_offdiag(sq)
        K = np.exp(-sq / self.sigma2)
        k = min(self.k, K.shape[0] - 1)
        self.kpca = KernelPCA(n_components=k, kernel="precomputed", eigen_solver="auto", random_state=0)
        self.Z_tr = self.kpca.fit_transform(K)
        self.phi_tr = phi_tr
        self.K_tr = K
        self.fit_s = time.perf_counter() - t
        return self

    def transform(self, phi_new):
        sq = euclidean_distances(phi_new, self.phi_tr, squared=True)
        return self.kpca.transform(np.exp(-sq / self.sigma2))


class DirectKernelPCA:
    """Kernel applied elementwise to symmetrized NCD distances (kept as an ablation).
    Not guaranteed PSD; kernel PCA keeps only the leading (positive) components."""

    def __init__(self, sigma2_mult=1.0, n_components=128):
        self.m, self.k = sigma2_mult, n_components

    def fit(self, D_tr):
        sq = D_tr.astype(np.float64) ** 2
        self.sigma2 = self.m * _median_offdiag(sq)
        K = np.exp(-sq / self.sigma2)
        self.min_eig_centered = min_eig(K, center=True)
        self.min_eig_raw = min_eig(K, center=False)
        k = min(self.k, K.shape[0] - 1)
        self.kpca = KernelPCA(n_components=k, kernel="precomputed", random_state=0)
        self.Z_tr = self.kpca.fit_transform(K)
        return self

    def transform(self, D_new_tr):
        return self.kpca.transform(np.exp(-(D_new_tr.astype(np.float64) ** 2) / self.sigma2))


def linear_pca(phi_tr, phi_others, k):
    k = min(k, phi_tr.shape[0] - 1, phi_tr.shape[1])
    pca = PCA(n_components=k, random_state=0).fit(phi_tr)
    return pca.transform(phi_tr), [pca.transform(x) for x in phi_others]


# ----------------------------------------------------------------------------------------------
# Simple alignment-free baselines (no compression across sequences)
# ----------------------------------------------------------------------------------------------
DNA = "ACGT"
AA = "ACDEFGHIKLMNPQRSTVWY"


def length_features(seqs):
    L = np.array([len(s) for s in seqs], dtype=float)
    return np.c_[L, np.log1p(L)]


def composition_features(seqs, alphabet):
    alph = DNA if alphabet == "dna" else AA
    X = np.zeros((len(seqs), len(alph) + 1 + (1 if alphabet == "dna" else 0)))
    for i, s in enumerate(seqs):
        n = max(1, len(s))
        for j, c in enumerate(alph):
            X[i, j] = s.count(c) / n
        X[i, len(alph)] = 1 - X[i, :len(alph)].sum()  # fraction of non-standard symbols
        if alphabet == "dna":
            X[i, -1] = (s.count("G") + s.count("C")) / n  # GC content
    return X


def kmer_features(seqs, alphabet, k=3):
    alph = DNA if alphabet == "dna" else AA
    from itertools import product
    vocab = {"".join(t): i for i, t in enumerate(product(alph, repeat=k))}
    X = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j in range(len(s) - k + 1):
            v = vocab.get(s[j:j + k])
            if v is not None:
                X[i, v] += 1
        tot = X[i].sum()
        if tot > 0:
            X[i] /= tot
    return X


def compressed_length_features(cx_by_comp, seqs):
    L = np.array([max(1, len(s)) for s in seqs], dtype=float)
    cols = []
    for c in sorted(cx_by_comp):
        cx = cx_by_comp[c].astype(float)
        cols += [cx, cx / L]
    return np.column_stack(cols + [L])
