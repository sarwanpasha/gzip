# gzip: compression-based representations for molecular sequences

Alignment-free embeddings for DNA and protein sequences built from the
Normalized Compression Distance (NCD) with standard compressors (Gzip, Bz2, LZMA),
a Gaussian kernel, and kernel PCA. The embeddings are fed to ordinary classifiers.
No neural network is trained to produce the representation, and no GPU is needed for it.

This repository contains the full pipeline: data preparation, splitting, NCD computation,
embedding, baselines, ablations, controls, statistics, timing, and table generation.

## Method

1. **NCD.** For sequences `x` and `y`, with `C(.)` the compressed length in bytes of the UTF-8 string:
   `NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))`.
   Both orders are computed and averaged: `D'(x, y) = (NCD(x, y) + NCD(y, x)) / 2`.
2. **Distance profile.** Each sequence `s` is represented by its distances to the training
   (reference) sequences: `phi(s) = [D'(s, r) for r in train]`.
3. **Kernel.** `K(s, t) = exp(-||phi(s) - phi(t)||^2 / sigma^2)`. This is a Gaussian (RBF) kernel
   between real vectors, so it is positive semi-definite. `sigma^2` is a multiple of the median
   squared distance between training profiles.
4. **Kernel PCA.** Fitted on the training kernel only. Validation and test sequences are projected
   out of sample using their kernel values against the training sequences.
5. **Classifier.** SVM, Naive Bayes, MLP, k-NN, Random Forest, Logistic Regression, Decision Tree.

Nothing that depends on the split (reference set, bandwidth, centering, kernel PCA,
hyperparameters) uses validation or test sequences. A new sequence is embedded by computing its
NCD to each training sequence and projecting.

## Evaluation protocol

- **Splits.** 60/10/30 train/validation/test. Two schemes:
  stratified random splits, and cluster-disjoint splits where whole MMseqs2 clusters
  (`easy-cluster`, coverage 0.8) go to one partition. Several identity thresholds are run;
  thresholds at which a usable split cannot be formed are reported and skipped.
- **Runs.** 10 runs per scheme. Each run uses a different split and model seed.
  All representations share the same splits, so comparisons are paired.
- **Model selection.** Kernel bandwidth, number of components, and classifier hyperparameters are
  chosen by validation macro-F1. The test partition is used only for the final numbers.
- **Metrics.** Accuracy; support-weighted precision, recall and F1; macro precision, recall and F1;
  ROC-AUC (one-vs-rest, macro-averaged, from predicted probabilities).
- **Statistics.** Mean, standard deviation and 95% confidence interval per configuration.
  Paired t-test and Wilcoxon signed-rank test between the best compression-based configuration and
  each baseline, with mean difference, confidence interval, Cohen's d_z, and Holm correction.

## What is compared

| Group | Representations |
|---|---|
| Compression-based | Gzip, Bz2, LZMA + kernel PCA |
| Foundation models (frozen, mean-pooled) | ESM-2 650M and ProtT5-XL-U50 for proteins; Nucleotide Transformer v2-500M for DNA |
| Simple baselines | sequence length; residue composition (and GC content); 3-mer spectrum; compressed length |
| Ablations | k-NN directly on NCD; distance profile without PCA; linear PCA; kernel applied elementwise to distances |
| Controls | shuffled training labels; residue-shuffled sequences (composition preserved); random Gaussian features; label-permutation null distribution |

Cost is also measured: CPU time of all compression calls, peak memory, time to embed one new
sequence on one CPU core, and foundation-model embedding time on GPU and on CPU.

## Installation

```bash
git clone https://github.com/sarwanpasha/gzip.git
cd gzip
bash scripts/setup_cpu.sh      # conda env in ./env with MMseqs2, scikit-learn, CPU PyTorch
bash scripts/setup_gpu.sh      # on the GPU machine instead: env with CUDA PyTorch for embeddings
```

Both scripts expect `conda` (set `CONDA=/path/to/conda` if it is not in `~/miniconda3` or
`~/anaconda3`). They create `./env` and keep model weights in `./tools`.
`setup_gpu.sh` pins PyTorch 1.12.1 (CUDA 11.3) so that older GPUs work; edit it for newer cards.
Without conda: `pip install -r requirements.txt` and install
[MMseqs2](https://github.com/soedinglab/MMseqs2) separately.

## Data

Put the input files in `data/raw/` and describe them in `config.json`. Supported formats:

| `format` | File layout |
|---|---|
| `tsv_seq_class` | tab-separated with a header; column 1 = sequence, column 2 = class |
| `fasta_label_last_field` | FASTA; the label is the last field of the header (`>id|...|label`, separator set by `header_sep`) |
| `csv` | columns `sequence`, `label`, and optionally `id` |

Each dataset entry also needs `"alphabet": "dna"` or `"protein"`.

The Human DNA dataset is public and is downloaded by `setup_cpu.sh`.
Sequences obtained from GISAID cannot be redistributed and are not included here;
download them from [GISAID](https://gisaid.org) under its terms of use.

## Running

Every stage runs in its own `tmux` session named `gzip_*`, and writes a log to `logs/`.
Reattach with `tmux attach -t gzip_<name>`.

```bash
bash scripts/run.sh prepare   # dataset report, splits, MMseqs2 clustering, C(x)
bash scripts/run.sh ncd       # all-pairs NCD for each dataset and compressor
bash scripts/run.sh fmgpu     # GPU machine: frozen foundation-model embeddings
bash scripts/run.sh fmcpu     # CPU timing of the foundation models
bash scripts/run.sh exp       # all representations x classifiers x runs
bash scripts/run.sh post      # statistics, permutation null, timing, LaTeX tables
bash scripts/run.sh status    # sessions and last log line of each
```

Parallelism is set with `NCD_PROCS`, `EXP_WORKERS` and `EXP_THREADS`
(for example `NCD_PROCS=8 bash scripts/run.sh ncd`).
If embeddings are computed on another machine, copy `data/emb/` back before `exp`.
All steps skip work that is already done, so they can be restarted.

## Outputs

```
data/processed/<dataset>/report.json     sizes, length statistics, duplicates, non-standard symbols,
                                         concatenation sizes vs. the 32 KiB Gzip window
data/splits/<dataset>/                   split indices, cluster assignments, cluster statistics,
                                         MMseqs2 version and exact commands
data/ncd/<dataset>/<compressor>/         C(x), symmetrized NCD matrix, compression timing
data/emb/<dataset>/                      foundation-model embeddings and timing
results/<dataset>/<scheme>/seed*.csv     one row per representation and classifier
results/summary/                         summary.csv, best_by_validation.csv, paired_tests.csv,
                                         kernel_grid_validation.csv, timing.csv, cluster_stats.csv
results/tables/                          LaTeX tables generated from the files above
```

## Code layout

| File | Purpose |
|---|---|
| `src/data.py` | load raw data, write common format and dataset report |
| `src/splits.py` | random and cluster-disjoint splits with statistics |
| `src/ncd.py` | compressed lengths and all-pairs NCD |
| `src/features.py` | distance profiles, kernel PCA, ablations, simple baselines |
| `src/classify.py` | classifiers, hyperparameter grids, metrics |
| `src/fm_embed.py` | frozen foundation-model embeddings |
| `src/run_experiment.py` | one (dataset, scheme, run): build all representations, evaluate |
| `src/aggregate.py` | summaries and paired statistical tests |
| `src/perm_null.py` | label-permutation null distribution |
| `src/timing_bench.py` | cost of embedding new sequences |
| `src/make_tables.py` | LaTeX tables |

## Compressor settings

| Compressor | Setting | Window / block |
|---|---|---|
| Gzip | `gzip.compress`, level 9 | 32 KiB sliding window |
| Bz2 | `bz2.compress`, level 9 | 900 kB blocks |
| LZMA | `lzma.compress`, xz preset 6 | 8 MiB dictionary |

Gzip cannot see redundancy between two sequences once their concatenation is longer than its
window. `report.json` lists how many pairs exceed it, including each sequence paired with itself.

## License

MIT
