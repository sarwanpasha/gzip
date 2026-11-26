#!/bin/bash
# CPU machine: conda env with MMseqs2 + scientific stack + CPU PyTorch (for CPU timing).
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
CONDA=${CONDA:-$HOME/miniconda3/bin/conda}
[ -d $W/env ] || $CONDA create -y -p $W/env -c conda-forge -c bioconda python=3.11 mmseqs2 numpy pandas scipy scikit-learn tmux
$W/env/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
$W/env/bin/pip install --no-cache-dir "transformers==4.44.2" sentencepiece protobuf
export HF_HOME=$W/tools/hf
mkdir -p $W/data/raw
[ -s $W/data/raw/human_dna.csv ] || $W/env/bin/python - <<PY
import pandas as pd
d = pd.read_csv("https://raw.githubusercontent.com/krishnaik06/DNA-Sequencing-Classifier/master/human_data.txt", sep="\\t")
names = ["G protein coupled receptors", "Tyrosine kinase", "Tyrosine phosphatase", "Synthetase", "Synthase", "Ion channel", "Transcription factor"]
pd.DataFrame({"id": ["seq%d" % i for i in range(len(d))], "sequence": d.iloc[:, 0], "label": [names[int(c)] for c in d.iloc[:, 1]]}).to_csv("$W/data/raw/human_dna.csv", index=False)
PY
$W/env/bin/python -c "import sklearn, torch, transformers; print('sklearn', sklearn.__version__, 'torch', torch.__version__)"
$W/env/bin/mmseqs version
cd $W/src && $W/env/bin/python -c "
from fm_embed import MODELS, load
for n in MODELS: load(n, 'cpu'); print('downloaded', n)"
echo SETUP_CPU_DONE
