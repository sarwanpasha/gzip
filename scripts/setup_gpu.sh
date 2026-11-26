#!/bin/bash
# GPU machine with an older GPU (e.g. Tesla K80, compute capability 3.7, CUDA 11.3 driver):
# PyTorch 1.12.1+cu113 is the last release line with K80 (sm_37) kernels.
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
CONDA=${CONDA:-$HOME/anaconda3/bin/conda}
[ -d $W/env ] || $CONDA create -y -p $W/env python=3.9 numpy pandas
$W/env/bin/pip install --no-cache-dir torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
$W/env/bin/pip install --no-cache-dir "transformers==4.33.3" sentencepiece "protobuf<4"
export HF_HOME=$W/tools/hf
$W/env/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print((torch.ones(2,2).cuda()@torch.ones(2,2).cuda()).sum().item())"
cd $W/src && $W/env/bin/python -c "
from fm_embed import MODELS, load
for n in MODELS: load(n, 'cpu'); print('downloaded', n)"
echo SETUP_GPU_DONE
