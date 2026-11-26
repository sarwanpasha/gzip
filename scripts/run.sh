#!/bin/bash
# tmux-based runner (no SLURM). Every job runs in its own tmux session named gzip_*;
# reattach with:  tmux attach -t gzip_<name>      list with:  tmux ls | grep gzip_
# Logs: logs/<session>.log
#
#  CPU machine:  bash scripts/run.sh prepare | ncd | fmcpu | exp | post | status
#  GPU machine:  bash scripts/run.sh fmgpu | status
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p $W/logs
NCD_PROCS=${NCD_PROCS:-5}     # processes per NCD session (9 sessions -> ~45 cores)
EXP_WORKERS=${EXP_WORKERS:-6} # parallel experiment sessions
EXP_THREADS=${EXP_THREADS:-4} # threads per experiment session (BLAS + random forest)
ENVV="export HF_HOME=$W/tools/hf PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PATH=$W/env/bin:\$PATH; cd $W/src"
PY=$W/env/bin/python
DATASETS=$($PY -c "import json;print(' '.join(json.load(open('$W/config.json'))['datasets']))")

launch() {  # launch <session> <commands>: writes logs/<session>.sh and runs it in tmux
  local s=$1; shift
  if tmux has-session -t "$s" 2>/dev/null; then echo "session $s already exists (attach: tmux attach -t $s)"; return; fi
  {
    echo "#!/bin/bash"
    echo "$ENVV"
    echo "set -o pipefail"
    echo "( $* ) 2>&1 | tee -a $W/logs/$s.log"
    echo 'echo "[finished, exit $?]" | tee -a '"$W/logs/$s.log"
    echo 'exec bash'
  } > $W/logs/$s.sh
  tmux new-session -d -s "$s" "bash $W/logs/$s.sh"
  echo "started tmux session $s"
}

case "${1:-}" in
prepare)
  CX="for ds in $DATASETS; do for c in gzip bz2 lzma; do for v in original shuffled; do python ncd.py cx --ds \$ds --comp \$c --variant \$v; done; done; done"
  launch gzip_prepare "python data.py && python splits.py && $CX && echo PREPARE_DONE" ;;
ncd)
  for ds in $DATASETS; do
    N=$(cd $W/src && $PY ncd.py nchunks --ds $ds)
    for c in gzip bz2 lzma; do
      launch gzip_ncd_${ds}_${c} "for v in original shuffled; do for ch in \$(seq 0 $((N-1))); do python ncd.py rows --ds $ds --comp $c --variant \$v --chunk \$ch --procs $NCD_PROCS || exit 1; done; python ncd.py merge --ds $ds --comp $c --variant \$v || exit 1; done; echo NCD_DONE"
    done
  done ;;
fmgpu)   # GPU machine (uses GPUs 0 and 1)
  P="for ds in $DATASETS; do a=\$(python -c \"import json;print(json.load(open('$W/config.json'))['datasets']['\$ds']['alphabet'])\"); [ \$a = protein ] || continue; for m in esm2_650m prott5_xl_u50; do python fm_embed.py --ds \$ds --model \$m --device cuda || exit 1; done; done"
  D="for ds in $DATASETS; do a=\$(python -c \"import json;print(json.load(open('$W/config.json'))['datasets']['\$ds']['alphabet'])\"); [ \$a = dna ] || continue; python fm_embed.py --ds \$ds --model nt_v2_500m --device cuda || exit 1; done"
  launch gzip_fm_gpu0 "export CUDA_VISIBLE_DEVICES=0; python data.py && $P && echo FM_GPU0_DONE"
  launch gzip_fm_gpu1 "export CUDA_VISIBLE_DEVICES=1; sleep 120; $D && echo FM_GPU1_DONE" ;;
fmcpu)   # CPU timing of the foundation models on the same CPU as the NCD jobs
  N=$($PY -c "import json;print(json.load(open('$W/config.json'))['cpu_timing_n_sequences'])")
  launch gzip_fm_cpu "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; for ds in $DATASETS; do a=\$(python -c \"import json;print(json.load(open('$W/config.json'))['datasets']['\$ds']['alphabet'])\"); for m in \$(python -c \"import json;print(' '.join(json.load(open('$W/config.json'))['foundation_models']['\$a']))\"); do taskset -c 0-7 python fm_embed.py --ds \$ds --model \$m --device cpu --limit $N || exit 1; done; done; echo FM_CPU_DONE" ;;
exp)
  (cd $W/src && $PY plan.py)
  NT=$(wc -l < $W/results/tasks.txt)
  for w in $(seq 0 $((EXP_WORKERS-1))); do
    launch gzip_exp_$w "export OMP_NUM_THREADS=$EXP_THREADS OPENBLAS_NUM_THREADS=$EXP_THREADS MKL_NUM_THREADS=$EXP_THREADS GZ_NJOBS=$EXP_THREADS; for t in \$(seq $w $EXP_WORKERS $((NT-1))); do python run_experiment.py --task \$t || exit 1; done; echo EXP_DONE"
  done ;;
post)
  launch gzip_post "export OMP_NUM_THREADS=8 GZ_NJOBS=8; python aggregate.py && for ds in $DATASETS; do python perm_null.py --ds \$ds && python timing_bench.py --ds \$ds --n-queries 20 || exit 1; done && python aggregate.py && python make_tables.py && echo POST_DONE" ;;
status)
  tmux ls 2>/dev/null | grep gzip_ || echo "no gzip_ sessions"
  for f in $W/logs/gzip_*.log; do [ -f "$f" ] && echo "== $(basename $f): $(tail -n 1 $f | cut -c1-150)"; done ;;
*)
  echo "usage: bash scripts/run.sh {prepare|ncd|fmgpu|fmcpu|exp|post|status}"; exit 1 ;;
esac
