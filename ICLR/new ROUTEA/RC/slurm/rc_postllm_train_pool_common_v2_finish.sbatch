#!/usr/bin/env bash
#SBATCH --job-name=M_pool_finish
#SBATCH --partition=dev_cpuonly,cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-M-pool-finish-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-M-pool-finish-%j.err
#SBATCH --export=NIL
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
PY='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
pids=()
for task_rows in 16-17 18-19 20-21 22-23; do
  "$PY" -u programs/analyze_rc_postllm_train_pool_common_v2.py --rows "$task_rows" --threads 2 --budget 460 &
  pids+=("$!")
done
status=0
for task_pid in "${pids[@]}"; do
  if ! wait "$task_pid"; then status=1; fi
done
if [[ "$status" != 0 ]]; then exit "$status"; fi
"$PY" -u programs/analyze_rc_postllm_train_pool_common_v2.py --summarize-only
"$PY" - <<'PY'
from pathlib import Path
import json
d=json.loads(Path('results/rc_postllm_train_pool_common_v2/summary.json').read_text())
assert d['status']=='ALL8_PROBE_COMPLETE' and d['completed_probe_queries']==8, d['status']
print('ALL8_FROZEN_PROBE_COMPLETE')
PY
