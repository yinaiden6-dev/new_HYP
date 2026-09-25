#!/usr/bin/env bash
#SBATCH --job-name=M_post_why
#SBATCH --partition=dev_cpuonly,cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-M-why-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-M-why-%j.err
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
for task_rows in 0-5 6-11 12-17 18-23; do
  "$PY" -u programs/analyze_rc_postllm_signal_decomposition_v1.py --rows "$task_rows" --threads 2 --budget 440 &
  pids+=("$!")
done
status=0
for task_pid in "${pids[@]}"; do
  if ! wait "$task_pid"; then status=1; fi
done
if [[ "$status" != 0 ]]; then exit "$status"; fi
"$PY" -u programs/analyze_rc_postllm_signal_decomposition_v1.py --summarize-only
"$PY" - <<'PY'
import json
from pathlib import Path
p=Path('results/rc_postllm_m_signal_decomposition_v1/summary.json')
d=json.loads(p.read_text())
assert d['status']=='ALL24_COMPLETE', d['status']
print(json.dumps({'status':d['status'],'completed_queries':d['completed_queries'],
                 'cpu_gpu_decision_disagreements':d['native_CPU_GPU_decision_disagreements']}))
PY
