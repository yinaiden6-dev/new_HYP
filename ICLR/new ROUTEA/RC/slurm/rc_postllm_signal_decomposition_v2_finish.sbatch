#!/usr/bin/env bash
#SBATCH --job-name=M_post_finish
#SBATCH --partition=dev_cpuonly,cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-M-finish-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-M-finish-%j.err
#SBATCH --export=NIL
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
PY='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
"$PY" -u programs/analyze_rc_postllm_signal_decomposition_v2.py --rows 23 --threads 2 --budget 180
"$PY" -u programs/analyze_rc_postllm_signal_decomposition_v2.py --summarize-only
"$PY" - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('results/rc_postllm_m_signal_decomposition_v2/summary.json').read_text())
assert d['status']=='ALL24_COMPLETE', d['status']
print('ALL24_COMPLETE')
PY
