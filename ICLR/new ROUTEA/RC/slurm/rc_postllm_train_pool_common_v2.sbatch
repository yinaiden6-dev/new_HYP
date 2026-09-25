#!/usr/bin/env bash
#SBATCH --job-name=M_post_pool
#SBATCH --partition=dev_cpuonly,cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-M-pool-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-M-pool-%j.err
#SBATCH --export=NIL
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
PY='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
"$PY" -u programs/analyze_rc_postllm_train_pool_common_v2.py --rows 16-23 --threads 2 --budget 240
"$PY" - <<'PY'
from pathlib import Path
import json
p=Path('results/rc_postllm_train_pool_common_v2/queries')
rows=[json.loads(x.read_text()) for x in p.glob('*/result.json')]
assert len([r for r in rows if r['split']=='probe'])==8, 'incomplete probe'
print('ALL8_FROZEN_PROBE_COMPLETE')
PY
