#!/usr/bin/env bash
#SBATCH --job-name=gz120_infer
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/gz120-infer-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/gz120-infer-%A_%a.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_roma_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-romav2/bin/python'
rc_tmp="${SLURM_TMPDIR:-/tmp}/gz120-infer-${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:?}"
mkdir -p "$rc_tmp"
export TMPDIR="$rc_tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME='/hkfs/work/workspace/scratch/ap7811-benchmark/third_party/model_cache/torch'
cd "$rc_root"
"$rc_python" - <<'PY_GUARD'
import hashlib,json
from pathlib import Path
p=Path('registry/rc_new_hyp_grozi120_runtime_launch_repair_authority_v2_20260913.json')
a=json.loads(p.read_text())
for b in a['sources'].values():
    assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()==b['sha256'], 'REPAIR_SOURCE_DRIFT:'+b['path']
assert a['training_updates']==0 and a['original_worker_unchanged']
PY_GUARD
case "${1:?stage}" in
 raw) exec timeout --signal=TERM --kill-after=10s 540s "$rc_python" programs/run_rc_new_hyp_grozi120_inference_v1.py raw --shard "$SLURM_ARRAY_TASK_ID" ;;
 roma) exec timeout --signal=TERM --kill-after=10s 840s "$rc_roma_python" programs/run_rc_new_hyp_grozi120_inference_v1.py roma --shard "$SLURM_ARRAY_TASK_ID" ;;
 *) exit 2 ;;
esac
