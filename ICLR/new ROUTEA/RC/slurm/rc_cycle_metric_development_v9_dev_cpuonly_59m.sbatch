#!/usr/bin/env bash
#SBATCH --job-name=cm9_develop
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=dev_cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --input=/dev/null
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_cycle_metric_v9-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_cycle_metric_v9-%j.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
umask 077
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-romav2/bin/python'
rc_runtime="${SLURM_TMPDIR:-/tmp}/rc-cycle-metric-${SLURM_JOB_ID:-manual}"
rc_mode="${1:?mode required}"
rc_lane="${2:-fresh}"
rc_authority="${3:-$rc_root/registry/rc_cycle_metric_development_authority_v9_20260909.json}"
case "$rc_mode" in prepare|train|reduce|validate) ;; *) exit 64 ;; esac
case "$rc_lane" in fresh|resume) ;; *) exit 64 ;; esac
mkdir -p "$rc_runtime"
export TMPDIR="$rc_runtime" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$rc_root"
"$rc_python" programs/run_rc_cycle_metric_development_v9.py --authority "$rc_authority" --mode sources
"$rc_python" programs/run_rc_cycle_metric_development_v9.py --authority "$rc_authority" --mode "$rc_mode" --lane "$rc_lane" --max-seconds 3200
if [[ "$rc_mode" == train ]]; then
    rc_status="$("$rc_python" - "$rc_root" "$rc_authority" "$rc_lane" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
authority = json.loads(pathlib.Path(sys.argv[2]).read_text())
state = json.loads((root / authority['work_rel'] / (sys.argv[3] + '_state.json')).read_text())
assert state['lane'] == sys.argv[3] and state['update_index'] <= 512
assert state['status'] in ('COMPLETE', 'RESUMABLE')
assert (state['status'] == 'COMPLETE') == (state['update_index'] == 512)
print(state['status'])
PY
)"
    if [[ "$rc_status" == RESUMABLE ]]; then
        : "${SLURM_JOB_ID:?resumable training requires Slurm job ID}"
        # Requeue this same immutable authority/checkpoint; afterok consumers stay blocked.
        exec /usr/bin/scontrol requeue "$SLURM_JOB_ID"
    fi
fi
