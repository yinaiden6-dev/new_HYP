#!/usr/bin/env bash
#SBATCH --job-name=post_grozi
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-grozi-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-grozi-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
export PYTHONPATH="$PWD/programs:$PWD/src:/hkfs/work/workspace/scratch/ap7811-benchmark"
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
role="${1:?role required}"
case "$role" in
  pilot) args=(programs/run_rc_postllm_grozi_external_v1.py --pilot --budget 490) ;;
  worker) args=(programs/run_rc_postllm_grozi_external_v1.py --task "${SLURM_ARRAY_TASK_ID:?}" --budget 490) ;;
  join) args=(programs/join_rc_postllm_grozi_external_v1.py) ;;
  *) exit 2 ;;
esac
set +e
timeout --signal=TERM --kill-after=15s 550s "$rc_python" -u "${args[@]}"
rc_status=$?
set -e
if [[ "$rc_status" == 75 || "$rc_status" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 12 ]]; then exit 75; fi
  rc_job="${SLURM_JOB_ID:?}"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then rc_job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID:?}"; fi
  scontrol requeue "$rc_job"
  exit 0
fi
exit "$rc_status"
