#!/usr/bin/env bash
#SBATCH --job-name=query_route
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/query-route-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/query-route-%A_%a.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_tmp="${SLURM_TMPDIR:-/tmp}/query-route-${SLURM_JOB_ID:?}"
mkdir -p "$rc_tmp"
export TMPDIR="$rc_tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
cd "$rc_root"
case "${1:?cache, fit or join required}" in
  cache) exec timeout --signal=TERM --kill-after=10s 540s "$rc_python" programs/run_rc_query_content_routing_oof4_v1.py cache ;;
  fit) exec timeout --signal=TERM --kill-after=10s 540s "$rc_python" programs/run_rc_query_content_routing_oof4_v1.py fit --fold "${SLURM_ARRAY_TASK_ID:?}" ;;
  join) exec timeout --signal=TERM --kill-after=10s 540s "$rc_python" programs/validate_rc_query_content_routing_oof4_v1.py join ;;
  *) exit 2 ;;
esac
