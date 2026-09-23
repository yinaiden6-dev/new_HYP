#!/usr/bin/env bash
#SBATCH --job-name=h593_snet
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-snet-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-snet-%A_%a.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_tmp="${SLURM_TMPDIR:-/tmp}/h593-snet-${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-single}"
mkdir -p "$rc_tmp"
export TMPDIR="$rc_tmp"
cd "$rc_root"
case "${1:?stage required}" in
 fit) exec timeout --signal=TERM --kill-after=10s 540s "$rc_python" programs/run_rc_h593_s_net_hold_optimization_v1.py fit --fold "${SLURM_ARRAY_TASK_ID:?}" ;;
 join) exec timeout --signal=TERM --kill-after=10s 540s "$rc_python" programs/run_rc_h593_s_net_hold_optimization_v1.py join ;;
 *) exit 2 ;;
esac
