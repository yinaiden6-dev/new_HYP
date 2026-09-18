#!/usr/bin/env bash
#SBATCH --job-name=proc128
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/proc128-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/proc128-%A_%a.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_roma_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-romav2/bin/python'
rc_tmp="${SLURM_TMPDIR:-/tmp}/proc128-${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-join}"
mkdir -p "$rc_tmp"
export TMPDIR="$rc_tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME='/hkfs/work/workspace/scratch/ap7811-benchmark/third_party/model_cache/torch'
cd "$rc_root"
case "${1:?stage}" in
 raw) exec timeout --signal=TERM --kill-after=10s 540s "$rc_python" programs/run_rc_new_hyp_processed128_v1.py raw --shard "$SLURM_ARRAY_TASK_ID" ;;
 roma) exec timeout --signal=TERM --kill-after=10s 840s "$rc_roma_python" programs/run_rc_new_hyp_processed128_v1.py roma --shard "$SLURM_ARRAY_TASK_ID" ;;
 join) export CUDA_VISIBLE_DEVICES=''; exec timeout --signal=TERM --kill-after=10s 540s "$rc_python" programs/analyze_rc_new_hyp_processed128_v1.py ;;
 *) exit 2 ;;
esac
