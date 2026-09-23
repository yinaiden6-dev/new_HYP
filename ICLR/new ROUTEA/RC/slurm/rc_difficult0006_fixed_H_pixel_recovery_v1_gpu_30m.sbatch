#!/usr/bin/env bash
#SBATCH --job-name=diff0006_H
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --input=/dev/null
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_difficult0006_fixed_H_pixel_recovery_v1-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_difficult0006_fixed_H_pixel_recovery_v1-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
umask 077
unset LD_LIBRARY_PATH
rc_workspace='/hkfs/work/workspace/scratch/ap7811-benchmark'
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
rc_python="$rc_workspace/.venv-colpali/bin/python"
rc_runtime="${SLURM_TMPDIR:-/tmp}/diff0006-H-${SLURM_JOB_ID:-manual}"
export PATH=/usr/local/bin:/usr/bin:/bin
mkdir -p "$rc_runtime/hf" "$rc_runtime/torch" "$rc_runtime/xdg" "$rc_runtime/tmp"
export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$rc_runtime/hf" TORCH_HOME="$rc_runtime/torch" XDG_CACHE_HOME="$rc_runtime/xdg" TMPDIR="$rc_runtime/tmp"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$rc_root"
"$rc_python" programs/run_rc_difficult0006_fixed_H_pixel_recovery_v1.py --phase run
exec "$rc_python" programs/run_rc_difficult0006_fixed_H_pixel_recovery_v1.py --phase validate
