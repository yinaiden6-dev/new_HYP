#!/usr/bin/env bash
#SBATCH --job-name=raw_interface6
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --input=/dev/null
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_difficult0006_raw_interface_bridge_v1-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_difficult0006_raw_interface_bridge_v1-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
umask 077
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_workspace='/hkfs/work/workspace/scratch/ap7811-benchmark'
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
rc_python="$rc_workspace/.venv-colpali/bin/python"
rc_runtime="${SLURM_TMPDIR:-/tmp}/raw-interface6-${SLURM_JOB_ID:-manual}"
mkdir -p "$rc_runtime"
export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$rc_runtime" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$rc_root"
"$rc_python" programs/run_rc_difficult0006_raw_interface_bridge_v1.py --phase run
exec "$rc_python" programs/run_rc_difficult0006_raw_interface_bridge_v1.py --phase validate
