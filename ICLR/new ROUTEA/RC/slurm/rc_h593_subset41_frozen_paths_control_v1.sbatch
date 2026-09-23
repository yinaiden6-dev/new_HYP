#!/usr/bin/env bash
#SBATCH --job-name=h41_ctl
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h41-pathctl-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h41-pathctl-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
rc_workspace='/hkfs/work/workspace/scratch/ap7811-benchmark'
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export TORCH_HOME="$rc_workspace/third_party/model_cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=''
cd "$rc_root"
exec timeout --signal=TERM --kill-after=10s 550s "$rc_workspace/.venv-romav2/bin/python" programs/run_rc_h593_subset41_frozen_paths_v1.py "${1:-control}" ${2:+--previous "$2"}
