#!/usr/bin/env bash
#SBATCH --job-name=roma_b7t0
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma_b7t0-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma_b7t0-%j.err
#SBATCH --export=NIL
set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark;root="$workspace/ICLR/new ROUTEA/RC";py="$workspace/.venv-colpali/bin/python";runtime=${SLURM_TMPDIR:-/tmp}/roma-b7t0-${SLURM_JOB_ID};mkdir -p "$runtime/home" "$runtime/hf" "$runtime/torch" "$runtime/xdg" "$runtime/tmp";export PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$root/src:$root/programs:$workspace" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HOME="$runtime/hf" TORCH_HOME="$runtime/torch" XDG_CACHE_HOME="$runtime/xdg" TMPDIR="$runtime/tmp" HOME="$runtime/home" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8;cd "$root";echo "job=$SLURM_JOB_ID host=$(hostname) stage=B7_VIDEO_TOKEN_FULLRANK_SHARD0";nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;exec "$py" programs/materialize_romav2_colnomic_b7_video_token_fullrank_shard_v1.py --shard 0
