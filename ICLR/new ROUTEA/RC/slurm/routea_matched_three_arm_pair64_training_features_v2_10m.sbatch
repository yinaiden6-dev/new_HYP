#!/usr/bin/env bash
#SBATCH --job-name=pair64_v2
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/pair64_v2-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/pair64_v2-%j.err
#SBATCH --export=NIL
set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
root="$workspace/ICLR/new ROUTEA/RC"
py="$workspace/.venv-romav2/bin/python"
model_cache="$workspace/third_party/model_cache"
runtime=${SLURM_TMPDIR:-/tmp}/pair64-v2-${SLURM_JOB_ID}
mkdir -p "$runtime/home" "$runtime/tmp" "$runtime/xdg" "$runtime/hf" "$runtime/torchinductor" "$runtime/triton"
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$root/src:$root/programs:$workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HOME="$runtime/home" TORCH_HOME="$model_cache/torch" HF_HOME="$runtime/hf"
export TMPDIR="$runtime/tmp" XDG_CACHE_HOME="$runtime/xdg"
export TORCHINDUCTOR_CACHE_DIR="$runtime/torchinductor" TRITON_CACHE_DIR="$runtime/triton"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$root"
echo "job=$SLURM_JOB_ID host=$(hostname) stage=PAIR64_V2_POST_FULL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$py" programs/materialize_routea_matched_three_arm_pair64_training_features_v2.py
"$py" programs/validate_routea_matched_three_arm_pair64_training_features_v2.py
