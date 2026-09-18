#!/usr/bin/env bash
#SBATCH --job-name=roma_se5
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma_se5-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma_se5-%j.err
#SBATCH --export=NIL

set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
root="$workspace/ICLR/new ROUTEA/RC"
python_bin="$workspace/.venv-colpali/bin/python"
runtime=${SLURM_TMPDIR:-/tmp}/roma-se5-${SLURM_JOB_ID}
mkdir -p "$runtime/home" "$runtime/hf" "$runtime/torch" "$runtime/xdg" "$runtime/tmp"
export PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$root/src:$root/programs:$workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$runtime/hf" TORCH_HOME="$runtime/torch" XDG_CACHE_HOME="$runtime/xdg"
export TMPDIR="$runtime/tmp" HOME="$runtime/home"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$root"
echo "job=$SLURM_JOB_ID host=$(hostname) stage=OPENED_EXACT_LABEL_PORTABILITY_E0_V5"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$python_bin" programs/run_romav2_colnomic_sealed_source_portability_e0_v5.py
