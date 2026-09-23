#!/usr/bin/env bash
#SBATCH --job-name=n2_p64m0
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --array=0-7%8
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_p64m0-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_p64m0-%A_%a.err
#SBATCH --export=NIL
set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark; root="$workspace/ICLR/new ROUTEA/RC"; py="$workspace/.venv-romav2/bin/python"; cache="$workspace/third_party/model_cache"; runtime=${SLURM_TMPDIR:-/tmp}/n2-p64m0-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
umask 077
test -d "$root"; test -x "$py"; test -d "$root/logs"
mkdir -p "$runtime/home" "$runtime/tmp" "$runtime/hf" "$runtime/xdg" "$runtime/torchinductor" "$runtime/triton"; export PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$root/src:$root/programs:$root/../route_a_core:$root/..:$workspace" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 HOME="$runtime/home" TMPDIR="$runtime/tmp" HF_HOME="$runtime/hf" TORCH_HOME="$cache/torch" XDG_CACHE_HOME="$runtime/xdg" TORCHINDUCTOR_CACHE_DIR="$runtime/torchinductor" TRITON_CACHE_DIR="$runtime/triton" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$root"
export ROUTEA_PAIR64_AUTHORITY_STAGE=m0_array
"$py" programs/validate_routea_n2_fresh_d1_pair64_training_input_execution_authority_v1.py
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
shard_dir=$(printf 'results/routea_n2_fresh_d1_matched_three_arm_pair64_training_features_v1/shard%02d' "$SLURM_ARRAY_TASK_ID")
if [[ ! -f "$shard_dir/payload.pt" || ! -f "$shard_dir/receipt.json" ]]; then
  "$py" programs/materialize_routea_n2_fresh_d1_matched_three_arm_pair64_training_features_shard_v1.py --shard "$SLURM_ARRAY_TASK_ID"
fi
[[ -f "$shard_dir/validation.json" ]] || "$py" programs/validate_routea_n2_fresh_d1_matched_three_arm_pair64_training_features_shard_v1.py --shard "$SLURM_ARRAY_TASK_ID"
