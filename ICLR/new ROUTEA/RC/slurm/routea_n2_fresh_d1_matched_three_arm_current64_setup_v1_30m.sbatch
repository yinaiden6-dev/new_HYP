#!/usr/bin/env bash
#SBATCH --job-name=n2_c64set
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_c64set-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_c64set-%j.err
#SBATCH --export=NIL

set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
root="$workspace/ICLR/new ROUTEA/RC"
py="$workspace/.venv-colpali/bin/python"
runtime=${SLURM_TMPDIR:-/tmp}/routea-n2-current64-setup-${SLURM_JOB_ID}
mkdir -p "$runtime/home" "$runtime/hf" "$runtime/torch" "$runtime/xdg" "$runtime/tmp"
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$root/src:$root/programs:$root/../route_a_core:$root/..:$workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$runtime/hf" TORCH_HOME="$runtime/torch" XDG_CACHE_HOME="$runtime/xdg"
export TMPDIR="$runtime/tmp" HOME="$runtime/home"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

cd "$root"
export ROUTEA_CURRENT64_AUTHORITY_STAGE=setup
"$py" programs/validate_routea_n2_fresh_d1_matched_three_arm_current64_staging_execution_authority_v1.py
echo "job=$SLURM_JOB_ID host=$(hostname) stage=N2_FRESH_D1_CURRENT64_SOURCE_AND_REFERENCE_SETUP"
source_root=results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1
reference_root=results/routea_n2_fresh_d1_matched_three_arm_reference_cache_v1
if [[ ! -f "$source_root/manifest.json" ]]; then
  "$py" programs/materialize_routea_n2_fresh_d1_matched_three_arm_source_manifest_v1.py
fi
if [[ ! -f "$source_root/independent_validation.json" ]]; then
  "$py" programs/validate_routea_n2_fresh_d1_matched_three_arm_source_manifest_v1.py
fi
if [[ ! -f "$reference_root/result.json" || ! -f "$reference_root/payload.pt" ]]; then
  "$py" programs/materialize_routea_n2_fresh_d1_matched_three_arm_reference_cache_v1.py
fi
if [[ ! -f "$reference_root/independent_validation.json" ]]; then
  "$py" programs/validate_routea_n2_fresh_d1_matched_three_arm_reference_cache_v1.py
fi
