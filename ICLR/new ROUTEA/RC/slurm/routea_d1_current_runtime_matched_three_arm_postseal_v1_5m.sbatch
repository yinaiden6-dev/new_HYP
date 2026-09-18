#!/usr/bin/env bash
#SBATCH --job-name=d1_3apost
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/d1_3apost-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/d1_3apost-%j.err
#SBATCH --export=NIL
set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
root="$workspace/ICLR/new ROUTEA/RC"
py="$workspace/.venv-colpali/bin/python"
runtime=${SLURM_TMPDIR:-/tmp}/d1-3apost-${SLURM_JOB_ID}
mkdir -p "$runtime/home" "$runtime/tmp"
export PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$root/src:$root/programs:$root/../route_a_core:$root/..:$workspace" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 HOME="$runtime/home" TMPDIR="$runtime/tmp" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$root"
echo "job=$SLURM_JOB_ID host=$(hostname) stage=D1_MATCHED_THREE_ARM_POSTSEAL"
"$py" programs/evaluate_routea_d1_current_runtime_matched_three_arm_postseal_v1.py
"$py" programs/validate_routea_d1_current_runtime_matched_three_arm_postseal_v1.py
