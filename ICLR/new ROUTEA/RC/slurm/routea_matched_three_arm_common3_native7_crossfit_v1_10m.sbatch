#!/usr/bin/env bash
#SBATCH --job-name=3a_crossfit
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/3a_crossfit-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/3a_crossfit-%j.err
#SBATCH --export=NIL
set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark;root="$workspace/ICLR/new ROUTEA/RC";py="$workspace/.venv-colpali/bin/python";runtime=${SLURM_TMPDIR:-/tmp}/3a-crossfit-${SLURM_JOB_ID};mkdir -p "$runtime/home" "$runtime/tmp";export PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$root/src:$root/programs:$root/../route_a_core:$root/..:$workspace" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 HOME="$runtime/home" TMPDIR="$runtime/tmp" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8;cd "$root";"$py" programs/run_routea_matched_three_arm_common3_native7_crossfit_v1.py;"$py" programs/validate_routea_matched_three_arm_common3_native7_crossfit_v1.py
