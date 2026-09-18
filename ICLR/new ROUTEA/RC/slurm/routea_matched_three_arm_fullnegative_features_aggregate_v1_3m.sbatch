#!/usr/bin/env bash
# Submit only with --dependency=afterok:<feature-array-jobid>.
#SBATCH --job-name=3a_featagg
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:03:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/3a_featagg-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/3a_featagg-%j.err
#SBATCH --export=NIL
set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark;root="$workspace/ICLR/new ROUTEA/RC";py="$workspace/.venv-romav2/bin/python";runtime=${SLURM_TMPDIR:-/tmp}/3a-featagg-${SLURM_JOB_ID};mkdir -p "$runtime/home" "$runtime/tmp";export PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$root/src:$root/programs:$workspace" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 HOME="$runtime/home" TMPDIR="$runtime/tmp" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4;cd "$root";exec "$py" programs/validate_routea_matched_three_arm_fullnegative_features_aggregate_v1.py
