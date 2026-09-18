#!/usr/bin/env bash
# Submit only with --dependency=afterok:<matched-three-arm-array-jobid>.
#SBATCH --job-name=d1_3aagg
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/d1_3aagg-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/d1_3aagg-%j.err
#SBATCH --export=NIL
set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
root="$workspace/ICLR/new ROUTEA/RC"
py="$workspace/.venv-romav2/bin/python"
runtime=${SLURM_TMPDIR:-/tmp}/d1-3aagg-${SLURM_JOB_ID}
mkdir -p "$runtime/home" "$runtime/tmp"
export PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$root/src:$root/programs:$workspace" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 HOME="$runtime/home" TMPDIR="$runtime/tmp" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$root"
echo "job=$SLURM_JOB_ID host=$(hostname) stage=D1_MATCHED_THREE_ARM_AGGREGATE"
exec "$py" programs/validate_routea_d1_current_runtime_matched_three_arm_aggregate_v1.py
