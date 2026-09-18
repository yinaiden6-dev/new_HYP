#!/usr/bin/env bash
#SBATCH --job-name=n2_c64agg
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_c64agg-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_c64agg-%j.err
#SBATCH --export=NIL

set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
root="$workspace/ICLR/new ROUTEA/RC"
py="$workspace/.venv-romav2/bin/python"
runtime=${SLURM_TMPDIR:-/tmp}/routea-n2-current64-aggregate-${SLURM_JOB_ID}
mkdir -p "$runtime/home" "$runtime/xdg" "$runtime/tmp"
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$root/src:$root/programs:$root/../route_a_core:$root/..:$workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export XDG_CACHE_HOME="$runtime/xdg" TMPDIR="$runtime/tmp" HOME="$runtime/home"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

cd "$root"
export ROUTEA_CURRENT64_AUTHORITY_STAGE=aggregate
"$py" programs/validate_routea_n2_fresh_d1_matched_three_arm_current64_staging_execution_authority_v1.py
echo "job=$SLURM_JOB_ID host=$(hostname) stage=N2_FRESH_D1_CURRENT64_PREJOIN_AGGREGATE"
"$py" programs/validate_routea_n2_fresh_d1_matched_three_arm_current64_prejoin_aggregate_v1.py
