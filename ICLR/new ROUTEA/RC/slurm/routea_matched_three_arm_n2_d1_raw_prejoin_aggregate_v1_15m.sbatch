#!/bin/bash
#SBATCH --job-name=n2_d1ragg
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail
umask 077
export PATH=/usr/bin:/bin
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
root="$workspace/ICLR/new ROUTEA/RC"
python="$workspace/.venv-colpali/bin/python"
test -x "$python"
test "${SLURM_SUBMIT_DIR:-}" = "$root"
runtime=${SLURM_TMPDIR:-/tmp}/routea-n2-d1-rawagg-${SLURM_JOB_ID}
mkdir -p "$runtime/home" "$runtime/tmp"
export HOME="$runtime/home" TMPDIR="$runtime/tmp" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/src:$workspace/ICLR/new ROUTEA/route_a_core"
exec "$python" "$root/programs/validate_routea_matched_three_arm_n2_d1_raw_prejoin_aggregate_v1.py"
