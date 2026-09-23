#!/usr/bin/env bash
#SBATCH --job-name=n2_p64agg
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_p64agg-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_p64agg-%j.err
#SBATCH --export=NIL
set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark; root="$workspace/ICLR/new ROUTEA/RC"; py="$workspace/.venv-colpali/bin/python"; runtime=${SLURM_TMPDIR:-/tmp}/n2-p64agg-${SLURM_JOB_ID}
umask 077
test -d "$root"; test -x "$py"; test -d "$root/logs"
mkdir -p "$runtime/home" "$runtime/tmp"; export PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$root/src:$root/programs:$root/../route_a_core:$root/..:$workspace" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 HOME="$runtime/home" TMPDIR="$runtime/tmp" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$root"
export ROUTEA_PAIR64_AUTHORITY_STAGE=aggregate
"$py" programs/validate_routea_n2_fresh_d1_pair64_training_input_execution_authority_v1.py
"$py" programs/validate_routea_n2_fresh_d1_matched_three_arm_pair64_training_features_aggregate_v1.py
