#!/bin/bash
#SBATCH --job-name=rcde_v124_e1p4
#SBATCH --partition=dev_accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128000M
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail
umask 077
rc_root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
rc_python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
rc_authority="$rc_root/registry/dino_rcde_track_r_v124_e1_dev_staged_smoke_authority_v4_20260826.json"
rc_program="$rc_root/programs/run_dino_rcde_track_r_v124_e1_dev_staged_producer_v4.py"
test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "${SLURM_JOB_PARTITION:?}" = dev_accelerated
test "${SLURM_CPUS_PER_TASK:?}" -eq 8
test "${SLURM_MEM_PER_NODE:?}" -eq 128000
test -n "${CUDA_VISIBLE_DEVICES:?}"
test -f "$rc_authority"; test ! -L "$rc_authority"; test "$(stat -c %a "$rc_authority")" = 444
test ! -L "$rc_program"
rc_tmp=$(mktemp -d "/tmp/rcde-v124-e1p4-${SLURM_JOB_ID}.XXXXXX")
trap 'chmod -R u+w "$rc_tmp" 2>/dev/null || true; rm -rf -- "$rc_tmp"' EXIT
export TMPDIR="$rc_tmp" PYTHONPATH="$rc_root/src:$rc_root/programs:/hkfs/work/workspace/scratch/ap7811-benchmark"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$rc_root"
rc_authority_sha=$(sha256sum "$rc_authority" | awk '{print $1}')
"$rc_python" "$rc_program" --authority "$rc_authority" --authority-sha256 "$rc_authority_sha"

