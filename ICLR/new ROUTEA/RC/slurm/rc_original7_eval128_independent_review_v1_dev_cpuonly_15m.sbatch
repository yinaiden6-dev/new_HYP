#!/usr/bin/env bash
#SBATCH --job-name=e128_independent_review
#SBATCH --partition=dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --dependency=afterok:5139200
#SBATCH --input=/dev/null
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_original7_eval128_independent_review_v1-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_original7_eval128_independent_review_v1-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
umask 077
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_review_script="$rc_root/programs/review_rc_original7_eval128_full_evidence_v1.py"
rc_runtime="${SLURM_TMPDIR:-/tmp}/original7-eval128-independent-review-${SLURM_JOB_ID:-manual}"
mkdir -p "$rc_runtime"
export TMPDIR="$rc_runtime" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$rc_root"
"$rc_python" - "$rc_review_script" <<'RC_VERIFY_REVIEWER_SHA'
import hashlib
import sys
from pathlib import Path
expected = '4ed80c0d155dced8a0d9717a9bdde800ba667ff2b93bd9908623107e00232dff'
observed = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
if observed != expected:
    raise SystemExit('FROZEN_INDEPENDENT_REVIEWER_SOURCE_SHA_MISMATCH')
print('FROZEN_INDEPENDENT_REVIEWER_SOURCE_SHA_PASS ' + observed, flush=True)
RC_VERIFY_REVIEWER_SHA
# Beijing 2026-09-11 24:00 is UTC 2026-09-11 16:00.
rc_seconds=$("$rc_python" -c 'from datetime import datetime,timezone;print(max(0,int((datetime(2026,9,11,16,tzinfo=timezone.utc)-datetime.now(timezone.utc)).total_seconds())))')
if (( rc_seconds <= 0 )); then
 echo 'USER_RESEARCH_DEADLINE_REACHED'
 exit 1
fi
# The reviewer itself requires completed result/validation and prejoin bindings before private reads.
timeout --signal=TERM --kill-after=10s "${rc_seconds}s" "$rc_python" "$rc_review_script" --review-validated-result
