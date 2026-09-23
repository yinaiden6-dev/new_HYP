#!/bin/bash
#SBATCH --job-name=rcde_tr_e1
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384
#SBATCH --time=01:15:00
#SBATCH --array=0-49%20
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NIL
set -euo pipefail
umask 077

rc_workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
rc_python="$rc_workspace/.venv-colpali/bin/python"
rc_authority="$rc_root/registry/current_authority_v118_20260821.json"
rc_authority_validation="$rc_root/results/dino_rcde_track_r_relative_e1_authority_v118_revision2_validation_v1/result.json"
rc_producer="$rc_root/programs/materialize_dino_rcde_track_r_relative_e1_episode_shard_v1.py"
rc_validator="$rc_root/programs/validate_dino_rcde_track_r_relative_e1_episode_shard_v1.py"
rc_producer_root="$rc_root/results/dino_rcde_track_r_relative_e1_shards_v1/producer"
rc_validation_root="$rc_root/results/dino_rcde_track_r_relative_e1_shards_v1/validation"
rc_launcher="$rc_root/slurm/dino_rcde_track_r_relative_e1_episode_shards_v1.sbatch"

test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "${SLURM_JOB_PARTITION:?}" = cpuonly
test "${SLURM_CPUS_PER_TASK:?}" -eq 4
test "${SLURM_MEM_PER_NODE:?}" -eq 16384
test -x "$rc_python"
test -f "$rc_authority"
test ! -L "$rc_authority"
test "$(stat -c '%a' "$rc_authority")" = 444
test -f "$rc_authority_validation"
test ! -L "$rc_authority_validation"
test "$(stat -c '%a' "$rc_authority_validation")" = 444
cmp -s "$0" "$rc_launcher"
"$rc_python" -c 'import hashlib,json,os,stat,sys; ap,vp=sys.argv[1:]; sha=lambda p: hashlib.sha256(open(p,"rb").read()).hexdigest(); logical=lambda v: hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!="logical_sha256"},sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")).hexdigest(); a=json.load(open(ap,encoding="utf-8")); v=json.load(open(vp,encoding="utf-8")); assert not os.path.islink(ap) and not os.path.islink(vp); assert stat.S_IMODE(os.stat(ap).st_mode)==0o444 and stat.S_IMODE(os.stat(vp).st_mode)==0o444; assert a.get("schema_version")=="rc_current_authority_v118_20260821" and a.get("status")=="DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARDS_AUTHORIZED" and a.get("authority_version")==118 and a.get("authority_revision")==2 and a.get("logical_sha256")==logical(a); assert a.get("independent_authority_validation_output")=="results/dino_rcde_track_r_relative_e1_authority_v118_revision2_validation_v1/result.json"; assert v.get("status")=="DINO_RCDE_TRACK_R_RELATIVE_E1_PARALLEL_AUTHORITY_V118_REVISION2_INDEPENDENT_VALIDATION_PASS" and v.get("validation_pass") is True and v.get("authority_version")==118 and v.get("authority_revision")==2 and v.get("authority_sha256")==sha(ap) and v.get("authority_logical_sha256")==a.get("logical_sha256") and v.get("logical_sha256")==logical(v); assert v.get("training_authorized") is False and v.get("scientific_GO_or_NO_GO") is None' "$rc_authority" "$rc_authority_validation"
rc_shard=${SLURM_ARRAY_TASK_ID:?}
test "$rc_shard" -ge 0
test "$rc_shard" -lt 50
rc_start=$((rc_shard*12))
rc_stop=$((rc_start+12))
rc_payload=$(printf '%s/shard_%03d_%03d.pt' "$rc_producer_root" "$rc_start" "$rc_stop")
rc_validation=$(printf '%s/shard_%03d_%03d.validation.json' "$rc_validation_root" "$rc_start" "$rc_stop")
test ! -e "$rc_payload"
test ! -e "$rc_validation"

export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$rc_root"
"$rc_python" "$rc_producer" \
  --authority "$rc_authority" \
  --shard-ordinal "$rc_shard" \
  --output-root "$rc_producer_root"
"$rc_python" "$rc_validator" \
  --authority "$rc_authority" \
  --shard "$rc_payload" \
  --shard-ordinal "$rc_shard" \
  --output "$rc_validation"

test -f "$rc_payload"
test -f "$rc_validation"
