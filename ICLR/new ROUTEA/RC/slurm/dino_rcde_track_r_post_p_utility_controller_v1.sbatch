#!/bin/bash
#SBATCH --job-name=rcde_tr_c6
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16384
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail
rc_root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
rc_python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
rc_self="$rc_root/slurm/dino_rcde_track_r_post_p_utility_controller_v1.sbatch"
rc_authority="$rc_root/registry/current_authority_v122_20260821.json"
rc_producer="$rc_root/results/dino_rcde_track_r_p_v2_utility_shards_v1/producer"
rc_validation="$rc_root/results/dino_rcde_track_r_p_v2_utility_shards_v1/validation"
rc_comparator="$rc_root/results/dino_rcde_track_r_p_v2_utility_comparator_v1/result.json"
rc_comparator_validation="$rc_root/results/dino_rcde_track_r_p_v2_utility_comparator_validation_v1/result.json"
rc_receipt="$rc_root/results/dino_rcde_track_r_autochain_v1/post_p_utility_submission.json"
test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test ! -e "$rc_comparator"
test ! -e "$rc_comparator_validation"
test ! -e "$rc_receipt"
cmp -s "$0" "$rc_self"
cd "$rc_root"
"$rc_python" "$rc_root/programs/reduce_dino_rcde_track_r_p_v2_utility_v1.py" --authority "$rc_authority" --producer-root "$rc_producer" --validation-root "$rc_validation" --output "$rc_comparator"
"$rc_python" "$rc_root/programs/validate_dino_rcde_track_r_p_v2_utility_comparator_v1.py" --authority "$rc_authority" --producer-root "$rc_producer" --validation-root "$rc_validation" --comparator "$rc_comparator" --output "$rc_comparator_validation"
test "$(jq -er '.status' "$rc_comparator_validation")" = DINO_RCDE_TRACK_R_P_V2_UTILITY_COMPARATOR_INDEPENDENT_VALIDATION_PASS
mkdir -p "$(dirname "$rc_receipt")"
"$rc_python" -c 'import json,os,pathlib,sys,tempfile;p=pathlib.Path(sys.argv[1]);v={"schema_version":"rc_dino_rcde_track_r_post_p_utility_submission_v1_20260822","status":"V122_P_UTILITY_COMPARATOR_COMPLETE_AWAITING_SEPARATE_V123_AUTHORITY","controller_job_id":int(os.environ["SLURM_JOB_ID"]),"automatic_stage_advance":False,"next_authorized_stage":None};fd,t=tempfile.mkstemp(prefix="."+p.name+".",suffix=".partial",dir=p.parent);os.close(fd);q=pathlib.Path(t);q.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");os.replace(q,p);p.chmod(0o444)' "$rc_receipt"
