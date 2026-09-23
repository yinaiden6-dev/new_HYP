#!/bin/bash
#SBATCH --job-name=rcde_tr_c4
#SBATCH --partition=dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail

rc_root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
rc_python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
rc_completion="$rc_root/results/dino_rcde_track_r_oof_prejoin_completion_v120/result.json"
rc_freezer="$rc_root/programs/freeze_dino_rcde_track_r_p_v2_utility_authority_v121.py"
rc_authority="$rc_root/registry/current_authority_v122_20260821.json"
rc_authority_validator="$rc_root/programs/validate_dino_rcde_track_r_p_v2_utility_authority_v122.py"
rc_authority_validation="$rc_root/results/dino_rcde_track_r_p_v2_utility_authority_validation_v1/result.json"
rc_launcher="$rc_root/slurm/dino_rcde_track_r_p_v2_utility_shards_v1.sbatch"
rc_next="$rc_root/slurm/dino_rcde_track_r_post_p_utility_shard0_controller_v1.sbatch"
rc_receipt="$rc_root/results/dino_rcde_track_r_autochain_v1/post_oof_v2_submission.json"
test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "$(jq -er '.status' "$rc_completion")" = DINO_RCDE_TRACK_R_OOF_PREJOIN_V120_COMPLETE
test ! -e "$rc_receipt"
cd "$rc_root"
"$rc_python" "$rc_freezer"
"$rc_python" "$rc_authority_validator" --authority "$rc_authority" --output "$rc_authority_validation"
test "$(jq -er '.status' "$rc_authority_validation")" = DINO_RCDE_TRACK_R_P_V2_UTILITY_AUTHORITY_INDEPENDENT_VALIDATION_PASS
rc_shard0=$(sbatch --parsable --array=0 "$rc_launcher")
rc_controller=$(sbatch --parsable --dependency="afterok:$rc_shard0" "$rc_next")
mkdir -p "$(dirname "$rc_receipt")"
"$rc_python" -c 'import json,os,pathlib,sys,tempfile;p=pathlib.Path(sys.argv[1]);v={"schema_version":"rc_dino_rcde_track_r_post_oof_v2_submission_v1_20260822","status":"V122_SHARD0_AND_CONTROLLER_SUBMITTED","registrar_job_id":int(os.environ["SLURM_JOB_ID"]),"shard0_job_id":int(sys.argv[2]),"controller_job_id":int(sys.argv[3])};fd,t=tempfile.mkstemp(prefix="."+p.name+".",suffix=".partial",dir=p.parent);os.close(fd);q=pathlib.Path(t);q.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");os.replace(q,p);p.chmod(0o444)' "$rc_receipt" "$rc_shard0" "$rc_controller"
