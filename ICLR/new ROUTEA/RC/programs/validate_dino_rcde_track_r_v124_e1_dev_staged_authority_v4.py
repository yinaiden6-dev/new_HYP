#!/usr/bin/env python3
"""Independent static validation of V124 E1 dev staged V4 authority."""

from __future__ import annotations
import json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from rc_aslo_xf import dino_rcde_track_r_v124_e1_dev_staged_authority_v4 as A  # noqa:E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_real_smoke_authority_v2 as P  # noqa:E402

AUTH=ROOT/A.AUTHORITY_PATH
OUT=ROOT/"registry/dino_rcde_track_r_v124_e1_dev_staged_smoke_authority_v4_20260826.independent_validation.json"

def main() -> None:
    sha=P.BASE.V2.file_sha256(AUTH); merged=A.read_authority(AUTH,sha)
    prod=(ROOT/A.OVERLAY_PATHS["producer_launcher"]).read_text(); val=(ROOT/A.OVERLAY_PATHS["validator_launcher"]).read_text()
    checks={
      "authority_valid":True,
      "parent_science_binding_preserved":merged["v_checkpoint_state_sha256"] is not None and merged["candidate_count"]==128,
      "both_dev_one_hour":all("#SBATCH --partition=dev_accelerated" in x and "#SBATCH --time=01:00:00" in x for x in (prod,val)),
      "producer_only":A.OVERLAY_PATHS["producer"] in prod and A.OVERLAY_PATHS["validator"] not in prod,
      "validator_only":A.OVERLAY_PATHS["validator"] in val and A.OVERLAY_PATHS["producer"] not in val,
      "producer_checkpoint_required":("family_commit.json" in val),
      "no_training_or_science":merged["model_backward_authorized"] is False and merged["model_update_authorized"] is False and merged["scientific_reduction_authorized"] is False,
      "new_outputs_absent":not os.path.lexists(ROOT/A.PRODUCER_NAMESPACE) and not os.path.lexists(ROOT/A.VALIDATION_NAMESPACE),
    }
    result={"schema_version":"rc_dino_rcde_track_r_v124_e1_dev_staged_authority_validation_v4_20260826","status":"DINO_RCDE_TRACK_R_V124_E1_DEV_STAGED_SMOKE_V4_AUTHORITY_INDEPENDENT_VALIDATION_PASS" if all(checks.values()) else "DINO_RCDE_TRACK_R_V124_E1_DEV_STAGED_SMOKE_V4_AUTHORITY_VALIDATION_FAIL","authority_sha256":sha,"checks":checks,"scientific_GO_or_NO_GO":None,"eligible_as_e1_result":False,"next_authorized_stage":None}
    result["logical_sha256"]=A.logical(result)
    if os.path.lexists(OUT): raise RuntimeError(f"validation exists: {OUT}")
    OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="ascii"); OUT.chmod(0o444); print(json.dumps({"status":result["status"],"output":str(OUT)},sort_keys=True)); raise SystemExit(0 if all(checks.values()) else 2)

if __name__=="__main__": main()
