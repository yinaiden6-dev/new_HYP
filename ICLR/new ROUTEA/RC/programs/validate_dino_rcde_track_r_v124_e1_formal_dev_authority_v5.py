#!/usr/bin/env python3
from __future__ import annotations
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from rc_aslo_xf import dino_rcde_track_r_v124_e1_formal_dev_authority_v5 as A  # noqa:E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_authority_v3 as P  # noqa:E402
AUTH=ROOT/A.AUTHORITY_PATH;OUT=ROOT/"registry/dino_rcde_track_r_v124_e1_formal_dev_staged_authority_v5_20260826.independent_validation.json"
def main():
 sha=P.V2.file_sha256(AUTH);m=A.read_authority(AUTH,sha);prod=(ROOT/A.OVERLAY_PATHS["producer_launcher"]).read_text();val=(ROOT/A.OVERLAY_PATHS["validator_launcher"]).read_text();c={"authority_valid":True,"formal_eligible":m["eligible_as_e1_result"] is True,"population":m["candidate_count"]==128 and m["expected_pair_arm_record_count"]==9 and m["expected_directional_term_count"]==36,"dev_one_hour":all("#SBATCH --partition=dev_accelerated" in x and "#SBATCH --time=01:00:00" in x for x in (prod,val)),"checkpoint":("family_commit.json" in val),"no_join_science":m["target_rival_join_authorized"] is False and m["scientific_reduction_authorized"] is False,"outputs_absent":not os.path.lexists(ROOT/A.PRODUCER_NAMESPACE) and not os.path.lexists(ROOT/A.VALIDATION_NAMESPACE)};r={"schema_version":"rc_dino_rcde_track_r_v124_e1_formal_dev_authority_validation_v5_20260826","status":"DINO_RCDE_TRACK_R_V124_E1_FORMAL_DEV_AUTHORITY_V5_INDEPENDENT_VALIDATION_PASS" if all(c.values()) else "DINO_RCDE_TRACK_R_V124_E1_FORMAL_DEV_AUTHORITY_V5_VALIDATION_FAIL","authority_sha256":sha,"checks":c,"scientific_GO_or_NO_GO":None,"next_authorized_stage":None};r["logical_sha256"]=A.logical(r)
 if os.path.lexists(OUT):raise RuntimeError(f"exists {OUT}")
 OUT.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",encoding="ascii");OUT.chmod(0o444);print(json.dumps({"status":r["status"],"output":str(OUT)},sort_keys=True));raise SystemExit(0 if all(c.values()) else 2)
if __name__=="__main__":main()

