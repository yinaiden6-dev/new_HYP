#!/usr/bin/env python3
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"programs"))
import run_romav2_colnomic_visibility_xf_six_case_v1 as core  # noqa:E402
from rc_aslo_xf.rgh_full600_source_v1 import RGHFull600SourceLoaderV1  # noqa:E402
from romav2 import RoMaV2  # noqa:E402
SELECTION=(394,396,271,285,359,532,368,517,560,341,357,56,519,563,511,345,66,455,356,520,284,399,461,344,364,518,570,459,358,371,72,75);HELDOUT_FOLD=1;SELECTION_NAMESPACE="ROMA_COLXF_FULLC128_V1";EXCLUDE_EXTRA=set();SOURCE=ROOT/"results/cw0_rgh_xf_v2_p0_a0_manifest_v2/source_manifest.json";GEOMETRY=ROOT/"cache/dino_rcde_colnomic_sr_full600_geometry_v2/full600_geometry_payload_v2.pt";FOLDS=ROOT/"protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json";OUTROOT=ROOT/"results/romav2_colnomic_visibility_xf_full_c128_prejoin_v1"
def main():
 p=argparse.ArgumentParser();p.add_argument("--shard",type=int,required=True);a=p.parse_args();assert 0<=a.shard<4;executions=SELECTION[8*a.shard:8*(a.shard+1)];records=json.loads(SOURCE.read_text())["records"];by={int(x["execution_ordinal"]):x for x in records};eligible=[]
 exclude={480,20,421,324,485,413}|set(EXCLUDE_EXTRA)
 for path in (ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v1/result.json",ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v2/result.json"):
  exclude|={int(x["execution_ordinal"]) for x in json.loads(path.read_text())["selection"]}
 for r in records:
  ex=int(r["execution_ordinal"])
  if int(r["inner_fold"])==HELDOUT_FOLD and ex not in exclude:eligible.append((hashlib.sha256(f"{SELECTION_NAMESPACE}|{r['query_id']}|{ex}".encode()).hexdigest(),ex))
 assert tuple(ex for _,ex in sorted(eligible)[:32])==SELECTION;out=OUTROOT/f"shard{a.shard:02d}/result.json"
 if out.exists():raise RuntimeError("immutable full-C128 shard exists")
 loader=RGHFull600SourceLoaderV1(rc_root=ROOT,geometry_payload_path=GEOMETRY,prejoin_schedule_path=FOLDS);torch.set_float32_matmul_precision("highest");torch.manual_seed(17);model=RoMaV2();rows=[]
 for ex in executions:
  source=loader.load_execution(ex);query=core.oriented(source.query.source_path);candidates=[]
  for pos,entry in enumerate(source.candidates):
   ref=entry.source;pred=model.match(query,core.oriented(ref.source_path));wq=core.cell_means(pred["overlap_AB"][0,...,0].detach().cpu(),source.query.colnomic_geometry);wr=core.cell_means(pred["overlap_BA"][0,...,0].detach().cpu(),ref.colnomic_geometry);real,mass,_=core.score(source.query.tokens,ref.tokens,wq,wr);qc,_,_=core.score(source.query.tokens,ref.tokens,wq.roll(max(1,wq.numel()//2)),wr);rc,_,_=core.score(source.query.tokens,ref.tokens,wq,wr.roll(max(1,wr.numel()//2)));candidates.append({"candidate_position":pos,"physical_row":int(entry.physical_row),"real_score":float(real),"query_control_score":float(qc),"reference_control_score":float(rc),"visibility_mass":float(mass),"query_map_sha256":hashlib.sha256(wq.contiguous().numpy().tobytes()).hexdigest(),"reference_map_sha256":hashlib.sha256(wr.contiguous().numpy().tobytes()).hexdigest()})
  rows.append({"execution_ordinal":ex,"query_id":source.query_id,"candidate_axis_sha256":source.candidate_axis_sha256,"candidate_count":128,"candidates":candidates,"target_role_read_count":0,"target_insertion_count":0});print(json.dumps({"shard":a.shard,"execution":ex,"candidate_count":128},sort_keys=True),flush=True)
 value={"schema_version":"rc_romav2_colnomic_visibility_xf_full_c128_prejoin_shard_v1_20260831","status":"ROMAV2_COLNOMIC_VISIBILITY_XF_FULL_C128_PREJOIN_SHARD_READY","claim_level":"TARGET_FREE_FULL_C128_SCORE_SEAL","shard":a.shard,"execution_ordinals":list(executions),"rows":rows,"target_role_read_count":0,"target_insertion_count":0,"opened_read_count":0,"sealed_read_count":0,"model_update_count":0,"logical_sha256":""};value["logical_sha256"]=core.logical(value);core.atomic(out,value);print(json.dumps({"status":value["status"],"shard":a.shard},sort_keys=True))
if __name__=="__main__":main()
