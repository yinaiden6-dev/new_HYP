#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"programs"))
import run_romav2_colnomic_visibility_xf_six_case_v1 as core  # noqa:E402
from rc_aslo_xf.rgh_full600_source_v1 import RGHFull600SourceLoaderV1  # noqa:E402
from rc_aslo_xf.rgh_v9_frozen_colnomic_base_v1 import FrozenColNomicBaseV1  # noqa:E402
from romav2 import RoMaV2  # noqa:E402
SOURCE=ROOT/"results/cw0_rgh_xf_v2_p0_a0_manifest_v2/source_manifest.json";ROLE=ROOT/"results/cw0_rgh_xf_v2_p0_a0_manifest_v2/role_shards";GEOMETRY=ROOT/"cache/dino_rcde_colnomic_sr_full600_geometry_v2/full600_geometry_payload_v2.pt";FOLDS=ROOT/"protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json";PRE=ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v1/result.json";OUT=ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v2/result.json";SIX={480,20,421,324,485,413}
def selection(records,base,exclude):
 strata={True:[],False:[]}
 for r in records:
  ex=int(r["execution_ordinal"])
  if int(r["inner_fold"])!=4 or ex in exclude:continue
  role=json.loads((ROLE/f"role_exec{ex:03d}.json").read_text())
  if role.get("target_naturally_present") is not True:continue
  scores=base.scores(ex,r["candidate_physical_rows"]);target=int(role["target_candidate_position"]);rank=sorted(range(128),key=lambda i:(-float(scores[i]),int(r["candidate_physical_rows"][i]))).index(target)+1;key=hashlib.sha256(f"ROMA_COLXF_BAL32_V2|{r['query_id']}|{ex}".encode()).hexdigest();strata[rank==1].append((key,ex,target,rank))
 return tuple(sorted(strata[False])[:16]+sorted(strata[True])[:16])
def derange(weights):return weights.roll(max(1,weights.numel()//2))
def main():
 if OUT.exists():raise RuntimeError("immutable V2 output exists")
 predecessor=json.loads(PRE.read_text());assert predecessor["status"]=="ROMAV2_COLNOMIC_VISIBILITY_XF_BALANCED32_NO_GO";records=json.loads(SOURCE.read_text())["records"];exclude=SIX|{int(x["execution_ordinal"]) for x in predecessor["selection"]};base=FrozenColNomicBaseV1(ROOT);chosen=selection(records,base,exclude);assert len(chosen)==32 and sum(x[3]>1 for x in chosen)==16;loader=RGHFull600SourceLoaderV1(rc_root=ROOT,geometry_payload_path=GEOMETRY,prejoin_schedule_path=FOLDS);torch.set_float32_matmul_precision("highest");torch.manual_seed(17);model=RoMaV2();rows=[]
 for _,ex,target,rank in chosen:
  source=loader.load_execution(ex);raw=base.scores(ex,[x.physical_row for x in source.candidates]);competitor=sorted((i for i in range(128) if i!=target),key=lambda i:(-float(raw[i]),int(source.candidates[i].physical_row)))[0];query=core.oriented(source.query.source_path);candidates=[]
  for role,pos in (("TARGET",target),("COMPETITOR",competitor)):
   ref=source.candidates[pos].source;pred=model.match(query,core.oriented(ref.source_path));wq=core.cell_means(pred["overlap_AB"][0,...,0].detach().cpu(),source.query.colnomic_geometry);wr=core.cell_means(pred["overlap_BA"][0,...,0].detach().cpu(),ref.colnomic_geometry);real,mass,_=core.score(source.query.tokens,ref.tokens,wq,wr);qc,_,_=core.score(source.query.tokens,ref.tokens,derange(wq),wr);rc,_,_=core.score(source.query.tokens,ref.tokens,wq,derange(wr));candidates.append({"role":role,"candidate_position":pos,"physical_row":int(source.candidates[pos].physical_row),"real_score":float(real),"query_control_score":float(qc),"reference_control_score":float(rc),"visibility_mass":float(mass),"query_map_sha256":hashlib.sha256(wq.contiguous().numpy().tobytes()).hexdigest(),"reference_map_sha256":hashlib.sha256(wr.contiguous().numpy().tobytes()).hexdigest(),"query_control_shift":max(1,wq.numel()//2),"reference_control_shift":max(1,wr.numel()//2)})
  t,c=candidates;rm=t["real_score"]-c["real_score"];qm=t["query_control_score"]-c["query_control_score"];xm=t["reference_control_score"]-c["reference_control_score"];rows.append({"execution_ordinal":ex,"query_id":source.query_id,"base_rank":rank,"base_correct":rank==1,"candidates":candidates,"real_margin":rm,"query_control_margin":qm,"reference_control_margin":xm,"local_correct":rm>0,"real_gt_query_control":rm>qm,"real_gt_reference_control":rm>xm,"maps_distinct":t["query_map_sha256"]!=c["query_map_sha256"] and t["reference_map_sha256"]!=c["reference_map_sha256"]})
 wrong=[x for x in rows if not x["base_correct"]];correct=[x for x in rows if x["base_correct"]];rescue=sum(x["local_correct"] for x in wrong);brk=sum(not x["local_correct"] for x in correct);pair=sum(x["local_correct"] for x in rows);qdrop=sum(x["real_gt_query_control"] for x in rows);rdrop=sum(x["real_gt_reference_control"] for x in rows);distinct=sum(x["maps_distinct"] for x in rows);gates={"population":len(rows)==32 and len(wrong)==len(correct)==16,"rescues":rescue>=11,"breaks":brk<=1,"pair_accuracy":pair>=26,"query_spatial":qdrop>=20,"reference_spatial":rdrop>=20,"candidate_binding":distinct==32};passed=all(gates.values());value={"schema_version":"rc_romav2_colnomic_visibility_xf_balanced32_v2_20260831","status":"ROMAV2_COLNOMIC_VISIBILITY_XF_BALANCED32_V2_GO" if passed else "ROMAV2_COLNOMIC_VISIBILITY_XF_BALANCED32_V2_NO_GO","claim_level":"FRESH_POSTJOIN_BALANCED_PAIR_SPATIAL_CONTROL_REPAIR_ONLY","contract_sha256":core.sha(ROOT/"plan/ROMAV2_COLNOMIC_VISIBILITY_XF_BALANCED32_V2_SPATIAL_CONTROL_CONTRACT_20260831.md"),"predecessor_sha256":core.sha(PRE),"selection_namespace":"ROMA_COLXF_BAL32_V2","excluded_executions":sorted(exclude),"selection":[{"execution_ordinal":x[1],"target_position":x[2],"base_rank":x[3]} for x in chosen],"summary":{"rescue":rescue,"break":brk,"pair_correct":pair,"query_spatial_real_gt_control":qdrop,"reference_spatial_real_gt_control":rdrop,"candidate_maps_distinct":distinct},"gates":gates,"rows":rows,"model_update_count":0,"training_count":0,"full_c128_scoring_count":0,"P0_authorized":False,"scientific_GO_or_NO_GO":None,"next_authorized_stage":"ROMAV2_COLNOMIC_VISIBILITY_XF_FULL_C128_GATE" if passed else None,"logical_sha256":""};value["logical_sha256"]=core.logical(value);core.atomic(OUT,value);print(json.dumps({"status":value["status"],"summary":value["summary"],"gates":gates},sort_keys=True))
if __name__=="__main__":main()

