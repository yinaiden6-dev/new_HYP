#!/usr/bin/env python3
"""Independently aggregate the eight Pair64 training-feature shard seals."""
from __future__ import annotations

from collections import Counter
import hashlib, json, os, tempfile
from pathlib import Path
from typing import Any, Mapping
import torch

ROOT=Path(__file__).resolve().parents[1]
SHARD_ROOT=ROOT/"results/routea_n2_fresh_d1_matched_three_arm_pair64_training_features_v1"
CURRENT64=ROOT/"results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1/validation.json"
P0_ROOT=ROOT/"results/routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1"
P0=P0_ROOT/"preseal.json";P0_VALIDATION=P0_ROOT/"independent_validation.json"
J0_ROOT=ROOT/"results/routea_n2_fresh_d1_matched_three_arm_pair64_postseal_fixed_pairs_v1"
J0=J0_ROOT/"fixed_pairs.json";J0_VALIDATION=J0_ROOT/"independent_validation.json"
P0_PRODUCER=ROOT/"programs/materialize_routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1.py";P0_VALIDATOR=ROOT/"programs/validate_routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1.py"
J0_PRODUCER=ROOT/"programs/materialize_routea_n2_fresh_d1_pair64_postseal_fixed_pairs_v1.py";J0_VALIDATOR=ROOT/"programs/validate_routea_n2_fresh_d1_pair64_postseal_fixed_pairs_v1.py"
M0_PRODUCER=ROOT/"programs/materialize_routea_n2_fresh_d1_matched_three_arm_pair64_training_features_shard_v1.py"
M0_VALIDATOR=ROOT/"programs/validate_routea_n2_fresh_d1_matched_three_arm_pair64_training_features_shard_v1.py"
OUT=SHARD_ROOT/"independent_aggregate_validation.json"
READY="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURE_SHARD_READY"
VALID="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURE_SHARD_VALIDATED"
STATUS="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUTS_VALIDATED"
NEXT="N2_FRESH_D1_MATCHED_THREE_ARM_SEVEN_PARAMETER_CROSSFIT_CONTRACT"
READY_NEXT="N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURE_SHARD_INDEPENDENT_VALIDATION"
VALID_NEXT="N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURE_AGGREGATE_VALIDATION"
EXPECTED={P0_PRODUCER:"068c8297a0b773c4efefa093f2887393588213dce7ca347cb5d6517cc22e13b5",P0_VALIDATOR:"239f67385076b72b3f28c4223a5c4208f7ee64ba90c22008d618aaa627adf6db",J0_PRODUCER:"5ef394b3cbc1688b8315b7ee7c2ad625a29f255b82d1018e48c4d4a3f5267fd4",J0_VALIDATOR:"a536c5fdcc37834dd78a6cc8b18a7f13d76c14091db625533059291e88ad8c16",M0_PRODUCER:"3e77fe479c9ccf1c4c7ca60a4c01e6b0947f7717609ba431755979d3e630fc2e",M0_VALIDATOR:"d8de67ccc9e5493157fa6aa64c57572927a72bda473186e2ef0eb620a8c6e415",CURRENT64:"24ea4c6e416fc9d5e0a57d22aaa957d5bb3f29e4f01022369a62dd7ff48b7859"}
CHECK_KEYS={"payload_scope_and_schema","input_bindings","access_boundary","record_population_and_formula","fresh_roma_replay","protected_field_poison_invariance","switch_label_geometry_and_feature_invariance","no_target_group_state_output","no_cbind_fullnegative_or_historical_maps","claim_boundary","receipt"}
FORBIDDEN={"target_identity","target_supergroup","target_state","target_naturally_in_fresh_c128","target_candidate_position","target_physical_row_provenance","n2_label_source_folds","corrected_identity"}

class AggregateError(RuntimeError): pass
def req(x:bool,m:str)->None:
 if not x: raise AggregateError(m)
def sha(p:Path)->str:
 req(p.is_file() and not p.is_symlink(),f"missing/nonregular {p}");h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def canon(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def logical(x:Mapping[str,Any])->str:
 y=dict(x);y.pop("logical_sha256",None);return canon(y)
def read(p:Path)->dict[str,Any]:
 x=json.loads(p.read_text());req(isinstance(x,dict),f"root {p}");return x
def walk_keys(x:Any)->set[str]:
 if isinstance(x,dict):return set(map(str,x))|set().union(*(walk_keys(v) for v in x.values()),set())
 if isinstance(x,(list,tuple)):return set().union(*(walk_keys(v) for v in x),set())
 return set()
def atomic(p:Path,x:Mapping[str,Any])->None:
 p.parent.mkdir(parents=True,exist_ok=True);s=json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n"
 if p.exists():req(p.is_file() and not p.is_symlink() and p.read_text()==s,f"immutable output drift {p}");return
 fd,n=tempfile.mkstemp(prefix=f".{p.name}.",suffix=".partial",dir=p.parent)
 try:
  with os.fdopen(fd,"w") as f:f.write(s);f.flush();os.fsync(f.fileno())
  os.link(n,p)
 finally:Path(n).unlink(missing_ok=True)

def main()->None:
 for p,h in EXPECTED.items():req(sha(p)==h,f"fixed hash drift {p}")
 cur=read(CURRENT64);req(cur.get("status")=="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATED" and all(cur.get("checks",{}).values()) and cur.get("logical_sha256")==logical(cur) and cur.get("current64_query_count")==64 and cur.get("pair64_pending_query_count")==64 and cur.get("contract_population_complete") is False,"current64 boundary")
 p0,p0v,j0,j0v=map(read,(P0,P0_VALIDATION,J0,J0_VALIDATION));p0sha=sha(P0);j0sha=sha(J0)
 req(p0.get("status")=="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_READY" and p0.get("logical_sha256")==logical(p0) and p0v.get("status")=="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_VALIDATED" and p0v.get("preseal_sha256")==p0sha and p0v.get("preseal_logical_sha256")==p0["logical_sha256"] and p0v.get("producer_sha256")==EXPECTED[P0_PRODUCER] and p0v.get("validator_sha256")==EXPECTED[P0_VALIDATOR] and p0v.get("logical_sha256")==logical(p0v) and all(p0v.get("checks",{}).values()),"P0 physical/logical binding")
 req(j0.get("status")=="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIRS_READY" and j0.get("logical_sha256")==logical(j0) and j0.get("p0_hash_before_target_join")==j0.get("p0_hash_after_target_join")==p0sha and j0v.get("status")=="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIRS_VALIDATED" and j0v.get("fixed_pairs_sha256")==j0sha and j0v.get("fixed_pairs_logical_sha256")==j0["logical_sha256"] and j0v.get("producer_sha256")==EXPECTED[J0_PRODUCER] and j0v.get("validator_sha256")==EXPECTED[J0_VALIDATOR] and j0v.get("logical_sha256")==logical(j0v) and all(j0v.get("checks",{}).values()) and j0v.get("checks",{}).get("pair64_current_eval32_query_identity_supergroup_disjoint") is True,"J0 physical/logical/disjoint binding")
 dis=j0v.get("pair64_current_eval32_disjointness",{});req(dis.get("pair64_query_count")==64 and dis.get("current_eval32_query_count")==32 and dis.get("pair64_distinct_target_identity_count")==dis.get("pair64_distinct_supergroup_count")==20 and dis.get("current_eval32_distinct_target_identity_count")==dis.get("current_eval32_distinct_supergroup_count")==11 and dis.get("query_id_overlap_count")==dis.get("target_identity_overlap_count")==dis.get("supergroup_overlap_count")==0 and all(isinstance(dis.get(k),str) and len(dis[k])==64 for k in ("pair64_target_identity_sha256","pair64_supergroup_sha256","current_eval32_target_identity_sha256","current_eval32_supergroup_sha256")),"J0 disjointness exact structure")
 seals=[];records=[];switch=hold=maps=0;all_checks=True
 for s in range(8):
  d=SHARD_ROOT/f"shard{s:02d}";pp=d/"payload.pt";rp=d/"receipt.json";vp=d/"validation.json"
  r,v=read(rp),read(vp);payload=torch.load(pp,map_location="cpu",weights_only=False,mmap=True)
  req(payload.get("status")==READY and r.get("status")==READY and v.get("status")==VALID,f"status shard{s}")
  req(payload.get("bindings",{}).get("p0_preseal_sha256")==p0sha and payload.get("bindings",{}).get("j0_fixed_pairs_sha256")==j0sha,"P0/J0 shard binding")
  req(payload.get("query_count")==payload.get("fixed_pair_count")==8 and payload.get("selected_endpoint_count")==payload.get("roma_pair_evaluation_count")==16 and v.get("query_count")==v.get("fixed_pair_count")==8 and v.get("selected_endpoint_count")==v.get("roma_pair_evaluation_count")==16,"shard exact totals")
  req(payload.get("access",{}).get("model_update_count")==v.get("access",{}).get("model_update_count")==0 and payload.get("access",{}).get("roma_pair_evaluation_count")==v.get("access",{}).get("roma_pair_evaluation_count")==16,"shard access totals")
  req(v.get("payload_sha256")==sha(pp) and v.get("receipt_sha256")==sha(rp) and v.get("producer_sha256")==EXPECTED[M0_PRODUCER] and v.get("validator_sha256")==EXPECTED[M0_VALIDATOR] and v.get("logical_sha256")==logical(v),f"seal shard{s}")
  req(set(v.get("checks",{}))==CHECK_KEYS and all(v["checks"].values()),f"checks shard{s}")
  req(payload.get("shard")==r.get("shard")==v.get("shard")==s and payload.get("shard_count")==8 and payload.get("next_authorized_stage")==r.get("next_authorized_stage")==READY_NEXT and v.get("next_authorized_stage")==VALID_NEXT,f"envelope shard{s}")
  rs=payload.get("records",[]);req(len(rs)==8 and not(FORBIDDEN&walk_keys(rs)),f"record scope shard{s}")
  for x in rs:
   req(set(x.get("native6_features",{}))=={"A_ALL","B_QUERY","C_PAIRED"} and all(tuple(torch.as_tensor(z).shape)==(1,6) and bool(torch.isfinite(torch.as_tensor(z)).all()) for z in x["native6_features"].values()),"native6 shape")
   req(x.get("roma_pair_evaluation_count")==2 and x.get("historical_postjoin_map_reuse_count")==x.get("cbind_feature_count")==x.get("fullnegative_feature_count")==x.get("j0_protected_field_semantic_read_count")==0,"record access")
  records.extend(rs);switch+=int(v["switch_label_count"]);hold+=int(v["hold_label_count"]);maps+=int(v["roma_pair_evaluation_count"])
  req(v.get("cbind_feature_count")==v.get("fullnegative_feature_count")==v.get("historical_postjoin_map_reuse_count")==0,"forbidden features")
  seals.append({"shard":s,"payload_sha256":sha(pp),"receipt_sha256":sha(rp),"validation_sha256":sha(vp),"validation_logical_sha256":v["logical_sha256"]})
 ords=[int(x["pair64_ordinal"]) for x in records];q=[str(x["query_id"]) for x in records];j0q=[str(x["query_id"]) for x in j0.get("records",[])]
 checks={"current64_validated_and_pair64_pending_boundary":True,"p0_j0_physical_logical_and_disjointness_bound":j0v.get("checks",{}).get("pair64_current_eval32_query_identity_supergroup_disjoint") is True,"eight_independently_validated_shards":len(seals)==8,"pair64_records_complete_in_exact_j0_order":ords==list(range(64)) and len(set(q))==64 and q==j0q,"exact_64_pairs_128_endpoints_maps":len(records)==64 and maps==128,"fixed_37_hold_27_switch":hold==37 and switch==27,"native_abc_each_1x6_finite":True,"no_target_group_state_or_candidate_identity_output":not(FORBIDDEN&walk_keys(records)),"no_cbind_fullnegative_or_historical_maps":True,"no_model_update":all(x.get("access",{}).get("model_update_count")==0 for x in [torch.load(SHARD_ROOT/f"shard{s:02d}/payload.pt",map_location="cpu",weights_only=False,mmap=True) for s in range(8)]),"no_scientific_claim_or_auto_advance":all(x.get("scientific_GO_or_NO_GO") is None and x.get("ownership_GO_or_NO_GO") is None and x.get("automatic_stage_advance") is False for x in [cur]+[read(SHARD_ROOT/f"shard{s:02d}/validation.json") for s in range(8)])}
 req(all(checks.values()),"aggregate checks")
 out={"version":"routea_n2_fresh_d1_matched_three_arm_pair64_training_features_aggregate_v1_20260904","status":STATUS,"claim_level":"INDEPENDENT_TRAINING_ONLY_PAIR64_FEATURE_AGGREGATE_NO_SCIENTIFIC_CLAIM","checks":checks,"population":{"query_count":64,"fixed_pair_count":64,"selected_endpoint_count":128,"roma_pair_evaluation_count":128,"switch_label_count":27,"hold_label_count":37},"query_id_order_sha256":canon(q),"record_sequence_sha256":canon([[x["pair64_ordinal"],x["query_id"],x["geometry_projection_sha256"]] for x in records]),"bindings":{"current64_validation_sha256":sha(CURRENT64),"current64_validation_logical_sha256":cur["logical_sha256"],"p0_preseal_sha256":p0sha,"p0_preseal_logical_sha256":p0["logical_sha256"],"p0_validation_sha256":sha(P0_VALIDATION),"p0_validation_logical_sha256":p0v["logical_sha256"],"p0_producer_sha256":sha(P0_PRODUCER),"p0_validator_sha256":sha(P0_VALIDATOR),"j0_fixed_pairs_sha256":j0sha,"j0_fixed_pairs_logical_sha256":j0["logical_sha256"],"j0_validation_sha256":sha(J0_VALIDATION),"j0_validation_logical_sha256":j0v["logical_sha256"],"j0_producer_sha256":sha(J0_PRODUCER),"j0_validator_sha256":sha(J0_VALIDATOR),"j0_pair64_current_eval32_disjointness_sha256":canon(dis),"j0_pair64_current_eval32_overlap_counts":{"query_id":dis["query_id_overlap_count"],"target_identity":dis["target_identity_overlap_count"],"supergroup":dis["supergroup_overlap_count"]},"m0_producer_sha256":sha(M0_PRODUCER),"m0_validator_sha256":sha(M0_VALIDATOR)},"shards":seals,"access":{"target_identity_read_count":0,"target_supergroup_read_count":0,"target_state_read_count":0,"corrected_identity_output_count":0,"roma_pair_evaluation_count":128,"cbind_feature_count":0,"fullnegative_feature_count":0,"historical_postjoin_map_reuse_count":0,"model_update_count":0,"external_read_count":0,"sealed_read_count":0},"scientific_GO_or_NO_GO":None,"ownership_GO_or_NO_GO":None,"automatic_stage_advance":False,"next_authorized_stage":NEXT,"logical_sha256":""};out["logical_sha256"]=logical(out);atomic(OUT,out);print(json.dumps({"status":STATUS,"checks":checks},sort_keys=True))
if __name__=="__main__":main()
