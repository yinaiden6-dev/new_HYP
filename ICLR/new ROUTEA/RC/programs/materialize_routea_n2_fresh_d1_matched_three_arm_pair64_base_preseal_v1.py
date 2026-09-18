#!/usr/bin/env python3
"""Materialize the target-free Pair64 fresh-D1 base preseal."""
from __future__ import annotations

from collections import Counter
import hashlib, json, os, tempfile
from pathlib import Path
from typing import Any, Mapping
import torch

ROOT=Path(__file__).resolve().parents[1]
ACTIVE=ROOT/"plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUT_EXECUTION_CONTRACT_V1_20260904.md"
SCOPE=ROOT/"plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_PAIR_SCOPE_CORRECTIVE_ADDENDUM_V1_20260904.md"
LABEL=ROOT/"plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_LABEL_AUTHORITY_CORRECTION_ADDENDUM_V1_20260904.md"
DISPOSITION=ROOT/"plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_ACTIVE_CONTRACT_DISPOSITION_AND_EXECUTION_CLARIFICATION_V1_20260904.md"
CURRENT64=ROOT/"results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1/validation.json"
CURRENT64_SOURCE=ROOT/"results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1/manifest.json"
PAIR_ROOT=ROOT/"results/routea_matched_three_arm_pair64_training_features_v2"
PAIR=PAIR_ROOT/"payload.pt";PAIR_RECEIPT=PAIR_ROOT/"receipt.json";PAIR_LINEAGE=PAIR_ROOT/"post_full_lineage.json";PAIR_VALID=PAIR_ROOT/"independent_validation.json"
BAL1=ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v1/result.json";BAL2=ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v2/result.json"
ROLE_FREE=ROOT/"results/dino_rcde_sr0_mt_role_free_pair_address_v1/role_free_pair_address_manifest.json"
LEDGER=ROOT/"cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json"
OOF_ROOT=ROOT/"results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1";OOF_AGG=OOF_ROOT/"independent_aggregate_validation.json"
GALLERY=ROOT.parents[2]/"colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
REPAIR_REGISTRY=ROOT/"registry/gallery_identity_repair_v1.json";REPAIR_RUNTIME=ROOT/"src/rc_aslo_xf/gallery_identity_repair.py";REPAIR_CONTRACT=ROOT/"protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json"
OUT_ROOT=ROOT/"results/routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1";OUT=OUT_ROOT/"preseal.json"
VERSION="routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1_20260904";READY="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_READY"
EXPECTED={ACTIVE:"bc53ef74ec71283d79247a2117f25e2c708c2d30bb6fda1fad194b5668c97ade",SCOPE:"19fa1ca3e5b898adec22c9148c397d7765af48aa23a5057d155115649fdbe124",LABEL:"9c94f79cb718e7a111a983fa045814028cba1d737553a9c60daa46f118f39035",DISPOSITION:"502de768d26584613bb89f329e214e431d959e821fe6caa3ab5cd1cb5ad538b8",CURRENT64:"24ea4c6e416fc9d5e0a57d22aaa957d5bb3f29e4f01022369a62dd7ff48b7859",PAIR:"d7be701ac4629059d22301c17b0f0d44b69b4b67666072b8fab7c3c0cb7716e3",PAIR_VALID:"657b316a83b80f20e8779bf985b2777b1533126c9b8e7ec7b552ea245febbf1b",BAL1:"77367e63ac835bb38340c1ae9c9d16018fde1c22b16a14a12f346d56201e404a",BAL2:"fd02c73f508f3f16acbafefb5b703dc864a7793ff7f688ff70efffdffa80f108",ROLE_FREE:"6f9999053068a50b0b8f890a11c24987e77532ed31b5361c16c16956b9564e0b",LEDGER:"df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec",OOF_AGG:"fc5ac9422ae6f2fe6485b906525f7ddccd316ba9b77c481bb65f679ea706982f",GALLERY:"11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc",REPAIR_REGISTRY:"9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f",REPAIR_RUNTIME:"995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d",REPAIR_CONTRACT:"867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650"}
SUPERSEDED={"07cff47995dfa838f6465c856b05aadbf787f3902ac09ccab1f17436fbd98ac8","aacf113cdd31902a3a2d0d2cab0df7d91824a4e2b75a37626be6ac57ee20e759"}

class P0Error(RuntimeError):pass
def require(x:bool,m:str)->None:
 if not x:raise P0Error(m)
def sha(p:Path)->str:
 require(p.is_file() and not p.is_symlink(),f"input absent: {p}");h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def canon(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def logical(x:Mapping[str,Any])->str:
 y=dict(x);y.pop("logical_sha256",None);return canon(y)
def read(p:Path)->dict[str,Any]:
 v=json.loads(p.read_text());require(isinstance(v,dict),f"bad json {p}");return v
def thash(x:torch.Tensor)->str:
 t=torch.as_tensor(x).detach().cpu().contiguous();h=hashlib.sha256();h.update(str(t.dtype).encode("ascii"));h.update(str(tuple(t.shape)).encode("ascii"));h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes());return h.hexdigest()
def reduced_rank(scores:torch.Tensor,labels:tuple[str,...])->list[int]:
 seen=set();out=[]
 for row in torch.argsort(scores,descending=True,stable=True).tolist():
  identity=labels[row]
  if identity not in seen:seen.add(identity);out.append(int(row))
 require(len(out)==5412,"corrected ranking population drift");return out
def atomic(p:Path,v:Mapping[str,Any])->None:
 p.parent.mkdir(parents=True,exist_ok=True);s=json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n"
 if p.exists():require(p.read_text()==s,"immutable output drift");return
 fd,n=tempfile.mkstemp(prefix=f".{p.name}.",suffix=".partial",dir=p.parent);q=Path(n)
 try:
  with os.fdopen(fd,"w") as f:f.write(s);f.flush();os.fsync(f.fileno())
  os.link(q,p)
 finally:q.unlink(missing_ok=True)

def corrected_labels()->tuple[tuple[str,...],str]:
 import sys;sys.path.insert(0,str(ROOT/"src"));from rc_aslo_xf.n2_corrected_d1_runtime_v1 import corrected_labels_from_legacy,validate_corrected_identity_axis
 g=torch.load(GALLERY,map_location="cpu",weights_only=False,mmap=True);labels=corrected_labels_from_legacy(g["setids"]);a=validate_corrected_identity_axis(labels);require(a["corrected_axis_sha256"]=="935ce029e3c8177fd9bc4b51f7b41fff8e241c3d587977c31d516b2f300efca4","corrected axis drift");return labels,a["corrected_axis_sha256"]

def build()->dict[str,Any]:
 for p,h in EXPECTED.items():require(sha(p)==h,f"authority hash drift: {p.name}")
 bindings={str(p):h for p,h in EXPECTED.items()};require(len(bindings)==len(EXPECTED) and not(SUPERSEDED&set(bindings.values())),"ambiguous or superseded contract binding")
 cur=read(CURRENT64);require(cur.get("status")=="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATED" and all(cur.get("checks",{}).values()) and cur.get("contract_population_complete") is False and cur.get("pair64_pending_query_count")==64,"current64 prerequisite failed")
 pv=read(PAIR_VALID);pr=read(PAIR_RECEIPT);pl=read(PAIR_LINEAGE);require(pv.get("status")=="ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED" and all(pv.get("checks",{}).values()) and pv.get("v2_payload_sha256")==sha(PAIR) and pv.get("v2_lineage_sha256")==sha(PAIR_LINEAGE) and pv.get("logical_sha256")==logical(pv) and pr.get("status")=="ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_READY" and pr.get("query_count")==64 and pr.get("payload_sha256")==sha(PAIR) and pr.get("logical_sha256")==logical(pr) and pl.get("status")=="ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_POST_FULL_READY" and pl.get("v2_payload_sha256")==sha(PAIR) and pl.get("logical_sha256")==logical(pl),"Pair64 V2 invalid")
 payload=torch.load(PAIR,map_location="cpu",weights_only=False,mmap=True);pair_ids=[str(x["query_id"]) for x in payload["records"]];require(len(pair_ids)==len(set(pair_ids))==64,"Pair64 roster drift")
 b1=read(BAL1);b2=read(BAL2);balanced=[str(x["query_id"]) for x in b1["rows"]]+[str(x["query_id"]) for x in b2["rows"]];require(pair_ids==balanced and len(b1["rows"])==len(b2["rows"])==32,"Balanced order disagreement")
 rf=read(ROLE_FREE);role_free_ids={str(x["query_id"]) for x in rf["pair_manifest"]["records"]};require(set(pair_ids)<=role_free_ids and rf.get("role_free") is True,"role-free roster disagreement")
 ledger=read(LEDGER);specs={str(x["query_id"]):x for x in ledger["queries"]};require(len(specs)==987 and ledger.get("target_label_read") is False and ledger.get("opened_runtime_read_count")==ledger.get("sealed_runtime_read_count")==0,"canonical ledger drift")
 oofa=read(OOF_AGG);require(oofa.get("status")=="ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_PREJOIN_AGGREGATE_VALIDATED" and all(oofa.get("checks",{}).values()) and oofa.get("logical_sha256")==logical(oofa),"OOF aggregate drift")
 seals={int(x["shard"]):x for x in oofa["shards"]};oof={}
 for s in range(16):
  p=OOF_ROOT/f"shard{s:02d}/payload.pt";require(seals[s]["payload_sha256"]==sha(p),"OOF shard drift");z=torch.load(p,map_location="cpu",weights_only=False,mmap=True)
  for i,x in enumerate(z["records"]):
   q=str(x["query_id"])
   if q in pair_ids:oof[q]=(s,i,p,x)
 require(set(oof)==set(pair_ids),"Pair64 OOF join incomplete")
 labels,axis_hash=corrected_labels();current_ids={str(x["query_id"]) for x in read(CURRENT64_SOURCE)["records"]};require(not(set(pair_ids)&current_ids),"Pair64/current64 overlap")
 records=[]
 for order,q in enumerate(pair_ids):
  spec=specs[q];s,i,p,x=oof[q];scores=torch.as_tensor(x["physical_row_scores"]);rank=torch.as_tensor(x["ranked_representative_physical_rows"]);c128=list(map(int,x["natural_c128_representative_physical_rows"].tolist()));ids=[labels[r] for r in c128]
  require(int(spec["query_ordinal"])==int(x["query_ordinal"]) and int(spec["heldout_fold"])==int(x["heldout_fold"]) and tuple(x["grid_shape"])==(int(spec["grid_h"]),int(spec["grid_w"])) and sha(Path(spec["path"]))==spec["source_image_sha256"] and len(c128)==len(set(c128))==len(set(ids))==128 and thash(x["adapted_image_tokens"])==x["adapted_image_tokens_sha256"] and thash(scores)==x["physical_row_scores_sha256"] and thash(rank)==x["ranked_representative_physical_rows_sha256"] and rank.tolist()==reduced_rank(scores,labels),f"canonical/OOF drift {q}")
  frame="EXIF_ORIENTED_BEFORE_RESIZE" if spec["track"]=="new_difficult_train" else "DECODED_RAW_BEFORE_EXIF"
  records.append({"pair64_ordinal":order,"pair_cohort":"BALANCED32_V1" if order<32 else "BALANCED32_V2","query_id":q,"canonical_query_ordinal":int(spec["query_ordinal"]),"canonical_heldout_fold":int(spec["heldout_fold"]),"canonical_track":str(spec["track"]),"query_source_path":str(spec["path"]),"query_source_image_sha256":str(spec["source_image_sha256"]),"query_grid_shape":[int(spec["grid_h"]),int(spec["grid_w"])],"preprocessing_frame":frame,"oof_pointer":{"relative_payload_path":str(p.relative_to(ROOT)),"payload_sha256":sha(p),"source_shard":s,"record_index":i,"record_query_id":q,"record_query_ordinal":int(x["query_ordinal"])},"oof_checkpoint_sha256":str(x["oof_checkpoint_sha256"]),"adapted_image_tokens":{"dtype":str(x["adapted_image_tokens"].dtype),"shape":list(x["adapted_image_tokens"].shape),"sha256":str(x["adapted_image_tokens_sha256"])},"physical_row_scores":{"dtype":str(scores.dtype),"shape":list(scores.shape),"sha256":str(x["physical_row_scores_sha256"])},"corrected_ranking":{"dtype":str(rank.dtype),"shape":list(rank.shape),"sha256":str(x["ranked_representative_physical_rows_sha256"])},"natural_c128":{"representative_physical_rows":c128,"corrected_identities":ids,"sha256":canon([[j,r,ids[j]] for j,r in enumerate(c128)])},"base_winner":{"candidate_position":0,"representative_physical_row":c128[0],"corrected_identity":ids[0],"score":float(scores[c128[0]]),"tie_rule":"score_descending_then_lower_physical_row"}})
 value={"version":VERSION,"status":READY,"claim_level":"TARGET_FREE_PAIR64_FRESH_D1_BASE_POINTER_PRESEAL_NO_MAPS_OR_FEATURES","active_contract":{"disposition_sha256":sha(DISPOSITION),"contract_sha256":sha(ACTIVE),"scope_correction_sha256":sha(SCOPE),"label_correction_sha256":sha(LABEL),"active_contract_count":1,"superseded_contract_hashes_rejected":sorted(SUPERSEDED)},"population":{"query_count":64,"cohort_counts":{"BALANCED32_V1":32,"BALANCED32_V2":32},"fold_counts":{str(k):v for k,v in sorted(Counter(r["canonical_heldout_fold"] for r in records).items())},"current64_overlap_count":0},"records":records,"query_id_order_sha256":canon(pair_ids),"corrected_axis_sha256":axis_hash,"bindings":{**bindings,"pair_receipt_sha256":sha(PAIR_RECEIPT),"pair_lineage_sha256":sha(PAIR_LINEAGE),"current64_source_manifest_sha256":sha(CURRENT64_SOURCE),"producer_sha256":sha(Path(__file__).resolve())},"poison_contract":{"pair64_non_query_fields_semantically_consumed":False,"balanced_non_query_fields_semantically_consumed":False,"role_free_member_read_count":0,"historical_execution_or_inner_fold_consumption_count":0,"query_id_order_mutation_must_abort":True,"canonical_or_oof_mutation_must_abort":True},"access":{"pair64_artifact_deserialization_count":1,"pair64_query_id_read_count":64,"pair64_protected_field_read_count":0,"balanced_query_id_read_count":64,"balanced_protected_field_read_count":0,"role_free_query_id_membership_read_count":64,"role_free_member_read_count":0,"target_identity_read_count":0,"target_position_read_count":0,"supergroup_read_count":0,"outcome_read_count":0,"action_read_count":0,"map_read_count":0,"feature_read_count":0,"model_forward_count":0,"model_update_count":0,"external_read_count":0,"sealed_read_count":0},"scientific_GO_or_NO_GO":None,"ownership_GO_or_NO_GO":None,"automatic_stage_advance":False,"next_authorized_stage":"N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIR_JOIN","logical_sha256":""};value["logical_sha256"]=logical(value);return value
def main()->None:
 v=build();atomic(OUT,v);print(json.dumps({"status":v["status"],"queries":64},sort_keys=True))
if __name__=="__main__":main()
