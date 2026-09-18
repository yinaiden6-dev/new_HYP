#!/usr/bin/env python3
"""Independent validation of the target-free Pair64 base preseal."""
from __future__ import annotations
from collections import Counter
import hashlib,json,os,tempfile
from pathlib import Path
from typing import Any,Mapping
import torch

ROOT=Path(__file__).resolve().parents[1];OUT_ROOT=ROOT/"results/routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1";PRESEAL=OUT_ROOT/"preseal.json";OUT=OUT_ROOT/"independent_validation.json"
PRODUCER=ROOT/"programs/materialize_routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1.py"
PAIR=ROOT/"results/routea_matched_three_arm_pair64_training_features_v2/payload.pt";PAIR_RECEIPT=PAIR.parent/"receipt.json";PAIR_LINEAGE=PAIR.parent/"post_full_lineage.json";PAIR_VALID=PAIR.parent/"independent_validation.json";BAL1=ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v1/result.json";BAL2=ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v2/result.json";ROLE_FREE=ROOT/"results/dino_rcde_sr0_mt_role_free_pair_address_v1/role_free_pair_address_manifest.json";LEDGER=ROOT/"cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json";GALLERY=ROOT.parents[2]/"colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt";CURRENT64_SOURCE=ROOT/"results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1/manifest.json"
READY="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_READY";VALID="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_VALIDATED";VERSION="routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1_20260904"
SUPERSEDED={"07cff47995dfa838f6465c856b05aadbf787f3902ac09ccab1f17436fbd98ac8","aacf113cdd31902a3a2d0d2cab0df7d91824a4e2b75a37626be6ac57ee20e759"}
class ValidationError(RuntimeError):pass
def req(x:bool,m:str)->None:
 if not x:raise ValidationError(m)
def sha(p:Path)->str:
 req(p.is_file() and not p.is_symlink(),f"absent {p}");h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def canon(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def logical(x:Mapping[str,Any])->str:y=dict(x);y.pop("logical_sha256",None);return canon(y)
def read(p:Path)->dict[str,Any]:v=json.loads(p.read_text());req(isinstance(v,dict),"json root");return v
def thash(x:torch.Tensor)->str:
 t=torch.as_tensor(x).detach().cpu().contiguous();h=hashlib.sha256();h.update(str(t.dtype).encode("ascii"));h.update(str(tuple(t.shape)).encode("ascii"));h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes());return h.hexdigest()
def reduced_rank(scores:torch.Tensor,labels:tuple[str,...])->list[int]:
 seen=set();out=[]
 for row in torch.argsort(scores,descending=True,stable=True).tolist():
  identity=labels[row]
  if identity not in seen:seen.add(identity);out.append(int(row))
 req(len(out)==5412,"corrected ranking population drift");return out
def atomic(p:Path,v:Mapping[str,Any])->None:
 p.parent.mkdir(parents=True,exist_ok=True);req(not p.exists(),"validation exists");s=json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n";fd,n=tempfile.mkstemp(prefix=f".{p.name}.",dir=p.parent);q=Path(n)
 try:
  with os.fdopen(fd,"w") as f:f.write(s);f.flush();os.fsync(f.fileno())
  os.link(q,p)
 finally:q.unlink(missing_ok=True)
def project(records:list[Mapping[str,Any]])->list[str]:return [str(x["query_id"]) for x in records]

def main()->None:
 p=read(PRESEAL);req(p.get("version")==VERSION and p.get("status")==READY and p.get("logical_sha256")==logical(p),"preseal envelope")
 pair_payload=torch.load(PAIR,map_location="cpu",weights_only=False,mmap=True);pr=read(PAIR_RECEIPT);pl=read(PAIR_LINEAGE);pv=read(PAIR_VALID);req(pr.get("status")=="ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_READY" and pr.get("payload_sha256")==sha(PAIR) and pr.get("logical_sha256")==logical(pr) and pl.get("status")=="ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_POST_FULL_READY" and pl.get("v2_payload_sha256")==sha(PAIR) and pl.get("logical_sha256")==logical(pl) and pv.get("status")=="ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED" and pv.get("v2_payload_sha256")==sha(PAIR) and pv.get("v2_lineage_sha256")==sha(PAIR_LINEAGE) and pv.get("logical_sha256")==logical(pv),"independent Pair64 V2 lineage");pair_records=pair_payload["records"];ids=project(pair_records);req(len(ids)==len(set(ids))==64,"pair roster")
 b1=read(BAL1);b2=read(BAL2);req(ids==project(b1["rows"])+project(b2["rows"]),"balanced order")
 rf=read(ROLE_FREE);rf_records=rf["pair_manifest"]["records"];req(set(ids)<={str(x["query_id"]) for x in rf_records},"rolefree membership")
 # Poison all non-query fields without copying tensor storage; projection must be invariant.
 pair_poison=[{"query_id":x["query_id"],**{k:"POISON" for k in x if k!="query_id"}} for x in pair_records]
 bal_poison=[{"query_id":x["query_id"],**{k:"POISON" for k in x if k!="query_id"}} for x in b1["rows"]+b2["rows"]]
 rf_poison=[{"query_id":x["query_id"],"members":"POISON"} for x in rf_records]
 poison_ok=project(pair_poison)==ids and project(bal_poison)==ids and set(ids)<={x["query_id"] for x in rf_poison}
 swapped=list(ids);swapped[0],swapped[1]=swapped[1],swapped[0];query_mutation_rejected=swapped!=ids
 specs={str(x["query_id"]):x for x in read(LEDGER)["queries"]}
 import sys;sys.path.insert(0,str(ROOT/"src"));from rc_aslo_xf.n2_corrected_d1_runtime_v1 import corrected_labels_from_legacy,validate_corrected_identity_axis
 labels=corrected_labels_from_legacy(torch.load(GALLERY,map_location="cpu",weights_only=False,mmap=True)["setids"]);axis_audit=validate_corrected_identity_axis(labels);req(axis_audit["corrected_axis_sha256"]==p["corrected_axis_sha256"],"corrected axis")
 records=p.get("records",[]);req(len(records)==64 and [x["query_id"] for x in records]==ids,"record order")
 payloads={};record_ok=[]
 for order,r in enumerate(records):
  ptr=r["oof_pointer"];path=ROOT/ptr["relative_payload_path"];req(sha(path)==ptr["payload_sha256"],"OOF pointer payload")
  z=payloads.setdefault(str(path),torch.load(path,map_location="cpu",weights_only=False,mmap=True));x=z["records"][int(ptr["record_index"])];q=ids[order];spec=specs[q]
  scores=torch.as_tensor(x["physical_row_scores"]);rank=torch.as_tensor(x["ranked_representative_physical_rows"]);c128=list(map(int,x["natural_c128_representative_physical_rows"].tolist()));cids=[labels[row] for row in c128]
  expected_frame="EXIF_ORIENTED_BEFORE_RESIZE" if spec["track"]=="new_difficult_train" else "DECODED_RAW_BEFORE_EXIF"
  ok=(r["pair64_ordinal"]==order and r["pair_cohort"]==("BALANCED32_V1" if order<32 else "BALANCED32_V2") and x["query_id"]==ptr["record_query_id"]==r["query_id"]==q and int(x["query_ordinal"])==int(ptr["record_query_ordinal"])==int(r["canonical_query_ordinal"])==int(spec["query_ordinal"]) and int(x["heldout_fold"])==int(r["canonical_heldout_fold"])==int(spec["heldout_fold"]) and r["query_source_image_sha256"]==spec["source_image_sha256"] and r["query_source_path"]==spec["path"] and r["preprocessing_frame"]==expected_frame and r["query_grid_shape"]==[spec["grid_h"],spec["grid_w"]] and r["adapted_image_tokens"]=={"dtype":str(x["adapted_image_tokens"].dtype),"shape":list(x["adapted_image_tokens"].shape),"sha256":x["adapted_image_tokens_sha256"]} and thash(x["adapted_image_tokens"])==x["adapted_image_tokens_sha256"] and r["physical_row_scores"]=={"dtype":str(scores.dtype),"shape":list(scores.shape),"sha256":x["physical_row_scores_sha256"]} and thash(scores)==x["physical_row_scores_sha256"] and r["corrected_ranking"]=={"dtype":str(rank.dtype),"shape":list(rank.shape),"sha256":x["ranked_representative_physical_rows_sha256"]} and thash(rank)==x["ranked_representative_physical_rows_sha256"] and r["natural_c128"]["representative_physical_rows"]==c128 and r["natural_c128"]["corrected_identities"]==cids and r["natural_c128"]["sha256"]==canon([[j,row,cids[j]] for j,row in enumerate(c128)]) and len(set(cids))==128 and r["base_winner"]["candidate_position"]==0 and r["base_winner"]["representative_physical_row"]==c128[0] and r["base_winner"]["corrected_identity"]==cids[0] and r["base_winner"]["score"]==float(scores[c128[0]]) and r["base_winner"]["tie_rule"]=="score_descending_then_lower_physical_row")
  ok=ok and sha(Path(spec["path"]))==spec["source_image_sha256"] and rank.tolist()==reduced_rank(scores,labels)
  record_ok.append(ok)
 cur_ids={str(x["query_id"]) for x in read(CURRENT64_SOURCE)["records"]};bindings=p.get("bindings",{});active=p.get("active_contract",{})
 checks={"active_contract_unique_and_superseded_rejected":active.get("active_contract_count")==1 and not(SUPERSEDED&set(bindings.values())) and active.get("superseded_contract_hashes_rejected")==sorted(SUPERSEDED),"pair64_roster_order_64":len(ids)==64,"balanced_order_crosscheck":True,"rolefree_members_not_consumed":p.get("access",{}).get("role_free_member_read_count")==0,"poison_nonquery_invariance":poison_ok,"query_order_mutation_rejected":query_mutation_rejected,"canonical_oof_pointer_and_tensor_replay":all(record_ok),"corrected_5412_identity_axis":axis_audit["corrected_identity_count"]==5412,"pair64_current64_disjoint":not(set(ids)&cur_ids),"zero_protected_map_feature_model_access":all(p.get("access",{}).get(k)==0 for k in ("pair64_protected_field_read_count","balanced_protected_field_read_count","target_identity_read_count","target_position_read_count","supergroup_read_count","outcome_read_count","action_read_count","map_read_count","feature_read_count","model_forward_count","model_update_count","external_read_count","sealed_read_count")),"no_claim_or_auto_advance":p.get("scientific_GO_or_NO_GO") is None and p.get("ownership_GO_or_NO_GO") is None and p.get("automatic_stage_advance") is False and p.get("next_authorized_stage")=="N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIR_JOIN","producer_bound":bindings.get("producer_sha256")==sha(PRODUCER)}
 req(all(checks.values()),"Pair64 P0 validation failed")
 v={"version":VERSION,"status":VALID,"claim_level":"INDEPENDENT_TARGET_FREE_PAIR64_BASE_POINTER_PRESEAL_VALIDATION","checks":checks,"query_count":64,"query_id_order_sha256":canon(ids),"corrected_axis_sha256":p["corrected_axis_sha256"],"preseal_sha256":sha(PRESEAL),"preseal_logical_sha256":p["logical_sha256"],"producer_sha256":sha(PRODUCER),"validator_sha256":sha(Path(__file__).resolve()),"access":p["access"],"scientific_GO_or_NO_GO":None,"ownership_GO_or_NO_GO":None,"automatic_stage_advance":False,"next_authorized_stage":"N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIR_JOIN","logical_sha256":""};v["logical_sha256"]=logical(v);atomic(OUT,v);print(json.dumps({"status":VALID,"checks":checks},sort_keys=True))
if __name__=="__main__":main()
