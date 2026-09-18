#!/usr/bin/env python3
"""Independent validation of the Pair64 postseal fixed-pair manifest.

The producer is never imported.  This validator independently reconstructs
the N2 fourfold target consensus, corrected-identity branch rule, and CW0
identity/supergroup cross-check without consuming any CW0 position or axis.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUT_EXECUTION_CONTRACT_V1_20260904.md"
SCOPE = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_PAIR_SCOPE_CORRECTIVE_ADDENDUM_V1_20260904.md"
LABEL_CORRECTION = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_LABEL_AUTHORITY_CORRECTION_ADDENDUM_V1_20260904.md"
DISPOSITION = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_ACTIVE_CONTRACT_DISPOSITION_AND_EXECUTION_CLARIFICATION_V1_20260904.md"
P0_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1"
P0 = P0_ROOT / "preseal.json"
P0_VALIDATION = P0_ROOT / "independent_validation.json"
P0_PRODUCER = ROOT / "programs/materialize_routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1.py"
P0_VALIDATOR = ROOT / "programs/validate_routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1.py"
CURRENT64_SOURCE_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1"
CURRENT64_SOURCE = CURRENT64_SOURCE_ROOT / "manifest.json"
CURRENT64_SOURCE_VALIDATION = CURRENT64_SOURCE_ROOT / "independent_validation.json"
LABEL_ROOT = ROOT / "results/routea_matched_three_arm_n2_d1_label_join_authority_v1"
LABEL_RESULT = LABEL_ROOT / "result.json"
LABEL_VALIDATION = LABEL_ROOT / "independent_validation.json"
CW_ROOT = ROOT / "results/cw0_rgh_xf_v2_p0_a0_manifest_v2"
CW_SOURCE = CW_ROOT / "source_manifest.json"
CW_ROLE = CW_ROOT / "role_manifest.json"
CW_RESULT = CW_ROOT / "result.json"
CW_VALIDATION = CW_ROOT / "independent_validation.json"
GALLERY = ROOT.parents[2] / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
PRODUCER = ROOT / "programs/materialize_routea_n2_fresh_d1_pair64_postseal_fixed_pairs_v1.py"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_pair64_postseal_fixed_pairs_v1"
FIXED_PAIRS = OUT_ROOT / "fixed_pairs.json"
OUT = OUT_ROOT / "independent_validation.json"

VERSION = "routea_n2_fresh_d1_matched_three_arm_pair64_postseal_fixed_pairs_v1_20260904"
READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIRS_READY"
VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIRS_VALIDATED"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_FRESH_ROMA_PAIR_FEATURES"
EXPECTED_BRANCH_COUNTS = {"TARGET_WINNER_NEGATIVE_HOLD": 37, "TARGET_NONWINNER_POSITIVE_SWITCH": 27, "TARGET_ABSENT_NEGATIVE_HOLD": 0}
EXPECTED_HASHES = {
    ACTIVE: "bc53ef74ec71283d79247a2117f25e2c708c2d30bb6fda1fad194b5668c97ade",
    SCOPE: "19fa1ca3e5b898adec22c9148c397d7765af48aa23a5057d155115649fdbe124",
    LABEL_CORRECTION: "9c94f79cb718e7a111a983fa045814028cba1d737553a9c60daa46f118f39035",
    DISPOSITION: "502de768d26584613bb89f329e214e431d959e821fe6caa3ab5cd1cb5ad538b8",
    LABEL_RESULT: "519cea43968abf8083f3d6c4b98e108d592f29976c470b01f464054d7c9a5861",
    LABEL_VALIDATION: "0b78e01d6940c19db40d606ce6012be190628b31d56e71851c22a97e83cfe3a8",
    CW_SOURCE: "e0be35125eddec391a92398e84103e77ca0a65f661452b9190c2b81f3fdbea23",
    CW_ROLE: "2f104f4fbf71bada1b6186fa3d0915fa7f8059c65d6798414ab00043e5835454",
    CW_RESULT: "1c3cd17e1349a84be7fb7af5d5b796874f3ec3d7a61539b3c6e535dddac04e08",
    CW_VALIDATION: "ae735624176e5e400ff5c16874b7f73b0505ce71f6814d0c4ed515d94455f8ac",
    GALLERY: "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc",
    P0_PRODUCER: "068c8297a0b773c4efefa093f2887393588213dce7ca347cb5d6517cc22e13b5",
    P0_VALIDATOR: "239f67385076b72b3f28c4223a5c4208f7ee64ba90c22008d618aaa627adf6db",
    CURRENT64_SOURCE: "da7ff4c0ee24bb72d1fa026e3e103e01e3cf3f9cd489e8b33f26835b4065ab58",
    CURRENT64_SOURCE_VALIDATION: "654453f5884fc2bc1848a54fe67cde37947f01cdc1b6493d3e764cda4cfb737e",
}
FOLD_HASHES = {0:"0a9c72582f0af05b9a83d28a6217ef62670a2a89b7bf433247e35e46d0e983f3",1:"c411861a4ca6b8f41792893f747d5f66561af718c6681b3d4c4bbece6b40ab72",2:"08b89f23d61012460da86191951202dbdb4e7aa4cffb75e2c673e5bb8c5f9a0f",3:"a559a678fe056353de607ba92b2bda1e675470be7a477ccdc54513c1395afb94",4:"d6014cb1c141efd7cb57a19eed9692b406efddf8ef0f52449690e06b1090c6b3"}


class ValidationError(RuntimeError): pass
def require(value: bool, message: str) -> None:
    if not value: raise ValidationError(message)
def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"input absent: {path}"); digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()
def canonical_sha256(value: Any) -> str: return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def logical_sha256(value: Mapping[str,Any]) -> str:
    payload=dict(value);payload.pop("logical_sha256",None);return canonical_sha256(payload)
def read_json(path: Path) -> dict[str,Any]:
    require(path.is_file() and not path.is_symlink(),f"JSON absent: {path}");value=json.loads(path.read_text());require(isinstance(value,dict),"JSON root");return value
def atomic_json(path: Path,value: Mapping[str,Any]) -> None:
    require(not path.exists(),f"immutable validation exists: {path}");encoded=json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n";path.parent.mkdir(parents=True,exist_ok=True);fd,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".partial",dir=path.parent);temporary=Path(name)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as handle: handle.write(encoded);handle.flush();os.fsync(handle.fileno())
        os.link(temporary,path)
    finally: temporary.unlink(missing_ok=True)


def corrected_labels() -> tuple[str,...]:
    import sys;sys.path.insert(0,str(ROOT/"src"))
    from rc_aslo_xf.n2_corrected_d1_runtime_v1 import corrected_labels_from_legacy,validate_corrected_identity_axis
    labels=corrected_labels_from_legacy(torch.load(GALLERY,map_location="cpu",weights_only=False,mmap=True)["setids"]);audit=validate_corrected_identity_axis(labels)
    require(audit.get("corrected_identity_count")==5412 and audit.get("corrected_axis_sha256")=="935ce029e3c8177fd9bc4b51f7b41fff8e241c3d587977c31d516b2f300efca4","corrected identity axis")
    return labels


def independent_consensus(query_ids:set[str],p0_by_query:Mapping[str,Mapping[str,Any]]) -> tuple[dict[str,dict[str,Any]],list[dict[str,Any]]]:
    result,validation=read_json(LABEL_RESULT),read_json(LABEL_VALIDATION)
    require(result.get("status")=="ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_READY" and result.get("logical_sha256")==logical_sha256(result) and validation.get("status")=="ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_VALIDATED" and validation.get("result_sha256")==sha256_file(LABEL_RESULT) and validation.get("logical_sha256")==logical_sha256(validation) and all(validation.get("checks",{}).values()),"label authority")
    seals={int(x["fold"]):x for x in validation["folds"]};occ=defaultdict(list);fold_bindings=[]
    for fold in range(5):
        path=LABEL_ROOT/f"fold_{fold}/payload.pt";require(sha256_file(path)==FOLD_HASHES[fold]==seals[fold]["payload_sha256"],f"fold {fold} label seal");payload=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
        for row in payload["records"]:
            if row["query_id"] in query_ids: occ[str(row["query_id"])].append((fold,row))
        fold_bindings.append({"fold":fold,"payload_sha256":FOLD_HASHES[fold]})
    output={}
    for query_id in query_ids:
        rows=occ[query_id];p0=p0_by_query[query_id];heldout=int(p0["canonical_heldout_fold"]);require(len(rows)==4 and {f for f,_ in rows}==set(range(5))-{heldout},f"fourfold occurrence {query_id}")
        values={(str(r["target_identity"]),str(r["supergroup"]),int(r["target_physical_row"]),int(r["query_ordinal"]),int(r["heldout_fold"]),str(r["track"])) for _,r in rows};require(len(values)==1,f"fourfold disagreement {query_id}");target,group,target_row,ordinal,row_fold,track=next(iter(values));require(ordinal==p0["canonical_query_ordinal"] and row_fold==heldout and track==p0["canonical_track"],f"canonical label disagreement {query_id}");output[query_id]={"target_identity":target,"target_supergroup":group,"target_physical_row_provenance":target_row,"source_folds":sorted(f for f,_ in rows)}
    return output,fold_bindings


def independent_eval32_consensus() -> dict[str,dict[str,Any]]:
    source=read_json(CURRENT64_SOURCE);validation=read_json(CURRENT64_SOURCE_VALIDATION)
    require(source.get("status")=="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_READY" and source.get("population",{}).get("role_counts")=={"EVAL":32,"TRAIN":32} and source.get("logical_sha256")==logical_sha256(source) and validation.get("status")=="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_VALIDATED" and validation.get("manifest_sha256")==sha256_file(CURRENT64_SOURCE) and validation.get("manifest_logical_sha256")==source["logical_sha256"] and validation.get("logical_sha256")==logical_sha256(validation) and all(validation.get("checks",{}).values()),"current64 source authority")
    eval_rows={str(row["query_id"]):row for row in source["records"] if row["role"]=="EVAL"};require(len(eval_rows)==32,"EVAL32 population")
    occurrences=defaultdict(list)
    for fold in range(5):
        payload=torch.load(LABEL_ROOT/f"fold_{fold}/payload.pt",map_location="cpu",weights_only=False,mmap=True)
        for row in payload["records"]:
            if row["query_id"] in eval_rows:occurrences[str(row["query_id"])].append((fold,row))
    output={}
    for query_id,source_row in eval_rows.items():
        rows=occurrences[query_id];heldout=int(source_row["heldout_fold"]);require(len(rows)==4 and {fold for fold,_ in rows}==set(range(5))-{heldout},f"EVAL32 occurrences {query_id}");values={(str(row["target_identity"]),str(row["supergroup"]),int(row["target_physical_row"]),int(row["query_ordinal"]),int(row["heldout_fold"])) for _,row in rows};require(len(values)==1,f"EVAL32 consensus {query_id}");target,group,physical,ordinal,row_fold=next(iter(values));require(ordinal==int(source_row["oof_query_ordinal"]) and row_fold==heldout,f"EVAL32 canonical role {query_id}");output[query_id]={"target_identity":target,"target_supergroup":group,"target_physical_row_provenance":physical}
    return output


def independent_cw0(query_ids:set[str]) -> tuple[dict[str,tuple[str,str]],str]:
    source,role,result,validation=map(read_json,(CW_SOURCE,CW_ROLE,CW_RESULT,CW_VALIDATION))
    require(source.get("status")=="RGH_P0_A0_SOURCE_MANIFEST_READY" and role.get("status")=="RGH_P0_A0_ROLE_MANIFEST_READY" and role.get("combined_role_runtime_read_authorized") is False and result.get("status")=="RGH_P0_A0_MANIFEST_V2_READY" and validation.get("status")=="RGH_P0_A0_MANIFEST_V2_INDEPENDENT_VALIDATION_PASS","CW0 envelope")
    source_by_query={str(row["query_id"]):row for row in source["records"]};seals={int(row["execution_ordinal"]):row for row in role["shards"]};require(len(source_by_query)==len(seals)==600 and query_ids<=set(source_by_query),"CW0 address population")
    found={};selected=[]
    for query_id in sorted(query_ids):
        seal=seals[int(source_by_query[query_id]["execution_ordinal"])];path=Path(seal["path"]);require(sha256_file(path)==seal["sha256"],"CW0 physical seal");row=read_json(path);require(row.get("logical_sha256")==seal["logical_sha256"] and str(row["query_id"])==query_id,"CW0 logical/query seal");require(query_id not in found,"CW0 duplicate");found[query_id]=(str(row["identity"]),str(row["supergroup"]));selected.append([query_id,seal["sha256"],seal["logical_sha256"]])
    require(set(found)==query_ids,"CW0 coverage");return found,canonical_sha256(selected)


def main() -> None:
    for path,expected in EXPECTED_HASHES.items(): require(sha256_file(path)==expected,f"fixed hash drift: {path}")
    p0=read_json(P0);p0v=read_json(P0_VALIDATION);p0_sha=sha256_file(P0)
    require(p0.get("status")=="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_READY" and p0.get("logical_sha256")==logical_sha256(p0) and p0v.get("status")=="ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_VALIDATED" and p0v.get("preseal_sha256")==p0_sha and p0v.get("preseal_logical_sha256")==p0["logical_sha256"] and p0v.get("producer_sha256")==sha256_file(P0_PRODUCER) and p0v.get("validator_sha256")==sha256_file(P0_VALIDATOR) and all(p0v.get("checks",{}).values()),"P0 validation")
    p0_records=p0["records"];p0_by_query={str(x["query_id"]):x for x in p0_records};require(len(p0_by_query)==64 and [x["pair64_ordinal"] for x in p0_records]==list(range(64)),"P0 population")
    query_ids=set(p0_by_query);consensus,fold_bindings=independent_consensus(query_ids,p0_by_query);eval_consensus=independent_eval32_consensus();pair_ids={row["target_identity"] for row in consensus.values()};pair_groups={row["target_supergroup"] for row in consensus.values()};eval_ids={row["target_identity"] for row in eval_consensus.values()};eval_groups={row["target_supergroup"] for row in eval_consensus.values()};disjointness={"pair64_query_count":len(query_ids),"current_eval32_query_count":len(eval_consensus),"pair64_distinct_target_identity_count":len(pair_ids),"pair64_distinct_supergroup_count":len(pair_groups),"current_eval32_distinct_target_identity_count":len(eval_ids),"current_eval32_distinct_supergroup_count":len(eval_groups),"query_id_overlap_count":len(query_ids&set(eval_consensus)),"target_identity_overlap_count":len(pair_ids&eval_ids),"supergroup_overlap_count":len(pair_groups&eval_groups),"pair64_target_identity_sha256":canonical_sha256(sorted(pair_ids)),"pair64_supergroup_sha256":canonical_sha256(sorted(pair_groups)),"current_eval32_target_identity_sha256":canonical_sha256(sorted(eval_ids)),"current_eval32_supergroup_sha256":canonical_sha256(sorted(eval_groups))};require(disjointness["pair64_distinct_target_identity_count"]==disjointness["pair64_distinct_supergroup_count"]==20 and disjointness["current_eval32_distinct_target_identity_count"]==disjointness["current_eval32_distinct_supergroup_count"]==11 and disjointness["query_id_overlap_count"]==disjointness["target_identity_overlap_count"]==disjointness["supergroup_overlap_count"]==0,"Pair64/EVAL32 disjointness");cw0,cw_hash=independent_cw0(query_ids);labels=corrected_labels();expected_records=[]
    for p in p0_records:
        q=str(p["query_id"]);target=consensus[q];target_row=target["target_physical_row_provenance"];require(labels[target_row]==target["target_identity"],f"target provenance {q}");c128=p["natural_c128"];rows=list(map(int,c128["representative_physical_rows"]));ids=list(map(str,c128["corrected_identities"]));require(ids==[labels[row] for row in rows] and len(set(ids))==128 and c128["sha256"]==canonical_sha256([[i,row,ids[i]] for i,row in enumerate(rows)]),f"C128 {q}")
        if target["target_identity"]==ids[0]: branch="TARGET_WINNER_NEGATIVE_HOLD";counter=1;switch=False;target_position=0
        elif target["target_identity"] in ids: branch="TARGET_NONWINNER_POSITIVE_SWITCH";counter=ids.index(target["target_identity"]);switch=True;target_position=counter
        else: branch="TARGET_ABSENT_NEGATIVE_HOLD";counter=1;switch=False;target_position=None
        record={"pair64_ordinal":int(p["pair64_ordinal"]),"pair_cohort":str(p["pair_cohort"]),"query_id":q,"canonical_query_ordinal":int(p["canonical_query_ordinal"]),"canonical_heldout_fold":int(p["canonical_heldout_fold"]),"canonical_track":str(p["canonical_track"]),"target_identity":target["target_identity"],"target_supergroup":target["target_supergroup"],"target_physical_row_provenance":target_row,"n2_label_source_folds":target["source_folds"],"target_state":branch,"target_naturally_in_fresh_c128":target_position is not None,"target_candidate_position":target_position,"winner":{"candidate_position":0,"representative_physical_row":rows[0],"corrected_identity":ids[0]},"counterpart":{"candidate_position":counter,"representative_physical_row":rows[counter],"corrected_identity":ids[counter]},"switch_label":switch,"preseal_record_sha256":canonical_sha256(p),"preseal_natural_c128_sha256":c128["sha256"],"target_insertion_count":0,"roma_evaluation_count":0,"feature_computation_count":0,"model_update_count":0};record["fixed_pair_sha256"]=canonical_sha256(record);expected_records.append(record)
    observed=Counter(x["target_state"] for x in expected_records);branch_counts={branch:int(observed.get(branch,0)) for branch in EXPECTED_BRANCH_COUNTS};cw_i=sum(cw0[q][0]!=consensus[q]["target_identity"] for q in query_ids);cw_g=sum(cw0[q][1]!=consensus[q]["target_supergroup"] for q in query_ids);require(branch_counts==EXPECTED_BRANCH_COUNTS and cw_i==cw_g==0,"branch/CW invariant")
    result=read_json(FIXED_PAIRS);expected_access={"pair64_query_count":64,"p0_target_bearing_read_count":0,"n2_pair64_label_record_occurrence_count":256,"n2_current_eval32_label_record_occurrence_count":128,"n2_nonheldout_records_per_query":4,"n2_four_record_disagreement_count":0,"cw0_secondary_role_shard_deserialization_count":64,"cw0_secondary_address_lookup_count":64,"cw0_secondary_query_count":64,"cw0_identity_disagreement_count":0,"cw0_supergroup_disagreement_count":0,"cw0_target_candidate_position_consumption_count":0,"cw0_execution_or_inner_fold_consumption_count":0,"current_eval32_target_identity_read_count":32,"current_eval32_supergroup_read_count":32,"historical_pair_axis_or_position_consumption_count":0,"target_insertion_count":0,"roma_evaluation_count":0,"feature_computation_count":0,"model_forward_count":0,"model_update_count":0,"external_read_count":0,"sealed_read_count":0}
    require(result.get("version")==VERSION and result.get("status")==READY and result.get("records")==expected_records and result.get("record_sequence_sha256")==canonical_sha256(expected_records) and result.get("population")=={"query_count":64,"fixed_pair_count":64,"selected_endpoint_count":128,"branch_counts":branch_counts,"switch_label_count":27,"hold_label_count":37} and result.get("pair64_current_eval32_disjointness")==disjointness and result.get("p0_hash_before_target_join")==result.get("p0_hash_after_target_join")==p0_sha and result.get("p0_unchanged_after_target_join") is True and result.get("access")==expected_access and result.get("bindings",{}).get("n2_fold_payloads")==fold_bindings and result.get("bindings",{}).get("current64_source_manifest_sha256")==sha256_file(CURRENT64_SOURCE) and result.get("bindings",{}).get("current64_source_validation_sha256")==sha256_file(CURRENT64_SOURCE_VALIDATION) and result.get("bindings",{}).get("cw0_selected64_role_shards_sha256")==cw_hash and result.get("bindings",{}).get("producer_sha256")==sha256_file(PRODUCER) and result.get("logical_sha256")==logical_sha256(result) and result.get("next_authorized_stage")==NEXT,"independent fixed-pair reconstruction differs")
    checks={"producer_not_imported":True,"p0_independently_validated_and_hash_unchanged":True,"n2_fourfold_consensus_64x4":True,"pair64_current_eval32_query_identity_supergroup_disjoint":True,"corrected_5412_identity_membership":True,"three_branch_counts_37_27_0":True,"cw0_secondary_identity_group_agreement_64":True,"cw0_position_axis_fold_consumption_zero":True,"fixed_pairs_64_endpoints_128":True,"target_insertion_zero":True,"no_roma_feature_or_model_update":True,"no_external_or_sealed_access":True,"no_scientific_claim_or_auto_advance":result.get("scientific_GO_or_NO_GO") is None and result.get("ownership_GO_or_NO_GO") is None and result.get("automatic_stage_advance") is False}
    require(all(checks.values()),"independent checks")
    value={"version":VERSION,"status":VALIDATED,"claim_level":"INDEPENDENT_POSTSEAL_TRAINING_ONLY_FIXED_PAIR_MANIFEST_VALIDATION","checks":checks,"population":result["population"],"pair64_current_eval32_disjointness":disjointness,"fixed_pairs_sha256":sha256_file(FIXED_PAIRS),"fixed_pairs_logical_sha256":result["logical_sha256"],"p0_preseal_sha256":p0_sha,"p0_validation_sha256":sha256_file(P0_VALIDATION),"producer_sha256":sha256_file(PRODUCER),"validator_sha256":sha256_file(Path(__file__).resolve()),"access":expected_access,"scientific_GO_or_NO_GO":None,"ownership_GO_or_NO_GO":None,"automatic_stage_advance":False,"next_authorized_stage":NEXT,"logical_sha256":""};value["logical_sha256"]=logical_sha256(value);atomic_json(OUT,value);print(json.dumps({"status":VALIDATED,"checks":checks,"branches":branch_counts},sort_keys=True))


if __name__ == "__main__": main()
