#!/usr/bin/env python3
"""Freeze U4 scientific reduction only after four Track-R replay validations PASS."""

from __future__ import annotations

import argparse,json,math
from pathlib import Path
from typing import Any
import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H


ROOT=Path(__file__).resolve().parents[1];AUTH="registry/h0/h0_u4_absolute_gate_authority_v24_20260823.json";STATUS="H0_U4_ABSOLUTE_GATE_REDUCTION_AUTHORIZED";TRACK_PASS="DINO_RCDE_TRACK_R_RELATIVE_V_FIT_FRESH_RESUME_VALIDATION_PASS"
def req(c:Any,m:str)->None:
    if not c:raise RuntimeError(m)
def read(p:str)->dict[str,Any]:v=json.loads((ROOT/p).read_text());req(isinstance(v,dict),f"JSON drift {p}");return v


def build()->dict[str,Any]:
    raw_path="results/dino_rcde_h0_u4_raw_aggregate_v2/result.json";rawv_path="results/dino_rcde_h0_u4_raw_aggregate_validation_v2/result.json";raw=read(raw_path);rawv=read(rawv_path);req(raw.get("status")=="H0_U4_V2_RAW_SCORE_AGGREGATE_QUARANTINED" and raw.get("scientific_metric_count")==0 and raw.get("quarantine_released") is False and rawv.get("status")=="H0_U4_V2_RAW_SCORE_AGGREGATE_INDEPENDENT_VALIDATION_PASS" and rawv.get("validation_pass") is True,"raw aggregate gate drift")
    track={}
    for fold in range(1,5):
        path=f"results/dino_rcde_track_r_relative_v_fits_v1/outer_fold{fold}/fresh_resume_validation.json";v=read(path);req(v.get("status")==TRACK_PASS and v.get("validation_pass") is True and v.get("fresh_resume_model_bit_exact") is True and v.get("fresh_resume_schedule_trace_exact") is True and v.get("relative_pair_loss_only") is True and v.get("donor_null_absolute_loss_count")==0 and v.get("outer_heldout_forward_count")==0,"Track-R fold validation gate drift");track[fold]=path
    u3=read("results/dino_rcde_h0_u3_formal_training_validation_v1/result.json");req(u3.get("status")=="H0_FULL_REFERENCE_UNARY_U3_FOUR_FOLD_TRAINING_VALIDATION_PASS" and u3.get("validation_pass") is True,"H U3 gate drift")
    bindings={"parent_v18":H.bind("registry/h0/h0_u4_quarantine_precompute_authority_v18_20260823.json",with_logical=True,immutable=True),"raw_aggregate_authority_v23":H.bind("registry/h0/h0_u4_raw_aggregate_authority_v23_20260823.json",with_logical=True,immutable=True),"raw_aggregate":H.bind(raw_path,with_logical=True,immutable=True),"raw_aggregate_validation":H.bind(rawv_path,with_logical=True,immutable=True),"h_u3_validation":H.bind("results/dino_rcde_h0_u3_formal_training_validation_v1/result.json",with_logical=True,immutable=True),"track_r_backend_repair_authority":H.bind("registry/dino_rcde_track_r_resume_backend_repair_authority_v2_20260823.json",with_logical=True,immutable=True),"contract":H.bind("plan/DINO_RCDE_H0_U4_OUTER_OOF_ABSOLUTE_GATE_CONTRACT_V1_20260823.md"),"reducer":H.bind("programs/reduce_dino_rcde_h0_u4_absolute_gate_v1.py"),"validator":H.bind("programs/validate_dino_rcde_h0_u4_absolute_gate_v1.py"),"freezer":H.bind("programs/freeze_dino_rcde_h0_u4_absolute_gate_authority_v24.py"),"launcher":H.bind("slurm/dino_rcde_h0_u4_absolute_gate_v1.sbatch"),"test_reducer":H.bind("tests/test_reduce_dino_rcde_h0_u4_absolute_gate_v1.py"),"test_validator":H.bind("tests/test_validate_dino_rcde_h0_u4_absolute_gate_v1.py"),"test_authority":H.bind("tests/test_freeze_dino_rcde_h0_u4_absolute_gate_authority_v24.py")}
    for fold,path in track.items():bindings[f"track_r_fresh_resume_validation_fold{fold}"]=H.bind(path,with_logical=True,immutable=True)
    value={"schema_version":"rc_h0_u4_absolute_gate_authority_v24_20260823","status":STATUS,"track":"H0","claim_level":"CONDITIONAL_ABSOLUTE_VERIFICATION_GATE_ONLY","track_r_four_fold_fresh_resume_gate_satisfied":True,"scientific_reducer_authorized":True,"unquarantine_authorized":True,"unquarantine_scope":"REGISTERED_U4_REDUCTION_ONLY","retrieval_or_ownership_claim_authorized":False,"training_authorized":False,"protected_access_authorized":False,"automatic_stage_advance":False,"scientific_GO_or_NO_GO":None,"next_authorized_stage":None,"metric_population":{"query_count":594,"matched_scored":567,"unmatched_unscored":27,"fallback_count":0},"gate_contract":{"loss_formula":"0.5 * mean_direction(softplus(-target) + softplus(donor))","null_loss":math.log(2.0),"minimum_loss_improvement_vs_track_r":0.01,"minimum_loss_improvement_vs_init":0.01,"minimum_group_balanced_win_rate":0.625,"minimum_positive_fold_count":3,"strict_positive_margin":True,"primary_aggregation":"recipient_supergroup_then_query_balanced"},"output":"results/dino_rcde_h0_u4_absolute_gate_v1/result.json","validation_output":"results/dino_rcde_h0_u4_absolute_gate_validation_v1/result.json","resource_contract":{"partitions":["dev_cpuonly","cpuonly"],"cpus_per_task":2,"memory_megabytes":8192,"walltime_seconds":600},"bindings":bindings};value["logical_sha256"]=H.logical(value);return value


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=ROOT/AUTH);a=p.parse_args();req(a.output.resolve(strict=False)==(ROOT/AUTH).resolve(strict=False),"output drift");v=build()
    if a.output.exists():req(json.loads(a.output.read_text())==v,"existing drift")
    else:H.atomic(a.output,v)
    print(json.dumps({"status":STATUS,"logical_sha256":v["logical_sha256"]},sort_keys=True))
if __name__=="__main__":main()
