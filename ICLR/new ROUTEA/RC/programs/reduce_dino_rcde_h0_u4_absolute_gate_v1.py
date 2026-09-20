#!/usr/bin/env python3
"""Registered U4 conditional absolute-verification reducer.

This file may be tested before authorization, but ``main`` must not run until
a later authority binds four independently passing Track-R fresh/resume
validations and explicitly releases the quarantined raw aggregate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256


ROOT=Path(__file__).resolve().parents[1]; AUTH_STATUS="H0_U4_ABSOLUTE_GATE_REDUCTION_AUTHORIZED"; ARMS=("INIT","TRACK_R","TRACK_H"); DIRECTIONS=("a_to_b","b_to_a")


class U4ReductionError(RuntimeError): pass
def require(c:Any,m:str)->None:
    if not c: raise U4ReductionError(m)
def fsha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def logical(v:Mapping[str,Any])->str:return canonical_sha256({k:x for k,x in v.items() if k!="logical_sha256"})
def softplus(x:float)->float:return max(x,0.0)+math.log1p(math.exp(-abs(x)))


def query_arm_metrics(row:Mapping[str,Any],arm:str)->dict[str,float]:
    require(arm in ARMS and set(row["arms"])==set(ARMS),"arm population drift"); value=row["arms"][arm]; target=value["target"];donor=value["donor"]
    require(set(target)==set(donor)==set(DIRECTIONS),"direction population drift")
    logits=[float(target[d]) for d in DIRECTIONS]+[float(donor[d]) for d in DIRECTIONS];require(all(math.isfinite(x) for x in logits),"nonfinite logit")
    # Mean over 2 roles x 2 directions.  All-zero logits therefore equal log(2).
    loss=0.25*sum(softplus(-float(target[d]))+softplus(float(donor[d])) for d in DIRECTIONS)
    margin=0.5*sum(float(target[d])-float(donor[d]) for d in DIRECTIONS)
    return {"loss":loss,"margin":margin,"win":float(margin>0.0)}


def mean(values:Iterable[float])->float:
    rows=list(values);require(bool(rows),"empty mean");return math.fsum(rows)/len(rows)


def summarize(rows:list[Mapping[str,Any]],arm:str)->dict[str,Any]:
    require(bool(rows),"empty summary population"); metrics=[query_arm_metrics(row,arm) for row in rows]
    groups:dict[str,list[dict[str,float]]]=defaultdict(list)
    for row,item in zip(rows,metrics,strict=True):groups[str(row["recipient_supergroup_sha256"])].append(item)
    group_rows=[]
    for group in sorted(groups):
        items=groups[group];group_rows.append({"recipient_supergroup_sha256":group,"query_count":len(items),"loss":mean(x["loss"] for x in items),"margin":mean(x["margin"] for x in items),"win_rate":mean(x["win"] for x in items)})
    primary={"loss":mean(x["loss"] for x in group_rows),"margin":mean(x["margin"] for x in group_rows),"win_rate":mean(x["win_rate"] for x in group_rows),"supergroup_count":len(group_rows),"query_count":len(rows)}
    query_balanced={"loss":mean(x["loss"] for x in metrics),"margin":mean(x["margin"] for x in metrics),"win_rate":mean(x["win"] for x in metrics),"query_count":len(rows)}
    return {"primary_recipient_supergroup_balanced":primary,"diagnostic_query_balanced":query_balanced,"supergroups":group_rows}


def reduce_rows(rows:list[Mapping[str,Any]])->dict[str,Any]:
    require(len(rows)==567 and len({str(r["outer_ledger_record_sha256"]) for r in rows})==567,"fixed matched population drift")
    arm_summaries={arm:summarize(rows,arm) for arm in ARMS}
    fold_summaries={}
    for fold in range(1,5):
        selected=[r for r in rows if int(r["outer_fold"])==fold];require(len(selected)=={1:145,2:140,3:136,4:146}[fold],"fold population drift")
        fold_summaries[str(fold)]={arm:summarize(selected,arm)["primary_recipient_supergroup_balanced"] for arm in ARMS}
    track_summaries={}
    for track in sorted({str(r["track"]) for r in rows}):
        selected=[r for r in rows if str(r["track"])==track];track_summaries[track]={arm:summarize(selected,arm)["diagnostic_query_balanced"] for arm in ARMS}
    h=arm_summaries["TRACK_H"]["primary_recipient_supergroup_balanced"];r=arm_summaries["TRACK_R"]["primary_recipient_supergroup_balanced"];i=arm_summaries["INIT"]["primary_recipient_supergroup_balanced"]
    gates={"track_h_loss_below_log2":h["loss"]<math.log(2.0),"track_h_loss_at_least_0p01_below_track_r":r["loss"]-h["loss"]>=0.01,"track_h_loss_at_least_0p01_below_init":i["loss"]-h["loss"]>=0.01,"track_h_win_rate_at_least_0p625":h["win_rate"]>=0.625,"track_h_margin_positive":h["margin"]>0.0,"track_h_positive_fold_margin_at_least_3_of_4":sum(fold_summaries[str(f)]["TRACK_H"]["margin"]>0.0 for f in range(1,5))>=3}
    return {"arm_summaries":arm_summaries,"fold_summaries":fold_summaries,"track_summaries":track_summaries,"gates":gates,"all_gates_pass":all(gates.values())}


def read(p:Path)->dict[str,Any]:
    v=json.loads(p.read_text());require(isinstance(v,dict),"JSON drift");return v
def atomic(p:Path,v:Mapping[str,Any])->None:
    require(not p.exists() and ROOT in p.resolve().parents,"output unsafe");p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix=f".{p.name}.",dir=p.parent)
    with os.fdopen(fd,"w",encoding="ascii") as h:h.write(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+"\n");h.flush();os.fsync(h.fileno())
    os.chmod(t,0o444);os.replace(t,p)


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--authority",type=Path,required=True);p.add_argument("--aggregate",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();authority=read(a.authority)
    require(authority.get("status")==AUTH_STATUS and authority.get("scientific_reducer_authorized") is True and authority.get("unquarantine_authorized") is True and authority.get("logical_sha256")==logical(authority),"scientific authority absent/drift")
    binding=authority["bindings"]["raw_aggregate"];require(a.aggregate.resolve()==(ROOT/binding["path"]).resolve() and fsha(a.aggregate)==binding["sha256"],"raw aggregate binding drift")
    for fold in range(1,5):
        v=read(ROOT/authority["bindings"][f"track_r_fresh_resume_validation_fold{fold}"]["path"]);require(v.get("status")=="DINO_RCDE_TRACK_R_RELATIVE_V_FIT_FRESH_RESUME_VALIDATION_PASS" and v.get("validation_pass") is True and v.get("fresh_resume_model_bit_exact") is True and v.get("fresh_resume_schedule_trace_exact") is True,"Track-R fresh/resume gate absent")
    raw=read(a.aggregate);require(raw.get("status")=="H0_U4_V2_RAW_SCORE_AGGREGATE_QUARANTINED" and raw.get("scientific_metric_count")==0 and raw.get("quarantine_released") is False and len(raw.get("rows",[]))==567 and len(raw.get("unmatched_rows",[]))==27,"raw aggregate drift")
    reduced=reduce_rows(raw["rows"]);go=bool(reduced["all_gates_pass"])
    value={"schema_version":"rc_dino_rcde_h0_u4_absolute_gate_result_v1_20260823","status":"H0_U4_CONDITIONAL_ABSOLUTE_VERIFICATION_GO" if go else "H0_U4_CONDITIONAL_ABSOLUTE_VERIFICATION_NO_GO","claim_level":"CONDITIONAL_ON_CORRECT_CANDIDATE_AND_SEALED_P_LOCK_ONLY","authority_logical_sha256":authority["logical_sha256"],"raw_aggregate_logical_sha256":raw["logical_sha256"],"metric_population":{"query_count":594,"matched_scored":567,"unmatched_unscored":27,"fallback_count":0},**reduced,"loss_normalization":"0.5 * mean_direction(target_BCE + donor_BCE)","null_loss":math.log(2.0),"quarantine_released_for_registered_reduction_only":True,"retrieval_or_ownership_claim_authorized":False,"scientific_GO_or_NO_GO":"GO" if go else "NO_GO","automatic_stage_advance":False,"next_authorized_stage":None};value["logical_sha256"]=logical(value);atomic(a.output,value);print(json.dumps({"status":value["status"],"scientific_GO_or_NO_GO":value["scientific_GO_or_NO_GO"]},sort_keys=True))


if __name__=="__main__":main()
