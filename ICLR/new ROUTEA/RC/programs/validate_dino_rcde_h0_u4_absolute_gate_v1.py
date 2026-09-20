#!/usr/bin/env python3
"""Independent recomputation of the registered U4 absolute-verification gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

import reduce_dino_rcde_h0_u4_absolute_gate_v1 as P


def sp(x:float)->float:
    return math.log1p(math.exp(x)) if x<=0.0 else x+math.log1p(math.exp(-x))


def qm(row:Mapping[str,Any],arm:str)->dict[str,float]:
    t=row["arms"][arm]["target"];d=row["arms"][arm]["donor"];directions=("a_to_b","b_to_a")
    loss=statistics.fmean([sp(-float(t[x])) for x in directions]+[sp(float(d[x])) for x in directions]);margin=statistics.fmean(float(t[x])-float(d[x]) for x in directions)
    return {"loss":loss,"margin":margin,"win":float(margin>0.0)}


def summary(rows:list[Mapping[str,Any]],arm:str)->dict[str,Any]:
    groups=defaultdict(list)
    for row in rows:groups[str(row["recipient_supergroup_sha256"])].append(qm(row,arm))
    gs=[]
    for group in sorted(groups):
        items=groups[group];gs.append({"recipient_supergroup_sha256":group,"query_count":len(items),"loss":statistics.fmean(x["loss"] for x in items),"margin":statistics.fmean(x["margin"] for x in items),"win_rate":statistics.fmean(x["win"] for x in items)})
    q=[qm(row,arm) for row in rows]
    return {"primary_recipient_supergroup_balanced":{"loss":statistics.fmean(x["loss"] for x in gs),"margin":statistics.fmean(x["margin"] for x in gs),"win_rate":statistics.fmean(x["win_rate"] for x in gs),"supergroup_count":len(gs),"query_count":len(rows)},"diagnostic_query_balanced":{"loss":statistics.fmean(x["loss"] for x in q),"margin":statistics.fmean(x["margin"] for x in q),"win_rate":statistics.fmean(x["win"] for x in q),"query_count":len(rows)},"supergroups":gs}


def recompute(rows:list[Mapping[str,Any]])->dict[str,Any]:
    arms={arm:summary(rows,arm) for arm in P.ARMS};folds={str(f):{arm:summary([r for r in rows if int(r["outer_fold"])==f],arm)["primary_recipient_supergroup_balanced"] for arm in P.ARMS} for f in range(1,5)};tracks={track:{arm:summary([r for r in rows if str(r["track"])==track],arm)["diagnostic_query_balanced"] for arm in P.ARMS} for track in sorted({str(r["track"]) for r in rows})}
    h=arms["TRACK_H"]["primary_recipient_supergroup_balanced"];r=arms["TRACK_R"]["primary_recipient_supergroup_balanced"];i=arms["INIT"]["primary_recipient_supergroup_balanced"]
    gates={"track_h_loss_below_log2":h["loss"]<math.log(2.0),"track_h_loss_at_least_0p01_below_track_r":r["loss"]-h["loss"]>=0.01,"track_h_loss_at_least_0p01_below_init":i["loss"]-h["loss"]>=0.01,"track_h_win_rate_at_least_0p625":h["win_rate"]>=0.625,"track_h_margin_positive":h["margin"]>0.0,"track_h_positive_fold_margin_at_least_3_of_4":sum(folds[str(f)]["TRACK_H"]["margin"]>0.0 for f in range(1,5))>=3}
    return {"arm_summaries":arms,"fold_summaries":folds,"track_summaries":tracks,"gates":gates,"all_gates_pass":all(gates.values())}


def close(a:Any,b:Any,path:str="root")->None:
    if isinstance(a,float) or isinstance(b,float):P.require(math.isclose(float(a),float(b),rel_tol=0.0,abs_tol=5e-15),f"float drift {path}");return
    P.require(type(a) is type(b),f"type drift {path}")
    if isinstance(a,dict):P.require(set(a)==set(b),f"key drift {path}");[close(a[k],b[k],f"{path}.{k}") for k in a]
    elif isinstance(a,list):P.require(len(a)==len(b),f"length drift {path}");[close(x,y,f"{path}[{i}]") for i,(x,y) in enumerate(zip(a,b,strict=True))]
    else:P.require(a==b,f"value drift {path}")


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--authority",type=Path,required=True);p.add_argument("--aggregate",type=Path,required=True);p.add_argument("--result",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();authority=P.read(a.authority);raw=P.read(a.aggregate);result=P.read(a.result)
    P.require(authority.get("status")==P.AUTH_STATUS and authority.get("logical_sha256")==P.logical(authority),"authority drift");P.require(result.get("logical_sha256")==P.logical(result),"result envelope drift");expected=recompute(raw["rows"])
    for key in ("arm_summaries","fold_summaries","track_summaries","gates","all_gates_pass"):close(result[key],expected[key],key)
    go=expected["all_gates_pass"];P.require(result["status"]==("H0_U4_CONDITIONAL_ABSOLUTE_VERIFICATION_GO" if go else "H0_U4_CONDITIONAL_ABSOLUTE_VERIFICATION_NO_GO") and result["scientific_GO_or_NO_GO"]==("GO" if go else "NO_GO") and result["metric_population"]=={"query_count":594,"matched_scored":567,"unmatched_unscored":27,"fallback_count":0} and result["retrieval_or_ownership_claim_authorized"] is False,"scientific result decision/claim drift")
    value={"schema_version":"rc_dino_rcde_h0_u4_absolute_gate_validation_v1_20260823","status":"H0_U4_ABSOLUTE_GATE_INDEPENDENT_VALIDATION_PASS","validation_pass":True,"authority_logical_sha256":authority["logical_sha256"],"raw_aggregate_logical_sha256":raw["logical_sha256"],"result_logical_sha256":result["logical_sha256"],"scientific_GO_or_NO_GO":result["scientific_GO_or_NO_GO"],"all_gates_recomputed":True,"loss_normalization_recomputed":True,"group_balancing_recomputed":True,"unmatched_fallback_count":0,"automatic_stage_advance":False,"next_authorized_stage":None};value["logical_sha256"]=P.logical(value);P.atomic(a.output,value);print(json.dumps({"status":value["status"],"scientific_GO_or_NO_GO":value["scientific_GO_or_NO_GO"]},sort_keys=True))


if __name__=="__main__":main()
