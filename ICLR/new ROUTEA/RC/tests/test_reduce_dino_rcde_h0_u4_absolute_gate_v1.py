from __future__ import annotations

import math

import reduce_dino_rcde_h0_u4_absolute_gate_v1 as R


def row(index:int,group:str,fold:int,target:float,donor:float)->dict:
    arm={"checkpoint_sha256":"a"*64,"target":{"a_to_b":target,"b_to_a":target},"donor":{"a_to_b":donor,"b_to_a":donor}}
    return {"outer_ledger_record_sha256":f"{index:064x}","outer_fold":fold,"execution_ordinal":index,"query_id":f"Q{index}","recipient_supergroup_sha256":group,"track":"fixture","arms":{name:arm for name in R.ARMS}}


def test_zero_logits_have_exact_registered_null_scale() -> None:
    value=R.query_arm_metrics(row(0,"g",1,0.0,0.0),"TRACK_H")
    assert value["loss"]==math.log(2.0)
    assert value["margin"]==0.0 and value["win"]==0.0


def test_primary_is_supergroup_then_query_balanced() -> None:
    rows=[row(0,"large",1,2.0,-2.0),row(1,"large",1,2.0,-2.0),row(2,"large",1,2.0,-2.0),row(3,"small",1,-2.0,2.0)]
    out=R.summarize(rows,"TRACK_H")
    q=out["diagnostic_query_balanced"]["margin"];g=out["primary_recipient_supergroup_balanced"]["margin"]
    assert q==2.0 and g==0.0


def test_loss_is_mean_over_four_role_direction_terms() -> None:
    value=R.query_arm_metrics(row(0,"g",1,1.0,-1.0),"TRACK_H")
    assert math.isclose(value["loss"],R.softplus(-1.0),rel_tol=0.0,abs_tol=1e-15)
