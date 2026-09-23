from __future__ import annotations

import math

import reduce_dino_rcde_h0_u4_absolute_gate_v1 as P
import validate_dino_rcde_h0_u4_absolute_gate_v1 as V


def test_independent_null_loss_matches_log2() -> None:
    arm={"target":{"a_to_b":0.0,"b_to_a":0.0},"donor":{"a_to_b":0.0,"b_to_a":0.0}}
    row={"arms":{name:arm for name in P.ARMS},"recipient_supergroup_sha256":"g"}
    assert V.qm(row,"TRACK_H")["loss"]==math.log(2.0)


def test_recursive_comparator_accepts_roundoff_but_not_gate_drift() -> None:
    V.close({"x":0.5},{"x":0.5+1e-16})
    try:V.close({"x":True},{"x":False})
    except P.U4ReductionError:return
    raise AssertionError("gate drift accepted")
