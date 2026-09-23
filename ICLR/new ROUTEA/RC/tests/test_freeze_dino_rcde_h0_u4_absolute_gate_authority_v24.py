from __future__ import annotations

import freeze_dino_rcde_h0_u4_absolute_gate_authority_v24 as F


def test_v24_only_builds_after_all_external_gates_pass() -> None:
    value=F.build()
    assert value["track_r_four_fold_fresh_resume_gate_satisfied"] is True
    assert value["scientific_reducer_authorized"] is True
    assert value["unquarantine_authorized"] is True
    assert value["unquarantine_scope"]=="REGISTERED_U4_REDUCTION_ONLY"
    assert value["retrieval_or_ownership_claim_authorized"] is False
    assert value["metric_population"]=={"query_count":594,"matched_scored":567,"unmatched_unscored":27,"fallback_count":0}
    assert value["gate_contract"]["loss_formula"].startswith("0.5 *")
    assert value["gate_contract"]["minimum_group_balanced_win_rate"]==0.625
    assert all(f"track_r_fresh_resume_validation_fold{fold}" in value["bindings"] for fold in range(1,5))
