from __future__ import annotations
import freeze_dino_rcde_h0_bias_calibrator_authority_v28 as F
def test_v28_is_one_parameter_optimization_only()->None:
    v=F.build();assert v["calibrator_family"]=="SHARED_BIAS_ONLY";assert v["calibrator_fit_authorized"] is True;assert v["calibrator_consumption_authorized"] is False;assert v["model_forward_authorized"] is False;assert v["fit_contract"]["parameters"]==1;assert v["fit_contract"]["regularization"] is None;assert v["qualification_gates"]["combined_lofo_improvement_vs_init_min"]==0.01;assert v["U4_role"]=="OPTIMIZATION_ONLY_NOT_CONFIRMATION";assert v["scientific_GO_or_NO_GO"] is None
