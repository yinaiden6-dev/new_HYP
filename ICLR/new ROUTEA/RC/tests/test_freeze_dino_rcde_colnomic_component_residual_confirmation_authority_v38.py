from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZER = (
    ROOT
    / "programs/freeze_dino_rcde_colnomic_component_residual_confirmation_authority_v38.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("raw_residual_v38", FREEZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v38_executes_only_unit_residual_on_582_queries() -> None:
    module = load_module()
    value = module.build()
    assert value["status"] == module.STATUS
    assert value["residual_reduction_authorized"] is True
    assert value["residual_weight"] == 1.0
    assert value["alternative_weight_count"] == 0
    assert value["normalization_threshold_hold_scan_authorized"] is False
    assert value["population_contract"]["development_execution_ordinals_excluded"] == list(range(12))
    assert value["population_contract"]["confirmation_query_count"] == 582
    assert value["model_forward_authorized"] is False
    assert value["logical_sha256"] == module.H.logical(value)


def test_v38_binds_full_raw_axis_and_pj1() -> None:
    module = load_module()
    bindings = module.build()["bindings"]
    assert bindings["raw_prejoin_ledger"]["path"].endswith("prejoin_ledger.pt")
    assert bindings["pj1_result"]["path"].endswith("postjoin_pj1_v1/result.json")
    assert bindings["preregistration_authority"]["path"].endswith(
        "residual_prereg_authority_v37_20260824.json"
    )
