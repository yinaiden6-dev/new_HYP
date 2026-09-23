from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZER = (
    ROOT / "programs/freeze_dino_rcde_h0_component_scale4_confirmation_authority_v36.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("component_scale4_v36", FREEZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v36_executes_only_the_preregistered_scale_on_582_queries() -> None:
    module = load_module()
    value = module.build()
    assert value["status"] == module.STATUS
    assert value["scale4_reduction_authorized"] is True
    assert value["scale_contract"]["scale"] == 4.0
    assert value["alternative_scale_count"] == 0
    assert value["threshold_or_hold_scan_authorized"] is False
    assert value["population_contract"]["development_execution_ordinals_excluded"] == list(range(12))
    assert value["population_contract"]["confirmation_query_count"] == 582
    assert value["model_forward_authorized"] is False
    assert value["next_authorized_stage"] is None
    assert value["logical_sha256"] == module.H.logical(value)


def test_v36_binds_pj1_and_unscaled_negative_comparator() -> None:
    module = load_module()
    bindings = module.build()["bindings"]
    assert bindings["pj1_result"]["path"].endswith("postjoin_pj1_v1/result.json")
    assert bindings["unscaled_pj2_result"]["path"].endswith("postjoin_pj2_v1/result.json")
    assert bindings["preregistration_authority"]["path"].endswith(
        "scale4_prereg_authority_v35_20260824.json"
    )
