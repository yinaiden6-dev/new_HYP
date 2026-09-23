from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZER = (
    ROOT / "programs/freeze_dino_rcde_h0_component_scale4_prereg_authority_v35.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("component_scale4_v35", FREEZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v35_freezes_one_nonexecutable_scale_only() -> None:
    module = load_module()
    value = module.build()
    assert value["status"] == module.STATUS
    assert value["scale"] == 4.0
    assert value["score_formula"] == "S_INIT + 4.0 * (S_TRACK_H - S_INIT)"
    assert value["alternative_scale_count"] == 0
    assert value["threshold_or_hold_scan_authorized"] is False
    assert value["execution_authorized"] is False
    assert value["semantic_join_authorized"] is False
    assert value["scientific_reduction_authorized"] is False
    assert value["scientific_GO_or_NO_GO"] is None
    assert value["logical_sha256"] == module.H.logical(value)


def test_v35_excludes_development_canary_and_binds_negative_history() -> None:
    module = load_module()
    value = module.build()
    assert value["confirmation_execution_ordinals_excluded"] == list(range(12))
    assert value["expected_confirmation_query_count"] == 582
    assert value["bindings"]["track_r_negative_result"]["path"].endswith(
        "dino_rcde_track_r_scientific_result_v1/result.json"
    )
    assert value["future_input_requirements"]["pj1_independent_validation_pass"] is True
