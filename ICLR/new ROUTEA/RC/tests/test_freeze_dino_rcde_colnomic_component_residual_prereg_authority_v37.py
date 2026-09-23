from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZER = (
    ROOT
    / "programs/freeze_dino_rcde_colnomic_component_residual_prereg_authority_v37.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("raw_component_v37", FREEZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v37_freezes_unit_weight_without_execution() -> None:
    module = load_module()
    value = module.build()
    assert value["status"] == module.STATUS
    assert value["residual_weight"] == 1.0
    assert value["score_formula"] == (
        "S_RAW_COLNOMIC + (S_TRACK_H_COMPONENT - S_INIT_COMPONENT)"
    )
    assert value["alternative_weight_count"] == 0
    assert value["normalization_threshold_hold_scan_authorized"] is False
    assert value["execution_authorized"] is False
    assert value["scientific_GO_or_NO_GO"] is None
    assert value["logical_sha256"] == module.H.logical(value)


def test_v37_binds_full_raw_axis_and_excludes_development() -> None:
    module = load_module()
    value = module.build()
    assert value["confirmation_execution_ordinals_excluded"] == list(range(12))
    assert value["expected_confirmation_query_count"] == 582
    assert value["bindings"]["raw_prejoin_ledger"]["path"].endswith(
        "prejoin_ledger.pt"
    )
    assert value["bindings"]["pj1_result"]["path"].endswith(
        "postjoin_pj1_v1/result.json"
    )
