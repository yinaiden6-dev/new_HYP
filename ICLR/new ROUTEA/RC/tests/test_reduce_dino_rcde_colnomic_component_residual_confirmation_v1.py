from __future__ import annotations

import importlib.util
from pathlib import Path

import reduce_dino_rcde_h0_c128_target_free_postjoin_pj2_v1 as PJ2


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = (
    ROOT / "programs/reduce_dino_rcde_colnomic_component_residual_confirmation_v1.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("raw_residual_reducer", PROGRAM)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate(row: int, identity: str, raw: float, fused: float):
    return {
        "candidate_physical_row": row,
        "candidate_corrected_identity": identity,
        "scores": {
            "RAW": {"score": raw, "score_bits": PJ2.float_bits(raw)},
            "FUSED": {"score": fused, "score_bits": PJ2.float_bits(fused)},
        },
    }


def test_fusion_can_rescue_while_preserving_stable_identity_tie() -> None:
    module = load_module()
    rows = [
        candidate(4, "target", 0.1, 0.4),
        candidate(9, "wrong", 0.2, 0.3),
        candidate(7, "wrong", 0.2, 0.3),
    ]
    raw = module.reduce_arm(rows, "RAW", "target")
    fused = module.reduce_arm(rows, "FUSED", "target")
    assert raw["strict_top1"] is False
    assert fused["strict_top1"] is True
    assert raw["strongest_wrong"]["physical_row"] == 7
    assert module.transition(raw, fused) == "rescue"


def test_raw_prejoin_requires_zero_semantic_join() -> None:
    module = load_module()
    payload = {
        "status": "RCDE_SR0_MT_I0_PREJOIN_LEDGER_SEALED",
        "query_count": 600,
        "candidate_count_per_query": 128,
        "target_or_label_join_count": 1,
        "label_read_count": 0,
        "records": [],
        "record_sequence_sha256": module.canonical([]),
    }
    try:
        module.validate_raw_prejoin(payload)
    except module.ResidualConfirmationError:
        pass
    else:
        raise AssertionError("semantic RAW ledger was accepted")


def test_only_unit_residual_and_582_confirmation_are_exposed() -> None:
    module = load_module()
    source = PROGRAM.read_text(encoding="utf-8")
    assert module.RESIDUAL_WEIGHT == 1.0
    assert module.EXCLUDED == tuple(range(12))
    assert "alternative_weight_count" in source
    assert "threshold" not in source.lower()
