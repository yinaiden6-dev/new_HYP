from __future__ import annotations

import importlib.util
from pathlib import Path

import reduce_dino_rcde_h0_c128_target_free_postjoin_pj2_v1 as PJ2


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "programs/reduce_dino_rcde_h0_component_scale4_confirmation_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("scale4_confirmation", PROGRAM)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate(row: int, identity: str, init: float, h: float, scale: float):
    return {
        "candidate_physical_row": row,
        "candidate_corrected_identity": identity,
        "scores": {
            "INIT": {"score": init, "score_bits": PJ2.float_bits(init)},
            "TRACK_H": {"score": h, "score_bits": PJ2.float_bits(h)},
            "SCALE4": {"score": scale, "score_bits": PJ2.float_bits(scale)},
        },
    }


def test_identity_reduction_and_stable_tie_are_arm_local() -> None:
    module = load_module()
    rows = [
        candidate(8, "target", 0.2, 0.2, 0.2),
        candidate(9, "wrong-a", 0.1, 0.3, 0.5),
        candidate(7, "wrong-a", 0.1, 0.3, 0.5),
        candidate(10, "wrong-b", 0.0, 0.0, 0.0),
    ]
    init = module.reduce_scored_arm(rows, "INIT", "target")
    scale = module.reduce_scored_arm(rows, "SCALE4", "target")
    assert init["strict_top1"] is True
    assert init["rank"] == 1
    assert scale["strict_top1"] is False
    assert scale["strongest_wrong"]["physical_row"] == 7
    assert scale["rank"] == 2


def test_transition_names_are_fixed() -> None:
    module = load_module()
    assert module.comparison_transition({"strict_top1": False}, {"strict_top1": True}) == "rescue"
    assert module.comparison_transition({"strict_top1": True}, {"strict_top1": False}) == "break"
    assert module.comparison_transition({"strict_top1": True}, {"strict_top1": True}) == "both_correct"
    assert module.comparison_transition({"strict_top1": False}, {"strict_top1": False}) == "both_wrong"


def test_confirmation_population_and_single_scale_are_frozen() -> None:
    module = load_module()
    source = PROGRAM.read_text(encoding="utf-8")
    assert module.EXCLUDED == tuple(range(12))
    assert sum(module.EXPECTED_FOLDS.values()) == 582
    assert module.SCALE == 4.0
    assert "alternative_scale_count" in source
    assert "threshold" not in source.lower()
