from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

from rc_aslo_xf.dino_rcde_track_r_statistics_v1 import ARMS, evaluate_track_r  # noqa: E402
from dino_rcde_track_r_statistics_independent_v1 import independent_evaluate_track_r  # noqa: E402
from seal_dino_rcde_track_r_i0_science_metadata_v1 import (  # noqa: E402
    TrackRMetadataSealError,
    build_sealed_metadata,
    file_sha256,
)


def _json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _science_rows(regional_gain: float, spatial: bool):
    rows = []
    for index in range(32):
        arm_rows = {}
        for arm_index, arm in enumerate(ARMS):
            gain = 0.0 if arm_index == 0 else regional_gain
            real = 1.0 + gain
            arm_rows[arm] = {
                "REAL": real,
                "C_DINO_V": -1.0,
                "P_QUERY": -0.5 if spatial else real,
                "P_REFERENCE": -0.5 if spatial else real,
            }
        rows.append(
            {
                "query_id": f"Q{index:03d}",
                "group_sha256": f"{index:064x}",
                "outer_fold": index % 4 + 1,
                "raw_margin": -1.0 if index < 16 else 1.0,
                "arms": arm_rows,
            }
        )
    return rows


def test_i0_seal_closes_actual_594_rows_and_49_isolated_groups() -> None:
    paths = {
        "loss_join": ROOT / "results/dino_rcde_sr0_mt_loss_join_v1/loss_join.json",
        "i0_postjoin": ROOT / "results/dino_rcde_sr0_mt_i0_v1/formal_job5075941/postjoin_ledger.json",
        "role_free_pair": ROOT / "results/dino_rcde_sr0_mt_role_free_pair_address_v1/role_free_pair_address_manifest.json",
        "fold_schedule": ROOT / "protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json",
    }
    value = build_sealed_metadata(
        loss_join=_json(paths["loss_join"].relative_to(ROOT).as_posix()),
        i0_postjoin=_json(paths["i0_postjoin"].relative_to(ROOT).as_posix()),
        role_free=_json(paths["role_free_pair"].relative_to(ROOT).as_posix()),
        fold_schedule=_json(paths["fold_schedule"].relative_to(ROOT).as_posix()),
        source_bindings={f"{name}_sha256": file_sha256(path) for name, path in paths.items()},
    )
    assert value["query_count"] == 594
    assert value["group_count"] == 49
    assert value["fold_group_count"] == {"1": 13, "2": 12, "3": 12, "4": 12}
    assert value["group_fold_isolation"] is True
    assert value["model_forward_count"] == 0
    assert all("supergroup" not in row and "identity" not in row for row in value["records"])

    poisoned = _json(paths["i0_postjoin"].relative_to(ROOT).as_posix())
    poisoned["records"][0]["target_physical_row"] += 1
    with pytest.raises(TrackRMetadataSealError, match="target/rival row/member"):
        build_sealed_metadata(
            loss_join=_json(paths["loss_join"].relative_to(ROOT).as_posix()),
            i0_postjoin=poisoned,
            role_free=_json(paths["role_free_pair"].relative_to(ROOT).as_posix()),
            fold_schedule=_json(paths["fold_schedule"].relative_to(ROOT).as_posix()),
            source_bindings={},
        )


@pytest.mark.parametrize("gain,spatial", ((0.0, True), (0.5, False), (0.5, True)))
def test_independent_statistics_is_exact_without_importing_production(
    gain: float, spatial: bool
) -> None:
    rows = _science_rows(gain, spatial)
    if gain and not spatial:
        for row in rows[:8]:
            row["arms"][ARMS[0]]["REAL"] = -0.1
            row["arms"][ARMS[0]]["P_QUERY"] = -0.1
            row["arms"][ARMS[0]]["P_REFERENCE"] = -0.1
    assert evaluate_track_r(rows, repetitions=199) == independent_evaluate_track_r(
        rows, repetitions=199
    )
    source = (ROOT / "programs/dino_rcde_track_r_statistics_independent_v1.py").read_text()
    assert "from rc_aslo_xf.dino_rcde_track_r_statistics_v1" not in source
    assert "import rc_aslo_xf.dino_rcde_track_r_statistics_v1" not in source
    assert "evaluate_track_r(" not in source.replace("independent_evaluate_track_r(", "")


def test_v123_is_claim_scoped_and_stops_after_independent_validation() -> None:
    contract = (ROOT / "plan/DINO_RCDE_TRACK_R_V123_SCIENTIFIC_REDUCTION_CONTRACT_V1_20260822.md").read_text()
    reducer = (ROOT / "programs/reduce_dino_rcde_track_r_science_v1.py").read_text()
    validator = (ROOT / "programs/validate_dino_rcde_track_r_science_v1.py").read_text()
    launcher = (ROOT / "slurm/dino_rcde_track_r_science_v1.sbatch").read_text()
    controller = (ROOT / "slurm/dino_rcde_track_r_post_p_utility_controller_v1.sbatch").read_text()
    promotion = (ROOT / "slurm/dino_rcde_track_r_v123_promotion_controller_v1.sbatch").read_text()
    assert "maxT covers only" in contract
    assert "Decision B may be GO while the spatial boundary is nonspatial" in contract
    assert '"full_gallery_retrieval_claim": False' in reducer
    assert '"hold_switch_claim": False' in reducer
    assert '"ownership_claim": False' in reducer
    assert "independent_evaluate_track_r" in validator
    assert "evaluate_track_r" not in validator.replace("independent_evaluate_track_r", "")
    assert "postjoin_model_forward_count" in reducer and "postjoin_model_forward_count" in validator
    assert "DINO_RCDE_TRACK_R_SCIENTIFIC_INDEPENDENT_VALIDATION_PASS" in launcher
    assert "automatic_stage_advance" in launcher
    assert "V122_P_UTILITY_COMPARATOR_COMPLETE_AWAITING_SEPARATE_V123_AUTHORITY" in controller
    assert "freeze_dino_rcde_track_r_science_authority_v123.py" not in controller
    assert "dino_rcde_track_r_science_v1.sbatch" not in controller
    assert "freeze_dino_rcde_track_r_science_authority_v123.py" in promotion
    assert "V123_AUTHORITY_FROZEN_AND_SCIENCE_DEPENDENCY_RELEASED" in promotion
    assert "\nsbatch " not in promotion and "=$(sbatch" not in promotion
    assert "v123_registration.json" in promotion and "v123_registration.json" in launcher
    assert "v123_promotion.json" in launcher


def test_v123_paths_and_parent_are_frozen() -> None:
    freezer = (ROOT / "programs/freeze_dino_rcde_track_r_science_authority_v123.py").read_text()
    assert 'AUTH = "registry/current_authority_v123_20260822.json"' in freezer
    assert 'PARENT = "registry/current_authority_v122_20260821.json"' in freezer
    assert "dino_rcde_track_r_p_v2_utility_comparator_validation_v1/result.json" in freezer
    assert "expected_i0_seal_logical_sha256" in freezer
    assert "for ordinal in range(50):" in freezer
    assert '"automatic_stage_advance": False' in freezer
    assert '"next_authorized_stage": None' in freezer
    assert 'PROMOTION_CONTROLLER = "slurm/dino_rcde_track_r_v123_promotion_controller_v1.sbatch"' in freezer
    assert '"v123_promotion_controller": H.bind(PROMOTION_CONTROLLER)' in freezer
    assert '"v123_autochain_registration"' in freezer


def test_v123_scheduler_registration_is_fixed_and_science_preserving() -> None:
    value = _json("results/dino_rcde_track_r_autochain_v1/v123_registration.json")
    assert value["status"] == "V123_AUTOMATIC_CONTINUATION_REGISTERED"
    assert value["repair_job_id"] == 5104783
    assert value["comparator_job_id"] == 5102487
    assert value["promotion_dependency"] == "afterok:5102487"
    assert value["promotion_job_id"] == 5104905
    assert value["science_job_id"] == 5104906
    assert value["science_dependency"] == "afterok:5104905"
    assert value["automatic_stage_advance_to_v123"] is True
    assert value["automatic_stage_advance_after_v123"] is False
    assert value["model_or_science_execution_count"] == 0
    assert value["scientific_GO_or_NO_GO"] is None
    assert value["bindings"]["post_p_controller"]["sha256"] == (
        _json("registry/current_authority_v122_20260821.json")["bindings"]["post_p_controller"]["sha256"]
    )
    validator_source = (
        ROOT / "programs/validate_dino_rcde_track_r_v123_autochain_registration_v1.py"
    ).read_text()
    assert "import register_dino_rcde_track_r_v123_autochain_v1" not in validator_source
