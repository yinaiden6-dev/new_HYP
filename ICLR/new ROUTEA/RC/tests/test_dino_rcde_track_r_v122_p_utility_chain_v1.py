from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))

import freeze_dino_rcde_track_r_p_v2_utility_authority_v121 as freezer  # noqa: E402


def source(relative: str) -> str:
    text = (ROOT / relative).read_text(encoding="utf-8")
    ast.parse(text) if relative.endswith(".py") else None
    return text


def test_v122_interface_and_target_free_boundary_are_fixed() -> None:
    assert freezer.AUTH == "registry/current_authority_v122_20260821.json"
    assert freezer.PARENT == "registry/current_authority_v121_20260821.json"
    assert freezer.STATUS == "DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARDS_AUTHORIZED"
    producer = source("programs/materialize_dino_rcde_track_r_p_v2_utility_shard_v1.py")
    assert "loss_join" not in producer
    assert '"target_rival_read_count": 0' in producer
    assert '"dino_token_read_count": 0' in producer
    assert "256 * len(rows)" in producer
    assert "OUTER_TRAIN_REFIT" in producer
    assert "score_compact_deployable_direction" in producer
    assert "decision_from_compact_output" in producer
    assert "fanout_compact_query_sources" not in producer


def test_v122_validator_is_full_replay_and_reducer_is_lineage_closed() -> None:
    validator = source("programs/validate_dino_rcde_track_r_p_v2_utility_shard_v1.py")
    reducer = source("programs/reduce_dino_rcde_track_r_p_v2_utility_v1.py")
    aggregate = source("programs/validate_dino_rcde_track_r_p_v2_utility_comparator_v1.py")
    assert "replay_materialize" in validator
    assert '"independent_full_replay": True' in validator
    assert "runtime_source_bindings" in reducer
    assert "validation.get(\"independent_full_replay\") is True" in reducer
    assert "comparator.get(\"records\") == records" in aggregate


def test_v122_staged_resources_and_controllers_are_exact() -> None:
    launcher = source("slurm/dino_rcde_track_r_p_v2_utility_shards_v1.sbatch")
    shard0 = source("slurm/dino_rcde_track_r_post_p_utility_shard0_controller_v1.sbatch")
    post_p = source("slurm/dino_rcde_track_r_post_p_utility_controller_v1.sbatch")
    for literal in (
        "#SBATCH --partition=cpuonly",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=65536",
        "#SBATCH --time=06:00:00",
        "#SBATCH --array=0",
    ):
        assert literal in launcher
    assert "--array=1-49%8" in shard0
    assert "DINO_RCDE_TRACK_R_P_V2_UTILITY_COMPARATOR_INDEPENDENT_VALIDATION_PASS" in post_p
    assert "V122_P_UTILITY_COMPARATOR_COMPLETE_AWAITING_SEPARATE_V123_AUTHORITY" in post_p
    assert "freeze_dino_rcde_track_r_science_authority_v123.py" not in post_p
    assert "dino_rcde_track_r_science_v1.sbatch" not in post_p
    assert "automatic_stage_advance" in post_p
    assert (ROOT / "programs/freeze_dino_rcde_track_r_science_authority_v123.py").is_file()
    assert (ROOT / "slurm/dino_rcde_track_r_science_v1.sbatch").is_file()
    for relative in (
        "slurm/dino_rcde_track_r_p_v2_utility_shards_v1.sbatch",
        "slurm/dino_rcde_track_r_post_oof_controller_v2.sbatch",
        "slurm/dino_rcde_track_r_post_p_utility_shard0_controller_v1.sbatch",
        "slurm/dino_rcde_track_r_post_p_utility_controller_v1.sbatch",
    ):
        subprocess.run(["bash", "-n", str(ROOT / relative)], check=True)


def test_v122_runtime_closure_contains_all_program_roots() -> None:
    paths, edges = freezer.discover_runtime_closure(freezer.RUNTIME_ROOTS)
    assert len(paths) > 20
    assert len(edges) > 20
    assert "programs/materialize_dino_rcde_sr0_mt_role_free_pair_feature_cache_shard_v1.py" in paths
    assert "src/rc_aslo_xf/dino_rcde_sr0_mt_p_runtime_v1.py" in paths


def test_v122_package_initializer_binding_is_the_real_package_initializer() -> None:
    authority = freezer.build()
    assert authority["bindings"]["package_initializer"]["path"] == "src/rc_aslo_xf/__init__.py"
    assert authority["authority_revision"] == 5
    assert authority["failed_job_id"] == 5102194
    assert authority["repair_scope"] == "EXECUTION_INTERFACE_ONLY_SINGLE_OUTER_HEAD"
    assert authority["bindings"]["repair_addendum"]["path"].endswith(
        "V122_SINGLE_OUTER_SCORER_REPAIR_ADDENDUM_V1_20260823.md"
    )
    assert authority["resource_contract_revision"] == 2
    assert authority["resource_repair_scope"] == "SCHEDULER_TIME_LIMIT_ONLY_4H"
    assert authority["resource_validation_repair_scope"] == "INDEPENDENT_RESOURCE_EVIDENCE_REPLAY"
    assert authority["resource_evidence_hardening_scope"] == "FULL_PATH_AND_SCHEDULER_RECEIPT_REPLAY"
    assert authority["full_shard0_canary_job_id"] == 5102372
    assert authority["resource_contract"]["walltime_seconds"] == 14400
    assert authority["resource_contract"]["launcher_header_walltime_seconds"] == 21600
    assert authority["resource_contract"]["scheduler_override_job_ids"] == [
        5102240,
        5102486,
    ]
