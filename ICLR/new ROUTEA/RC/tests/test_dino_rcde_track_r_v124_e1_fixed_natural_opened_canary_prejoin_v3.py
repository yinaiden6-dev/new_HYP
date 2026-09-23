from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))
sys.path.insert(0, str(ROOT / "tests"))

import test_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v2 as V2TEST  # noqa: E402
import freeze_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v3 as REVIEW  # noqa: E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_atomic_family_v3 as ATOMIC  # noqa: E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_authority_v3 as AUTH  # noqa: E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_atomic_family_v2 as ATOMIC_V2  # noqa: E402


V2_REJECTION = ROOT / "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_v2_terminal_rejection_v1_20260826.json"


def artifact() -> dict:
    value = {
        "schema_version": "test_v3_artifact",
        "status": "READY",
        "tensor": torch.arange(5),
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = AUTH.V2.canonical_sha256(
        {
            "schema_version": value["schema_version"],
            "status": value["status"],
            "tensor_sha256": "test-v3",
            "scientific_GO_or_NO_GO": None,
            "automatic_stage_advance": False,
            "next_authorized_stage": None,
        }
    )
    return value


def result_builder(_path: Path) -> dict:
    value = {
        "schema_version": "test_v3_result",
        "status": "READY",
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = AUTH.V2.logical_sha256(value)
    return value


def authority_fixture() -> dict:
    bindings = {
        key: {"path": "dummy", "bytes": 0, "sha256": "0" * 64}
        for key in AUTH.required_bindings()
    }
    explicit = {
        "launcher_test_full_c128_c_dino_v1": "tests/test_dino_rcde_track_r_full_c128_c_dino_v1.py",
        "launcher_test_v_runtime_v2": "tests/test_dino_rcde_sr0_mt_v_runtime_v2.py",
        "launcher_test_phase_b_all_patch_v1": "tests/test_dino_rcde_track_r_phase_b_all_patch_v1.py",
        "launcher_test_e1_v3": "tests/test_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3.py",
    }
    for key, path in explicit.items():
        bindings[key]["path"] = path
    value = {
        "schema_version": AUTH.SCHEMA,
        "status": AUTH.STATUS,
        "bindings": bindings,
        "independent_review_status": "PASS",
        "execution_ordinal": 0,
        "query_id": "DIFFICULT-0000",
        "source_fold": 2,
        "candidate_count": 128,
        "candidate_axis_sha256": AUTH.CANDIDATE_AXIS_SHA256,
        "pair_sha256": AUTH.PAIR_SHA256,
        "v_checkpoint_file_sha256": AUTH.V_CHECKPOINT_FILE_SHA256,
        "v_checkpoint_state_sha256": AUTH.V_CHECKPOINT_STATE_SHA256,
        "family_sequence": list(AUTH.FAMILIES),
        "arm_sequence": list(AUTH.ARMS),
        "expected_pair_arm_record_count": 9,
        "expected_directional_term_count": 36,
        "device_schedule": {
            "all_patch_device": "cpu",
            "regional_device": "cuda:0",
            "model_load_count": 1,
            "model_device_transition_count": 1,
        },
        "resource_contract": {
            "partition": "accelerated",
            "allowed_partitions": ["accelerated"],
            "gpu_count": 1,
            "cpus_per_task": 8,
            "memory_megabytes": 128000,
            "walltime_seconds": 7200,
        },
        "model_load_authorized": True,
        "model_forward_authorized": True,
        "p_training_authorized": False,
        "v_training_authorized": False,
        "model_backward_authorized": False,
        "model_update_authorized": False,
        "target_rival_join_authorized": False,
        "postjoin_authorized": False,
        "scientific_reduction_authorized": False,
        "automatic_submit_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "import_closure": list(AUTH.discover_import_closure()),
        "import_closure_sha256": AUTH.V2.canonical_sha256(
            list(AUTH.discover_import_closure())
        ),
    }
    value["logical_sha256"] = AUTH.V2.logical_sha256(value)
    return value


def publish(public: Path, authority: str = "a" * 64) -> None:
    ATOMIC.publish_producer_family(
        public,
        authority_sha256=authority,
        artifact=artifact(),
        result_builder=result_builder,
    )


def test_v2_candidate_absent_and_terminally_rejected() -> None:
    value = json.loads(V2_REJECTION.read_text(encoding="ascii"))
    assert value["terminal_review_decision"] == "FAIL"
    assert value["review_v2_authority_created"] is False
    assert value["smoke_executed"] is False
    assert value["submission_executed"] is False
    assert value["logical_sha256"] == AUTH.V2.logical_sha256(value)
    assert not (
        ROOT
        / "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v2_20260826.json"
    ).exists()


def test_raw_public_directory_symlink_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real-family"
    publish(real)
    alias = tmp_path / "alias-family"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ATOMIC.E1AtomicFamilyV3Error, match="symlink"):
        ATOMIC.validate_family(
            alias,
            authority_sha256="a" * 64,
            family_role="E1_V3_PRODUCER_FAMILY",
            producer=True,
        )


def test_member_symlink_rejected(tmp_path: Path) -> None:
    public = tmp_path / "member-family"
    publish(public)
    member = public / "producer_result.json"
    backup = tmp_path / "producer_result.backup.json"
    member.chmod(0o644)
    member.rename(backup)
    member.symlink_to(backup)
    with pytest.raises(ATOMIC.E1AtomicFamilyV3Error, match="regular file"):
        ATOMIC.validate_family(
            public,
            authority_sha256="a" * 64,
            family_role="E1_V3_PRODUCER_FAMILY",
            producer=True,
        )


def test_staging_symlink_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = tmp_path / "staging-family"

    def symlink_stage(*, prefix: str, dir: Path):
        real = Path(dir) / "real-stage-target"
        real.mkdir()
        link = Path(dir) / f"{prefix}symlink"
        link.symlink_to(real, target_is_directory=True)
        return str(link)

    monkeypatch.setattr(ATOMIC.tempfile, "mkdtemp", symlink_stage)
    with pytest.raises(ATOMIC.E1AtomicFamilyV3Error, match="symlink"):
        publish(public)
    assert not public.exists()


def test_family_rows_bind_each_member_logical_sha(tmp_path: Path) -> None:
    public = tmp_path / "logical-family"
    publish(public)
    family = json.loads(
        (public / "family_commit.json").read_text(encoding="ascii")
    )
    assert family["member_count"] == 6
    assert all(
        set(row) == {"path", "bytes", "sha256", "logical_sha256"}
        and len(row["logical_sha256"]) == 64
        for row in family["ordered_members"]
    )
    ATOMIC.validate_family(
        public,
        authority_sha256="a" * 64,
        family_role="E1_V3_PRODUCER_FAMILY",
        producer=True,
    )


def test_tampered_member_logical_row_rejected(tmp_path: Path) -> None:
    public = tmp_path / "logical-tamper"
    publish(public)
    path = public / "family_commit.json"
    path.chmod(0o644)
    family = json.loads(path.read_text(encoding="ascii"))
    family["ordered_members"][0]["logical_sha256"] = "0" * 64
    family["logical_sha256"] = AUTH.V2.logical_sha256(family)
    path.write_text(json.dumps(family, sort_keys=True, indent=2) + "\n")
    path.chmod(0o444)
    with pytest.raises(
        ATOMIC.E1AtomicFamilyV3Error, match="logical member"
    ):
        ATOMIC.validate_family(
            public,
            authority_sha256="a" * 64,
            family_role="E1_V3_PRODUCER_FAMILY",
            producer=True,
        )


def test_v3_atomic_crash_and_reuse(tmp_path: Path) -> None:
    early = tmp_path / "early"
    with pytest.raises(ATOMIC.InjectedFamilyCrash):
        ATOMIC.publish_producer_family(
            early,
            authority_sha256="b" * 64,
            artifact=artifact(),
            result_builder=result_builder,
            crash_after="AFTER_RESULT_COMMIT",
        )
    assert not early.exists()
    public = tmp_path / "reuse"
    publish(public, authority="b" * 64)

    def forbidden(_path: Path):
        raise AssertionError("builder called during V3 reuse")

    _, _, reused = ATOMIC.publish_producer_family(
        public,
        authority_sha256="b" * 64,
        artifact=artifact(),
        result_builder=forbidden,
    )
    assert reused is True


def test_launcher_tests_are_explicit_and_in_transitive_closure() -> None:
    closure = set(AUTH.discover_import_closure())
    tests = {
        "tests/test_dino_rcde_track_r_full_c128_c_dino_v1.py",
        "tests/test_dino_rcde_sr0_mt_v_runtime_v2.py",
        "tests/test_dino_rcde_track_r_phase_b_all_patch_v1.py",
        "tests/test_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3.py",
    }
    assert tests.issubset(closure)
    assert (
        "programs/freeze_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v3.py"
        in closure
    )
    value = authority_fixture()
    AUTH.validate_authority_envelope(value)
    assert all(
        AUTH.closure_key(path) in value["bindings"] for path in closure
    )


def test_missing_launcher_test_and_extra_binding_rejected() -> None:
    missing = authority_fixture()
    missing["bindings"].pop("launcher_test_phase_b_all_patch_v1")
    missing["logical_sha256"] = AUTH.V2.logical_sha256(missing)
    with pytest.raises(AUTH.E1AuthorityV3Error, match="whitelist"):
        AUTH.validate_authority_envelope(missing)
    extra = authority_fixture()
    extra["bindings"]["result_roles"] = {
        "path": "valid",
        "bytes": 1,
        "sha256": "f" * 64,
    }
    extra["logical_sha256"] = AUTH.V2.logical_sha256(extra)
    with pytest.raises(AUTH.E1AuthorityV3Error, match="whitelist"):
        AUTH.validate_authority_envelope(extra)


def test_review_v3_builds_but_remains_absent_and_nonexecuting() -> None:
    value = REVIEW.build_authority()
    assert value["status"] == REVIEW.STATUS
    assert value["terminal_independent_review_status"] == "PENDING"
    assert value["execution_authorized"] is False
    assert value["smoke_authorized"] is False
    assert value["submission_authorized"] is False
    assert value["logical_sha256"] == REVIEW.logical_sha256(value)
    assert not REVIEW.OUTPUT.exists()


def test_v3_launcher_is_accelerated_only_and_runs_exact_four_tests() -> None:
    launcher = (
        ROOT
        / "slurm/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=accelerated" in launcher
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "dev_accelerated-h100" not in launcher
    assert "dev_cpuonly" not in launcher
    assert "\nsbatch " not in launcher
    for path in (
        "tests/test_dino_rcde_track_r_full_c128_c_dino_v1.py",
        "tests/test_dino_rcde_sr0_mt_v_runtime_v2.py",
        "tests/test_dino_rcde_track_r_phase_b_all_patch_v1.py",
        "tests/test_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3.py",
    ):
        assert launcher.count(path) == 1


def test_v2_positive_deep_gate_receipt_remains_bound() -> None:
    # V3 changes publication/closure only; bind the V2 terminal-tested deep core.
    rejection = json.loads(V2_REJECTION.read_text(encoding="ascii"))
    by_path = {row["path"]: row["sha256"] for row in rejection["reviewed_file_bindings"]}
    assert by_path[
        "programs/run_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v2.py"
    ]
    assert by_path[
        "programs/validate_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_independent_v2.py"
    ]
    assert V2TEST.producer.deep_validate_c_col_population is not None
    assert V2TEST.independent.deep_validate_c_col_independent is not None
