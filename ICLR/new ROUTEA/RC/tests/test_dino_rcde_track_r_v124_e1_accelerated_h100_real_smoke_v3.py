from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_dino_rcde_track_r_v124_e1_accelerated_h100_real_smoke_v3 as producer  # noqa: E402
import validate_dino_rcde_track_r_v124_e1_accelerated_h100_real_smoke_independent_v3 as validator  # noqa: E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_real_smoke_resource_authority_v3 as AUTH  # noqa: E402


def relogical(value: dict) -> None:
    value["logical_sha256"] = AUTH.BASE.V2.logical_sha256(value)


def test_h100_candidate_exact126_and_zero_start_successor() -> None:
    value = AUTH.build_candidate()
    assert len(value["bindings"]) == len(value["binding_modes"]) == 129
    assert len(value["import_closure"]) == 65
    assert len(value["smoke_overlay_paths"]) == 14
    assert len(value["smoke_overlay_import_closure"]) == 10
    assert value["resource_contract"] == {
        "partition": "accelerated-h100",
        "allowed_partitions": ["accelerated-h100"],
        "gpu_count": 1,
        "cpus_per_task": 8,
        "memory_megabytes": 128000,
        "walltime_seconds": 7200,
    }
    assert value["never_consume_superseded_v2_outputs"] is True
    assert value["superseded_v2_smoke_output_read_authorized"] is False
    assert value["eligible_as_e1_result"] is False
    cancellation = value["bindings"]["job5110073_zero_start_cancellation"]
    assert cancellation["path"] == AUTH.OVERLAY_PATHS[
        "job5110073_zero_start_cancellation"
    ]
    assert not (ROOT / AUTH.AUTHORITY_PATH).exists()


@pytest.mark.parametrize(
    "partition", ["accelerated", "dev_accelerated-h100", "cpuonly"]
)
def test_wrong_partition_alias_poison(partition: str) -> None:
    value = copy.deepcopy(AUTH.build_candidate())
    value["resource_contract"]["partition"] = partition
    value["resource_contract"]["allowed_partitions"] = [partition]
    relogical(value)
    with pytest.raises(Exception):
        AUTH.validate_authority_envelope(value)


def test_missing_or_swapped_v2_disposition_bindings_poison() -> None:
    missing = copy.deepcopy(AUTH.build_candidate())
    missing["bindings"].pop("job5110073_zero_start_cancellation")
    missing["binding_modes"].pop("job5110073_zero_start_cancellation")
    missing["smoke_overlay_paths"].pop(
        "job5110073_zero_start_cancellation"
    )
    relogical(missing)
    with pytest.raises(Exception):
        AUTH.validate_authority_envelope(missing)
    swapped = copy.deepcopy(AUTH.build_candidate())
    first = swapped["bindings"]["superseded_smoke_v2_authority"]
    second = swapped["bindings"]["job5110073_zero_start_cancellation"]
    swapped["bindings"]["superseded_smoke_v2_authority"] = second
    swapped["bindings"]["job5110073_zero_start_cancellation"] = first
    relogical(swapped)
    with pytest.raises(Exception):
        AUTH.validate_authority_envelope(swapped)


def test_superseded_v2_output_read_authorization_poison() -> None:
    value = copy.deepcopy(AUTH.build_candidate())
    value["superseded_v2_smoke_output_read_authorized"] = True
    relogical(value)
    with pytest.raises(Exception):
        AUTH.validate_authority_envelope(value)


@pytest.mark.parametrize(
    "key",
    [
        "smoke_producer_v1_grandparent",
        "smoke_validator_v1_grandparent",
        "smoke_authority_v1_grandparent",
    ],
)
def test_missing_v1_grandparent_binding_poison(key: str) -> None:
    value = copy.deepcopy(AUTH.build_candidate())
    value["bindings"].pop(key)
    value["binding_modes"].pop(key)
    value["smoke_overlay_paths"].pop(key)
    relogical(value)
    with pytest.raises(Exception):
        AUTH.validate_authority_envelope(value)


def test_overlay_closure_rejects_unbound_local_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = AUTH._module_file
    unbound = (
        ROOT
        / "programs/reject_dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v1_terminal_v1.py"
    )

    def injected(module: str):
        if module == "argparse":
            return unbound
        return original(module)

    monkeypatch.setattr(AUTH, "_module_file", injected)
    with pytest.raises(Exception, match="unbound local import"):
        AUTH.discover_overlay_import_closure()


def test_literal_h100_launcher_paths_and_partition_exact_once() -> None:
    value = AUTH.build_candidate()
    launcher = (
        ROOT / AUTH.OVERLAY_PATHS["smoke_launcher"]
    ).read_text(encoding="utf-8")
    AUTH.validate_launcher_source(value, launcher)
    assert launcher.count("#SBATCH --partition=accelerated-h100") == 1
    assert "dev_accelerated-h100" not in launcher
    assert "#SBATCH --partition=accelerated\n" not in launcher
    for path in AUTH.LAUNCHER_TEST_PATHS:
        assert launcher.count(path) == 1
    for path in AUTH.LAUNCHER_EXECUTABLE_PATHS.values():
        assert launcher.count(path) == 1


def test_h100_overlay_ast_closure_and_fresh_namespaces() -> None:
    closure = set(AUTH.discover_overlay_import_closure())
    assert len(closure) == 10
    assert AUTH.OVERLAY_PATHS["smoke_producer_v2_parent"] in closure
    assert AUTH.OVERLAY_PATHS["smoke_validator_v2_parent"] in closure
    assert AUTH.OVERLAY_PATHS["smoke_authority_v2_parent"] in closure
    assert AUTH.OVERLAY_PATHS["smoke_producer_v1_grandparent"] in closure
    assert AUTH.OVERLAY_PATHS["smoke_validator_v1_grandparent"] in closure
    assert AUTH.OVERLAY_PATHS["smoke_authority_v1_grandparent"] in closure
    assert "accelerated_h100_real_smoke_v3_committed" in AUTH.PRODUCER_NAMESPACE
    assert (
        "accelerated_h100_real_smoke_validation_v3_committed"
        in AUTH.VALIDATION_NAMESPACE
    )
    assert "accelerated_real_smoke_v2_committed" not in AUTH.PRODUCER_NAMESPACE


def test_h100_wrappers_configure_fresh_nonpromotable_paths() -> None:
    producer._configure()
    validator._configure()
    assert producer.PUBLIC_DIR == ROOT / AUTH.PRODUCER_NAMESPACE
    assert validator.PRODUCER_DIR == ROOT / AUTH.PRODUCER_NAMESPACE
    assert validator.PUBLIC_DIR == ROOT / AUTH.VALIDATION_NAMESPACE
    assert producer.STATUS.endswith("NONPROMOTABLE")
    assert validator.STATUS.endswith("NONPROMOTABLE")


def test_h100_authority_and_all_output_namespaces_absent() -> None:
    for relative in (
        AUTH.AUTHORITY_PATH,
        AUTH.PRODUCER_NAMESPACE,
        AUTH.VALIDATION_NAMESPACE,
        AUTH.SUPERSEDED_V2_PRODUCER,
        AUTH.SUPERSEDED_V2_VALIDATION,
        AUTH.FORMAL_PRODUCER,
        AUTH.FORMAL_VALIDATION,
    ):
        path = AUTH._raw_path(relative)
        assert not path.exists()
        assert not path.is_symlink()
