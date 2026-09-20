"""Exact V3 authority whitelist including launcher-test transitive closure."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
from typing import Any, Mapping

from . import dino_rcde_track_r_v124_e1_authority_v2 as V2


RC_ROOT = V2.RC_ROOT
PACKAGE_ROOT = V2.PACKAGE_ROOT
SCHEMA = "rc_dino_rcde_track_r_v124_e1_v3_authority_20260826"
STATUS = "DINO_RCDE_TRACK_R_V124_E1_V3_EXECUTION_AUTHORIZED"
PAIR_SHA256 = V2.PAIR_SHA256
CANDIDATE_AXIS_SHA256 = V2.CANDIDATE_AXIS_SHA256
V_CHECKPOINT_FILE_SHA256 = V2.V_CHECKPOINT_FILE_SHA256
V_CHECKPOINT_STATE_SHA256 = V2.V_CHECKPOINT_STATE_SHA256
FAMILIES = V2.FAMILIES
ARMS = V2.ARMS

BASE_BINDINGS = frozenset(
    {
        "contract_v1",
        "contract_v2",
        "contract_v3",
        "review_v1_authority",
        "review_v1_terminal_rejection",
        "review_v2_terminal_rejection",
        "review_v3_authority",
        "review_v3_terminal_validation",
        "v124_core_authority",
        "v124_core_result",
        "v124_core_validation",
        "phase_a_retry1_authority",
        "phase_a_artifact",
        "phase_a_artifact_receipt",
        "phase_a_artifact_commit",
        "phase_a_normalized_validation",
        "phase_a_normalized_validation_receipt",
        "phase_a_normalized_validation_commit",
        "phase_a_final_validation",
        "phase_a_final_validation_receipt",
        "phase_a_final_validation_commit",
        "phase_b_authority",
        "phase_b_result",
        "phase_b_validation",
        "role_free_pair_manifest",
        "role_free_pair_validation",
        "gallery_cache",
        "identity_contract",
        "identity_registry",
        "canonical_geometry_payload",
        "canonical_geometry_receipt",
        "redacted_schedule",
        "redacted_cache_index",
        "fold_schedule",
        "source_manifest",
        "source_manifest_validation",
        "source_artifact",
        "source_artifact_validation",
        "v121_authority",
        "fold2_v_checkpoint",
        "producer_v3",
        "independent_validator_v3",
        "atomic_family_v3",
        "authority_runtime_v3",
        "tests_v3",
        "launcher_v3",
        "launcher_test_full_c128_c_dino_v1",
        "launcher_test_v_runtime_v2",
        "launcher_test_phase_b_all_patch_v1",
        "launcher_test_e1_v3",
    }
)

CLOSURE_ROOTS = (
    "programs/run_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3.py",
    "programs/validate_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_independent_v3.py",
    "src/rc_aslo_xf/dino_rcde_track_r_v124_e1_atomic_family_v3.py",
    "src/rc_aslo_xf/dino_rcde_track_r_v124_e1_authority_v3.py",
    "programs/freeze_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v3.py",
    "tests/test_dino_rcde_track_r_full_c128_c_dino_v1.py",
    "tests/test_dino_rcde_sr0_mt_v_runtime_v2.py",
    "tests/test_dino_rcde_track_r_phase_b_all_patch_v1.py",
    "tests/test_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3.py",
)


class E1AuthorityV3Error(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise E1AuthorityV3Error(message)


def _module_file(module: str) -> Path | None:
    package = V2._module_file(module)
    if package is not None:
        return package
    test = RC_ROOT / "tests" / f"{module}.py"
    return test if test.is_file() else None


def _imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    output: set[Path] = set()
    in_package = PACKAGE_ROOT in path.parents
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                file = _module_file(alias.name)
                if file is not None:
                    output.add(file.resolve())
        elif isinstance(node, ast.ImportFrom):
            if node.level and in_package:
                if node.module:
                    file = PACKAGE_ROOT / f"{node.module.split('.')[-1]}.py"
                    if file.is_file():
                        output.add(file.resolve())
                else:
                    for alias in node.names:
                        file = PACKAGE_ROOT / f"{alias.name}.py"
                        if file.is_file():
                            output.add(file.resolve())
            elif node.module:
                file = _module_file(node.module)
                if file is not None:
                    output.add(file.resolve())
                if node.module == "rc_aslo_xf":
                    for alias in node.names:
                        file = PACKAGE_ROOT / f"{alias.name}.py"
                        if file.is_file():
                            output.add(file.resolve())
    if in_package:
        output.add((PACKAGE_ROOT / "__init__.py").resolve())
    return output


def discover_import_closure() -> tuple[str, ...]:
    pending = [(RC_ROOT / relative).resolve() for relative in CLOSURE_ROOTS]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        require(
            path.is_file() and not path.is_symlink() and RC_ROOT in path.parents,
            f"V3 closure root/dependency absent: {path}",
        )
        if path in seen:
            continue
        seen.add(path)
        pending.extend(item for item in _imports(path) if item not in seen)
    return tuple(
        sorted(path.relative_to(RC_ROOT).as_posix() for path in seen)
    )


def closure_key(relative: str) -> str:
    return "import_closure__" + re.sub(
        r"[^a-zA-Z0-9]+", "_", relative
    ).strip("_")


def required_bindings() -> frozenset[str]:
    return BASE_BINDINGS | frozenset(
        closure_key(relative) for relative in discover_import_closure()
    )


def validate_authority_envelope(value: Mapping[str, Any]) -> None:
    bindings = value.get("bindings")
    require(
        isinstance(bindings, Mapping)
        and set(bindings) == required_bindings(),
        "E1 V3 exact binding/test/import whitelist drift",
    )
    for row in bindings.values():
        require(
            isinstance(row, Mapping)
            and set(row) == {"path", "bytes", "sha256"},
            "E1 V3 binding field set drift",
        )
    require(
        value.get("schema_version") == SCHEMA
        and value.get("status") == STATUS
        and value.get("logical_sha256") == V2.logical_sha256(value)
        and value.get("independent_review_status") == "PASS"
        and value.get("execution_ordinal") == 0
        and value.get("query_id") == "DIFFICULT-0000"
        and value.get("source_fold") == 2
        and value.get("candidate_count") == 128
        and value.get("candidate_axis_sha256") == CANDIDATE_AXIS_SHA256
        and value.get("pair_sha256") == PAIR_SHA256
        and value.get("v_checkpoint_file_sha256")
        == V_CHECKPOINT_FILE_SHA256
        and value.get("v_checkpoint_state_sha256")
        == V_CHECKPOINT_STATE_SHA256
        and tuple(value.get("family_sequence", ())) == FAMILIES
        and tuple(value.get("arm_sequence", ())) == ARMS
        and value.get("expected_pair_arm_record_count") == 9
        and value.get("expected_directional_term_count") == 36
        and value.get("device_schedule")
        == {
            "all_patch_device": "cpu",
            "regional_device": "cuda:0",
            "model_load_count": 1,
            "model_device_transition_count": 1,
        }
        and value.get("resource_contract")
        == {
            "partition": "accelerated",
            "allowed_partitions": ["accelerated"],
            "gpu_count": 1,
            "cpus_per_task": 8,
            "memory_megabytes": 128000,
            "walltime_seconds": 7200,
        }
        and value.get("model_load_authorized") is True
        and value.get("model_forward_authorized") is True
        and value.get("p_training_authorized") is False
        and value.get("v_training_authorized") is False
        and value.get("model_backward_authorized") is False
        and value.get("model_update_authorized") is False
        and value.get("target_rival_join_authorized") is False
        and value.get("postjoin_authorized") is False
        and value.get("scientific_reduction_authorized") is False
        and value.get("automatic_submit_authorized") is False
        and value.get("scientific_GO_or_NO_GO") is None
        and value.get("automatic_stage_advance") is False
        and value.get("next_authorized_stage") is None
        and value.get("import_closure") == list(discover_import_closure())
        and value.get("import_closure_sha256")
        == V2.canonical_sha256(list(discover_import_closure())),
        "E1 V3 authority envelope/capability drift",
    )
    explicit_test_paths = {
        "launcher_test_full_c128_c_dino_v1": "tests/test_dino_rcde_track_r_full_c128_c_dino_v1.py",
        "launcher_test_v_runtime_v2": "tests/test_dino_rcde_sr0_mt_v_runtime_v2.py",
        "launcher_test_phase_b_all_patch_v1": "tests/test_dino_rcde_track_r_phase_b_all_patch_v1.py",
        "launcher_test_e1_v3": "tests/test_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3.py",
    }
    require(
        all(bindings[key]["path"] == path for key, path in explicit_test_paths.items()),
        "launcher test explicit binding drift",
    )


def read_authority(path: Path, expected_sha256: str) -> dict[str, Any]:
    authority_path = path.resolve()
    require(
        authority_path.is_file()
        and not authority_path.is_symlink()
        and V2.file_sha256(authority_path) == expected_sha256,
        "E1 V3 authority physical drift",
    )
    value = json.loads(authority_path.read_text(encoding="ascii"))
    validate_authority_envelope(value)
    for name, row in value["bindings"].items():
        raw = Path(row["path"])
        bound = (raw if raw.is_absolute() else RC_ROOT / raw).resolve()
        require(
            bound.is_file()
            and not bound.is_symlink()
            and bound.stat().st_size == row["bytes"]
            and V2.file_sha256(bound) == row["sha256"],
            f"E1 V3 bound file drift: {name}",
        )
    for relative in discover_import_closure():
        require(
            value["bindings"][closure_key(relative)]["path"] == relative,
            f"E1 V3 closure path drift: {relative}",
        )
    return value


__all__ = [
    "ARMS",
    "BASE_BINDINGS",
    "CANDIDATE_AXIS_SHA256",
    "CLOSURE_ROOTS",
    "E1AuthorityV3Error",
    "FAMILIES",
    "PAIR_SHA256",
    "SCHEMA",
    "STATUS",
    "V_CHECKPOINT_FILE_SHA256",
    "V_CHECKPOINT_STATE_SHA256",
    "closure_key",
    "discover_import_closure",
    "read_authority",
    "required_bindings",
    "validate_authority_envelope",
]
