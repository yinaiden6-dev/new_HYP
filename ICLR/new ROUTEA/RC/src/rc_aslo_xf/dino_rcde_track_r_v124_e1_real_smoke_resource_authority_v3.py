"""Literal accelerated-h100 resource successor for non-promotable smoke V3."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping

from . import dino_rcde_track_r_v124_e1_authority_v3 as BASE
from . import dino_rcde_track_r_v124_e1_real_smoke_authority_v2 as V2


OVERLAY_PATHS = MappingProxyType(
    {
        "smoke_producer": "programs/run_dino_rcde_track_r_v124_e1_accelerated_h100_real_smoke_v3.py",
        "smoke_independent_validator": "programs/validate_dino_rcde_track_r_v124_e1_accelerated_h100_real_smoke_independent_v3.py",
        "smoke_launcher": "slurm/dino_rcde_track_r_v124_e1_accelerated_h100_real_smoke_v3.sbatch",
        "smoke_authority_runtime": "src/rc_aslo_xf/dino_rcde_track_r_v124_e1_real_smoke_resource_authority_v3.py",
        "smoke_tests": "tests/test_dino_rcde_track_r_v124_e1_accelerated_h100_real_smoke_v3.py",
        "smoke_producer_v2_parent": "programs/run_dino_rcde_track_r_v124_e1_accelerated_real_smoke_v2.py",
        "smoke_validator_v2_parent": "programs/validate_dino_rcde_track_r_v124_e1_accelerated_real_smoke_independent_v2.py",
        "smoke_authority_v2_parent": "src/rc_aslo_xf/dino_rcde_track_r_v124_e1_real_smoke_authority_v2.py",
        "superseded_smoke_v2_authority": "registry/dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v2_20260826.json",
        "superseded_smoke_v2_authority_validation": "registry/dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v2_20260826.independent_validation_receipt_v1.json",
        "job5110073_zero_start_cancellation": "registry/dino_rcde_track_r_v124_e1_smoke_job5110073_zero_start_cancellation_v1_20260826.json",
        "smoke_producer_v1_grandparent": "programs/run_dino_rcde_track_r_v124_e1_accelerated_real_smoke_v1.py",
        "smoke_validator_v1_grandparent": "programs/validate_dino_rcde_track_r_v124_e1_accelerated_real_smoke_independent_v1.py",
        "smoke_authority_v1_grandparent": "src/rc_aslo_xf/dino_rcde_track_r_v124_e1_real_smoke_authority_v1.py",
    }
)
AUTHORITY_PATH = "registry/dino_rcde_track_r_v124_e1_accelerated_h100_real_smoke_authority_v3_20260826.json"
PRODUCER_NAMESPACE = "results/dino_rcde_track_r_v124_e1_v3_accelerated_h100_real_smoke_v3_committed"
VALIDATION_NAMESPACE = "results/dino_rcde_track_r_v124_e1_v3_accelerated_h100_real_smoke_validation_v3_committed"
SUPERSEDED_V2_PRODUCER = V2.PRODUCER_NAMESPACE
SUPERSEDED_V2_VALIDATION = V2.VALIDATION_NAMESPACE
FORMAL_PRODUCER = V2.FORMAL_PRODUCER_NAMESPACE
FORMAL_VALIDATION = V2.FORMAL_VALIDATION_NAMESPACE
LAUNCHER_TEST_PATHS = (
    "tests/test_dino_rcde_track_r_full_c128_c_dino_v1.py",
    "tests/test_dino_rcde_sr0_mt_v_runtime_v2.py",
    "tests/test_dino_rcde_track_r_phase_b_all_patch_v1.py",
    "tests/test_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3.py",
    OVERLAY_PATHS["smoke_tests"],
)
LAUNCHER_EXECUTABLE_PATHS = MappingProxyType(
    {
        "smoke_producer": OVERLAY_PATHS["smoke_producer"],
        "smoke_independent_validator": OVERLAY_PATHS[
            "smoke_independent_validator"
        ],
        "smoke_launcher": OVERLAY_PATHS["smoke_launcher"],
    }
)


class SmokeResourceAuthorityV3Error(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise SmokeResourceAuthorityV3Error(message)


def required_bindings() -> frozenset[str]:
    return BASE.required_bindings() | frozenset(OVERLAY_PATHS)


def _module_file(module: str) -> Path | None:
    if module.startswith("rc_aslo_xf."):
        path = BASE.RC_ROOT / "src/rc_aslo_xf" / f"{module.split('.')[-1]}.py"
        return path if path.is_file() else None
    for directory in ("programs", "tests"):
        path = BASE.RC_ROOT / directory / f"{module}.py"
        if path.is_file():
            return path
    return None


def discover_overlay_import_closure() -> tuple[str, ...]:
    expected = {
        path for path in OVERLAY_PATHS.values() if path.endswith(".py")
    }
    pending = [
        (BASE.RC_ROOT / OVERLAY_PATHS[key]).resolve()
        for key in (
            "smoke_producer",
            "smoke_independent_validator",
            "smoke_authority_runtime",
            "smoke_tests",
        )
    ]
    seen = set()
    base_closure = set(BASE.discover_import_closure())
    while pending:
        path = pending.pop()
        relative = path.relative_to(BASE.RC_ROOT).as_posix()
        require(relative in expected, f"foreign H100 overlay closure: {relative}")
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
                if node.module == "rc_aslo_xf":
                    modules.extend(
                        f"rc_aslo_xf.{alias.name}" for alias in node.names
                    )
            for module in modules:
                candidate = _module_file(module)
                if candidate is None:
                    continue
                candidate_relative = candidate.relative_to(
                    BASE.RC_ROOT
                ).as_posix()
                if candidate_relative in expected:
                    if candidate.resolve() not in seen:
                        pending.append(candidate.resolve())
                else:
                    require(
                        candidate_relative in base_closure,
                        "unbound local import outside base and H100 overlay: "
                        + candidate_relative,
                    )
    discovered = tuple(
        sorted(path.relative_to(BASE.RC_ROOT).as_posix() for path in seen)
    )
    require(set(discovered) == expected, "H100 overlay AST closure drift")
    return discovered


def _raw_path(path_value: str | Path) -> Path:
    raw = Path(path_value)
    return Path(
        os.path.abspath(
            os.fspath(raw if raw.is_absolute() else BASE.RC_ROOT / raw)
        )
    )


def _raw_regular_file(
    path_value: str | Path,
    *,
    expected_mode: int,
    expected_lexical: str | Path,
) -> Path:
    raw = _raw_path(path_value)
    require(raw == _raw_path(expected_lexical), "H100 exact lexical path drift")
    require(os.path.lexists(raw), "H100 bound path absent")
    metadata = os.lstat(raw)
    require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_mode & 0o777 == expected_mode,
        "H100 raw bound file type/mode/symlink drift",
    )
    base = BASE.RC_ROOT if BASE.RC_ROOT in raw.parents else Path("/")
    cursor = base
    for part in raw.relative_to(base).parts:
        cursor /= part
        if os.path.lexists(cursor):
            require(
                not stat.S_ISLNK(os.lstat(cursor).st_mode),
                "H100 symlink component forbidden",
            )
    require(raw.resolve() == raw, "H100 bound path resolves through alias")
    return raw


def _reject_namespace_symlinks() -> None:
    for relative in (
        PRODUCER_NAMESPACE,
        VALIDATION_NAMESPACE,
        SUPERSEDED_V2_PRODUCER,
        SUPERSEDED_V2_VALIDATION,
        FORMAL_PRODUCER,
        FORMAL_VALIDATION,
    ):
        raw = _raw_path(relative)
        if os.path.lexists(raw):
            require(
                not stat.S_ISLNK(os.lstat(raw).st_mode),
                f"H100 smoke/formal namespace symlink: {relative}",
            )


def validate_launcher_source(value: Mapping[str, Any], source: str) -> None:
    for path in LAUNCHER_TEST_PATHS:
        require(source.count(path) == 1, f"H100 launcher test path drift: {path}")
    for key, path in LAUNCHER_EXECUTABLE_PATHS.items():
        require(
            value["bindings"][key]["path"] == path
            and source.count(path) == 1,
            f"H100 launcher executable path drift: {key}",
        )
    require(
        "#SBATCH --partition=accelerated-h100" in source
        and "#SBATCH --gres=gpu:1" in source
        and "dev_accelerated-h100" not in source
        and "#SBATCH --partition=accelerated\n" not in source
        and "accelerated_real_smoke_v2.py" not in source
        and "accelerated_real_smoke_v2.sbatch" not in source,
        "H100 launcher partition/alias drift",
    )


def validate_authority_envelope(value: Mapping[str, Any]) -> None:
    bindings = value.get("bindings")
    require(
        isinstance(bindings, Mapping)
        and set(bindings) == required_bindings(),
        "H100 smoke exact126 binding set drift",
    )
    projected = dict(value)
    projected["bindings"] = {
        key: bindings[key] for key in BASE.required_bindings()
    }
    projected["resource_contract"] = {
        "partition": "accelerated",
        "allowed_partitions": ["accelerated"],
        "gpu_count": 1,
        "cpus_per_task": 8,
        "memory_megabytes": 128000,
        "walltime_seconds": 7200,
    }
    projected["logical_sha256"] = BASE.V2.logical_sha256(projected)
    BASE.validate_authority_envelope(projected)
    require(
        value.get("logical_sha256") == BASE.V2.logical_sha256(value)
        and value.get("execution_scope")
        == "EXECUTION0_ACCELERATED_H100_REAL_SMOKE_V3_ONLY_NONPROMOTABLE"
        and value.get("resource_contract")
        == {
            "partition": "accelerated-h100",
            "allowed_partitions": ["accelerated-h100"],
            "gpu_count": 1,
            "cpus_per_task": 8,
            "memory_megabytes": 128000,
            "walltime_seconds": 7200,
        }
        and value.get("real_smoke") is True
        and value.get("eligible_as_e1_result") is False
        and value.get("never_consume_superseded_v2_outputs") is True
        and value.get("superseded_v2_smoke_output_read_authorized") is False
        and value.get("output_contract")
        == {
            "producer_family": PRODUCER_NAMESPACE,
            "validation_family": VALIDATION_NAMESPACE,
            "append_only": True,
            "atomic_family_required": True,
            "exact_committed_reuse": True,
            "overwrite_or_repair_authorized": False,
            "eligible_as_e1_result": False,
        }
        and value.get("smoke_overlay_paths") == dict(OVERLAY_PATHS)
        and value.get("smoke_overlay_import_closure")
        == list(discover_overlay_import_closure())
        and value.get("smoke_overlay_binding_count") == len(OVERLAY_PATHS)
        and isinstance(value.get("binding_modes"), Mapping)
        and set(value["binding_modes"]) == set(bindings)
        and value.get("authority_file_mode") == 0o444
        and value.get("target_rival_join_authorized") is False
        and value.get("postjoin_authorized") is False
        and value.get("scientific_reduction_authorized") is False
        and value.get("automatic_submit_authorized") is False
        and value.get("scientific_GO_or_NO_GO") is None
        and value.get("automatic_stage_advance") is False
        and value.get("next_authorized_stage") is None,
        "H100 smoke resource/nonpromotable boundary drift",
    )
    paths = []
    for key, expected in OVERLAY_PATHS.items():
        row = bindings[key]
        require(
            isinstance(row, Mapping)
            and set(row) == {"path", "bytes", "sha256"}
            and row["path"] == expected,
            f"H100 smoke key/path drift: {key}",
        )
        paths.append(str(row["path"]))
    require(len(paths) == len(set(paths)) == 14, "H100 overlay alias drift")
    explicit_tests = {
        "launcher_test_full_c128_c_dino_v1": LAUNCHER_TEST_PATHS[0],
        "launcher_test_v_runtime_v2": LAUNCHER_TEST_PATHS[1],
        "launcher_test_phase_b_all_patch_v1": LAUNCHER_TEST_PATHS[2],
        "launcher_test_e1_v3": LAUNCHER_TEST_PATHS[3],
        "smoke_tests": LAUNCHER_TEST_PATHS[4],
    }
    require(
        all(bindings[key]["path"] == path for key, path in explicit_tests.items()),
        "H100 executed test path drift",
    )
    launcher = _raw_path(bindings["smoke_launcher"]["path"])
    validate_launcher_source(value, launcher.read_text(encoding="utf-8"))


def _binding(path_value: str) -> dict[str, object]:
    path = _raw_regular_file(
        path_value,
        expected_mode=_raw_path(path_value).stat().st_mode & 0o777,
        expected_lexical=path_value,
    )
    return {
        "path": path_value,
        "bytes": path.stat().st_size,
        "sha256": BASE.V2.file_sha256(path),
    }


def build_candidate() -> dict[str, Any]:
    v2_path = _raw_path(
        OVERLAY_PATHS["superseded_smoke_v2_authority"]
    )
    v2 = json.loads(v2_path.read_text(encoding="ascii"))
    v2_validation_path = _raw_path(
        OVERLAY_PATHS["superseded_smoke_v2_authority_validation"]
    )
    v2_validation = json.loads(
        v2_validation_path.read_text(encoding="ascii")
    )
    cancellation_path = _raw_path(
        OVERLAY_PATHS["job5110073_zero_start_cancellation"]
    )
    cancellation = json.loads(cancellation_path.read_text(encoding="ascii"))
    require(
        BASE.V2.file_sha256(v2_path)
        == "260a06a45b8f4d43f8a557606a7a0bdc3aa7904053bf4170f5ef60d71332f0b3"
        and v2_validation.get("validation_pass") is True
        and v2_validation.get("authority_sha256")
        == "260a06a45b8f4d43f8a557606a7a0bdc3aa7904053bf4170f5ef60d71332f0b3"
        and cancellation.get("job_id") == 5110073
        and cancellation.get("state") == "CANCELLED"
        and cancellation.get("elapsed") == "00:00:00"
        and cancellation.get("start_time") is None
        and cancellation.get("smoke_output_created") is False
        and cancellation.get("model_forward_count") == 0,
        "superseded V2/cancellation disposition drift",
    )
    for relative in (
        PRODUCER_NAMESPACE,
        VALIDATION_NAMESPACE,
        SUPERSEDED_V2_PRODUCER,
        SUPERSEDED_V2_VALIDATION,
        FORMAL_PRODUCER,
        FORMAL_VALIDATION,
    ):
        require(
            not os.path.lexists(_raw_path(relative)),
            f"H100 candidate output exists/symlink: {relative}",
        )
    rows = {}
    for key in BASE.required_bindings():
        row = v2.get("bindings", {}).get(key)
        require(isinstance(row, Mapping), f"V2 base binding absent: {key}")
        observed = _binding(str(row["path"]))
        require(
            observed["bytes"] == row["bytes"]
            and observed["sha256"] == row["sha256"],
            f"V2 base binding drift: {key}",
        )
        rows[key] = observed
    for key, path in OVERLAY_PATHS.items():
        rows[key] = _binding(path)
    require(set(rows) == required_bindings(), "H100 candidate exact126 drift")
    value: dict[str, Any] = {
        "schema_version": BASE.SCHEMA,
        "status": BASE.STATUS,
        "bindings": rows,
        "independent_review_status": "PASS",
        "submission_review_status": "PENDING",
        "execution_scope": "EXECUTION0_ACCELERATED_H100_REAL_SMOKE_V3_ONLY_NONPROMOTABLE",
        "real_smoke": True,
        "eligible_as_e1_result": False,
        "never_consume_superseded_v2_outputs": True,
        "superseded_v2_smoke_output_read_authorized": False,
        "formal_or_full594_execution_authorized": False,
        "manual_submission_requires_independent_review": True,
        "execution_ordinal": 0,
        "query_id": "DIFFICULT-0000",
        "source_fold": 2,
        "candidate_count": 128,
        "candidate_axis_sha256": BASE.CANDIDATE_AXIS_SHA256,
        "pair_sha256": BASE.PAIR_SHA256,
        "v_checkpoint_file_sha256": BASE.V_CHECKPOINT_FILE_SHA256,
        "v_checkpoint_state_sha256": BASE.V_CHECKPOINT_STATE_SHA256,
        "family_sequence": list(BASE.FAMILIES),
        "arm_sequence": list(BASE.ARMS),
        "expected_pair_arm_record_count": 9,
        "expected_directional_term_count": 36,
        "device_schedule": {
            "all_patch_device": "cpu",
            "regional_device": "cuda:0",
            "model_load_count": 1,
            "model_device_transition_count": 1,
        },
        "resource_contract": {
            "partition": "accelerated-h100",
            "allowed_partitions": ["accelerated-h100"],
            "gpu_count": 1,
            "cpus_per_task": 8,
            "memory_megabytes": 128000,
            "walltime_seconds": 7200,
        },
        "output_contract": {
            "producer_family": PRODUCER_NAMESPACE,
            "validation_family": VALIDATION_NAMESPACE,
            "append_only": True,
            "atomic_family_required": True,
            "exact_committed_reuse": True,
            "overwrite_or_repair_authorized": False,
            "eligible_as_e1_result": False,
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
        "smoke_authorized": True,
        "submission_authorized": False,
        "base_v3_import_closure_count": 65,
        "base_v3_binding_count": 115,
        "smoke_overlay_binding_count": len(OVERLAY_PATHS),
        "smoke_overlay_paths": dict(OVERLAY_PATHS),
        "smoke_overlay_import_closure": list(
            discover_overlay_import_closure()
        ),
        "smoke_overlay_import_closure_sha256": BASE.V2.canonical_sha256(
            list(discover_overlay_import_closure())
        ),
        "binding_modes": {
            key: _raw_path(row["path"]).stat().st_mode & 0o777
            for key, row in rows.items()
        },
        "authority_file_mode": 0o444,
        "launcher_test_paths": list(LAUNCHER_TEST_PATHS),
        "launcher_executable_paths": dict(LAUNCHER_EXECUTABLE_PATHS),
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "import_closure": list(BASE.discover_import_closure()),
        "import_closure_sha256": BASE.V2.canonical_sha256(
            list(BASE.discover_import_closure())
        ),
    }
    value["logical_sha256"] = BASE.V2.logical_sha256(value)
    validate_authority_envelope(value)
    return value


def read_authority(path: Path, expected_sha256: str) -> dict[str, Any]:
    authority_path = _raw_regular_file(
        path,
        expected_mode=0o444,
        expected_lexical=AUTHORITY_PATH,
    )
    require(BASE.V2.file_sha256(authority_path) == expected_sha256, "H100 authority SHA drift")
    value = json.loads(authority_path.read_text(encoding="ascii"))
    validate_authority_envelope(value)
    resolved = []
    for name, row in value["bindings"].items():
        bound = _raw_regular_file(
            row["path"],
            expected_mode=value["binding_modes"][name],
            expected_lexical=row["path"],
        )
        require(
            bound.stat().st_size == row["bytes"]
            and BASE.V2.file_sha256(bound) == row["sha256"],
            f"H100 bound file drift: {name}",
        )
        if name in OVERLAY_PATHS:
            resolved.append(bound)
    require(
        len({(path.stat().st_dev, path.stat().st_ino) for path in resolved})
        == 14,
        "H100 overlay physical alias drift",
    )
    _reject_namespace_symlinks()
    return value


__all__ = [
    "AUTHORITY_PATH",
    "LAUNCHER_EXECUTABLE_PATHS",
    "LAUNCHER_TEST_PATHS",
    "OVERLAY_PATHS",
    "PRODUCER_NAMESPACE",
    "VALIDATION_NAMESPACE",
    "discover_overlay_import_closure",
    "read_authority",
    "required_bindings",
    "validate_authority_envelope",
    "validate_launcher_source",
    "build_candidate",
]
