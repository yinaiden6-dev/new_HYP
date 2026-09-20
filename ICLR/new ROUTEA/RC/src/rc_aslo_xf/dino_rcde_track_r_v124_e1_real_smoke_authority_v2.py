"""Exact path-map authority runtime for non-promotable real-smoke V2."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping

from . import dino_rcde_track_r_v124_e1_authority_v3 as BASE


OVERLAY_PATHS = MappingProxyType(
    {
        "smoke_producer": "programs/run_dino_rcde_track_r_v124_e1_accelerated_real_smoke_v2.py",
        "smoke_independent_validator": "programs/validate_dino_rcde_track_r_v124_e1_accelerated_real_smoke_independent_v2.py",
        "smoke_launcher": "slurm/dino_rcde_track_r_v124_e1_accelerated_real_smoke_v2.sbatch",
        "smoke_authority_runtime": "src/rc_aslo_xf/dino_rcde_track_r_v124_e1_real_smoke_authority_v2.py",
        "smoke_tests": "tests/test_dino_rcde_track_r_v124_e1_accelerated_real_smoke_v2.py",
        "smoke_producer_v1_parent": "programs/run_dino_rcde_track_r_v124_e1_accelerated_real_smoke_v1.py",
        "smoke_validator_v1_parent": "programs/validate_dino_rcde_track_r_v124_e1_accelerated_real_smoke_independent_v1.py",
        "smoke_authority_v1_parent": "src/rc_aslo_xf/dino_rcde_track_r_v124_e1_real_smoke_authority_v1.py",
        "smoke_authority_v1_terminal_rejection": "registry/dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v1_20260826.terminal_rejection_v1.json",
        "smoke_authority_freezer": "programs/freeze_dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v2.py",
        "smoke_authority_independent_reviewer": "programs/validate_dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v2.py",
    }
)
PRODUCER_NAMESPACE = "results/dino_rcde_track_r_v124_e1_v3_accelerated_real_smoke_v2_committed"
VALIDATION_NAMESPACE = "results/dino_rcde_track_r_v124_e1_v3_accelerated_real_smoke_validation_v2_committed"
AUTHORITY_PATH = "registry/dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v2_20260826.json"
FORMAL_PRODUCER_NAMESPACE = "results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3_committed"
FORMAL_VALIDATION_NAMESPACE = "results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_validation_v3_committed"
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


class E1RealSmokeAuthorityV2Error(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise E1RealSmokeAuthorityV2Error(message)


def required_bindings() -> frozenset[str]:
    return BASE.required_bindings() | frozenset(OVERLAY_PATHS)


def _local_module_file(module: str) -> Path | None:
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
    roots = [
        BASE.RC_ROOT / OVERLAY_PATHS[key]
        for key in (
            "smoke_producer",
            "smoke_independent_validator",
            "smoke_authority_runtime",
            "smoke_tests",
        )
    ]
    pending = [path.resolve() for path in roots]
    seen = set()
    while pending:
        path = pending.pop()
        relative = path.relative_to(BASE.RC_ROOT).as_posix()
        require(relative in expected, f"foreign overlay closure root: {relative}")
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
                candidate = _local_module_file(module)
                if candidate is None:
                    continue
                candidate_relative = candidate.relative_to(
                    BASE.RC_ROOT
                ).as_posix()
                if candidate_relative in expected and candidate.resolve() not in seen:
                    pending.append(candidate.resolve())
    discovered = tuple(
        sorted(path.relative_to(BASE.RC_ROOT).as_posix() for path in seen)
    )
    require(
        set(discovered) == expected,
        "smoke V2 overlay AST closure missing/extra path",
    )
    return discovered


def _raw_path(path_value: str | Path) -> Path:
    raw = Path(path_value)
    return Path(
        os.path.abspath(
            os.fspath(raw if raw.is_absolute() else BASE.RC_ROOT / raw)
        )
    )


def _assert_no_symlink_components(path: Path) -> None:
    raw = _raw_path(path)
    base = BASE.RC_ROOT if BASE.RC_ROOT in raw.parents else Path("/")
    cursor = base
    for part in raw.relative_to(base).parts:
        cursor = cursor / part
        if os.path.lexists(cursor):
            metadata = os.lstat(cursor)
            require(
                not stat.S_ISLNK(metadata.st_mode),
                f"raw symlink component forbidden: {cursor}",
            )


def _raw_regular_file(
    path_value: str | Path,
    *,
    expected_mode: int,
    expected_lexical: str | Path | None = None,
) -> Path:
    raw = _raw_path(path_value)
    if expected_lexical is not None:
        require(
            raw == _raw_path(expected_lexical),
            "exact canonical lexical path drift",
        )
    require(os.path.lexists(raw), f"raw bound path absent: {raw}")
    metadata = os.lstat(raw)
    require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_mode & 0o777 == expected_mode,
        f"raw bound file type/mode/symlink drift: {raw}",
    )
    _assert_no_symlink_components(raw)
    require(raw.resolve() == raw, "bound path resolves through alias")
    return raw


def _reject_namespace_symlinks() -> None:
    for relative in (
        PRODUCER_NAMESPACE,
        VALIDATION_NAMESPACE,
        FORMAL_PRODUCER_NAMESPACE,
        FORMAL_VALIDATION_NAMESPACE,
    ):
        raw = _raw_path(relative)
        if os.path.lexists(raw):
            require(
                not stat.S_ISLNK(os.lstat(raw).st_mode),
                f"dangling/resolved namespace symlink forbidden: {relative}",
            )


def validate_launcher_source(value: Mapping[str, Any], source: str) -> None:
    for path in LAUNCHER_TEST_PATHS:
        require(
            source.count(path) == 1,
            f"launcher test path count drift: {path}",
        )
    for key, path in LAUNCHER_EXECUTABLE_PATHS.items():
        require(
            value["bindings"][key]["path"] == path
            and source.count(path) == 1,
            f"launcher executable path count/key drift: {key}",
        )
    require(
        "accelerated_real_smoke_v1.py" not in source
        and "accelerated_real_smoke_v1.sbatch" not in source
        and source.count('"$rc_python" -m pytest') == 1
        and source.count('"$rc_python" "$rc_producer"') == 1
        and source.count('"$rc_python" "$rc_validator"') == 1,
        "launcher alias/command population drift",
    )


def _validate_launcher(value: Mapping[str, Any]) -> None:
    launcher_row = value["bindings"]["smoke_launcher"]
    raw = Path(launcher_row["path"])
    launcher = (BASE.RC_ROOT / raw).resolve()
    validate_launcher_source(value, launcher.read_text(encoding="utf-8"))


def validate_authority_envelope(value: Mapping[str, Any]) -> None:
    bindings = value.get("bindings")
    require(
        isinstance(bindings, Mapping)
        and set(bindings) == required_bindings(),
        "smoke V2 exact base/overlay key set drift",
    )
    projected = dict(value)
    projected["bindings"] = {
        key: bindings[key] for key in BASE.required_bindings()
    }
    projected["logical_sha256"] = BASE.V2.logical_sha256(projected)
    BASE.validate_authority_envelope(projected)
    require(
        value.get("logical_sha256") == BASE.V2.logical_sha256(value)
        and value.get("execution_scope")
        == "EXECUTION0_ACCELERATED_REAL_SMOKE_V2_ONLY_NONPROMOTABLE"
        and value.get("real_smoke") is True
        and value.get("eligible_as_e1_result") is False
        and value.get("submission_review_status") == "PENDING"
        and value.get("submission_authorized") is False
        and value.get("manual_submission_requires_independent_review") is True
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
        and value.get("base_v3_import_closure_count") == 65
        and value.get("base_v3_binding_count") == 115
        and value.get("smoke_overlay_binding_count") == len(OVERLAY_PATHS)
        and isinstance(value.get("binding_modes"), Mapping)
        and set(value["binding_modes"]) == set(bindings)
        and all(
            type(mode) is int and 0 <= mode <= 0o777
            for mode in value["binding_modes"].values()
        )
        and value.get("authority_file_mode") == 0o444
        and value.get("smoke_overlay_paths") == dict(OVERLAY_PATHS)
        and value.get("smoke_overlay_import_closure")
        == list(discover_overlay_import_closure())
        and value.get("smoke_overlay_import_closure_sha256")
        == BASE.V2.canonical_sha256(
            list(discover_overlay_import_closure())
        )
        and value.get("launcher_test_paths") == list(LAUNCHER_TEST_PATHS)
        and value.get("launcher_executable_paths")
        == dict(LAUNCHER_EXECUTABLE_PATHS)
        and value.get("target_rival_join_authorized") is False
        and value.get("postjoin_authorized") is False
        and value.get("scientific_reduction_authorized") is False
        and value.get("automatic_submit_authorized") is False
        and value.get("scientific_GO_or_NO_GO") is None
        and value.get("automatic_stage_advance") is False
        and value.get("next_authorized_stage") is None,
        "smoke V2 non-promotable/path-map boundary drift",
    )
    paths = []
    for key, expected in OVERLAY_PATHS.items():
        row = bindings[key]
        require(
            isinstance(row, Mapping)
            and set(row) == {"path", "bytes", "sha256"}
            and row["path"] == expected,
            f"smoke V2 key/path binding drift: {key}",
        )
        paths.append(str(row["path"]))
    require(
        len(paths) == len(set(paths)) == len(OVERLAY_PATHS),
        "smoke V2 duplicate/alias overlay path drift",
    )
    explicit_tests = {
        "launcher_test_full_c128_c_dino_v1": LAUNCHER_TEST_PATHS[0],
        "launcher_test_v_runtime_v2": LAUNCHER_TEST_PATHS[1],
        "launcher_test_phase_b_all_patch_v1": LAUNCHER_TEST_PATHS[2],
        "launcher_test_e1_v3": LAUNCHER_TEST_PATHS[3],
        "smoke_tests": LAUNCHER_TEST_PATHS[4],
    }
    require(
        all(bindings[key]["path"] == path for key, path in explicit_tests.items()),
        "smoke V2 executed test key/path drift",
    )
    _validate_launcher(value)


def read_authority(path: Path, expected_sha256: str) -> dict[str, Any]:
    authority_path = _raw_regular_file(
        path,
        expected_mode=0o444,
        expected_lexical=AUTHORITY_PATH,
    )
    require(
        BASE.V2.file_sha256(authority_path) == expected_sha256,
        "smoke V2 authority physical drift",
    )
    value = json.loads(authority_path.read_text(encoding="ascii"))
    validate_authority_envelope(value)
    resolved_overlay = []
    for name, row in value["bindings"].items():
        bound = _raw_regular_file(
            row["path"],
            expected_mode=value["binding_modes"][name],
            expected_lexical=row["path"],
        )
        require(
            bound.is_file()
            and not bound.is_symlink()
            and bound.stat().st_size == row["bytes"]
            and BASE.V2.file_sha256(bound) == row["sha256"],
            f"smoke V2 bound file drift: {name}",
        )
        if name in OVERLAY_PATHS:
            resolved_overlay.append(bound)
    require(
        len(
            {
                (os.stat(path).st_dev, os.stat(path).st_ino)
                for path in resolved_overlay
            }
        )
        == len(OVERLAY_PATHS),
        "smoke V2 overlay hardlink/physical alias drift",
    )
    _reject_namespace_symlinks()
    return value


__all__ = [
    "E1RealSmokeAuthorityV2Error",
    "LAUNCHER_EXECUTABLE_PATHS",
    "LAUNCHER_TEST_PATHS",
    "OVERLAY_PATHS",
    "PRODUCER_NAMESPACE",
    "VALIDATION_NAMESPACE",
    "read_authority",
    "required_bindings",
    "validate_authority_envelope",
    "validate_launcher_source",
    "discover_overlay_import_closure",
    "_raw_regular_file",
    "_reject_namespace_symlinks",
]
