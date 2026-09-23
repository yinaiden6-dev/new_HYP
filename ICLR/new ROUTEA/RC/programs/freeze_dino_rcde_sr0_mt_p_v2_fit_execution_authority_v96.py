#!/usr/bin/env python3
"""Freeze V96 authority for the 16 formal P-V2 fits and exact-resume checks.

This freezer is intentionally declarative.  It binds the immutable V95
manifest population and the complete V2 fit implementation, but it neither
runs training nor makes a produced checkpoint consumable.  The only enabled
scopes are the formal fit and its independent exact-resume validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
AUTH = "registry/current_authority_v96_20260818.json"
PARENT = "registry/current_authority_v95_20260818.json"
V95_INDEX = "results/dino_rcde_sr0_mt_p_v2_training_manifests_v1/index.json"
V95_VALIDATION = (
    "results/dino_rcde_sr0_mt_p_v2_training_manifest_validation_v1/result.json"
)
FIT_OUTPUT_ROOT = "results/dino_rcde_sr0_mt_p_v2_formal_fits_v1"
VALIDATION_OUTPUT_ROOT = (
    "results/dino_rcde_sr0_mt_p_v2_formal_fit_validation_v1"
)
SCHEMA = "rc_current_authority_v96_20260818"
STATUS = "RCDE_SR0_MT_P_V2_FORMAL_FIT_EXECUTION_AUTHORIZED"

EXPECTED_SCOPE = {
    "p_v2_exact_resume_validation": True,
    "p_v2_formal_fit_execution": True,
}

TEST_PATHS = (
    "tests/test_dino_rcde_sr0_mt_p_v2_fit_runtime_v1.py",
    "tests/test_run_dino_rcde_sr0_mt_p_v2_fit_v1.py",
    "tests/test_validate_dino_rcde_sr0_mt_p_v2_fit_v1.py",
    "tests/test_freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96.py",
)


class AuthorityError(RuntimeError):
    """A V95 lineage, implementation binding, or V96 boundary drifted."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def canonical(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return canonical(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(
    relative: str,
    *,
    with_logical: bool = False,
    immutable: bool = False,
) -> dict[str, Any]:
    raw = Path(relative)
    require(
        not raw.is_absolute() and ".." not in raw.parts,
        f"unsafe binding path {relative}",
    )
    path = (ROOT / raw).resolve(strict=True)
    require(
        path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink(),
        f"invalid binding path {relative}",
    )
    if immutable:
        require(
            (path.stat().st_mode & 0o777) == 0o444,
            f"non-immutable input artifact {relative}",
        )
    result: dict[str, Any] = {
        "path": relative,
        "sha256": file_sha(path),
        "bytes": path.stat().st_size,
    }
    if with_logical:
        value = json.loads(path.read_text(encoding="utf-8"))
        require(
            isinstance(value, dict)
            and value.get("logical_sha256") == logical(value),
            f"logical hash drift {relative}",
        )
        result["logical_sha256"] = value["logical_sha256"]
    return result


def latest() -> tuple[int, Path]:
    pattern = re.compile(r"current_authority_v([1-9][0-9]*)_[0-9]{8}\.json")
    rows: list[tuple[int, Path]] = []
    for path in (ROOT / "registry").glob("current_authority_v*_*.json"):
        match = pattern.fullmatch(path.name)
        if match is not None and path.is_file() and not path.is_symlink():
            rows.append((int(match.group(1)), path.resolve()))
    require(rows, "authority registry is empty")
    version = max(item[0] for item in rows)
    paths = [path for item_version, path in rows if item_version == version]
    require(len(paths) == 1, "latest authority version is aliased")
    return version, paths[0]


def expected_fit_ids() -> list[str]:
    output: list[str] = []
    for outer in range(1, 5):
        output.extend(
            f"P_OUTER{outer}_INNER{inner}_FIT"
            for inner in range(1, 5)
            if inner != outer
        )
        output.append(f"P_OUTER{outer}_OUTER_REFIT")
    return output


def _validate_parent_and_manifests() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any]
]:
    parent_path = (ROOT / PARENT).resolve(strict=True)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    require(
        parent.get("schema_version") == "rc_current_authority_v95_20260818"
        and parent.get("status")
        == "RCDE_SR0_MT_P_V2_TRAINING_MANIFESTS_AUTHORIZED"
        and parent.get("authorized_scope")
        == {"p_v2_training_manifest_materialization_only": True},
        "V95 parent identity/scope drift",
    )
    for key in (
        "training_authorized",
        "model_load_authorized",
        "consumable_checkpoint_authorized",
        "automatic_stage_advance",
    ):
        require(parent.get(key) is False, f"V95 parent {key} drift")
    require(
        parent.get("scientific_GO_or_NO_GO") is None
        and parent.get("fit_count") == 16
        and parent.get("inner_fit_count") == 12
        and parent.get("outer_refit_count") == 4,
        "V95 parent count/claim drift",
    )

    index = json.loads((ROOT / V95_INDEX).read_text(encoding="utf-8"))
    validation = json.loads(
        (ROOT / V95_VALIDATION).read_text(encoding="utf-8")
    )
    require(
        index.get("schema_version")
        == "rc_dino_rcde_sr0_mt_p_v2_training_manifest_index_v1_20260818"
        and index.get("status")
        == "RCDE_SR0_MT_P_V2_TRAINING_MANIFEST_INDEX_READY"
        and index.get("logical_sha256") == logical(index),
        "V95 manifest index drift",
    )
    rows = index.get("fits")
    require(
        isinstance(rows, list)
        and len(rows) == 16
        and index.get("fit_count") == 16
        and index.get("inner_fit_count") == 12
        and index.get("outer_refit_count") == 4
        and index.get("fit_sequence_sha256") == canonical(rows)
        and [row.get("fit_id") for row in rows] == expected_fit_ids(),
        "V95 manifest population/order drift",
    )
    require(
        validation.get("schema_version")
        == "rc_dino_rcde_sr0_mt_p_v2_training_manifest_validation_v1_20260818"
        and validation.get("status")
        == "RCDE_SR0_MT_P_V2_TRAINING_MANIFESTS_INDEPENDENT_VALIDATION_PASS"
        and validation.get("validation_pass") is True
        and validation.get("fit_count") == 16
        and validation.get("inner_fit_count") == 12
        and validation.get("outer_refit_count") == 4
        and validation.get("total_episode_instances") == 5346
        and validation.get("training_authorized") is False
        and validation.get("consumable_checkpoint_authorized") is False
        and validation.get("scientific_GO_or_NO_GO") is None
        and validation.get("logical_sha256") == logical(validation),
        "V95 independent validation PASS absent",
    )
    require(
        validation.get("manifest_index_sha256") == file_sha(ROOT / V95_INDEX)
        and validation.get("manifest_index_logical_sha256")
        == index.get("logical_sha256"),
        "V95 validation/index binding drift",
    )

    for row, fit_id in zip(rows, expected_fit_ids(), strict=True):
        expected_path = (
            "results/dino_rcde_sr0_mt_p_v2_training_manifests_v1/"
            f"{fit_id}/fit_manifest.json"
        )
        require(
            row.get("fit_id") == fit_id and row.get("path") == expected_path,
            f"V95 manifest index address drift {fit_id}",
        )
        path = (ROOT / expected_path).resolve(strict=True)
        require(
            path.is_file()
            and not path.is_symlink()
            and (path.stat().st_mode & 0o777) == 0o444,
            f"V95 manifest is not immutable {fit_id}",
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        require(
            manifest.get("schema_version")
            == "rc_dino_rcde_sr0_mt_p_v2_training_manifest_v1_20260818"
            and manifest.get("status")
            == "RCDE_SR0_MT_P_V2_TRAINING_MANIFEST_READY"
            and manifest.get("fit_id") == fit_id
            and manifest.get("logical_sha256")
            == logical(manifest)
            == row.get("logical_sha256")
            and manifest.get("training_authorized") is False
            and manifest.get("consumable_checkpoint_authorized") is False,
            f"V95 fit manifest drift {fit_id}",
        )
    return parent, index, validation


def build() -> dict[str, Any]:
    version, latest_path = latest()
    parent_path = (ROOT / PARENT).resolve(strict=True)
    self_path = (ROOT / AUTH).resolve(strict=False)
    require(
        (version == 95 and latest_path == parent_path)
        or (
            version == 96
            and latest_path == self_path
            and self_path.is_file()
            and not self_path.is_symlink()
        ),
        "V95/V96 authority state drift",
    )
    _parent, index, _validation = _validate_parent_and_manifests()

    bindings: dict[str, Any] = {
        "parent_authority_v95": bind(PARENT, with_logical=True, immutable=True),
        "v95_manifest_index": bind(
            V95_INDEX, with_logical=True, immutable=True
        ),
        "v95_manifest_validation": bind(
            V95_VALIDATION, with_logical=True, immutable=True
        ),
        "contract": bind(
            "plan/DINO_RCDE_SR0_MT_P_V2_FORMAL_FIT_EXECUTION_CONTRACT_V1_20260818.md"
        ),
        "loss_join": bind(
            "results/dino_rcde_sr0_mt_loss_join_v1/loss_join.json",
            with_logical=True,
            immutable=True,
        ),
        "loss_join_validation": bind(
            "results/dino_rcde_sr0_mt_loss_join_validation_v1/result.json",
            with_logical=True,
            immutable=True,
        ),
        "pair_aggregate": bind(
            "results/dino_rcde_sr0_mt_role_free_pair_feature_cache_virtual_aggregate_v1/virtual_aggregate.json",
            with_logical=True,
            immutable=True,
        ),
        "pair_aggregate_validation": bind(
            "results/dino_rcde_sr0_mt_role_free_pair_feature_cache_virtual_aggregate_validation_v1/result.json",
            with_logical=True,
            immutable=True,
        ),
        "sidecar_manifest": bind(
            "results/dino_rcde_sr0_mt_training_consumer_e0_full594_structure_sidecar_validation_v1/manifest.json",
            with_logical=True,
            immutable=True,
        ),
        "p_head": bind(
            "src/rc_aslo_xf/dino_rcde_cw1_multitile_sr0_p_v1.py"
        ),
        "role_free_scorer": bind(
            "src/rc_aslo_xf/dino_rcde_sr0_mt_p_role_free_e0_consumer_v1.py"
        ),
        "p_lock_core": bind(
            "src/rc_aslo_xf/dino_rcde_sr0_mt_p_lock_v2.py"
        ),
        "p_selector": bind(
            "src/rc_aslo_xf/dino_rcde_sr0_mt_p_selector_v2.py"
        ),
        "p_v2_adapter": bind(
            "src/rc_aslo_xf/dino_rcde_sr0_mt_p_natural_adapter_v2.py"
        ),
        "fit_runtime": bind(
            "src/rc_aslo_xf/dino_rcde_sr0_mt_p_v2_fit_runtime_v1.py"
        ),
        "runner": bind("programs/run_dino_rcde_sr0_mt_p_v2_fit_v1.py"),
        "validator": bind(
            "programs/validate_dino_rcde_sr0_mt_p_v2_fit_v1.py"
        ),
        "launcher": bind(
            "slurm/dino_rcde_sr0_mt_p_v2_formal_fit_v1.sbatch"
        ),
        "freezer": bind(
            "programs/freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96.py"
        ),
    }
    for ordinal, row in enumerate(index["fits"]):
        bindings[f"fit_manifest_{ordinal:02d}"] = bind(
            str(row["path"]), with_logical=True, immutable=True
        )
    for ordinal, relative in enumerate(TEST_PATHS):
        bindings[f"test_{ordinal:02d}"] = bind(relative)

    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "stage": "SR0_MT_P_V2_FORMAL_FIT_EXECUTION",
        "claim_level": "ENGINEERING_16_P_V2_FITS_AND_EXACT_RESUME_EXECUTION_ONLY",
        "authorized_scope": EXPECTED_SCOPE,
        "fit_manifest_index": V95_INDEX,
        "fit_output_root": FIT_OUTPUT_ROOT,
        "validation_output_root": VALIDATION_OUTPUT_ROOT,
        "fit_count": 16,
        "inner_fit_count": 12,
        "outer_refit_count": 4,
        "pair_feature_payload_read_authorized": True,
        "sidecar_shard_payload_read_authorized": True,
        "validated_loss_join_read_authorized": True,
        "validated_membership_read_authorized": True,
        "target_rival_loss_role_read_authorized": True,
        "model_load_authorized": True,
        "model_forward_authorized": True,
        "model_backward_authorized": True,
        "model_update_authorized": True,
        "training_authorized": True,
        "checkpoint_deserialization_authorized": True,
        "resume_state_read_authorized": True,
        "d1_score_rank_winner_read_authorized": False,
        "retrieval_outcome_read_authorized": False,
        "opened_read_authorized": False,
        "sealed_read_authorized": False,
        "C8_read_authorized": False,
        "S8_read_authorized": False,
        "target_insertion_authorized": False,
        "p_lock_materialization_authorized": False,
        "donor_matching_authorized": False,
        "v_training_authorized": False,
        "heldout_scoring_authorized": False,
        "consumable_checkpoint_authorized": False,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
        "resource_contract": {
            "partition": "cpuonly",
            "array_spec": "0-15%4",
            "array_task_count": 16,
            "array_max_concurrency": 4,
            "cpus_per_task": 1,
            "memory_megabytes": 16384,
            "walltime_seconds": 14400,
        },
        "bindings": bindings,
    }
    value["logical_sha256"] = logical(value)
    return value


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable authority already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.chmod(0o444)
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / AUTH)
    args = parser.parse_args()
    require(
        args.output.resolve(strict=False) == (ROOT / AUTH).resolve(strict=False),
        "authority output path drift",
    )
    value = build()
    atomic(args.output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": str(args.output),
                "logical_sha256": value["logical_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
