#!/usr/bin/env python3
"""Freeze V120 after all four Track-R relative V fits validate exactly."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import torch

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H


ROOT = Path(__file__).resolve().parents[1]
AUTH = "registry/current_authority_v121_20260821.json"
PARENT = "registry/current_authority_v120_20260821.json"
STATUS = "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARDS_AUTHORIZED"
PARENT_SCHEMA = "rc_current_authority_v120_20260821"
PARENT_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_V_FIT_AUTHORIZED"
GALLERY_CACHE = (
    ROOT.parents[3]
    / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
).resolve()

PRODUCER = "programs/materialize_dino_rcde_track_r_oof_prejoin_shard_v1.py"
VALIDATOR = "programs/validate_dino_rcde_track_r_oof_prejoin_shard_v1.py"
FINALIZER = "programs/finalize_dino_rcde_track_r_oof_prejoin_v120.py"
LINEAGE = "programs/dino_rcde_track_r_v121_lineage_v1.py"
RUNTIME_ENTRY_VALIDATOR = (
    "programs/validate_dino_rcde_track_r_v121_runtime_binding_v1.py"
)
LAUNCHER = "slurm/dino_rcde_track_r_oof_prejoin_shards_v1.sbatch"
POST_VFIT = "slurm/dino_rcde_track_r_post_vfit_controller_v1.sbatch"
POST_OOF = "slurm/dino_rcde_track_r_post_oof_controller_v1.sbatch"
CANARY_RESULT = "results/dino_rcde_track_r_v120_40gb_memory_canary_v2/result.json"
CANARY_VALIDATION = "results/dino_rcde_track_r_v120_40gb_memory_canary_v2/validation.json"
PACKAGE_INITIALIZER = "src/rc_aslo_xf/__init__.py"
RUNTIME_ROOTS = (
    PRODUCER,
    VALIDATOR,
    FINALIZER,
    LINEAGE,
    RUNTIME_ENTRY_VALIDATOR,
    PACKAGE_INITIALIZER,
)


def req(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _module_path(module: str) -> Path | None:
    if module.startswith("rc_aslo_xf"):
        candidate = (ROOT / "src" / Path(*module.split("."))).with_suffix(".py")
    else:
        candidate = (ROOT / "programs" / Path(*module.split("."))).with_suffix(".py")
    return candidate.resolve() if candidate.is_file() else None


def _local_imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source_root = (ROOT / "src").resolve()
    if source_root in path.resolve().parents:
        relative = path.resolve().relative_to(source_root).with_suffix("")
        package = list(relative.parts[:-1])
    else:
        package = []
    output: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:]
                if node.level > 1:
                    base = base[: -(node.level - 1)]
                module = ".".join(
                    base + ((node.module or "").split(".") if node.module else [])
                )
            else:
                module = node.module or ""
            modules.append(module)
            modules.extend(
                ".".join(part for part in (module, alias.name) if part)
                for alias in node.names
                if alias.name != "*"
            )
        for module in modules:
            imported = _module_path(module)
            if imported is not None:
                output.add(imported)
    return output


def discover_runtime_import_closure(
    roots: Iterable[str] = RUNTIME_ROOTS,
) -> tuple[list[str], list[dict[str, str]]]:
    root_paths = {(ROOT / relative).resolve(strict=True) for relative in roots}
    pending = list(root_paths)
    visited = set(root_paths)
    closure: set[Path] = set()
    edges: set[tuple[str, str]] = set()
    while pending:
        importer = pending.pop()
        for imported in _local_imports(importer):
            edges.add(
                (
                    importer.relative_to(ROOT).as_posix(),
                    imported.relative_to(ROOT).as_posix(),
                )
            )
            if imported not in root_paths:
                closure.add(imported)
            if imported not in visited:
                visited.add(imported)
                pending.append(imported)
    paths = sorted(path.relative_to(ROOT).as_posix() for path in closure)
    edge_rows = [
        {"importer": importer, "imported": imported}
        for importer, imported in sorted(edges)
    ]
    return paths, edge_rows


def latest() -> tuple[int, Path]:
    pattern = re.compile(r"current_authority_v([1-9][0-9]*)_[0-9]{8}\.json")
    rows = [(int(match.group(1)), path.resolve()) for path in (ROOT / "registry").glob("current_authority_v*_*.json") if (match := pattern.fullmatch(path.name)) and path.is_file() and not path.is_symlink()]
    version = max(item[0] for item in rows)
    paths = [path for item_version, path in rows if item_version == version]
    req(len(paths) == 1, "latest authority alias")
    return version, paths[0]


def external_bind(path: Path) -> dict[str, Any]:
    req(path == GALLERY_CACHE and path.is_file() and not path.is_symlink(), "external gallery cache path drift")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def build() -> dict[str, Any]:
    version, current = latest()
    parent_path = (ROOT / PARENT).resolve(strict=True)
    self_path = (ROOT / AUTH).resolve(strict=False)
    req((version == 120 and current == parent_path) or (version == 121 and current == self_path and self_path.is_file()), "V120/V121 authority state drift")
    parent = json.loads(parent_path.read_text())
    req(
        parent.get("schema_version") == PARENT_SCHEMA
        and parent.get("status") == PARENT_STATUS
        and parent.get("authority_version") == 120
        and parent.get("authority_revision") == 5
        and parent.get("execution_node_schema_repair", {}).get(
            "training_or_model_change_authorized"
        )
        is False
        and parent.get("three_arm_autograd_repair", {}).get(
            "model_architecture_changed"
        )
        is False
        and parent.get("three_arm_autograd_repair", {}).get(
            "forward_loss_scalar_changed"
        )
        is False
        and parent.get("three_arm_autograd_repair", {}).get(
            "pre_full_fit_canary_required"
        )
        is True
        and parent.get("logical_sha256") == H.logical(parent)
        and (parent_path.stat().st_mode & 0o777) == 0o444,
        "V120 parent physical/logical drift",
    )
    parent_sha256 = H.file_sha(parent_path)
    parent_logical_sha256 = str(parent["logical_sha256"])
    canary_result_path = ROOT / CANARY_RESULT
    canary_validation_path = ROOT / CANARY_VALIDATION
    canary_result = json.loads(canary_result_path.read_text())
    canary_validation = json.loads(canary_validation_path.read_text())
    req(
        canary_result.get("status")
        == "DINO_RCDE_TRACK_R_V120_R5_40GB_MEMORY_CANARY_PASS"
        and canary_validation.get("status")
        == "DINO_RCDE_TRACK_R_V120_R5_40GB_MEMORY_CANARY_INDEPENDENT_VALIDATION_PASS"
        and canary_validation.get("validation_pass") is True
        and canary_result.get("authority_revision") == 5
        and canary_result.get("authority_sha256") == parent_sha256
        and canary_result.get("authority_logical_sha256")
        == parent_logical_sha256
        and canary_validation.get("authority_revision") == 5
        and canary_validation.get("authority_sha256") == parent_sha256
        and canary_validation.get("authority_logical_sha256")
        == parent_logical_sha256
        and canary_validation.get("result_sha256") == H.file_sha(canary_result_path)
        and canary_validation.get("result_logical_sha256")
        == canary_result.get("logical_sha256")
        and canary_result.get("completed_update_count") == 8
        and canary_result.get("effective_queries_per_update") == 4
        and canary_result.get("optimizer_step_count") == 8
        and canary_result.get("autograd_contract")
        == "THREE_ARM_EXECUTION_WITH_LOCAL_ONLY_AUTOGRAD_V1"
        and canary_result.get("mandatory_arm_execution_count_per_episode") == 3
        and canary_result.get("required_free_fraction") == 0.05
        and canary_result.get("gpu_free_fraction_at_peak_lower_bound", -1.0)
        >= 0.05,
        "V120-r5 three-arm 40GB canary independent PASS absent",
    )
    fit_receipts = []
    checkpoint_bindings = {}
    for fold in (1, 2, 3, 4):
        root = ROOT / f"results/dino_rcde_track_r_relative_v_fits_v1/outer_fold{fold}"
        validation_path = root / "fresh_resume_validation.json"
        checkpoint_path = root / "primary/checkpoint_update2048.pt"
        result_path = root / "primary/result.json"
        validation = json.loads(validation_path.read_text())
        result = json.loads(result_path.read_text())
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True, mmap=True
        )
        req(
            validation.get("status") == "DINO_RCDE_TRACK_R_RELATIVE_V_FIT_FRESH_RESUME_VALIDATION_PASS"
            and validation.get("validation_pass") is True
            and validation.get("outer_fold") == fold
            and validation.get("authority_sha256") == parent_sha256
            and validation.get("authority_logical_sha256")
            == parent_logical_sha256
            and validation.get("fresh_resume_model_bit_exact") is True
            and validation.get("relative_pair_loss_only") is True
            and validation.get("donor_null_absolute_loss_count") == 0
            and validation.get("logical_sha256") == H.logical(validation)
            and result.get("status") == "DINO_RCDE_TRACK_R_RELATIVE_V_FIT_COMPLETE"
            and result.get("completed_updates") == 2048
            and result.get("authority_sha256") == parent_sha256
            and result.get("authority_logical_sha256")
            == parent_logical_sha256
            and result.get("logical_sha256") == H.logical(result)
            and result.get("final_checkpoint_sha256") == H.file_sha(checkpoint_path),
            f"Track-R fold{fold} validation drift",
        )
        req(
            isinstance(checkpoint, dict)
            and checkpoint.get("status")
            == "DINO_RCDE_TRACK_R_RELATIVE_CHECKPOINT_COMPLETE"
            and checkpoint.get("outer_fold") == fold
            and checkpoint.get("authority_sha256") == parent_sha256
            and checkpoint.get("authority_logical_sha256")
            == parent_logical_sha256,
            f"Track-R fold{fold} checkpoint V120 lineage drift",
        )
        fit_receipts.append({
            "outer_fold": fold,
            "validation": H.bind(validation_path.relative_to(ROOT).as_posix(), with_logical=True, immutable=True),
            "result": H.bind(result_path.relative_to(ROOT).as_posix(), with_logical=True),
            "checkpoint": H.bind(checkpoint_path.relative_to(ROOT).as_posix()),
        })
        checkpoint_bindings[f"track_r_checkpoint_fold{fold}"] = H.bind(checkpoint_path.relative_to(ROOT).as_posix())
    role_free = json.loads((ROOT / "results/dino_rcde_sr0_mt_role_free_pair_address_v1/role_free_pair_address_manifest.json").read_text())
    role_validation = json.loads((ROOT / "results/dino_rcde_sr0_mt_role_free_pair_address_validation_v1/result.json").read_text())
    req(
        role_free.get("status") == "RCDE_SR0_MT_ROLE_FREE_PAIR_ADDRESS_READY"
        and role_free.get("query_count") == 594
        and role_free.get("role_free") is True
        and role_validation.get("status") == "RCDE_SR0_MT_ROLE_FREE_PAIR_ADDRESS_INDEPENDENT_VALIDATION_PASS"
        and role_validation.get("validation_pass") is True,
        "role-free pair address PASS absent",
    )
    closure_paths, closure_edges = discover_runtime_import_closure()
    closure_rows = [H.bind(path) for path in closure_paths]
    bindings = {
        "parent_authority_v120": H.bind(PARENT, with_logical=True, immutable=True),
        "v120_three_arm_40gb_canary_result": H.bind(
            CANARY_RESULT, with_logical=True, immutable=True
        ),
        "v120_three_arm_40gb_canary_validation": H.bind(
            CANARY_VALIDATION, with_logical=True, immutable=True
        ),
        "fit_receipts": {"count": 4, "rows": fit_receipts, "logical_sha256": H.logical({"rows": fit_receipts})},
        **checkpoint_bindings,
        "role_free_pair_manifest": H.bind("results/dino_rcde_sr0_mt_role_free_pair_address_v1/role_free_pair_address_manifest.json", with_logical=True, immutable=True),
        "role_free_pair_validation": H.bind("results/dino_rcde_sr0_mt_role_free_pair_address_validation_v1/result.json", with_logical=True, immutable=True),
        "p_v2_lock_aggregate": H.bind("results/dino_rcde_sr0_mt_p_v2_formal_lock_aggregate_v1/result.json", with_logical=True, immutable=True),
        "fold_schedule": H.bind("protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json", with_logical=True),
        "geometry_payload": H.bind("cache/dino_rcde_colnomic_sr_full600_geometry_v2/full600_geometry_payload_v2.pt"),
        "redacted_schedule": H.bind("results/dino_rcde_r1_oof_redacted_inputs_v1_0/redacted_oof_schedule.json", with_logical=True),
        "redacted_cache_index": H.bind("results/dino_rcde_r1_oof_redacted_inputs_v1_0/redacted_cache_index.json", with_logical=True),
        "gallery_cache": external_bind(GALLERY_CACHE),
        "gallery_identity_repair": H.bind("registry/gallery_identity_repair_v1.json"),
        "gallery_identity_contract": H.bind("protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json"),
        "producer": H.bind(PRODUCER),
        "validator": H.bind(VALIDATOR),
        "finalizer": H.bind(FINALIZER),
        "lineage_runtime": H.bind(LINEAGE),
        "runtime_entry_validator": H.bind(RUNTIME_ENTRY_VALIDATOR),
        "package_initializer": H.bind(PACKAGE_INITIALIZER),
        "runtime_import_closure": {
            "count": len(closure_rows),
            "logical_sha256": H.logical({"rows": closure_rows}),
            "rows": closure_rows,
            "edge_count": len(closure_edges),
            "edges_sha256": H.logical({"edges": closure_edges}),
        },
        "adapter": H.bind("src/rc_aslo_xf/dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1.py"),
        "controls": H.bind("src/rc_aslo_xf/dino_rcde_sr0_mt_controls_v1.py"),
        "test": H.bind("tests/test_dino_rcde_track_r_pipeline_programs_v1.py"),
        "lineage_test": H.bind(
            "tests/test_dino_rcde_track_r_v121_oof_lineage_v1.py"
        ),
        "freezer": H.bind("programs/freeze_dino_rcde_track_r_oof_prejoin_authority_v120.py"),
        "launcher": H.bind(LAUNCHER),
        "post_vfit_controller": H.bind(POST_VFIT),
        "post_oof_controller": H.bind(POST_OOF),
    }
    authority = {
        "schema_version": "rc_current_authority_v121_20260821",
        "status": STATUS,
        "stage": "TRACK_R_TARGET_FREE_OUTER_OOF_PREJOIN",
        "claim_level": "TARGET_FREE_ROLE_FREE_PAIR_THREE_ARM_AND_CONTROL_PREJOIN_ONLY",
        "parent_v120_authority_revision": 5,
        "parent_v120_authority_sha256": parent_sha256,
        "parent_v120_authority_logical_sha256": parent_logical_sha256,
        "all_fit_outputs_parent_authority_match": True,
        "query_count": 594,
        "excluded_execution_ordinals": [25, 26, 101, 346, 354, 470],
        "shard_count": 50,
        "shard_size": 12,
        "cache_model_checkpoint_logical_sha256": "f901d9bd056bb65e5fcb02c72e869f6d0cf8d7472467d22fbc340442feca034c",
        "oof_prejoin_shard_materialization_authorized": True,
        "oof_prejoin_shard_validation_authorized": True,
        "role_free_pair_read_authorized": True,
        "label_target_rival_read_authorized": False,
        "model_load_authorized": True,
        "model_forward_authorized": True,
        "model_backward_authorized": False,
        "model_update_authorized": False,
        "scientific_reduction_authorized": False,
        "protected_access_authorized": False,
        "scheduler_continuation_authorized": True,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
        "producer_root": "results/dino_rcde_track_r_oof_prejoin_v1/producer",
        "validation_root": "results/dino_rcde_track_r_oof_prejoin_v1/validation",
        "resource_contract": {"allowed_partitions": ["accelerated", "dev_accelerated-h100"], "partition_semantic_binding": False, "gpu_model_pinned": False, "array_spec": "0-49%20", "cpus_per_task": 8, "memory_megabytes": 128000, "gpus_per_node": 1, "walltime_seconds": 7200},
        "bindings": bindings,
    }
    authority["logical_sha256"] = H.logical(authority)
    return authority


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / AUTH)
    args = parser.parse_args()
    req(args.output.resolve(strict=False) == (ROOT / AUTH).resolve(strict=False), "output drift")
    value = build()
    H.atomic(args.output, value)
    print(json.dumps({"status": value["status"], "logical_sha256": value["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
