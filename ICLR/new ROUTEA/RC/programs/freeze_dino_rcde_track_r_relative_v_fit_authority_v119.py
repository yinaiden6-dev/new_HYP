#!/usr/bin/env python3
"""Freeze V119 and exact fresh/resume execution nodes after E1 ledgers PASS."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H


ROOT = Path(__file__).resolve().parents[1]
AUTH = "registry/current_authority_v120_20260821.json"
PARENT = "registry/current_authority_v119_20260821.json"
STATUS = "DINO_RCDE_TRACK_R_RELATIVE_V_FIT_AUTHORIZED"
NODE_ROOT = "protocols/dino_rcde_track_r_relative_v_fit_nodes_v1"
RUNNER = "programs/run_dino_rcde_track_r_relative_v_fit_v1.py"
VALIDATOR = "programs/validate_dino_rcde_track_r_relative_v_fit_v1.py"
LAUNCHER = "slurm/dino_rcde_track_r_relative_v_fit_v1.sbatch"
PACKAGE_INITIALIZER = "src/rc_aslo_xf/__init__.py"
RUNTIME_TEST = "tests/test_dino_rcde_track_r_v120_runtime_lineage_v1.py"


def req(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def latest() -> tuple[int, Path]:
    pattern = re.compile(r"current_authority_v([1-9][0-9]*)_[0-9]{8}\.json")
    rows = [(int(match.group(1)), path.resolve()) for path in (ROOT / "registry").glob("current_authority_v*_*.json") if (match := pattern.fullmatch(path.name)) and path.is_file() and not path.is_symlink()]
    version = max(item[0] for item in rows)
    paths = [path for item_version, path in rows if item_version == version]
    req(len(paths) == 1, "latest authority alias")
    return version, paths[0]


def _module_path(module: str) -> Path | None:
    if module.startswith("rc_aslo_xf"):
        candidate = (ROOT / "src" / Path(*module.split("."))).with_suffix(".py")
    elif "." not in module:
        candidate = (ROOT / "programs" / module).with_suffix(".py")
    else:
        return None
    return candidate.resolve() if candidate.is_file() else None


def _local_imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    output: set[Path] = set()
    src = (ROOT / "src").resolve()
    package = (
        list(path.resolve().relative_to(src).with_suffix("").parts[:-1])
        if src in path.resolve().parents
        else []
    )
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
            local = _module_path(module)
            if local is not None:
                output.add(local)
    return output


def discover_runtime_closure(
    roots: Iterable[str],
) -> tuple[list[str], list[dict[str, str]]]:
    root_paths = [(ROOT / relative).resolve(strict=True) for relative in roots]
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
            closure.add(imported)
            if imported not in visited:
                visited.add(imported)
                pending.append(imported)
    return (
        sorted(path.relative_to(ROOT).as_posix() for path in closure),
        [
            {"importer": importer, "imported": imported}
            for importer, imported in sorted(edges)
        ],
    )


def ensure_node(relative: str, arguments: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / relative
    value = {
        "schema_version": "rc_dino_rcde_track_r_relative_v_fit_execution_node_v1_20260821",
        "program": "run_dino_rcde_track_r_relative_v_fit_v1.py",
        "arguments": arguments,
    }
    value["logical_sha256"] = H.logical(value)
    if path.exists():
        observed = json.loads(path.read_text())
        req(observed == value and (path.stat().st_mode & 0o777) == 0o444, f"execution node drift: {relative}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        H.atomic(path, value)
    return H.bind(relative, with_logical=True, immutable=True)


def build() -> dict[str, Any]:
    version, current = latest()
    parent_path = (ROOT / PARENT).resolve(strict=True)
    self_path = (ROOT / AUTH).resolve(strict=False)
    req((version == 119 and current == parent_path) or (version == 120 and current == self_path and self_path.is_file()), "V119/V120 authority state drift")
    parent = json.loads(parent_path.read_text())
    req(parent.get("status") == "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_REDUCTION_AUTHORIZED", "V119 parent drift")
    index_path = ROOT / "results/dino_rcde_track_r_relative_e1_ledgers_v1/index.json"
    validation_path = ROOT / "results/dino_rcde_track_r_relative_e1_ledgers_validation_v1/result.json"
    index = json.loads(index_path.read_text())
    validation = json.loads(validation_path.read_text())
    req(
        index.get("status") == "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_LEDGER_INDEX_READY"
        and index.get("total_episode_count") == 1782
        and validation.get("status") == "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_LEDGERS_INDEPENDENT_VALIDATION_PASS"
        and validation.get("validation_pass") is True
        and validation.get("index_sha256") == H.file_sha(index_path),
        "E1 ledger independent PASS absent",
    )
    matched = json.loads((ROOT / "results/dino_rcde_r1_main_matched_foldset_v1_0/matched_foldset_manifest.json").read_text())
    node_bindings = {}
    for fold in (1, 2, 3, 4):
        context = next(item["context"] for item in matched["folds"] if item["outer_fold"] == fold)
        common = {
            "protocol": "protocols/dino_rcde_track_r_relative_v_fit_protocol_v1_20260821.json",
            "authority": AUTH,
            "episode_ledger": f"results/dino_rcde_track_r_relative_e1_ledgers_v1/outer_fold{fold}.episodes.pt",
            "episode_index": "results/dino_rcde_track_r_relative_e1_ledgers_v1/index.json",
            "episode_validation": "results/dino_rcde_track_r_relative_e1_ledgers_validation_v1/result.json",
            "redacted_schedule": "results/dino_rcde_r1_oof_redacted_inputs_v1_0/redacted_oof_schedule.json",
            "redacted_cache_index": "results/dino_rcde_r1_oof_redacted_inputs_v1_0/redacted_cache_index.json",
            "redacted_cache_root": "runtime/dino_rcde_p0_v1_2/stable_fp16_cache",
            "geometry_payload": "cache/dino_rcde_colnomic_sr_full600_geometry_v2/full600_geometry_payload_v2.pt",
            "context_checkpoint": context["checkpoint"]["path"],
            "outer_fold": fold,
            "device": "cuda",
        }
        nodes = (
            ("primary", {**common, "output_dir": f"results/dino_rcde_track_r_relative_v_fits_v1/outer_fold{fold}/primary", "stop_after_updates": 2048, "resume": False}),
            ("resume_part1", {**common, "output_dir": f"results/dino_rcde_track_r_relative_v_fits_v1/outer_fold{fold}/resume_replay", "stop_after_updates": 1024, "resume": False}),
            ("resume_part2", {**common, "output_dir": f"results/dino_rcde_track_r_relative_v_fits_v1/outer_fold{fold}/resume_replay", "stop_after_updates": 2048, "resume": True}),
        )
        for name, arguments in nodes:
            relative = f"{NODE_ROOT}/fold{fold}_{name}.json"
            node_bindings[f"fold{fold}_{name}"] = ensure_node(relative, arguments)
    closure_paths, closure_edges = discover_runtime_closure(
        (RUNNER, VALIDATOR, PACKAGE_INITIALIZER)
    )
    closure_rows = [
        {"module": relative.removesuffix(".py").replace("/", "."), **H.bind(relative)}
        for relative in closure_paths
    ]
    bindings = {
        "parent_authority_v119": H.bind(PARENT, with_logical=True, immutable=True),
        "episode_index": H.bind(index_path.relative_to(ROOT).as_posix(), with_logical=True, immutable=True),
        "episode_validation": H.bind(validation_path.relative_to(ROOT).as_posix(), with_logical=True, immutable=True),
        "protocol": H.bind("protocols/dino_rcde_track_r_relative_v_fit_protocol_v1_20260821.json"),
        "runner": H.bind(RUNNER),
        "validator": H.bind(VALIDATOR),
        "package_initializer": H.bind(PACKAGE_INITIALIZER),
        "runtime_import_closure": {
            "root_programs": [RUNNER, VALIDATOR, PACKAGE_INITIALIZER],
            "module_count": len(closure_rows),
            "modules": closure_rows,
            "module_sequence_sha256": H.logical({"rows": closure_rows}),
            "import_edges": closure_edges,
            "import_edge_sequence_sha256": H.logical({"rows": closure_edges}),
        },
        "input_common": H.bind("programs/dino_rcde_track_r_input_common_v1.py"),
        "runtime": H.bind("src/rc_aslo_xf/dino_rcde_sr0_mt_v_runtime_v1.py"),
        "adapter": H.bind("src/rc_aslo_xf/dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1.py"),
        "geometry_payload": H.bind("cache/dino_rcde_colnomic_sr_full600_geometry_v2/full600_geometry_payload_v2.pt"),
        "redacted_schedule": H.bind("results/dino_rcde_r1_oof_redacted_inputs_v1_0/redacted_oof_schedule.json", with_logical=True),
        "redacted_cache_index": H.bind("results/dino_rcde_r1_oof_redacted_inputs_v1_0/redacted_cache_index.json", with_logical=True),
        "matched_foldset": H.bind("results/dino_rcde_r1_main_matched_foldset_v1_0/matched_foldset_manifest.json"),
        "test": H.bind("tests/test_dino_rcde_track_r_pipeline_programs_v1.py"),
        "runtime_lineage_test": H.bind(RUNTIME_TEST),
        "freezer": H.bind("programs/freeze_dino_rcde_track_r_relative_v_fit_authority_v119.py"),
        "launcher": H.bind(LAUNCHER),
        "post_vfit_controller": H.bind("slurm/dino_rcde_track_r_post_vfit_controller_v1.sbatch"),
        "execution_nodes": {"count": 12, "rows": node_bindings, "logical_sha256": H.logical({"rows": node_bindings})},
    }
    for fold in (1, 2, 3, 4):
        context = next(item["context"] for item in matched["folds"] if item["outer_fold"] == fold)
        bindings[f"context_checkpoint_fold{fold}"] = H.bind(context["checkpoint"]["path"])
        bindings[f"episode_ledger_fold{fold}"] = H.bind(f"results/dino_rcde_track_r_relative_e1_ledgers_v1/outer_fold{fold}.episodes.pt", immutable=True)
    authority = {
        "schema_version": "rc_current_authority_v120_20260821",
        "status": STATUS,
        "stage": "TRACK_R_RELATIVE_V_FIT",
        "claim_level": "ENGINEERING_FOUR_RELATIVE_ONLY_V_FITS_AND_FRESH_RESUME_VALIDATION",
        "authorized_outer_folds": [1, 2, 3, 4],
        "relative_v_training_authorized": True,
        "relative_v_validation_authorized": True,
        "model_load_authorized": True,
        "model_forward_authorized": True,
        "model_backward_authorized": True,
        "model_update_authorized": True,
        "donor_null_absolute_loss_authorized": False,
        "heldout_scoring_authorized": False,
        "protected_access_authorized": False,
        "scheduler_continuation_authorized": True,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
        "resource_contract": {"allowed_partitions": ["accelerated", "dev_accelerated-h100"], "partition_semantic_binding": False, "gpu_model_pinned": False, "array_spec": "1-4%4", "cpus_per_task": 4, "memory_megabytes": 64000, "gpus_per_node": 1, "walltime_seconds": 28800},
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
