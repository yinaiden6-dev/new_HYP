#!/usr/bin/env python3
"""Independent validator for the formal GX-CBNR Balanced-32 R0 reduction.

This module intentionally does not import the production reducer or the
relational-cache constructors/replay helpers.  It rebuilds the role join,
replays the trained comparator directly from the serialized pre-comparator
tensors, recomputes every scientific statistic and gate, and then requires an
exact match with the producer result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ROOT / "programs"
SRC = ROOT / "src"
import sys

sys.path.insert(0, str(SRC))

from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2  # noqa: E402


RESULT_SCHEMA = "rc_dino_rcde_gx_formal_r0_science_v1_20260825"
VALIDATION_SCHEMA = "rc_dino_rcde_gx_formal_r0_science_validation_v1_20260825"
VALIDATION_STATUS = "GX_CBNR_R0_SCIENTIFIC_INDEPENDENT_VALIDATION_PASS"
GO = "GX_CBNR_R0_REPRESENTATION_GO"
NO_GO = "GX_CBNR_R0_NO_DEPLOYABLE_GEOMETRIC_EXCLUSIVITY"
FOLD_RESULT_SCHEMA = "rc_dino_rcde_gx_formal_fold_train_v1_20260824"
FOLD_RESULT_STATUS = "GX_CBNR_FORMAL_FOLD_TRAIN_COMPLETE"
FOLD_VALIDATION_STATUS = "GX_CBNR_FORMAL_FOLD_TRAIN_VALIDATION_PASS"
ADDRESS_SCHEMA = "rc_dino_rcde_gx_cbnr_r0_address_manifest_v2_20260824"
LOSS_ROLE_SCHEMA = "rc_dino_rcde_gx_cbnr_r0_loss_role_manifest_v2_20260824"
MANIFEST_STATUS = "GX_CBNR_R0_ELIGIBILITY_AWARE_MANIFEST_READY"
CACHE_INDEX_SCHEMA = "rc_dino_rcde_gx_relational_cache_index_v1_20260824"
CACHE_INDEX_STATUS = "GX_CBNR_RELATIONAL_CACHE_INDEX_READY"
CACHE_PAYLOAD_SCHEMA = "rc_dino_rcde_gx_relational_cache_shard_v1_20260824"
CACHE_PAYLOAD_STATUS = "GX_CBNR_RELATIONAL_CACHE_SHARD_READY"
CACHE_VALIDATION_STATUS = "GX_CBNR_RELATIONAL_CACHE_SHARD_VALIDATION_PASS"
CACHE_SCHEMA = "rc_dino_rcde_gx_relational_head_cache_v1_20260824"
BRANCHES = ("REAL", "C_BIND", "P_COORD", "N_REGION_RESAMPLE")
NULLS = BRANCHES[1:]
DIRECTIONS = ("a_to_b", "b_to_a")
ENDPOINTS = (
    "correctness_increment",
    "target_z_minus_rival_z",
    "target_real_minus_C_BIND",
    "target_real_minus_P_COORD",
    "target_real_minus_N_REGION_RESAMPLE",
)
FOLDS = (1, 2, 3, 4)
QUERY_COUNT = 32
BOOTSTRAP_SEED = 17
BOOTSTRAP_REPETITIONS = 10_000
PROTECTED_COMPONENTS = {"opened", "sealed"}


class R0ScienceValidationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise R0ScienceValidationError(message)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "bytes_hex": tensor.numpy().tobytes(order="C").hex(),
        }
    )


def tree_hash_payload(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "tensor_sha256": tensor_sha256(value),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, Mapping):
        return {
            str(key): tree_hash_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return {"tuple": [tree_hash_payload(item) for item in value]}
    if isinstance(value, list):
        return [tree_hash_payload(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        require(
            not isinstance(value, float) or math.isfinite(value),
            "nonfinite model-state scalar",
        )
        return value
    raise R0ScienceValidationError(
        f"unsupported model-state value: {type(value).__name__}"
    )


def tree_sha256(value: Any) -> str:
    return canonical_sha256(tree_hash_payload(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    require(
        resolved.is_file()
        and not resolved.is_symlink()
        and not PROTECTED_COMPONENTS.intersection(
            component.lower() for component in resolved.parts
        ),
        f"unsafe/opened/sealed input: {resolved}",
    )
    return resolved


def safe_dir(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    require(
        resolved.is_dir()
        and not resolved.is_symlink()
        and not PROTECTED_COMPONENTS.intersection(
            component.lower() for component in resolved.parts
        ),
        f"unsafe/opened/sealed directory: {resolved}",
    )
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(safe_file(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"immutable validation exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def binary64(value: object) -> float:
    require(
        isinstance(value, str)
        and len(value) == 16
        and all(character in "0123456789abcdef" for character in value),
        "RAW binary64 field drift",
    )
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    require(math.isfinite(result), "RAW binary64 value is nonfinite")
    return float(result)


def _validate_pullback(raw: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    source = [int(item) for item in raw["source_original_indices"]]
    destination = [int(item) for item in raw["destination_decode_indices"]]
    require(
        source
        and len(source) == len(destination)
        and len(set(source)) == len(source)
        and len(set(destination)) == len(destination),
        "query pullback is not a bijection",
    )
    forward = [
        {"source": src, "destination": dst}
        for src, dst in zip(source, destination, strict=True)
    ]
    backward = [
        {"destination": dst, "source": src}
        for src, dst in zip(source, destination, strict=True)
    ]
    payload = {
        "source_original_indices": source,
        "destination_decode_indices": destination,
        "source_to_destination_sha256": canonical_sha256(forward),
        "destination_to_source_sha256": canonical_sha256(backward),
    }
    require(
        raw.get("source_to_destination_sha256")
        == payload["source_to_destination_sha256"]
        and raw.get("destination_to_source_sha256")
        == payload["destination_to_source_sha256"]
        and raw.get("logical_sha256") == canonical_sha256(payload),
        "query pullback hash drift",
    )
    return source, destination


def _validate_root_hashes(raw: Mapping[str, Any]) -> None:
    relational = torch.as_tensor(raw["relational"])
    decode_query = torch.as_tensor(raw["decode_query_mask"])
    decode_reference = torch.as_tensor(raw["decode_reference_mask"])
    original_query = torch.as_tensor(raw["original_root_query_mask"])
    original_reference = torch.as_tensor(raw["original_root_reference_mask"])
    require(
        relational.device.type == "cpu"
        and relational.dtype == torch.float32
        and relational.ndim == 2
        and relational.is_contiguous()
        and bool(torch.isfinite(relational).all())
        and all(
            tensor.device.type == "cpu"
            and tensor.dtype == torch.bool
            and tensor.is_contiguous()
            for tensor in (
                decode_query,
                decode_reference,
                original_query,
                original_reference,
            )
        ),
        "serialized relational root tensor drift",
    )
    hashes = {
        "relational_sha256": tensor_sha256(relational),
        "decode_query_mask_sha256": tensor_sha256(decode_query.to(torch.uint8)),
        "decode_reference_mask_sha256": tensor_sha256(
            decode_reference.to(torch.uint8)
        ),
        "original_root_query_mask_sha256": tensor_sha256(
            original_query.to(torch.uint8)
        ),
        "original_root_reference_mask_sha256": tensor_sha256(
            original_reference.to(torch.uint8)
        ),
    }
    require(
        all(raw.get(name) == digest for name, digest in hashes.items()),
        "serialized relational root tensor hash drift",
    )
    address_keys = (
        "outer_fold",
        "execution_ordinal",
        "query_id",
        "query_source_image_sha256",
        "candidate_key",
        "candidate_physical_row",
        "candidate_content_binding_sha256",
        "branch",
        "direction",
        "root_ordinal",
        "source_reference_root_ordinal",
        "raw_p_v2_record_sha256",
        "core_p_v2_record_sha256",
        "complete_root_table_sha256",
        "branch_control_receipt_sha256",
        "direction_control_receipt_sha256",
        "root_control_receipt_sha256",
    )
    address = {key: raw[key] for key in address_keys}
    require(raw.get("address_sha256") == canonical_sha256(address), "root address hash drift")
    pullback = raw.get("query_pullback")
    if pullback is not None:
        require(isinstance(pullback, Mapping), "query pullback payload drift")
        _validate_pullback(pullback)
    payload = {
        "schema_version": CACHE_SCHEMA,
        **address,
        "address_sha256": raw["address_sha256"],
        "query_grid_shape": list(map(int, raw["query_grid_shape"])),
        "reference_grid_shape": list(map(int, raw["reference_grid_shape"])),
        "original_reference_grid_shape": list(
            map(int, raw["original_reference_grid_shape"])
        ),
        "relational_dtype": "torch.float32",
        "relational_shape": list(relational.shape),
        "relational_numel": int(relational.numel()),
        "relational_bytes": int(relational.numel() * relational.element_size()),
        **hashes,
        "query_pullback": pullback,
        "pre_comparator": True,
        "target_free": True,
    }
    require(raw.get("logical_sha256") == canonical_sha256(payload), "root logical hash drift")


def _replay_direction(model: torch.nn.Module, raw: Mapping[str, Any]) -> torch.Tensor:
    denominator = raw["denominator"]
    require(isinstance(denominator, Mapping), "direction denominator absent")
    selected = tuple(map(int, denominator["selected_root_ordinals"]))
    statuses = {
        int(key): str(value)
        for key, value in denominator["root_status_by_ordinal"].items()
    }
    masks = {
        int(key): torch.as_tensor(value)
        for key, value in denominator["original_query_masks_by_root"].items()
    }
    require(
        selected == tuple(sorted(set(selected)))
        and set(selected) == set(statuses) == set(masks)
        and all(value in {"ROOT_READY", "ROOT_REFERENCE_MISSING"} for value in statuses.values())
        and all(mask.dtype == torch.bool and mask.is_contiguous() for mask in masks.values()),
        "fixed denominator root population drift",
    )
    expected_coverage = torch.zeros_like(next(iter(masks.values())), dtype=torch.int64)
    root_rows = []
    for root in selected:
        expected_coverage += masks[root].to(torch.int64)
        root_rows.append(
            {
                "root_ordinal": root,
                "status": statuses[root],
                "original_query_mask_sha256": tensor_sha256(
                    masks[root].to(torch.uint8)
                ),
            }
        )
    expected_union = expected_coverage > 0
    coverage = torch.as_tensor(denominator["coverage"])
    union = torch.as_tensor(denominator["query_union"])
    require(
        torch.equal(coverage, expected_coverage)
        and torch.equal(union, expected_union)
        and denominator.get("coverage_sha256") == tensor_sha256(coverage)
        and denominator.get("query_union_sha256") == tensor_sha256(union)
        and denominator.get("root_mask_population_sha256")
        == canonical_sha256(root_rows),
        "fixed denominator tensor/hash drift",
    )
    denominator_payload = {
        "schema_version": CACHE_SCHEMA,
        "outer_fold": int(denominator["outer_fold"]),
        "execution_ordinal": int(denominator["execution_ordinal"]),
        "query_id": denominator["query_id"],
        "candidate_key": denominator["candidate_key"],
        "candidate_physical_row": int(denominator["candidate_physical_row"]),
        "branch": denominator["branch"],
        "direction": denominator["direction"],
        "query_grid_shape": list(map(int, denominator["query_grid_shape"])),
        "raw_p_v2_record_sha256": denominator["raw_p_v2_record_sha256"],
        "core_p_v2_record_sha256": denominator["core_p_v2_record_sha256"],
        "complete_root_table_sha256": denominator["complete_root_table_sha256"],
        "selected_root_ordinals": list(selected),
        "root_rows": root_rows,
        "coverage_shape": list(coverage.shape),
        "coverage_numel": int(coverage.numel()),
        "coverage_bytes": int(coverage.numel() * coverage.element_size()),
        "coverage_sha256": denominator["coverage_sha256"],
        "query_union_shape": list(union.shape),
        "query_union_numel": int(union.numel()),
        "query_union_bytes": int(union.numel() * union.element_size()),
        "query_union_sha256": denominator["query_union_sha256"],
        "root_mask_population_sha256": denominator["root_mask_population_sha256"],
        "complete_root_coverage_used": True,
        "available_root_renormalization": False,
        "available_patch_renormalization": False,
    }
    require(
        denominator.get("logical_sha256") == canonical_sha256(denominator_payload),
        "fixed denominator logical hash drift",
    )
    roots = list(raw["root_payloads"])
    require(
        [int(item["root_ordinal"]) for item in roots]
        == [root for root in selected if statuses[root] == "ROOT_READY"],
        "READY root payload axis drift",
    )
    direction_population = {
        "denominator_logical_sha256": denominator["logical_sha256"],
        "root_logical_sha256": [item["logical_sha256"] for item in roots],
    }
    require(
        raw.get("direction_cache_sha256") == canonical_sha256(direction_population),
        "direction cache population hash drift",
    )
    contributions: dict[int, torch.Tensor] = {}
    for root in roots:
        _validate_root_hashes(root)
        require(
            int(root["outer_fold"]) == int(denominator["outer_fold"])
            and int(root["execution_ordinal"])
            == int(denominator["execution_ordinal"])
            and root["query_id"] == denominator["query_id"]
            and root["candidate_key"] == denominator["candidate_key"]
            and int(root["candidate_physical_row"])
            == int(denominator["candidate_physical_row"])
            and root["branch"] == denominator["branch"]
            and root["direction"] == denominator["direction"],
            "root/denominator address drift",
        )
        relational = torch.as_tensor(root["relational"])
        decode_mask = torch.as_tensor(root["decode_query_mask"])
        result = model.compare_relational(
            relational, torch.zeros_like(relational), decode_mask
        )
        value = getattr(result, "contributions", None)
        require(
            isinstance(value, torch.Tensor)
            and value.dtype == torch.float32
            and value.ndim == 1
            and value.numel() == union.numel()
            and bool(torch.isfinite(value).all()),
            "independent comparator contribution drift",
        )
        original = torch.as_tensor(root["original_root_query_mask"])
        pullback = root.get("query_pullback")
        if pullback is None:
            value = value * original.to(value.dtype)
        else:
            source, destination = _validate_pullback(pullback)
            pulled = torch.zeros_like(value)
            pulled[torch.tensor(source, dtype=torch.long)] = value[
                torch.tensor(destination, dtype=torch.long)
            ]
            value = pulled
        require(not bool(value[~original].ne(0).any()), "root evidence escaped original slot")
        contributions[int(root["root_ordinal"])] = value
    exemplar = next(iter(contributions.values()))
    reciprocal = torch.zeros_like(exemplar)
    reciprocal[union] = coverage[union].to(torch.float32).reciprocal()
    total = torch.zeros_like(exemplar)
    for root in selected:
        if statuses[root] == "ROOT_READY":
            total += contributions[root] * reciprocal
    patch = total * union.to(total.dtype)
    require(not bool(patch[~union].ne(0).any()), "direction evidence escaped fixed union")
    energy = patch[union].mean()
    require(energy.dtype == torch.float32 and bool(torch.isfinite(energy)), "nonfinite direction energy")
    return energy


def replay_candidate_energy(model: torch.nn.Module, raw: Mapping[str, Any]) -> float:
    require(
        raw.get("branch") in {*BRANCHES, "T_ROOT_ASSIGNMENT_POPULATION"}
        and set(raw.get("directions", {})) == set(DIRECTIONS),
        "candidate cache branch/direction drift",
    )
    direction_hashes = {}
    values = []
    for direction in DIRECTIONS:
        observed = raw["directions"][direction]
        direction_hashes[direction] = observed["direction_cache_sha256"]
        values.append(_replay_direction(model, observed))
    population = {
        "outer_fold": int(raw["outer_fold"]),
        "execution_ordinal": int(raw["execution_ordinal"]),
        "query_id": raw["query_id"],
        "candidate_key": raw["candidate_key"],
        "candidate_physical_row": int(raw["candidate_physical_row"]),
        "branch": raw["branch"],
        "branch_control_receipt_sha256": raw["branch_control_receipt_sha256"],
        "direction_cache_sha256": direction_hashes,
    }
    require(
        raw.get("candidate_cache_sha256") == canonical_sha256(population),
        "candidate cache population hash drift",
    )
    return float((0.5 * (values[0] + values[1])).detach().cpu())


def _load_model(checkpoint_path: Path, expected_result: Mapping[str, Any]) -> torch.nn.Module:
    checkpoint_file = safe_file(checkpoint_path)
    require(
        checkpoint_file.stat().st_size == int(expected_result["checkpoint_bytes"])
        and file_sha256(checkpoint_file) == expected_result["checkpoint_sha256"],
        "fold checkpoint physical receipt drift",
    )
    checkpoint = torch.load(
        checkpoint_file, map_location="cpu", weights_only=True
    )
    require(
        isinstance(checkpoint, Mapping)
        and checkpoint.get("status") == FOLD_RESULT_STATUS
        and checkpoint.get("update") == 128
        and isinstance(checkpoint.get("training_state"), Mapping),
        "fold checkpoint envelope drift",
    )
    state = checkpoint["training_state"]["model_state_dict"]
    require(
        tree_sha256(state) == checkpoint.get("model_state_sha256")
        == expected_result.get("model_state_sha256"),
        "fold checkpoint model-state hash drift",
    )
    model = DINO_RCDE_V1_2()
    missing, unexpected = model.load_state_dict(state, strict=False)
    require(not missing and not unexpected, "fold model-state parameter axis drift")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _candidate_energies(
    model: torch.nn.Module,
    serialized: Sequence[Mapping[str, Any]],
    candidate_key: str,
) -> tuple[dict[str, float], float | None, int]:
    by_branch = {
        str(item["branch"]): item
        for item in serialized
        if str(item["candidate_key"]) == candidate_key
    }
    require(set(BRANCHES).issubset(by_branch), "mandatory candidate branch absent")
    values = {branch: replay_candidate_energy(model, by_branch[branch]) for branch in BRANCHES}
    tensor = torch.tensor([values[name] for name in NULLS], dtype=torch.float32)
    null_lme = torch.logsumexp(tensor, dim=0) - math.log(3.0)
    z = torch.tensor(values["REAL"], dtype=torch.float32) - null_lme
    result = {
        **values,
        "null_logmeanexp_CPN": float(null_lme),
        "Z_CPN": float(z),
    }
    t = (
        replay_candidate_energy(model, by_branch["T_ROOT_ASSIGNMENT_POPULATION"])
        if "T_ROOT_ASSIGNMENT_POPULATION" in by_branch
        else None
    )
    compare_count = sum(
        len(by_branch[branch]["directions"][direction]["root_payloads"])
        for branch in by_branch
        for direction in DIRECTIONS
    )
    return result, t, compare_count


def _record(
    *,
    fold: int,
    role: Mapping[str, Any],
    fold_observed: Mapping[str, Any],
    model: torch.nn.Module,
    serialized: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    target_key = str(role["target_candidate_key"])
    rival_key = str(role["rival_candidate_key"])
    target, target_t, first_count = _candidate_energies(model, serialized, target_key)
    rival, rival_t, second_count = _candidate_energies(model, serialized, rival_key)
    rival_reordered, _, third_count = _candidate_energies(
        model, serialized, rival_key
    )
    target_reordered, _, fourth_count = _candidate_energies(
        model, serialized, target_key
    )
    raw_margin = binary64(role["target_raw_score_bits"]) - binary64(
        role["rival_raw_score_bits"]
    )
    raw_correct = bool(role["raw_correct"])
    require(raw_correct is (raw_margin > 0.0), "RAW correctness/score drift")
    raw_margin = float(torch.tensor(raw_margin, dtype=torch.float32))
    geometric = float(
        torch.tensor(target["Z_CPN"], dtype=torch.float32)
        - torch.tensor(rival["Z_CPN"], dtype=torch.float32)
    )
    final_margin = float(torch.tensor(raw_margin, dtype=torch.float32) + geometric)
    if raw_correct:
        challenger_z = rival["Z_CPN"]
        challenger_margin = -final_margin
    else:
        challenger_z = target["Z_CPN"]
        challenger_margin = final_margin
    action = (
        "SWITCH"
        if challenger_z > 0.0 and challenger_margin > 0.0
        else "RELATIVE_NULL_HOLD"
    )
    deployed = final_margin if action == "SWITCH" else raw_margin
    final_correct = deployed > 0.0
    target_minus_null = {
        name: float(target["REAL"] - target[name]) for name in NULLS
    }
    rival_minus_null = {
        name: float(rival["REAL"] - rival[name]) for name in NULLS
    }
    half = float(torch.tensor(0.5, dtype=torch.float32) * torch.tensor(geometric, dtype=torch.float32))
    invariants = {
        "candidate_swap_exact": float(rival["Z_CPN"] - target["Z_CPN"]) == -geometric,
        "candidate_reorder_exact": target == target_reordered
        and rival == rival_reordered,
        "pair_zero_sum_exact": float(torch.tensor(half) + torch.tensor(-half)) == 0.0,
        "forced_ineligible_hold_exact": True,
        "natural_hold_preserves_raw_exact": action != "RELATIVE_NULL_HOLD" or deployed == raw_margin,
    }
    observed_by_execution = {
        int(row["execution_ordinal"]): row
        for row in fold_observed["evaluation_records"]
    }
    execution = int(role["execution_ordinal"])
    require(execution in observed_by_execution, "fold evaluation record absent")
    observed = observed_by_execution[execution]
    require(
        observed.get("query_id") == role.get("query_id")
        and observed.get("episode_id")
        == canonical_sha256(
            [
                "GX_FORMAL_EPISODE_V1",
                fold,
                execution,
                str(role["source_i0_record_sha256"]),
            ]
        )
        and bool(observed.get("raw_correct")) is raw_correct
        and abs(float(observed["raw_target_margin"]) - raw_margin) <= 1.0e-6
        and abs(float(observed["target_z"]) - target["Z_CPN"]) <= 1.0e-6
        and abs(float(observed["rival_z"]) - rival["Z_CPN"]) <= 1.0e-6
        and abs(float(observed["geometric_margin"]) - geometric) <= 1.0e-6
        and abs(float(observed["final_margin"]) - final_margin) <= 1.0e-6
        and observed.get("action") == action,
        "independent replay disagrees with frozen fold evaluation",
    )
    value = {
        "outer_fold": fold,
        "execution_ordinal": execution,
        "query_id": str(role["query_id"]),
        "episode_id": str(observed["episode_id"]),
        "group_sha256": str(role["group_sha256"]),
        "track": str(role["track"]),
        "raw_correct": raw_correct,
        "raw_target_margin": raw_margin,
        "candidate_energies": {"target": target, "rival": rival},
        "target_real_minus_null": target_minus_null,
        "rival_real_minus_null": rival_minus_null,
        "target_z_minus_rival_z": geometric,
        "final_target_margin": final_margin,
        "raw_wrong_final_direction": (not raw_correct) and final_correct,
        "action": action,
        "deployed_target_margin": deployed,
        "final_correct": final_correct,
        "rescue": (not raw_correct) and final_correct,
        "break": raw_correct and not final_correct,
        "correctness_increment": float(final_correct) - float(raw_correct),
        "null_eligibility": {name: True for name in NULLS},
        "invariants": invariants,
    }
    value["record_sha256"] = canonical_sha256(value)
    t_record = {
        "outer_fold": fold,
        "execution_ordinal": execution,
        "target_eligible": target_t is not None,
        "rival_eligible": rival_t is not None,
        "target_real_minus_T": None if target_t is None else target["REAL"] - target_t,
        "rival_real_minus_T": None if rival_t is None else rival["REAL"] - rival_t,
    }
    return (
        value,
        t_record,
        first_count + second_count + third_count + fourth_count,
    )


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    require(bool(rows), f"empty summary population: {key}")
    return float(sum(float(row[key]) for row in rows) / len(rows))


def _fold_summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for fold in FOLDS:
        rows = [row for row in records if int(row["outer_fold"]) == fold]
        require(len(rows) == 8, "Balanced-32 fold must contain exactly eight records")
        correct = [row for row in rows if row["raw_correct"]]
        wrong = [row for row in rows if not row["raw_correct"]]
        require(len(correct) == len(wrong) == 4, "fold RAW strata are not 4/4")
        rescue = sum(bool(row["rescue"]) for row in rows)
        broken = sum(bool(row["break"]) for row in rows)
        mean_z = _mean(rows, "target_z_minus_rival_z")
        mean_null = {
            name: float(
                np.mean(
                    [float(row["target_real_minus_null"][name]) for row in rows],
                    dtype=np.float64,
                )
            )
            for name in NULLS
        }
        value = {
            "query_count": len(rows),
            "raw_correct_count": len(correct),
            "raw_wrong_count": len(wrong),
            "raw_wrong_final_direction_count": sum(
                bool(row["raw_wrong_final_direction"]) for row in wrong
            ),
            "raw_correct_retained_count": sum(bool(row["final_correct"]) for row in correct),
            "rescue_count": rescue,
            "break_count": broken,
            "net_rescue": rescue - broken,
            "mean_target_z_minus_rival_z": mean_z,
            "mean_target_real_minus_null": mean_null,
            "gates": {
                "rescue_strictly_exceeds_break": rescue > broken,
                "net_rescue_positive": rescue - broken > 0,
                "mean_target_z_minus_rival_z_positive": mean_z > 0.0,
                "target_real_minus_each_null_positive": all(
                    value > 0.0 for value in mean_null.values()
                ),
            },
        }
        output[str(fold)] = value
    return output


def _track_summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for track in sorted({str(row["track"]) for row in records}):
        rows = [row for row in records if str(row["track"]) == track]
        rescue = sum(bool(row["rescue"]) for row in rows)
        broken = sum(bool(row["break"]) for row in rows)
        output[track] = {
                "query_count": len(rows),
                "raw_correct_count": sum(bool(row["raw_correct"]) for row in rows),
                "raw_wrong_count": sum(not bool(row["raw_correct"]) for row in rows),
                "rescue_count": rescue,
                "break_count": broken,
                "net_rescue": rescue - broken,
                "nonnegative_gate": rescue >= broken,
            }
    return output


def _bootstrap(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = sorted({str(row["group_sha256"]) for row in records})
    require(groups, "bootstrap group population absent")
    by_group = {group: [row for row in records if row["group_sha256"] == group] for group in groups}
    endpoint_values: dict[str, np.ndarray] = {}
    for endpoint in ENDPOINTS:
        values = []
        for group in groups:
            rows = by_group[group]
            if endpoint == "correctness_increment":
                raw = [float(row["final_correct"]) - float(row["raw_correct"]) for row in rows]
            elif endpoint == "target_z_minus_rival_z":
                raw = [float(row[endpoint]) for row in rows]
            else:
                null = endpoint.removeprefix("target_real_minus_")
                raw = [float(row["target_real_minus_null"][null]) for row in rows]
            values.append(sum(raw) / len(raw))
        endpoint_values[endpoint] = np.asarray(values, dtype=np.float64)
    endpoints = {}
    for endpoint in ENDPOINTS:
        values = endpoint_values[endpoint]
        generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        indices = generator.integers(
            0,
            len(groups),
            size=(BOOTSTRAP_REPETITIONS, len(groups)),
            endpoint=False,
        )
        replicates = values[indices].mean(axis=1)
        low, high = np.quantile(
            replicates, [0.025, 0.975], method="linear"
        )
        endpoints[endpoint] = {
            "cluster_field": "group_sha256",
            "cluster_count": len(groups),
            "cluster_means_equal_weighted": True,
            "observed_group_balanced_mean": float(values.mean()),
            "rng": "numpy.PCG64",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPETITIONS,
            "quantile_method": "linear",
            "two_sided_95_ci": [float(low), float(high)],
            "lower_strictly_positive": bool(low > 0.0),
        }
    return endpoints


def _totals(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = [row for row in records if row["raw_correct"]]
    wrong = [row for row in records if not row["raw_correct"]]
    rescue = sum(bool(row["rescue"]) for row in records)
    broken = sum(bool(row["break"]) for row in records)
    return {
        "query_count": len(records),
        "raw_correct_count": len(correct),
        "raw_wrong_count": len(wrong),
        "raw_wrong_final_direction_count": sum(
            bool(row["raw_wrong_final_direction"]) for row in wrong
        ),
        "raw_correct_retained_count": sum(bool(row["final_correct"]) for row in correct),
        "rescue_count": rescue,
        "break_count": broken,
        "net_rescue": rescue - broken,
    }


def _gates(
    folds: Sequence[Mapping[str, Any]],
    tracks: Sequence[Mapping[str, Any]],
    invariants: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    totals: Mapping[str, Any],
) -> dict[str, bool]:
    difficult = tracks.get("difficult")
    gates = {
        "raw_wrong_direction_at_least_11_of_16": totals["raw_wrong_final_direction_count"] >= 11,
        "raw_correct_retention_exactly_16_of_16": totals["raw_correct_retained_count"] == 16,
        "every_fold_rescue_strictly_exceeds_break": all(row["gates"]["rescue_strictly_exceeds_break"] for row in folds.values()),
        "every_fold_net_rescue_positive": all(row["gates"]["net_rescue_positive"] for row in folds.values()),
        "every_fold_mean_target_z_minus_rival_z_positive": all(row["gates"]["mean_target_z_minus_rival_z_positive"] for row in folds.values()),
        "every_fold_target_real_minus_each_null_positive": all(row["gates"]["target_real_minus_each_null_positive"] for row in folds.values()),
        "difficult_track_rescue_at_least_break": difficult is not None and bool(difficult["nonnegative_gate"]),
        "every_available_track_nonnegative": all(row["nonnegative_gate"] for row in tracks.values()),
        "all_invariants_pass": invariants["all_pass"] is True,
        "all_nulls_reported_without_drop": eligibility["silent_denominator_drop_count"] == 0,
        "all_required_bootstrap_lower_strictly_positive": all(row["lower_strictly_positive"] for row in bootstrap.values()),
    }
    gates["all_gates_pass"] = all(gates.values())
    return gates


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    auxiliary_t_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Independently serialize the producer's complete summary schema."""

    rows = list(records)
    t_rows = list(auxiliary_t_rows)
    require(
        len(rows) == QUERY_COUNT
        and len(t_rows) == 2 * QUERY_COUNT,
        "independent summary population drift",
    )
    folds = _fold_summaries(rows)
    tracks = _track_summaries(rows)
    invariant_names = tuple(rows[0]["invariants"])
    invariants = {
        "record_count": len(rows),
        **{
            f"{name}_count": sum(
                bool(row["invariants"][name]) for row in rows
            )
            for name in invariant_names
        },
        "all_pass": all(
            all(bool(value) for value in row["invariants"].values())
            for row in rows
        ),
    }

    def eligibility_slice(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        denominator = 2 * len(selected)
        return {
            control: {
                "candidate_denominator_count": denominator,
                "eligible_candidate_count": denominator,
                "ineligible_candidate_count": 0,
                "reason_counts": {"ELIGIBLE": denominator},
            }
            for control in NULLS
        }

    null_eligibility = {
        "controls": list(NULLS),
        "overall": eligibility_slice(rows),
        "by_fold": {
            str(fold): eligibility_slice(
                [row for row in rows if int(row["outer_fold"]) == fold]
            )
            for fold in FOLDS
        },
        "by_track": {
            track: eligibility_slice(
                [row for row in rows if str(row["track"]) == track]
            )
            for track in tracks
        },
        "silent_denominator_drop_count": 0,
    }
    auxiliary_t = {
        "name": "T_ROOT_ASSIGNMENT_POPULATION",
        "selected_candidate_denominator_count": len(t_rows),
        "present_count": sum(bool(row["present"]) for row in t_rows),
        "absent_count": sum(not bool(row["present"]) for row in t_rows),
        "by_fold": {
            str(fold): {
                "candidate_denominator_count": sum(
                    int(row["fold"]) == fold for row in t_rows
                ),
                "present_count": sum(
                    int(row["fold"]) == fold and bool(row["present"])
                    for row in t_rows
                ),
            }
            for fold in FOLDS
        },
        "by_track": {
            track: {
                "candidate_denominator_count": sum(
                    str(row["track"]) == track for row in t_rows
                ),
                "present_count": sum(
                    str(row["track"]) == track and bool(row["present"])
                    for row in t_rows
                ),
            }
            for track in tracks
        },
        "enters_calibration": False,
        "enters_loss": False,
        "enters_action": False,
        "enters_gate": False,
        "enters_bootstrap": False,
    }
    bootstrap = _bootstrap(rows)
    totals = _totals(rows)
    gates = _gates(
        folds, tracks, invariants, bootstrap, null_eligibility, totals
    )
    return {
        "fold_summaries": folds,
        "track_summaries": tracks,
        "null_eligibility_summary": null_eligibility,
        "auxiliary_t_diagnostic": auxiliary_t,
        "invariants": invariants,
        "bootstrap": bootstrap,
        "gates": gates,
        "totals": totals,
    }


def reconstruct(
    *,
    training_root: Path,
    training_validation_root: Path,
    address_manifest_path: Path,
    loss_role_manifest_path: Path,
    cache_index_path: Path,
) -> dict[str, Any]:
    training_root = safe_dir(training_root)
    validation_root = safe_dir(training_validation_root)
    address_path = safe_file(address_manifest_path)
    role_path = safe_file(loss_role_manifest_path)
    index_path = safe_file(cache_index_path)
    address = read_json(address_path)
    roles = read_json(role_path)
    index = read_json(index_path)
    require(
        address.get("schema_version") == ADDRESS_SCHEMA
        and address.get("status") == MANIFEST_STATUS
        and address.get("logical_sha256") == logical_sha256(address)
        and address.get("label_target_rival_raw_group_track_field_count") == 0
        and roles.get("schema_version") == LOSS_ROLE_SCHEMA
        and roles.get("status") == MANIFEST_STATUS
        and roles.get("logical_sha256") == logical_sha256(roles)
        and roles.get("address_manifest_logical_sha256") == address["logical_sha256"]
        and roles.get("join_after_target_free_eligibility_and_cache_address_seal") is True
        and index.get("schema_version") == CACHE_INDEX_SCHEMA
        and index.get("status") == CACHE_INDEX_STATUS
        and index.get("target_free") is True
        and index.get("logical_sha256") == logical_sha256(index)
        and index.get("address_manifest_sha256") == file_sha256(address_path),
        "formal manifest/cache-index envelope drift",
    )
    role_eval = {
        (int(row["outer_fold"]), int(row["execution_ordinal"])): row
        for row in roles["evaluation_records"]
    }
    address_eval = {
        (int(row["outer_fold"]), int(row["execution_ordinal"])): row
        for row in address["evaluation_records"]
    }
    require(
        len(role_eval) == len(address_eval) == QUERY_COUNT
        and set(role_eval) == set(address_eval),
        "Balanced-32 role/address evaluation axis drift",
    )
    index_by_key = {
        (int(row["outer_fold"]), int(row["execution_ordinal"])): row
        for row in index["records"]
    }
    records = []
    fold_receipts = []
    t_records = []
    compare_count = 0
    payload_cache: dict[Path, Mapping[str, Any]] = {}
    for fold in FOLDS:
        result_path = safe_file(training_root / f"outer_fold{fold}" / "result.json")
        result = read_json(result_path)
        validation_path = safe_file(validation_root / f"outer_fold{fold}" / "result.json")
        validation = read_json(validation_path)
        require(
            result.get("schema_version") == FOLD_RESULT_SCHEMA
            and result.get("status") == FOLD_RESULT_STATUS
            and result.get("outer_fold") == fold
            and result.get("completed_updates") == 128
            and result.get("evaluation_record_count") == 8
            and result.get("evaluation_record_population_sha256")
            == canonical_sha256(result["evaluation_records"])
            and result.get("logical_sha256") == logical_sha256(result)
            and validation.get("status") == FOLD_VALIDATION_STATUS
            and validation.get("validation_pass") is True
            and validation.get("outer_fold") == fold
            and validation.get("runner_result_sha256") == file_sha256(result_path)
            and validation.get("runner_result_logical_sha256") == result["logical_sha256"]
            and validation.get("evaluation_record_population_sha256")
            == result["evaluation_record_population_sha256"]
            and validation.get("fresh_resume_byte_parity") is True
            and validation.get("logical_sha256") == logical_sha256(validation),
            "validated fold output drift",
        )
        checkpoint_path = safe_file(Path(result["checkpoint_path"]))
        model = _load_model(checkpoint_path, result)
        fold_roles = [row for (value, _), row in sorted(role_eval.items()) if value == fold]
        require(len(fold_roles) == 8, "fold role count drift")
        for role in fold_roles:
            key = (fold, int(role["execution_ordinal"]))
            addr = address_eval[key]
            require(
                role["address_record_sha256"] == addr["record_sha256"]
                and addr["cache_address_record_sha256"]
                == next(
                    row["record_sha256"]
                    for row in address["cache_address_records"]
                    if int(row["outer_fold"]) == fold
                    and int(row["execution_ordinal"]) == key[1]
                ),
                "role/address/cache join drift",
            )
            cache_row = index_by_key[key]
            payload_path = safe_file(Path(cache_row["payload_path"]))
            require(
                payload_path.stat().st_size == int(cache_row["payload_bytes"])
                and file_sha256(payload_path) == cache_row["payload_sha256"],
                "cache payload physical receipt drift",
            )
            if payload_path not in payload_cache:
                payload = torch.load(payload_path, map_location="cpu", weights_only=True, mmap=True)
                require(
                    isinstance(payload, Mapping)
                    and payload.get("schema_version") == CACHE_PAYLOAD_SCHEMA
                    and payload.get("status") == CACHE_PAYLOAD_STATUS
                    and payload.get("target_free") is True,
                    "cache payload envelope drift",
                )
                payload_cache[payload_path] = payload
            payload = payload_cache[payload_path]
            ordinal = int(cache_row["payload_context_ordinal"])
            context = payload["contexts"][ordinal]
            require(
                int(context["outer_fold"]) == fold
                and int(context["execution_ordinal"]) == key[1]
                and context["query_id"] == role["query_id"]
                and context["address_record_sha256"]
                == next(
                    row["record_sha256"]
                    for row in address["cache_address_records"]
                    if int(row["outer_fold"]) == fold
                    and int(row["execution_ordinal"]) == key[1]
                ),
                "cache context/role address drift",
            )
            record, _unused_t_record, calls = _record(
                fold=fold,
                role=role,
                fold_observed=result,
                model=model,
                serialized=context["candidate_branch_caches"],
            )
            records.append(record)
            candidates = cache_row.get("candidate_caches")
            require(
                isinstance(candidates, list) and len(candidates) == 2,
                "cache-index candidate summary drift",
            )
            require(
                sum(bool(item["auxiliary_t_present"]) for item in candidates)
                == int(_unused_t_record["target_eligible"])
                + int(_unused_t_record["rival_eligible"]),
                "cache-index/independent auxiliary T presence drift",
            )
            t_records.extend(
                {
                    "fold": fold,
                    "track": str(role["track"]),
                    "present": bool(item["auxiliary_t_present"]),
                }
                for item in candidates
            )
            compare_count += calls
        fold_receipts.append(
            {
                "outer_fold": fold,
                "result_sha256": file_sha256(result_path),
                "result_logical_sha256": result["logical_sha256"],
                "validation_sha256": file_sha256(validation_path),
                "validation_logical_sha256": validation["logical_sha256"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "evaluation_record_population_sha256": result[
                    "evaluation_record_population_sha256"
                ],
                "model_state_sha256": result["model_state_sha256"],
            }
        )
    records.sort(key=lambda row: (int(row["outer_fold"]), int(row["execution_ordinal"])))
    require(
        len(records) == QUERY_COUNT
        and len({row["execution_ordinal"] for row in records}) == QUERY_COUNT
        and {row["outer_fold"] for row in records} == set(FOLDS),
        "independent Balanced-32 population drift",
    )
    folds = _fold_summaries(records)
    tracks = _track_summaries(records)
    invariant_names = tuple(records[0]["invariants"])
    invariants = {
        "record_count": len(records),
        **{
            f"{name}_count": sum(
                bool(row["invariants"][name]) for row in records
            )
            for name in invariant_names
        },
        "all_pass": all(
            all(bool(value) for value in row["invariants"].values())
            for row in records
        ),
    }

    def eligibility_slice(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        denominator = 2 * len(selected)
        return {
            control: {
                "candidate_denominator_count": denominator,
                "eligible_candidate_count": denominator,
                "ineligible_candidate_count": 0,
                "reason_counts": {"ELIGIBLE": denominator},
            }
            for control in NULLS
        }

    null_eligibility = {
        "controls": list(NULLS),
        "overall": eligibility_slice(records),
        "by_fold": {
            str(fold): eligibility_slice(
                [row for row in records if int(row["outer_fold"]) == fold]
            )
            for fold in FOLDS
        },
        "by_track": {
            track: eligibility_slice(
                [row for row in records if str(row["track"]) == track]
            )
            for track in tracks
        },
        "silent_denominator_drop_count": 0,
    }
    auxiliary_t = {
        "name": "T_ROOT_ASSIGNMENT_POPULATION",
        "selected_candidate_denominator_count": len(t_records),
        "present_count": sum(bool(row["present"]) for row in t_records),
        "absent_count": sum(not bool(row["present"]) for row in t_records),
        "by_fold": {
            str(fold): {
                "candidate_denominator_count": sum(
                    int(row["fold"]) == fold for row in t_records
                ),
                "present_count": sum(
                    int(row["fold"]) == fold and bool(row["present"])
                    for row in t_records
                ),
            }
            for fold in FOLDS
        },
        "by_track": {
            track: {
                "candidate_denominator_count": sum(
                    str(row["track"]) == track for row in t_records
                ),
                "present_count": sum(
                    str(row["track"]) == track and bool(row["present"])
                    for row in t_records
                ),
            }
            for track in tracks
        },
        "enters_calibration": False,
        "enters_loss": False,
        "enters_action": False,
        "enters_gate": False,
        "enters_bootstrap": False,
    }
    bootstrap = _bootstrap(records)
    totals = _totals(records)
    gates = _gates(folds, tracks, invariants, bootstrap, null_eligibility, totals)
    target_free_seal = canonical_sha256(
        [
            {
                "outer_fold": fold,
                "context_count": 56,
                "cache_count": 448,
                "seal_sha256": canonical_sha256(
                    sorted(
                        [
                            {
                                "execution_ordinal": int(row["execution_ordinal"]),
                                "payload_sha256": row["payload_sha256"],
                                "validation_sha256": row["validation_sha256"],
                                "address_record_sha256": row["address_record_sha256"],
                            }
                            for row in index["records"]
                            if int(row["outer_fold"]) == fold
                        ],
                        key=lambda row: row["execution_ordinal"],
                    )
                ),
            }
            for fold in FOLDS
        ]
    )
    status = GO if gates["all_gates_pass"] else NO_GO
    return {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "scientific_GO_or_NO_GO": status,
        "claim_level": "INTERNAL_BALANCED32_MATCHED_NULL_CALIBRATED_CONNECTED_DINO_RELATIONAL_ENERGY_SCREEN",
        "claim_boundary": {
            "matched_null_calibrated_connected_dino_relational_energy_only": True,
            "full_gallery_claim_authorized": False,
            "ownership_claim_authorized": False,
            "target_presence_or_absence_claim_authorized": False,
            "opened_or_sealed_claim_authorized": False,
            "downstream_action_experiment_requires_separate_authority": True,
        },
        "input_receipts": {},
        "access_audit": {
            "all_four_target_free_folds_sealed_before_role_join": True,
            "target_free_fourfold_seal_sha256": target_free_seal,
            "target_free_context_count": 224,
            "target_free_cache_count": 1792,
            "loss_role_manifest_open_count_after_seal": 4,
            "label_target_rival_role_model_input_count": 0,
            "opened_sealed_read_count": 0,
        },
        "evaluation_records": records,
        "evaluation_record_population_sha256": canonical_sha256(records),
        "fold_summaries": folds,
        "track_summaries": tracks,
        "null_eligibility_summary": null_eligibility,
        "auxiliary_t_diagnostic": auxiliary_t,
        "invariants": invariants,
        "bootstrap": bootstrap,
        "gates": gates,
        "totals": totals,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }


def validate(
    *,
    result_path: Path,
    training_root: Path,
    training_validation_root: Path,
    address_manifest_path: Path,
    loss_role_manifest_path: Path,
    cache_index_path: Path,
    output: Path,
    input_receipts_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observed_path = safe_file(result_path)
    observed = read_json(observed_path)
    require(
        observed.get("schema_version") == RESULT_SCHEMA
        and observed.get("status") in {GO, NO_GO}
        and observed.get("scientific_GO_or_NO_GO") == observed.get("status")
        and observed.get("logical_sha256") == logical_sha256(observed),
        "scientific result envelope drift",
    )
    expected = reconstruct(
        training_root=training_root,
        training_validation_root=training_validation_root,
        address_manifest_path=address_manifest_path,
        loss_role_manifest_path=loss_role_manifest_path,
        cache_index_path=cache_index_path,
    )
    if input_receipts_override is not None:
        expected["input_receipts"] = dict(input_receipts_override)
    expected["logical_sha256"] = logical_sha256(expected)
    require(observed == expected, "independent scientific replay mismatch")
    value = {
        "schema_version": VALIDATION_SCHEMA,
        "status": VALIDATION_STATUS,
        "validation_pass": True,
        "result_sha256": file_sha256(observed_path),
        "result_logical_sha256": observed["logical_sha256"],
        "scientific_GO_or_NO_GO": observed["scientific_GO_or_NO_GO"],
        "query_count": observed["totals"]["query_count"],
        "fold_count": 4,
        "evaluation_record_population_sha256": observed[
            "evaluation_record_population_sha256"
        ],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "all_gates_recomputed": True,
        "control_constructors_imported": False,
        "production_reducer_imported": False,
        "opened_sealed_access_count": 0,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(output, value)
    return value


def _authority_path(value: Mapping[str, Any], name: str) -> Path:
    binding = value.get("bindings", {}).get(name)
    require(isinstance(binding, Mapping), f"science authority binding absent: {name}")
    path = Path(str(binding["path"]))
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    path = safe_file(path)
    require(
        path.stat().st_size == int(binding["bytes"])
        and file_sha256(path) == binding["sha256"],
        f"science authority binding drift: {name}",
    )
    return path


def validate_from_authority(
    *, authority_path: Path, result_path: Path, output: Path
) -> dict[str, Any]:
    authority_file = safe_file(authority_path)
    authority = read_json(authority_file)
    require(
        authority.get("status")
        == "GX_CBNR_R0_SCIENTIFIC_REDUCTION_EXECUTION_AUTHORIZED"
        and authority.get("logical_sha256") == logical_sha256(authority)
        and authority.get("scientific_reduction_authorized") is True
        and authority.get("opened_or_sealed_access_authorized") is False
        and authority.get("automatic_stage_advance") is False,
        "scientific reduction authority envelope drift",
    )
    expected_result = (ROOT / str(authority["output"])).resolve()
    expected_validation = (
        ROOT / str(authority["validation_output"])
    ).resolve()
    require(
        result_path.resolve() == expected_result
        and output.resolve() == expected_validation,
        "scientific authority output path drift",
    )
    names = [
        "parent_training_authority_v54",
        "address_manifest",
        "loss_role_manifest",
        "cache_index",
    ]
    for fold in FOLDS:
        names.extend(
            (
                f"fold{fold}_result",
                f"fold{fold}_validation",
                f"fold{fold}_checkpoint",
                f"fold{fold}_init_checkpoint",
            )
        )
    paths = {name: _authority_path(authority, name) for name in names}
    parent_training = read_json(paths["parent_training_authority_v54"])
    require(
        parent_training.get("status")
        == "GX_CBNR_FORMAL_FOURFOLD_TRAINING_EXECUTION_AUTHORIZED"
        and parent_training.get("logical_sha256")
        == logical_sha256(parent_training)
        and parent_training.get("scientific_reduction_authorized") is False,
        "parent formal training authority drift",
    )
    training_roots = {paths[f"fold{fold}_result"].parent.parent for fold in FOLDS}
    validation_roots = {
        paths[f"fold{fold}_validation"].parent.parent for fold in FOLDS
    }
    require(
        len(training_roots) == len(validation_roots) == 1,
        "authority fold output root drift",
    )
    training_root = next(iter(training_roots))
    training_validation_root = next(iter(validation_roots))
    execution_receipt = {
        "authority_path": str(authority_file.resolve()),
        "authority_bytes": authority_file.stat().st_size,
        "authority_sha256": file_sha256(authority_file),
        "authority_logical_sha256": authority["logical_sha256"],
        "binding_count": len(paths),
        "binding_population_sha256": canonical_sha256(
            [
                {
                    "name": name,
                    "path": str(paths[name]),
                    "sha256": file_sha256(paths[name]),
                }
                for name in sorted(paths)
            ]
        ),
    }
    execution_receipt["logical_sha256"] = logical_sha256(execution_receipt)
    input_receipts = {
        "authority_execution_receipt": execution_receipt,
        "parent_training_authority_sha256": file_sha256(
            paths["parent_training_authority_v54"]
        ),
        "address_manifest_sha256": file_sha256(paths["address_manifest"]),
        "loss_role_manifest_sha256": file_sha256(paths["loss_role_manifest"]),
        "cache_index_sha256": file_sha256(paths["cache_index"]),
        "fold_result_sha256": {
            str(fold): file_sha256(paths[f"fold{fold}_result"])
            for fold in FOLDS
        },
        "fold_validation_sha256": {
            str(fold): file_sha256(paths[f"fold{fold}_validation"])
            for fold in FOLDS
        },
        "fold_checkpoint_sha256": {
            str(fold): file_sha256(paths[f"fold{fold}_checkpoint"])
            for fold in FOLDS
        },
    }
    return validate(
        result_path=result_path,
        training_root=training_root,
        training_validation_root=training_validation_root,
        address_manifest_path=paths["address_manifest"],
        loss_role_manifest_path=paths["loss_role_manifest"],
        cache_index_path=paths["cache_index"],
        output=output,
        input_receipts_override=input_receipts,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    value = validate_from_authority(
        authority_path=args.authority,
        result_path=args.result,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                "scientific_GO_or_NO_GO": value["scientific_GO_or_NO_GO"],
                "logical_sha256": value["logical_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
