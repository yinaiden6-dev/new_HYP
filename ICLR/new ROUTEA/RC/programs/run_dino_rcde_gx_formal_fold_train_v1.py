#!/usr/bin/env python3
"""Formal one-fold GX head trainer over independently validated caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.dino_rcde_gx_cbnr_v3 import gx_action
from rc_aslo_xf.dino_rcde_gx_formal_head_train_v1 import (
    EXPECTED_BRANCHES,
    RAW_CORRECT,
    RAW_WRONG,
    STATE_SCHEMA_VERSION,
    FormalHeadEpisodeV1,
    FormalHeadTrainingStateV1,
    UpdateScheduleRecordV1,
    canonical_sha256,
    episode_loss,
    export_training_state,
    gradient_coverage_receipt,
    resume_training,
    run_updates,
    start_training,
    tensor_sha256,
    tree_sha256,
)
from rc_aslo_xf.dino_rcde_gx_relational_head_cache_v1 import (
    AUXILIARY_BRANCHES,
    TRAINING_BRANCHES,
    make_candidate_branch_relational_cache,
    make_direction_relational_cache,
    make_fixed_denominator_cache,
    make_query_pullback,
    make_root_relational_payload,
)
from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2


SCHEMA_VERSION = "rc_dino_rcde_gx_formal_fold_train_v1_20260824"
RESULT_STATUS = "GX_CBNR_FORMAL_FOLD_TRAIN_COMPLETE"
INTERRUPTED_STATUS = "GX_CBNR_FORMAL_FOLD_TRAIN_INTERRUPTED"
ABORT = "GX_CBNR_FORMAL_FOLD_TRAIN_ABORT"
CHECKPOINT_SCHEMA = "rc_dino_rcde_gx_formal_fold_checkpoint_v1_20260824"
CACHE_INDEX_SCHEMA = "rc_dino_rcde_gx_relational_cache_index_v1_20260824"
CACHE_INDEX_STATUS = "GX_CBNR_RELATIONAL_CACHE_INDEX_READY"
CACHE_INDEX_REQUIRED_RECORD_FIELDS = {
    "outer_fold",
    "execution_ordinal",
    "query_id",
    "address_record_sha256",
    "payload_path",
    "payload_bytes",
    "payload_sha256",
    "payload_context_ordinal",
    "validation_path",
    "validation_bytes",
    "validation_sha256",
    "record_sha256",
}
CACHE_PAYLOAD_SCHEMA = "rc_dino_rcde_gx_relational_cache_shard_v1_20260824"
CACHE_PAYLOAD_STATUS = "GX_CBNR_RELATIONAL_CACHE_SHARD_READY"
CACHE_VALIDATION_STATUS = "GX_CBNR_RELATIONAL_CACHE_SHARD_VALIDATION_PASS"
ADDRESS_SCHEMA = "rc_dino_rcde_gx_cbnr_r0_address_manifest_v2_20260824"
LOSS_ROLE_SCHEMA = "rc_dino_rcde_gx_cbnr_r0_loss_role_manifest_v2_20260824"
MANIFEST_STATUS = "GX_CBNR_R0_ELIGIBILITY_AWARE_MANIFEST_READY"
FOLD_CONTEXT_COUNT = 56
TRAIN_CONTEXT_COUNT = 48
EVALUATION_CONTEXT_COUNT = 8
TRAINING_RECORD_COUNT = 128
OPENED_SEALED_COMPONENTS = {"opened", "sealed"}


class FoldTrainError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise FoldTrainError(message)


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_input_path(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    require(
        resolved.is_file()
        and not resolved.is_symlink()
        and not OPENED_SEALED_COMPONENTS.intersection(
            component.lower() for component in resolved.parts
        ),
        f"unsafe/opened/sealed input path: {resolved}",
    )
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    path = safe_input_path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FoldTrainError(f"cannot read JSON: {path}") from error
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def validate_execution_authority(
    authority_path: Path, outer_fold: int, output_dir: Path
) -> dict[str, Any]:
    authority_file = safe_input_path(authority_path)
    authority = read_json(authority_file)
    require(
        authority.get("status")
        == "GX_CBNR_FORMAL_FOURFOLD_TRAINING_EXECUTION_AUTHORIZED"
        and authority.get("logical_sha256") == logical_sha256(authority)
        and authority.get("formal_r0_authorized") is True
        and authority.get("fourfold_training_authorized") is True
        and authority.get("opened_or_sealed_access_authorized") is False,
        "formal execution authority envelope drift",
    )
    expected = authority.get("fold_outputs", {}).get(str(outer_fold))
    authorized_output_dirs: set[Path] = set()
    if isinstance(expected, Mapping):
        authorized_output_dirs.add(
            (ROOT / str(expected["output_dir"])).resolve(strict=False)
        )
        segments = expected.get("segment_outputs", {})
        if isinstance(segments, Mapping):
            authorized_output_dirs.update(
                (ROOT / str(row["output_dir"])).resolve(strict=False)
                for row in segments.values()
                if isinstance(row, Mapping) and "output_dir" in row
            )
    require(
        isinstance(expected, Mapping)
        and output_dir.resolve(strict=False) in authorized_output_dirs,
        "formal fold output authorization drift",
    )
    bindings = authority.get("bindings")
    require(isinstance(bindings, Mapping), "formal authority bindings absent")
    checked = []
    for name, binding in sorted(bindings.items()):
        if name == "runtime_import_closure":
            continue
        require(isinstance(binding, Mapping), f"authority binding malformed: {name}")
        path = Path(str(binding["path"]))
        path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == int(binding["bytes"])
            and file_sha256(path) == binding["sha256"],
            f"authority binding drift: {name}",
        )
        checked.append({"name": name, "sha256": binding["sha256"]})
    closure = bindings.get("runtime_import_closure")
    require(
        isinstance(closure, Mapping)
        and isinstance(closure.get("rows"), list)
        and len(closure["rows"]) == int(closure["count"])
        and closure.get("logical_sha256")
        == logical_sha256({"rows": closure["rows"]}),
        "formal authority runtime closure envelope drift",
    )
    for index, binding in enumerate(closure["rows"]):
        path = Path(str(binding["path"]))
        path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == int(binding["bytes"])
            and file_sha256(path) == binding["sha256"],
            f"formal runtime closure drift: {index}",
        )
    receipt = {
        "authority_sha256": file_sha256(authority_file),
        "authority_logical_sha256": authority["logical_sha256"],
        "simple_binding_count": len(checked),
        "simple_binding_population_sha256": canonical_sha256(checked),
        "runtime_import_closure_count": len(closure["rows"]),
        "runtime_import_closure_logical_sha256": closure["logical_sha256"],
        "expected_fold_outputs": dict(expected),
    }
    receipt["logical_sha256"] = logical_sha256(receipt)
    return receipt


def decode_binary64(value: object) -> float:
    require(
        isinstance(value, str)
        and len(value) == 16
        and all(character in "0123456789abcdef" for character in value),
        "RAW binary64 field drift",
    )
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    require(math.isfinite(result), "RAW binary64 is nonfinite")
    return result


def deserialize_pullback(value: object):
    if value is None:
        return None
    require(isinstance(value, Mapping), "query pullback payload drift")
    pullback = make_query_pullback(
        value["source_original_indices"], value["destination_decode_indices"]
    )
    require(
        pullback.payload() == value,
        "query pullback serialized receipt drift",
    )
    return pullback


def deserialize_candidate_cache(value: Mapping[str, Any]):
    require(
        str(value.get("branch")) in {*TRAINING_BRANCHES, *AUXILIARY_BRANCHES},
        "serialized candidate branch drift",
    )
    directions = {}
    for direction in ("a_to_b", "b_to_a"):
        observed_direction = value["directions"][direction]
        raw_denominator = observed_direction["denominator"]
        status = {
            int(key): str(item)
            for key, item in raw_denominator["root_status_by_ordinal"].items()
        }
        masks = {
            int(key): item
            for key, item in raw_denominator[
                "original_query_masks_by_root"
            ].items()
        }
        denominator = make_fixed_denominator_cache(
            outer_fold=int(raw_denominator["outer_fold"]),
            execution_ordinal=int(raw_denominator["execution_ordinal"]),
            query_id=str(raw_denominator["query_id"]),
            candidate_key=str(raw_denominator["candidate_key"]),
            candidate_physical_row=int(
                raw_denominator["candidate_physical_row"]
            ),
            branch=str(raw_denominator["branch"]),
            direction=str(raw_denominator["direction"]),
            query_grid_shape=tuple(raw_denominator["query_grid_shape"]),
            raw_p_v2_record_sha256=str(
                raw_denominator["raw_p_v2_record_sha256"]
            ),
            core_p_v2_record_sha256=str(
                raw_denominator["core_p_v2_record_sha256"]
            ),
            complete_root_table_sha256=str(
                raw_denominator["complete_root_table_sha256"]
            ),
            selected_root_ordinals=tuple(
                map(int, raw_denominator["selected_root_ordinals"])
            ),
            root_status_by_ordinal=status,
            original_query_masks_by_root=masks,
        )
        require(
            denominator.logical_sha256 == raw_denominator["logical_sha256"]
            and torch.equal(denominator.coverage, raw_denominator["coverage"])
            and torch.equal(
                denominator.query_union, raw_denominator["query_union"]
            ),
            "serialized fixed denominator drift",
        )
        roots = []
        for raw in observed_direction["root_payloads"]:
            root = make_root_relational_payload(
                outer_fold=int(raw["outer_fold"]),
                execution_ordinal=int(raw["execution_ordinal"]),
                query_id=str(raw["query_id"]),
                query_source_image_sha256=str(
                    raw["query_source_image_sha256"]
                ),
                candidate_key=str(raw["candidate_key"]),
                candidate_physical_row=int(raw["candidate_physical_row"]),
                candidate_content_binding_sha256=str(
                    raw["candidate_content_binding_sha256"]
                ),
                branch=str(raw["branch"]),
                direction=str(raw["direction"]),
                root_ordinal=int(raw["root_ordinal"]),
                source_reference_root_ordinal=int(
                    raw["source_reference_root_ordinal"]
                ),
                raw_p_v2_record_sha256=str(
                    raw["raw_p_v2_record_sha256"]
                ),
                core_p_v2_record_sha256=str(
                    raw["core_p_v2_record_sha256"]
                ),
                complete_root_table_sha256=str(
                    raw["complete_root_table_sha256"]
                ),
                branch_control_receipt_sha256=str(
                    raw["branch_control_receipt_sha256"]
                ),
                direction_control_receipt_sha256=str(
                    raw["direction_control_receipt_sha256"]
                ),
                root_control_receipt_sha256=str(
                    raw["root_control_receipt_sha256"]
                ),
                query_grid_shape=tuple(raw["query_grid_shape"]),
                reference_grid_shape=tuple(raw["reference_grid_shape"]),
                original_reference_grid_shape=tuple(
                    raw["original_reference_grid_shape"]
                ),
                relational=raw["relational"],
                decode_query_mask=raw["decode_query_mask"],
                decode_reference_mask=raw["decode_reference_mask"],
                original_root_query_mask=raw[
                    "original_root_query_mask"
                ],
                original_root_reference_mask=raw[
                    "original_root_reference_mask"
                ],
                query_pullback=deserialize_pullback(raw.get("query_pullback")),
            )
            require(
                root.logical_sha256 == raw["logical_sha256"]
                and root.address_sha256 == raw["address_sha256"],
                "serialized relational root drift",
            )
            roots.append(root)
        direction_cache = make_direction_relational_cache(denominator, roots)
        require(
            direction_cache.direction_cache_sha256
            == observed_direction["direction_cache_sha256"],
            "serialized direction cache SHA drift",
        )
        directions[direction] = direction_cache
    cache = make_candidate_branch_relational_cache(
        branch_control_receipt_sha256=str(
            value["branch_control_receipt_sha256"]
        ),
        directions=directions,
    )
    require(
        cache.candidate_cache_sha256 == value["candidate_cache_sha256"],
        "serialized candidate cache SHA drift",
    )
    cache.validate_bytes()
    return cache


def default_model_loader(path: Path, outer_fold: int):
    path = safe_input_path(path)
    import run_dino_rcde_h0_u4_precompute_chunk_v1 as CHECKPOINT_INPUT

    model, validated = CHECKPOINT_INPUT.load_model(
        path, arm="INIT", fold=outer_fold, device=torch.device("cpu")
    )
    receipt = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "outer_fold": outer_fold,
        "model_state_sha256": tree_sha256(model.state_dict()),
        "validated_checkpoint_receipt": dict(validated),
    }
    return model, receipt


def validate_cache_index(index: Mapping[str, Any]) -> None:
    require(
        index.get("schema_version") == CACHE_INDEX_SCHEMA
        and index.get("status") == CACHE_INDEX_STATUS
        and index.get("target_free") is True
        and index.get("logical_sha256") == logical_sha256(index)
        and isinstance(index.get("records"), list)
        and index.get("record_count") == len(index["records"]),
        "relational cache-index envelope drift",
    )
    for record in index["records"]:
        require(
            CACHE_INDEX_REQUIRED_RECORD_FIELDS.issubset(record)
            and
            record.get("record_sha256")
            == canonical_sha256(
                {
                    key: item
                    for key, item in record.items()
                    if key != "record_sha256"
                }
            ),
            "relational cache-index record SHA drift",
        )


def load_target_free_fold(
    *,
    address_manifest_path: Path,
    cache_index_path: Path,
    outer_fold: int,
    expected_init_checkpoint_sha256: str,
) -> dict[str, Any]:
    address_path = safe_input_path(address_manifest_path)
    cache_index_file = safe_input_path(cache_index_path)
    address = read_json(address_path)
    index = read_json(cache_index_file)
    require(
        address.get("schema_version") == ADDRESS_SCHEMA
        and address.get("status") == MANIFEST_STATUS
        and address.get("logical_sha256") == logical_sha256(address)
        and address.get("label_target_rival_raw_group_track_field_count") == 0
        and address.get("loss_role_manifest_path_or_sha_read_count") == 0,
        "address manifest target-free envelope drift",
    )
    validate_cache_index(index)
    require(
        index.get("address_manifest_sha256") == file_sha256(address_path)
        and index.get("address_manifest_logical_sha256")
        == address["logical_sha256"],
        "cache-index/address manifest binding drift",
    )
    address_contexts = {
        int(row["execution_ordinal"]): row
        for row in address["cache_address_records"]
        if int(row["outer_fold"]) == outer_fold
    }
    records = [
        row for row in index["records"] if int(row["outer_fold"]) == outer_fold
    ]
    require(
        len(address_contexts) == len(records) == FOLD_CONTEXT_COUNT
        and len({int(row["execution_ordinal"]) for row in records})
        == FOLD_CONTEXT_COUNT,
        "fold cache-index must contain exactly 56 contexts",
    )
    payload_cache: dict[str, Mapping[str, Any]] = {}
    validation_cache: dict[str, Mapping[str, Any]] = {}
    contexts = {}
    cache_file_receipts = []
    for record in records:
        execution = int(record["execution_ordinal"])
        require(execution in address_contexts, "cache-index execution outside address manifest")
        address_row = address_contexts[execution]
        require(
            record["address_record_sha256"] == address_row["record_sha256"],
            "cache-index address-record binding drift",
        )
        payload_path = safe_input_path(Path(record["payload_path"]))
        validation_path = safe_input_path(Path(record["validation_path"]))
        require(
            payload_path.stat().st_size == record["payload_bytes"]
            and file_sha256(payload_path) == record["payload_sha256"]
            and validation_path.stat().st_size == record["validation_bytes"]
            and file_sha256(validation_path) == record["validation_sha256"],
            "cache-index payload/validation physical binding drift",
        )
        payload_key = str(payload_path)
        if payload_key not in payload_cache:
            payload = torch.load(
                payload_path, map_location="cpu", weights_only=True, mmap=True
            )
            require(
                isinstance(payload, Mapping)
                and payload.get("schema_version") == CACHE_PAYLOAD_SCHEMA
                and payload.get("status") == CACHE_PAYLOAD_STATUS
                and payload.get("target_free") is True
                and payload.get("outer_fold") == outer_fold
                and payload.get("checkpoint_sha256")
                == [expected_init_checkpoint_sha256]
                and payload.get("logical_sha256")
                == canonical_sha256(
                    {
                        key: item
                        for key, item in payload.items()
                        if key not in {"contexts", "logical_sha256"}
                    }
                ),
                "relational payload envelope drift",
            )
            payload_cache[payload_key] = payload
        payload = payload_cache[payload_key]
        validation_key = str(validation_path)
        if validation_key not in validation_cache:
            validation = read_json(validation_path)
            require(
                validation.get("status") == CACHE_VALIDATION_STATUS
                and validation.get("validation_pass") is True
                and validation.get("logical_sha256")
                == logical_sha256(validation)
                and validation.get("payload_sha256")
                == record["payload_sha256"]
                and validation.get("address_manifest_sha256")
                == file_sha256(address_path),
                "relational cache independent validation drift",
            )
            checkpoint_receipt = validation.get("checkpoint_receipt")
            if checkpoint_receipt is not None:
                require(
                    isinstance(checkpoint_receipt, Mapping)
                    and checkpoint_receipt.get("checkpoint_sha256")
                    == expected_init_checkpoint_sha256,
                    "cache validation/init checkpoint binding drift",
                )
            validation_cache[validation_key] = validation
        ordinal = int(record["payload_context_ordinal"])
        require(0 <= ordinal < len(payload["contexts"]), "payload context ordinal drift")
        context = payload["contexts"][ordinal]
        require(
            int(context["outer_fold"]) == outer_fold
            and int(context["execution_ordinal"]) == execution
            and context["address_record_sha256"] == address_row["record_sha256"],
            "payload context address drift",
        )
        candidate_keys = [
            row["candidate_key"] for row in address_row["candidate_cache_addresses"]
        ]
        caches: dict[str, dict[str, Any]] = {
            key: {} for key in candidate_keys
        }
        for serialized in context["candidate_branch_caches"]:
            key = str(serialized["candidate_key"])
            branch = str(serialized["branch"])
            require(key in caches, "payload candidate outside address pair")
            if branch in EXPECTED_BRANCHES:
                require(branch not in caches[key], "duplicate mandatory cache")
                cache = deserialize_candidate_cache(serialized)
                require(
                    cache.outer_fold == outer_fold
                    and cache.execution_ordinal == execution
                    and cache.query_id == context["query_id"]
                    and cache.candidate_key == key,
                    "payload candidate cache context address drift",
                )
                caches[key][branch] = cache
            else:
                require(
                    branch in AUXILIARY_BRANCHES,
                    "unknown payload cache branch",
                )
        require(
            all(set(value) == set(EXPECTED_BRANCHES) for value in caches.values()),
            "context missing REAL/C/P/N relational caches",
        )
        contexts[execution] = {
            "address": address_row,
            "caches_by_candidate": caches,
            "payload_sha256": record["payload_sha256"],
            "validation_sha256": record["validation_sha256"],
        }
        cache_file_receipts.append(
            {
                "execution_ordinal": execution,
                "payload_sha256": record["payload_sha256"],
                "validation_sha256": record["validation_sha256"],
                "address_record_sha256": record["address_record_sha256"],
            }
        )
    return {
        "address_manifest": address,
        "cache_index": index,
        "contexts": contexts,
        "cache_file_receipts": sorted(
            cache_file_receipts, key=lambda row: row["execution_ordinal"]
        ),
        "target_free_context_count": len(contexts),
        "target_free_cache_count": len(contexts)
        * 2
        * len(EXPECTED_BRANCHES),
        "target_free_seal_sha256": canonical_sha256(
            sorted(cache_file_receipts, key=lambda row: row["execution_ordinal"])
        ),
    }


def episode_id(outer_fold: int, execution: int, source_sha: str) -> str:
    return canonical_sha256(["GX_FORMAL_EPISODE_V1", outer_fold, execution, source_sha])


def make_episode(
    role: Mapping[str, Any], context: Mapping[str, Any], outer_fold: int
) -> FormalHeadEpisodeV1:
    execution = int(role["execution_ordinal"])
    target_score = decode_binary64(role["target_raw_score_bits"])
    rival_score = decode_binary64(role["rival_raw_score_bits"])
    margin = target_score - rival_score
    raw_correct = bool(role["raw_correct"])
    require(raw_correct is (margin > 0.0), "loss-role RAW correctness drift")
    target_key = str(role["target_candidate_key"])
    rival_key = str(role["rival_candidate_key"])
    caches = context["caches_by_candidate"]
    require(
        target_key in caches
        and rival_key in caches
        and target_key != rival_key,
        "loss-role target/rival outside anonymous cache pair",
    )
    return FormalHeadEpisodeV1(
        episode_id=episode_id(
            outer_fold, execution, str(role["source_i0_record_sha256"])
        ),
        stratum=RAW_CORRECT if raw_correct else RAW_WRONG,
        raw_target_margin=torch.tensor(margin, dtype=torch.float32),
        target_candidate_key=target_key,
        rival_candidate_key=rival_key,
        target_branches=caches[target_key],
        rival_branches=caches[rival_key],
    )


def open_loss_roles_and_build_fold(
    *,
    loss_role_manifest_path: Path,
    target_free: Mapping[str, Any],
    outer_fold: int,
) -> dict[str, Any]:
    roles_path = safe_input_path(loss_role_manifest_path)
    roles = read_json(roles_path)
    address = target_free["address_manifest"]
    require(
        roles.get("schema_version") == LOSS_ROLE_SCHEMA
        and roles.get("status") == MANIFEST_STATUS
        and roles.get("logical_sha256") == logical_sha256(roles)
        and roles.get("address_manifest_logical_sha256")
        == address["logical_sha256"]
        and roles.get("join_after_target_free_eligibility_and_cache_address_seal")
        is True,
        "loss-role manifest envelope drift",
    )
    address_training = {
        int(row["update"]): row
        for row in address["training_records"]
        if int(row["outer_fold"]) == outer_fold
    }
    role_training = {
        int(row["update"]): row
        for row in roles["training_records"]
        if int(row["outer_fold"]) == outer_fold
    }
    address_eval = {
        int(row["execution_ordinal"]): row
        for row in address["evaluation_records"]
        if int(row["outer_fold"]) == outer_fold
    }
    role_eval = {
        int(row["execution_ordinal"]): row
        for row in roles["evaluation_records"]
        if int(row["outer_fold"]) == outer_fold
    }
    require(
        set(address_training) == set(role_training) == set(range(1, 129))
        and set(address_eval) == set(role_eval)
        and len(address_eval) == EVALUATION_CONTEXT_COUNT,
        "fold loss-role training/evaluation population drift",
    )
    contexts = target_free["contexts"]
    episodes = {}
    schedule_members = []
    for update in range(1, 129):
        addr = address_training[update]
        role = role_training[update]
        require(
            role["address_record_sha256"] == addr["record_sha256"]
            and role["raw_correct_count"] == role["raw_wrong_count"] == 1
            and len(addr["members"]) == len(role["members"]) == 2,
            "explicit fold update role/address drift",
        )
        by_execution = {
            int(row["execution_ordinal"]): row for row in addr["members"]
        }
        update_episodes = []
        for member in role["members"]:
            execution = int(member["execution_ordinal"])
            require(
                execution in by_execution
                and execution in contexts
                and by_execution[execution]["cache_address_record_sha256"]
                == contexts[execution]["address"]["record_sha256"],
                "loss role/cache address join drift",
            )
            candidate = make_episode(member, contexts[execution], outer_fold)
            previous = episodes.setdefault(candidate.episode_id, candidate)
            require(
                previous.logical_sha256 == candidate.logical_sha256,
                "repeated loss-role episode payload drift",
            )
            update_episodes.append(previous)
        correct = next(
            item for item in update_episodes if item.stratum == RAW_CORRECT
        )
        wrong = next(item for item in update_episodes if item.stratum == RAW_WRONG)
        schedule_members.append((correct, wrong))
    correct_pool = tuple(
        sorted(
            (item for item in episodes.values() if item.stratum == RAW_CORRECT),
            key=lambda item: item.episode_id,
        )
    )
    wrong_pool = tuple(
        sorted(
            (item for item in episodes.values() if item.stratum == RAW_WRONG),
            key=lambda item: item.episode_id,
        )
    )
    schedule = []
    for update, (correct, wrong) in enumerate(schedule_members, start=1):
        payload = {
            "update": update,
            "raw_correct_episode_id": correct.episode_id,
            "raw_wrong_episode_id": wrong.episode_id,
            "raw_correct_episode_sha256": correct.logical_sha256,
            "raw_wrong_episode_sha256": wrong.logical_sha256,
        }
        schedule.append(
            UpdateScheduleRecordV1(
                **payload, record_sha256=canonical_sha256(payload)
            )
        )
    evaluations = []
    for execution in sorted(role_eval):
        require(
            role_eval[execution]["address_record_sha256"]
            == address_eval[execution]["record_sha256"]
            and address_eval[execution]["cache_address_record_sha256"]
            == contexts[execution]["address"]["record_sha256"],
            "evaluation loss-role/cache address drift",
        )
        evaluations.append(
            (
                make_episode(role_eval[execution], contexts[execution], outer_fold),
                role_eval[execution],
            )
        )
    return {
        "loss_role_manifest": roles,
        "correct_pool": correct_pool,
        "wrong_pool": wrong_pool,
        "schedule": tuple(schedule),
        "evaluations": tuple(evaluations),
        "loss_role_manifest_sha256": file_sha256(roles_path),
    }


def evaluate_fold(model: torch.nn.Module, evaluations) -> list[dict[str, Any]]:
    records = []
    with torch.no_grad():
        for episode, role in evaluations:
            loss = episode_loss(model, episode)
            raw = episode.raw_target_margin.to(loss.target_z.device)
            if episode.stratum == RAW_CORRECT:
                action = gx_action(
                    raw_challenger_minus_winner=-raw,
                    challenger_z=loss.rival_z,
                    winner_z=loss.target_z,
                    all_nulls_eligible=True,
                )
            else:
                action = gx_action(
                    raw_challenger_minus_winner=raw,
                    challenger_z=loss.target_z,
                    winner_z=loss.rival_z,
                    all_nulls_eligible=True,
                )
            value = {
                "execution_ordinal": int(role["execution_ordinal"]),
                "query_id": role["query_id"],
                "episode_id": episode.episode_id,
                "raw_correct": episode.stratum == RAW_CORRECT,
                "raw_target_margin": float(raw.cpu()),
                "target_z": float(loss.target_z.cpu()),
                "rival_z": float(loss.rival_z.cpu()),
                "geometric_margin": float(loss.geometric_margin.cpu()),
                "final_margin": float(loss.final_margin.cpu()),
                "loss": float(loss.total.cpu()),
                "action": action.action,
                "relative_null_hold_not_target_absence": True,
                "target_absence_claimed": False,
                "track": role.get("track"),
                "group_sha256": role.get("group_sha256"),
            }
            value["record_sha256"] = canonical_sha256(value)
            records.append(value)
    return records


def formal_state_payload(state: FormalHeadTrainingStateV1) -> dict[str, Any]:
    return {
        "schema_version": state.schema_version,
        "completed_updates": state.completed_updates,
        "schedule_sha256": state.schedule_sha256,
        "parameter_contract": dict(state.parameter_contract),
        "model_state_dict": dict(state.model_state_dict),
        "optimizer_state_dict": dict(state.optimizer_state_dict),
        "trace": [dict(row) for row in state.trace],
        "torch_cpu_rng_state": state.torch_cpu_rng_state,
        "model_state_sha256": state.model_state_sha256,
        "optimizer_state_sha256": state.optimizer_state_sha256,
        "trace_sha256": state.trace_sha256,
        "logical_sha256": state.logical_sha256,
    }


def formal_state_from_payload(value: Mapping[str, Any]) -> FormalHeadTrainingStateV1:
    return FormalHeadTrainingStateV1(
        schema_version=str(value["schema_version"]),
        completed_updates=int(value["completed_updates"]),
        schedule_sha256=str(value["schedule_sha256"]),
        parameter_contract=dict(value["parameter_contract"]),
        model_state_dict=dict(value["model_state_dict"]),
        optimizer_state_dict=dict(value["optimizer_state_dict"]),
        trace=tuple(value["trace"]),
        torch_cpu_rng_state=value["torch_cpu_rng_state"],
        model_state_sha256=str(value["model_state_sha256"]),
        optimizer_state_sha256=str(value["optimizer_state_sha256"]),
        trace_sha256=str(value["trace_sha256"]),
        logical_sha256=str(value["logical_sha256"]),
    )


def atomic_torch(path: Path, value: object) -> None:
    require(not path.exists(), f"immutable torch output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"immutable JSON output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def run_fold(
    *,
    address_manifest_path: Path,
    loss_role_manifest_path: Path,
    cache_index_path: Path,
    outer_fold: int,
    init_checkpoint_path: Path,
    output_dir: Path,
    stop_after_update: int = 128,
    resume_checkpoint_path: Path | None = None,
    authority_execution_receipt: Mapping[str, Any] | None = None,
    model_loader: Callable[[Path, int], tuple[torch.nn.Module, Mapping[str, Any]]]
    = default_model_loader,
) -> dict[str, Any]:
    require(outer_fold in (1, 2, 3, 4), "outer fold drift")
    if authority_execution_receipt is not None:
        require(
            authority_execution_receipt.get("authority_sha256")
            and authority_execution_receipt.get("expected_fold_outputs"),
            "formal authority execution receipt drift",
        )
    require(not output_dir.exists(), "formal fold output directory exists")
    init_checkpoint = safe_input_path(init_checkpoint_path)
    init_checkpoint_sha256 = file_sha256(init_checkpoint)
    # Target-free front half: loss-role path is neither opened nor passed here.
    target_free = load_target_free_fold(
        address_manifest_path=address_manifest_path,
        cache_index_path=cache_index_path,
        outer_fold=outer_fold,
        expected_init_checkpoint_sha256=init_checkpoint_sha256,
    )
    model, init_receipt = model_loader(init_checkpoint, outer_fold)
    require(isinstance(model, torch.nn.Module), "model loader returned no module")
    require(
        init_receipt.get("sha256") == init_checkpoint_sha256,
        "model loader/init checkpoint SHA drift",
    )
    target_free_seal = canonical_sha256(
        {
            "outer_fold": outer_fold,
            "target_free_context_count": target_free["target_free_context_count"],
            "target_free_cache_count": target_free["target_free_cache_count"],
            "target_free_seal_sha256": target_free["target_free_seal_sha256"],
            "init_checkpoint_receipt": dict(init_receipt),
        }
    )
    # Only after all 56 contexts and model state are sealed may roles be opened.
    joined = open_loss_roles_and_build_fold(
        loss_role_manifest_path=loss_role_manifest_path,
        target_free=target_free,
        outer_fold=outer_fold,
    )
    if resume_checkpoint_path is None:
        trainer = start_training(
            model,
            joined["correct_pool"],
            joined["wrong_pool"],
            joined["schedule"],
        )
        resume_from_update = 0
    else:
        resume_path = safe_input_path(resume_checkpoint_path)
        resume_payload = torch.load(
            resume_path, map_location="cpu", weights_only=True
        )
        require(
            isinstance(resume_payload, Mapping)
            and resume_payload.get("schema_version") == CHECKPOINT_SCHEMA
            and resume_payload.get("status") == INTERRUPTED_STATUS
            and resume_payload.get("outer_fold") == outer_fold
            and resume_payload.get("update")
            == resume_payload.get("training_state", {}).get("completed_updates")
            and 0 < int(resume_payload.get("update", 0)) < 128
            and resume_payload.get("target_free_seal_sha256")
            == target_free_seal
            and resume_payload.get("address_manifest_sha256")
            == file_sha256(safe_input_path(address_manifest_path))
            and resume_payload.get("loss_role_manifest_sha256")
            == joined["loss_role_manifest_sha256"],
            "resume checkpoint input binding drift",
        )
        require(
            resume_payload.get("cache_index_sha256")
            == file_sha256(safe_input_path(cache_index_path))
            and resume_payload.get("init_checkpoint_receipt", {}).get("sha256")
            == init_checkpoint_sha256
            and (
                authority_execution_receipt is None
                or (
                    resume_payload.get("authority_sha256")
                    == authority_execution_receipt["authority_sha256"]
                    and resume_payload.get("authority_logical_sha256")
                    == authority_execution_receipt["authority_logical_sha256"]
                )
            ),
            "resume checkpoint physical/authority binding drift",
        )
        state = formal_state_from_payload(resume_payload["training_state"])
        trainer = resume_training(
            model,
            joined["correct_pool"],
            joined["wrong_pool"],
            joined["schedule"],
            state,
        )
        resume_from_update = state.completed_updates
    run_updates(trainer, stop_after_updates=stop_after_update)
    state = export_training_state(trainer)
    gradient_coverage = gradient_coverage_receipt(
        state.trace, state.parameter_contract
    )
    evaluations = (
        evaluate_fold(model, joined["evaluations"])
        if stop_after_update == 128
        else []
    )
    # Publish a stage as one immutable directory.  A scheduler signal can arrive
    # between any two individual file writes; writing directly into output_dir
    # would then leave an apparently existing, but incomplete, stage that a
    # continuation cannot safely consume.  The hidden sibling is deliberately
    # outside the authorized final path and is renamed only after checkpoint,
    # trace and result have all been fsync'ed and closed.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent)
    )
    checkpoint_path = staging_dir / f"checkpoint_update{stop_after_update}.pt"
    published_checkpoint_path = (
        output_dir / f"checkpoint_update{stop_after_update}.pt"
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": (
            RESULT_STATUS if stop_after_update == 128 else INTERRUPTED_STATUS
        ),
        "outer_fold": outer_fold,
        "authority_execution_receipt": None
        if authority_execution_receipt is None
        else dict(authority_execution_receipt),
        "authority_sha256": None
        if authority_execution_receipt is None
        else authority_execution_receipt["authority_sha256"],
        "authority_logical_sha256": None
        if authority_execution_receipt is None
        else authority_execution_receipt["authority_logical_sha256"],
        "update": stop_after_update,
        "resume_from_update": resume_from_update,
        "target_free_seal_sha256": target_free_seal,
        "address_manifest_sha256": file_sha256(
            safe_input_path(address_manifest_path)
        ),
        "loss_role_manifest_sha256": joined[
            "loss_role_manifest_sha256"
        ],
        "cache_index_sha256": file_sha256(safe_input_path(cache_index_path)),
        "init_checkpoint_receipt": dict(init_receipt),
        "training_state": formal_state_payload(state),
        "model_state_sha256": state.model_state_sha256,
        "optimizer_state_sha256": state.optimizer_state_sha256,
        "trace_sha256": state.trace_sha256,
        "gradient_coverage": gradient_coverage,
        "all_trainable_tensors_finite_ever_nonzero": gradient_coverage[
            "all_tensors_finite"
        ]
        and gradient_coverage["all_tensors_ever_nonzero"],
    }
    atomic_torch(checkpoint_path, checkpoint)
    trace_path = staging_dir / "trace.json"
    published_trace_path = output_dir / "trace.json"
    trace_value = {
        "schema_version": SCHEMA_VERSION,
        "outer_fold": outer_fold,
        "completed_updates": state.completed_updates,
        "schedule_sha256": state.schedule_sha256,
        "trace": [dict(row) for row in state.trace],
        "trace_sha256": state.trace_sha256,
    }
    trace_value["logical_sha256"] = logical_sha256(trace_value)
    atomic_json(trace_path, trace_value)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            RESULT_STATUS if stop_after_update == 128 else INTERRUPTED_STATUS
        ),
        "outer_fold": outer_fold,
        "completed_updates": state.completed_updates,
        "resume_from_update": resume_from_update,
        "authority_execution_receipt": None
        if authority_execution_receipt is None
        else dict(authority_execution_receipt),
        "authority_sha256": None
        if authority_execution_receipt is None
        else authority_execution_receipt["authority_sha256"],
        "authority_logical_sha256": None
        if authority_execution_receipt is None
        else authority_execution_receipt["authority_logical_sha256"],
        "target_free_context_count_loaded_before_loss_role": target_free[
            "target_free_context_count"
        ],
        "target_free_cache_count_loaded_before_loss_role": target_free[
            "target_free_cache_count"
        ],
        "target_free_seal_before_loss_role": True,
        "target_free_seal_sha256": target_free_seal,
        "loss_role_open_count": 1,
        "opened_sealed_read_count": 0,
        "selected_raw_correct_count": len(joined["correct_pool"]),
        "selected_raw_wrong_count": len(joined["wrong_pool"]),
        "explicit_schedule_record_count": len(joined["schedule"]),
        "schedule_sha256": state.schedule_sha256,
        "checkpoint_path": str(published_checkpoint_path.resolve()),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "trace_path": str(published_trace_path.resolve()),
        "trace_bytes": trace_path.stat().st_size,
        "trace_file_sha256": file_sha256(trace_path),
        "model_state_sha256": state.model_state_sha256,
        "optimizer_state_sha256": state.optimizer_state_sha256,
        "trace_sha256": state.trace_sha256,
        "gradient_coverage": gradient_coverage,
        "all_trainable_tensors_finite_ever_nonzero": gradient_coverage[
            "all_tensors_finite"
        ]
        and gradient_coverage["all_tensors_ever_nonzero"],
        "evaluation_record_count": len(evaluations),
        "evaluation_records": evaluations,
        "evaluation_record_population_sha256": canonical_sha256(evaluations),
        "gx_action_call_count": len(evaluations),
        "all_actions_relative_null_only": all(
            row["action"] in {"SWITCH", "RELATIVE_NULL_HOLD"}
            and row["target_absence_claimed"] is False
            for row in evaluations
        ),
        "formal_scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    result["logical_sha256"] = logical_sha256(result)
    try:
        atomic_json(staging_dir / "result.json", result)
        os.replace(staging_dir, output_dir)
        staging_dir = Path()
        return result
    finally:
        # Normal Python failures are recoverable without touching any committed
        # output.  SIGKILL may leave only a hidden staging sibling; it is never
        # treated as a checkpoint by the launcher.
        if staging_dir != Path() and staging_dir.exists():
            shutil.rmtree(staging_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--address-manifest", type=Path, required=True)
    parser.add_argument("--loss-role-manifest", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-after-update", type=int, default=128)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        authority_receipt = validate_execution_authority(
            args.authority.resolve(strict=True),
            args.outer_fold,
            args.output_dir.resolve(strict=False),
        )
        value = run_fold(
            address_manifest_path=args.address_manifest,
            loss_role_manifest_path=args.loss_role_manifest,
            cache_index_path=args.cache_index,
            outer_fold=args.outer_fold,
            init_checkpoint_path=args.init_checkpoint,
            output_dir=args.output_dir.resolve(strict=False),
            stop_after_update=args.stop_after_update,
            resume_checkpoint_path=args.resume_checkpoint,
            authority_execution_receipt=authority_receipt,
        )
        print(
            json.dumps(
                {
                    "status": value["status"],
                    "outer_fold": value["outer_fold"],
                    "completed_updates": value["completed_updates"],
                    "evaluation_record_count": value[
                        "evaluation_record_count"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": ABORT,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
