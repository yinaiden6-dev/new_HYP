#!/usr/bin/env python3
"""Materialize one target-free Track-R outer-OOF pair/control shard.

Each query consumes the independently frozen two-member role-free pair address,
its outer-refit P-V2 head, and its fold-local trained Track-R checkpoint.  No
target/rival role, RAW score, correctness, or scientific result is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))
sys.path.insert(0, str(RC_ROOT / "programs"))

from rc_aslo_xf.dino_rcde_cw1_multitile_vdecode_v1 import (  # noqa: E402
    CandidateReferenceFieldV1,
    QueryTokenFieldV1,
)
from rc_aslo_xf.dino_rcde_sr0_mt_controls_v1 import (  # noqa: E402
    _p_reference_score,
    c_dino_v_derangement,
    decode_c_dino_pair,
    move_c_dino_locks_to_device,
    permute_query_content,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402
from rc_aslo_xf.dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1 import (  # noqa: E402
    GeometryNamespaceBindingV1,
    outer_refit_head_spec,
    project_natural_v2_candidate_to_v1_runtime,
    select_natural_v2_head_records,
)
from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (  # noqa: E402
    CandidatePLockV1,
    decode_pair,
    move_locks_to_device,
    serialize_arm_evidence,
    state_dict_sha256,
    tensor_sha256,
)
from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2  # noqa: E402

from dino_rcde_track_r_v121_lineage_v1 import (  # noqa: E402
    read_authority as read_v121_authority,
    validate_runtime_bindings,
)
from rc_aslo_xf.gallery_identity_repair import build_identity_map  # noqa: E402

import dino_rcde_sr0_mt_v_input_common_v1 as v_input  # noqa: E402
import dino_rcde_track_r_input_common_v1 as track_input  # noqa: E402


AUTHORITY_PATH = "registry/current_authority_v121_20260821.json"
AUTHORITY_SCHEMA = "rc_current_authority_v121_20260821"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARDS_AUTHORIZED"
GALLERY_CACHE = (
    RC_ROOT.parents[3]
    / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
).resolve()
OUTPUT_SCHEMA = "rc_dino_rcde_track_r_oof_prejoin_shard_v1_20260821"
OUTPUT_STATUS = "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARD_READY"
PAIR_SCHEMA = "rc_dino_rcde_sr0_mt_p_role_free_pair_address_v1_20260815"
PAIR_MANIFEST_STATUS = "RCDE_SR0_MT_ROLE_FREE_PAIR_ADDRESS_READY"
LOCK_ARTIFACT_SCHEMA = "rc_dino_rcde_sr0_mt_p_v2_formal_lock_artifact_v1_20260819"
CHECKPOINT_SCHEMA = "rc_dino_rcde_sr0_mt_v_checkpoint_v1"
SHARD_SIZE = 12
SHARD_COUNT = 50
QUERY_COUNT = 600
EXCLUDED = (25, 26, 101, 346, 354, 470)
ARMS = (
    "ALL_PATCH_SAME_MODEL",
    "CW1_QUERY_MULTITILE_FULL_REFERENCE",
    "CW1_QUERY_MULTITILE_LOCAL_COMPONENT_SET",
)


class TrackROOFError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackROOFError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def safe_path(path: Path, *, must_exist: bool = True, file: bool = True) -> Path:
    value = path.resolve()
    require(RC_ROOT == value or RC_ROOT in value.parents, f"path escapes RC root: {value}")
    require(
        not {"c8", "s8", "opened", "sealed"}.intersection(
            part.lower() for part in value.parts
        ),
        f"protected path requested: {value}",
    )
    if must_exist:
        require((value.is_file() if file else value.is_dir()) and not value.is_symlink(), f"input absent/unsafe: {value}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(safe_path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def binding_path(authority: Mapping[str, Any], name: str) -> Path:
    item = authority.get("bindings", {}).get(name)
    require(isinstance(item, Mapping) and set(item) >= {"path", "sha256"}, f"binding absent: {name}")
    path = safe_path(RC_ROOT / str(item["path"]))
    require(file_sha256(path) == item["sha256"], f"binding hash drift: {name}")
    return path


def external_gallery_binding_path(authority: Mapping[str, Any]) -> Path:
    item = authority.get("bindings", {}).get("gallery_cache")
    require(
        isinstance(item, Mapping) and set(item) >= {"path", "sha256"},
        "external gallery binding absent",
    )
    path = Path(str(item["path"])).resolve()
    require(
        path == GALLERY_CACHE and path.is_file() and not path.is_symlink(),
        "external gallery cache path drift",
    )
    require(file_sha256(path) == item["sha256"], "external gallery cache hash drift")
    return path


def atomic_torch(path: Path, value: object) -> None:
    output = safe_path(path, must_exist=False)
    require(not output.exists(), f"immutable prejoin shard exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, output)
    output.chmod(0o444)


def _identity_labels(gallery_cache_path: Path) -> tuple[str, ...]:
    gallery = torch.load(
        safe_path(gallery_cache_path), map_location="cpu", weights_only=True, mmap=True
    )
    legacy = gallery.get("setids") if isinstance(gallery, Mapping) else None
    require(isinstance(legacy, (list, tuple)) and len(legacy) == 5413, "gallery legacy identity axis drift")
    return build_identity_map(tuple(map(str, legacy))).labels


def _load_model(
    path: Path,
    outer_fold: int,
    device: torch.device,
    expected_v120_authority_sha256: str,
    expected_v120_authority_logical_sha256: str,
) -> tuple[DINO_RCDE_V1_2, str]:
    checkpoint_path = safe_path(path)
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    require(
        isinstance(payload, Mapping)
        and payload.get("schema_version") == CHECKPOINT_SCHEMA
        and payload.get("status") == "DINO_RCDE_TRACK_R_RELATIVE_CHECKPOINT_COMPLETE"
        and payload.get("outer_fold") == outer_fold
        and payload.get("authority_sha256") == expected_v120_authority_sha256
        and payload.get("authority_logical_sha256")
        == expected_v120_authority_logical_sha256
        and payload.get("arm") == "RCDE_CONTEXT_TRACK_R_RELATIVE"
        and payload.get("update") == 2048
        and payload.get("relative_pair_loss_only") is True,
        "Track-R fold checkpoint drift",
    )
    model = DINO_RCDE_V1_2().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    state = state_dict_sha256(model)
    require(state == payload.get("final_state_sha256"), "Track-R fold checkpoint state drift")
    model.eval()
    return model, state


def _max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    left = torch.as_tensor(first, dtype=torch.float64).detach().cpu()
    right = torch.as_tensor(second, dtype=torch.float64).detach().cpu()
    require(left.shape == right.shape, "numeric comparison shape drift")
    return 0.0 if left.numel() == 0 else float((left - right).abs().max())


def _serialize_control_evidence(evidence) -> dict[str, Any]:
    return serialize_arm_evidence(evidence)


def materialize(
    authority_path: Path,
    shard_ordinal: int,
    output_root: Path,
    *,
    device: str,
) -> dict[str, Any]:
    require(0 <= shard_ordinal < SHARD_COUNT, "shard ordinal drift")
    authority_path = safe_path(authority_path)
    authority, authority_sha256 = read_v121_authority(
        authority_path,
        root=RC_ROOT,
        expected_relative=AUTHORITY_PATH,
        expected_schema=AUTHORITY_SCHEMA,
        expected_status=AUTHORITY_STATUS,
    )
    require(
        authority.get("schema_version") == AUTHORITY_SCHEMA
        and authority.get("status") == AUTHORITY_STATUS
        and authority.get("oof_prejoin_shard_materialization_authorized") is True
        and authority.get("role_free_pair_read_authorized") is True
        and authority.get("label_target_rival_read_authorized") is False
        and authority.get("model_load_authorized") is True
        and authority.get("model_forward_authorized") is True
        and authority.get("model_backward_authorized") is False
        and authority.get("model_update_authorized") is False
        and authority.get("scientific_reduction_authorized") is False
        and authority.get("protected_access_authorized") is False,
        "V121 prejoin authority drift",
    )
    validate_runtime_bindings(
        authority,
        root=RC_ROOT,
        required_names=(
            "producer",
            "validator",
            "finalizer",
            "lineage_runtime",
            "runtime_entry_validator",
            "package_initializer",
            "launcher",
            "post_vfit_controller",
            "post_oof_controller",
        ),
    )
    expected_v120_authority_sha256 = authority.get(
        "parent_v120_authority_sha256"
    )
    expected_v120_authority_logical_sha256 = authority.get(
        "parent_v120_authority_logical_sha256"
    )
    require(
        isinstance(expected_v120_authority_sha256, str)
        and isinstance(expected_v120_authority_logical_sha256, str)
        and authority.get("all_fit_outputs_parent_authority_match") is True,
        "V121 parent V120 lineage receipt drift",
    )
    aggregate = read_json(binding_path(authority, "p_v2_lock_aggregate"))
    folds = read_json(binding_path(authority, "fold_schedule"))
    pair_outer = read_json(binding_path(authority, "role_free_pair_manifest"))
    pair_validation = read_json(binding_path(authority, "role_free_pair_validation"))
    require(
        pair_outer.get("status") == PAIR_MANIFEST_STATUS
        and pair_outer.get("role_free") is True
        and pair_outer.get("query_count") == 594
        and pair_outer.get("excluded_execution_ordinals") == list(EXCLUDED)
        and pair_validation.get("validation_pass") is True,
        "role-free pair address closure drift",
    )
    pair_manifest = pair_outer.get("pair_manifest")
    require(isinstance(pair_manifest, Mapping) and isinstance(pair_manifest.get("records"), list), "role-free pair manifest absent")
    pair_by_execution = {int(item["execution_ordinal"]): item for item in pair_manifest["records"]}
    aggregate_by_execution = {int(item["execution_ordinal"]): item for item in aggregate["rows"]}
    fold_records = tuple(folds["records"])
    require(
        len(fold_records) == QUERY_COUNT
        and len({int(item["query_ordinal"]) for item in fold_records}) == QUERY_COUNT,
        "OOF fold record population/historical-ordinal drift",
    )
    fold_by_execution = dict(enumerate(fold_records))
    expected_eligible = set(range(QUERY_COUNT)) - set(EXCLUDED)
    require(
        set(pair_by_execution) == set(aggregate_by_execution) == expected_eligible,
        "OOF eligible execution-axis population drift",
    )
    for execution in expected_eligible:
        fold = fold_by_execution[execution]
        pair = pair_by_execution[execution]
        aggregate_row = aggregate_by_execution[execution]
        require(
            pair.get("query_id") == aggregate_row.get("query_id") == fold.get("query_id")
            and aggregate_row.get("source_fold") == fold.get("inner_fold"),
            f"OOF execution-axis query/fold closure drift: {execution}",
        )
    geometry_queries, geometry_references, geometry_sha = track_input.load_geometry_payload(
        binding_path(authority, "geometry_payload")
    )
    cache_index, _, cache_index_sha, schedule_sha, schedule_logical = v_input.load_schedule_cache_index(
        binding_path(authority, "redacted_schedule"),
        binding_path(authority, "redacted_cache_index"),
    )
    cache_root = safe_path(RC_ROOT / "runtime/dino_rcde_p0_v1_2/stable_fp16_cache", file=False)
    identities = _identity_labels(external_gallery_binding_path(authority))
    expected_model_sha = str(authority["cache_model_checkpoint_logical_sha256"])
    target_device = torch.device(device)
    require(device != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    models: dict[int, tuple[DINO_RCDE_V1_2, str]] = {}
    start = shard_ordinal * SHARD_SIZE
    stop = min(QUERY_COUNT, start + SHARD_SIZE)
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    model_forward_count = 0
    for execution in range(start, stop):
        if execution in EXCLUDED:
            require(execution not in pair_by_execution and execution not in aggregate_by_execution, "excluded query leaked")
            exclusions.append({"execution_ordinal": execution, "reason": "NATURAL_C128_TARGET_MISS"})
            continue
        pair = pair_by_execution[execution]
        fold = fold_by_execution[execution]
        outer_fold = int(fold["inner_fold"])
        aggregate_row = aggregate_by_execution[execution]
        artifact_path = safe_path(RC_ROOT / str(aggregate_row["lock_artifact_path"]))
        require(file_sha256(artifact_path) == aggregate_row["lock_artifact_sha256"], "P-V2 artifact hash drift")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True, mmap=True)
        require(
            isinstance(artifact, Mapping)
            and artifact.get("schema_version") == LOCK_ARTIFACT_SCHEMA
            and artifact.get("target_free") is True
            and artifact.get("execution_ordinal") == execution
            and artifact.get("source_fold") == outer_fold
            and artifact.get("record_count") == 1024,
            "P-V2 artifact envelope drift",
        )
        records = artifact.get("records")
        require(isinstance(records, list), "P-V2 records absent")
        spec = outer_refit_head_spec(source_fold=outer_fold)
        selected = select_natural_v2_head_records(records, spec=spec, expected_candidate_count=128)
        by_key: dict[str, dict[str, Mapping[str, Any]]] = {}
        for record in selected:
            by_key.setdefault(record["candidate"]["candidate_key"], {})[
                record["direction"]
            ] = record
        members = pair.get("members")
        require(isinstance(members, list) and len(members) == 2, "role-free pair members drift")
        q_geometry = geometry_queries[execution]
        query = v_input.make_query_field(
            {
                "execution_ordinal": execution,
                "query_source_image_sha256": pair["query_source_image_sha256"],
            },
            cache_root=cache_root,
            index=cache_index,
            expected_model_sha256=expected_model_sha,
        )
        locks: dict[str, CandidatePLockV1] = {}
        for member in members:
            key = str(member["candidate_key"])
            row = int(member["candidate_physical_row"])
            require(key in by_key and row in geometry_references, "pair member absent from outer-refit P-V2 axis")
            candidate = v_input.make_candidate_field(
                row,
                str(member["candidate_reference_source_sha256"]),
                cache_root=cache_root,
                index=cache_index,
                expected_model_sha256=expected_model_sha,
            )
            r_geometry = geometry_references[row]
            projection = project_natural_v2_candidate_to_v1_runtime(
                query=query,
                candidate=candidate,
                direction_records=by_key[key],
                spec=spec,
                query_geometry_binding=GeometryNamespaceBindingV1(
                    source_image_sha256=query.source_image_sha256,
                    grid_shape=query.grid_shape,
                    canonical_geometry_sha256=q_geometry["dino_geometry"]["geometry_sha256"],
                    cache_geometry_record_sha256=query.geometry_record_sha256,
                ),
                reference_geometry_binding=GeometryNamespaceBindingV1(
                    source_image_sha256=candidate.source_image_sha256,
                    grid_shape=candidate.grid_shape,
                    canonical_geometry_sha256=r_geometry["dino_geometry"]["geometry_sha256"],
                    cache_geometry_record_sha256=candidate.geometry_record_sha256,
                ),
            )
            locks[key] = projection.runtime_lock
        left, right = (str(member["candidate_key"]) for member in members)
        if outer_fold not in models:
            models[outer_fold] = _load_model(
                binding_path(authority, f"track_r_checkpoint_fold{outer_fold}"),
                outer_fold,
                target_device,
                expected_v120_authority_sha256,
                expected_v120_authority_logical_sha256,
            )
        model, expected_state = models[outer_fold]
        before_state = state_dict_sha256(model)
        query_device = QueryTokenFieldV1(
            layers=query.layers.to(target_device),
            grid_shape=query.grid_shape,
            valid_patch_mask=query.valid_patch_mask,
            source_image_sha256=query.source_image_sha256,
            source_key=query.source_key,
            cache_payload_sha256=query.cache_payload_sha256,
            geometry_record_sha256=query.geometry_record_sha256,
            tokens_sha256=query.tokens_sha256,
        )
        locks_device = move_locks_to_device(locks, target_device)
        with torch.no_grad():
            real = decode_pair(model, query_device, locks_device, left, right)
            reordered = decode_pair(
                model,
                query_device,
                dict(reversed(tuple(locks_device.items()))),
                left,
                right,
            )
            swapped = decode_pair(model, query_device, locks_device, right, left)
            p_query_field, p_query_receipt = permute_query_content(
                query_device,
                namespace="TRACK_R_OOF_P_QUERY_V1_SEED17",
                query_id=str(pair["query_id"]),
            )
            p_query = decode_pair(model, p_query_field, locks_device, left, right)
            identity_map = {
                key: identities[locks[key].candidate.physical_gallery_row] for key in locks
            }
            c_dino_cpu, c_dino_receipts, c_dino_ineligible = c_dino_v_derangement(
                locks,
                namespace="TRACK_R_OOF_C_DINO_PAIR_V1_SEED17",
                query_id=str(pair["query_id"]),
                corrected_identity_by_physical_row={
                    locks[key].candidate.physical_gallery_row: identity_map[key]
                    for key in locks
                },
            )
            require(not c_dino_ineligible, "role-free pair C_DINO is ineligible")
            c_dino, c_dino_transport = decode_c_dino_pair(
                model,
                query_device,
                move_c_dino_locks_to_device(c_dino_cpu, target_device),
                left,
                right,
            )
        model_forward_count += 5
        real_serialized = serialize_arm_evidence(real)
        reordered_serialized = serialize_arm_evidence(reordered)
        swapped_serialized = serialize_arm_evidence(swapped)
        p_query_serialized = serialize_arm_evidence(p_query)
        reorder_error = max(
            _max_abs(
                real.by_name()[arm].scalar_contributions,
                reordered.by_name()[arm].scalar_contributions,
            )
            for arm in real.by_name()
        )
        swap_error = max(
            float(
                (
                    real.by_name()[arm].logit + swapped.by_name()[arm].logit
                )
                .abs()
                .detach()
                .cpu()
            )
            for arm in real.by_name()
        )
        p_reference: dict[str, Any] = {}
        with torch.no_grad():
            for arm in real.by_name():
                contributions, margin = _p_reference_score(
                    model,
                    query_device,
                    locks_device,
                    left,
                    right,
                    arm,
                    namespace="TRACK_R_OOF_P_REFERENCE_V1_SEED17",
                    query_id=str(pair["query_id"]),
                )
                p_reference[arm] = {
                    "logit": float(margin.detach().cpu()),
                    "scalar_contributions": contributions.detach().cpu(),
                    "scalar_contributions_sha256": tensor_sha256(
                        contributions.detach().cpu()
                    ),
                }
        model_forward_count += 3
        require(before_state == state_dict_sha256(model) == expected_state, "OOF forward mutated Track-R model")
        row_value: dict[str, Any] = {
            "query_id": pair["query_id"],
            "execution_ordinal": execution,
            "query_source_image_sha256": pair["query_source_image_sha256"],
            "outer_fold": outer_fold,
            "pair_sha256": pair["pair_sha256"],
            "candidate_axis_sha256": pair["candidate_axis_sha256"],
            "member_addresses": members,
            "member_order": [left, right],
            "checkpoint_state_sha256": expected_state,
            "real": real_serialized,
            "candidate_reorder": reordered_serialized,
            "pair_swap": swapped_serialized,
            "P_QUERY": p_query_serialized,
            "P_QUERY_receipt": p_query_receipt.logical_record(),
            "C_DINO_V": serialize_arm_evidence(c_dino),
            "C_DINO_V_receipts": [receipt.logical_record() for receipt in c_dino_receipts],
            "C_DINO_V_mask_transport_sha256": dict(c_dino_transport),
            "P_REFERENCE": p_reference,
            "candidate_reorder_max_abs": reorder_error,
            "pair_swap_max_abs": swap_error,
            "real_control_direction_inspection_count": 0,
        }
        row_value["record_sha256"] = logical_sha256(row_value)
        rows.append(row_value)
        del artifact, records, selected, by_key, query, locks, query_device, locks_device
    value: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "status": OUTPUT_STATUS,
        "claim_level": "TARGET_FREE_OUTER_OOF_ROLE_FREE_PAIR_THREE_ARM_AND_CONTROLS_ONLY",
        "shard_ordinal": shard_ordinal,
        "execution_start": start,
        "execution_stop": stop,
        "query_count": len(rows),
        "excluded_query_count": len(exclusions),
        "rows": rows,
        "row_sequence_sha256": canonical_sha256([item["record_sha256"] for item in rows]),
        "exclusions": exclusions,
        "source_bindings": {
            "authority_sha256": authority_sha256,
            "parent_v120_authority_sha256": expected_v120_authority_sha256,
            "parent_v120_authority_logical_sha256": (
                expected_v120_authority_logical_sha256
            ),
            "role_free_pair_manifest_sha256": file_sha256(
                binding_path(authority, "role_free_pair_manifest")
            ),
            "p_v2_lock_aggregate_sha256": file_sha256(
                binding_path(authority, "p_v2_lock_aggregate")
            ),
            "geometry_payload_sha256": geometry_sha,
            "redacted_cache_index_sha256": cache_index_sha,
            "redacted_schedule_sha256": schedule_sha,
            "redacted_schedule_logical_sha256": schedule_logical,
        },
        "access_audit": {
            "model_forward_count": model_forward_count,
            "model_backward_count": 0,
            "model_update_count": 0,
            "label_target_rival_read_count": 0,
            "raw_score_rank_correctness_outcome_read_count": 0,
            "role_free_pair_read_count": len(rows),
            "corrected_identity_control_only_read_count": len(rows) * 2,
            "scientific_reduction_count": 0,
            "protected_access_count": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical_sha256(value)
    output = safe_path(output_root, must_exist=False) / f"shard_{start:03d}_{stop:03d}.pt"
    atomic_torch(output, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--shard-ordinal", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    value = materialize(
        args.authority, args.shard_ordinal, args.output_root, device=args.device
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                "shard_ordinal": value["shard_ordinal"],
                "query_count": value["query_count"],
                "logical_sha256": value["logical_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
