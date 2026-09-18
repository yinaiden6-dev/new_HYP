#!/usr/bin/env python3
"""Independent zero-RoMa replay of one RoMa-RGH natural-S0 shard.

This validator deliberately does not import the producer.  It reconstructs
all five proposal populations, every frozen H/H0 family, and every RAW-
ColNomic readout from the immutable per-query atom-bank checkpoints plus the
sanitized mathematical input.  Neither image bytes nor labels are read.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "programs"), str(ROOT / "src")]

from roma_rgh_reference_first_s0_sanitized_loader_v1 import (  # noqa: E402
    canonical_sha256,
    load_sanitized_s0_shard,
)
from rc_aslo_xf.r0_natural_binding_v1 import (  # noqa: E402
    spatial_permutation_is_affine,
)
from rc_aslo_xf.roma_rgh_reference_first_s0_runtime_v1 import (  # noqa: E402
    FrozenOpaqueDonorPlan,
    apply_frozen_opaque_donor_plan,
    freeze_opaque_donor_plan,
    opaque_candidate_axis_sha256,
)
from rc_aslo_xf.roma_rgh_reference_first_v1 import (  # noqa: E402
    DEFAULT_HYPOTHESIS_SLOT_CAPACITY,
    FrozenQueryOnlyFamily,
    RawColNomicGrid,
    build_query_only_family,
    coordinate_destroy_bank,
    freeze_hypothesis_family,
    h0_marginal,
    logical_sha256,
    tensor_sha256,
    validate_family,
    validate_reference_first_bank,
)


INPUT_ROOT = ROOT / "results/rc_lth_p_only_sanitized_train_input_v2"
DONOR_PLAN_SCHEMA = "roma_rgh_reference_first_s0_opaque_donor_plans_v1_20260910"
DONOR_PLAN_STATUS = "ROMA_RGH_REFERENCE_FIRST_S0_OPAQUE_DONOR_PLANS_READY"
DONOR_VALIDATION_SCHEMA = "roma_rgh_reference_first_s0_opaque_donor_plans_validation_v1_20260910"
DONOR_VALIDATION_STATUS = "ROMA_RGH_REFERENCE_FIRST_S0_OPAQUE_DONOR_PLANS_VALIDATION_PASS"
CHECKPOINT_SCHEMA = "roma_rgh_reference_first_natural_s0_query_checkpoint_v1_20260910"
SHARD_SCHEMA = "roma_rgh_reference_first_natural_s0_shard_v1_20260910"
SHARD_STATUS = "ROMA_RGH_REFERENCE_FIRST_NATURAL_S0_SHARD_PREJOIN_READY"
AUTHORITY_V1_SCHEMA = "roma_rgh_reference_first_natural_s0_shard_execution_authority_v1_20260910"
AUTHORITY_V2_SCHEMA = "roma_rgh_reference_first_natural_s0_shard_execution_authority_v2_20260910"
AUTHORITY_STATUS = "ROMA_RGH_REFERENCE_FIRST_NATURAL_S0_SHARD_EXECUTION_AUTHORIZED"
VALIDATION_SCHEMA = "roma_rgh_reference_first_natural_s0_shard_validation_v1_20260910"
VALIDATION_STATUS = "ROMA_RGH_REFERENCE_FIRST_NATURAL_S0_SHARD_INDEPENDENT_VALIDATION_PASS"
REPLAY_AUTHORITY_SCHEMA = "roma_rgh_reference_first_natural_s0_replay_authority_v1_20260910"
REPLAY_AUTHORITY_V2_SCHEMA = "roma_rgh_reference_first_natural_s0_replay_authority_v2_20260910"
REPLAY_AUTHORITY_STATUS = "ROMA_RGH_REFERENCE_FIRST_NATURAL_S0_REPLAY_AUTHORIZED"
CONTROL_NAMESPACES = (
    "REAL",
    "C_P_BIND",
    "C_WHOLE",
    "P_QUERY_COORD",
    "P_REFERENCE_COORD",
)
READOUT_NAMES = (
    "all_patch_full_reference",
    "query_only_full_reference",
    "rgh_query_full_reference",
    "rgh_paired_region",
)
READOUT_NAMES_BY_NAMESPACE = {
    namespace: (("rgh_query_full_reference",) if namespace == "C_P_BIND" else READOUT_NAMES)
    for namespace in CONTROL_NAMESPACES
}
HEX64 = frozenset("0123456789abcdef")


class ShardValidationAbort(RuntimeError):
    """Fail-closed independent replay error."""


def require(condition: object, code: str) -> None:
    if not bool(condition):
        raise ShardValidationAbort(code)


@dataclass(frozen=True)
class ValidationShape:
    query_count: int = 4
    candidate_count: int = 128
    slot_capacity: int = DEFAULT_HYPOTHESIS_SLOT_CAPACITY


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in HEX64 for character in value)
    )


def _read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"JSON_INPUT_INVALID:{path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _resolve_bound_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_replay_authority(
    path: Path,
    *,
    shard_index: int,
    output_path: Path,
    authority_v1_path: Path,
    authority_v2_path: Path,
) -> tuple[dict[str, Any], str]:
    authority = _read_json(path.resolve())
    schema = authority.get("schema_version")
    expected_shards = (
        list(range(8))
        if schema == REPLAY_AUTHORITY_SCHEMA
        else [0, 2, 4, 5, 6, 7]
    )
    require(
        schema in {REPLAY_AUTHORITY_SCHEMA, REPLAY_AUTHORITY_V2_SCHEMA}
        and authority.get("status") == REPLAY_AUTHORITY_STATUS
        and authority.get("shard_indices") == expected_shards
        and shard_index in expected_shards
        and authority.get("model_load_count") == 0
        and authority.get("roma_forward_count") == 0
        and authority.get("label_or_identity_read_count") == 0
        and authority.get("automatic_stage_advance") is False
        and authority.get("scientific_GO_or_NO_GO") is None
        and authority.get("next_authorized_stage") is None,
        "REPLAY_AUTHORITY_ENVELOPE_DRIFT",
    )
    bindings = authority.get("bindings")
    require(isinstance(bindings, dict) and bool(bindings), "REPLAY_AUTHORITY_BINDINGS_ABSENT")
    for item in bindings.values():
        require(
            isinstance(item, dict)
            and set(item) == {"path", "sha256"}
            and _resolve_bound_path(item["path"]).is_file()
            and not _resolve_bound_path(item["path"]).is_symlink()
            and sha256_file(_resolve_bound_path(item["path"])) == item["sha256"],
            "REPLAY_AUTHORITY_BINDING_HASH_DRIFT",
        )
    require(
        _resolve_bound_path(bindings["shard_validator"]["path"]) == Path(__file__).resolve()
        and _resolve_bound_path(bindings["authority_v1"]["path"]) == authority_v1_path.resolve()
        and _resolve_bound_path(bindings["authority_v2"]["path"]) == authority_v2_path.resolve()
        and output_path.resolve()
        == (ROOT / authority["output_root"] / f"shard{shard_index:02d}" / "independent_validation.json").resolve(),
        "REPLAY_AUTHORITY_RUNTIME_BINDING_DRIFT",
    )
    return authority, sha256_file(path.resolve())


def _result_logical_sha256(value: Mapping[str, Any]) -> str:
    return logical_sha256({key: item for key, item in value.items() if key != "logical_sha256"})


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    """Independent copy of the sealed checkpoint's public descriptor contract."""

    return {
        "schema_version": value["schema_version"],
        "shard_index": value["shard_index"],
        "query_index": value["query_index"],
        "query_resource_key": value["query_resource_key"],
        "candidate_keys": list(value["candidate_keys"]),
        "candidate_axis_sha256": value["candidate_axis_sha256"],
        "source_provenance_sha256": value["source_provenance_sha256"],
        "source_p_input_sha256": value["source_p_input_sha256"],
        "source_broker_sha256": value["source_broker_sha256"],
        "authority_sha256": value["authority_sha256"],
        "donor_indices": list(value["donor_indices"]),
        "donor_plan_logical_sha256": value["donor_plan_logical_sha256"],
        "planner_validation_sha256": value["planner_validation_sha256"],
        "slot_capacity": value["slot_capacity"],
        "control_namespaces": list(value["control_namespaces"]),
        "readout_names_by_namespace": {
            namespace: list(value["readout_names_by_namespace"][namespace])
            for namespace in CONTROL_NAMESPACES
        },
        "real_bank_sha256": [bank.logical_sha256 for bank in value["real_banks"]],
        "query_visibility_sha256": tensor_sha256(value["query_visibility"]),
        "query_only_family_sha256": value["query_only_family_sha256"],
        "query_coordinate_permutation_sha256": tensor_sha256(value["query_coordinate_permutation"]),
        "reference_coordinate_permutation_sha256": [
            tensor_sha256(item) for item in value["reference_coordinate_permutations"]
        ],
        "query_valid_mask_sha256": tensor_sha256(value["query_valid_mask"]),
        "reference_valid_mask_sha256": [tensor_sha256(item) for item in value["reference_valid_masks"]],
        "family_receipts_sha256": canonical_sha256(value["family_receipts"]),
        "score_tensor_sha256": {
            namespace: {
                name: tensor_sha256(value["scores"][namespace][name])
                for name in READOUT_NAMES_BY_NAMESPACE[namespace]
            }
            for namespace in CONTROL_NAMESPACES
        },
        "model_forward_count": value["model_forward_count"],
        "forbidden_input_read_count": value["forbidden_input_read_count"],
        "dense_tensor_persist_count": value["dense_tensor_persist_count"],
        "training_update_count": value["training_update_count"],
    }


def _family_receipt(family: Any, verification: Mapping[str, Any], *, qfull_only: bool) -> dict[str, Any]:
    validate_family(family)
    receipt = {
        "family_logical_sha256": family.logical_sha256,
        "source_bank_sha256": family.source_bank_sha256,
        "h1_count": family.h1_count,
        "slot_capacity": family.slot_capacity,
        "slot_state_sha256": canonical_sha256(
            [(slot.state, slot.logical_sha256) for slot in family.slots]
        ),
        "qfull_slot_sha256": tensor_sha256(verification["qfull_slot_scores"]),
    }
    if not qfull_only:
        receipt.update(
            {
                "paired_eligible": verification["paired_eligible"],
                "paired_slot_sha256": tensor_sha256(verification["paired_slot_scores"]),
            }
        )
    return receipt


def _complete_derangement(value: torch.Tensor, count: int, code: str) -> torch.Tensor:
    order = torch.as_tensor(value, dtype=torch.int64).detach().cpu().contiguous()
    require(order.shape == (count,), f"{code}_SHAPE_DRIFT")
    require(torch.equal(torch.sort(order).values, torch.arange(count)), f"{code}_NOT_PERMUTATION")
    require(not bool(order.eq(torch.arange(count)).any()), f"{code}_HAS_FIXED_POINT")
    return order


def _normalize_similarity(query: RawColNomicGrid, reference: RawColNomicGrid) -> torch.Tensor:
    q = torch.as_tensor(query.tokens).detach().cpu().contiguous()
    r = torch.as_tensor(reference.tokens).detach().cpu().contiguous()
    qvalid = torch.as_tensor(query.valid_mask, dtype=torch.bool).detach().cpu().contiguous()
    rvalid = torch.as_tensor(reference.valid_mask, dtype=torch.bool).detach().cpu().contiguous()
    require(q.ndim == r.ndim == 2 and q.shape[1] == r.shape[1], "RAW_TOKEN_DIMENSION_DRIFT")
    require(qvalid.shape == (q.shape[0],) and bool(qvalid.any()), "QUERY_RAW_VALID_MASK_DRIFT")
    require(rvalid.shape == (r.shape[0],) and bool(rvalid.any()), "REFERENCE_RAW_VALID_MASK_DRIFT")
    similarity = F.normalize(q.to(torch.float64), dim=1) @ F.normalize(r.to(torch.float64), dim=1).T
    similarity[:, ~rvalid] = -torch.inf
    return similarity.contiguous()


def _score_family_from_similarity(
    family: Any,
    similarity: torch.Tensor,
    query: RawColNomicGrid,
    reference: RawColNomicGrid,
) -> dict[str, Any]:
    """Replay the frozen unary equations while reusing one q-by-r matrix."""

    item = validate_family(family)
    qvalid = torch.as_tensor(query.valid_mask, dtype=torch.bool).detach().cpu().contiguous()
    rvalid = torch.as_tensor(reference.valid_mask, dtype=torch.bool).detach().cpu().contiguous()
    require(query.resource_key == item.query_resource_key, "P_V_QUERY_RESOURCE_DRIFT")
    require(reference.resource_key == item.verification_reference_resource_key, "P_V_REFERENCE_RESOURCE_DRIFT")
    require(tuple(query.grid_shape) == tuple(item.query_grid_shape), "P_V_QUERY_GRID_DRIFT")
    if item.source_reference_resource_key == item.verification_reference_resource_key:
        require(tuple(reference.grid_shape) == tuple(item.reference_grid_shape), "P_V_REFERENCE_GRID_DRIFT")
    require(similarity.shape == (qvalid.numel(), rvalid.numel()), "SIMILARITY_AXIS_DRIFT")
    all_score = similarity[qvalid].max(dim=1).values.mean()
    qslots = torch.zeros(item.slot_capacity, dtype=torch.float64)
    pslots = torch.zeros(item.slot_capacity, dtype=torch.float64)
    paired_eligible = bool(
        reference.resource_key == item.source_reference_resource_key
        and tuple(reference.grid_shape) == tuple(item.reference_grid_shape)
    )
    for slot in item.slots:
        if slot.state == "H0":
            continue
        qindex = slot.query_content_indices
        require(bool(qvalid[qindex].all()), "H1_READS_INVALID_QUERY_CONTENT")
        qslots[slot.slot] = similarity[qindex].max(dim=1).values.mean()
        if paired_eligible:
            refs = slot.reference_content_indices[rvalid[slot.reference_content_indices]]
            require(refs.numel() > 0, "PAIRED_COMPONENT_EMPTY")
            pslots[slot.slot] = similarity[qindex][:, refs].max(dim=1).values.mean()
    return {
        "all_patch_full_reference": all_score,
        "rgh_query_full_reference": h0_marginal(qslots),
        "rgh_paired_region": h0_marginal(pslots) if paired_eligible else pslots.abs().sum() * 0.0,
        "paired_eligible": paired_eligible,
        "qfull_slot_scores": qslots,
        "paired_slot_scores": pslots,
    }


def _score_query_only_from_similarity(
    family: FrozenQueryOnlyFamily,
    similarity: torch.Tensor,
    query: RawColNomicGrid,
    reference: RawColNomicGrid,
) -> torch.Tensor:
    require(family.query_resource_key == query.resource_key, "QUERY_ONLY_QUERY_BINDING_DRIFT")
    require(reference.resource_key in family.candidate_keys, "QUERY_ONLY_REFERENCE_BINDING_DRIFT")
    base = validate_family(family.family)
    qvalid = torch.as_tensor(query.valid_mask, dtype=torch.bool).detach().cpu().contiguous()
    require(tuple(query.grid_shape) == tuple(base.query_grid_shape), "QUERY_ONLY_GRID_DRIFT")
    scores = torch.zeros(base.slot_capacity, dtype=torch.float64)
    for slot in base.slots:
        if slot.state == "H1":
            indices = slot.query_content_indices
            require(bool(qvalid[indices].all()), "QUERY_ONLY_READS_INVALID_CONTENT")
            scores[slot.slot] = similarity[indices].max(dim=1).values.mean()
    return h0_marginal(scores)


def _checkpoint_envelope(value: Mapping[str, Any], *, shape: ValidationShape) -> None:
    expected = {
        "schema_version", "shard_index", "query_index", "query_resource_key",
        "candidate_keys", "candidate_axis_sha256", "source_provenance_sha256",
        "source_p_input_sha256", "source_broker_sha256", "authority_sha256",
        "donor_indices", "donor_plan_logical_sha256", "planner_validation_sha256",
        "slot_capacity", "control_namespaces", "readout_names_by_namespace",
        "real_banks", "query_visibility", "query_only_family_sha256",
        "query_coordinate_permutation", "reference_coordinate_permutations",
        "query_valid_mask", "reference_valid_masks", "family_receipts", "scores",
        "model_forward_count", "forbidden_input_read_count", "dense_tensor_persist_count",
        "training_update_count", "logical_sha256",
    }
    require(set(value) == expected, "QUERY_CHECKPOINT_KEYS_DRIFT")
    require(value["schema_version"] == CHECKPOINT_SCHEMA, "QUERY_CHECKPOINT_SCHEMA_DRIFT")
    keys = tuple(value["candidate_keys"])
    require(
        len(keys) == shape.candidate_count
        and len(set(keys)) == shape.candidate_count
        and value["candidate_axis_sha256"] == opaque_candidate_axis_sha256(keys),
        "QUERY_CHECKPOINT_CANDIDATE_AXIS_DRIFT",
    )
    require(tuple(value["control_namespaces"]) == CONTROL_NAMESPACES, "CONTROL_AXIS_DRIFT")
    require(value["readout_names_by_namespace"] == READOUT_NAMES_BY_NAMESPACE, "READOUT_AXIS_DRIFT")
    require(value["slot_capacity"] == shape.slot_capacity, "SLOT_CAPACITY_DRIFT")
    require(
        value["model_forward_count"] == shape.candidate_count
        and value["forbidden_input_read_count"] == 0
        and value["dense_tensor_persist_count"] == 0
        and value["training_update_count"] == 0,
        "CHECKPOINT_ACCESS_OR_FORWARD_DRIFT",
    )
    require(
        _is_sha256(value["logical_sha256"])
        and value["logical_sha256"] == canonical_sha256(_checkpoint_descriptor(value)),
        "CHECKPOINT_LOGICAL_HASH_DRIFT",
    )


def _validate_descriptor_authority(
    descriptor: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    result_authority_sha256: str,
    v1_authority_sha256: str,
) -> None:
    if result_authority_sha256 == v1_authority_sha256:
        require(
            "authority_sha256" not in descriptor
            and checkpoint.get("authority_sha256") == v1_authority_sha256,
            "V1_CHECKPOINT_DESCRIPTOR_SCHEMA_DRIFT",
        )
    else:
        require(
            descriptor.get("authority_sha256") == checkpoint.get("authority_sha256"),
            "V2_CHECKPOINT_DESCRIPTOR_AUTHORITY_DRIFT",
        )


def replay_query_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    bundle: Any,
    record_index: int,
    donor_plan: FrozenOpaqueDonorPlan,
    allowed_authority_sha256: Sequence[str],
    shape: ValidationShape = ValidationShape(),
) -> dict[str, Any]:
    """Rebuild one sealed query without loading RoMa or image pixels."""

    _checkpoint_envelope(checkpoint, shape=shape)
    record = bundle.math.records[record_index]
    keys = tuple(checkpoint["candidate_keys"])
    require(checkpoint["shard_index"] == bundle.math.shard_index, "QUERY_SHARD_BINDING_DRIFT")
    require(checkpoint["query_index"] == record_index, "QUERY_INDEX_BINDING_DRIFT")
    require(checkpoint["query_resource_key"] == record.resource_key, "QUERY_KEY_BINDING_DRIFT")
    require(keys == tuple(record.candidate_keys), "QUERY_SOURCE_CANDIDATE_AXIS_DRIFT")
    require(checkpoint["candidate_axis_sha256"] == record.candidate_set_sha256, "QUERY_SOURCE_AXIS_HASH_DRIFT")
    require(checkpoint["source_provenance_sha256"] == bundle.math.provenance_digest_sha256, "QUERY_SOURCE_PROVENANCE_DRIFT")
    require(checkpoint["source_p_input_sha256"] == bundle.p_input_sha256, "QUERY_SOURCE_P_INPUT_DRIFT")
    require(checkpoint["source_broker_sha256"] == bundle.broker_sha256, "QUERY_SOURCE_BROKER_DRIFT")
    require(checkpoint["authority_sha256"] in set(allowed_authority_sha256), "CHECKPOINT_AUTHORITY_NOT_ACCEPTED")
    require(
        donor_plan.query_resource_key == record.resource_key
        and donor_plan.candidate_axis_sha256 == record.candidate_set_sha256
        and tuple(checkpoint["donor_indices"]) == donor_plan.donor_indices
        and checkpoint["donor_plan_logical_sha256"] == donor_plan.logical_sha256
        and checkpoint["planner_validation_sha256"] == donor_plan.planner_validation_sha256,
        "QUERY_DONOR_PLAN_BINDING_DRIFT",
    )

    real = tuple(validate_reference_first_bank(bank) for bank in checkpoint["real_banks"])
    require(len(real) == shape.candidate_count, "REAL_BANK_COUNT_DRIFT")
    qvalid = torch.as_tensor(checkpoint["query_valid_mask"], dtype=torch.bool).detach().cpu().contiguous()
    rvalid = tuple(
        torch.as_tensor(item, dtype=torch.bool).detach().cpu().contiguous()
        for item in checkpoint["reference_valid_masks"]
    )
    require(len(rvalid) == shape.candidate_count, "REFERENCE_VALID_MASK_COUNT_DRIFT")
    for destination, bank in enumerate(real):
        require(
            bank.control_namespace == "REAL"
            and bank.query_resource_key == record.resource_key
            and bank.destination_candidate_key == keys[destination]
            and bank.source_reference_resource_key == keys[destination]
            and bank.verification_reference_resource_key == keys[destination]
            and torch.equal(bank.query_valid_axis, qvalid)
            and torch.equal(bank.reference_valid_axis, rvalid[destination]),
            "REAL_BANK_BINDING_DRIFT",
        )

    c_p_bind = apply_frozen_opaque_donor_plan(
        real, donor_plan=donor_plan, namespace="C_P_BIND", proposal_only=True
    )
    c_whole = apply_frozen_opaque_donor_plan(
        real, donor_plan=donor_plan, namespace="C_WHOLE", proposal_only=False
    )
    qperm = _complete_derangement(
        checkpoint["query_coordinate_permutation"], qvalid.numel(), "QUERY_COORDINATE_PERMUTATION"
    )
    require(not spatial_permutation_is_affine(qperm, tuple(record.grid_shape)), "QUERY_COORDINATE_PERMUTATION_AFFINE")
    require(torch.equal(qvalid, qvalid[qperm]), "QUERY_COORDINATE_VALID_STRATUM_DRIFT")
    p_query = tuple(
        coordinate_destroy_bank(
            bank,
            query_permutation=qperm,
            reference_permutation=torch.arange(bank.atom_count),
            namespace="P_QUERY_COORD",
        )
        for bank in real
    )
    p_reference = []
    for destination, bank in enumerate(real):
        permutation = _complete_derangement(
            checkpoint["reference_coordinate_permutations"][destination],
            bank.atom_count,
            "REFERENCE_COORDINATE_PERMUTATION",
        )
        require(
            not spatial_permutation_is_affine(permutation, tuple(bank.reference_grid_shape)),
            "REFERENCE_COORDINATE_PERMUTATION_AFFINE",
        )
        require(torch.equal(rvalid[destination], rvalid[destination][permutation]), "REFERENCE_COORDINATE_VALID_STRATUM_DRIFT")
        p_reference.append(
            coordinate_destroy_bank(
                bank,
                query_permutation=torch.arange(qvalid.numel()),
                reference_permutation=permutation,
                namespace="P_REFERENCE_COORD",
            )
        )
    populations = {
        "REAL": real,
        "C_P_BIND": c_p_bind,
        "C_WHOLE": c_whole,
        "P_QUERY_COORD": p_query,
        "P_REFERENCE_COORD": tuple(p_reference),
    }

    query = RawColNomicGrid(record.resource_key, tuple(record.grid_shape), record.tokens, qvalid)
    references = tuple(
        RawColNomicGrid(
            key,
            tuple(bundle.math.reference_assets[key].grid_shape),
            bundle.math.reference_assets[key].tokens,
            rvalid[index],
        )
        for index, key in enumerate(keys)
    )
    similarities = tuple(_normalize_similarity(query, reference) for reference in references)
    visibility = torch.as_tensor(checkpoint["query_visibility"])
    require(
        visibility.dtype == torch.float64
        and visibility.shape == (shape.candidate_count, qvalid.numel())
        and bool(torch.isfinite(visibility).all()),
        "QUERY_VISIBILITY_DRIFT",
    )
    visibility_hashes = tuple(tensor_sha256(item) for item in visibility)
    query_only = build_query_only_family(
        record.resource_key,
        tuple(record.grid_shape),
        visibility,
        keys,
        visibility_hashes,
        slot_capacity=shape.slot_capacity,
    )
    require(query_only.logical_sha256 == checkpoint["query_only_family_sha256"], "QUERY_ONLY_FAMILY_REPLAY_DRIFT")

    replay_scores: dict[str, dict[str, torch.Tensor]] = {}
    replay_receipts: dict[str, list[dict[str, Any]]] = {}
    donors = donor_plan.donor_indices
    for namespace, population in populations.items():
        rows: dict[str, list[torch.Tensor]] = {
            name: [] for name in READOUT_NAMES_BY_NAMESPACE[namespace]
        }
        receipts: list[dict[str, Any]] = []
        for destination, bank in enumerate(population):
            family = freeze_hypothesis_family(bank, slot_capacity=shape.slot_capacity)
            reference_index = donors[destination] if namespace == "C_WHOLE" else destination
            replay = _score_family_from_similarity(
                family, similarities[reference_index], query, references[reference_index]
            )
            rows["rgh_query_full_reference"].append(replay["rgh_query_full_reference"])
            if namespace != "C_P_BIND":
                rows["all_patch_full_reference"].append(replay["all_patch_full_reference"])
                rows["query_only_full_reference"].append(
                    _score_query_only_from_similarity(
                        query_only, similarities[reference_index], query, references[reference_index]
                    )
                )
                rows["rgh_paired_region"].append(replay["rgh_paired_region"])
            receipts.append(_family_receipt(family, replay, qfull_only=namespace == "C_P_BIND"))
        replay_scores[namespace] = {
            name: torch.stack(values).to(torch.float64).contiguous()
            for name, values in rows.items()
        }
        replay_receipts[namespace] = receipts

    require(set(checkpoint["scores"]) == set(CONTROL_NAMESPACES), "SEALED_SCORE_NAMESPACE_DRIFT")
    require(set(checkpoint["family_receipts"]) == set(CONTROL_NAMESPACES), "SEALED_RECEIPT_NAMESPACE_DRIFT")
    for namespace in CONTROL_NAMESPACES:
        require(set(checkpoint["scores"][namespace]) == set(READOUT_NAMES_BY_NAMESPACE[namespace]), f"SEALED_READOUT_AXIS_DRIFT:{namespace}")
        require(checkpoint["family_receipts"][namespace] == replay_receipts[namespace], f"FAMILY_RECEIPT_REPLAY_DRIFT:{namespace}")
        for name in READOUT_NAMES_BY_NAMESPACE[namespace]:
            require(
                torch.equal(torch.as_tensor(checkpoint["scores"][namespace][name]), replay_scores[namespace][name]),
                f"SCORE_REPLAY_DRIFT:{namespace}:{name}",
            )

    return {
        "query_index": record_index,
        "query_resource_key": record.resource_key,
        "candidate_axis_sha256": record.candidate_set_sha256,
        "checkpoint_logical_sha256": checkpoint["logical_sha256"],
        "checkpoint_authority_sha256": checkpoint["authority_sha256"],
        "legal_h1_candidate_count_by_namespace": {
            namespace: sum(int(item["h1_count"]) > 0 for item in replay_receipts[namespace])
            for namespace in CONTROL_NAMESPACES
        },
        "family_receipts_sha256": canonical_sha256(replay_receipts),
        "score_tensor_sha256": {
            namespace: {
                name: tensor_sha256(replay_scores[namespace][name])
                for name in READOUT_NAMES_BY_NAMESPACE[namespace]
            }
            for namespace in CONTROL_NAMESPACES
        },
        "replayed_family_count": shape.candidate_count * len(CONTROL_NAMESPACES),
        "replayed_similarity_matrix_count": shape.candidate_count,
    }


def _load_authorities(v1_path: Path, v2_path: Path) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    v1 = _read_json(v1_path)
    v2 = _read_json(v2_path)
    v1_sha = sha256_file(v1_path)
    v2_sha = sha256_file(v2_path)
    require(v1.get("schema_version") == AUTHORITY_V1_SCHEMA and v1.get("status") == AUTHORITY_STATUS, "AUTHORITY_V1_ENVELOPE_DRIFT")
    require(v2.get("schema_version") == AUTHORITY_V2_SCHEMA and v2.get("status") == AUTHORITY_STATUS, "AUTHORITY_V2_ENVELOPE_DRIFT")
    require(
        v2.get("predecessor_authority", {}).get("sha256") == v1_sha
        and v2.get("accepted_checkpoint_authority_sha256") == [v1_sha]
        and v2.get("output_root") == v1.get("output_root")
        and v2.get("shard_indices") == [1, 3]
        and v2.get("repair_resume_shards") == [1, 3],
        "AUTHORITY_V1_V2_LINEAGE_DRIFT",
    )
    return v1, v2, v1_sha, v2_sha


def _validated_donor_plans(
    plan_path: Path,
    validation_path: Path,
    *,
    bundle: Any,
    shape: ValidationShape,
) -> tuple[dict[str, FrozenOpaqueDonorPlan], str, str]:
    plan = _read_json(plan_path)
    validation = _read_json(validation_path)
    plan_sha = sha256_file(plan_path)
    validation_sha = sha256_file(validation_path)
    require(plan.get("schema_version") == DONOR_PLAN_SCHEMA and plan.get("status") == DONOR_PLAN_STATUS, "DONOR_PLAN_ENVELOPE_DRIFT")
    require(
        validation.get("schema_version") == DONOR_VALIDATION_SCHEMA
        and validation.get("status") == DONOR_VALIDATION_STATUS
        and validation.get("plan_artifact_sha256") == plan_sha
        and all(value is True for value in validation.get("checks", {}).values()),
        "DONOR_VALIDATION_ENVELOPE_DRIFT",
    )
    raw_by_query = {item["query_resource_key"]: item for item in plan.get("plans", [])}
    output: dict[str, FrozenOpaqueDonorPlan] = {}
    for record in bundle.math.records:
        require(record.resource_key in raw_by_query, "DONOR_QUERY_PLAN_ABSENT")
        raw = raw_by_query[record.resource_key]
        require(
            raw.get("candidate_axis_sha256") == record.candidate_set_sha256
            and raw.get("fixed_point_free") is True
            and raw.get("corrected_identity_disjoint") is True
            and raw.get("result_blind_hash_rule") is True
            and raw.get("source_hashes") == {"p_input_sha256": bundle.p_input_sha256},
            "DONOR_QUERY_PLAN_BINDING_DRIFT",
        )
        donors = raw.get("donor_indices")
        require(isinstance(donors, list) and len(donors) == shape.candidate_count, "DONOR_INDEX_AXIS_DRIFT")
        output[record.resource_key] = freeze_opaque_donor_plan(
            query_resource_key=record.resource_key,
            candidate_axis_sha256=record.candidate_set_sha256,
            donor_indices=donors,
            fixed_point_free=True,
            corrected_identity_disjoint=True,
            planner_validation_sha256=validation_sha,
        )
    return output, plan_sha, validation_sha


def validate_shard(
    *,
    replay_authority_path: Path,
    authority_v1_path: Path,
    authority_v2_path: Path,
    shard_root: Path,
    shard_index: int,
    donor_plan_path: Path,
    donor_validation_path: Path,
    output_path: Path,
    shape: ValidationShape = ValidationShape(),
) -> dict[str, Any]:
    _replay_authority, replay_authority_sha = _load_replay_authority(
        replay_authority_path,
        shard_index=shard_index,
        output_path=output_path,
        authority_v1_path=authority_v1_path,
        authority_v2_path=authority_v2_path,
    )
    v1, v2, v1_sha, v2_sha = _load_authorities(authority_v1_path, authority_v2_path)
    expected_output_root = (ROOT / str(v1["output_root"])).resolve()
    require(shard_root.resolve() == expected_output_root / f"shard{shard_index:02d}", "SHARD_OUTPUT_ROOT_DRIFT")
    require(output_path.resolve() == shard_root.resolve() / "independent_validation.json", "VALIDATION_OUTPUT_PATH_DRIFT")
    result_path = shard_root / "result.json"
    result = _read_json(result_path)
    require(result.get("schema_version") == SHARD_SCHEMA and result.get("status") == SHARD_STATUS, "SHARD_RESULT_ENVELOPE_DRIFT")
    require(result.get("logical_sha256") == _result_logical_sha256(result), "SHARD_RESULT_LOGICAL_HASH_DRIFT")
    require(result.get("shard_index") == shard_index and result.get("query_count") == shape.query_count, "SHARD_RESULT_AXIS_DRIFT")
    require(
        result.get("candidate_count_per_query") == shape.candidate_count
        and result.get("candidate_occurrence_count") == shape.query_count * shape.candidate_count
        and result.get("slot_capacity") == shape.slot_capacity
        and result.get("control_namespaces") == list(CONTROL_NAMESPACES)
        and result.get("readout_names_by_namespace")
        == {name: list(values) for name, values in READOUT_NAMES_BY_NAMESPACE.items()}
        and result.get("sealed_model_forward_count") == shape.query_count * shape.candidate_count
        and result.get("model_load_upper_bound_per_process") == 1
        and result.get("forbidden_input_read_count") == 0
        and result.get("dense_tensor_persist_count") == 0
        and result.get("training_update_count") == 0
        and result.get("scientific_GO_or_NO_GO") is None
        and result.get("next_authorized_stage") is None,
        "SHARD_RESULT_EXECUTION_OR_CLAIM_DRIFT",
    )
    expected_result_authority = v2_sha if shard_index in (1, 3) else v1_sha
    require(result.get("authority_sha256") == expected_result_authority, "SHARD_RESULT_AUTHORITY_DRIFT")
    allowed_authorities = (v1_sha, v2_sha) if shard_index in (1, 3) else (v1_sha,)

    source_root = INPUT_ROOT / f"shard{shard_index:02d}"
    receipt = _read_json(source_root / "receipt.json")
    bundle = load_sanitized_s0_shard(
        source_root / "p_input.pt",
        source_root / "broker.pt",
        expected_p_input_sha256=receipt["p_input_sha256"],
        expected_broker_sha256=receipt["broker_sha256"],
    )
    donor_plans, donor_sha, donor_validation_sha = _validated_donor_plans(
        donor_plan_path, donor_validation_path, bundle=bundle, shape=shape
    )
    require(
        result.get("source_p_input_sha256") == bundle.p_input_sha256
        and result.get("source_broker_sha256") == bundle.broker_sha256
        and result.get("donor_plan_file_sha256") == donor_sha
        and result.get("planner_validation_file_sha256") == donor_validation_sha,
        "SHARD_RESULT_SOURCE_BINDING_DRIFT",
    )

    descriptors = result.get("query_checkpoints")
    require(isinstance(descriptors, list) and len(descriptors) == shape.query_count, "SHARD_CHECKPOINT_DESCRIPTOR_COUNT_DRIFT")
    replay = []
    observed_authorities: list[str] = []
    for index, descriptor in enumerate(descriptors):
        require(descriptor.get("query_index") == index, "SHARD_CHECKPOINT_DESCRIPTOR_ORDER_DRIFT")
        checkpoint_path = (shard_root / str(descriptor["path"])).resolve()
        require(checkpoint_path.parent == (shard_root / "query_checkpoints").resolve(), "CHECKPOINT_PATH_ESCAPE")
        require(checkpoint_path.is_file() and not checkpoint_path.is_symlink(), "CHECKPOINT_FILE_INVALID")
        require(stat.S_IMODE(checkpoint_path.stat().st_mode) == 0o444, "CHECKPOINT_FILE_NOT_IMMUTABLE")
        checkpoint_sha = sha256_file(checkpoint_path)
        require(checkpoint_sha == descriptor.get("sha256"), "CHECKPOINT_PHYSICAL_HASH_DRIFT")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(isinstance(checkpoint, dict), "CHECKPOINT_NOT_DICT")
        require(
            descriptor.get("query_resource_key") == checkpoint.get("query_resource_key")
            and descriptor.get("logical_sha256") == checkpoint.get("logical_sha256"),
            "CHECKPOINT_DESCRIPTOR_BINDING_DRIFT",
        )
        _validate_descriptor_authority(
            descriptor,
            checkpoint,
            result_authority_sha256=expected_result_authority,
            v1_authority_sha256=v1_sha,
        )
        summary = replay_query_checkpoint(
            checkpoint,
            bundle=bundle,
            record_index=index,
            donor_plan=donor_plans[checkpoint["query_resource_key"]],
            allowed_authority_sha256=allowed_authorities,
            shape=shape,
        )
        summary["checkpoint_path"] = str(checkpoint_path.relative_to(ROOT))
        summary["checkpoint_sha256"] = checkpoint_sha
        replay.append(summary)
        observed_authorities.append(checkpoint["authority_sha256"])

    if shard_index in (1, 3):
        require(observed_authorities == [v1_sha, v1_sha, v1_sha, v2_sha], "MIXED_CHECKPOINT_AUTHORITY_DISTRIBUTION_DRIFT")
        require(result.get("checkpoint_authority_sha256_set") == sorted((v1_sha, v2_sha)), "MIXED_RESULT_AUTHORITY_SET_DRIFT")
        require(result.get("accepted_predecessor_checkpoint_count") == 3, "MIXED_PREDECESSOR_COUNT_DRIFT")
    else:
        require(observed_authorities == [v1_sha] * shape.query_count, "V1_CHECKPOINT_AUTHORITY_DISTRIBUTION_DRIFT")

    checks = {
        "v1_v2_authority_lineage_bound": True,
        "shard_result_and_source_bound": True,
        "checkpoint_physical_and_logical_hashes_bound": True,
        "mixed_checkpoint_authority_exact": True,
        "complete_anonymous_c128_axes": True,
        "opaque_identity_disjoint_donor_plans_bound": True,
        "real_atom_banks_revalidated": True,
        "all_five_control_populations_rebuilt": True,
        "all_families_and_h0_rebuilt": True,
        "all_readouts_exactly_replayed": True,
        "all_family_receipts_exactly_replayed": True,
        "coordinate_controls_complete_nondegenerate_and_stratum_preserving": True,
        "zero_roma_forward": True,
        "zero_image_read": True,
        "zero_label_or_identity_read": True,
        "zero_training_update": True,
    }
    validation: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA,
        "status": VALIDATION_STATUS,
        "claim_level": "ANONYMOUS_FOUR_QUERY_PREJOIN_ZERO_FORWARD_INDEPENDENT_REPLAY_ONLY",
        "checks": checks,
        "shard_index": shard_index,
        "query_count": shape.query_count,
        "candidate_count_per_query": shape.candidate_count,
        "candidate_occurrence_count": shape.query_count * shape.candidate_count,
        "replayed_family_count": sum(item["replayed_family_count"] for item in replay),
        "replayed_similarity_matrix_count": sum(item["replayed_similarity_matrix_count"] for item in replay),
        "query_replay": replay,
        "checkpoint_authority_sha256_counts": {
            key: observed_authorities.count(key) for key in sorted(set(observed_authorities))
        },
        "authority_v1_sha256": v1_sha,
        "authority_v2_sha256": v2_sha,
        "replay_authority_sha256": replay_authority_sha,
        "shard_result_path": str(result_path.resolve().relative_to(ROOT)),
        "shard_result_sha256": sha256_file(result_path),
        "shard_result_logical_sha256": result["logical_sha256"],
        "donor_plan_sha256": donor_sha,
        "donor_validation_sha256": donor_validation_sha,
        "validator_model_load_count": 0,
        "validator_roma_forward_count": 0,
        "validator_image_read_count": 0,
        "validator_label_or_identity_read_count": 0,
        "training_update_count": 0,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
    }
    validation["logical_sha256"] = logical_sha256(validation)
    if output_path.exists():
        existing = _read_json(output_path)
        require(existing == validation, "EXISTING_VALIDATION_DRIFT")
    else:
        _atomic_json(output_path, validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-authority", type=Path, required=True)
    parser.add_argument("--authority-v1", type=Path, required=True)
    parser.add_argument("--authority-v2", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--donor-plans", type=Path, required=True)
    parser.add_argument("--donor-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.shard_index in range(8), "SHARD_INDEX_INVALID")
    result = validate_shard(
        replay_authority_path=args.replay_authority.resolve(),
        authority_v1_path=args.authority_v1.resolve(),
        authority_v2_path=args.authority_v2.resolve(),
        shard_root=args.shard_root.resolve(),
        shard_index=args.shard_index,
        donor_plan_path=args.donor_plans.resolve(),
        donor_validation_path=args.donor_validation.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps({"status": result["status"], "shard_index": result["shard_index"], "checks": result["checks"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
