"""Shared contracts for N2 current-runtime fold-local D1 training.

This module contains no target lookup and no optimizer loop.  It centralizes
the immutable predecessor, corrected-gallery, cache, hashing and publication
rules used by the target-free prejoin and the later target-bearing trainer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import torch

from rc_aslo_xf.n2_corrected_d1_runtime_v1 import (
    CANONICAL_FOLD_COUNTS,
    CANONICAL_QUERY_COUNT,
    corrected_labels_from_legacy,
    validate_canonical_987_ledger,
    validate_corrected_identity_axis,
)


VERSION = "routea_n2_current_runtime_fold_local_d1_training_v1_20260903"
PREJOIN_SHARD_STATUS = "ROUTEA_N2_D1_RAW_PREJOIN_SHARD_READY"
PREJOIN_SHARD_VALID_STATUS = "ROUTEA_N2_D1_RAW_PREJOIN_SHARD_VALIDATED"
PREJOIN_AGGREGATE_VALID_STATUS = "ROUTEA_N2_D1_RAW_PREJOIN_AGGREGATE_VALIDATED"
TRAIN_STATUS = "ROUTEA_N2_CURRENT_RUNTIME_D1_FOLD_TRAINING_COMPLETE"
TRAIN_VALID_STATUS = "ROUTEA_N2_CURRENT_RUNTIME_D1_FOLD_TRAINING_VALIDATED"
TOKEN_AGGREGATE_STATUS = "ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_AGGREGATE_READY"
TOKEN_VALID_STATUS = "ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_INDEPENDENT_VALIDATION_PASS"
NEXT_TOKEN_STAGE = "N2_CURRENT_RUNTIME_FOLD_LOCAL_D1_TRAINING_CONTRACT"

QUERY_COUNT = CANONICAL_QUERY_COUNT
PHYSICAL_ROWS = 5_413
CORRECTED_IDENTITIES = 5_412
SHARD_COUNT = 16
FORMAL_FOLDS = (1, 2, 3, 4)
OPTIONAL_FOLDS = (0,)
STEPS = 800
CHECKPOINT_INTERVAL = 25
SEED = 17
FOLD_SEED_STRIDE = 1009
PARAMETER_COUNT = 49_792
NEGATIVE_COUNT = 63
CORRECTED_ELIGIBLE_IDENTITY_COUNTS = {0: 5_314, 1: 5_325, 2: 5_319, 3: 5_300, 4: 5_330}
EXPECTED_PARAMETER_SCHEMA = (
    ("norm.weight", (128,)),
    ("norm.bias", (128,)),
    ("fc1.weight", (128, 128)),
    ("fc1.bias", (128,)),
    ("fc2.weight", (128, 128)),
    ("fc2.bias", (128,)),
    ("delta_head.weight", (128, 128)),
    ("delta_head.bias", (128,)),
)
EXPECTED_PARAMETER_SCHEMA_SHA256 = "8594609ae4028a220fe3720a73545e1e6e654b79dda0ba0e39a78ea2541473f1"
EXPECTED_ALGORITHM_DEPENDENCY_HASHES = {
    "d1_core_sha256": "a3973e45f97f2157883acb96b64c614d9fa93ec80187e6608f462fe841737274",
    "domain_alignment_sha256": "2967270e1959966413f64902d07425e543e1bead72e0f08097b1f443a5a0ebe7",
    "domain_safe_sha256": "53bd5436ac8a8992bf244ad422be217eb355330cc38d3479b1d83d60a674cea2",
    "maxsim_runtime_sha256": "a8e267a9c033b3d6a116b2445e34614eb5f7f173df90d44952bf7a6befd054df",
    "corrected_d1_runtime_sha256": "fb0cb8f1fdeefc66aaaa587529a66b02a25331ceab706c6c05be8572da0a8c6b",
    "gallery_identity_repair_runtime_sha256": "995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d",
    "gallery_identity_repair_registry_sha256": "9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f",
    "gallery_identity_repair_contract_sha256": "867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650",
}
EXPECTED_LABEL_SOURCE_HASHES = {
    "outcome_manifest_sha256": "c9ccf836c94b84bf4097be8e59ad74e6bd606ec882af33cadfddafaa52917b10",
    "difficult_manifest_sha256": "9d5a066c3fac9fb792ba7d3ef50dd6379cd4c41c592ecb7eba8a8c7669a32a56",
    "new_difficult_manifest_sha256": "279d6db5b554ed6ed5df6f2a806a62ca8aa19e927506d70f64f36e53be1db538",
    "upstreams_sha256": "6f0399baa46a0c774a7214486467320eb0e778fe91cefd2b2010bc7512f75eac",
    "split_preflight_sha256": "c4bbafcbf9ba4c3332fd91b15fa742e72ac07611596aca52a8b50725a72d66ab",
}

FORBIDDEN_PARAMETER_FRAGMENTS = (
    "identity", "gallery", "candidate", "reference", "row", "slot",
    "embedding_table", "lookup_table",
)
FORBIDDEN_PREJOIN_FIELDS = (
    "identity", "target", "supergroup", "group_id", "target_row",
    "correct", "winner", "challenger", "negative", "outcome", "action",
)


class N2D1ContractError(RuntimeError):
    """A fail-closed N2 D1 lineage, split, cache or training error."""


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "logical_sha256"}
    return canonical_sha256(body)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _algorithm_dependency_paths(root: Path) -> dict[str, Path]:
    routea = root.parent
    return {
        "d1_core_sha256": routea / "route_a_core/route_a/o1_c6direct_m1_d1.py",
        "domain_alignment_sha256": routea / "route_a_core/route_a/domain_alignment.py",
        "domain_safe_sha256": routea / "route_a_core/route_a/domain_safe.py",
        "maxsim_runtime_sha256": routea / "route_a_core/route_a/o1_c6direct_m1_runtime.py",
        "corrected_d1_runtime_sha256": root / "src/rc_aslo_xf/n2_corrected_d1_runtime_v1.py",
        "gallery_identity_repair_runtime_sha256": root / "src/rc_aslo_xf/gallery_identity_repair.py",
        "gallery_identity_repair_registry_sha256": root / "registry/gallery_identity_repair_v1.json",
        "gallery_identity_repair_contract_sha256": root / "protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json",
    }


def _selected_dependency_hashes(root: Path, roles: set[str]) -> dict[str, str]:
    dependencies = _algorithm_dependency_paths(root)
    require(roles <= set(dependencies), "unknown algorithm dependency role")
    selected = {role: dependencies[role] for role in sorted(roles)}
    for path in selected.values():
        require_regular(path, "algorithm dependency")
    observed = {role: sha256_file(path) for role, path in selected.items()}
    require(
        observed == {role: EXPECTED_ALGORITHM_DEPENDENCY_HASHES[role] for role in sorted(roles)},
        "frozen algorithm dependency drift",
    )
    return observed


def algorithm_dependency_hashes(root: Path) -> dict[str, str]:
    return _selected_dependency_hashes(root, set(EXPECTED_ALGORITHM_DEPENDENCY_HASHES))


def raw_prejoin_dependency_hashes(root: Path) -> dict[str, str]:
    roles = {
        "maxsim_runtime_sha256",
        "corrected_d1_runtime_sha256",
        "gallery_identity_repair_runtime_sha256",
        "gallery_identity_repair_registry_sha256",
        "gallery_identity_repair_contract_sha256",
    }
    return _selected_dependency_hashes(root, roles)


def label_join_dependency_hashes(root: Path) -> dict[str, str]:
    roles = {
        "corrected_d1_runtime_sha256",
        "gallery_identity_repair_runtime_sha256",
        "gallery_identity_repair_registry_sha256",
        "gallery_identity_repair_contract_sha256",
    }
    return _selected_dependency_hashes(root, roles)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(tensor_sha256(state[name]).encode("ascii"))
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise N2D1ContractError(message)


def require_regular(path: Path, role: str) -> None:
    require(path.exists() and path.is_file() and not path.is_symlink(), f"{role} absent or non-regular: {path}")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f"refusing symlink output: {path}")
    if path.exists():
        require(json.loads(path.read_text()) == dict(value), f"immutable JSON drift: {path}")
        return
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f"refusing symlink output: {path}")
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        require(_exact_object(existing, value), f"immutable torch artifact drift: {path}")
        return
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _exact_object(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_exact_object(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)  # type: ignore[arg-type]
            and all(_exact_object(a, b) for a, b in zip(left, right))  # type: ignore[arg-type]
        )
    return left == right


def validate_token_authority(root: Path) -> dict[str, str]:
    aggregate_path = root / "results/routea_matched_three_arm_n2_token_cache_v1/aggregate.json"
    validation_path = root / "results/routea_matched_three_arm_n2_token_cache_v1/independent_validation.json"
    require_regular(aggregate_path, "N2 token aggregate")
    require_regular(validation_path, "N2 token independent validation")
    aggregate = json.loads(aggregate_path.read_text())
    validation = json.loads(validation_path.read_text())
    aggregate_keys = {
        "schema_version", "status", "claim_level", "checks", "query_count",
        "mode_counts", "track_counts", "fold_counts", "grid_counts",
        "tensor_manifest_sha256", "query_id_ordinal_sha256",
        "shared_encoder_parameter_schema_sha256", "shared_encoder_parameter_count",
        "shared_d1_parameter_schema_sha256", "shared_d1_parameter_count", "shards",
        "bindings", "access", "scientific_GO_or_NO_GO", "n2_d1_training_authorized",
        "external_execution_authorized", "next_authorized_stage", "logical_sha256",
    }
    validation_keys = {
        "schema_version", "status", "claim_level", "checks", "query_count",
        "mode_counts", "fresh_forward_audit", "aggregate_sha256",
        "shared_encoder_parameter_schema_sha256", "shared_encoder_parameter_count",
        "shared_d1_parameter_schema_sha256", "shared_d1_parameter_count",
        "reference_defined_model_boundary", "access",
        "runtime_diagnostics_not_in_model_identity", "scientific_GO_or_NO_GO",
        "n2_d1_training_authorized", "external_execution_authorized",
        "next_authorized_stage", "logical_sha256",
    }
    require(
        set(aggregate) == aggregate_keys
        and aggregate.get("status") == TOKEN_AGGREGATE_STATUS
        and aggregate.get("query_count") == QUERY_COUNT
        and isinstance(aggregate.get("checks"), Mapping)
        and bool(aggregate["checks"])
        and all(value is True for value in aggregate["checks"].values())
        and isinstance(aggregate.get("shards"), list)
        and len(aggregate["shards"]) == SHARD_COUNT
        and [item.get("shard") for item in aggregate["shards"]] == list(range(SHARD_COUNT))
        and aggregate.get("logical_sha256") == logical_sha256(aggregate),
        "N2 token aggregate is not a complete immutable authority",
    )
    require(
        set(validation) == validation_keys
        and validation.get("status") == TOKEN_VALID_STATUS
        and validation.get("query_count") == QUERY_COUNT
        and isinstance(validation.get("checks"), Mapping)
        and bool(validation["checks"])
        and all(value is True for value in validation["checks"].values())
        and validation.get("next_authorized_stage") == NEXT_TOKEN_STAGE
        and validation.get("logical_sha256") == logical_sha256(validation)
        and validation.get("aggregate_sha256") == sha256_file(aggregate_path),
        "N2 token independent validation does not authorize D1",
    )
    require(
        aggregate.get("shared_d1_parameter_count") == PARAMETER_COUNT
        and aggregate.get("shared_d1_parameter_schema_sha256") == EXPECTED_PARAMETER_SCHEMA_SHA256
        and validation.get("shared_d1_parameter_count") == PARAMETER_COUNT
        and validation.get("shared_d1_parameter_schema_sha256") == EXPECTED_PARAMETER_SCHEMA_SHA256
        and validation.get("reference_defined_model_boundary", {}).get(
            "per_identity_or_gallery_row_or_slot_parameter_count"
        )
        == 0
        and validation.get("reference_defined_model_boundary", {}).get(
            "new_reference_requires_parameter_update"
        )
        is False,
        "token authority shared reference-defined D1 schema drift",
    )
    for seal in aggregate["shards"]:
        shard = int(seal["shard"])
        base = root / f"results/routea_matched_three_arm_n2_token_cache_v1/shard{shard:02d}"
        require(
            seal.get("payload_sha256") == sha256_file(base / "payload.pt")
            and seal.get("receipt_sha256") == sha256_file(base / "receipt.json")
            and seal.get("validation_sha256") == sha256_file(base / "validation.json"),
            f"token aggregate shard seal drift at shard {shard}",
        )
    return {
        "token_aggregate_sha256": sha256_file(aggregate_path),
        "token_aggregate_logical_sha256": str(aggregate["logical_sha256"]),
        "token_validation_sha256": sha256_file(validation_path),
        "token_validation_logical_sha256": str(validation["logical_sha256"]),
        "token_query_id_ordinal_sha256": str(aggregate["query_id_ordinal_sha256"]),
    }


def sanitized_query_index(root: Path) -> list[dict[str, Any]]:
    """Reconstruct the opaque query axis without opening path-bearing ledger rows."""

    output: list[dict[str, Any]] = []
    for shard in range(SHARD_COUNT):
        payload = load_token_shard(root, shard)
        for record in payload["records"]:
            output.append(
                {
                    "query_id": str(record["query_id"]),
                    "query_ordinal": int(record["query_ordinal"]),
                    "heldout_fold": int(record["heldout_fold"]),
                    "track": str(record["track"]),
                    "grid_h": int(record["grid_shape"][0]),
                    "grid_w": int(record["grid_shape"][1]),
                    "source_image_sha256": str(record["source_image_sha256"]),
                }
            )
    validate_canonical_987_ledger(output)
    require([row["query_ordinal"] for row in output] == list(range(QUERY_COUNT)), "sanitized query order drift")
    return output


def expected_shard_interval(shard: int) -> range:
    require(type(shard) is int and 0 <= shard < SHARD_COUNT, "invalid shard")
    return range(QUERY_COUNT * shard // SHARD_COUNT, QUERY_COUNT * (shard + 1) // SHARD_COUNT)


def load_token_shard(root: Path, shard: int) -> dict[str, Any]:
    base = root / f"results/routea_matched_three_arm_n2_token_cache_v1/shard{shard:02d}"
    payload_path = base / "payload.pt"
    receipt_path = base / "receipt.json"
    validation_path = base / "validation.json"
    for path, role in ((payload_path, "token payload"), (receipt_path, "token receipt"), (validation_path, "token validation")):
        require_regular(path, role)
    receipt = json.loads(receipt_path.read_text())
    validation = json.loads(validation_path.read_text())
    receipt_keys = {
        "schema_version", "status", "shard", "query_count",
        "query_ordinal_begin", "query_ordinal_end_exclusive", "mode_counts",
        "payload_sha256", "model_update_count", "next_authorized_stage",
        "logical_sha256",
    }
    validation_keys = {
        "schema_version", "status", "claim_level", "shard", "checks",
        "query_count", "reuse_count", "fresh_count", "payload_sha256",
        "receipt_sha256", "model_update_count", "scientific_GO_or_NO_GO",
        "next_authorized_stage", "logical_sha256",
    }
    require(
        set(receipt) == receipt_keys
        and set(validation) == validation_keys
        and validation.get("status") == "ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_SHARD_VALIDATED"
        and isinstance(validation.get("checks"), Mapping)
        and bool(validation["checks"])
        and all(value is True for value in validation["checks"].values())
        and validation.get("payload_sha256") == sha256_file(payload_path)
        and validation.get("receipt_sha256") == sha256_file(receipt_path)
        and validation.get("logical_sha256") == logical_sha256(validation),
        f"token shard {shard} is not independently validated",
    )
    payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
    records = payload.get("records", []) if isinstance(payload, Mapping) else []
    expected = list(expected_shard_interval(shard))
    payload_keys = {
        "schema_version", "status", "claim_level", "shard", "shard_count",
        "query_ordinal_begin", "query_ordinal_end_exclusive", "records",
        "model", "bindings", "access",
    }
    model_keys = {
        "stable_encoder_fingerprint_sha256", "encoder_parameter_schema_sha256",
        "encoder_parameter_count", "encoder_trainable_parameter_count",
        "d1_checkpoint_load_count", "d1_model_update_count",
        "hardware_identity_is_authority",
    }
    record_keys = {
        "query_id", "query_ordinal", "heldout_fold", "track",
        "source_image_sha256", "source_exif_orientation", "decode_frame",
        "raw_size_hw", "oriented_size_hw", "grid_shape", "image_tokens",
        "image_tokens_sha256", "template_tokens", "template_tokens_sha256",
        "materialization_mode", "current64_source_shard",
        "current64_source_payload_sha256", "model_update_count",
    }
    require(
        set(payload) == payload_keys
        and set(payload.get("model", {})) == model_keys
        and bool(records)
        and all(set(record) == record_keys for record in records)
        and [int(record.get("query_ordinal", -1)) for record in records] == expected
        and payload.get("status") == "ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_SHARD_READY"
        and receipt.get("payload_sha256") == sha256_file(payload_path),
        f"token shard {shard} population or envelope drift",
    )
    return dict(payload)


def validate_prejoin_record(record: Mapping[str, Any]) -> None:
    required = {
        "query_id", "query_ordinal", "heldout_fold", "track",
        "physical_row_scores", "physical_row_scores_sha256",
        "image_tokens_sha256", "template_tokens_sha256",
    }
    require(set(record) == required, "RAW prejoin record schema drift")
    lowered = {str(key).lower() for key in record}
    require(
        not any(fragment in key for key in lowered for fragment in FORBIDDEN_PREJOIN_FIELDS),
        "RAW prejoin contains a protected semantic field",
    )
    scores = record["physical_row_scores"]
    require(
        isinstance(scores, torch.Tensor)
        and scores.dtype == torch.float32
        and scores.shape == (PHYSICAL_ROWS,)
        and bool(torch.isfinite(scores).all())
        and tensor_sha256(scores) == record["physical_row_scores_sha256"],
        "RAW prejoin score tensor drift",
    )


def load_gallery(root: Path) -> tuple[list[torch.Tensor], tuple[str, ...], dict[str, Any]]:
    gallery_path = root.parents[2] / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
    require_regular(gallery_path, "frozen full gallery")
    require(
        sha256_file(gallery_path) == "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc",
        "full gallery hash drift",
    )
    payload = torch.load(gallery_path, map_location="cpu", weights_only=False, mmap=True)
    legacy = list(map(str, payload.get("setids", [])))
    references = list(payload.get("passage_emb", []))
    corrected = corrected_labels_from_legacy(legacy)
    axis = validate_corrected_identity_axis(corrected)
    require(len(references) == PHYSICAL_ROWS, "gallery reference population drift")
    require(
        all(
            isinstance(value, torch.Tensor)
            and value.ndim == 2
            and value.shape[0] > 0
            and value.shape[1] == 128
            and value.is_floating_point()
            and bool(torch.isfinite(value).all())
            for value in references
        ),
        "gallery reference token schema drift",
    )
    return references, corrected, {
        "gallery_sha256": sha256_file(gallery_path),
        "corrected_axis_sha256": axis["corrected_axis_sha256"],
        "physical_row_count": PHYSICAL_ROWS,
        "corrected_identity_count": CORRECTED_IDENTITIES,
    }


def parameter_manifest(module: torch.nn.Module) -> list[dict[str, Any]]:
    output = []
    for name, parameter in module.named_parameters():
        lowered = name.lower()
        require(
            not any(fragment in lowered for fragment in FORBIDDEN_PARAMETER_FRAGMENTS),
            f"identity/gallery-specific parameter forbidden: {name}",
        )
        output.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": int(parameter.numel()),
            }
        )
    observed = tuple((item["name"], tuple(item["shape"])) for item in output)
    schema_for_hash = [
        {"name": item["name"], "shape": item["shape"], "dtype": item["dtype"]}
        for item in output
    ]
    require(observed == EXPECTED_PARAMETER_SCHEMA, "D1 exact eight-key parameter schema drift")
    require(sum(item["numel"] for item in output) == PARAMETER_COUNT, "D1 parameter count drift")
    require(canonical_sha256(schema_for_hash) == EXPECTED_PARAMETER_SCHEMA_SHA256, "D1 parameter schema hash drift")
    return output


def corrected_mask_audit(labels: Sequence[str], legal: torch.Tensor) -> dict[str, Any]:
    validate_corrected_identity_axis(labels)
    require(legal.dtype == torch.bool and legal.shape == (PHYSICAL_ROWS,), "legal mask schema drift")
    states: dict[str, set[bool]] = defaultdict(set)
    for identity, keep in zip(labels, legal.tolist()):
        states[str(identity)].add(bool(keep))
    require(all(len(values) == 1 for values in states.values()), "legal mask splits a corrected identity")
    return {
        "eligible_physical_rows": int(legal.sum().item()),
        "excluded_physical_rows": PHYSICAL_ROWS - int(legal.sum().item()),
        "eligible_corrected_identities": sum(next(iter(values)) for values in states.values()),
        "excluded_corrected_identities": sum(not next(iter(values)) for values in states.values()),
        "legal_rows_sha256": canonical_sha256(torch.nonzero(legal).flatten().tolist()),
    }


def select_corrected_top63(
    physical_scores: torch.Tensor,
    corrected_labels: Sequence[str],
    legal: torch.Tensor,
    *,
    target_identity: str,
) -> torch.Tensor:
    """Select negatives only after corrected-identity max and lower-row ties.

    The input scores are already sealed target-free.  The target identity is
    consulted only to remove its complete corrected component after reduction.
    """

    require(
        isinstance(physical_scores, torch.Tensor)
        and physical_scores.dtype == torch.float32
        and physical_scores.shape == (PHYSICAL_ROWS,)
        and bool(torch.isfinite(physical_scores).all()),
        "negative selection requires finite float32 [5413] RAW scores",
    )
    labels = tuple(map(str, corrected_labels))
    audit = corrected_mask_audit(labels, legal)
    require(target_identity in labels, "training target absent from corrected gallery")
    target_rows = [row for row, identity in enumerate(labels) if identity == target_identity]
    require(target_rows and all(bool(legal[row]) for row in target_rows), "training target is not legal in this fold")
    # Stable descending sort over the canonical ascending physical-row axis is
    # exactly: per-identity max, lower-row tie break, then score-desc/row-asc.
    # The first encountered row for each corrected identity is its reduced
    # representative.  This avoids constructing 5,412 tiny tensors per query.
    ranked_rows = torch.argsort(physical_scores, descending=True, stable=True).tolist()
    seen: set[str] = set()
    selected: list[int] = []
    for row in ranked_rows:
        if not bool(legal[row]):
            continue
        identity = labels[row]
        if identity in seen:
            continue
        seen.add(identity)
        if identity == target_identity:
            continue
        selected.append(int(row))
        if len(selected) == NEGATIVE_COUNT:
            break
    require(len(selected) == NEGATIVE_COUNT, "fewer than 63 legal corrected negative identities")
    require(len({labels[row] for row in selected}) == NEGATIVE_COUNT, "negative identity uniqueness drift")
    require(target_identity not in {labels[row] for row in selected}, "target leaked into negatives")
    require(
        audit["eligible_corrected_identities"] >= NEGATIVE_COUNT + 1,
        "legal corrected gallery too small",
    )
    return torch.tensor(selected, dtype=torch.long)


def fold_seed(fold: int) -> int:
    require(fold in FORMAL_FOLDS + OPTIONAL_FOLDS, "fold must be in 0..4")
    return SEED + FOLD_SEED_STRIDE * fold


def counter(values: Iterable[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


__all__ = [
    "CHECKPOINT_INTERVAL", "CORRECTED_IDENTITIES", "EXPECTED_ALGORITHM_DEPENDENCY_HASHES", "EXPECTED_PARAMETER_SCHEMA",
    "EXPECTED_PARAMETER_SCHEMA_SHA256", "EXPECTED_LABEL_SOURCE_HASHES", "FORMAL_FOLDS",
    "NEGATIVE_COUNT", "N2D1ContractError", "OPTIONAL_FOLDS",
    "PARAMETER_COUNT", "PHYSICAL_ROWS", "PREJOIN_AGGREGATE_VALID_STATUS",
    "PREJOIN_SHARD_STATUS", "PREJOIN_SHARD_VALID_STATUS", "QUERY_COUNT",
    "SEED", "SHARD_COUNT", "STEPS", "TRAIN_STATUS", "TRAIN_VALID_STATUS",
    "VERSION", "algorithm_dependency_hashes", "raw_prejoin_dependency_hashes",
    "label_join_dependency_hashes", "atomic_json", "atomic_torch", "canonical_sha256", "corrected_mask_audit",
    "CORRECTED_ELIGIBLE_IDENTITY_COUNTS", "counter", "expected_shard_interval", "fold_seed", "load_gallery",
    "load_token_shard", "logical_sha256", "parameter_manifest", "sanitized_query_index",
    "require", "require_regular", "sha256_file", "state_dict_sha256",
    "select_corrected_top63", "tensor_sha256", "validate_prejoin_record", "validate_token_authority",
]
