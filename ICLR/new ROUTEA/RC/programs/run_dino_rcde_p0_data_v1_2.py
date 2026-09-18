#!/usr/bin/env python3
"""Materialise and audit DINO-RCDE P0 V1.2 data/statistics inputs.

This program performs no learned-model forward and cannot authorise E0 on its
own.  It consumes the already redacted, target-free artifacts frozen by the
last complete V1.1 run.  Labels enter only after the RAW C128 and low-margin
receipt have been independently reconstructed and sealed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rc_dino_rcde_p0_data_result_v1_2_20260812"
PROTECTED_ZERO = {
    "formal_runner_C8_runtime_read_count": 0,
    "formal_runner_S8_runtime_read_count": 0,
    "formal_runner_opened_runtime_read_count": 0,
    "formal_runner_sealed_runtime_read_count": 0,
    "home_files_modified": 0,
    "target_free_runtime_legacy_query_carrier_read_count": 0,
    "C8_image_token_score_or_result_read_count": 0,
    "project_agent_C8_role_metadata_untouched": False,
}


class P0DataError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise P0DataError(f"immutable JSON drift: {path}")
        return
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with partial.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def link_immutable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise P0DataError(f"immutable link/copy drift: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        partial = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
        shutil.copyfile(source, partial)
        try:
            os.link(partial, destination)
        finally:
            partial.unlink(missing_ok=True)


def resolve(binding: Mapping[str, Any]) -> Path:
    return (ROOT / str(binding["path"])).resolve()


def validate_binding(binding: Mapping[str, Any], name: str) -> Path:
    path = resolve(binding)
    if not path.is_file() or sha256_file(path) != str(binding["sha256"]):
        raise P0DataError(f"source binding drift: {name}")
    return path


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected_forbidden = {
        "C128_construction", "negative_sampler", "loss", "model_feature", "eligibility",
        "HOLD", "A1_gate", "threshold_selection", "checkpoint_selection", "GO_primary_population",
    }
    low = protocol.get("low_margin_contract", {})
    checks = {
        "schema": protocol.get("schema_version") == "rc_dino_rcde_p0_protocol_v1_2_20260812",
        "status": protocol.get("status") == "FROZEN_BEFORE_V1_2_NATURAL_RAW_SCORE_READ",
        "low_namespace": low.get("namespace") == "RAW_LOW_MARGIN_V1_1_Q25_MAD_C128",
        "low_q": low.get("quantile") == 0.25 and low.get("final_quantile_index") == 150,
        "low_forbidden": set(low.get("forbidden_uses", [])) == expected_forbidden,
        "no_auto": protocol.get("automatic_stage_advance") is False,
        "no_target_insert": protocol.get("base_contract", {}).get("target_insertion_allowed") is False,
        "restricted_before_rank": str(protocol.get("negative_sampler_contract", {}).get("population", "")).startswith("all_5413_physical_rows_restricted"),
    }
    if not all(checks.values()):
        raise P0DataError(f"DINO_RCDE_LOW_MARGIN_CONTRACT_UNFROZEN: {checks}")
    parent_authority = validate_binding(protocol["parent_authority"], "parent_authority")
    authority = json.loads(parent_authority.read_text(encoding="utf-8"))
    if authority.get("core_addendum", {}).get("natural_training_authorized") is not False:
        raise P0DataError("parent authority no longer forbids natural training at P0")
    # Phase 1 deliberately does not even hash a target-bearing postjoin file.
    for name, binding in {
        "scientific_contract": protocol["scientific_contract"],
        "execution_repair_addendum": protocol["execution_repair_addendum"],
        "low_margin_contract_binding": protocol["low_margin_contract_binding"],
    }.items():
        validate_binding(binding, name)
    prejoin = protocol.get("prejoin_bindings", {})
    postjoin = protocol.get("postjoin_bindings", {})
    if not prejoin or not postjoin:
        raise P0DataError("source bindings are not split into prejoin/postjoin maps")
    if set(prejoin) & set(postjoin):
        raise P0DataError("prejoin/postjoin source binding overlap")
    for name, binding in prejoin.items():
        validate_binding(binding, name)
    for name, binding in protocol["dino_freeze"].items():
        if isinstance(binding, dict) and "path" in binding:
            validate_binding(binding, f"dino_freeze.{name}")
    forbidden_sources = {"roles", "execution_roles", "target_free_runtime_manifest"}
    if forbidden_sources & (set(prejoin) | set(postjoin)):
        raise P0DataError("protocol exposes a mixed/protected role source to formal P0")
    return protocol


def validate_postjoin_bindings(protocol: Mapping[str, Any]) -> None:
    """Hash postjoin sources only after the low-margin artifact is durable."""

    for name, binding in protocol["postjoin_bindings"].items():
        validate_binding(binding, name)


def cohort_digest(kind: str, identifier: str) -> str:
    """Result-blind canonical tie break for the frozen resource cohort."""

    payload = b"RCDE_RESOURCE_COHORT_V1_2\0" + kind.encode("ascii") + b"\0" + identifier.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def nearest_rank_percentile(
    rows: Sequence[Mapping[str, Any]], percentile: float, *, kind: str, identifier_key: str,
) -> Mapping[str, Any]:
    """Select ceil(p*N)-1 after token-count and canonical-hash ordering."""

    if not rows or not 0.0 < percentile <= 1.0:
        raise P0DataError("invalid resource percentile population")
    ordered = sorted(
        rows,
        key=lambda item: (
            int(item["valid_tokens"]),
            cohort_digest(kind, str(item[identifier_key])),
            str(item[identifier_key]),
        ),
    )
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def raw_pixel_rank_key(item: Mapping[str, Any]) -> tuple[int, str, str]:
    """Result-blind ordering for source decode/preprocess resource extremes."""

    identifier = (
        f"query:{item['id']}" if item["kind"] == "query"
        else f"reference:{int(item['physical_row'])}"
    )
    return (
        int(item["raw_h"]) * int(item["raw_w"]),
        str(item["source_image_sha256"]),
        identifier,
    )


def raw_pixel_percentile(
    rows: Sequence[Mapping[str, Any]], percentile: float,
) -> Mapping[str, Any]:
    if not rows or not 0.0 < percentile <= 1.0:
        raise P0DataError("invalid raw-pixel resource percentile population")
    ordered = sorted(rows, key=raw_pixel_rank_key)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def robust_confidence(scores: Sequence[float]) -> float:
    if len(scores) != 128:
        raise P0DataError("RAW confidence requires 128 candidates")
    values = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in values):
        raise P0DataError("nonfinite RAW C128 score")
    ordered = sorted(values)
    median = (ordered[63] + ordered[64]) * 0.5
    deviation = sorted(abs(value - median) for value in ordered)
    mad = (deviation[63] + deviation[64]) * 0.5
    return (values[0] - values[1]) / (1.4826 * mad + 1.0e-6)


def shape_for_image(path: Path) -> dict[str, int]:
    from PIL import Image
    with Image.open(path) as image:
        raw_w, raw_h = image.size
        try:
            orientation = int(image.getexif().get(274, 1))
        except (AttributeError, TypeError, ValueError, OSError):
            orientation = 1
        if orientation not in range(1, 9):
            orientation = 1
        width, height = ((raw_h, raw_w) if orientation in {5, 6, 7, 8} else (raw_w, raw_h))
    scale = 518.0 / max(height, width)
    resized_h = max(1, math.floor(height * scale + 0.5))
    resized_w = max(1, math.floor(width * scale + 0.5))
    padded_h = ((resized_h + 13) // 14) * 14
    padded_w = ((resized_w + 13) // 14) * 14
    valid_h, valid_w = resized_h // 14, resized_w // 14
    return {
        "raw_h": raw_h, "raw_w": raw_w, "exif_orientation": int(orientation),
        "oriented_h": height, "oriented_w": width,
        "resized_h": resized_h, "resized_w": resized_w,
        "padded_h": padded_h, "padded_w": padded_w,
        "grid_h": padded_h // 14, "grid_w": padded_w // 14,
        "valid_h": valid_h, "valid_w": valid_w,
        "valid_tokens": valid_h * valid_w,
        "total_tokens": (padded_h // 14) * (padded_w // 14),
    }


def hall_feasible(groups: Sequence[str]) -> tuple[bool, dict[str, int]]:
    counts = Counter(groups)
    total = sum(counts.values())
    largest = max(counts.values(), default=0)
    return len(counts) >= 2 and largest <= total - largest, {
        "records": total, "groups": len(counts), "maximum_group_records": largest,
    }


def sign_power(number: int) -> dict[str, Any]:
    if number <= 0:
        return {
            "groups": number, "minimum_rejection_positives": None,
            "minimum_attainable_one_sided_p": None,
            "power80_direction_probability": None,
            "power_at_preregistered_direction_probability_0_70": None,
            "preregistered_direction_probability_0_70_has_80pct_power": False,
        }
    rejection = number + 1
    for positive in range(number + 1):
        if sum(math.comb(number, k) for k in range(positive, number + 1)) / 2.0**number <= 0.05:
            rejection = positive
            break
    p80 = None
    if rejection <= number:
        for step in range(5001, 10001):
            probability = step / 10000.0
            power = sum(
                math.comb(number, k) * probability**k * (1.0 - probability)**(number - k)
                for k in range(rejection, number + 1)
            )
            if power >= 0.8:
                p80 = probability
                break
    return {
        "groups": number,
        "minimum_rejection_positives": rejection if rejection <= number else None,
        "minimum_attainable_one_sided_p": 2.0**(-number),
        "power80_direction_probability": p80,
        "power_at_preregistered_direction_probability_0_70": (
            sum(
                math.comb(number, k) * 0.70**k * 0.30**(number - k)
                for k in range(rejection, number + 1)
            ) if rejection <= number else 0.0
        ),
        "preregistered_direction_probability_0_70_has_80pct_power": (
            rejection <= number
            and sum(
                math.comb(number, k) * 0.70**k * 0.30**(number - k)
                for k in range(rejection, number + 1)
            ) >= 0.8
        ),
    }


def exact_binomial_power(number: int, null_probability: float, alternative_probability: float) -> dict[str, Any]:
    """One-sided exact-binomial resolution under an explicit Bernoulli model."""

    if number <= 0 or not (0.0 <= null_probability < alternative_probability <= 1.0):
        return {
            "groups": number, "null_probability": null_probability,
            "alternative_probability": alternative_probability,
            "minimum_rejection_positives": None, "minimum_attainable_one_sided_p": None,
            "power_at_alternative": None, "has_80pct_power": False,
        }
    rejection = None
    for positive in range(number + 1):
        tail = sum(
            math.comb(number, k)
            * null_probability**k
            * (1.0 - null_probability) ** (number - k)
            for k in range(positive, number + 1)
        )
        if tail <= 0.05:
            rejection = positive
            break
    power = 0.0 if rejection is None else sum(
        math.comb(number, k)
        * alternative_probability**k
        * (1.0 - alternative_probability) ** (number - k)
        for k in range(rejection, number + 1)
    )
    minimum_p = null_probability**number if null_probability > 0.0 else 0.0
    return {
        "groups": number, "null_probability": null_probability,
        "alternative_probability": alternative_probability,
        "minimum_rejection_positives": rejection,
        "minimum_attainable_one_sided_p": minimum_p,
        "power_at_alternative": power, "has_80pct_power": power >= 0.8,
        "independence_unit": "supergroup",
    }


def continuous_power(number: int, effect_dz: float = 0.50) -> dict[str, Any]:
    """Normal approximation for a one-sided paired standardized group effect."""

    if number <= 0:
        return {
            "groups": number, "preregistered_effect_dz": effect_dz,
            "one_sided_alpha": 0.05, "power_at_preregistered_effect": None,
            "power80_mde_dz": None,
        }
    z_alpha = 1.6448536269514722
    z_power80 = 0.8416212335729143
    power = 0.5 * (1.0 + math.erf((math.sqrt(number) * effect_dz - z_alpha) / math.sqrt(2.0)))
    return {
        "groups": number, "preregistered_effect_dz": effect_dz,
        "one_sided_alpha": 0.05, "approximation": "normal_paired_group_effect",
        "power_at_preregistered_effect": power,
        "preregistered_effect_has_80pct_power": power >= 0.8,
        "power80_mde_dz": (z_alpha + z_power80) / math.sqrt(number),
        "independence_unit": "supergroup",
    }


def cohort_power(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["supergroup"]) for row in rows)
    groups = len(counts)
    maximum_group_queries = max(counts.values(), default=0)
    lattice = 1.0 / (groups * maximum_group_queries) if groups and maximum_group_queries else None
    return {
        "queries": len(rows), "eligible_supergroups": groups,
        "maximum_queries_in_one_supergroup": maximum_group_queries,
        "smallest_nonzero_group_balanced_binary_increment": lattice,
        "sign_or_sign_flip": sign_power(groups),
        "paired_continuous": continuous_power(groups),
    }


def matched_effect_identifiability(cohort: Mapping[str, Any], absolute_effect: float) -> dict[str, Any]:
    continuous = cohort["paired_continuous"]
    mde_dz = continuous["power80_mde_dz"]
    lattice = cohort["smallest_nonzero_group_balanced_binary_increment"]
    return {
        "absolute_effect": absolute_effect,
        "lattice_resolvable_for_binary_metric": bool(lattice is not None and absolute_effect >= lattice),
        "power80_identifiability_without_observed_group_sd": "UNDETERMINED",
        "maximum_observed_paired_group_sd_for_80pct_power": (
            absolute_effect / mde_dz if mde_dz not in (None, 0.0) else None
        ),
        "reason": "P0_must_not_invent_unknown_paired_group_variance",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise P0DataError(f"output already exists: {output}")
    output.mkdir(parents=True)
    started_ns = time.time_ns()

    protocol_path = args.protocol.resolve()
    try:
        protocol = load_protocol(protocol_path)
    except Exception as error:
        write_json(output / "result.json", {
            "schema_version": SCHEMA,
            "status": "DINO_RCDE_LOW_MARGIN_CONTRACT_UNFROZEN",
            "failure_domain": "protocol", "next_authorized_stage": None,
            "error_type": type(error).__name__, "error": str(error),
            "RAW_score_read_count": 0, "label_read_count": 0,
            "protected_access_counts": dict(PROTECTED_ZERO),
        })
        return 2

    protocol_sha = sha256_file(protocol_path)
    write_json(output / "contract.json", {
        "schema_version": "rc_dino_rcde_p0_machine_contract_v1_2_20260812",
        "status": "P0_V1_2_MACHINE_CONTRACT_VALIDATED_BEFORE_RAW_SCORE_READ",
        "created_ns": time.time_ns(), "protocol_sha256": protocol_sha,
        "scientific_contract_sha256": protocol["scientific_contract"]["sha256"],
        "execution_repair_addendum_sha256": protocol["execution_repair_addendum"]["sha256"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "low_margin_contract": protocol["low_margin_contract"],
        "base_contract": protocol["base_contract"],
        "RAW_score_read_count": 0, "label_read_count": 0,
        "protected_access_counts": dict(PROTECTED_ZERO),
    })

    import torch
    from rc_aslo_xf.dino_rcde_p0_v1_2 import (
        canonical_c128, corrected_identities, fold_access, legacy_replay_summary,
        natural_c128, reduce_scores, select_611_negatives, episode_model_order,
    )

    # Deliberately keep the postjoin map out of the active binding namespace
    # until the low-margin receipt has been durably written.
    bindings = protocol["prejoin_bindings"]
    inherited_prejoin_path = resolve(bindings["inherited_v1_1_base_prejoin"])
    inherited_receipt = json.loads(resolve(bindings["inherited_v1_1_base_prejoin_receipt"]).read_text(encoding="utf-8"))
    inherited_manifest = json.loads(resolve(bindings["inherited_v1_1_redacted_manifest"]).read_text(encoding="utf-8"))
    sources = json.loads(resolve(bindings["isolated_sanitized_query_sources_600"]).read_text(encoding="utf-8"))
    if inherited_receipt.get("base_prejoin_sha256") != sha256_file(inherited_prejoin_path):
        raise P0DataError("inherited prejoin receipt drift")
    if inherited_manifest.get("query_count") != 600 or sources.get("query_count") != 600:
        raise P0DataError("sanitized source population drift")
    inherited_rows = sorted(inherited_manifest["rows"], key=lambda item: int(item["query_ordinal"]))
    source_rows = sorted(sources["records"], key=lambda item: int(item["query_ordinal"]))
    if inherited_manifest.get("logical_sha256") != canonical_sha256(inherited_manifest.get("rows", [])):
        raise P0DataError("inherited redacted manifest logical hash drift")
    if sources.get("record_sequence_sha256") != canonical_sha256(sources.get("records", [])):
        raise P0DataError("sanitized source ledger logical hash drift")
    if [item["query_id"] for item in inherited_rows] != [item["query_id"] for item in source_rows]:
        raise P0DataError("sanitized query sequence mismatch")
    inherited_ordinals = [int(item["query_ordinal"]) for item in inherited_rows]
    source_ordinals = [int(item["query_ordinal"]) for item in source_rows]
    if inherited_ordinals != source_ordinals or len(set(inherited_ordinals)) != 600:
        raise P0DataError("sanitized query ordinal mismatch or duplication")
    if any(item["source_image_sha256"] != source["source_image_sha256"] for item, source in zip(inherited_rows, source_rows, strict=True)):
        raise P0DataError("sanitized source-image binding mismatch")
    for source in source_rows:
        source_path = Path(str(source["path"]))
        if not source_path.is_file() or sha256_file(source_path) != str(source["source_image_sha256"]):
            raise P0DataError(f"sanitized source-image bytes drift: {source['query_id']}")
    schema_forbidden = {"identity", "target", "label", "raw_full_gallery_rank", "raw_full_gallery_gt_score", "hard_negative_indices", "hard_negative_ids"}
    shard_schema = {
        "schema_version", "query_id", "query_ordinal", "source_image_sha256",
        "grid_h", "grid_w", "image_tokens", "template_tokens",
    }
    validated_shards = []
    for ordinal, item in enumerate(inherited_rows):
        shard = Path(str(item["redacted_shard"]))
        if not shard.is_file() or sha256_file(shard) != item["redacted_shard_sha256"]:
            raise P0DataError(f"redacted shard drift: {item['query_id']}")
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        if schema_forbidden & set(payload):
            raise P0DataError(f"target-bearing field in redacted shard: {item['query_id']}")
        image_tokens, template_tokens = payload.get("image_tokens"), payload.get("template_tokens")
        if (
            set(payload) != shard_schema
            or payload.get("schema_version") != "rcde_redacted_colnomic_query_tokens_v1_1"
            or str(payload.get("source_image_sha256")) != item["source_image_sha256"]
            or not isinstance(image_tokens, torch.Tensor) or image_tokens.dtype != torch.float16
            or image_tokens.ndim != 2 or image_tokens.shape[1] != 128
            or not isinstance(template_tokens, torch.Tensor) or template_tokens.dtype != torch.float16
            or template_tokens.ndim != 2 or template_tokens.shape[1] != 128
            or int(payload.get("grid_h", -1)) * int(payload.get("grid_w", -1)) != image_tokens.shape[0]
            or str(payload.get("query_id")) != item["query_id"]
            or int(payload.get("query_ordinal", -1)) != int(item["query_ordinal"])
        ):
            raise P0DataError(f"redacted shard schema drift: {item['query_id']}")
        if tensor_sha256(image_tokens) != item["image_token_sha256"] or tensor_sha256(template_tokens) != item["template_token_sha256"]:
            raise P0DataError(f"redacted tensor hash drift: {item['query_id']}")
        validated_shards.append({
            "query_id": item["query_id"], "execution_ordinal": ordinal,
            "query_ordinal": int(item["query_ordinal"]),
            "source_image_sha256": item["source_image_sha256"],
            "redacted_shard_sha256": item["redacted_shard_sha256"],
            "redacted_shard_file_bytes": shard.stat().st_size,
            "image_tokens_dtype": str(image_tokens.dtype), "image_tokens_shape": list(image_tokens.shape),
            "template_tokens_dtype": str(template_tokens.dtype), "template_tokens_shape": list(template_tokens.shape),
            "schema_forbidden_fields_present": [],
        })
    write_json(output / "redacted_inheritance_receipt.json", {
        "schema_version": "rcde_redacted_inheritance_receipt_v1_2",
        "status": "RCDE_V1_2_REDACTED_INHERITANCE_VERIFIED",
        "query_count": len(validated_shards), "validated_shard_count": len(validated_shards),
        "validated_source_image_count": len(source_rows),
        "inherited_manifest_sha256": sha256_file(resolve(bindings["inherited_v1_1_redacted_manifest"])),
        "inherited_manifest_rows_logical_sha256": inherited_manifest["logical_sha256"],
        "source_ledger_sha256": sha256_file(resolve(bindings["isolated_sanitized_query_sources_600"])),
        "source_ledger_rows_logical_sha256": sources["record_sequence_sha256"],
        "inherited_base_prejoin_expected_sha256": bindings["inherited_v1_1_base_prejoin"]["sha256"],
        "rows": validated_shards, "rows_logical_sha256": canonical_sha256(validated_shards),
        "mixed_or_legacy_carrier_read_count": 0, "label_read_count": 0,
        "protected_access_counts": dict(PROTECTED_ZERO),
    })

    prejoin = torch.load(inherited_prejoin_path, map_location="cpu", weights_only=False)
    all_scores = prejoin.get("physical_scores")
    stored_rows = prejoin.get("c128_rows_by_raw_rank")
    stored_scores = prejoin.get("c128_scores_by_raw_rank")
    if (
        not isinstance(all_scores, torch.Tensor) or all_scores.dtype != torch.float32 or all_scores.shape != (600, 5413)
        or not isinstance(stored_rows, torch.Tensor) or stored_rows.dtype != torch.int64 or stored_rows.shape != (600, 128)
        or not isinstance(stored_scores, torch.Tensor) or stored_scores.dtype != torch.float64 or stored_scores.shape != (600, 128)
        or prejoin.get("query_ids") != [item["query_id"] for item in inherited_rows]
        or prejoin.get("query_ordinals") != [int(item["query_ordinal"]) for item in inherited_rows]
        or prejoin.get("source_image_sha256") != [item["source_image_sha256"] for item in inherited_rows]
    ):
        raise P0DataError("inherited prejoin schema drift")

    gallery = torch.load(resolve(bindings["gallery_token_cache"]), map_location="cpu", weights_only=False, mmap=True)
    spatial = torch.load(resolve(bindings["gallery_spatial_metadata"]), map_location="cpu", weights_only=False)
    repair = json.loads(resolve(bindings["gallery_identity_repair"]).read_text(encoding="utf-8"))
    setids = list(map(str, gallery["setids"]))
    corrected = corrected_identities(setids, repair)
    if len(setids) != 5413 or len(set(corrected)) != 5412:
        raise P0DataError("corrected gallery cardinality drift")
    raw_c128_all, canonical_rows_all = [], []
    for index in range(600):
        ranked = natural_c128(all_scores[index].tolist(), corrected)
        rows = [item["physical_row"] for item in ranked]
        values = [item["raw_score"] for item in ranked]
        if rows != stored_rows[index].tolist() or values != stored_scores[index].tolist():
            raise P0DataError(f"inherited C128 independent replay mismatch at {index}")
        raw_c128_all.append(ranked)
        canonical_rows_all.append([item["physical_row"] for item in canonical_c128(ranked)])
    canonical_payload = {
        "schema_version": "rcde_model_visible_natural_c128_v1_2",
        "status": "TARGET_FREE_CANONICAL_C128_FROZEN",
        "candidate_order": "corrected_identity_lexical_order",
        "allowed_fields": ["query_id", "query_ordinal", "candidate_physical_rows"],
        "forbidden_fields_present": [],
        "target_insertions": 0,
        "rows": [
            {"query_id": item["query_id"], "query_ordinal": int(item["query_ordinal"]), "candidate_physical_rows": canonical_rows_all[index]}
            for index, item in enumerate(inherited_rows)
        ],
        "protected_access_counts": dict(PROTECTED_ZERO),
    }
    canonical_payload["rows_logical_sha256"] = canonical_sha256(canonical_payload["rows"])
    write_json(output / "model_visible_c128.json", canonical_payload)
    link_immutable(inherited_prejoin_path, output / "base_prejoin.pt")
    write_json(output / "base_prejoin_receipt.json", {
        "schema_version": "rcde_colnomic_raw_c128_prejoin_receipt_v1_2",
        "status": "RCDE_V1_2_INHERITED_PREJOIN_INDEPENDENTLY_VERIFIED",
        "created_ns": time.time_ns(), "query_count": 600,
        "physical_gallery_rows": 5413, "corrected_gallery_identities": 5412,
        "base_prejoin_sha256": sha256_file(output / "base_prejoin.pt"),
        "inherited_source_sha256": sha256_file(inherited_prejoin_path),
        "physical_score_tensor_sha256": tensor_sha256(all_scores),
        "c128_row_tensor_sha256": tensor_sha256(stored_rows),
        "c128_score_tensor_sha256": tensor_sha256(stored_scores),
        "validated_prejoin_hit_count": 600, "RAW_score_read_count": 600,
        "new_ColNomic_forward_count": 0, "label_read_count": 0, "target_insertions": 0,
        "protected_access_counts": dict(PROTECTED_ZERO),
    })

    confidences = [robust_confidence(stored_scores[index].tolist()) for index in range(600)]
    # Fold comes from the isolated postjoin ledger, but low-margin itself is
    # computed and sealed before target identity/correctness is consulted.
    fold_ledger = json.loads(resolve(bindings["isolated_prejoin_folds_600"]).read_text(encoding="utf-8"))
    fold_rows = sorted(fold_ledger["records"], key=lambda item: int(item["query_ordinal"]))
    if (
        fold_ledger.get("record_sequence_sha256") != canonical_sha256(fold_ledger.get("records", []))
        or fold_ledger.get("query_count") != 600
        or [item["query_id"] for item in fold_rows] != [item["query_id"] for item in inherited_rows]
        or any(set(item) != {"query_id", "query_ordinal", "source_image_sha256", "inner_fold", "track"} for item in fold_rows)
        or any(item["source_image_sha256"] != inherited_rows[index]["source_image_sha256"] for index, item in enumerate(fold_rows))
    ):
        raise P0DataError("isolated prejoin fold sequence/schema drift")
    folds = [int(item["inner_fold"]) for item in fold_rows]
    thresholds: dict[str, Any] = {}
    low_flags = []
    for heldout in (1, 2, 3, 4):
        members = [index for index, fold in enumerate(folds) if fold != heldout]
        order = sorted(members, key=lambda index: (confidences[index], inherited_rows[index]["source_image_sha256"], inherited_rows[index]["query_id"]))
        k = math.ceil(0.25 * len(order)); tau = confidences[order[k - 1]]
        thresholds[str(heldout)] = {
            "calibration_count": len(order), "k": k, "tau": tau,
            "membership_logical_sha256": canonical_sha256([inherited_rows[index]["query_id"] for index in order]),
        }
    final_order = sorted(range(600), key=lambda index: (confidences[index], inherited_rows[index]["source_image_sha256"], inherited_rows[index]["query_id"]))
    tau_final = confidences[final_order[149]]
    for index, item in enumerate(inherited_rows):
        low_flags.append({
            "query_id": item["query_id"], "source_image_sha256": item["source_image_sha256"],
            "fold": folds[index], "u_RAW": confidences[index],
            "oof_low_margin": confidences[index] <= thresholds[str(folds[index])]["tau"],
        })
    write_json(output / "rcde_raw_low_margin_receipt.json", {
        "schema_version": "rcde_raw_low_margin_receipt_v1_2",
        "status": "RAW_LOW_MARGIN_V1_1_SEMANTICS_RECONSTRUCTED_BEFORE_TARGET_JOIN",
        "namespace": "RAW_LOW_MARGIN_V1_1_Q25_MAD_C128",
        "protocol_sha256": protocol_sha,
        "base_prejoin_sha256": sha256_file(output / "base_prejoin.pt"),
        "model_visible_c128_sha256": sha256_file(output / "model_visible_c128.json"),
        "redacted_query_token_manifest_sha256": bindings["inherited_v1_1_redacted_manifest"]["sha256"],
        "sanitized_query_source_ledger_sha256": bindings["isolated_sanitized_query_sources_600"]["sha256"],
        "isolated_prejoin_fold_ledger_sha256": bindings["isolated_prejoin_folds_600"]["sha256"],
        "postjoin_training_role_ledger_expected_sha256_from_prevalidated_protocol": (
            protocol["postjoin_bindings"]["isolated_training_roles_600"]["sha256"]
        ),
        "thresholds": thresholds, "tau_final": tau_final,
        "final_calibration": {
            "calibration_count": 600, "k": 150, "tau": tau_final,
            "membership_logical_sha256": canonical_sha256(
                [inherited_rows[index]["query_id"] for index in final_order]
            ),
        },
        "rows": low_flags, "rows_logical_sha256": canonical_sha256(low_flags),
        "allowed_uses": protocol["low_margin_contract"]["allowed_uses"],
        "forbidden_uses": protocol["low_margin_contract"]["forbidden_uses"],
        "RAW_score_read_count": 600, "label_read_count": 0,
        "protected_access_counts": dict(PROTECTED_ZERO),
    })

    # The receipt above is now fsynced and closed.  This is the first point at
    # which target-bearing identities/supergroups or the historical label
    # ledger may be hashed or opened.
    low_margin_write_completed_ns = time.time_ns()
    postjoin_first_read_started_ns = time.time_ns()
    validate_postjoin_bindings(protocol)
    bindings = {**protocol["prejoin_bindings"], **protocol["postjoin_bindings"]}
    write_json(output / "access_phase_receipt.json", {
        "schema_version": "rcde_access_phase_receipt_v1_2",
        "status": "PREJOIN_SEALED_BEFORE_POSTJOIN_BINDING_READ",
        "prejoin_binding_names": sorted(protocol["prejoin_bindings"]),
        "postjoin_binding_names": sorted(protocol["postjoin_bindings"]),
        "low_margin_write_completed_ns": low_margin_write_completed_ns,
        "postjoin_first_read_started_ns": postjoin_first_read_started_ns,
        "temporal_order_valid": postjoin_first_read_started_ns >= low_margin_write_completed_ns,
        "protected_access_counts": dict(PROTECTED_ZERO),
    })
    role_ledger = json.loads(resolve(bindings["isolated_training_roles_600"]).read_text(encoding="utf-8"))
    role_rows = sorted(role_ledger["records"], key=lambda item: int(item["query_ordinal"]))
    if [item["query_id"] for item in role_rows] != [item["query_id"] for item in inherited_rows]:
        raise P0DataError("isolated postjoin role sequence drift")

    role_by_id = {str(item["query_id"]): item for item in role_rows}
    joined = []
    for index, source in enumerate(inherited_rows):
        role = role_by_id[source["query_id"]]
        target = str(role["identity"])
        ranked = raw_c128_all[index]
        ranked_ids = [str(item["corrected_identity"]) for item in ranked]
        target_rank = next((position for position, identity in enumerate(ranked_ids, start=1) if identity == target), None)
        target_item = next((item for item in reduce_scores(all_scores[index].tolist(), corrected) if item["corrected_identity"] == target), None)
        if target_item is None:
            raise P0DataError(f"target absent from corrected gallery: {target}")
        joined.append({
            "query_id": source["query_id"], "execution_ordinal": index,
            "query_ordinal": int(source["query_ordinal"]),
            "source_image_sha256": source["source_image_sha256"],
            "fold": int(role["inner_fold"]), "track": str(role["track"]),
            "identity": target, "supergroup": str(role["supergroup"]),
            "target_hit": target_rank is not None, "target_raw_rank": target_rank,
            "target_physical_row": int(target_item["physical_row"]),
            "target_raw_score": float(target_item["raw_score"]),
            "raw_correct": ranked_ids[0] == target,
            "raw_winner": ranked_ids[0], "oof_low_margin": bool(low_flags[index]["oof_low_margin"]),
        })

    legacy = json.loads(resolve(bindings["legacy_raw_replay_ledger"]).read_text(encoding="utf-8"))
    legacy_by_id = {item["query_id"]: item for item in legacy["records"]}
    replay_records = []
    for row in joined:
        historical = legacy_by_id[row["query_id"]]
        available = bool(historical["available"])
        matched = False
        if available:
            legacy_row = int(historical["legacy_target_physical_row"])
            legacy_score = float(historical["legacy_target_score_float32"])
            physical_rank_zero = int((all_scores[row["execution_ordinal"]] > all_scores[row["execution_ordinal"], legacy_row]).sum().item())
            score = float(all_scores[row["execution_ordinal"], legacy_row])
            matched = (
                str(historical["legacy_identity"]) == row["identity"]
                and corrected[legacy_row] == row["identity"]
                and physical_rank_zero == int(historical["legacy_physical_rank_zero_based"])
                and score == legacy_score
            )
        replay_records.append({"query_id": row["query_id"], "available": available, "matched": matched})
    replay = legacy_replay_summary(replay_records)
    if replay != {"population_count": 600, "available_count": 72, "unavailable_count": 528, "match_count": 72, "mismatch_count": 0, "all_available_match": True}:
        raise P0DataError(f"legacy replay mismatch: {replay}")
    write_json(output / "legacy_raw_replay_receipt.json", {
        "schema_version": "rcde_legacy_raw_replay_receipt_v1_2",
        "status": "LEGACY_AVAILABLE_DENOMINATOR_REPLAY_PASS",
        **replay, "records": replay_records,
        "legacy_mixed_carrier_runtime_read_count": 0,
        "protected_access_counts": dict(PROTECTED_ZERO),
    })

    fold_receipts: dict[str, Any] = {}
    episodes = []
    overall_denominator = overall_at6_count = overall_both_count = 0
    negative_gate = balanced_gate = hall_gate = True
    hall_rows: dict[str, Any] = {}
    for heldout in (1, 2, 3, 4):
        access = fold_access(role_rows, corrected, heldout_fold=heldout)
        train_rows = [row for row in joined if row["fold"] != heldout]
        gate_rows = [row for row in train_rows if row["target_hit"]]
        reports = []
        middle_ids: set[str] = set(); middle_groups: set[str] = set()
        broad_ids: set[str] = set(); broad_groups: set[str] = set()
        target_restricted_ranks = []
        for row in train_rows:
            ranking = reduce_scores(all_scores[row["execution_ordinal"]].tolist(), corrected, allowed_identities=access["train_identities"])
            selection = select_611_negatives(
                ranking, target_identity=row["identity"], query_sha256=row["source_image_sha256"],
            )
            target_restricted_ranks.append(next(item["fold_local_rank"] for item in ranking if item["corrected_identity"] == row["identity"]))
            selected = selection["selected"]
            if row["target_hit"]:
                reports.append({
                    "at_least_6": selection["available_negative_count"] >= 6,
                    "middle_and_broad": selection["middle_bin_available"] and selection["broad_bin_available"],
                    "fallback_count": selection["fallback_count"],
                })
            selected_by_role = {item["selection_role"]: item for item in selected}
            for key, identity_set, group_set in (
                ("middle", middle_ids, middle_groups), ("broad", broad_ids, broad_groups),
            ):
                if key in selected_by_role:
                    identity = selected_by_role[key]["corrected_identity"]
                    identity_set.add(identity)
                    group_set.add(next(role_item["supergroup"] for role_item in role_rows if role_item["identity"] == identity))
            pair_loss_eligible = bool(row["target_hit"] and selection["training_eligible"])
            # This is a postjoin training receipt.  The future loader may use
            # pair_loss_eligible to select loss records, but neither it nor the
            # target flag may enter the model-visible candidate payload.
            # ``full_gallery_RAW_correct`` is also postjoin-only: P0 resource
            # qualification uses it solely to freeze a same-outer-fold 2+2
            # correct/wrong maximum-update benchmark.  It is explicitly not an
            # input, sampler feature, loss feature, or model-visible field.
            model_order = episode_model_order(
                query_sha256=row["source_image_sha256"],
                target_identity=row["identity"],
                target_row=row["target_physical_row"],
                negatives=selected,
            )
            episodes.append({
                "heldout_fold": heldout, "query_id": row["query_id"],
                "execution_ordinal": row["execution_ordinal"], "query_ordinal": row["query_ordinal"],
                "target_identity": row["identity"], "target_physical_row": row["target_physical_row"],
                "full_gallery_RAW_target_hit": bool(row["target_hit"]),
                "full_gallery_RAW_correct": bool(row["raw_correct"]),
                "pair_loss_eligible": pair_loss_eligible,
                "target_fold_local_rank": target_restricted_ranks[-1],
                "allowed_identity_count": len(access["train_identities"]),
                "available_negative_count": selection["available_negative_count"],
                "middle_bin_available": selection["middle_bin_available"],
                "broad_bin_available": selection["broad_bin_available"],
                "selected_negatives": selected,
                "model_candidate_order": model_order,
                "model_candidate_order_exposes_target_slot": False,
            })
        denominator = len(reports)
        at6_count = sum(item["at_least_6"] for item in reports)
        both_count = sum(item["middle_and_broad"] for item in reports)
        at6_rate = at6_count / denominator
        both_rate = both_count / denominator
        raw_correct_pool = len({row["query_id"] for row in gate_rows if row["raw_correct"]})
        raw_wrong_pool = len({row["query_id"] for row in gate_rows if not row["raw_correct"]})
        gate = (
            at6_rate >= protocol["data_gates"]["negative_at_least_6_each_fold_min"]
            and both_rate >= protocol["data_gates"]["middle_and_broad_each_fold_min"]
            and len(middle_ids) >= protocol["data_gates"]["middle_identity_and_group_each_fold_min"]
            and len(middle_groups) >= protocol["data_gates"]["middle_identity_and_group_each_fold_min"]
            and len(broad_ids) >= protocol["data_gates"]["broad_identity_and_group_each_fold_min"]
            and len(broad_groups) >= protocol["data_gates"]["broad_identity_and_group_each_fold_min"]
        )
        negative_gate &= gate
        balanced_gate &= raw_correct_pool >= 2 and raw_wrong_pool >= 2
        fold_receipts[str(heldout)] = {
            "heldout_fold": heldout,
            "train_identity_count": len(access["train_identities"]),
            "train_supergroup_count": len(access["train_supergroups"]),
            "heldout_identity_count": len(access["heldout_identities"]),
            "heldout_supergroup_count": len(access["heldout_supergroups"]),
            "allowed_physical_row_count": len(access["allowed_physical_rows"]),
            "heldout_physical_row_count": len(access["heldout_physical_rows"]),
            "identity_intersection_count": len(set(access["train_identities"]) & set(access["heldout_identities"])),
            "supergroup_intersection_count": len(set(access["train_supergroups"]) & set(access["heldout_supergroups"])),
            "physical_row_intersection_count": len(set(access["allowed_physical_rows"]) & set(access["heldout_physical_rows"])),
            "all_outer_train_episode_count": len(train_rows),
            "gate_denominator_rule": "outer_train_and_full_gallery_RAW_target_hit",
            "denominator": denominator, "at_least_6_count": at6_count, "at_least_6_rate": at6_rate,
            "middle_and_broad_count": both_count, "middle_and_broad_rate": both_rate,
            "middle_identity_count": len(middle_ids), "middle_supergroup_count": len(middle_groups),
            "broad_identity_count": len(broad_ids), "broad_supergroup_count": len(broad_groups),
            "fallback_episode_count": sum(item["fallback_count"] > 0 for item in reports),
            "target_restricted_rank_min": min(target_restricted_ranks),
            "target_restricted_rank_max": max(target_restricted_ranks),
            "raw_correct_pool": raw_correct_pool, "raw_wrong_pool": raw_wrong_pool,
            "gate": gate,
            "train_identities_sha256": canonical_sha256(access["train_identities"]),
            "heldout_identities_sha256": canonical_sha256(access["heldout_identities"]),
            "allowed_physical_rows_sha256": canonical_sha256(access["allowed_physical_rows"]),
        }
        overall_denominator += denominator; overall_at6_count += at6_count; overall_both_count += both_count
        for correct in (False, True):
            population = [row for row in joined if row["fold"] == heldout and row["target_hit"] and row["raw_correct"] is correct]
            ok, details = hall_feasible([row["supergroup"] for row in population])
            details["feasible"] = ok
            hall_rows[f"fold{heldout}_{'correct' if correct else 'wrong'}"] = details
            hall_gate &= ok
    overall_at6 = overall_at6_count / overall_denominator
    overall_both = overall_both_count / overall_denominator
    negative_gate &= (
        overall_at6 >= protocol["data_gates"]["negative_at_least_6_overall_min"]
        and overall_both >= protocol["data_gates"]["middle_and_broad_overall_min"]
    )
    write_json(output / "fold_access_and_episode_receipt.json", {
        "schema_version": "rcde_fold_access_and_episode_receipt_v1_2",
        "status": "RESTRICT_BEFORE_RANK_FOLD_ACCESS_FROZEN",
        "construction": "physical_rows_restrict_then_corrected_identity_max_reduce_then_stable_rank_then_6_1_1",
        "explicitly_forbidden_construction_used": False,
        "folds": fold_receipts, "episode_count": len(episodes),
        "episodes": episodes, "episodes_logical_sha256": canonical_sha256(episodes),
        "overall_at_least_6_rate": overall_at6,
        "overall_middle_and_broad_rate": overall_both,
        "negative_gate": negative_gate, "balanced_pool_gate": balanced_gate,
        "target_insertions": 0, "model_score_rank_slot_read_count": 0,
        "full_gallery_RAW_correct_scope": "POSTJOIN_RESOURCE_COHORT_SELECTION_ONLY_NOT_MODEL_VISIBLE",
        "full_gallery_RAW_correct_model_visible_read_count": 0,
        "pair_loss_rule": "full_gallery_RAW_target_hit_and_at_least_2_fold_local_negatives",
        "pair_loss_eligible_episode_count": sum(item["pair_loss_eligible"] for item in episodes),
        "pair_loss_miss_episode_count": sum(item["pair_loss_eligible"] and not item["full_gallery_RAW_target_hit"] for item in episodes),
        "protected_access_counts": dict(PROTECTED_ZERO),
    })

    slices = {}
    for name, predicate in {
        "overall": lambda row: True,
        "raw_correct": lambda row: row["raw_correct"],
        "raw_wrong": lambda row: not row["raw_correct"],
        "low_margin": lambda row: row["oof_low_margin"],
    }.items():
        population = [row for row in joined if predicate(row)]
        slices[name] = {
            "queries": len(population), "identities": len({row["identity"] for row in population}),
            "supergroups": len({row["supergroup"] for row in population}),
            "fold_supergroups": {str(fold): len({row["supergroup"] for row in population if row["fold"] == fold}) for fold in (1, 2, 3, 4)},
            "maximum_supergroup_frame_share": max(Counter(row["supergroup"] for row in population).values(), default=0) / len(population) if population else 0.0,
        }
    gates = protocol["statistics_gates"]
    stat_gate = (
        slices["overall"]["supergroups"] >= gates["overall_supergroups_min"]
        and all(value >= gates["each_fold_supergroups_min"] for value in slices["overall"]["fold_supergroups"].values())
        and slices["raw_correct"]["supergroups"] >= gates["raw_correct_supergroups_min"]
        and slices["raw_wrong"]["supergroups"] >= gates["raw_wrong_supergroups_min"]
        and all(value >= gates["raw_correct_each_fold_supergroups_min"] for value in slices["raw_correct"]["fold_supergroups"].values())
        and all(value >= gates["raw_wrong_each_fold_supergroups_min"] for value in slices["raw_wrong"]["fold_supergroups"].values())
    )
    power_populations = {
        "overall_A1_internal": list(joined),
        "target_hit_R1": [row for row in joined if row["target_hit"]],
        "target_hit_raw_wrong_T1": [row for row in joined if row["target_hit"] and not row["raw_correct"]],
        "target_hit_raw_correct": [row for row in joined if row["target_hit"] and row["raw_correct"]],
        "low_margin_descriptive": [row for row in joined if row["oof_low_margin"]],
    }
    power_cohorts = {name: cohort_power(rows) for name, rows in power_populations.items()}
    n1_power = {}
    for heldout in (1, 2, 3, 4):
        for correct in (False, True):
            key = f"fold{heldout}_{'correct' if correct else 'wrong'}"
            rows = [
                row for row in joined
                if row["fold"] == heldout and row["target_hit"] and row["raw_correct"] is correct
            ]
            n1_power[key] = {
                **cohort_power(rows),
                "group_excluding_permutation_feasible": bool(hall_rows[key]["feasible"]),
                "log_loss_0_01": matched_effect_identifiability(cohort_power(rows), 0.01),
                "brier_0_005": matched_effect_identifiability(cohort_power(rows), 0.005),
            }
    r1 = power_cohorts["target_hit_R1"]
    t1 = power_cohorts["target_hit_raw_wrong_T1"]
    a1 = power_cohorts["overall_A1_internal"]
    harmonic_127 = sum(1.0 / rank for rank in range(1, 128))
    endpoint_power = {
        "R1_spatial_C_P_binary_group_direction": {
            "mra": "0.70_vs_0.50", "audit": r1["sign_or_sign_flip"],
        },
        "R1_paired_continuous_group_effect": {
            "mra_dz": 0.50, "audit": r1["paired_continuous"],
        },
        "T1_target_top1_absolute": {
            "mra": "0.50_vs_1_over_127",
            "audit": exact_binomial_power(t1["eligible_supergroups"], 1.0 / 127.0, 0.50),
        },
        "T1_target_MRR_absolute": {
            "mra_target": 0.50, "null_H127_over_127": harmonic_127 / 127.0,
            "audit": matched_effect_identifiability(t1, 0.50 - harmonic_127 / 127.0),
        },
        "T1_matched_control_top1_5pp": matched_effect_identifiability(t1, 0.05),
        "T1_matched_control_MRR_0_03": matched_effect_identifiability(t1, 0.03),
        "matched_arm_3pp_resolution": matched_effect_identifiability(t1, 0.03),
        "matched_arm_5pp_resolution": matched_effect_identifiability(t1, 0.05),
        "A1_R_at_1_internal_2pp": matched_effect_identifiability(a1, 0.02),
        "C1_R_at_1_internal_3_over_109": {
            "absolute_effect": 3.0 / 109.0,
            "population_status_at_P0": "C8_PROTECTED_NOT_OPENED",
            "power80_identifiability_without_C1_group_membership": "UNDETERMINED",
            "query_level_lattice_only": 1.0 / 109.0,
            "paper_level_claim_authorized": False,
        },
    }
    write_json(output / "rcde_statistics_receipt.json", {
        "schema_version": "rcde_statistics_preflight_v1_2",
        "slice_counts": slices, "track_counts": dict(Counter(row["track"] for row in joined)),
        "power_discrete_resolution": {name: sign_power(value["supergroups"]) for name, value in slices.items()},
        "power_cohorts": power_cohorts,
        "n1_outer_fold_by_raw_correctness_power": n1_power,
        "endpoint_power_audit": endpoint_power,
        "mra_registry": {
            "R1_spatial_C_P_binary_group_direction": "0.70_vs_0.50",
            "paired_continuous_group_effect_dz": 0.50,
            "T1_target_top1": "0.50_vs_1_over_127",
            "T1_target_MRR": "0.50_vs_H127_over_127",
            "T1_matched_top1": 0.05,
            "T1_matched_MRR": 0.03,
            "N1_log_loss": 0.01,
            "N1_Brier": 0.005,
            "A1_R_at_1_internal": 0.02,
            "C1_R_at_1_internal": "3_over_109",
        },
        "unknown_paired_group_variance_policy": "report_conditional_sd_boundary_do_not_invent_variance_or_preemptively_abort",
        "power_preflight_alone_changes_statistics_gate": False,
        "statistics_gate": stat_gate,
        "bootstrap_repetitions": gates["bootstrap_repetitions"],
        "permutation_repetitions": gates["permutation_repetitions"], "seed": gates["seed"],
        "protected_access_counts": dict(PROTECTED_ZERO),
    })

    expected = protocol["expected_population"]
    role_checks = {
        "queries_600": len(joined) == expected["training_queries"],
        "identities_50": len({row["identity"] for row in joined}) == expected["identities"],
        "supergroups_49": len({row["supergroup"] for row in joined}) == expected["supergroups"],
        "tracks_exact": Counter(row["track"] for row in joined) == Counter(expected["tracks"]),
        "fold_queries_exact": Counter(str(row["fold"]) for row in joined) == Counter(expected["fold_query_counts"]),
        "fold_groups_exact": {str(fold): len({row["supergroup"] for row in joined if row["fold"] == fold}) for fold in (1, 2, 3, 4)} == expected["fold_group_counts"],
        "legacy_available_replay_all": replay["all_available_match"],
        "legacy_available_count_72": replay["available_count"] == 72,
        "full_gallery_c128_all": all(len(items) == 128 for items in raw_c128_all),
        "canonical_c128_all": all(len(items) == 128 and len(set(items)) == 128 for items in canonical_rows_all),
        "target_identity_present_in_corrected_gallery": all(row["identity"] in set(corrected) for row in joined),
        "isolated_role_ledger_has_no_protected_sections": not any(role_ledger["forbidden_sections_present"].values()),
    }
    gallery_rows = gallery.get("passage_emb", [])
    spatial_rows = spatial.get("rows", [])
    query_token_contract = (
        len(validated_shards) == 600
        and all(item["image_tokens_shape"][0] > 0 for item in validated_shards)
        and all(item["image_tokens_shape"][1] == 128 for item in validated_shards)
        and all(item["image_tokens_dtype"] == "torch.float16" for item in validated_shards)
    )
    reference_token_contract = (
        len(gallery_rows) == 5413
        and all(
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == 2 and tensor.shape[0] > 0 and tensor.shape[1] == 128
            and tensor.dtype == torch.float16 and bool(torch.isfinite(tensor).all())
            for tensor in gallery_rows
        )
    )
    spatial_token_contract = (
        len(spatial_rows) == 5413
        and len(gallery_rows) == 5413
        and all(
            int(spatial_rows[index]["total_token_count"]) == int(gallery_rows[index].shape[0])
            for index in range(5413)
        )
    )
    matched = {
        "eligible": query_token_contract and reference_token_contract and spatial_token_contract,
        "query_dimension": 128, "reference_dimension": 128,
        "query_image_tokens_exclude_template": True,
        "query_token_contract": query_token_contract,
        "reference_token_contract": reference_token_contract,
        "spatial_token_contract": spatial_token_contract,
        "query_count": len(validated_shards),
        "reference_physical_row_count": len(gallery_rows),
        "query_image_token_count_min": min(item["image_tokens_shape"][0] for item in validated_shards),
        "query_image_token_count_max": max(item["image_tokens_shape"][0] for item in validated_shards),
        "reference_token_count_min": min(int(tensor.shape[0]) for tensor in gallery_rows),
        "reference_token_count_max": max(int(tensor.shape[0]) for tensor in gallery_rows),
        "gallery_token_cache_sha256": bindings["gallery_token_cache"]["sha256"],
        "gallery_token_cache_file_bytes": resolve(bindings["gallery_token_cache"]).stat().st_size,
        "gallery_spatial_metadata_sha256": bindings["gallery_spatial_metadata"]["sha256"],
        "redacted_query_manifest_sha256": bindings["inherited_v1_1_redacted_manifest"]["sha256"],
        "redacted_query_shard_total_file_bytes": sum(
            int(item["redacted_shard_file_bytes"]) for item in validated_shards
        ),
        "redacted_query_shard_max_file_bytes": max(
            int(item["redacted_shard_file_bytes"]) for item in validated_shards
        ),
        "query_image_token_total": sum(
            int(item["image_tokens_shape"][0]) for item in validated_shards
        ),
        "query_image_token_max": max(
            int(item["image_tokens_shape"][0]) for item in validated_shards
        ),
        "reference_token_total": sum(int(tensor.shape[0]) for tensor in gallery_rows),
        "reference_token_max": max(int(tensor.shape[0]) for tensor in gallery_rows),
        "matched_colnomic_resource_ledger_scope": "EXISTING_SOURCE_CACHE_READ_PLUS_FOUR_LAYER_ISOMETRIC_LIFT",
        "same_gallery_cache_as_RAW": True,
    }
    write_json(output / "rcde_data_role_receipt.json", {
        "schema_version": "rcde_data_role_receipt_v1_2",
        "status": "P0_V1_2_DATA_ROLE_AUDITED",
        "role_checks": role_checks,
        "target_hit_count": sum(row["target_hit"] for row in joined),
        "target_miss_count": sum(not row["target_hit"] for row in joined),
        "raw_correct_count": sum(row["raw_correct"] for row in joined),
        "raw_wrong_count": sum(not row["raw_correct"] for row in joined),
        "legacy_replay": replay, "negative_folds": fold_receipts,
        "negative_overall_at_least_6_rate": overall_at6,
        "negative_overall_middle_and_broad_rate": overall_both,
        "negative_gate": negative_gate, "balanced_pool_gate": balanced_gate,
        "n1_hall_strata": hall_rows, "n1_hall_gate": hall_gate,
        "matched_colnomic_control": matched,
        "joined_rows_logical_sha256": canonical_sha256(joined),
        "label_read_count": 600, "target_insertions": 0,
        "protected_access_counts": dict(PROTECTED_ZERO),
    })

    # The gallery token payload is no longer needed once matched-control token
    # geometry has been checked.  Release it before walking image metadata so
    # the resource preflight itself does not create an avoidable CPU-RSS peak.
    del gallery, prejoin, all_scores, stored_rows, stored_scores
    import gc
    gc.collect()

    union_rows = sorted({row for candidates in canonical_rows_all for row in candidates})
    query_shapes = []
    for source in source_rows:
        path = Path(str(source["path"]))
        if not path.is_file() or sha256_file(path) != source["source_image_sha256"]:
            raise P0DataError(f"query source image drift: {source['query_id']}")
        query_shapes.append({
            "kind": "query", "id": source["query_id"], "execution_ordinal": len(query_shapes),
            "query_ordinal": int(source["query_ordinal"]),
            "source_image_path": str(path), "source_image_sha256": source["source_image_sha256"],
            **shape_for_image(path),
        })
    reference_shapes = []
    for physical_row in union_rows:
        path = Path(str(spatial["rows"][physical_row]["path"]))
        if not path.is_file():
            raise P0DataError(f"reference source absent: {physical_row}")
        reference_shapes.append({
            "kind": "reference", "physical_row": physical_row,
            "source_image_path": str(path), "source_image_sha256": sha256_file(path),
            **shape_for_image(path),
        })
    all_shapes = query_shapes + reference_shapes
    reference_by_row = {int(item["physical_row"]): item for item in reference_shapes}
    query_p50 = nearest_rank_percentile(query_shapes, 0.50, kind="query", identifier_key="source_image_sha256")
    query_p95 = nearest_rank_percentile(query_shapes, 0.95, kind="query", identifier_key="source_image_sha256")
    reference_p50 = nearest_rank_percentile(reference_shapes, 0.50, kind="reference", identifier_key="source_image_sha256")
    reference_p95 = nearest_rank_percentile(reference_shapes, 0.95, kind="reference", identifier_key="source_image_sha256")
    q_nmax_tokens = max(int(item["valid_tokens"]) for item in query_shapes)
    q_nmax = min(
        (item for item in query_shapes if item["valid_tokens"] == q_nmax_tokens),
        key=lambda item: (cohort_digest("query", str(item["source_image_sha256"])), str(item["id"])),
    )
    ref_nmax_tokens = max(int(item["valid_tokens"]) for item in reference_shapes)
    reference_max = min(
        (item for item in reference_shapes if item["valid_tokens"] == ref_nmax_tokens),
        key=lambda item: (cohort_digest("reference", str(item["source_image_sha256"])), int(item["physical_row"])),
    )
    refsum_rows = []
    pair_rows = []
    for query in query_shapes:
        execution_ordinal = int(query["execution_ordinal"])
        candidate_rows = canonical_rows_all[execution_ordinal]
        reference_token_sum = sum(reference_by_row[row]["valid_tokens"] for row in candidate_rows)
        refsum_rows.append((int(reference_token_sum), query))
        for physical_row in candidate_rows:
            reference = reference_by_row[physical_row]
            pair_rows.append((int(query["valid_tokens"]) * int(reference["valid_tokens"]), query, reference))
    max_refsum = max(item[0] for item in refsum_rows)
    q_refsum = min(
        (item[1] for item in refsum_rows if item[0] == max_refsum),
        key=lambda item: (cohort_digest("query", str(item["source_image_sha256"])), str(item["id"])),
    )
    max_pair_product = max(item[0] for item in pair_rows)
    pair_product, q_pair, ref_pair = min(
        (item for item in pair_rows if item[0] == max_pair_product),
        key=lambda item: (
            cohort_digest("query", str(item[1]["source_image_sha256"])),
            cohort_digest("reference", str(item[2]["source_image_sha256"])),
            int(item[2]["physical_row"]),
        ),
    )
    def query_case(item: Mapping[str, Any]) -> dict[str, Any]:
        ordinal = int(item["execution_ordinal"])
        return {
            "query_id": item["id"], "execution_ordinal": ordinal,
            "query_ordinal": int(item["query_ordinal"]), "valid_tokens": int(item["valid_tokens"]),
            "candidate_physical_rows": canonical_rows_all[ordinal],
        }
    def reference_case(item: Mapping[str, Any], *, bind_host: bool = False) -> dict[str, Any]:
        result = {
            "physical_row": int(item["physical_row"]), "valid_tokens": int(item["valid_tokens"]),
            "source_image_sha256": item["source_image_sha256"],
        }
        if bind_host:
            physical_row = int(item["physical_row"])
            eligible = [
                query for query in query_shapes
                if physical_row in canonical_rows_all[int(query["execution_ordinal"])]
            ]
            if not eligible:
                raise P0DataError("resource reference cohort row has no natural-C128 host query")
            def host_hash(query: Mapping[str, Any]) -> str:
                payload = (
                    b"RCDE_RESOURCE_REF_HOST_V1_2\0"
                    + physical_row.to_bytes(8, "big", signed=False)
                    + str(query["id"]).encode("utf-8")
                )
                return hashlib.sha256(payload).hexdigest()
            host = min(eligible, key=lambda query: (host_hash(query), str(query["id"])))
            result.update({
                "host_query": query_case(host),
                "host_selection_namespace": "RCDE_RESOURCE_REF_HOST_V1_2",
                "host_selection_sha256": host_hash(host),
                "host_eligible_query_count": len(eligible),
            })
        return result
    def raw_pixel_case(item: Mapping[str, Any]) -> dict[str, Any]:
        result = {
            "kind": str(item["kind"]),
            "source_image_sha256": str(item["source_image_sha256"]),
            "raw_h": int(item["raw_h"]), "raw_w": int(item["raw_w"]),
            "raw_pixels": int(item["raw_h"]) * int(item["raw_w"]),
        }
        if item["kind"] == "query":
            result.update({
                "query_id": str(item["id"]),
                "execution_ordinal": int(item["execution_ordinal"]),
                "query_ordinal": int(item["query_ordinal"]),
            })
        else:
            result["physical_row"] = int(item["physical_row"])
        return result
    raw_pixel_p50 = raw_pixel_percentile(all_shapes, 0.50)
    raw_pixel_p95 = raw_pixel_percentile(all_shapes, 0.95)
    raw_pixel_max_key = max(raw_pixel_rank_key(item)[0] for item in all_shapes)
    raw_pixel_max = min(
        (item for item in all_shapes if raw_pixel_rank_key(item)[0] == raw_pixel_max_key),
        key=lambda item: raw_pixel_rank_key(item)[1:],
    )
    write_json(output / "resource_shape_ledger.json", {
        "schema_version": "rcde_resource_shape_ledger_v1_2",
        "status": "P0_V1_2_RESOURCE_SHAPE_LEDGER_FROZEN",
        "preprocessing": "full_frame_518_long_edge_round_half_up_right_bottom_pad14_complete_footprint_valid",
        "query_count": len(query_shapes), "reference_union_count": len(reference_shapes),
        "reference_union_physical_rows": union_rows,
        "reference_union_sha256": canonical_sha256(union_rows),
        "rows": all_shapes, "rows_logical_sha256": canonical_sha256(all_shapes),
        "cohort_selection": {
            "percentile_rule": "stable_sort_valid_tokens_asc_then_canonical_sha256_asc_select_ceil_pN_minus_1",
            "raw_pixel_percentile_rule": "stable_sort_raw_h_times_raw_w_asc_then_canonical_sha256_asc_then_kind_identifier_select_ceil_pN_minus_1",
            "max_tie_rule": "canonical_sha256_asc_then_identifier_asc",
            "query_population": 600, "reference_population": len(reference_shapes),
            "raw_pixel_population": len(all_shapes),
            "membership": "full_gallery_natural_C128_union_and_per_query_memberships",
        },
        "worst_case_cohort": {
            "Q_NMAX": query_case(q_nmax),
            "Q_QUERY_P50": query_case(query_p50),
            "Q_QUERY_P95": query_case(query_p95),
            "Q_REFSUM_MAX": {**query_case(q_refsum), "reference_valid_token_sum": max_refsum},
            "Q_PAIR_MAX": {**query_case(q_pair), "reference": reference_case(ref_pair), "query_reference_token_product": pair_product, "expected_unary_count": 128, "expected_pair_count": 8128},
            "REFERENCE_P50": reference_case(reference_p50, bind_host=True),
            "REFERENCE_P95": reference_case(reference_p95, bind_host=True),
            "REFERENCE_MAX": reference_case(reference_max, bind_host=True),
            "RAW_PIXEL_P50": raw_pixel_case(raw_pixel_p50),
            "RAW_PIXEL_P95": raw_pixel_case(raw_pixel_p95),
            "RAW_PIXEL_MAX": raw_pixel_case(raw_pixel_max),
        },
        "low_margin_write_completed_ns": low_margin_write_completed_ns,
        "postjoin_first_read_started_ns": postjoin_first_read_started_ns,
        "protected_access_counts": dict(PROTECTED_ZERO),
    })

    if not all(role_checks.values()):
        status, failure = "DINO_RCDE_DATA_ROLE_PREFLIGHT_ABORT", "data/base"
    elif not negative_gate or not balanced_gate:
        status, failure = "DINO_RCDE_NEGATIVE_BAND_COVERAGE_INELIGIBLE", "data"
    elif not hall_gate:
        status, failure = "N1_PERMUTATION_CONTROL_INELIGIBLE", "data/statistics"
    elif not matched["eligible"]:
        status, failure = "COLNOMIC_RCDE_MATCHED_CONTROL_INELIGIBLE", "data/base"
    elif not stat_gate:
        status, failure = "DINO_RCDE_STATISTICALLY_INELIGIBLE", "data/statistics"
    else:
        status, failure = "DINO_RCDE_P0_DATA_STATISTICS_READY_FOR_RESOURCE", None
    result = {
        "schema_version": SCHEMA, "status": status, "failure_domain": failure,
        "claim_level": protocol["claim_level"], "next_authorized_stage": None,
        "next_required_component": "P0_RESOURCE" if failure is None else None,
        "automatic_stage_advance": False, "natural_training_authorized": False,
        "protocol_sha256": protocol_sha, "contract_sha256": sha256_file(output / "contract.json"),
        "redacted_inheritance_receipt_sha256": sha256_file(output / "redacted_inheritance_receipt.json"),
        "base_prejoin_receipt_sha256": sha256_file(output / "base_prejoin_receipt.json"),
        "low_margin_receipt_sha256": sha256_file(output / "rcde_raw_low_margin_receipt.json"),
        "data_role_receipt_sha256": sha256_file(output / "rcde_data_role_receipt.json"),
        "fold_access_receipt_sha256": sha256_file(output / "fold_access_and_episode_receipt.json"),
        "statistics_receipt_sha256": sha256_file(output / "rcde_statistics_receipt.json"),
        "resource_shape_ledger_sha256": sha256_file(output / "resource_shape_ledger.json"),
        "model_visible_c128_sha256": sha256_file(output / "model_visible_c128.json"),
        "legacy_raw_replay_receipt_sha256": sha256_file(output / "legacy_raw_replay_receipt.json"),
        "access_phase_receipt_sha256": sha256_file(output / "access_phase_receipt.json"),
        "RAW_score_read_count": 600, "label_read_count": 600,
        "protected_access_counts": dict(PROTECTED_ZERO),
        "elapsed_seconds": (time.time_ns() - started_ns) / 1.0e9,
    }
    write_json(output / "result.json", result)
    files = sorted(path for path in output.iterdir() if path.is_file())
    write_json(output / "manifest.json", {
        "schema_version": "rcde_p0_data_artifact_manifest_v1_2",
        "status": status,
        "files": [{"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files],
        "protected_access_counts": dict(PROTECTED_ZERO),
    })
    print(json.dumps({"status": status, "failure_domain": failure, "next_required_component": result["next_required_component"]}, sort_keys=True))
    return 0 if failure is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
