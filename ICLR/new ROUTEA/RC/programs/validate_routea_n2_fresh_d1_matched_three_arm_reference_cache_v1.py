#!/usr/bin/env python3
"""Independently validate the fresh-D1 current64 reference-token cache."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_routea_d1_current_runtime_bridge_e0_v1 as e0  # noqa: E402
from rc_aslo_xf.conditional_colnomic_ms_proposal import (  # noqa: E402
    ValidatedSpatialCacheResolver,
    gallery_work_ordinal_by_physical_row,
)
from rc_aslo_xf.conditional_rep_sources import (  # noqa: E402
    build_gallery_source,
    build_prejoin_sources,
)
from rc_aslo_xf.cw1_sr0_s8_feature_runtime_v1 import (  # noqa: E402
    LEGACY_SPATIAL_UNION_COUNT,
    SPATIAL_CACHE_ROOT,
    tensor_sha256,
)


CONTRACT = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_REMATERIALIZATION_CONTRACT_V1_20260904.md"
DEFAULT_SOURCE_MANIFEST = (
    ROOT
    / "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1/manifest.json"
)
DEFAULT_SOURCE_VALIDATION = (
    ROOT
    / "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1/independent_validation.json"
)
OLD_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_reference_cache_v1"
OLD_PAYLOAD = OLD_ROOT / "payload.pt"
OLD_RESULT = OLD_ROOT / "result.json"
OLD_VALIDATION = OLD_ROOT / "independent_validation.json"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_reference_cache_v1"
PAYLOAD = OUT_ROOT / "payload.pt"
RESULT = OUT_ROOT / "result.json"
OUT = OUT_ROOT / "independent_validation.json"
PRODUCER = ROOT / "programs/materialize_routea_n2_fresh_d1_matched_three_arm_reference_cache_v1.py"
RESOLVER_SOURCE = ROOT / "src/rc_aslo_xf/cw1_sr0_s8_feature_runtime_v1.py"

EXPECTED_UNION_COUNT = 2805
EXPECTED_REUSE_COUNT = 2801
EXPECTED_PAIR64_PENDING = 64
EXPECTED_CURRENT64_QUERY_COUNT = 64
EXPECTED_MISSING_ROWS = (467, 1488, 2435, 5159)
EXPECTED_OLD_COUNT = 3004
EXPECTED_OLD_PAYLOAD_SHA256 = (
    "ac9126534abb558683d07a13f6cb469ba5f46170ac70e6897997f46d3a26b53c"
)
EXPECTED_OLD_RESULT_SHA256 = (
    "f981c399db9e18debc2c80a9050bfadf6a4a4d1b0fce06fc4826e0fac2c71152"
)
EXPECTED_OLD_VALIDATION_SHA256 = (
    "284f973b04d79a7d9fdb8b57097552e198d11029f60f706d4fe92ec6f31f0543"
)
REFERENCE_KEYS = {
    "physical_row",
    "source_path",
    "source_image_sha256",
    "grid_shape",
    "tokens",
    "tokens_sha256",
    "source_kind",
    "source_logical_sha256",
}
PAYLOAD_KEYS = {
    "schema_version",
    "status",
    "claim_level",
    "contract_population_complete",
    "current64_query_count",
    "pair64_pending",
    "reference_union",
    "references",
    "source_kind_counts",
    "reuse",
    "bindings",
    "access",
}
RESULT_KEYS = {
    "schema_version",
    "status",
    "claim_level",
    "contract_population_complete",
    "current64_query_count",
    "pair64_pending",
    "reference_count",
    "source_kind_counts",
    "reuse",
    "payload_sha256",
    "bindings",
    "access",
    "scientific_GO_or_NO_GO",
    "ownership_GO_or_NO_GO",
    "automatic_stage_advance",
    "next_authorized_stage",
    "logical_sha256",
}
FORBIDDEN_SOURCE_KEYS = {
    "target_identity",
    "target_label",
    "target_row",
    "target_position",
    "target_rank",
    "target_correct",
    "correctness",
    "supergroup",
    "outcome",
    "switch_label",
    "rescue",
    "break",
}


class IndependentValidationError(RuntimeError):
    """The independent current64 reference-cache replay failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentValidationError(message)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _walk_keys(value: Any) -> Sequence[str]:
    output: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            output.append(str(key))
            output.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(_walk_keys(child))
    return output


def read_source_manifest(
    path: Path, validation_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[int], bool]:
    require(path.is_file(), "source manifest absent")
    manifest = json.loads(path.read_text())
    require(isinstance(manifest, dict), "source manifest is not an object")
    required = {
        "version",
        "status",
        "claim_level",
        "population",
        "records",
        "old_map_reuse_pairs",
        "missing_map_pairs",
        "reference_reuse",
        "bindings",
        "access",
        "scientific_GO_or_NO_GO",
        "automatic_stage_advance",
        "next_authorized_stage",
        "logical_sha256",
    }
    interface = required.issubset(manifest)
    union_a = manifest.get("fresh_reference_union")
    union_b = manifest.get("union_physical_rows")
    aliases_agree = not (
        union_a is not None and union_b is not None and union_a != union_b
    )
    union_value = union_a if union_a is not None else union_b
    require(isinstance(union_value, list), "fresh reference union absent")
    union = sorted(union_value)
    bindings = manifest.get("bindings", {})
    access = manifest.get("access", {})
    zero_access = isinstance(access, Mapping) and all(
        value == 0
        for key, value in access.items()
        if any(term in str(key) for term in ("target", "label", "supergroup", "outcome"))
    )
    valid = (
        interface
        and isinstance(manifest.get("status"), str)
        and manifest["status"].endswith("_READY")
        and "ABORT" not in manifest["status"]
        and "NO_GO" not in manifest["status"]
        and isinstance(manifest.get("claim_level"), str)
        and "TARGET_FREE" in manifest["claim_level"]
        and manifest.get("logical_sha256") == e0.logical_sha256(manifest)
        and aliases_agree
        and union_value == union
        and len(union) == len(set(union)) == EXPECTED_UNION_COUNT
        and all(type(row) is int and row in range(5413) for row in union)
        and isinstance(bindings, Mapping)
        and bindings.get("rematerialization_contract_sha256")
        == e0.sha256_file(CONTRACT)
        and isinstance(manifest.get("reference_reuse"), Mapping)
        and not (set(_walk_keys(manifest)) & FORBIDDEN_SOURCE_KEYS)
        and zero_access
        and manifest.get("scientific_GO_or_NO_GO") is None
        and manifest.get("automatic_stage_advance") is False
    )
    require(validation_path.is_file(), "source-manifest independent validation absent")
    validation = json.loads(validation_path.read_text())
    validation_valid = (
        isinstance(validation, dict)
        and validation.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_VALIDATED"
        and isinstance(validation.get("checks"), Mapping)
        and bool(validation["checks"])
        and all(type(value) is bool and value for value in validation["checks"].values())
        and validation.get("manifest_sha256") == e0.sha256_file(path)
        and validation.get("manifest_logical_sha256") == manifest.get("logical_sha256")
        and validation.get("logical_sha256") == e0.logical_sha256(validation)
        and validation.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_COMPLETION"
    )
    return manifest, validation, union, valid and validation_valid


def read_old_cache() -> tuple[dict[str, Any], dict[int, dict[str, Any]], bool]:
    hashes = (
        e0.sha256_file(OLD_PAYLOAD) == EXPECTED_OLD_PAYLOAD_SHA256
        and e0.sha256_file(OLD_RESULT) == EXPECTED_OLD_RESULT_SHA256
        and e0.sha256_file(OLD_VALIDATION) == EXPECTED_OLD_VALIDATION_SHA256
    )
    result = json.loads(OLD_RESULT.read_text())
    validation = json.loads(OLD_VALIDATION.read_text())
    payload = torch.load(OLD_PAYLOAD, map_location="cpu", weights_only=False, mmap=True)
    old_union = payload.get("reference_union")
    references = payload.get("references")
    valid = (
        hashes
        and result.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_READY"
        and result.get("logical_sha256") == e0.logical_sha256(result)
        and result.get("payload_sha256") == EXPECTED_OLD_PAYLOAD_SHA256
        and validation.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_VALIDATED"
        and validation.get("logical_sha256") == e0.logical_sha256(validation)
        and validation.get("payload_sha256") == EXPECTED_OLD_PAYLOAD_SHA256
        and all(validation.get("checks", {}).values())
        and isinstance(old_union, list)
        and old_union == sorted(old_union)
        and len(old_union) == len(set(old_union)) == EXPECTED_OLD_COUNT
        and isinstance(references, dict)
        and set(references) == set(old_union)
    )
    return payload, references, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        "--source-validation", type=Path, default=DEFAULT_SOURCE_VALIDATION
    )
    args = parser.parse_args()
    require(not OUT.exists(), f"immutable validation exists: {OUT}")
    require(PAYLOAD.is_file() and RESULT.is_file(), "producer outputs absent")

    manifest, source_validation, fresh_union, source_manifest_valid = read_source_manifest(
        args.source_manifest, args.source_validation
    )
    old_payload, old_references, old_cache_valid = read_old_cache()
    payload = torch.load(PAYLOAD, map_location="cpu", weights_only=False, mmap=True)
    result = json.loads(RESULT.read_text())
    old_rows = set(old_payload["reference_union"])
    reused_rows = sorted(set(fresh_union) & old_rows)
    missing_rows = sorted(set(fresh_union) - old_rows)
    partition_valid = (
        len(reused_rows) == EXPECTED_REUSE_COUNT
        and tuple(missing_rows) == EXPECTED_MISSING_ROWS
    )

    gallery = build_gallery_source(verify_cache_file_sha256=True)
    legacy = build_prejoin_sources(gallery=gallery)
    legacy_union = tuple(
        sorted(
            {
                row
                for source_record in legacy.records
                for row in source_record.candidate_physical_rows
            }
        )
    )
    require(
        len(legacy_union) == LEGACY_SPATIAL_UNION_COUNT,
        "independent 4,976-row spatial union drift",
    )
    work_by_row = gallery_work_ordinal_by_physical_row(legacy_union)
    spatial_cache = ValidatedSpatialCacheResolver(SPATIAL_CACHE_ROOT)
    references = payload.get("references", {})
    reference_checks: list[bool] = []
    exact_reuse_count = 0
    independently_resolved_count = 0
    source_kinds: Counter[str] = Counter()
    reused_set = set(reused_rows)
    for ordinal, physical_row in enumerate(fresh_union):
        record = references.get(physical_row, {}) if isinstance(references, dict) else {}
        valid = set(record) == REFERENCE_KEYS
        if not valid:
            reference_checks.append(False)
            continue
        observed = torch.as_tensor(record["tokens"]).detach().cpu().contiguous()
        raw_path = gallery.raw_paths[physical_row]
        common = (
            record["physical_row"] == physical_row
            and Path(record["source_path"]).resolve() == raw_path.resolve()
            and record["source_image_sha256"] == e0.sha256_file(raw_path)
            and observed.dtype == torch.float16
            and tuple(observed.shape)
            == (math.prod(tuple(record["grid_shape"])), 128)
            and bool(torch.isfinite(observed).all())
            and record["tokens_sha256"] == tensor_sha256(observed)
            and isinstance(record["source_logical_sha256"], str)
            and len(record["source_logical_sha256"]) == 64
        )
        if physical_row in reused_set:
            old = old_references[physical_row]
            expected = torch.as_tensor(old["tokens"]).detach().cpu().contiguous()
            non_tensor_exact = all(
                record[key] == old[key] for key in REFERENCE_KEYS - {"tokens"}
            )
            exact = common and non_tensor_exact and torch.equal(observed, expected)
            exact_reuse_count += int(exact)
        else:
            if physical_row not in work_by_row:
                exact = False
            else:
                entry = spatial_cache.get("gallery", work_by_row[physical_row])
                expected = entry.tokens.detach().cpu().contiguous()
                exact = (
                    common
                    and entry.physical_row == physical_row
                    and tuple(record["grid_shape"]) == tuple(entry.grid_shape)
                    and torch.equal(observed, expected)
                    and record["tokens_sha256"] == entry.tokens_sha256
                    and record["source_kind"] == "VALIDATED_4976_CACHE"
                    and record["source_logical_sha256"]
                    == entry.entry_logical_sha256
                )
            independently_resolved_count += int(exact)
        source_kinds[str(record["source_kind"])] += 1
        reference_checks.append(bool(exact))
        if (ordinal + 1) % 256 == 0 or ordinal + 1 == len(fresh_union):
            print(
                json.dumps(
                    {
                        "event": "fresh_d1_current64_reference_cache_validation_progress",
                        "done": ordinal + 1,
                        "total": len(fresh_union),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    expected_reuse = {
        "old_reference_count": EXPECTED_OLD_COUNT,
        "fresh_current64_reference_count": EXPECTED_UNION_COUNT,
        "reused_reference_count": EXPECTED_REUSE_COUNT,
        "resolver_reference_count": len(EXPECTED_MISSING_ROWS),
        "reused_physical_rows_sha256": e0.logical_sha256(
            {"rows": reused_rows, "logical_sha256": ""}
        ),
        "resolver_physical_rows": list(EXPECTED_MISSING_ROWS),
        "token_or_metadata_change_count_on_reuse": 0,
    }
    expected_bindings = {
        "contract_sha256": e0.sha256_file(CONTRACT),
        "source_manifest_path": str(args.source_manifest.resolve()),
        "source_manifest_sha256": e0.sha256_file(args.source_manifest),
        "source_manifest_logical_sha256": manifest["logical_sha256"],
        "source_manifest_validation_path": str(args.source_validation.resolve()),
        "source_manifest_validation_sha256": e0.sha256_file(args.source_validation),
        "source_manifest_validation_logical_sha256": source_validation[
            "logical_sha256"
        ],
        "old_reference_cache_payload_sha256": EXPECTED_OLD_PAYLOAD_SHA256,
        "old_reference_cache_result_sha256": EXPECTED_OLD_RESULT_SHA256,
        "old_reference_cache_validation_sha256": EXPECTED_OLD_VALIDATION_SHA256,
        "gallery_cache_sha256": gallery.gallery_cache_sha256,
        "gallery_corrected_mapping_sha256": gallery.corrected_mapping_sha256,
        "hybrid_resolver_source_sha256": e0.sha256_file(RESOLVER_SOURCE),
        "producer_sha256": e0.sha256_file(PRODUCER),
    }
    expected_access = {
        "source_manifest_read_count": 1,
        "source_manifest_validation_read_count": 1,
        "old_reference_cache_payload_read_count": 1,
        "target_identity_read_count": 0,
        "target_label_read_count": 0,
        "target_role_read_count": 0,
        "supergroup_read_count": 0,
        "outcome_read_count": 0,
        "sealed_read_count": 0,
        "hybrid_resolver_call_count": len(EXPECTED_MISSING_ROWS),
        "encoder_model_load_count": 0,
        "encoder_model_forward_count": 0,
        "model_update_count": 0,
    }
    expected_kind_counts = dict(sorted(source_kinds.items()))
    checks = {
        "source_manifest_interface_and_target_free_seal": source_manifest_valid,
        "old_3004_cache_independently_bound": old_cache_valid,
        "fresh_union_partition_2801_plus_4": partition_valid,
        "payload_exact_schema": set(payload) == PAYLOAD_KEYS,
        "current64_staging_boundary": payload.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_READY"
        and payload.get("claim_level")
        == "TARGET_FREE_CURRENT64_REFERENCE_TOKEN_UNION_STAGING_ONLY"
        and payload.get("contract_population_complete") is False
        and payload.get("current64_query_count") == EXPECTED_CURRENT64_QUERY_COUNT
        and payload.get("pair64_pending") == EXPECTED_PAIR64_PENDING,
        "fresh_union_exact": payload.get("reference_union") == fresh_union
        and isinstance(references, dict)
        and set(references) == set(fresh_union),
        "reused_records_byte_and_metadata_exact": exact_reuse_count
        == EXPECTED_REUSE_COUNT,
        "four_new_rows_independently_resolved": independently_resolved_count
        == len(EXPECTED_MISSING_ROWS),
        "all_reference_records_valid": len(reference_checks)
        == EXPECTED_UNION_COUNT
        and all(reference_checks),
        "source_kind_population": payload.get("source_kind_counts")
        == expected_kind_counts,
        "reuse_receipt": payload.get("reuse") == expected_reuse,
        "bindings": payload.get("bindings") == expected_bindings,
        "access": payload.get("access") == expected_access,
        "producer_result": set(result) == RESULT_KEYS
        and result.get("schema_version")
        == "routea_n2_fresh_d1_matched_three_arm_current64_reference_cache_result_v1_20260904"
        and result.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_READY"
        and result.get("claim_level")
        == "TARGET_FREE_CURRENT64_REFERENCE_TOKEN_UNION_STAGING_ONLY"
        and result.get("contract_population_complete") is False
        and result.get("current64_query_count") == EXPECTED_CURRENT64_QUERY_COUNT
        and result.get("pair64_pending") == EXPECTED_PAIR64_PENDING
        and result.get("reference_count") == EXPECTED_UNION_COUNT
        and result.get("source_kind_counts") == expected_kind_counts
        and result.get("reuse") == expected_reuse
        and result.get("payload_sha256") == e0.sha256_file(PAYLOAD)
        and result.get("bindings") == expected_bindings
        and result.get("access") == expected_access
        and result.get("scientific_GO_or_NO_GO") is None
        and result.get("ownership_GO_or_NO_GO") is None
        and result.get("automatic_stage_advance") is False
        and result.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARDS"
        and result.get("logical_sha256") == e0.logical_sha256(result),
    }
    passed = all(checks.values())
    value = {
        "schema_version": "routea_n2_fresh_d1_matched_three_arm_current64_reference_cache_validation_v1_20260904",
        "status": (
            "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_VALIDATED"
            if passed
            else "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_VALIDATION_ABORT"
        ),
        "claim_level": "INDEPENDENT_TARGET_FREE_CURRENT64_REFERENCE_TOKEN_UNION_STAGING_ONLY",
        "checks": checks,
        "contract_population_complete": False,
        "current64_query_count": EXPECTED_CURRENT64_QUERY_COUNT,
        "pair64_pending": EXPECTED_PAIR64_PENDING,
        "reference_count": len(fresh_union),
        "reused_reference_count": exact_reuse_count,
        "independently_resolved_reference_count": independently_resolved_count,
        "source_kind_counts": expected_kind_counts,
        "source_manifest_sha256": e0.sha256_file(args.source_manifest),
        "producer_result_sha256": e0.sha256_file(RESULT),
        "payload_sha256": e0.sha256_file(PAYLOAD),
        "target_identity_read_count": 0,
        "model_update_count": 0,
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": (
            "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARDS"
            if passed
            else None
        ),
        "validator_sha256": e0.sha256_file(Path(__file__).resolve()),
        "logical_sha256": "",
    }
    value["logical_sha256"] = e0.logical_sha256(value)
    atomic_json(OUT, value)
    print(
        json.dumps(
            {"status": value["status"], "checks": checks}, sort_keys=True
        ),
        flush=True,
    )
    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
