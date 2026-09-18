#!/usr/bin/env python3
"""Materialize the current64 fresh-D1 reference-token union without labels.

The 2,805-row union is supplied by the separately sealed source manifest.  The
2,801-row intersection with the prior 3,004-row cache is copied byte-for-byte
at tensor and metadata level.  Only four genuinely new rows are resolved via
the frozen HybridSpatialReferenceResolver.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_routea_d1_current_runtime_bridge_e0_v1 as e0  # noqa: E402
from rc_aslo_xf.conditional_rep_sources import build_gallery_source  # noqa: E402
from rc_aslo_xf.cw1_sr0_s8_feature_runtime_v1 import (  # noqa: E402
    HybridSpatialReferenceResolver,
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


class ReferenceCacheError(RuntimeError):
    """A target-free source or exact-reuse contract failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReferenceCacheError(message)


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


def load_source_manifest(
    path: Path, validation_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    require(path.is_file(), f"source manifest absent: {path}")
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
    require(required.issubset(manifest), "source manifest required interface absent")
    status = manifest.get("status")
    require(
        isinstance(status, str)
        and status.endswith("_READY")
        and "ABORT" not in status
        and "NO_GO" not in status,
        "source manifest status is not READY",
    )
    require(
        isinstance(manifest.get("claim_level"), str)
        and "TARGET_FREE" in manifest["claim_level"],
        "source manifest is not target-free",
    )
    require(
        manifest.get("logical_sha256") == e0.logical_sha256(manifest),
        "source manifest logical hash drift",
    )
    union_a = manifest.get("fresh_reference_union")
    union_b = manifest.get("union_physical_rows")
    if union_a is not None and union_b is not None:
        require(union_a == union_b, "source manifest union aliases disagree")
    union_value = union_a if union_a is not None else union_b
    require(isinstance(union_value, list), "fresh reference union absent")
    require(
        all(type(row) is int and row in range(5413) for row in union_value),
        "fresh reference union contains invalid rows",
    )
    union = sorted(union_value)
    require(
        union_value == union
        and len(union) == len(set(union)) == EXPECTED_UNION_COUNT,
        "fresh current64 reference union population drift",
    )
    bindings = manifest.get("bindings")
    require(isinstance(bindings, Mapping), "source manifest bindings absent")
    require(
        bindings.get("rematerialization_contract_sha256")
        == e0.sha256_file(CONTRACT),
        "source manifest contract binding drift",
    )
    require(
        isinstance(manifest.get("reference_reuse"), Mapping),
        "source manifest reference-reuse receipt absent",
    )
    require(
        not (set(_walk_keys(manifest)) & FORBIDDEN_SOURCE_KEYS),
        "source manifest contains forbidden postjoin fields",
    )
    access = manifest.get("access")
    require(isinstance(access, Mapping), "source manifest access receipt absent")
    for key, value in access.items():
        if any(term in str(key) for term in ("target", "label", "supergroup", "outcome")):
            require(value == 0, f"source manifest forbidden access is nonzero: {key}")
    require(
        manifest.get("scientific_GO_or_NO_GO") is None
        and manifest.get("automatic_stage_advance") is False,
        "source manifest claim boundary drift",
    )
    require(validation_path.is_file(), "source-manifest independent validation absent")
    validation = json.loads(validation_path.read_text())
    require(
        isinstance(validation, dict)
        and validation.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_VALIDATED"
        and isinstance(validation.get("checks"), Mapping)
        and bool(validation["checks"])
        and all(type(value) is bool and value for value in validation["checks"].values())
        and validation.get("manifest_sha256") == e0.sha256_file(path)
        and validation.get("manifest_logical_sha256") == manifest["logical_sha256"]
        and validation.get("logical_sha256") == e0.logical_sha256(validation)
        and validation.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_COMPLETION",
        "source-manifest independent validation drift",
    )
    return manifest, validation, union


def validate_old_cache() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    require(
        e0.sha256_file(OLD_PAYLOAD) == EXPECTED_OLD_PAYLOAD_SHA256
        and e0.sha256_file(OLD_RESULT) == EXPECTED_OLD_RESULT_SHA256
        and e0.sha256_file(OLD_VALIDATION) == EXPECTED_OLD_VALIDATION_SHA256,
        "old 3,004-row cache physical hash drift",
    )
    result = json.loads(OLD_RESULT.read_text())
    validation = json.loads(OLD_VALIDATION.read_text())
    require(
        result.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_READY"
        and result.get("logical_sha256") == e0.logical_sha256(result)
        and result.get("payload_sha256") == EXPECTED_OLD_PAYLOAD_SHA256,
        "old reference-cache result drift",
    )
    require(
        validation.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_VALIDATED"
        and validation.get("logical_sha256") == e0.logical_sha256(validation)
        and validation.get("payload_sha256") == EXPECTED_OLD_PAYLOAD_SHA256
        and all(validation.get("checks", {}).values()),
        "old reference-cache validation drift",
    )
    payload = torch.load(OLD_PAYLOAD, map_location="cpu", weights_only=False, mmap=True)
    old_union = payload.get("reference_union")
    references = payload.get("references")
    require(
        isinstance(old_union, list)
        and old_union == sorted(old_union)
        and len(old_union) == len(set(old_union)) == EXPECTED_OLD_COUNT
        and isinstance(references, dict)
        and set(references) == set(old_union),
        "old reference-cache payload population drift",
    )
    return payload, references


def checked_old_record(
    record: Mapping[str, Any], physical_row: int, raw_path: Path
) -> dict[str, Any]:
    require(set(record) == REFERENCE_KEYS, "old reference record schema drift")
    tokens = torch.as_tensor(record["tokens"]).detach().cpu().contiguous()
    grid = tuple(record["grid_shape"])
    require(
        record["physical_row"] == physical_row
        and Path(record["source_path"]).resolve() == raw_path.resolve()
        and record["source_image_sha256"] == e0.sha256_file(raw_path)
        and len(grid) == 2
        and all(type(item) is int and item > 0 for item in grid)
        and tokens.dtype == torch.float16
        and tokens.shape == (math.prod(grid), 128)
        and bool(torch.isfinite(tokens).all())
        and record["tokens_sha256"] == tensor_sha256(tokens)
        and record["source_kind"]
        in {"VALIDATED_4976_CACHE", "CPU_FROZEN_PROCESSOR_RECOVERY"}
        and isinstance(record["source_logical_sha256"], str)
        and len(record["source_logical_sha256"]) == 64,
        f"old reference record replay failed for row {physical_row}",
    )
    output = dict(record)
    output["tokens"] = tokens.clone()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        "--source-validation", type=Path, default=DEFAULT_SOURCE_VALIDATION
    )
    args = parser.parse_args()
    require(not OUT_ROOT.exists(), f"immutable output exists: {OUT_ROOT}")

    manifest, source_validation, fresh_union = load_source_manifest(
        args.source_manifest, args.source_validation
    )
    old_payload, old_references = validate_old_cache()
    old_rows = set(old_payload["reference_union"])
    reused_rows = sorted(set(fresh_union) & old_rows)
    missing_rows = sorted(set(fresh_union) - old_rows)
    require(
        len(reused_rows) == EXPECTED_REUSE_COUNT
        and tuple(missing_rows) == EXPECTED_MISSING_ROWS,
        "fresh/old reference-cache partition drift",
    )

    gallery = build_gallery_source(verify_cache_file_sha256=True)
    require(len(gallery.raw_paths) == 5413, "gallery physical-row population drift")
    resolver = HybridSpatialReferenceResolver(gallery_source=gallery)
    references: dict[int, dict[str, Any]] = {}
    source_kinds: Counter[str] = Counter()
    reused_set = set(reused_rows)
    for ordinal, physical_row in enumerate(fresh_union):
        raw_path = gallery.raw_paths[physical_row]
        if physical_row in reused_set:
            record = checked_old_record(
                old_references[physical_row], physical_row, raw_path
            )
        else:
            resolved = resolver.resolve(physical_row)
            tokens = resolved.tokens.detach().cpu().contiguous()
            require(
                resolved.source_kind == "VALIDATED_4976_CACHE"
                and tokens.dtype == torch.float16
                and tokens.shape == (math.prod(resolved.grid_shape), 128)
                and bool(torch.isfinite(tokens).all())
                and resolved.tokens_sha256 == tensor_sha256(tokens),
                f"new reference resolver replay failed for row {physical_row}",
            )
            record = {
                "physical_row": physical_row,
                "source_path": str(raw_path),
                "source_image_sha256": e0.sha256_file(raw_path),
                "grid_shape": tuple(map(int, resolved.grid_shape)),
                "tokens": tokens.clone(),
                "tokens_sha256": str(resolved.tokens_sha256),
                "source_kind": str(resolved.source_kind),
                "source_logical_sha256": str(resolved.source_logical_sha256),
            }
        source_kinds[str(record["source_kind"])] += 1
        references[physical_row] = record
        if (ordinal + 1) % 256 == 0 or ordinal + 1 == len(fresh_union):
            print(
                json.dumps(
                    {
                        "event": "fresh_d1_current64_reference_cache_progress",
                        "done": ordinal + 1,
                        "total": len(fresh_union),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    bindings = {
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
        "producer_sha256": e0.sha256_file(Path(__file__).resolve()),
    }
    reuse = {
        "old_reference_count": EXPECTED_OLD_COUNT,
        "fresh_current64_reference_count": EXPECTED_UNION_COUNT,
        "reused_reference_count": EXPECTED_REUSE_COUNT,
        "resolver_reference_count": len(missing_rows),
        "reused_physical_rows_sha256": e0.logical_sha256(
            {"rows": reused_rows, "logical_sha256": ""}
        ),
        "resolver_physical_rows": missing_rows,
        "token_or_metadata_change_count_on_reuse": 0,
    }
    access = {
        "source_manifest_read_count": 1,
        "source_manifest_validation_read_count": 1,
        "old_reference_cache_payload_read_count": 1,
        "target_identity_read_count": 0,
        "target_label_read_count": 0,
        "target_role_read_count": 0,
        "supergroup_read_count": 0,
        "outcome_read_count": 0,
        "sealed_read_count": 0,
        "hybrid_resolver_call_count": len(missing_rows),
        "encoder_model_load_count": 0,
        "encoder_model_forward_count": 0,
        "model_update_count": 0,
    }
    payload = {
        "schema_version": "routea_n2_fresh_d1_matched_three_arm_current64_reference_cache_payload_v1_20260904",
        "status": "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_READY",
        "claim_level": "TARGET_FREE_CURRENT64_REFERENCE_TOKEN_UNION_STAGING_ONLY",
        "contract_population_complete": False,
        "current64_query_count": EXPECTED_CURRENT64_QUERY_COUNT,
        "pair64_pending": EXPECTED_PAIR64_PENDING,
        "reference_union": fresh_union,
        "references": references,
        "source_kind_counts": dict(sorted(source_kinds.items())),
        "reuse": reuse,
        "bindings": bindings,
        "access": access,
    }
    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".n2-fresh-d1-current64-refcache-", dir=OUT_ROOT.parent)
    )
    try:
        staged_payload = staging / "payload.pt"
        staged_result = staging / "result.json"
        torch.save(payload, staged_payload)
        result = {
            "schema_version": "routea_n2_fresh_d1_matched_three_arm_current64_reference_cache_result_v1_20260904",
            "status": payload["status"],
            "claim_level": payload["claim_level"],
            "contract_population_complete": False,
            "current64_query_count": EXPECTED_CURRENT64_QUERY_COUNT,
            "pair64_pending": EXPECTED_PAIR64_PENDING,
            "reference_count": len(fresh_union),
            "source_kind_counts": payload["source_kind_counts"],
            "reuse": reuse,
            "payload_sha256": e0.sha256_file(staged_payload),
            "bindings": bindings,
            "access": access,
            "scientific_GO_or_NO_GO": None,
            "ownership_GO_or_NO_GO": None,
            "automatic_stage_advance": False,
            "next_authorized_stage": "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARDS",
            "logical_sha256": "",
        }
        result["logical_sha256"] = e0.logical_sha256(result)
        staged_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.rename(staging, OUT_ROOT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(
        json.dumps(
            {
                "status": result["status"],
                "reference_count": result["reference_count"],
                "reused": EXPECTED_REUSE_COUNT,
                "resolved": missing_rows,
                "contract_population_complete": False,
                "pair64_pending": EXPECTED_PAIR64_PENDING,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
