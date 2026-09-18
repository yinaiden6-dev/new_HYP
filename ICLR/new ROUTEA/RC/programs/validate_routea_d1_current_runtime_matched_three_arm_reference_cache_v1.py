#!/usr/bin/env python3
"""Independent low-level replay of the matched-three-arm reference cache."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_routea_d1_current_runtime_bridge_e0_v1 as e0  # noqa: E402
from rc_aslo_xf.colnomic_proposal_tokens import proposal_grid_sha256  # noqa: E402
from rc_aslo_xf.conditional_colnomic_ms_proposal import (  # noqa: E402
    ValidatedSpatialCacheResolver,
    gallery_work_ordinal_by_physical_row,
)
from rc_aslo_xf.conditional_rep_sources import (  # noqa: E402
    build_gallery_source,
    build_prejoin_sources,
    canonical_sha256,
)
from rc_aslo_xf.cw1_sr0_s8_feature_runtime_v1 import (  # noqa: E402
    SCHEMA_VERSION,
    SPATIAL_CACHE_ROOT,
    _default_gallery_embedding_loader,
    _default_processor_factory,
    _default_processor_source_receipt,
    _default_recovery_grid_builder,
    tensor_sha256,
)


CONTRACT = ROOT / "plan/ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_V1_20260902.md"
PREJOIN = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1"
PREJOIN_VALIDATION = PREJOIN / "validation.json"
JOIN_HANDOFF = ROOT / "results/routea_d1_current_runtime_64_label_join_v1/independent_validation.json"
OUT_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_reference_cache_v1"
PAYLOAD = OUT_ROOT / "payload.pt"
RESULT = OUT_ROOT / "result.json"
OUT = OUT_ROOT / "independent_validation.json"
EXPECTED_UNION = 3004
EXPECTED_SOURCE_KINDS = Counter(
    {"VALIDATED_4976_CACHE": 2994, "CPU_FROZEN_PROCESSOR_RECOVERY": 10}
)
HANDOFF_KEYS = {
    "schema_version",
    "status",
    "checks",
    "producer_result_sha256",
    "query_count",
    "target_role_read_count",
    "model_update_count",
    "next_authorized_stage",
    "logical_sha256",
}
HANDOFF_CHECK_KEYS = {
    "access",
    "bindings",
    "case_population",
    "cases_exact",
    "claim_boundary",
    "decision",
    "summary_all",
    "summary_by_fold",
    "summary_by_role",
    "summary_by_track",
}
PAYLOAD_KEYS = {
    "schema_version",
    "status",
    "claim_level",
    "reference_union",
    "references",
    "source_kind_counts",
    "bindings",
    "access",
}
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
BINDING_KEYS = {
    "contract_sha256",
    "prejoin_validation_sha256",
    "label_join_handoff_sha256",
    "gallery_cache_sha256",
    "gallery_corrected_mapping_sha256",
    "prejoin_shards",
}
ACCESS_KEYS = {
    "label_join_handoff_read_count",
    "target_identity_read_count",
    "target_role_read_count",
    "supergroup_read_count",
    "sealed_read_count",
    "encoder_model_load_count",
    "encoder_model_forward_count",
    "model_update_count",
}
RESULT_KEYS = {
    "schema_version",
    "status",
    "claim_level",
    "reference_count",
    "source_kind_counts",
    "payload_sha256",
    "bindings",
    "access",
    "next_authorized_stage",
    "logical_sha256",
}


def atomic_json(path: Path, value: dict) -> None:
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


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"immutable reference-cache validation exists: {OUT}")
    result = json.loads(RESULT.read_text())
    payload = torch.load(PAYLOAD, map_location="cpu", weights_only=False, mmap=True)
    prejoin_validation = json.loads(PREJOIN_VALIDATION.read_text())
    handoff = json.loads(JOIN_HANDOFF.read_text())

    handoff_exact = (
        set(handoff) == HANDOFF_KEYS
        and handoff.get("schema_version")
        == "routea_d1_current_runtime_64_label_join_independent_validation_v1_20260902"
        and handoff.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_64_LABEL_JOIN_INDEPENDENT_VALIDATION_PASS"
        and handoff.get("logical_sha256") == e0.logical_sha256(handoff)
        and set(handoff.get("checks", {})) == HANDOFF_CHECK_KEYS
        and all(type(value) is bool and value for value in handoff["checks"].values())
        and handoff.get("query_count") == 64
        and handoff.get("target_role_read_count") == 64
        and handoff.get("model_update_count") == 0
        and handoff.get("next_authorized_stage")
        == "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_TARGET_FREE_MATERIALIZATION"
    )
    sealed_shards = {
        int(item["shard"]): item for item in prejoin_validation.get("shards", [])
    }
    expected_union: set[int] = set()
    observed_shards: list[dict] = []
    for shard in range(8):
        payload_path = PREJOIN / f"shard{shard:02d}/payload.pt"
        receipt_path = PREJOIN / f"shard{shard:02d}/receipt.json"
        validation_path = PREJOIN / f"shard{shard:02d}/validation.json"
        sealed = sealed_shards.get(shard, {})
        if not (
            sealed.get("payload_sha256") == e0.sha256_file(payload_path)
            and sealed.get("receipt_sha256") == e0.sha256_file(receipt_path)
            and sealed.get("validation_sha256") == e0.sha256_file(validation_path)
        ):
            raise RuntimeError("independent prejoin shard seal drift")
        value = torch.load(
            payload_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        for record in value["records"]:
            expected_union.update(map(int, record["raw_candidate_physical_rows"]))
            expected_union.update(map(int, record["d1_candidate_physical_rows"]))
        observed_shards.append(
            {
                "shard": shard,
                "payload_sha256": sealed["payload_sha256"],
                "receipt_sha256": sealed["receipt_sha256"],
                "validation_sha256": sealed["validation_sha256"],
            }
        )
    expected_rows = sorted(expected_union)
    references = payload.get("references", {})

    gallery = build_gallery_source(verify_cache_file_sha256=True)
    legacy_prejoin = build_prejoin_sources(gallery=gallery)
    legacy_union = tuple(
        sorted(
            {
                row
                for record in legacy_prejoin.records
                for row in record.candidate_physical_rows
            }
        )
    )
    if len(legacy_union) != 4976:
        raise RuntimeError("independent legacy spatial union drift")
    work_by_row = gallery_work_ordinal_by_physical_row(legacy_union)
    spatial_cache = ValidatedSpatialCacheResolver(SPATIAL_CACHE_ROOT)
    processor = None
    processor_receipt = None
    gallery_embeddings = None
    source_kinds: Counter[str] = Counter()
    reference_checks: list[bool] = []

    for ordinal, physical_row in enumerate(expected_rows):
        record = references.get(physical_row, {})
        if set(record) != REFERENCE_KEYS:
            raise RuntimeError("reference record schema or forbidden-field drift")
        if physical_row in work_by_row:
            entry = spatial_cache.get("gallery", work_by_row[physical_row])
            if entry.physical_row != physical_row:
                raise RuntimeError("independent cached physical-row drift")
            expected_tokens = entry.tokens.detach().cpu().contiguous()
            expected_grid = tuple(entry.grid_shape)
            expected_kind = "VALIDATED_4976_CACHE"
            expected_logical = entry.entry_logical_sha256
        else:
            if processor is None:
                processor = _default_processor_factory()
                processor_receipt = dict(_default_processor_source_receipt())
                gallery_embeddings = _default_gallery_embedding_loader()
            cached = gallery_embeddings[physical_row].detach().cpu().contiguous()
            grid = _default_recovery_grid_builder(
                cached, gallery.raw_paths[physical_row], processor
            )
            expected_tokens = (
                grid.tokens.detach().cpu().to(torch.float16).contiguous()
            )
            expected_grid = tuple(grid.grid_shape)
            expected_kind = "CPU_FROZEN_PROCESSOR_RECOVERY"
            recovery_payload = {
                "schema_version": SCHEMA_VERSION,
                "source_kind": expected_kind,
                "physical_row": physical_row,
                "raw_path": str(gallery.raw_paths[physical_row]),
                "raw_file_sha256": e0.sha256_file(gallery.raw_paths[physical_row]),
                "frozen_valid_embedding_sha256": tensor_sha256(cached),
                "grid_shape": list(expected_grid),
                "image_token_indices": grid.image_token_indices.tolist(),
                "proposal_grid_sha256": proposal_grid_sha256(grid),
                "tokens_sha256": tensor_sha256(expected_tokens),
                "processor_source_logical_sha256": processor_receipt[
                    "logical_sha256"
                ],
                "encoder_model_load_count": 0,
                "encoder_model_forward_count": 0,
            }
            expected_logical = canonical_sha256(recovery_payload)
        source_kinds[expected_kind] += 1
        observed_tokens = torch.as_tensor(record["tokens"])
        reference_checks.append(
            record["physical_row"] == physical_row
            and Path(record["source_path"]).resolve()
            == gallery.raw_paths[physical_row].resolve()
            and record["source_image_sha256"]
            == e0.sha256_file(gallery.raw_paths[physical_row])
            and tuple(record["grid_shape"]) == expected_grid
            and observed_tokens.dtype == torch.float16
            and observed_tokens.shape == (expected_grid[0] * expected_grid[1], 128)
            and bool(torch.isfinite(observed_tokens).all())
            and torch.equal(observed_tokens, expected_tokens)
            and record["tokens_sha256"] == tensor_sha256(expected_tokens)
            and record["source_kind"] == expected_kind
            and record["source_logical_sha256"] == expected_logical
        )
        if (ordinal + 1) % 256 == 0 or ordinal + 1 == len(expected_rows):
            print(
                json.dumps(
                    {
                        "event": "reference_cache_validation_progress",
                        "done": ordinal + 1,
                        "total": len(expected_rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    expected_access = {
        "label_join_handoff_read_count": 1,
        "target_identity_read_count": 0,
        "target_role_read_count": 0,
        "supergroup_read_count": 0,
        "sealed_read_count": 0,
        "encoder_model_load_count": 0,
        "encoder_model_forward_count": 0,
        "model_update_count": 0,
    }
    expected_bindings = {
        "contract_sha256": e0.sha256_file(CONTRACT),
        "prejoin_validation_sha256": e0.sha256_file(PREJOIN_VALIDATION),
        "label_join_handoff_sha256": e0.sha256_file(JOIN_HANDOFF),
        "gallery_cache_sha256": gallery.gallery_cache_sha256,
        "gallery_corrected_mapping_sha256": gallery.corrected_mapping_sha256,
        "prejoin_shards": observed_shards,
    }
    expected_counts = dict(sorted(EXPECTED_SOURCE_KINDS.items()))
    checks = {
        "handoff_exact_sanitized": handoff_exact,
        "prejoin_authority": prejoin_validation.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and prejoin_validation.get("logical_sha256")
        == e0.logical_sha256(prejoin_validation)
        and all(prejoin_validation.get("checks", {}).values()),
        "payload_exact_schema": set(payload) == PAYLOAD_KEYS
        and set(payload.get("bindings", {})) == BINDING_KEYS
        and set(payload.get("access", {})) == ACCESS_KEYS,
        "payload_envelope": payload.get("schema_version")
        == "routea_d1_current_runtime_matched_three_arm_reference_cache_payload_v1_20260902"
        and payload.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_READY"
        and payload.get("claim_level") == "TARGET_FREE_REFERENCE_TOKEN_UNION_ONLY",
        "bindings": payload.get("bindings") == expected_bindings,
        "union_population": expected_rows == payload.get("reference_union")
        and len(expected_rows) == len(references) == EXPECTED_UNION,
        "source_kind_population": source_kinds == EXPECTED_SOURCE_KINDS
        and payload.get("source_kind_counts") == expected_counts,
        "reference_replay": all(reference_checks),
        "access": payload.get("access") == expected_access,
        "producer_result_exact": set(result) == RESULT_KEYS
        and result.get("schema_version")
        == "routea_d1_current_runtime_matched_three_arm_reference_cache_result_v1_20260902"
        and result.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_READY"
        and result.get("claim_level") == "TARGET_FREE_REFERENCE_TOKEN_UNION_ONLY"
        and result.get("reference_count") == EXPECTED_UNION
        and result.get("source_kind_counts") == expected_counts
        and result.get("payload_sha256") == e0.sha256_file(PAYLOAD)
        and result.get("bindings") == expected_bindings
        and result.get("access") == expected_access
        and result.get("next_authorized_stage")
        == "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_SHARDS"
        and result.get("logical_sha256") == e0.logical_sha256(result),
    }
    passed = all(checks.values())
    value = {
        "schema_version": "routea_d1_current_runtime_matched_three_arm_reference_cache_validation_v1_20260902",
        "status": (
            "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_VALIDATED"
            if passed
            else "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_VALIDATION_ABORT"
        ),
        "checks": checks,
        "reference_count": len(expected_rows),
        "source_kind_counts": dict(sorted(source_kinds.items())),
        "producer_result_sha256": e0.sha256_file(RESULT),
        "payload_sha256": e0.sha256_file(PAYLOAD),
        "target_identity_read_count": 0,
        "model_update_count": 0,
        "next_authorized_stage": (
            "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_SHARDS"
            if passed
            else None
        ),
        "logical_sha256": "",
    }
    value["logical_sha256"] = e0.logical_sha256(value)
    atomic_json(OUT, value)
    print(json.dumps({"status": value["status"], "checks": checks}, sort_keys=True))
    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
