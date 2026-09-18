#!/usr/bin/env python3
"""Materialize the target-free 3,004-row reference union for matched A/B/C."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_routea_d1_current_runtime_bridge_e0_v1 as e0  # noqa: E402
from rc_aslo_xf.conditional_rep_sources import build_gallery_source  # noqa: E402
from rc_aslo_xf.cw1_sr0_s8_feature_runtime_v1 import (  # noqa: E402
    HybridSpatialReferenceResolver,
)


CONTRACT = ROOT / "plan/ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_V1_20260902.md"
PREJOIN = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1"
PREJOIN_VALIDATION = PREJOIN / "validation.json"
JOIN_HANDOFF = ROOT / "results/routea_d1_current_runtime_64_label_join_v1/independent_validation.json"
OUT_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_reference_cache_v1"
PAYLOAD = OUT_ROOT / "payload.pt"
RESULT = OUT_ROOT / "result.json"
EXPECTED_UNION = 3004
EXPECTED_SOURCE_KINDS = Counter(
    {"VALIDATED_4976_CACHE": 2994, "CPU_FROZEN_PROCESSOR_RECOVERY": 10}
)


def main() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError(f"immutable matched-three-arm reference cache exists: {OUT_ROOT}")
    prejoin_validation = json.loads(PREJOIN_VALIDATION.read_text())
    handoff = json.loads(JOIN_HANDOFF.read_text())
    if (
        prejoin_validation.get("status")
        != "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        or prejoin_validation.get("logical_sha256")
        != e0.logical_sha256(prejoin_validation)
        or not all(prejoin_validation.get("checks", {}).values())
        or handoff.get("status")
        != "ROUTEA_D1_CURRENT_RUNTIME_64_LABEL_JOIN_INDEPENDENT_VALIDATION_PASS"
        or handoff.get("next_authorized_stage")
        != "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_TARGET_FREE_MATERIALIZATION"
        or handoff.get("logical_sha256") != e0.logical_sha256(handoff)
        or handoff.get("query_count") != 64
    ):
        raise RuntimeError("prejoin or target-free next-stage handoff is not closed")

    sealed_shards = {
        int(item["shard"]): item for item in prejoin_validation["shards"]
    }
    if set(sealed_shards) != set(range(8)):
        raise RuntimeError("prejoin aggregate shard population drift")
    selected_rows: set[int] = set()
    shard_bindings: list[dict] = []
    for shard in range(8):
        payload_path = PREJOIN / f"shard{shard:02d}/payload.pt"
        receipt_path = PREJOIN / f"shard{shard:02d}/receipt.json"
        validation_path = PREJOIN / f"shard{shard:02d}/validation.json"
        sealed = sealed_shards[shard]
        if not (
            sealed["payload_sha256"] == e0.sha256_file(payload_path)
            and sealed["receipt_sha256"] == e0.sha256_file(receipt_path)
            and sealed["validation_sha256"] == e0.sha256_file(validation_path)
        ):
            raise RuntimeError("prejoin shard aggregate seal drift")
        payload = torch.load(
            payload_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        for record in payload["records"]:
            selected_rows.update(map(int, record["raw_candidate_physical_rows"]))
            selected_rows.update(map(int, record["d1_candidate_physical_rows"]))
        shard_bindings.append(
            {
                "shard": shard,
                "payload_sha256": sealed["payload_sha256"],
                "receipt_sha256": sealed["receipt_sha256"],
                "validation_sha256": sealed["validation_sha256"],
            }
        )
    union = sorted(selected_rows)
    if len(union) != EXPECTED_UNION:
        raise RuntimeError("matched-three-arm reference union drift")

    gallery = build_gallery_source(verify_cache_file_sha256=True)
    resolver = HybridSpatialReferenceResolver(gallery_source=gallery)
    references: dict[int, dict] = {}
    source_kinds: Counter[str] = Counter()
    for ordinal, physical_row in enumerate(union):
        resolved = resolver.resolve(physical_row)
        source_path = gallery.raw_paths[physical_row]
        source_kind = str(resolved.source_kind)
        source_kinds[source_kind] += 1
        references[physical_row] = {
            "physical_row": physical_row,
            "source_path": str(source_path),
            "source_image_sha256": e0.sha256_file(source_path),
            "grid_shape": tuple(map(int, resolved.grid_shape)),
            "tokens": resolved.tokens.detach().cpu().contiguous(),
            "tokens_sha256": str(resolved.tokens_sha256),
            "source_kind": source_kind,
            "source_logical_sha256": str(resolved.source_logical_sha256),
        }
        if (ordinal + 1) % 128 == 0 or ordinal + 1 == len(union):
            print(
                json.dumps(
                    {
                        "event": "matched_three_arm_reference_ready",
                        "done": ordinal + 1,
                        "total": len(union),
                        "source_kinds": dict(sorted(source_kinds.items())),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if source_kinds != EXPECTED_SOURCE_KINDS:
        raise RuntimeError("reference cache/recovery source-kind population drift")

    value = {
        "schema_version": "routea_d1_current_runtime_matched_three_arm_reference_cache_payload_v1_20260902",
        "status": "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_READY",
        "claim_level": "TARGET_FREE_REFERENCE_TOKEN_UNION_ONLY",
        "reference_union": union,
        "references": references,
        "source_kind_counts": dict(sorted(source_kinds.items())),
        "bindings": {
            "contract_sha256": e0.sha256_file(CONTRACT),
            "prejoin_validation_sha256": e0.sha256_file(PREJOIN_VALIDATION),
            "label_join_handoff_sha256": e0.sha256_file(JOIN_HANDOFF),
            "gallery_cache_sha256": gallery.gallery_cache_sha256,
            "gallery_corrected_mapping_sha256": gallery.corrected_mapping_sha256,
            "prejoin_shards": shard_bindings,
        },
        "access": {
            "label_join_handoff_read_count": 1,
            "target_identity_read_count": 0,
            "target_role_read_count": 0,
            "supergroup_read_count": 0,
            "sealed_read_count": 0,
            "encoder_model_load_count": 0,
            "encoder_model_forward_count": 0,
            "model_update_count": 0,
        },
    }
    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".d1-matched-three-arm-refcache-", dir=OUT_ROOT.parent)
    )
    stage_payload = staging / "payload.pt"
    stage_result = staging / "result.json"
    torch.save(value, stage_payload)
    result = {
        "schema_version": "routea_d1_current_runtime_matched_three_arm_reference_cache_result_v1_20260902",
        "status": "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_READY",
        "claim_level": value["claim_level"],
        "reference_count": len(references),
        "source_kind_counts": value["source_kind_counts"],
        "payload_sha256": e0.sha256_file(stage_payload),
        "bindings": value["bindings"],
        "access": value["access"],
        "next_authorized_stage": "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_SHARDS",
        "logical_sha256": "",
    }
    result["logical_sha256"] = e0.logical_sha256(result)
    try:
        stage_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.rename(staging, OUT_ROOT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(
        json.dumps(
            {
                "status": result["status"],
                "reference_count": result["reference_count"],
                "source_kind_counts": result["source_kind_counts"],
                "payload_sha256": result["payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
