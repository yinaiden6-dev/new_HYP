#!/usr/bin/env python3
"""Independent, target-free validation of one N2 D1 RAW score shard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
ROUTEA = ROOT.parent
sys.path[:0] = [str(ROOT / "src"), str(ROUTEA / "route_a_core")]

from rc_aslo_xf.n2_current_runtime_d1_training_v1 import (  # noqa: E402
    PREJOIN_SHARD_STATUS,
    PREJOIN_SHARD_VALID_STATUS,
    SHARD_COUNT,
    VERSION,
    raw_prejoin_dependency_hashes,
    atomic_json,
    canonical_sha256,
    expected_shard_interval,
    load_gallery,
    load_token_shard,
    logical_sha256,
    require,
    sha256_file,
    tensor_sha256,
    validate_prejoin_record,
    validate_token_authority,
)
from route_a.o1_c6direct_m1_runtime import chunked_sum_maxsim, pad_reference_tokens  # noqa: E402


CONTRACT = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_N2_CURRENT_RUNTIME_FOLD_LOCAL_D1_TRAINING_V1_20260903.md"
RUNTIME = ROOT / "src/rc_aslo_xf/n2_current_runtime_d1_training_v1.py"
PRODUCER = ROOT / "programs/materialize_routea_matched_three_arm_n2_d1_raw_prejoin_shard_v1.py"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_d1_raw_prejoin_v1"


def expected_bindings() -> dict[str, str]:
    values = validate_token_authority(ROOT)
    values.update(
        {
            **raw_prejoin_dependency_hashes(ROOT),
            "contract_sha256": sha256_file(CONTRACT),
            "runtime_sha256": sha256_file(RUNTIME),
            "producer_sha256": sha256_file(PRODUCER),
        }
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    shard = args.shard
    require(0 <= shard < SHARD_COUNT, "shard must be in 0..15")
    require(torch.cuda.is_available(), "RAW score replay requires CUDA")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS workspace drift")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    base = OUT_ROOT / f"shard{shard:02d}"
    payload_path = base / "payload.pt"
    receipt_path = base / "receipt.json"
    require(payload_path.is_file() and receipt_path.is_file(), "RAW prejoin shard absent")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
    receipt = json.loads(receipt_path.read_text())
    token_payload = load_token_shard(ROOT, shard)
    records = payload.get("records", []) if isinstance(payload, Mapping) else []
    token_records = token_payload["records"]
    expected = list(expected_shard_interval(shard))
    bindings = expected_bindings()
    require(
        set(payload) == {
            "version", "status", "claim_level", "shard", "shard_count",
            "query_ordinal_begin", "query_ordinal_end_exclusive", "records",
            "gallery", "bindings", "access",
        }
        and payload.get("version") == VERSION
        and payload.get("status") == PREJOIN_SHARD_STATUS
        and payload.get("shard") == shard
        and payload.get("shard_count") == SHARD_COUNT
        and [record.get("query_ordinal") for record in records] == expected
        and len(records) == len(token_records)
        and payload.get("bindings") == bindings,
        "RAW prejoin shard envelope or binding drift",
    )
    require(
        receipt.get("status") == PREJOIN_SHARD_STATUS
        and receipt.get("payload_sha256") == sha256_file(payload_path)
        and receipt.get("bindings") == bindings
        and receipt.get("logical_sha256") == logical_sha256(receipt),
        "RAW prejoin receipt drift",
    )
    for record, token in zip(records, token_records):
        validate_prejoin_record(record)
        require(
            record["query_id"] == token["query_id"]
            and record["query_ordinal"] == token["query_ordinal"]
            and record["heldout_fold"] == token["heldout_fold"]
            and record["track"] == token["track"]
            and record["image_tokens_sha256"] == token["image_tokens_sha256"]
            and record["template_tokens_sha256"] == token["template_tokens_sha256"],
            "RAW prejoin/token sanitized join drift",
        )

    # One preregistered replay per shard: the first canonical ordinal.  This
    # validator never opens a target or path-bearing query ledger.
    references, _, gallery_receipt = load_gallery(ROOT)
    device = torch.device("cuda")
    padded, mask = pad_reference_tokens([value.detach().to(torch.float16) for value in references])
    padded, mask = padded.to(device), mask.to(device)
    token = token_records[0]
    query = torch.cat(
        (
            torch.as_tensor(token["image_tokens"]).float().to(device),
            torch.as_tensor(token["template_tokens"]).float().to(device),
        ),
        dim=0,
    )
    with torch.inference_mode():
        replay = chunked_sum_maxsim(query, padded, mask, row_chunk_size=32).cpu().float()
    stored = torch.as_tensor(records[0]["physical_row_scores"])
    max_abs = float((replay - stored).abs().max())
    cosine = float(torch.nn.functional.cosine_similarity(replay.double(), stored.double(), dim=0))
    require(max_abs <= 1.0e-5 and cosine >= 0.999999999, "independent RAW score replay drift")

    checks = {
        "token_predecessor_complete_and_all_shards_sealed": True,
        "target_free_exact_record_schema": True,
        "canonical_shard_interval_and_token_join": True,
        "corrected_5412_gallery_binding": payload.get("gallery") == gallery_receipt,
        "one_query_full_5413_score_replay": max_abs <= 1.0e-5 and cosine >= 0.999999999,
        "zero_target_identity_group_model_access": payload.get("access")
        == {
            "query_target_identity_read_count": 0,
            "query_supergroup_read_count": 0,
            "corrected_gallery_identity_axis_read_count": 1,
            "encoder_load_count": 0,
            "encoder_update_count": 0,
            "d1_checkpoint_load_count": 0,
            "d1_update_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        },
    }
    require(bool(checks) and all(checks.values()), "RAW prejoin independent check failed")
    value = {
        "version": VERSION,
        "status": PREJOIN_SHARD_VALID_STATUS,
        "claim_level": "INDEPENDENT_TARGET_FREE_RAW_SCORE_VALIDATION_NO_TRAINING",
        "shard": shard,
        "checks": checks,
        "query_count": len(records),
        "query_ordinal_begin": expected[0],
        "query_ordinal_end_exclusive": expected[-1] + 1,
        "score_manifest_sha256": canonical_sha256(
            [[record["query_id"], record["query_ordinal"], record["physical_row_scores_sha256"]] for record in records]
        ),
        "payload_sha256": sha256_file(payload_path),
        "receipt_sha256": sha256_file(receipt_path),
        "bindings": bindings,
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "replay": {
            "query_id": records[0]["query_id"],
            "query_ordinal": records[0]["query_ordinal"],
            "maximum_absolute_error": max_abs,
            "cosine_similarity": cosine,
        },
        "next_authorized_stage": "N2_D1_RAW_PREJOIN_AGGREGATE_VALIDATION",
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(base / "validation.json", value)
    print(json.dumps({"status": value["status"], "shard": shard, "max_abs": max_abs}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
