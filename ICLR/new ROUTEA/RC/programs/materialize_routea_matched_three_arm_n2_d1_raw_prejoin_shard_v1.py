#!/usr/bin/env python3
"""Materialize one target-free current-runtime RAW full-gallery score shard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
ROUTEA = ROOT.parent
sys.path[:0] = [str(ROOT / "src"), str(ROUTEA / "route_a_core")]

from rc_aslo_xf.n2_current_runtime_d1_training_v1 import (  # noqa: E402
    PREJOIN_SHARD_STATUS,
    SHARD_COUNT,
    VERSION,
    raw_prejoin_dependency_hashes,
    atomic_json,
    atomic_torch,
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
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_d1_raw_prejoin_v1"
WORK_ROOT = ROOT / "tmp/routea_matched_three_arm_n2_d1_raw_prejoin_v1"
ROW_CHUNK = 32


def bindings() -> dict[str, str]:
    values = validate_token_authority(ROOT)
    values.update(
        {
            **raw_prejoin_dependency_hashes(ROOT),
            "contract_sha256": sha256_file(CONTRACT),
            "runtime_sha256": sha256_file(RUNTIME),
            "producer_sha256": sha256_file(Path(__file__).resolve()),
        }
    )
    return values


def load_progress(path: Path, ordinal: int, token: Mapping[str, Any], expected_bindings: Mapping[str, str]) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    require(
        isinstance(value, Mapping)
        and value.get("version") == VERSION
        and value.get("status") == PREJOIN_SHARD_STATUS
        and value.get("query_ordinal") == ordinal
        and value.get("query_id") == token["query_id"]
        and value.get("bindings") == dict(expected_bindings),
        f"RAW prejoin progress drift at ordinal {ordinal}",
    )
    record = value.get("record")
    require(isinstance(record, Mapping), "RAW prejoin progress record absent")
    validate_prejoin_record(record)
    return dict(record)


def run(shard: int) -> dict[str, Any]:
    require(torch.cuda.is_available(), "formal RAW prejoin requires CUDA")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS_WORKSPACE_CONFIG drift")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    expected_bindings = bindings()
    token_payload = load_token_shard(ROOT, shard)
    token_records = token_payload["records"]
    expected_ordinals = list(expected_shard_interval(shard))
    require(
        [int(record["query_ordinal"]) for record in token_records] == expected_ordinals,
        "token/prejoin shard interval mismatch",
    )

    final_dir = OUT_ROOT / f"shard{shard:02d}"
    payload_path = final_dir / "payload.pt"
    receipt_path = final_dir / "receipt.json"
    if payload_path.exists() or receipt_path.exists():
        require(payload_path.exists() and receipt_path.exists(), "partial committed RAW prejoin shard")
        receipt = json.loads(receipt_path.read_text())
        require(
            receipt.get("status") == PREJOIN_SHARD_STATUS
            and receipt.get("payload_sha256") == sha256_file(payload_path)
            and receipt.get("bindings") == expected_bindings
            and receipt.get("logical_sha256") == logical_sha256(receipt),
            "existing RAW prejoin shard is stale or corrupt",
        )
        return receipt

    references, corrected_labels, gallery_receipt = load_gallery(ROOT)
    del corrected_labels
    device = torch.device("cuda")
    padded, mask = pad_reference_tokens([value.detach().to(torch.float16) for value in references])
    padded = padded.to(device)
    mask = mask.to(device)
    del references

    work_dir = WORK_ROOT / f"shard{shard:02d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for local_index, (ordinal, token) in enumerate(zip(expected_ordinals, token_records)):
        require(int(token["query_ordinal"]) == ordinal, f"sanitized token order drift at ordinal {ordinal}")
        progress_path = work_dir / f"query_{ordinal:04d}.pt"
        if progress_path.exists():
            record = load_progress(progress_path, ordinal, token, expected_bindings)
        else:
            image = torch.as_tensor(token["image_tokens"]).float().to(device)
            template = torch.as_tensor(token["template_tokens"]).float().to(device)
            query = torch.cat((image, template), dim=0)
            with torch.inference_mode():
                scores = chunked_sum_maxsim(
                    query, padded, mask, row_chunk_size=ROW_CHUNK
                ).detach().cpu().float().contiguous()
            record = {
                "query_id": str(token["query_id"]),
                "query_ordinal": ordinal,
                "heldout_fold": int(token["heldout_fold"]),
                "track": str(token["track"]),
                "physical_row_scores": scores,
                "physical_row_scores_sha256": tensor_sha256(scores),
                "image_tokens_sha256": str(token["image_tokens_sha256"]),
                "template_tokens_sha256": str(token["template_tokens_sha256"]),
            }
            validate_prejoin_record(record)
            atomic_torch(
                progress_path,
                {
                    "version": VERSION,
                    "status": PREJOIN_SHARD_STATUS,
                    "query_id": record["query_id"],
                    "query_ordinal": ordinal,
                    "bindings": expected_bindings,
                    "record": record,
                },
            )
        records.append(record)
        print(
            json.dumps(
                {
                    "event": "n2_d1_raw_prejoin_progress",
                    "shard": shard,
                    "done": local_index + 1,
                    "total": len(token_records),
                    "query_ordinal": ordinal,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    require([record["query_ordinal"] for record in records] == expected_ordinals, "RAW prejoin order drift")
    payload = {
        "version": VERSION,
        "status": PREJOIN_SHARD_STATUS,
        "claim_level": "TARGET_FREE_RAW_FULL_GALLERY_SCORE_CACHE_NO_TRAINING",
        "shard": shard,
        "shard_count": SHARD_COUNT,
        "query_ordinal_begin": expected_ordinals[0],
        "query_ordinal_end_exclusive": expected_ordinals[-1] + 1,
        "records": records,
        "gallery": gallery_receipt,
        "bindings": expected_bindings,
        "access": {
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

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".shard{shard:02d}.", suffix=".partial", dir=OUT_ROOT))
    try:
        torch.save(payload, temporary_dir / "payload.pt")
        receipt = {
            "version": VERSION,
            "status": PREJOIN_SHARD_STATUS,
            "claim_level": payload["claim_level"],
            "shard": shard,
            "query_count": len(records),
            "query_ordinal_begin": expected_ordinals[0],
            "query_ordinal_end_exclusive": expected_ordinals[-1] + 1,
            "score_manifest_sha256": canonical_sha256(
                [
                    [record["query_id"], record["query_ordinal"], record["physical_row_scores_sha256"]]
                    for record in records
                ]
            ),
            "payload_sha256": sha256_file(temporary_dir / "payload.pt"),
            "bindings": expected_bindings,
            "model_update_count": 0,
            "next_authorized_stage": "N2_D1_RAW_PREJOIN_SHARD_INDEPENDENT_VALIDATION",
            "logical_sha256": "",
        }
        receipt["logical_sha256"] = logical_sha256(receipt)
        atomic_json(temporary_dir / "receipt.json", receipt)
        os.rename(temporary_dir, final_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    require(0 <= args.shard < SHARD_COUNT, "shard must be in 0..15")
    value = run(args.shard)
    print(json.dumps({"status": value["status"], "shard": args.shard}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
