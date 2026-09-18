#!/usr/bin/env python3
"""Materialize one target-free fresh-D1 full-gallery OOF score shard."""

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

from rc_aslo_xf.n2_current_runtime_d1_oof_scoring_v1 import (  # noqa: E402
    OOF_SHARD_STATUS,
    SHARD_COUNT,
    VERSION,
    base_dependency_bindings,
    expected_shard_interval,
    fold_paths,
    load_token_shard,
    logical_sha256,
    ranked_representative_rows,
    require,
    sha256_file,
    tensor_sha256,
    validate_five_fold_authority,
    validate_oof_record,
)
from rc_aslo_xf.n2_current_runtime_d1_training_v1 import (  # noqa: E402
    atomic_json,
    atomic_torch,
    load_gallery,
    parameter_manifest,
)
from route_a import o1_c6direct_m1_d1 as d1  # noqa: E402
from route_a.o1_c6direct_m1_runtime import chunked_sum_maxsim, pad_reference_tokens  # noqa: E402


OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1"
WORK_ROOT = ROOT / "tmp/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1"
ROW_CHUNK = 32


def bindings() -> dict[str, Any]:
    value = base_dependency_bindings(ROOT)
    value["producer_sha256"] = sha256_file(Path(__file__).resolve())
    return value


def load_progress(
    path: Path,
    ordinal: int,
    token: Mapping[str, Any],
    expected_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    require(
        isinstance(value, Mapping)
        and value.get("version") == VERSION
        and value.get("status") == OOF_SHARD_STATUS
        and value.get("query_ordinal") == ordinal
        and value.get("query_id") == token["query_id"]
        and value.get("bindings") == dict(expected_bindings),
        f"fresh-D1 OOF progress drift at ordinal {ordinal}",
    )
    record = value.get("record")
    require(isinstance(record, Mapping), "fresh-D1 OOF progress record absent")
    validate_oof_record(record)
    return dict(record)


def run(shard: int) -> dict[str, Any]:
    require(0 <= shard < SHARD_COUNT, "shard must be in 0..15")
    require(torch.cuda.is_available(), "formal fresh-D1 OOF scoring requires CUDA")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS_WORKSPACE_CONFIG drift")
    d1.configure_deterministic_runtime()
    torch.set_float32_matmul_precision("highest")

    expected_bindings = bindings()
    _fold_seals, fold_states = validate_five_fold_authority(ROOT)
    token_payload = load_token_shard(ROOT, shard)
    token_records = token_payload["records"]
    expected_ordinals = list(expected_shard_interval(shard))
    require(
        [int(record["query_ordinal"]) for record in token_records] == expected_ordinals,
        "token/OOF shard interval mismatch",
    )

    final_dir = OUT_ROOT / f"shard{shard:02d}"
    payload_path = final_dir / "payload.pt"
    receipt_path = final_dir / "receipt.json"
    if payload_path.exists() or receipt_path.exists():
        require(payload_path.exists() and receipt_path.exists(), "partial committed fresh-D1 OOF shard")
        receipt = json.loads(receipt_path.read_text())
        require(
            receipt.get("status") == OOF_SHARD_STATUS
            and receipt.get("payload_sha256") == sha256_file(payload_path)
            and receipt.get("bindings") == expected_bindings
            and receipt.get("logical_sha256") == logical_sha256(receipt),
            "existing fresh-D1 OOF shard is stale or corrupt",
        )
        return receipt

    references, corrected_labels, gallery_receipt = load_gallery(ROOT)
    device = torch.device("cuda")
    padded, reference_mask = pad_reference_tokens(
        [value.detach().to(torch.float16) for value in references]
    )
    padded = padded.to(device)
    reference_mask = reference_mask.to(device)
    del references

    adapters: dict[int, torch.nn.Module] = {}
    checkpoint_hashes: dict[int, str] = {}
    for fold, state in fold_states.items():
        adapter = d1.build_fresh_d1_adapter().to(device).eval()
        adapter.load_state_dict(state, strict=True)
        parameter_manifest(adapter)
        adapter.requires_grad_(False)
        adapters[fold] = adapter
        checkpoint_hashes[fold] = sha256_file(fold_paths(ROOT, fold)["checkpoint"])

    work_dir = WORK_ROOT / f"shard{shard:02d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for local_index, (ordinal, token) in enumerate(zip(expected_ordinals, token_records, strict=True)):
        require(int(token["query_ordinal"]) == ordinal, "sanitized token order drift")
        progress_path = work_dir / f"query_{ordinal:04d}.pt"
        if progress_path.exists():
            record = load_progress(progress_path, ordinal, token, expected_bindings)
        else:
            fold = int(token["heldout_fold"])
            require(fold in adapters, f"OOF checkpoint absent for fold {fold}")
            image = torch.as_tensor(token["image_tokens"]).float().to(device)
            template = torch.as_tensor(token["template_tokens"]).float().to(device)
            grid = tuple(map(int, token["grid_shape"]))
            with torch.inference_mode():
                adapted = adapters[fold](image, grid[0], grid[1]).detach().cpu().float().contiguous()
                scores = chunked_sum_maxsim(
                    torch.cat((adapted.to(device), template), dim=0),
                    padded,
                    reference_mask,
                    row_chunk_size=ROW_CHUNK,
                ).detach().cpu().float().contiguous()
            ranked_list = ranked_representative_rows(scores, corrected_labels)
            ranked = torch.tensor(ranked_list, dtype=torch.int32)
            boundary_gap = float(scores[ranked_list[127]]) - float(scores[ranked_list[128]])
            require(boundary_gap >= 0.0, "fresh-D1 C128 boundary order drift")
            record = {
                "query_id": str(token["query_id"]),
                "query_ordinal": ordinal,
                "heldout_fold": fold,
                "track": str(token["track"]),
                "grid_shape": grid,
                "image_tokens_sha256": str(token["image_tokens_sha256"]),
                "template_tokens_sha256": str(token["template_tokens_sha256"]),
                "adapted_image_tokens": adapted,
                "adapted_image_tokens_sha256": tensor_sha256(adapted),
                "physical_row_scores": scores,
                "physical_row_scores_sha256": tensor_sha256(scores),
                "ranked_representative_physical_rows": ranked,
                "ranked_representative_physical_rows_sha256": tensor_sha256(ranked),
                "natural_c128_representative_physical_rows": ranked[:128].clone(),
                "c128_boundary_score_gap": boundary_gap,
                "c128_boundary_exact_tie": boundary_gap == 0.0,
                "oof_checkpoint_sha256": checkpoint_hashes[fold],
            }
            validate_oof_record(record)
            atomic_torch(
                progress_path,
                {
                    "version": VERSION,
                    "status": OOF_SHARD_STATUS,
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
                    "event": "n2_fresh_d1_oof_prejoin_progress",
                    "shard": shard,
                    "done": local_index + 1,
                    "total": len(token_records),
                    "query_ordinal": ordinal,
                    "fold": int(token["heldout_fold"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    require([record["query_ordinal"] for record in records] == expected_ordinals, "OOF output order drift")
    payload = {
        "version": VERSION,
        "status": OOF_SHARD_STATUS,
        "claim_level": "TARGET_FREE_FRESH_D1_FULL_GALLERY_OOF_PREJOIN_NO_SCIENTIFIC_CLAIM",
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
            "d1_checkpoint_load_count": 5,
            "d1_model_update_count": 0,
            "target_insertion_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        },
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".shard{shard:02d}.", suffix=".partial", dir=OUT_ROOT)
    )
    try:
        torch.save(payload, temporary_dir / "payload.pt")
        receipt = {
            "version": VERSION,
            "status": OOF_SHARD_STATUS,
            "claim_level": payload["claim_level"],
            "shard": shard,
            "query_count": len(records),
            "query_ordinal_begin": expected_ordinals[0],
            "query_ordinal_end_exclusive": expected_ordinals[-1] + 1,
            "fold_counts": {
                str(fold): sum(record["heldout_fold"] == fold for record in records)
                for fold in range(5)
            },
            "score_ranking_manifest_sha256": canonical_manifest(records),
            "payload_sha256": sha256_file(temporary_dir / "payload.pt"),
            "bindings": expected_bindings,
            "model_update_count": 0,
            "target_insertion_count": 0,
            "next_authorized_stage": "N2_FRESH_D1_OOF_PREJOIN_SHARD_INDEPENDENT_VALIDATION",
            "logical_sha256": "",
        }
        receipt["logical_sha256"] = logical_sha256(receipt)
        atomic_json(temporary_dir / "receipt.json", receipt)
        os.rename(temporary_dir, final_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return receipt


def canonical_manifest(records: list[dict[str, Any]]) -> str:
    from rc_aslo_xf.n2_current_runtime_d1_oof_scoring_v1 import canonical_sha256

    return canonical_sha256(
        [
            [
                record["query_id"],
                record["query_ordinal"],
                record["heldout_fold"],
                record["adapted_image_tokens_sha256"],
                record["physical_row_scores_sha256"],
                record["ranked_representative_physical_rows_sha256"],
                record["oof_checkpoint_sha256"],
            ]
            for record in records
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    value = run(args.shard)
    print(json.dumps({"status": value["status"], "shard": args.shard}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
