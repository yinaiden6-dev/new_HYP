#!/usr/bin/env python3
"""Independent validation of one target-free fresh-D1 OOF score shard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
ROUTEA = ROOT.parent
sys.path[:0] = [str(ROOT / "src"), str(ROUTEA / "route_a_core")]

from rc_aslo_xf.n2_current_runtime_d1_oof_scoring_v1 import (  # noqa: E402
    OOF_SHARD_STATUS,
    OOF_SHARD_VALID_STATUS,
    SHARD_COUNT,
    VERSION,
    base_dependency_bindings,
    canonical_sha256,
    expected_shard_interval,
    fold_paths,
    load_token_shard,
    logical_sha256,
    require,
    sha256_file,
    tensor_sha256,
    validate_five_fold_authority,
    validate_oof_record,
)
from rc_aslo_xf.n2_current_runtime_d1_training_v1 import atomic_json, load_gallery, parameter_manifest  # noqa: E402
from rc_aslo_xf.n2_corrected_d1_runtime_v1 import reduce_corrected_full_gallery_scores  # noqa: E402
from route_a import o1_c6direct_m1_d1 as d1  # noqa: E402
from route_a.o1_c6direct_m1_runtime import pad_reference_tokens  # noqa: E402


PRODUCER = ROOT / "programs/materialize_routea_matched_three_arm_n2_d1_oof_prejoin_shard_v1.py"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1"
ROW_CHUNK = 32


def expected_bindings() -> dict[str, Any]:
    value = base_dependency_bindings(ROOT)
    value["producer_sha256"] = sha256_file(PRODUCER)
    return value


def direct_sum_maxsim(
    query: torch.Tensor,
    references: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    require(
        query.ndim == 2
        and query.shape[1] == 128
        and references.ndim == 3
        and references.shape[0] == 5413
        and references.shape[2] == 128
        and mask.shape == references.shape[:2],
        "independent full-gallery replay shape drift",
    )
    chunks = []
    for start in range(0, references.shape[0], ROW_CHUNK):
        stop = min(start + ROW_CHUNK, references.shape[0])
        similarity = torch.einsum(
            "qd,rjd->qrj", query.float(), references[start:stop].float()
        )
        chunks.append(
            similarity.masked_fill(~mask[start:stop][None], float("-inf"))
            .amax(dim=-1)
            .sum(dim=0)
        )
    scores = torch.cat(chunks).cpu().float().contiguous()
    require(scores.shape == (5413,) and bool(torch.isfinite(scores).all()), "independent scores invalid")
    return scores


def score_ranking_manifest(records: list[Mapping[str, Any]]) -> str:
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
    shard = args.shard
    require(0 <= shard < SHARD_COUNT, "shard must be in 0..15")
    require(torch.cuda.is_available(), "fresh-D1 OOF score replay requires CUDA")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS workspace drift")
    d1.configure_deterministic_runtime()
    torch.set_float32_matmul_precision("highest")

    base = OUT_ROOT / f"shard{shard:02d}"
    payload_path = base / "payload.pt"
    receipt_path = base / "receipt.json"
    require(payload_path.is_file() and receipt_path.is_file(), "fresh-D1 OOF shard absent")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
    receipt = json.loads(receipt_path.read_text())
    records = payload.get("records", []) if isinstance(payload, Mapping) else []
    token_payload = load_token_shard(ROOT, shard)
    token_records = token_payload["records"]
    expected = list(expected_shard_interval(shard))
    bindings = expected_bindings()
    _fold_seals, states = validate_five_fold_authority(ROOT)
    require(
        set(payload) == {
            "version", "status", "claim_level", "shard", "shard_count",
            "query_ordinal_begin", "query_ordinal_end_exclusive", "records",
            "gallery", "bindings", "access",
        }
        and payload.get("version") == VERSION
        and payload.get("status") == OOF_SHARD_STATUS
        and payload.get("shard") == shard
        and payload.get("shard_count") == SHARD_COUNT
        and [record.get("query_ordinal") for record in records] == expected
        and len(records) == len(token_records)
        and payload.get("bindings") == bindings,
        "fresh-D1 OOF shard envelope or binding drift",
    )
    require(
        receipt.get("status") == OOF_SHARD_STATUS
        and receipt.get("payload_sha256") == sha256_file(payload_path)
        and receipt.get("bindings") == bindings
        and receipt.get("score_ranking_manifest_sha256") == score_ranking_manifest(records)
        and receipt.get("logical_sha256") == logical_sha256(receipt),
        "fresh-D1 OOF receipt drift",
    )

    _references, corrected, gallery_receipt = load_gallery(ROOT)
    del _references
    checkpoint_hashes = {
        fold: sha256_file(fold_paths(ROOT, fold)["checkpoint"]) for fold in range(5)
    }
    record_checks: list[bool] = []
    for record, token in zip(records, token_records, strict=True):
        validate_oof_record(record)
        scores = torch.as_tensor(record["physical_row_scores"])
        # Independently realize max-per-corrected-identity reduction for every
        # record; do not reuse the producer's stable physical-row scan.
        reduction = reduce_corrected_full_gallery_scores(scores, corrected)
        ranked = list(reduction.ranked_representative_rows)
        boundary_gap = float(scores[ranked[127]]) - float(scores[ranked[128]])
        record_checks.append(
            record["query_id"] == token["query_id"]
            and record["query_ordinal"] == token["query_ordinal"]
            and record["heldout_fold"] == token["heldout_fold"]
            and record["track"] == token["track"]
            and tuple(record["grid_shape"]) == tuple(token["grid_shape"])
            and record["image_tokens_sha256"] == token["image_tokens_sha256"]
            and record["template_tokens_sha256"] == token["template_tokens_sha256"]
            and record["adapted_image_tokens"].shape == token["image_tokens"].shape
            and record["ranked_representative_physical_rows"].tolist() == ranked
            and record["natural_c128_representative_physical_rows"].tolist() == ranked[:128]
            and record["c128_boundary_score_gap"] == boundary_gap
            and record["c128_boundary_exact_tie"] is (boundary_gap == 0.0)
            and record["oof_checkpoint_sha256"] == checkpoint_hashes[int(token["heldout_fold"])]
        )
    require(all(record_checks), "fresh-D1 OOF record/token/ranking replay failed")

    # Fresh direct replay of the first canonical query in the shard.
    references, _, _ = load_gallery(ROOT)
    device = torch.device("cuda")
    padded, mask = pad_reference_tokens([value.detach().to(torch.float16) for value in references])
    padded, mask = padded.to(device), mask.to(device)
    token = token_records[0]
    record = records[0]
    fold = int(token["heldout_fold"])
    adapter = d1.build_fresh_d1_adapter().to(device).eval()
    adapter.load_state_dict(states[fold], strict=True)
    parameter_manifest(adapter)
    adapter.requires_grad_(False)
    with torch.inference_mode():
        image = torch.as_tensor(token["image_tokens"]).float().to(device)
        grid = tuple(map(int, token["grid_shape"]))
        adapted = adapter(image, grid[0], grid[1]).cpu().float().contiguous()
        query = torch.cat(
            (adapted.to(device), torch.as_tensor(token["template_tokens"]).float().to(device)),
            dim=0,
        )
        replay_scores = direct_sum_maxsim(query, padded, mask)
    stored_adapted = torch.as_tensor(record["adapted_image_tokens"])
    stored_scores = torch.as_tensor(record["physical_row_scores"])
    adapted_max_abs = float((adapted - stored_adapted).abs().max())
    score_max_abs = float((replay_scores - stored_scores).abs().max())
    score_cosine = float(
        torch.nn.functional.cosine_similarity(replay_scores.double(), stored_scores.double(), dim=0)
    )
    independent_reduction = reduce_corrected_full_gallery_scores(replay_scores, corrected)
    replay_ranked = list(independent_reduction.ranked_representative_rows)
    require(
        adapted_max_abs <= 1.0e-6
        and score_max_abs <= 1.0e-5
        and score_cosine >= 0.999999999
        and replay_ranked == record["ranked_representative_physical_rows"].tolist(),
        "fresh-D1 direct full-gallery replay drift",
    )

    checks = {
        "five_fold_final_step_authority_complete": True,
        "token_and_raw_prejoin_predecessors_valid": True,
        "target_free_exact_record_schema": True,
        "canonical_shard_interval_and_token_join": True,
        "query_routes_to_own_heldout_fold_checkpoint": True,
        "corrected_5413_row_5412_identity_ranking_and_c128": True,
        "one_query_direct_adapter_and_full_gallery_replay": True,
        "zero_target_group_update_insertion_external_sealed_access": payload.get("access")
        == {
            "query_target_identity_read_count": 0,
            "query_supergroup_read_count": 0,
            "corrected_gallery_identity_axis_read_count": 1,
            "d1_checkpoint_load_count": 5,
            "d1_model_update_count": 0,
            "target_insertion_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        },
        "gallery_receipt_exact": payload.get("gallery") == gallery_receipt,
    }
    require(all(checks.values()), "fresh-D1 OOF independent checks failed")
    value = {
        "version": VERSION,
        "status": OOF_SHARD_VALID_STATUS,
        "claim_level": "INDEPENDENT_TARGET_FREE_FRESH_D1_OOF_SCORE_VALIDATION_NO_SCIENTIFIC_CLAIM",
        "shard": shard,
        "checks": checks,
        "query_count": len(records),
        "query_ordinal_begin": expected[0],
        "query_ordinal_end_exclusive": expected[-1] + 1,
        "score_ranking_manifest_sha256": score_ranking_manifest(records),
        "payload_sha256": sha256_file(payload_path),
        "receipt_sha256": sha256_file(receipt_path),
        "bindings": bindings,
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "replay": {
            "query_id": record["query_id"],
            "query_ordinal": record["query_ordinal"],
            "heldout_fold": fold,
            "adapted_token_maximum_absolute_error": adapted_max_abs,
            "physical_score_maximum_absolute_error": score_max_abs,
            "physical_score_cosine_similarity": score_cosine,
        },
        "model_update_count": 0,
        "target_insertion_count": 0,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": "N2_FRESH_D1_OOF_PREJOIN_AGGREGATE_VALIDATION",
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(base / "validation.json", value)
    print(json.dumps({"status": value["status"], "shard": shard, "score_max_abs": score_max_abs}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
