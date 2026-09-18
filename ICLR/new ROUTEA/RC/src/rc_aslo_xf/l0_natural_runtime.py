"""Strict V2 receipt validation for the L0 natural-pair funnel.

The receipt schemas are exact whitelists.  This is intentional: a rehashed
artifact carrying an alias such as ``target_label`` or ``d1_scores`` must fail
before it can reach any target-free scorer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from rc_aslo_xf.gallery_identity_repair import repair_contract_sha256


CANDIDATE_COUNT = 128
PREJOIN_VERSION = "l0_natural_hardneg_v2_identityfix_fold_local_fullgallery_c128_prejoin_v1"
SCORE_VERSION = "l0_natural_hardneg_v2_identityfix_target_free_score_fold_v1"

PREJOIN_KEYS = {
    "version",
    "contract_sha256",
    "fold",
    "query_ordinal",
    "query_id",
    "source_query_ledger_logical_sha256",
    "d1_checkpoint_sha256",
    "candidate_mask",
    "full_gallery_row_count",
    "full_gallery_label_count",
    "gallery_identity_repair_contract_sha256",
    "full_gallery_row_label_mapping_sha256_receipt_only",
    "candidate_physical_rows_in_hash_permutation",
    "candidate_row_label_mapping_sha256_receipt_only",
    "full_physical_score_sha256_receipt_only",
    "candidate_context_logical_sha256_receipt_only",
    "target_free_flags",
    "logical_sha256",
}
PREJOIN_LOGICAL_KEYS = tuple(sorted(PREJOIN_KEYS - {"logical_sha256"}))

SCORE_KEYS = {
    "version",
    "fold",
    "query_ordinal",
    "query_id",
    "contract_sha256",
    "head_checkpoint_sha256",
    "c0_prejoin_logical_sha256",
    "candidate_physical_rows_in_hash_permutation",
    "candidate_row_label_mapping_sha256_receipt_only",
    "real_candidate_evidence",
    "C_binding_candidate_evidence",
    "C_binding_offset",
    "target_join_performed",
    "target_label_read",
    "D1_score_rank_slot_winner_read",
    "opened_runtime_read_count",
    "sealed_runtime_read_count",
    "a10_runtime_read_count",
    "home_files_modified",
    "logical_sha256",
}
SCORE_LOGICAL_KEYS = tuple(sorted(SCORE_KEYS - {"logical_sha256"}))

TARGET_FREE_FLAGS = {
    "target_join_performed": False,
    "target_label_read": False,
    "historical_identity_loader_calls": 0,
    "D1_score_rank_slot_winner_serialized": False,
    "training_legal_mask_reads": 0,
    "opened_runtime_read_count": 0,
    "sealed_runtime_read_count": 0,
    "a10_runtime_read_count": 0,
    "home_files_modified": 0,
}


class L0RuntimeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    if set(keys) - set(payload):
        raise L0RuntimeError("L0 receipt omits a logical-hash field")
    return hashlib.sha256(
        json.dumps({key: payload[key] for key in keys}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prejoin_logical_sha(payload: dict[str, Any]) -> str:
    return _canonical_sha(payload, PREJOIN_LOGICAL_KEYS)


def load_prejoin(
    path: Path,
    *,
    fold: int,
    ordinal: int,
    contract_sha256: str,
    d1_checkpoint_sha256: str,
    source_query_ledger_logical_sha256: str,
) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise L0RuntimeError("L0 target-free C0 prejoin is not a dictionary")
    rows = value.get("candidate_physical_rows_in_hash_permutation", [])
    sha_fields = (
        "source_query_ledger_logical_sha256",
        "d1_checkpoint_sha256",
        "full_gallery_row_label_mapping_sha256_receipt_only",
        "candidate_row_label_mapping_sha256_receipt_only",
        "full_physical_score_sha256_receipt_only",
        "candidate_context_logical_sha256_receipt_only",
        "logical_sha256",
    )
    if (
        set(value) != PREJOIN_KEYS
        or value.get("version") != PREJOIN_VERSION
        or value.get("contract_sha256") != contract_sha256
        or value.get("fold") != fold
        or value.get("query_ordinal") != ordinal
        or not isinstance(value.get("query_id"), str)
        or value.get("source_query_ledger_logical_sha256") != source_query_ledger_logical_sha256
        or value.get("d1_checkpoint_sha256") != d1_checkpoint_sha256
        or value.get("candidate_mask") != "all_true_5413_rows"
        or value.get("full_gallery_row_count") != 5413
        or value.get("full_gallery_label_count") != 5412
        or value.get("gallery_identity_repair_contract_sha256") != repair_contract_sha256()
        or len(rows) != CANDIDATE_COUNT
        or len(set(rows)) != CANDIDATE_COUNT
        or not all(isinstance(row, int) and 0 <= row < 5413 for row in rows)
        or value.get("target_free_flags") != TARGET_FREE_FLAGS
        or any(not isinstance(value.get(name), str) or len(value[name]) != 64 for name in sha_fields)
        or value.get("logical_sha256") != prejoin_logical_sha(value)
    ):
        raise L0RuntimeError("L0 target-free C0 prejoin receipt drift")
    return value


def score_logical_sha(payload: dict[str, Any]) -> str:
    return _canonical_sha(payload, SCORE_LOGICAL_KEYS)


def load_target_free_score(
    path: Path,
    *,
    fold: int,
    ordinal: int,
    contract_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise L0RuntimeError("L0 target-free score is not a dictionary")
    vectors = ("real_candidate_evidence", "C_binding_candidate_evidence")
    if (
        set(value) != SCORE_KEYS
        or value.get("version") != SCORE_VERSION
        or value.get("fold") != fold
        or value.get("query_ordinal") != ordinal
        or not isinstance(value.get("query_id"), str)
        or value.get("contract_sha256") != contract_sha256
        or value.get("head_checkpoint_sha256") != checkpoint_sha256
        or not isinstance(value.get("c0_prejoin_logical_sha256"), str)
        or len(value["c0_prejoin_logical_sha256"]) != 64
        or len(value.get("candidate_physical_rows_in_hash_permutation", [])) != CANDIDATE_COUNT
        or len(set(value["candidate_physical_rows_in_hash_permutation"])) != CANDIDATE_COUNT
        or not isinstance(value.get("candidate_row_label_mapping_sha256_receipt_only"), str)
        or len(value["candidate_row_label_mapping_sha256_receipt_only"]) != 64
        or not isinstance(value.get("C_binding_offset"), int)
        or value.get("target_join_performed") is not False
        or value.get("target_label_read") is not False
        or value.get("D1_score_rank_slot_winner_read") is not False
        or any(value.get(name) != 0 for name in ("opened_runtime_read_count", "sealed_runtime_read_count", "a10_runtime_read_count", "home_files_modified"))
        or any(len(value.get(name, [])) != CANDIDATE_COUNT for name in vectors)
        or not all(torch.isfinite(torch.tensor(value[name], dtype=torch.float64)).all() for name in vectors)
        or value.get("logical_sha256") != score_logical_sha(value)
    ):
        raise L0RuntimeError("L0 target-free score receipt drift")
    return value
