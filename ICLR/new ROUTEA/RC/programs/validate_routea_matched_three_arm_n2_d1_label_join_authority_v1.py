#!/usr/bin/env python3
"""Independent validation of N2 D1 train-only label-join payloads."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
ROUTEA = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.n2_current_runtime_d1_training_v1 import (  # noqa: E402
    CORRECTED_ELIGIBLE_IDENTITY_COUNTS,
    EXPECTED_LABEL_SOURCE_HASHES,
    VERSION,
    label_join_dependency_hashes,
    atomic_json,
    canonical_sha256,
    corrected_mask_audit,
    load_gallery,
    logical_sha256,
    require,
    sanitized_query_index,
    sha256_file,
    tensor_sha256,
    validate_token_authority,
)


OUTCOME = ROUTEA / "results/b3_domain_alignment/train_cache_v1/manifest.json"
DIFFICULT = ROUTEA / "results/b4_learned_ownership/difficult_train_cache_v1/manifest.json"
NEW_DIFFICULT = ROUTEA / "results/route_a_c2_symmetric_ownership/new_difficult_token_cache_v2/manifest.json"
UPSTREAMS = ROOT / "registry/upstream_inputs.json"
CONTRACT = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_N2_CURRENT_RUNTIME_FOLD_LOCAL_D1_TRAINING_V1_20260903.md"
RUNTIME = ROOT / "src/rc_aslo_xf/n2_current_runtime_d1_training_v1.py"
PRODUCER = ROOT / "programs/materialize_routea_matched_three_arm_n2_d1_label_join_authority_v1.py"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_d1_label_join_authority_v1"
RESULT = OUT_ROOT / "result.json"
OUT = OUT_ROOT / "independent_validation.json"
IDENTITY_REPAIR_REGISTRY = ROOT / "registry/gallery_identity_repair_v1.json"
IDENTITY_REPAIR_CONTRACT = ROOT / "protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json"
EXPECTED_QUERY_COUNTS = {0: (775, 212), 1: (782, 205), 2: (798, 189), 3: (799, 188), 4: (794, 193)}


def read(path: Path) -> dict[str, Any]:
    require(path.exists() and path.is_file() and not path.is_symlink(), f"input absent: {path}")
    return json.loads(path.read_text())


def independent_manifest_map(allowed: set[str]) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    result: dict[str, tuple[str, str]] = {}
    rejected: list[str] = []

    def insert(key: str, identity: str, track: str) -> None:
        require(key not in result, f"duplicate three-manifest key: {key}")
        result[key] = (identity, track)

    outcome = read(OUTCOME)
    difficult = read(DIFFICULT)
    new = read(NEW_DIFFICULT)
    require(outcome.get("status") == "B3D_TRAIN_CACHE_COMPLETE", "outcome status")
    require(difficult.get("status") == "B4_DIFFICULT_QUERY_CACHE_COMPLETE", "difficult status")
    require(new.get("status") == "C2_NATURAL_QUERY_TOKEN_ONLY_CACHE_COMPLETE", "new status")
    for row in outcome["rows"]:
        query_id = f"OUTCOME-{int(row['index']):04d}"
        if query_id not in allowed:
            rejected.append(query_id)
            continue
        insert(query_id, str(row["identity"]), "outcome")
    for row in difficult["rows"]:
        query_id = f"DIFFICULT-{int(row['index']):04d}"
        if query_id not in allowed:
            rejected.append(query_id)
            continue
        insert(query_id, str(row["identity"]), "difficult")
    for row in new["rows"]:
        if row.get("split") == "train":
            query_id = str(row["query_id"])
            if query_id not in allowed:
                rejected.append(query_id)
                continue
            insert(query_id, str(row["reference_label"]), "new_difficult_train")
    return result, {
        "outcome_rows": len(outcome["rows"]),
        "difficult_rows": len(difficult["rows"]),
        "new_difficult_train_rows": sum(row.get("split") == "train" for row in new["rows"]),
        "three_manifest_train_row_total": len(outcome["rows"]) + len(difficult["rows"])
        + sum(row.get("split") == "train" for row in new["rows"]),
        "target_label_semantic_read_count": len(result),
        "rejected_before_label_read_count": len(rejected),
        "rejected_before_label_read_query_ids": sorted(rejected),
    }


def main() -> None:
    token_bindings = validate_token_authority(ROOT)
    source_axis = sanitized_query_index(ROOT)
    allowed_query_ids = {row["query_id"] for row in source_axis}
    manifest, source_counts = independent_manifest_map(allowed_query_ids)
    expected_rejected = [f"OUTCOME-{index:04d}" for index in range(519, 528)]
    expected_population_audit = {
        **source_counts,
        "optimization_join_count": 987,
        "explicitly_excluded_opened_count": 9,
        "explicitly_excluded_query_ids_sha256": canonical_sha256(expected_rejected),
        "missing_optimization_query_count": 0,
    }
    require(
        source_counts == {
            "outcome_rows": 822,
            "difficult_rows": 134,
            "new_difficult_train_rows": 40,
            "three_manifest_train_row_total": 996,
            "target_label_semantic_read_count": 987,
            "rejected_before_label_read_count": 9,
            "rejected_before_label_read_query_ids": expected_rejected,
        }
        and set(manifest) == allowed_query_ids,
        "independent three-manifest population/read boundary drift",
    )
    upstreams = read(UPSTREAMS)
    preflight_path = Path(upstreams["primary_line"]["identity_supergroup_preflight"]["path"])
    preflight = read(preflight_path)
    identity_rows = {
        str(row["identity"]): dict(row)
        for row in preflight["identities"]
        if row.get("final_role") == "optimization"
    }
    require(len(identity_rows) == 81, "independent optimization identity count")
    _, corrected, corrected_receipt = load_gallery(ROOT)
    positions: dict[str, list[int]] = defaultdict(list)
    for physical, identity in enumerate(corrected):
        positions[identity].append(physical)

    expected_join: dict[str, dict[str, Any]] = {}
    for source in source_axis:
        query_id = source["query_id"]
        require(query_id in manifest, f"independent manifest join missing {query_id}")
        identity, track = manifest[query_id]
        meta = identity_rows.get(identity)
        require(meta is not None and track == source["track"], f"independent identity/track join drift {query_id}")
        require(len(positions.get(identity, [])) == 1, f"positive row ambiguous for {identity}")
        expected_join[query_id] = {
            "query_id": query_id,
            "query_ordinal": int(source["query_ordinal"]),
            "heldout_fold": int(source["heldout_fold"]),
            "track": track,
            "target_identity": identity,
            "supergroup": str(meta["group_id"]),
            "target_physical_row": positions[identity][0],
        }
    require(len(expected_join) == 987, "independent 987 join failed")

    expected_bindings = {
        **label_join_dependency_hashes(ROOT),
        "contract_sha256": sha256_file(CONTRACT),
        "runtime_sha256": sha256_file(RUNTIME),
        "producer_sha256": sha256_file(PRODUCER),
        "outcome_manifest_sha256": sha256_file(OUTCOME),
        "difficult_manifest_sha256": sha256_file(DIFFICULT),
        "new_difficult_manifest_sha256": sha256_file(NEW_DIFFICULT),
        "upstreams_sha256": sha256_file(UPSTREAMS),
        "split_preflight_sha256": sha256_file(preflight_path),
        **token_bindings,
    }
    require(
        {role: expected_bindings[role] for role in EXPECTED_LABEL_SOURCE_HASHES}
        == EXPECTED_LABEL_SOURCE_HASHES,
        "independent frozen manifest/split source hash drift",
    )
    fold_seals = []
    checks: dict[str, bool] = {
        "three_manifest_unique_query_id_join_987": len(expected_join) == 987,
        "nine_opened_rows_rejected_before_label_read": source_counts["rejected_before_label_read_query_ids"]
        == expected_rejected and source_counts["target_label_semantic_read_count"] == 987,
        "positive_rows_unique_and_biogen_not_positive": all(
            row["target_identity"] != "Biogen_21" for row in expected_join.values()
        ),
        "token_axis_contains_no_source_paths": all("path" not in row for row in source_axis),
    }
    for fold in range(5):
        path = OUT_ROOT / f"fold_{fold}/payload.pt"
        require(path.exists() and path.is_file() and not path.is_symlink(), f"fold {fold} label payload absent")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records = payload.get("records", []) if isinstance(payload, Mapping) else []
        expected = sorted(
            (dict(row) for row in expected_join.values() if row["heldout_fold"] != fold),
            key=lambda row: row["query_ordinal"],
        )
        fold_spec = preflight["folds"][fold]
        train_ids = set(map(str, fold_spec["train_identities"]))
        heldout_ids = set(map(str, fold_spec["heldout_identities"]))
        train_groups = set(map(str, fold_spec["train_group_ids"]))
        heldout_groups = set(map(str, fold_spec["heldout_group_ids"]))
        excluded = set(map(int, preflight["gallery_masks"][fold]["excluded_row_indices"]))
        legal = torch.tensor([row not in excluded for row in range(5_413)], dtype=torch.bool)
        audit = corrected_mask_audit(corrected, legal)
        require(
            set(payload) == {
                "version", "status", "claim_level", "fold", "records",
                "legal_physical_row_mask", "legal_physical_row_mask_sha256",
                "mask_audit", "train_query_count", "heldout_query_count",
                "train_identity_count", "heldout_identity_count",
                "train_supergroup_count", "heldout_supergroup_count",
                "heldout_identity_sha256", "heldout_supergroup_sha256",
                "bindings", "access",
            }
            and payload.get("version") == VERSION
            and payload.get("status") == "ROUTEA_N2_D1_FOLD_TRAIN_LABEL_JOIN_AUTHORITY_READY"
            and payload.get("fold") == fold
            and records == expected
            and torch.equal(payload.get("legal_physical_row_mask"), legal)
            and payload.get("legal_physical_row_mask_sha256") == tensor_sha256(legal)
            and payload.get("mask_audit") == audit
            and audit["eligible_corrected_identities"] == CORRECTED_ELIGIBLE_IDENTITY_COUNTS[fold]
            and (len(records), 987 - len(records)) == EXPECTED_QUERY_COUNTS[fold]
            and {row["target_identity"] for row in records} == train_ids
            and {row["supergroup"] for row in records} == train_groups
            and all(row["target_identity"] not in heldout_ids for row in records)
            and all(row["supergroup"] not in heldout_groups for row in records)
            and all(bool(legal[row["target_physical_row"]]) for row in records)
            and payload.get("bindings") == expected_bindings
            and payload.get("access")
            == {
                "target_label_join_count": len(records),
                "heldout_target_label_published_count": 0,
                "query_path_published_count": 0,
                "model_update_count": 0,
                "external_read_count": 0,
                "sealed_read_count": 0,
            },
            f"fold {fold} train-only label payload drift",
        )
        fold_seals.append(
            {
                "fold": fold,
                "payload_sha256": sha256_file(path),
                "train_query_count": len(records),
                "record_manifest_sha256": canonical_sha256(
                    [[row["query_id"], row["query_ordinal"], row["target_identity"], row["supergroup"], row["target_physical_row"]] for row in records]
                ),
                "legal_mask_sha256": tensor_sha256(legal),
            }
        )
        checks[f"fold_{fold}_train_only_split_and_corrected_mask"] = True

    result = read(RESULT)
    require(
        result.get("status") == "ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_READY"
        and result.get("folds") == fold_seals
        and result.get("manifest_population_audit") == expected_population_audit
        and result.get("bindings") == expected_bindings
        and result.get("logical_sha256") == logical_sha256(result),
        "label-join producer result drift",
    )
    value = {
        "version": VERSION,
        "status": "ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_VALIDATED",
        "claim_level": "INDEPENDENT_TRAIN_ONLY_LABEL_JOIN_VALIDATION_NO_MODEL_UPDATE",
        "checks": checks,
        "query_count": 987,
        "corrected_gallery": corrected_receipt,
        "folds": fold_seals,
        "result_sha256": sha256_file(RESULT),
        "bindings": expected_bindings,
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": "N2_D1_RAW_PREJOIN_AND_FOLD_LOCAL_TRAINING",
        "logical_sha256": "",
    }
    require(bool(checks) and all(checks.values()), "independent label-join check failed")
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(OUT, value)
    print(json.dumps({"status": value["status"], "checks": checks}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
