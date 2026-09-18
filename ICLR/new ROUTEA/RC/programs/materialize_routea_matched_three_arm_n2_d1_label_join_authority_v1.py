#!/usr/bin/env python3
"""Create train-only fold label payloads from the three frozen manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
ROUTEA = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.n2_current_runtime_d1_training_v1 import (  # noqa: E402
    CORRECTED_ELIGIBLE_IDENTITY_COUNTS,
    EXPECTED_LABEL_SOURCE_HASHES,
    PARAMETER_COUNT,
    VERSION,
    label_join_dependency_hashes,
    atomic_json,
    atomic_torch,
    canonical_sha256,
    corrected_mask_audit,
    load_gallery,
    logical_sha256,
    require,
    sha256_file,
    sanitized_query_index,
    tensor_sha256,
    validate_token_authority,
)


OUTCOME = ROUTEA / "results/b3_domain_alignment/train_cache_v1/manifest.json"
DIFFICULT = ROUTEA / "results/b4_learned_ownership/difficult_train_cache_v1/manifest.json"
NEW_DIFFICULT = ROUTEA / "results/route_a_c2_symmetric_ownership/new_difficult_token_cache_v2/manifest.json"
UPSTREAMS = ROOT / "registry/upstream_inputs.json"
CONTRACT = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_N2_CURRENT_RUNTIME_FOLD_LOCAL_D1_TRAINING_V1_20260903.md"
RUNTIME = ROOT / "src/rc_aslo_xf/n2_current_runtime_d1_training_v1.py"
VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_d1_label_join_authority_v1.py"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_d1_label_join_authority_v1"
IDENTITY_REPAIR_REGISTRY = ROOT / "registry/gallery_identity_repair_v1.json"
IDENTITY_REPAIR_CONTRACT = ROOT / "protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json"

EXPECTED_QUERY_COUNTS = {0: (775, 212), 1: (782, 205), 2: (798, 189), 3: (799, 188), 4: (794, 193)}


def read_json(path: Path) -> dict[str, Any]:
    require(path.exists() and path.is_file() and not path.is_symlink(), f"input absent: {path}")
    return json.loads(path.read_text())


def manifest_identity_map(allowed_query_ids: set[str]) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    output: dict[str, tuple[str, str]] = {}
    rejected_before_label_read: list[str] = []

    def add(query_id: str, identity: str, track: str) -> None:
        require(query_id not in output, f"duplicate manifest query id: {query_id}")
        output[query_id] = (identity, track)

    outcome = read_json(OUTCOME)
    require(outcome.get("status") == "B3D_TRAIN_CACHE_COMPLETE", "outcome manifest status drift")
    for row in outcome.get("rows", []):
        query_id = f"OUTCOME-{int(row['index']):04d}"
        if query_id not in allowed_query_ids:
            rejected_before_label_read.append(query_id)
            continue
        add(query_id, str(row["identity"]), "outcome")

    difficult = read_json(DIFFICULT)
    require(difficult.get("status") == "B4_DIFFICULT_QUERY_CACHE_COMPLETE", "difficult manifest status drift")
    for row in difficult.get("rows", []):
        query_id = f"DIFFICULT-{int(row['index']):04d}"
        if query_id not in allowed_query_ids:
            rejected_before_label_read.append(query_id)
            continue
        add(query_id, str(row["identity"]), "difficult")

    new = read_json(NEW_DIFFICULT)
    require(new.get("status") == "C2_NATURAL_QUERY_TOKEN_ONLY_CACHE_COMPLETE", "new_difficult manifest status drift")
    for row in new.get("rows", []):
        if row.get("split") == "train":
            query_id = str(row["query_id"])
            if query_id not in allowed_query_ids:
                rejected_before_label_read.append(query_id)
                continue
            add(query_id, str(row["reference_label"]), "new_difficult_train")
    return output, {
        "outcome_rows": len(outcome.get("rows", [])),
        "difficult_rows": len(difficult.get("rows", [])),
        "new_difficult_train_rows": sum(row.get("split") == "train" for row in new.get("rows", [])),
        "three_manifest_train_row_total": 822 + 134 + sum(row.get("split") == "train" for row in new.get("rows", [])),
        "target_label_semantic_read_count": len(output),
        "rejected_before_label_read_count": len(rejected_before_label_read),
        "rejected_before_label_read_query_ids": sorted(rejected_before_label_read),
    }


def run() -> dict[str, Any]:
    token_bindings = validate_token_authority(ROOT)
    # This process consumes the path-free, already validated token-cache view;
    # it never opens the canonical ledger's source path or historical shards.
    ledger = sanitized_query_index(ROOT)
    allowed_query_ids = {row["query_id"] for row in ledger}
    source, source_counts = manifest_identity_map(allowed_query_ids)
    upstreams = read_json(UPSTREAMS)
    preflight_path = Path(upstreams["primary_line"]["identity_supergroup_preflight"]["path"])
    preflight = read_json(preflight_path)
    require(preflight.get("seed") == 17 and len(preflight.get("folds", [])) == 5, "split preflight drift")
    identities = {
        str(row["identity"]): dict(row)
        for row in preflight.get("identities", [])
        if row.get("final_role") == "optimization"
    }
    require(len(identities) == 81, "optimization identity authority drift")

    _, corrected, gallery_receipt = load_gallery(ROOT)
    positions: dict[str, list[int]] = defaultdict(list)
    for physical_row, identity in enumerate(corrected):
        positions[identity].append(physical_row)

    joined: list[dict[str, Any]] = []
    for row in ledger:
        query_id = str(row["query_id"])
        manifest = source.get(query_id)
        require(manifest is not None, f"canonical query absent from all three manifests: {query_id}")
        identity, track = manifest
        identity_row = identities.get(identity)
        require(
            track == row["track"] and identity_row is not None,
            f"manifest track/optimization identity drift: {query_id}",
        )
        target_rows = positions.get(identity, [])
        # Biogen_21 has two byte-identical rows but is not one of the 81
        # optimization targets.  A positive row is therefore never arbitrary.
        require(len(target_rows) == 1, f"optimization target is not one physical row: {identity}")
        joined.append(
            {
                "query_id": query_id,
                "query_ordinal": int(row["query_ordinal"]),
                "heldout_fold": int(row["heldout_fold"]),
                "track": track,
                "target_identity": identity,
                "supergroup": str(identity_row["group_id"]),
                "target_physical_row": int(target_rows[0]),
            }
        )
    require(len(joined) == len(ledger) == 987, "three-manifest query join is not 987/987")
    expected_rejected = [f"OUTCOME-{index:04d}" for index in range(519, 528)]
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
        and set(source) == allowed_query_ids,
        "three-manifest extra rows were not the exact nine explicitly excluded opened rows",
    )
    manifest_population_audit = {
        **source_counts,
        "optimization_join_count": len(joined),
        "explicitly_excluded_opened_count": 9,
        "explicitly_excluded_query_ids_sha256": canonical_sha256(expected_rejected),
        "missing_optimization_query_count": len(allowed_query_ids - set(source)),
    }

    source_hashes = {
        **label_join_dependency_hashes(ROOT),
        "contract_sha256": sha256_file(CONTRACT),
        "runtime_sha256": sha256_file(RUNTIME),
        "producer_sha256": sha256_file(Path(__file__).resolve()),
        "outcome_manifest_sha256": sha256_file(OUTCOME),
        "difficult_manifest_sha256": sha256_file(DIFFICULT),
        "new_difficult_manifest_sha256": sha256_file(NEW_DIFFICULT),
        "upstreams_sha256": sha256_file(UPSTREAMS),
        "split_preflight_sha256": sha256_file(preflight_path),
        **token_bindings,
    }
    require(
        {role: source_hashes[role] for role in EXPECTED_LABEL_SOURCE_HASHES}
        == EXPECTED_LABEL_SOURCE_HASHES,
        "frozen three-manifest/split source hash drift",
    )
    fold_seals = []
    for fold in range(5):
        fold_spec = preflight["folds"][fold]
        train_ids = set(map(str, fold_spec["train_identities"]))
        heldout_ids = set(map(str, fold_spec["heldout_identities"]))
        train_groups = set(map(str, fold_spec["train_group_ids"]))
        heldout_groups = set(map(str, fold_spec["heldout_group_ids"]))
        require(not (train_ids & heldout_ids) and not (train_groups & heldout_groups), "identity/supergroup split overlap")

        train = [dict(row) for row in joined if row["heldout_fold"] != fold]
        heldout = [row for row in joined if row["heldout_fold"] == fold]
        require((len(train), len(heldout)) == EXPECTED_QUERY_COUNTS[fold], f"fold {fold} query counts drift")
        require(
            {row["target_identity"] for row in train} == train_ids
            and {row["target_identity"] for row in heldout} == heldout_ids
            and {row["supergroup"] for row in train} == train_groups
            and {row["supergroup"] for row in heldout} == heldout_groups,
            f"fold {fold} identity/supergroup membership drift",
        )

        mask_spec = preflight["gallery_masks"][fold]
        excluded = set(map(int, mask_spec["excluded_row_indices"]))
        legal = torch.tensor([row not in excluded for row in range(5_413)], dtype=torch.bool)
        mask_audit = corrected_mask_audit(corrected, legal)
        require(
            mask_audit["eligible_corrected_identities"] == CORRECTED_ELIGIBLE_IDENTITY_COUNTS[fold]
            and all(bool(legal[row["target_physical_row"]]) for row in train)
            and all(not bool(legal[row["target_physical_row"]]) for row in heldout),
            f"fold {fold} corrected legal-mask closure drift",
        )

        train.sort(key=lambda row: row["query_ordinal"])
        payload = {
            "version": VERSION,
            "status": "ROUTEA_N2_D1_FOLD_TRAIN_LABEL_JOIN_AUTHORITY_READY",
            "claim_level": "TRAIN_ONLY_LABEL_JOIN_AUTHORITY_NO_MODEL_UPDATE",
            "fold": fold,
            "records": train,
            "legal_physical_row_mask": legal,
            "legal_physical_row_mask_sha256": tensor_sha256(legal),
            "mask_audit": mask_audit,
            "train_query_count": len(train),
            "heldout_query_count": len(heldout),
            "train_identity_count": len(train_ids),
            "heldout_identity_count": len(heldout_ids),
            "train_supergroup_count": len(train_groups),
            "heldout_supergroup_count": len(heldout_groups),
            "heldout_identity_sha256": canonical_sha256(sorted(heldout_ids)),
            "heldout_supergroup_sha256": canonical_sha256(sorted(heldout_groups)),
            "bindings": source_hashes,
            "access": {
                "target_label_join_count": len(train),
                "heldout_target_label_published_count": 0,
                "query_path_published_count": 0,
                "model_update_count": 0,
                "external_read_count": 0,
                "sealed_read_count": 0,
            },
        }
        fold_dir = OUT_ROOT / f"fold_{fold}"
        payload_path = fold_dir / "payload.pt"
        atomic_torch(payload_path, payload)
        fold_seals.append(
            {
                "fold": fold,
                "payload_sha256": sha256_file(payload_path),
                "train_query_count": len(train),
                "record_manifest_sha256": canonical_sha256(
                    [
                        [row["query_id"], row["query_ordinal"], row["target_identity"], row["supergroup"], row["target_physical_row"]]
                        for row in train
                    ]
                ),
                "legal_mask_sha256": tensor_sha256(legal),
            }
        )

    result = {
        "version": VERSION,
        "status": "ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_READY",
        "claim_level": "ISOLATED_TRAIN_LABEL_JOIN_AUTHORITY_NO_MODEL_UPDATE",
        "query_count": len(joined),
        "track_counts": dict(sorted(Counter(row["track"] for row in joined).items())),
        "optimization_identity_count": len(identities),
        "manifest_population_audit": manifest_population_audit,
        "corrected_gallery": gallery_receipt,
        "folds": fold_seals,
        "bindings": source_hashes,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": "N2_D1_LABEL_JOIN_AUTHORITY_INDEPENDENT_VALIDATION",
        "logical_sha256": "",
    }
    result["logical_sha256"] = logical_sha256(result)
    atomic_json(OUT_ROOT / "result.json", result)
    return result


def main() -> None:
    value = run()
    print(json.dumps({"status": value["status"], "folds": len(value["folds"])}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
