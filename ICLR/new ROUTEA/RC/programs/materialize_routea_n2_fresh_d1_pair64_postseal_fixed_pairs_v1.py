#!/usr/bin/env python3
"""Join postseal exact labels and freeze one training pair per Pair64 query.

The primary label is the four-record consensus from the independently
validated N2 fivefold label authority.  CW0 is read only as a secondary
identity/supergroup equality check.  No historical candidate position, axis,
score, map, feature, action, or fold can select an endpoint.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUT_EXECUTION_CONTRACT_V1_20260904.md"
SCOPE = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_PAIR_SCOPE_CORRECTIVE_ADDENDUM_V1_20260904.md"
LABEL_CORRECTION = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_LABEL_AUTHORITY_CORRECTION_ADDENDUM_V1_20260904.md"
DISPOSITION = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_ACTIVE_CONTRACT_DISPOSITION_AND_EXECUTION_CLARIFICATION_V1_20260904.md"
P0_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1"
P0 = P0_ROOT / "preseal.json"
P0_VALIDATION = P0_ROOT / "independent_validation.json"
P0_PRODUCER = ROOT / "programs/materialize_routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1.py"
P0_VALIDATOR = ROOT / "programs/validate_routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1.py"
CURRENT64_SOURCE_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1"
CURRENT64_SOURCE = CURRENT64_SOURCE_ROOT / "manifest.json"
CURRENT64_SOURCE_VALIDATION = CURRENT64_SOURCE_ROOT / "independent_validation.json"
LABEL_ROOT = ROOT / "results/routea_matched_three_arm_n2_d1_label_join_authority_v1"
LABEL_RESULT = LABEL_ROOT / "result.json"
LABEL_VALIDATION = LABEL_ROOT / "independent_validation.json"
CW_ROOT = ROOT / "results/cw0_rgh_xf_v2_p0_a0_manifest_v2"
CW_SOURCE = CW_ROOT / "source_manifest.json"
CW_ROLE = CW_ROOT / "role_manifest.json"
CW_RESULT = CW_ROOT / "result.json"
CW_VALIDATION = CW_ROOT / "independent_validation.json"
GALLERY = ROOT.parents[2] / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_pair64_postseal_fixed_pairs_v1"
OUT = OUT_ROOT / "fixed_pairs.json"

VERSION = "routea_n2_fresh_d1_matched_three_arm_pair64_postseal_fixed_pairs_v1_20260904"
READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIRS_READY"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_FRESH_ROMA_PAIR_FEATURES"
EXPECTED_BRANCH_COUNTS = {
    "TARGET_WINNER_NEGATIVE_HOLD": 37,
    "TARGET_NONWINNER_POSITIVE_SWITCH": 27,
    "TARGET_ABSENT_NEGATIVE_HOLD": 0,
}
EXPECTED_HASHES = {
    ACTIVE: "bc53ef74ec71283d79247a2117f25e2c708c2d30bb6fda1fad194b5668c97ade",
    SCOPE: "19fa1ca3e5b898adec22c9148c397d7765af48aa23a5057d155115649fdbe124",
    LABEL_CORRECTION: "9c94f79cb718e7a111a983fa045814028cba1d737553a9c60daa46f118f39035",
    DISPOSITION: "502de768d26584613bb89f329e214e431d959e821fe6caa3ab5cd1cb5ad538b8",
    LABEL_RESULT: "519cea43968abf8083f3d6c4b98e108d592f29976c470b01f464054d7c9a5861",
    LABEL_VALIDATION: "0b78e01d6940c19db40d606ce6012be190628b31d56e71851c22a97e83cfe3a8",
    CW_SOURCE: "e0be35125eddec391a92398e84103e77ca0a65f661452b9190c2b81f3fdbea23",
    CW_ROLE: "2f104f4fbf71bada1b6186fa3d0915fa7f8059c65d6798414ab00043e5835454",
    CW_RESULT: "1c3cd17e1349a84be7fb7af5d5b796874f3ec3d7a61539b3c6e535dddac04e08",
    CW_VALIDATION: "ae735624176e5e400ff5c16874b7f73b0505ce71f6814d0c4ed515d94455f8ac",
    GALLERY: "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc",
    P0_PRODUCER: "068c8297a0b773c4efefa093f2887393588213dce7ca347cb5d6517cc22e13b5",
    P0_VALIDATOR: "239f67385076b72b3f28c4223a5c4208f7ee64ba90c22008d618aaa627adf6db",
    CURRENT64_SOURCE: "da7ff4c0ee24bb72d1fa026e3e103e01e3cf3f9cd489e8b33f26835b4065ab58",
    CURRENT64_SOURCE_VALIDATION: "654453f5884fc2bc1848a54fe67cde37947f01cdc1b6493d3e764cda4cfb737e",
}
FOLD_PAYLOAD_HASHES = {
    0: "0a9c72582f0af05b9a83d28a6217ef62670a2a89b7bf433247e35e46d0e983f3",
    1: "c411861a4ca6b8f41792893f747d5f66561af718c6681b3d4c4bbece6b40ab72",
    2: "08b89f23d61012460da86191951202dbdb4e7aa4cffb75e2c673e5bb8c5f9a0f",
    3: "a559a678fe056353de607ba92b2bda1e675470be7a477ccdc54513c1395afb94",
    4: "d6014cb1c141efd7cb57a19eed9692b406efddf8ef0f52449690e06b1090c6b3",
}


class J0Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise J0Error(message)


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"input absent/non-regular: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("logical_sha256", None)
    return canonical_sha256(payload)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"JSON input absent: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON root is not object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        require(path.is_file() and not path.is_symlink() and path.read_text() == encoded, "immutable J0 output drift")
        return
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validated_p0() -> tuple[dict[str, Any], dict[str, Any], str]:
    before = sha256_file(P0)
    preseal = read_json(P0)
    validation = read_json(P0_VALIDATION)
    require(
        preseal.get("status") == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_READY"
        and preseal.get("logical_sha256") == logical_sha256(preseal)
        and len(preseal.get("records", [])) == 64
        and preseal.get("population", {}).get("query_count") == 64
        and preseal.get("corrected_axis_sha256") == "935ce029e3c8177fd9bc4b51f7b41fff8e241c3d587977c31d516b2f300efca4"
        and preseal.get("next_authorized_stage") == "N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIR_JOIN",
        "Pair64 P0 producer seal failed",
    )
    require(
        validation.get("status") == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_VALIDATED"
        and validation.get("preseal_sha256") == before
        and validation.get("preseal_logical_sha256") == preseal["logical_sha256"]
        and validation.get("query_count") == 64
        and validation.get("producer_sha256") == sha256_file(P0_PRODUCER)
        and validation.get("validator_sha256") == sha256_file(P0_VALIDATOR)
        and all(validation.get("checks", {}).values())
        and validation.get("logical_sha256") == logical_sha256(validation)
        and validation.get("next_authorized_stage") == "N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIR_JOIN",
        "Pair64 P0 independent validation failed",
    )
    return preseal, validation, before


def corrected_labels() -> tuple[str, ...]:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from rc_aslo_xf.n2_corrected_d1_runtime_v1 import corrected_labels_from_legacy, validate_corrected_identity_axis

    gallery = torch.load(GALLERY, map_location="cpu", weights_only=False, mmap=True)
    labels = corrected_labels_from_legacy(gallery["setids"])
    audit = validate_corrected_identity_axis(labels)
    require(
        len(labels) == 5413
        and audit.get("corrected_identity_count") == 5412
        and audit.get("corrected_axis_sha256") == "935ce029e3c8177fd9bc4b51f7b41fff8e241c3d587977c31d516b2f300efca4",
        "corrected gallery authority failed",
    )
    return labels


def label_consensus(query_ids: set[str], preseal_by_query: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    result = read_json(LABEL_RESULT)
    validation = read_json(LABEL_VALIDATION)
    require(
        result.get("status") == "ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_READY"
        and result.get("query_count") == 987
        and result.get("logical_sha256") == logical_sha256(result)
        and validation.get("status") == "ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_VALIDATED"
        and validation.get("query_count") == 987
        and validation.get("result_sha256") == sha256_file(LABEL_RESULT)
        and all(validation.get("checks", {}).values())
        and validation.get("logical_sha256") == logical_sha256(validation),
        "N2 label authority envelope failed",
    )
    occurrences: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    fold_seals = {int(item["fold"]): item for item in validation["folds"]}
    fold_bindings = []
    for fold in range(5):
        path = LABEL_ROOT / f"fold_{fold}/payload.pt"
        require(
            sha256_file(path) == FOLD_PAYLOAD_HASHES[fold]
            and fold_seals[fold]["payload_sha256"] == FOLD_PAYLOAD_HASHES[fold],
            f"N2 fold {fold} label payload seal drift",
        )
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        require(payload.get("status") == "ROUTEA_N2_D1_FOLD_TRAIN_LABEL_JOIN_AUTHORITY_READY" and int(payload["fold"]) == fold, f"N2 fold {fold} label payload envelope")
        for record in payload["records"]:
            query_id = str(record["query_id"])
            if query_id in query_ids:
                occurrences[query_id].append((fold, dict(record)))
        fold_bindings.append({"fold": fold, "payload_sha256": FOLD_PAYLOAD_HASHES[fold]})
    output: dict[str, dict[str, Any]] = {}
    for query_id in query_ids:
        rows = occurrences.get(query_id, [])
        preseal = preseal_by_query[query_id]
        heldout = int(preseal["canonical_heldout_fold"])
        require(len(rows) == 4 and {fold for fold, _ in rows} == set(range(5)) - {heldout}, f"N2 fourfold occurrence failure: {query_id}")
        tuples = {
            (
                str(row["target_identity"]),
                str(row["supergroup"]),
                int(row["target_physical_row"]),
                int(row["query_ordinal"]),
                int(row["heldout_fold"]),
                str(row["track"]),
            )
            for _, row in rows
        }
        require(len(tuples) == 1, f"N2 fourfold label disagreement: {query_id}")
        target, group, physical, ordinal, row_heldout, track = next(iter(tuples))
        require(
            ordinal == int(preseal["canonical_query_ordinal"])
            and row_heldout == heldout
            and track == preseal["canonical_track"],
            f"N2 label/canonical role disagreement: {query_id}",
        )
        output[query_id] = {
            "target_identity": target,
            "target_supergroup": group,
            "target_physical_row_provenance": physical,
            "source_folds": sorted(fold for fold, _ in rows),
            "consensus_occurrence_count": 4,
        }
    require(len(output) == 64, "N2 consensus did not cover Pair64")
    return output, fold_bindings


def current_eval_consensus() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source = read_json(CURRENT64_SOURCE)
    validation = read_json(CURRENT64_SOURCE_VALIDATION)
    require(
        source.get("status") == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_READY"
        and source.get("population", {}).get("query_count") == 64
        and source.get("population", {}).get("role_counts") == {"EVAL": 32, "TRAIN": 32}
        and source.get("logical_sha256") == logical_sha256(source)
        and validation.get("status") == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_VALIDATED"
        and validation.get("manifest_sha256") == sha256_file(CURRENT64_SOURCE)
        and validation.get("manifest_logical_sha256") == source["logical_sha256"]
        and all(validation.get("checks", {}).values())
        and validation.get("logical_sha256") == logical_sha256(validation),
        "current64 source authority failed before disjointness audit",
    )
    eval_rows = {
        str(row["query_id"]): row for row in source["records"] if row["role"] == "EVAL"
    }
    require(len(eval_rows) == 32, "current EVAL32 query population drift")
    occurrences: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for fold in range(5):
        payload = torch.load(
            LABEL_ROOT / f"fold_{fold}/payload.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        for record in payload["records"]:
            query_id = str(record["query_id"])
            if query_id in eval_rows:
                occurrences[query_id].append((fold, dict(record)))
    output: dict[str, dict[str, Any]] = {}
    for query_id, source_row in eval_rows.items():
        rows = occurrences.get(query_id, [])
        heldout = int(source_row["heldout_fold"])
        require(
            len(rows) == 4
            and {fold for fold, _ in rows} == set(range(5)) - {heldout},
            f"EVAL32 fourfold label occurrence failure: {query_id}",
        )
        values = {
            (
                str(row["target_identity"]),
                str(row["supergroup"]),
                int(row["target_physical_row"]),
                int(row["query_ordinal"]),
                int(row["heldout_fold"]),
            )
            for _, row in rows
        }
        require(len(values) == 1, f"EVAL32 fourfold label disagreement: {query_id}")
        target, group, physical, ordinal, row_heldout = next(iter(values))
        require(
            ordinal == int(source_row["oof_query_ordinal"])
            and row_heldout == heldout,
            f"EVAL32 source/label canonical role disagreement: {query_id}",
        )
        output[query_id] = {
            "target_identity": target,
            "target_supergroup": group,
            "target_physical_row_provenance": physical,
        }
    return output, {
        "manifest_sha256": sha256_file(CURRENT64_SOURCE),
        "validation_sha256": sha256_file(CURRENT64_SOURCE_VALIDATION),
    }


def cw0_secondary(query_ids: set[str]) -> tuple[dict[str, tuple[str, str]], str]:
    source = read_json(CW_SOURCE)
    role = read_json(CW_ROLE)
    result = read_json(CW_RESULT)
    validation = read_json(CW_VALIDATION)
    require(
        source.get("status") == "RGH_P0_A0_SOURCE_MANIFEST_READY"
        and source.get("logical_sha256") == logical_sha256(source)
        and role.get("status") == "RGH_P0_A0_ROLE_MANIFEST_READY"
        and role.get("role_shard_count") == 600
        and role.get("logical_sha256") == logical_sha256(role)
        and role.get("combined_role_runtime_read_authorized") is False
        and result.get("status") == "RGH_P0_A0_MANIFEST_V2_READY"
        and result.get("logical_sha256") == logical_sha256(result)
        and validation.get("status") == "RGH_P0_A0_MANIFEST_V2_INDEPENDENT_VALIDATION_PASS"
        and validation.get("role_receipt_count") == 600
        and validation.get("logical_sha256") == logical_sha256(validation),
        "CW0 secondary authority envelope failed",
    )
    source_by_query = {str(row["query_id"]): row for row in source["records"]}
    require(len(source_by_query) == 600 and query_ids <= set(source_by_query), "CW0 source query-address population failed")
    seals_by_execution = {int(seal["execution_ordinal"]): seal for seal in role["shards"]}
    require(len(seals_by_execution) == 600, "CW0 role seal-address population failed")
    found: dict[str, tuple[str, str]] = {}
    selected_seals = []
    for query_id in sorted(query_ids):
        execution = int(source_by_query[query_id]["execution_ordinal"])
        seal = seals_by_execution[execution]
        path = Path(seal["path"])
        require(sha256_file(path) == seal["sha256"], "CW0 role-shard physical seal drift")
        row = read_json(path)
        require(row.get("logical_sha256") == seal["logical_sha256"], "CW0 role-shard logical seal drift")
        require(str(row["query_id"]) == query_id and query_id not in found, f"CW0 secondary address/query mismatch: {query_id}")
        found[query_id] = (str(row["identity"]), str(row["supergroup"]))
        selected_seals.append([query_id, seal["sha256"], seal["logical_sha256"]])
    require(set(found) == query_ids, "CW0 secondary did not cover Pair64")
    return found, canonical_sha256(selected_seals)


def main() -> None:
    for path, expected in EXPECTED_HASHES.items():
        require(sha256_file(path) == expected, f"fixed authority hash drift: {path}")
    preseal, p0_validation, before_hash = validated_p0()
    records = preseal["records"]
    require([int(row["pair64_ordinal"]) for row in records] == list(range(64)), "Pair64 P0 order drift")
    by_query = {str(row["query_id"]): row for row in records}
    require(len(by_query) == 64, "Pair64 P0 query IDs not unique")
    query_ids = set(by_query)
    consensus, fold_bindings = label_consensus(query_ids, by_query)
    eval_consensus, current64_bindings = current_eval_consensus()
    pair_identities = {row["target_identity"] for row in consensus.values()}
    pair_groups = {row["target_supergroup"] for row in consensus.values()}
    eval_identities = {row["target_identity"] for row in eval_consensus.values()}
    eval_groups = {row["target_supergroup"] for row in eval_consensus.values()}
    disjointness_audit = {
        "pair64_query_count": len(query_ids),
        "current_eval32_query_count": len(eval_consensus),
        "pair64_distinct_target_identity_count": len(pair_identities),
        "pair64_distinct_supergroup_count": len(pair_groups),
        "current_eval32_distinct_target_identity_count": len(eval_identities),
        "current_eval32_distinct_supergroup_count": len(eval_groups),
        "query_id_overlap_count": len(query_ids & set(eval_consensus)),
        "target_identity_overlap_count": len(pair_identities & eval_identities),
        "supergroup_overlap_count": len(pair_groups & eval_groups),
        "pair64_target_identity_sha256": canonical_sha256(sorted(pair_identities)),
        "pair64_supergroup_sha256": canonical_sha256(sorted(pair_groups)),
        "current_eval32_target_identity_sha256": canonical_sha256(sorted(eval_identities)),
        "current_eval32_supergroup_sha256": canonical_sha256(sorted(eval_groups)),
    }
    require(
        disjointness_audit["pair64_distinct_target_identity_count"] == 20
        and disjointness_audit["pair64_distinct_supergroup_count"] == 20
        and disjointness_audit["current_eval32_distinct_target_identity_count"] == 11
        and disjointness_audit["current_eval32_distinct_supergroup_count"] == 11
        and disjointness_audit["query_id_overlap_count"] == 0
        and disjointness_audit["target_identity_overlap_count"] == 0
        and disjointness_audit["supergroup_overlap_count"] == 0,
        "Pair64/current EVAL32 identity-supergroup disjointness failed",
    )
    cw0, cw0_selected_hash = cw0_secondary(query_ids)
    labels = corrected_labels()

    output_records = []
    cw_identity_disagreements = 0
    cw_group_disagreements = 0
    for preseal_record in records:
        query_id = str(preseal_record["query_id"])
        target = consensus[query_id]
        cw_identity_disagreements += int(cw0[query_id][0] != target["target_identity"])
        cw_group_disagreements += int(cw0[query_id][1] != target["target_supergroup"])
        target_row = int(target["target_physical_row_provenance"])
        require(0 <= target_row < 5413 and labels[target_row] == target["target_identity"], f"target-row provenance does not name target: {query_id}")
        c128 = preseal_record["natural_c128"]
        rows = list(map(int, c128["representative_physical_rows"]))
        identities = list(map(str, c128["corrected_identities"]))
        require(
            len(rows) == len(set(rows)) == len(identities) == len(set(identities)) == 128
            and identities == [labels[row] for row in rows]
            and c128["sha256"] == canonical_sha256([[i, row, identities[i]] for i, row in enumerate(rows)]),
            f"sealed corrected C128 drift: {query_id}",
        )
        winner = preseal_record["base_winner"]
        require(
            winner["candidate_position"] == 0
            and winner["representative_physical_row"] == rows[0]
            and winner["corrected_identity"] == identities[0],
            f"sealed base winner drift: {query_id}",
        )
        target_identity = target["target_identity"]
        if target_identity == identities[0]:
            branch = "TARGET_WINNER_NEGATIVE_HOLD"
            counterpart_position = 1
            switch_label = False
            target_position = 0
        elif target_identity in identities:
            branch = "TARGET_NONWINNER_POSITIVE_SWITCH"
            counterpart_position = identities.index(target_identity)
            switch_label = True
            target_position = counterpart_position
        else:
            branch = "TARGET_ABSENT_NEGATIVE_HOLD"
            counterpart_position = 1
            switch_label = False
            target_position = None
        require(counterpart_position != 0, f"fixed pair endpoints are not distinct: {query_id}")
        record = {
            "pair64_ordinal": int(preseal_record["pair64_ordinal"]),
            "pair_cohort": str(preseal_record["pair_cohort"]),
            "query_id": query_id,
            "canonical_query_ordinal": int(preseal_record["canonical_query_ordinal"]),
            "canonical_heldout_fold": int(preseal_record["canonical_heldout_fold"]),
            "canonical_track": str(preseal_record["canonical_track"]),
            "target_identity": target_identity,
            "target_supergroup": target["target_supergroup"],
            "target_physical_row_provenance": target_row,
            "n2_label_source_folds": target["source_folds"],
            "target_state": branch,
            "target_naturally_in_fresh_c128": target_position is not None,
            "target_candidate_position": target_position,
            "winner": {
                "candidate_position": 0,
                "representative_physical_row": rows[0],
                "corrected_identity": identities[0],
            },
            "counterpart": {
                "candidate_position": counterpart_position,
                "representative_physical_row": rows[counterpart_position],
                "corrected_identity": identities[counterpart_position],
            },
            "switch_label": switch_label,
            "preseal_record_sha256": canonical_sha256(preseal_record),
            "preseal_natural_c128_sha256": c128["sha256"],
            "target_insertion_count": 0,
            "roma_evaluation_count": 0,
            "feature_computation_count": 0,
            "model_update_count": 0,
        }
        record["fixed_pair_sha256"] = canonical_sha256(record)
        output_records.append(record)

    observed_branches = Counter(record["target_state"] for record in output_records)
    branch_counts = {
        branch: int(observed_branches.get(branch, 0))
        for branch in EXPECTED_BRANCH_COUNTS
    }
    require(
        branch_counts == EXPECTED_BRANCH_COUNTS
        and cw_identity_disagreements == 0
        and cw_group_disagreements == 0
        and sum(record["switch_label"] for record in output_records) == 27
        and sha256_file(P0) == before_hash,
        "Pair64 fixed-pair branch/CW0/P0 invariant failed",
    )
    bindings = {
        "active_contract_sha256": sha256_file(ACTIVE),
        "scope_correction_sha256": sha256_file(SCOPE),
        "label_correction_sha256": sha256_file(LABEL_CORRECTION),
        "active_disposition_sha256": sha256_file(DISPOSITION),
        "p0_preseal_sha256": before_hash,
        "p0_preseal_logical_sha256": preseal["logical_sha256"],
        "p0_validation_sha256": sha256_file(P0_VALIDATION),
        "p0_validation_logical_sha256": p0_validation["logical_sha256"],
        "p0_producer_sha256": sha256_file(P0_PRODUCER),
        "p0_validator_sha256": sha256_file(P0_VALIDATOR),
        "current64_source_manifest_sha256": current64_bindings["manifest_sha256"],
        "current64_source_validation_sha256": current64_bindings["validation_sha256"],
        "n2_label_result_sha256": sha256_file(LABEL_RESULT),
        "n2_label_validation_sha256": sha256_file(LABEL_VALIDATION),
        "n2_fold_payloads": fold_bindings,
        "cw0_source_manifest_sha256": sha256_file(CW_SOURCE),
        "cw0_role_manifest_sha256": sha256_file(CW_ROLE),
        "cw0_result_sha256": sha256_file(CW_RESULT),
        "cw0_validation_sha256": sha256_file(CW_VALIDATION),
        "cw0_selected64_role_shards_sha256": cw0_selected_hash,
        "gallery_sha256": sha256_file(GALLERY),
        "corrected_axis_sha256": preseal["corrected_axis_sha256"],
        "producer_sha256": sha256_file(Path(__file__).resolve()),
    }
    access = {
        "pair64_query_count": 64,
        "p0_target_bearing_read_count": 0,
        "n2_pair64_label_record_occurrence_count": 256,
        "n2_current_eval32_label_record_occurrence_count": 128,
        "n2_nonheldout_records_per_query": 4,
        "n2_four_record_disagreement_count": 0,
        "cw0_secondary_role_shard_deserialization_count": 64,
        "cw0_secondary_address_lookup_count": 64,
        "cw0_secondary_query_count": 64,
        "cw0_identity_disagreement_count": 0,
        "cw0_supergroup_disagreement_count": 0,
        "cw0_target_candidate_position_consumption_count": 0,
        "cw0_execution_or_inner_fold_consumption_count": 0,
        "current_eval32_target_identity_read_count": 32,
        "current_eval32_supergroup_read_count": 32,
        "historical_pair_axis_or_position_consumption_count": 0,
        "target_insertion_count": 0,
        "roma_evaluation_count": 0,
        "feature_computation_count": 0,
        "model_forward_count": 0,
        "model_update_count": 0,
        "external_read_count": 0,
        "sealed_read_count": 0,
    }
    value: dict[str, Any] = {
        "version": VERSION,
        "status": READY,
        "claim_level": "POSTSEAL_TRAINING_ONLY_FIXED_PAIR_MANIFEST_NO_MAPS_FEATURES_OR_SCIENTIFIC_CLAIM",
        "population": {
            "query_count": 64,
            "fixed_pair_count": 64,
            "selected_endpoint_count": 128,
            "branch_counts": branch_counts,
            "switch_label_count": 27,
            "hold_label_count": 37,
        },
        "pair64_current_eval32_disjointness": disjointness_audit,
        "records": output_records,
        "record_sequence_sha256": canonical_sha256(output_records),
        "p0_hash_before_target_join": before_hash,
        "p0_hash_after_target_join": sha256_file(P0),
        "p0_unchanged_after_target_join": True,
        "bindings": bindings,
        "access": access,
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": NEXT,
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(OUT, value)
    print(json.dumps({"status": READY, "branches": branch_counts, "pairs": 64}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
