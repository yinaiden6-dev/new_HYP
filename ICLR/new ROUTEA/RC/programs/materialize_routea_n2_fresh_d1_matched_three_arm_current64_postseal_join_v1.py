#!/usr/bin/env python3
"""Create physically separated TRAIN32/EVAL32 postseal label ledgers.

All current64 maps, features, candidate axes, and base scores are sealed and
independently validated before this process reads the N2 fivefold label
authority.  Target membership is recomputed by corrected exact identity on
each fresh C128.  No model or action is computed here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_SEVEN_PARAMETER_CROSSFIT_CONTRACT_V1_20260904.md"
)
TWO_GATE = (
    ROOT
    / "plan/ROUTEA_N2_CANDIDATE_RECALL_AND_OWNERSHIP_TWO_GATE_ADDENDUM_V1_20260903.md"
)
CURRENT_ROOT = (
    ROOT / "results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1"
)
CURRENT_AGGREGATE = CURRENT_ROOT / "validation.json"
SOURCE_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1"
SOURCE = SOURCE_ROOT / "manifest.json"
SOURCE_VALIDATION = SOURCE_ROOT / "independent_validation.json"
LABEL_ROOT = ROOT / "results/routea_matched_three_arm_n2_d1_label_join_authority_v1"
LABEL_RESULT = LABEL_ROOT / "result.json"
LABEL_VALIDATION = LABEL_ROOT / "independent_validation.json"
PAIR_ROOT = (
    ROOT / "results/routea_n2_fresh_d1_matched_three_arm_pair64_postseal_fixed_pairs_v1"
)
PAIR_FIXED = PAIR_ROOT / "fixed_pairs.json"
PAIR_VALIDATION = PAIR_ROOT / "independent_validation.json"
PAIR_AGGREGATE = (
    ROOT
    / "results/routea_n2_fresh_d1_matched_three_arm_pair64_training_features_v1/independent_aggregate_validation.json"
)
GALLERY = (
    ROOT.parents[2]
    / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
)
REPAIR_REGISTRY = ROOT / "registry/gallery_identity_repair_v1.json"
REPAIR_RUNTIME = ROOT / "src/rc_aslo_xf/gallery_identity_repair.py"
REPAIR_CONTRACT = (
    ROOT / "protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json"
)
OUT_ROOT = (
    ROOT / "results/routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1"
)
TRAIN_OUT = OUT_ROOT / "J1_TRAIN32_LABEL_LEDGER.json"
EVAL_OUT = OUT_ROOT / "J1_EVAL32_LABEL_LEDGER.json"
RESULT_OUT = OUT_ROOT / "result.json"

VERSION = "routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1_20260904"
READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_POSTSEAL_JOIN_READY"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING"
EXPECTED_ROLE_POPULATION = {
    "TRAIN": {
        "query_count": 32,
        "identity_count": 12,
        "supergroup_count": 12,
        "TARGET_WINNER": 27,
        "TARGET_NONWINNER": 5,
        "TARGET_ABSENT": 0,
    },
    "EVAL": {
        "query_count": 32,
        "identity_count": 11,
        "supergroup_count": 11,
        "TARGET_WINNER": 24,
        "TARGET_NONWINNER": 8,
        "TARGET_ABSENT": 0,
    },
}
EXPECTED_HASHES = {
    CONTRACT: "1e129df85e71a14eefa814b477bf304771eeda631d14d4d68cc6c2a3b8342f79",
    TWO_GATE: "cb39aeaf3b0826b408b84063f0345bbcb384cb8ceef9ff5ae8760e60d1a86765",
    CURRENT_AGGREGATE: "24ea4c6e416fc9d5e0a57d22aaa957d5bb3f29e4f01022369a62dd7ff48b7859",
    SOURCE: "da7ff4c0ee24bb72d1fa026e3e103e01e3cf3f9cd489e8b33f26835b4065ab58",
    SOURCE_VALIDATION: "654453f5884fc2bc1848a54fe67cde37947f01cdc1b6493d3e764cda4cfb737e",
    LABEL_RESULT: "519cea43968abf8083f3d6c4b98e108d592f29976c470b01f464054d7c9a5861",
    LABEL_VALIDATION: "0b78e01d6940c19db40d606ce6012be190628b31d56e71851c22a97e83cfe3a8",
    PAIR_FIXED: "b89b7b43bf3a74353b2482451d73be999a6e1f36fb81cd4233c5867eb7daccad",
    PAIR_VALIDATION: "786c3fd06dc4cb3f4761ea4783a57c60e55bcce1a6c46dc43e38ab51e11d1633",
    PAIR_AGGREGATE: "90e83f9598f3c9cfcf1a1d80282666be213a70abd98c4f8465ae01a3d6c3074e",
    GALLERY: "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc",
    REPAIR_REGISTRY: "9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f",
    REPAIR_RUNTIME: "995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d",
    REPAIR_CONTRACT: "867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650",
}
FOLD_HASHES = {
    0: "0a9c72582f0af05b9a83d28a6217ef62670a2a89b7bf433247e35e46d0e983f3",
    1: "c411861a4ca6b8f41792893f747d5f66561af718c6681b3d4c4bbece6b40ab72",
    2: "08b89f23d61012460da86191951202dbdb4e7aa4cffb75e2c673e5bb8c5f9a0f",
    3: "a559a678fe056353de607ba92b2bda1e675470be7a477ccdc54513c1395afb94",
    4: "d6014cb1c141efd7cb57a19eed9692b406efddf8ef0f52449690e06b1090c6b3",
}


class J1Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise J1Error(message)


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"input absent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("logical_sha256", None)
    return canonical_sha256(payload)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"JSON absent: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "JSON root")
    return value


def corrected_labels() -> tuple[str, ...]:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from rc_aslo_xf.n2_corrected_d1_runtime_v1 import (
        corrected_labels_from_legacy,
        validate_corrected_identity_axis,
    )

    labels = corrected_labels_from_legacy(
        torch.load(GALLERY, map_location="cpu", weights_only=False, mmap=True)["setids"]
    )
    audit = validate_corrected_identity_axis(labels)
    require(
        len(labels) == 5413
        and audit.get("corrected_identity_count") == 5412
        and audit.get("corrected_axis_sha256")
        == "935ce029e3c8177fd9bc4b51f7b41fff8e241c3d587977c31d516b2f300efca4",
        "corrected axis",
    )
    return labels


def load_current_records() -> (
    tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]
):
    aggregate = read_json(CURRENT_AGGREGATE)
    source = read_json(SOURCE)
    source_validation = read_json(SOURCE_VALIDATION)
    require(
        aggregate.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATED"
        and aggregate.get("current64_query_count") == 64
        and aggregate.get("role_counts") == {"EVAL": 32, "TRAIN": 32}
        and all(aggregate.get("checks", {}).values())
        and aggregate.get("logical_sha256") == logical_sha256(aggregate),
        "current aggregate",
    )
    require(
        source.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_READY"
        and source_validation.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_VALIDATED"
        and source_validation.get("manifest_sha256") == sha256_file(SOURCE)
        and all(source_validation.get("checks", {}).values()),
        "current source",
    )
    seals = {int(item["shard"]): item for item in aggregate["shards"]}
    records = []
    bindings = []
    for shard in range(8):
        base = CURRENT_ROOT / f"shard{shard:02d}"
        payload_path = base / "payload.pt"
        receipt_path = base / "receipt.json"
        validation_path = base / "validation.json"
        seal = seals[shard]
        require(
            seal["payload_sha256"] == sha256_file(payload_path)
            and seal["receipt_sha256"] == sha256_file(receipt_path)
            and seal["validation_sha256"] == sha256_file(validation_path),
            f"current shard{shard} seal",
        )
        payload = torch.load(
            payload_path, map_location="cpu", weights_only=False, mmap=True
        )
        require(
            payload.get("status")
            == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARD_READY"
            and len(payload.get("records", [])) == 8,
            f"current shard{shard} payload",
        )
        records.extend(payload["records"])
        bindings.append(
            {
                "shard": shard,
                "payload_sha256": seal["payload_sha256"],
                "receipt_sha256": seal["receipt_sha256"],
                "validation_sha256": seal["validation_sha256"],
            }
        )
    source_by_query = {str(row["query_id"]): row for row in source["records"]}
    require(
        len(records) == 64
        and len({str(row["query_id"]) for row in records}) == 64
        and set(source_by_query) == {str(row["query_id"]) for row in records},
        "current record population",
    )
    return records, bindings, source_by_query


def label_consensus(
    query_ids: set[str], source_by_query: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    result, validation = read_json(LABEL_RESULT), read_json(LABEL_VALIDATION)
    require(
        result.get("status") == "ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_READY"
        and validation.get("status") == "ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_VALIDATED"
        and validation.get("result_sha256") == sha256_file(LABEL_RESULT)
        and all(validation.get("checks", {}).values()),
        "label authority",
    )
    seals = {int(item["fold"]): item for item in validation["folds"]}
    occ = defaultdict(list)
    fold_bindings = []
    for fold in range(5):
        path = LABEL_ROOT / f"fold_{fold}/payload.pt"
        require(
            sha256_file(path) == FOLD_HASHES[fold] == seals[fold]["payload_sha256"],
            f"label fold{fold}",
        )
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        for row in payload["records"]:
            if row["query_id"] in query_ids:
                occ[str(row["query_id"])].append((fold, row))
        fold_bindings.append({"fold": fold, "payload_sha256": FOLD_HASHES[fold]})
    output = {}
    for query_id in query_ids:
        source = source_by_query[query_id]
        heldout = int(source["heldout_fold"])
        rows = occ[query_id]
        require(
            len(rows) == 4 and {fold for fold, _ in rows} == set(range(5)) - {heldout},
            f"label occurrences {query_id}",
        )
        values = {
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
        require(len(values) == 1, f"label disagreement {query_id}")
        target, group, target_row, ordinal, row_fold, track = next(iter(values))
        require(
            ordinal == int(source["oof_query_ordinal"]) and row_fold == heldout,
            f"canonical role {query_id}",
        )
        output[query_id] = {
            "target_identity": target,
            "target_supergroup": group,
            "target_physical_row_provenance": target_row,
            "canonical_query_ordinal": ordinal,
            "canonical_heldout_fold": heldout,
            "canonical_track": track,
            "source_folds": sorted(fold for fold, _ in rows),
        }
    return output, fold_bindings


def make_role_ledger(
    role: str, records: list[dict[str, Any]], labels: tuple[str, ...]
) -> dict[str, Any]:
    output = []
    for record in sorted(
        (row for row in records if row["role_membership"] == role),
        key=lambda row: int(row["execution_ordinal"]),
    ):
        target = record["_target"]
        axis = list(map(int, record["candidate_physical_rows"]))
        identities = [labels[row] for row in axis]
        positions = [
            index
            for index, identity in enumerate(identities)
            if identity == target["target_identity"]
        ]
        require(
            len(positions) <= 1,
            f"duplicate target identity in C128 {record['query_id']}",
        )
        target_position = positions[0] if positions else None
        state = (
            "TARGET_WINNER"
            if target_position == 0
            else "TARGET_NONWINNER" if target_position is not None else "TARGET_ABSENT"
        )
        target_row = target["target_physical_row_provenance"]
        require(
            labels[target_row] == target["target_identity"]
            and int(record["base_winner_position"]) == 0,
            f"target/base provenance {record['query_id']}",
        )
        binding = {
            "candidate_axis_sha256": record["candidate_axis_sha256"],
            "base_scores_sha256": tensor_sha256(record["base_scores"]),
            "feature_sha256": record["feature_sha256"],
            "adapted_image_tokens_sha256": record["adapted_image_tokens_sha256"],
            "physical_row_scores_sha256": record["physical_row_scores_sha256"],
            "oof_checkpoint_sha256": record["oof_checkpoint_sha256"],
        }
        output.append(
            {
                "query_id": str(record["query_id"]),
                "execution_ordinal": int(record["execution_ordinal"]),
                "canonical_query_ordinal": target["canonical_query_ordinal"],
                "canonical_heldout_fold": target["canonical_heldout_fold"],
                "canonical_track": target["canonical_track"],
                "target_identity": target["target_identity"],
                "target_supergroup": target["target_supergroup"],
                "target_physical_row_provenance": target_row,
                "target_state": state,
                "target_candidate_position": target_position,
                "target_representative_physical_row": (
                    axis[target_position] if target_position is not None else None
                ),
                "base_winner_position": 0,
                "base_winner_representative_physical_row": axis[0],
                "base_winner_corrected_identity": identities[0],
                "n2_label_source_folds": target["source_folds"],
                "targetfree_record_binding": binding,
                "target_insertion_count": 0,
            }
        )
    counts = Counter(row["target_state"] for row in output)
    identities = {row["target_identity"] for row in output}
    groups = {row["target_supergroup"] for row in output}
    expected = EXPECTED_ROLE_POPULATION[role]
    require(
        len(output) == expected["query_count"]
        and len(identities) == expected["identity_count"]
        and len(groups) == expected["supergroup_count"]
        and all(
            counts.get(state, 0) == expected[state]
            for state in ("TARGET_WINNER", "TARGET_NONWINNER", "TARGET_ABSENT")
        ),
        f"{role} population",
    )
    policy = {
        "TRAIN": {
            "readable_by": ["T0_NATIVE7_TRAINING"],
            "forbidden_to": ["A0_TARGET_FREE_ACTION_SEAL", "E0_POSTSEAL_EVALUATION"],
        },
        "EVAL": {
            "readable_by": ["E0_POSTSEAL_EVALUATION"],
            "forbidden_to": ["T0_NATIVE7_TRAINING", "A0_TARGET_FREE_ACTION_SEAL"],
        },
    }[role]
    value = {
        "version": VERSION,
        "status": f"ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_J1_{role}32_LABEL_LEDGER_READY",
        "claim_level": f"POSTSEAL_{role}32_LABEL_LEDGER_NO_MODEL_OR_ACTION",
        "role": role,
        "population": {
            "query_count": len(output),
            "identity_count": len(identities),
            "supergroup_count": len(groups),
            "target_winner_count": counts.get("TARGET_WINNER", 0),
            "target_nonwinner_count": counts.get("TARGET_NONWINNER", 0),
            "target_absent_count": counts.get("TARGET_ABSENT", 0),
        },
        "records": output,
        "record_sequence_sha256": canonical_sha256(output),
        "access_policy": policy,
        "target_insertion_count": 0,
        "model_update_count": 0,
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    return value


def main() -> None:
    require(not OUT_ROOT.exists(), f"immutable J1 output exists: {OUT_ROOT}")
    for path, expected in EXPECTED_HASHES.items():
        require(sha256_file(path) == expected, f"fixed hash drift {path}")
    records, shards, source_by_query = load_current_records()
    query_ids = {str(row["query_id"]) for row in records}
    consensus, folds = label_consensus(query_ids, source_by_query)
    for record in records:
        record["_target"] = consensus[str(record["query_id"])]
    labels = corrected_labels()
    train = make_role_ledger("TRAIN", records, labels)
    evals = make_role_ledger("EVAL", records, labels)
    pair = read_json(PAIR_FIXED)
    pair_q = {row["query_id"] for row in pair["records"]}
    pair_i = {row["target_identity"] for row in pair["records"]}
    pair_g = {row["target_supergroup"] for row in pair["records"]}
    train_q = {row["query_id"] for row in train["records"]}
    train_i = {row["target_identity"] for row in train["records"]}
    train_g = {row["target_supergroup"] for row in train["records"]}
    eval_q = {row["query_id"] for row in evals["records"]}
    eval_i = {row["target_identity"] for row in evals["records"]}
    eval_g = {row["target_supergroup"] for row in evals["records"]}
    overlaps = {
        "TRAIN32_EVAL32": {
            "query": len(train_q & eval_q),
            "identity": len(train_i & eval_i),
            "supergroup": len(train_g & eval_g),
        },
        "PAIR64_TRAIN32": {
            "query": len(pair_q & train_q),
            "identity": len(pair_i & train_i),
            "supergroup": len(pair_g & train_g),
        },
        "PAIR64_EVAL32": {
            "query": len(pair_q & eval_q),
            "identity": len(pair_i & eval_i),
            "supergroup": len(pair_g & eval_g),
        },
    }
    require(
        all(value == 0 for block in overlaps.values() for value in block.values()),
        "three-population disjointness",
    )
    bindings = {
        "contract_sha256": sha256_file(CONTRACT),
        "two_gate_sha256": sha256_file(TWO_GATE),
        "current64_aggregate_sha256": sha256_file(CURRENT_AGGREGATE),
        "current64_aggregate_logical_sha256": read_json(CURRENT_AGGREGATE)[
            "logical_sha256"
        ],
        "current64_source_sha256": sha256_file(SOURCE),
        "current64_source_validation_sha256": sha256_file(SOURCE_VALIDATION),
        "n2_label_result_sha256": sha256_file(LABEL_RESULT),
        "n2_label_validation_sha256": sha256_file(LABEL_VALIDATION),
        "n2_label_fold_payloads": folds,
        "pair64_fixed_pairs_sha256": sha256_file(PAIR_FIXED),
        "pair64_validation_sha256": sha256_file(PAIR_VALIDATION),
        "pair64_training_input_aggregate_sha256": sha256_file(PAIR_AGGREGATE),
        "corrected_axis_sha256": "935ce029e3c8177fd9bc4b51f7b41fff8e241c3d587977c31d516b2f300efca4",
        "current64_shards": shards,
        "producer_sha256": sha256_file(Path(__file__).resolve()),
    }
    access = {
        "targetfree_current64_query_count": 64,
        "n2_label_record_occurrence_count": 256,
        "target_identity_read_count": 64,
        "target_supergroup_read_count": 64,
        "target_physical_row_provenance_read_count": 64,
        "historical_candidate_position_read_count": 0,
        "target_insertion_count": 0,
        "model_forward_count": 0,
        "model_update_count": 0,
        "external_read_count": 0,
        "sealed_read_count": 0,
    }
    staging = Path(tempfile.mkdtemp(prefix=".current64-j1-", dir=OUT_ROOT.parent))
    try:
        for name, value in ((TRAIN_OUT.name, train), (EVAL_OUT.name, evals)):
            (staging / name).write_text(
                json.dumps(
                    value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
                )
                + "\n"
            )
        result = {
            "version": VERSION,
            "status": READY,
            "claim_level": "POSTSEAL_CURRENT64_LABEL_JOIN_ONLY_NO_MODEL_OR_ACTION",
            "population": {
                "TRAIN32": train["population"],
                "EVAL32": evals["population"],
                "Pair64": {
                    "query_count": 64,
                    "identity_count": 20,
                    "supergroup_count": 20,
                },
            },
            "three_population_overlap_counts": overlaps,
            "train32_ledger": {
                "path": TRAIN_OUT.name,
                "sha256": sha256_file(staging / TRAIN_OUT.name),
                "logical_sha256": train["logical_sha256"],
            },
            "eval32_ledger": {
                "path": EVAL_OUT.name,
                "sha256": sha256_file(staging / EVAL_OUT.name),
                "logical_sha256": evals["logical_sha256"],
            },
            "bindings": bindings,
            "access": access,
            "scientific_GO_or_NO_GO": None,
            "ownership_GO_or_NO_GO": None,
            "automatic_stage_advance": False,
            "next_authorized_stage": NEXT,
            "logical_sha256": "",
        }
        result["logical_sha256"] = logical_sha256(result)
        (staging / RESULT_OUT.name).write_text(
            json.dumps(
                result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            + "\n"
        )
        os.rename(staging, OUT_ROOT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(
        json.dumps(
            {
                "status": READY,
                "TRAIN32": train["population"],
                "EVAL32": evals["population"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
