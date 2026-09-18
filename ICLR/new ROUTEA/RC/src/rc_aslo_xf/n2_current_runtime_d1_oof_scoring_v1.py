"""Shared fail-closed boundary for N2 fresh-D1 full-gallery OOF scoring.

This module contains no query-target join.  Producer and independent validator
share only immutable schemas, hashing helpers and predecessor checks; the
validator never imports the producer.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rc_aslo_xf.n2_current_runtime_d1_training_v1 import (
    PARAMETER_COUNT,
    PHYSICAL_ROWS,
    QUERY_COUNT,
    SHARD_COUNT,
    TRAIN_STATUS,
    TRAIN_VALID_STATUS,
    VERSION as TRAIN_VERSION,
    algorithm_dependency_hashes,
    canonical_sha256,
    expected_shard_interval,
    load_token_shard,
    logical_sha256,
    parameter_manifest,
    require,
    require_regular,
    sha256_file,
    state_dict_sha256,
    tensor_sha256,
    validate_token_authority,
)
from rc_aslo_xf.n2_corrected_d1_runtime_v1 import (
    CORRECTED_IDENTITY_COUNT,
    NATURAL_CANDIDATE_COUNT,
    validate_corrected_identity_axis,
)


VERSION = "routea_n2_current_runtime_d1_oof_scoring_v1_20260903"
OOF_SHARD_STATUS = "ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_PREJOIN_SHARD_READY"
OOF_SHARD_VALID_STATUS = "ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_PREJOIN_SHARD_VALIDATED"
OOF_AGGREGATE_VALID_STATUS = "ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_PREJOIN_AGGREGATE_VALIDATED"
FOLDS = tuple(range(5))
EXPECTED_FOLD_COUNTS = {0: 212, 1: 205, 2: 189, 3: 188, 4: 193}
EXPECTED_TRACK_COUNTS = {"outcome": 813, "difficult": 134, "new_difficult_train": 40}
CORRECTED_IDENTITIES = CORRECTED_IDENTITY_COUNT
CANDIDATE_COUNT = NATURAL_CANDIDATE_COUNT
EXPECTED_TWO_GATE_ADDENDUM_SHA256 = "cb39aeaf3b0826b408b84063f0345bbcb384cb8ceef9ff5ae8760e60d1a86765"
EXPECTED_TRAIN_CONTRACT_SHA256 = "fb1e836fb80061b1b6de9f1e57cd3801e38a47c0c58757e42b936ad4a4dac42e"
EXPECTED_OOF_CONTRACT_SHA256 = "7bbe3b5c66721e7ab5433a4dbe7af2901546330fa95b6a0ecee95aa83e8dfa1e"
EXPECTED_R5_RESULT_SHA256 = "42185a44380b9522053e0983ed5f2b222bc08eaba38c631217551e924704d846"
EXPECTED_FIXED235_LEDGER_SHA256 = "46d3566f0eea4ea07e2f070f2d5894855fb963964774af92b9b9aae507eaee85"
EXPECTED_FIXED235_LEDGER_LOGICAL_SHA256 = "9fa4b2f6c9492d814cf8a705602aea64528599c921cee4d01b9b9368f26cc521"
EXPECTED_FIXED235_VALIDATION_SHA256 = "821cea7b941050c0730e60f68efc55fe295659e84e806283b8b160d726fc52c7"
EXPECTED_FIXED235_VALIDATION_LOGICAL_SHA256 = "48c6b241f761bb6fadba3a74f892abad389661803fe8e19f8460a8de13fa1b1e"
EXPECTED_R5_LEDGER_SHA256 = {
    0: "37e28d4810168a27bae491ac21688547142c7fb26fd01cb49062f93814215500",
    1: "b549bbd0f2b10771b352538e73f75db8ec026a162ae4dcc760aadb0f44ecc9bb",
    2: "2d40209992b4f27e8e4d236c2e62d02e6cee8caf986aec0d4c7f53049e7d84f6",
    3: "836f0ae1d9cbc2c8ab93987ec2a7db797db0828610c806d5cac2b30c87283098",
    4: "06aef5a9009aabbb676d6195534950d2b611450e5be0157ae8a8b7ddc415f15e",
}
EXECUTION_AUTHORITY_NAME = "ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_SCORING_EXECUTION_AUTHORITY_V3_20260903.json"


def contract_paths(root: Path) -> dict[str, Path]:
    return {
        "oof_contract_sha256": root / "plan/ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_SCORING_AND_CANDIDATE_RECALL_GATE_V1_20260903.md",
        "two_gate_addendum_sha256": root / "plan/ROUTEA_N2_CANDIDATE_RECALL_AND_OWNERSHIP_TWO_GATE_ADDENDUM_V1_20260903.md",
        "train_contract_sha256": root / "plan/ROUTEA_MATCHED_THREE_ARM_N2_CURRENT_RUNTIME_FOLD_LOCAL_D1_TRAINING_V1_20260903.md",
        "oof_runtime_sha256": root / "src/rc_aslo_xf/n2_current_runtime_d1_oof_scoring_v1.py",
    }


def execution_code_paths(root: Path) -> dict[str, Path]:
    return {
        "runtime_sha256": root / "src/rc_aslo_xf/n2_current_runtime_d1_oof_scoring_v1.py",
        "producer_sha256": root / "programs/materialize_routea_matched_three_arm_n2_d1_oof_prejoin_shard_v1.py",
        "shard_validator_sha256": root / "programs/validate_routea_matched_three_arm_n2_d1_oof_prejoin_shard_v1.py",
        "aggregate_validator_sha256": root / "programs/validate_routea_matched_three_arm_n2_d1_oof_prejoin_aggregate_v1.py",
        "postseal_evaluator_sha256": root / "programs/evaluate_routea_matched_three_arm_n2_d1_candidate_recall_gate_v1.py",
        "postseal_validator_sha256": root / "programs/validate_routea_matched_three_arm_n2_d1_candidate_recall_gate_v1.py",
    }


def validate_execution_authority(root: Path) -> dict[str, str]:
    path = root / f"plan/{EXECUTION_AUTHORITY_NAME}"
    require_regular(path, "fresh-D1 OOF execution authority")
    value = json.loads(path.read_text())
    code_paths = execution_code_paths(root)
    for role, code_path in code_paths.items():
        require_regular(code_path, role)
    observed_code = {role: sha256_file(code_path) for role, code_path in code_paths.items()}
    require(
        value.get("version") == "routea_n2_current_runtime_d1_oof_scoring_execution_authority_v3_20260903"
        and value.get("status") == "FROZEN_BEFORE_N2_FRESH_D1_OOF_SCORING"
        and value.get("base_contract_sha256") == EXPECTED_OOF_CONTRACT_SHA256
        and value.get("two_gate_addendum_sha256") == EXPECTED_TWO_GATE_ADDENDUM_SHA256
        and value.get("fixed235_ledger_sha256") == EXPECTED_FIXED235_LEDGER_SHA256
        and value.get("fixed235_ledger_logical_sha256") == EXPECTED_FIXED235_LEDGER_LOGICAL_SHA256
        and value.get("fixed235_validation_sha256") == EXPECTED_FIXED235_VALIDATION_SHA256
        and value.get("fixed235_validation_logical_sha256") == EXPECTED_FIXED235_VALIDATION_LOGICAL_SHA256
        and value.get("code_sha256") == observed_code
        and value.get("target_free_prejoin") is True
        and value.get("automatic_stage_advance") is False
        and value.get("logical_sha256") == logical_sha256(value),
        "fresh-D1 OOF execution authority drift",
    )
    return {
        "execution_authority_sha256": sha256_file(path),
        "execution_authority_logical_sha256": str(value["logical_sha256"]),
    }


def fold_paths(root: Path, fold: int) -> dict[str, Path]:
    require(fold in FOLDS, "OOF fold must be in 0..4")
    base = root / f"results/routea_matched_three_arm_n2_current_runtime_d1_folds_v1/fold_{fold}"
    return {
        "result": base / "result.json",
        "checkpoint": base / "checkpoint.pt",
        "validation": base / "independent_validation.json",
    }


def validate_five_fold_authority(root: Path) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, torch.Tensor]]]:
    """Validate all five final checkpoints without reading a held-out target."""

    seals: dict[int, dict[str, str]] = {}
    states: dict[int, dict[str, torch.Tensor]] = {}
    for fold in FOLDS:
        paths = fold_paths(root, fold)
        for role, path in paths.items():
            require_regular(path, f"fold {fold} {role}")
        result = json.loads(paths["result"].read_text())
        validation = json.loads(paths["validation"].read_text())
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        require(
            result.get("status") == TRAIN_STATUS
            and result.get("version") == TRAIN_VERSION
            and result.get("fold") == fold
            and result.get("optimizer_steps") == 800
            and result.get("checkpoint_sha256") == sha256_file(paths["checkpoint"])
            and result.get("logical_sha256") == logical_sha256(result),
            f"fold {fold} result authority drift",
        )
        require(
            validation.get("status") == TRAIN_VALID_STATUS
            and validation.get("version") == TRAIN_VERSION
            and validation.get("fold") == fold
            and validation.get("parameter_count") == PARAMETER_COUNT
            and isinstance(validation.get("checks"), Mapping)
            and bool(validation["checks"])
            and all(value is True for value in validation["checks"].values())
            and validation.get("result_sha256") == sha256_file(paths["result"])
            and validation.get("checkpoint_sha256") == sha256_file(paths["checkpoint"])
            and validation.get("logical_sha256") == logical_sha256(validation)
            and validation.get("next_authorized_stage")
            == "N2_CURRENT_RUNTIME_D1_OOF_SCORING_AND_THREE_ARM_REMATERIALIZATION_CONTRACT",
            f"fold {fold} independent validation drift",
        )
        state = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
        receipt = checkpoint.get("receipt") if isinstance(checkpoint, Mapping) else None
        require(
            checkpoint.get("status") == TRAIN_STATUS
            and checkpoint.get("version") == TRAIN_VERSION
            and checkpoint.get("fold") == fold
            and isinstance(state, Mapping)
            and isinstance(receipt, Mapping)
            and receipt.get("optimizer_steps") == 800
            and receipt.get("parameter_count") == PARAMETER_COUNT
            and receipt.get("checkpoint_selection") == "final_step_800_only_no_heldout_selection"
            and receipt.get("final_state_dict_sha256") == state_dict_sha256(state)
            and result.get("final_state_dict_sha256") == state_dict_sha256(state)
            and validation.get("final_state_dict_sha256") == state_dict_sha256(state),
            f"fold {fold} checkpoint state drift",
        )
        states[fold] = {
            str(name): torch.as_tensor(value).detach().cpu().contiguous()
            for name, value in state.items()
        }
        seals[fold] = {
            "result_sha256": sha256_file(paths["result"]),
            "result_logical_sha256": str(result["logical_sha256"]),
            "checkpoint_sha256": sha256_file(paths["checkpoint"]),
            "checkpoint_state_dict_sha256": state_dict_sha256(states[fold]),
            "validation_sha256": sha256_file(paths["validation"]),
            "validation_logical_sha256": str(validation["logical_sha256"]),
        }
    require(len(seals) == len(states) == 5, "five-fold OOF checkpoint authority incomplete")
    return seals, states


def validate_raw_prejoin_authority(root: Path) -> dict[str, str]:
    path = root / "results/routea_matched_three_arm_n2_d1_raw_prejoin_v1/independent_aggregate_validation.json"
    require_regular(path, "target-free RAW aggregate validation")
    value = json.loads(path.read_text())
    require(
        value.get("status") == "ROUTEA_N2_D1_RAW_PREJOIN_AGGREGATE_VALIDATED"
        and value.get("query_count") == QUERY_COUNT
        and isinstance(value.get("checks"), Mapping)
        and bool(value["checks"])
        and all(item is True for item in value["checks"].values())
        and value.get("logical_sha256") == logical_sha256(value),
        "target-free RAW aggregate is not valid",
    )
    return {
        "raw_prejoin_aggregate_validation_sha256": sha256_file(path),
        "raw_prejoin_aggregate_validation_logical_sha256": str(value["logical_sha256"]),
        "raw_prejoin_score_manifest_sha256": str(value["score_manifest_sha256"]),
    }


def base_dependency_bindings(root: Path) -> dict[str, Any]:
    paths = contract_paths(root)
    for role, path in paths.items():
        require_regular(path, role)
    require(
        sha256_file(paths["two_gate_addendum_sha256"]) == EXPECTED_TWO_GATE_ADDENDUM_SHA256,
        "mandatory candidate-recall/ownership addendum drift",
    )
    require(
        sha256_file(paths["oof_contract_sha256"]) == EXPECTED_OOF_CONTRACT_SHA256,
        "fresh-D1 OOF scoring contract drift",
    )
    require(
        sha256_file(paths["train_contract_sha256"]) == EXPECTED_TRAIN_CONTRACT_SHA256,
        "fresh-D1 training contract drift",
    )
    fold_seals, _ = validate_five_fold_authority(root)
    return {
        **validate_token_authority(root),
        **validate_raw_prejoin_authority(root),
        **algorithm_dependency_hashes(root),
        **validate_execution_authority(root),
        **{role: sha256_file(path) for role, path in paths.items()},
        "fold_seals": {str(fold): seal for fold, seal in sorted(fold_seals.items())},
    }


def ranked_representative_rows(
    physical_scores: torch.Tensor, corrected_labels: Sequence[str]
) -> list[int]:
    """Target-free max-per-corrected-identity ranking with lower-row ties."""

    labels = tuple(map(str, corrected_labels))
    validate_corrected_identity_axis(labels)
    require(
        isinstance(physical_scores, torch.Tensor)
        and physical_scores.dtype == torch.float32
        and physical_scores.shape == (PHYSICAL_ROWS,)
        and bool(torch.isfinite(physical_scores).all()),
        "OOF scoring requires finite float32 [5413] scores",
    )
    order = torch.argsort(physical_scores, descending=True, stable=True).cpu().tolist()
    seen: set[str] = set()
    ranked: list[int] = []
    for row in order:
        identity = labels[row]
        if identity in seen:
            continue
        seen.add(identity)
        ranked.append(int(row))
    require(
        len(ranked) == len(seen) == CORRECTED_IDENTITIES
        and len(set(ranked)) == CORRECTED_IDENTITIES,
        "corrected full-gallery ranking population drift",
    )
    return ranked


def validate_oof_record(record: Mapping[str, Any]) -> None:
    required = {
        "query_id", "query_ordinal", "heldout_fold", "track",
        "grid_shape", "image_tokens_sha256", "template_tokens_sha256",
        "adapted_image_tokens", "adapted_image_tokens_sha256",
        "physical_row_scores", "physical_row_scores_sha256",
        "ranked_representative_physical_rows",
        "ranked_representative_physical_rows_sha256",
        "natural_c128_representative_physical_rows",
        "c128_boundary_score_gap", "c128_boundary_exact_tie",
        "oof_checkpoint_sha256",
    }
    require(set(record) == required, "fresh-D1 OOF prejoin record schema drift")
    forbidden = ("target", "supergroup", "correct", "winner", "challenger", "negative", "action", "outcome")
    require(
        not any(fragment in str(key).lower() for key in record for fragment in forbidden),
        "fresh-D1 OOF prejoin contains a protected query-semantic field",
    )
    image = record["adapted_image_tokens"]
    scores = record["physical_row_scores"]
    ranked = record["ranked_representative_physical_rows"]
    c128 = record["natural_c128_representative_physical_rows"]
    require(
        isinstance(image, torch.Tensor)
        and image.dtype == torch.float32
        and image.ndim == 2
        and image.shape[1] == 128
        and bool(torch.isfinite(image).all())
        and tensor_sha256(image) == record["adapted_image_tokens_sha256"],
        "fresh-D1 adapted image token tensor drift",
    )
    require(
        isinstance(scores, torch.Tensor)
        and scores.dtype == torch.float32
        and scores.shape == (PHYSICAL_ROWS,)
        and bool(torch.isfinite(scores).all())
        and tensor_sha256(scores) == record["physical_row_scores_sha256"],
        "fresh-D1 full-gallery score tensor drift",
    )
    require(
        isinstance(ranked, torch.Tensor)
        and ranked.dtype == torch.int32
        and ranked.shape == (CORRECTED_IDENTITIES,)
        and int(ranked.min()) >= 0
        and int(ranked.max()) < PHYSICAL_ROWS
        and torch.unique(ranked).numel() == CORRECTED_IDENTITIES
        and tensor_sha256(ranked) == record["ranked_representative_physical_rows_sha256"],
        "fresh-D1 complete corrected ranking drift",
    )
    require(
        isinstance(c128, torch.Tensor)
        and c128.dtype == torch.int32
        and c128.shape == (CANDIDATE_COUNT,)
        and torch.equal(c128, ranked[:CANDIDATE_COUNT]),
        "fresh-D1 natural C128 drift",
    )
    gap = record["c128_boundary_score_gap"]
    require(
        type(gap) is float
        and gap >= 0.0
        and gap == float(scores[int(ranked[127])]) - float(scores[int(ranked[128])])
        and record["c128_boundary_exact_tie"] is (gap == 0.0),
        "fresh-D1 C128 rank-128/129 boundary diagnostic drift",
    )


def r5_paths(root: Path) -> tuple[Path, dict[int, Path]]:
    base = root / "results/r5_independent_raw_pixel_local_index_a0_v1/postjoin_base_wrong_union_fullgallery_v3"
    return base / "result.json", {
        fold: base / f"fold_{fold}/rank_union_audit.json" for fold in FOLDS
    }


def validate_r5_fixed_authority(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Validate all-987 target roles and the separately frozen fixed235 ledger.

    The fixed denominator is consumed from ledger V2.  It is never rebuilt by
    filtering the all-987 rank audits in a new-model evaluator.
    """

    result_path, ledger_paths = r5_paths(root)
    require_regular(result_path, "frozen R5-V3 result")
    require(sha256_file(result_path) == EXPECTED_R5_RESULT_SHA256, "frozen R5-V3 result drift")
    result = json.loads(result_path.read_text())
    require(
        result.get("status") == "R5V3_BASE_WRONG_UNION_CANDIDATE_RECALL_STOP"
        and result.get("metrics", {}).get("base_wrong_count") == 235
        and result.get("metrics", {}).get("deep_tail_base_wrong_count") == 22
        and result.get("metrics", {}).get("base_wrong_D1_C128_target_coverage") == 190 / 235
        and result.get("metrics", {}).get("base_wrong_union_target_coverage") == 194 / 235,
        "frozen R5-V3 metric envelope drift",
    )
    records: list[dict[str, Any]] = []
    seals: dict[str, str] = {"r5_result_sha256": sha256_file(result_path)}
    for fold, path in ledger_paths.items():
        require_regular(path, f"frozen R5 fold {fold} ledger")
        require(sha256_file(path) == EXPECTED_R5_LEDGER_SHA256[fold], f"frozen R5 fold {fold} ledger drift")
        value = json.loads(path.read_text())
        require(
            value.get("status") == "R5V3_POSTJOIN_BASE_WRONG_UNION_FOLD_AUDIT_COMPLETE"
            and value.get("fold") == fold
            and value.get("record_count") == EXPECTED_FOLD_COUNTS[fold]
            and isinstance(value.get("records"), list),
            f"frozen R5 fold {fold} ledger envelope drift",
        )
        records.extend(value["records"])
        seals[f"r5_fold_{fold}_ledger_sha256"] = sha256_file(path)
    require(
        len(records) == len({str(row["query_id"]) for row in records}) == QUERY_COUNT
        and Counter(int(row["fold"]) for row in records) == Counter(EXPECTED_FOLD_COUNTS)
        and sum(not bool(row["D1_top1_correct"]) for row in records) == 235
        and sum(
            (not bool(row["D1_top1_correct"])) and int(row["D1_target_rank"]) > 512
            for row in records
        ) == 22,
        "frozen R5 987/235/22 population drift",
    )
    fixed_root = root / "results/routea_n2_r5v3_fixed235_evaluation_ledger_v2"
    fixed_path = fixed_root / "ledger.json"
    fixed_validation_path = fixed_root / "independent_validation.json"
    require_regular(fixed_path, "fixed235 ledger V2")
    require_regular(fixed_validation_path, "fixed235 ledger V2 validation")
    require(
        sha256_file(fixed_path) == EXPECTED_FIXED235_LEDGER_SHA256
        and sha256_file(fixed_validation_path) == EXPECTED_FIXED235_VALIDATION_SHA256,
        "fixed235 V2 physical seal drift",
    )
    fixed = json.loads(fixed_path.read_text())
    fixed_validation = json.loads(fixed_validation_path.read_text())
    fixed_records = fixed.get("records", []) if isinstance(fixed, Mapping) else []
    require(
        fixed.get("version") == "routea_n2_r5v3_fixed235_evaluation_ledger_v2"
        and fixed.get("status") == "ROUTEA_N2_R5V3_FIXED235_EVALUATION_LEDGER_READY"
        and fixed.get("logical_sha256") == EXPECTED_FIXED235_LEDGER_LOGICAL_SHA256
        and fixed.get("logical_sha256") == logical_sha256(fixed)
        and fixed.get("bindings", {}).get("oof_scoring_contract_sha256")
        == EXPECTED_OOF_CONTRACT_SHA256
        and fixed.get("bindings", {}).get("two_gate_addendum_sha256")
        == EXPECTED_TWO_GATE_ADDENDUM_SHA256
        and fixed.get("bindings", {}).get("r5_result_sha256")
        == EXPECTED_R5_RESULT_SHA256
        and len(fixed_records) == len({str(row["query_id"]) for row in fixed_records}) == 235
        and sum(bool(row["fixed_deep_tail22"]) for row in fixed_records) == 22,
        "fixed235 ledger V2 envelope or population drift",
    )
    require(
        fixed_validation.get("version") == "routea_n2_r5v3_fixed235_evaluation_ledger_v2"
        and fixed_validation.get("status")
        == "ROUTEA_N2_R5V3_FIXED235_EVALUATION_LEDGER_VALIDATED"
        and isinstance(fixed_validation.get("checks"), Mapping)
        and len(fixed_validation["checks"]) == 13
        and all(value is True for value in fixed_validation["checks"].values())
        and fixed_validation.get("ledger_sha256") == EXPECTED_FIXED235_LEDGER_SHA256
        and fixed_validation.get("logical_sha256") == EXPECTED_FIXED235_VALIDATION_LOGICAL_SHA256
        and fixed_validation.get("logical_sha256") == logical_sha256(fixed_validation)
        and fixed_validation.get("next_authorized_stage") is None,
        "fixed235 ledger V2 independent validation drift",
    )
    all_by_id = {str(row["query_id"]): row for row in records}
    for row in fixed_records:
        source = all_by_id.get(str(row["query_id"]))
        require(
            source is not None
            and row["fold"] == source["fold"]
            and row["global_query_index"] == source["global_query_index"]
            and row["track"] == source["track"]
            and row["target_identity"] == source["identity"]
            and row["target_supergroup"] == source["supergroup"]
            and row["old_D1_top1_correct"] is False
            and row["old_D1_target_rank"] == source["D1_target_rank"]
            and row["old_D1_target_in_C128"] == source["D1_target_in_C128"],
            "fixed235 ledger V2 does not match its frozen all-987 role source",
        )
    seals.update(
        {
            "fixed235_ledger_sha256": EXPECTED_FIXED235_LEDGER_SHA256,
            "fixed235_ledger_logical_sha256": EXPECTED_FIXED235_LEDGER_LOGICAL_SHA256,
            "fixed235_validation_sha256": EXPECTED_FIXED235_VALIDATION_SHA256,
            "fixed235_validation_logical_sha256": EXPECTED_FIXED235_VALIDATION_LOGICAL_SHA256,
        }
    )
    return result, records, [dict(row) for row in fixed_records], seals


__all__ = [
    "CANDIDATE_COUNT", "CORRECTED_IDENTITIES", "EXPECTED_FOLD_COUNTS",
    "EXPECTED_R5_LEDGER_SHA256", "EXPECTED_R5_RESULT_SHA256",
    "EXPECTED_TRACK_COUNTS", "FOLDS", "OOF_AGGREGATE_VALID_STATUS",
    "OOF_SHARD_STATUS", "OOF_SHARD_VALID_STATUS", "SHARD_COUNT", "VERSION",
    "base_dependency_bindings", "canonical_sha256", "expected_shard_interval",
    "fold_paths", "load_token_shard", "logical_sha256", "parameter_manifest",
    "ranked_representative_rows", "require", "require_regular", "r5_paths",
    "sha256_file", "state_dict_sha256", "tensor_sha256",
    "validate_five_fold_authority", "validate_oof_record",
    "validate_r5_fixed_authority", "validate_token_authority",
]
