#!/usr/bin/env python3
"""Independently validate one R1 main-arm/fold training artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
UPDATE_COUNT = 2_048
EXPECTED_PARAMETER_COUNT = 65_125
ORDER_NAMESPACE = "DINO_RCDE_R1_MAIN_QUERY_ORDER_V1_5"
DIRECTION_NAMESPACE = "DINO_RCDE_R1_PAIR_DIRECTION_V1_5"


class ValidationAbort(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_parts(namespace: str, *parts: object) -> str:
    return hashlib.sha256("\0".join((namespace, *(str(part) for part in parts))).encode()).hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationAbort(f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_bound(path_text: str) -> Path:
    path = (ROOT / path_text).resolve()
    if ROOT not in path.parents:
        raise ValidationAbort("binding escapes RC root")
    if {part.lower() for part in path.parts}.intersection({"c8", "opened", "sealed"}):
        raise ValidationAbort("protected path binding")
    return path


def fold_number(value: object) -> int:
    text = str(value).strip().upper()
    return int(text[1:] if text.startswith("F") else text)


def target_first(fold: int, query_id: str, rival_row: int) -> bool:
    return int(hash_parts(DIRECTION_NAMESPACE, fold, query_id, rival_row), 16) % 2 == 0


def ordered_pools(ledger: Mapping[str, Any], roles: Mapping[str, Any], fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    role_rows = roles.get("rows", roles.get("records", []))
    role_by_query = {str(row["query_id"]): row for row in role_rows}
    pools: tuple[list[dict[str, Any]], list[dict[str, Any]]] = ([], [])
    for episode in ledger.get("episodes", []):
        if fold_number(episode.get("heldout_fold")) != fold or episode.get("pair_loss_eligible") is not True:
            continue
        query_id = str(episode["query_id"])
        role = role_by_query.get(query_id)
        if role is None or fold_number(role["inner_fold"]) == fold:
            raise ValidationAbort("outer-train restriction failed")
        pools[0 if episode.get("full_gallery_RAW_correct") is True else 1].append(episode)
    key = lambda row: hash_parts(ORDER_NAMESPACE, fold, row["query_id"])
    pools[0].sort(key=key)
    pools[1].sort(key=key)
    if len(pools[0]) < 2 or len(pools[1]) < 2:
        raise ValidationAbort("balancing pool gate failed")
    return pools


def update_episodes(correct: Sequence[dict[str, Any]], wrong: Sequence[dict[str, Any]], update: int) -> list[dict[str, Any]]:
    rows = [
        correct[(2 * update) % len(correct)],
        correct[(2 * update + 1) % len(correct)],
        wrong[(2 * update) % len(wrong)],
        wrong[(2 * update + 1) % len(wrong)],
    ]
    if len({str(row["query_id"]) for row in rows}) != 4:
        raise ValidationAbort("duplicate query in expected schedule")
    return rows


def expected_schedule(ledger: Mapping[str, Any], roles: Mapping[str, Any], fold: int) -> tuple[str, int, int, int]:
    correct, wrong = ordered_pools(ledger, roles, fold)
    digest = hashlib.sha256()
    pair_count = 0
    for update in range(UPDATE_COUNT):
        query_rows = []
        for episode in update_episodes(correct, wrong, update):
            rivals = [int(row["physical_row"]) for row in episode["selected_negatives"]]
            pair_count += len(rivals)
            query_rows.append(
                {
                    "query_id": str(episode["query_id"]),
                    "query_ordinal": int(episode["query_ordinal"]),
                    "target_physical_row": int(episode["target_physical_row"]),
                    "rivals": rivals,
                    "target_first": [target_first(fold, str(episode["query_id"]), row) for row in rivals],
                }
            )
        digest.update(json_bytes({"update": update + 1, "queries": query_rows}) + b"\n")
    return digest.hexdigest(), pair_count, len(correct), len(wrong)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    authority_path = args.authority.resolve()
    output_dir = args.output_dir.resolve()
    validation_output = args.validation_output.resolve()
    if validation_output.exists():
        raise ValidationAbort("immutable validation output already exists")
    protocol = read_json(protocol_path)
    authority = read_json(authority_path)
    checks: dict[str, bool] = {}
    checks["protocol_schema"] = protocol.get("schema_version") == "rc_dino_rcde_r1_main_train_protocol_v1_5_20260813"
    checks["authority_digest"] = file_sha256(authority_path) == protocol.get("authority", {}).get("sha256")
    checks["authority_status"] = authority.get("status") == protocol.get("authority", {}).get("required_status")
    checks["training_authorized"] = authority.get("natural_training_authorized") is True
    checks["no_automatic_advance"] = protocol.get("automatic_stage_advance") is False
    for name, binding in protocol.get("bindings", {}).items():
        checks[f"binding_{name}"] = file_sha256(resolve_bound(str(binding["path"]))) == binding["sha256"]
    result_path = output_dir / "train_result.json"
    result = read_json(result_path)
    arm = str(result.get("arm"))
    fold = int(result.get("outer_fold", -1))
    checks["result_status"] = result.get("status") == "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAIN_COMPLETE"
    checks["scope"] = arm in protocol.get("training", {}).get("arms", []) and fold in protocol.get("training", {}).get("outer_folds", [])
    checks["protocol_result_binding"] = result.get("protocol_sha256") == file_sha256(protocol_path)
    checks["authority_result_binding"] = result.get("authority_sha256") == file_sha256(authority_path)
    checks["parameter_count"] = int(result.get("parameter_count", -1)) == EXPECTED_PARAMETER_COUNT
    training = result.get("training", {})
    checks["update_count"] = int(training.get("update_count", -1)) == UPDATE_COUNT
    checks["loss_trace_count"] = len(training.get("loss_trace", [])) == UPDATE_COUNT
    checks["finite_loss_trace"] = all(
        isinstance(row.get("mean_pair_loss"), (int, float))
        and float("-inf") < float(row["mean_pair_loss"]) < float("inf")
        and int(row.get("update", -1)) == index + 1
        for index, row in enumerate(training.get("loss_trace", []))
    )
    roles = read_json(resolve_bound(protocol["bindings"]["training_roles"]["path"]))
    ledger = read_json(resolve_bound(protocol["bindings"]["episode_ledger"]["path"]))
    schedule_sha, pair_count, correct_count, wrong_count = expected_schedule(ledger, roles, fold)
    checks["schedule"] = training.get("schedule_sha256") == schedule_sha
    checks["pair_count"] = int(training.get("pair_count", -1)) == pair_count
    checks["pool_counts"] = (
        int(training.get("correct_pool_count", -1)) == correct_count
        and int(training.get("wrong_pool_count", -1)) == wrong_count
    )
    checks["matched_tiles"] = training.get("query_tile_rows") == 8 and training.get("reference_tile_rows") == 16
    bag = result.get("bag_permutation", {})
    checks["arm_semantics"] = (
        arm == "RCDE_BAG" and bag.get("enabled") is True and bag.get("fixed_point_free_cyclic_shift") is True
    ) or (arm == "RCDE_CONTEXT" and bag.get("enabled") is False)
    protected = result.get("protected_access_counts", {})
    checks["protected_access_zero"] = bool(protected) and all(int(value) == 0 for value in protected.values())
    checkpoint_path = output_dir / str(result.get("checkpoint", ""))
    checks["checkpoint_digest"] = checkpoint_path.is_file() and file_sha256(checkpoint_path) == result.get("checkpoint_sha256")
    if checks["checkpoint_digest"]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", {})
        checks["checkpoint_metadata"] = (
            checkpoint.get("schema_version") == "rc_dino_rcde_r1_main_checkpoint_v1_5"
            and checkpoint.get("arm") == arm
            and int(checkpoint.get("outer_fold", -1)) == fold
            and int(checkpoint.get("update", -1)) == UPDATE_COUNT
            and checkpoint.get("protocol_sha256") == file_sha256(protocol_path)
            and checkpoint.get("authority_sha256") == file_sha256(authority_path)
        )
        checks["checkpoint_parameter_count"] = sum(int(value.numel()) for value in state.values()) == EXPECTED_PARAMETER_COUNT
        checks["state_changed"] = checkpoint.get("initial_state_sha256") != checkpoint.get("final_state_sha256")
        checks["state_result_binding"] = checkpoint.get("final_state_sha256") == result.get("final_state_sha256")
    else:
        checks["checkpoint_metadata"] = False
        checks["checkpoint_parameter_count"] = False
        checks["state_changed"] = False
        checks["state_result_binding"] = False
    passed = all(checks.values())
    validation = {
        "schema_version": "rc_dino_rcde_r1_main_train_validation_v1_5",
        "status": "DINO_RCDE_R1_MAIN_ARM_FOLD_VALIDATION_PASS" if passed else "DINO_RCDE_R1_MAIN_ARM_FOLD_VALIDATION_FAIL",
        "claim_level": "engineering_training_validation_only_not_scientific_GO_or_NO_GO",
        "arm": arm,
        "outer_fold": fold,
        "checks": checks,
        "result_sha256": file_sha256(result_path),
        "checkpoint_sha256": file_sha256(checkpoint_path) if checkpoint_path.is_file() else None,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_required_stage": "R1_MAIN_EIGHT_TASK_AGGREGATION" if passed else None,
    }
    atomic_json(validation_output, validation)
    if not passed:
        raise ValidationAbort("R1 main training validation failed")


if __name__ == "__main__":
    main()
