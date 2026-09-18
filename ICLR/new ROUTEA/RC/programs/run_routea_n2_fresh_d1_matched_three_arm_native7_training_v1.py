#!/usr/bin/env python3
"""Train the three fixed NATIVE7 arms without opening EVAL labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_SEVEN_PARAMETER_CROSSFIT_CONTRACT_V1_20260904.md"
CURRENT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1"
CURRENT_VALIDATION = CURRENT_ROOT / "validation.json"
PAIR_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_pair64_training_features_v1"
PAIR_VALIDATION = PAIR_ROOT / "independent_aggregate_validation.json"
J1_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1"
J1_TRAIN = J1_ROOT / "J1_TRAIN32_LABEL_LEDGER.json"
J1_PUBLIC_VALIDATION = J1_ROOT / "independent_validation.json"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_native7_training_v1"
HEADS = OUT_ROOT / "heads.pt"
RESULT = OUT_ROOT / "result.json"

VERSION = "routea_n2_fresh_d1_matched_three_arm_native7_training_v1_20260904"
STATUS = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_COMPLETE"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_INDEPENDENT_VALIDATION"
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
FEATURE_NAMES = (
    "fresh_d1_base_gap_over_c128_std",
    "symmetric_real_score",
    "symmetric_visibility_mass",
    "symmetric_score_over_mass",
    "symmetric_query_roll_response",
    "symmetric_reference_roll_response",
)
STEPS = 2000
SEED = 17
LEARNING_RATE = 0.03
WEIGHT_DECAY = 1e-3
HOLD_WEIGHT = 4.0
EXPECTED_HASHES = {
    CONTRACT: "1e129df85e71a14eefa814b477bf304771eeda631d14d4d68cc6c2a3b8342f79",
    CURRENT_VALIDATION: "24ea4c6e416fc9d5e0a57d22aaa957d5bb3f29e4f01022369a62dd7ff48b7859",
    PAIR_VALIDATION: "90e83f9598f3c9cfcf1a1d80282666be213a70abd98c4f8465ae01a3d6c3074e",
}


class TrainingError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"input absent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
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


def parameter_sha256(weight: torch.Tensor, bias: float) -> str:
    vector = torch.cat(
        [weight.detach().cpu().to(torch.float64).flatten(), torch.tensor([bias], dtype=torch.float64)]
    )
    return tensor_sha256(vector)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"JSON absent: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "JSON root is not object")
    return value


def finite_tensor(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    tensor = torch.as_tensor(value)
    require(tensor.shape == shape and bool(torch.isfinite(tensor).all()), f"{name} invalid")


def load_pair64() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation = read_json(PAIR_VALIDATION)
    require(
        validation.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUTS_VALIDATED"
        and validation.get("logical_sha256") == logical_sha256(validation)
        and all(validation.get("checks", {}).values())
        and validation.get("population")
        == {
            "query_count": 64,
            "fixed_pair_count": 64,
            "selected_endpoint_count": 128,
            "roma_pair_evaluation_count": 128,
            "switch_label_count": 27,
            "hold_label_count": 37,
        },
        "Pair64 aggregate authority drift",
    )
    seals = {int(item["shard"]): item for item in validation["shards"]}
    require(set(seals) == set(range(8)), "Pair64 shard seal set drift")
    records = []
    bindings = []
    for shard in range(8):
        directory = PAIR_ROOT / f"shard{shard:02d}"
        payload_path = directory / "payload.pt"
        receipt_path = directory / "receipt.json"
        validation_path = directory / "validation.json"
        seal = seals[shard]
        require(
            seal["payload_sha256"] == sha256_file(payload_path)
            and seal["receipt_sha256"] == sha256_file(receipt_path)
            and seal["validation_sha256"] == sha256_file(validation_path),
            f"Pair64 shard {shard} seal drift",
        )
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        require(
            payload.get("status")
            == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURE_SHARD_READY"
            and len(payload.get("records", [])) == 8,
            f"Pair64 shard {shard} payload drift",
        )
        for record in payload["records"]:
            projected = {
                "pair64_ordinal": int(record["pair64_ordinal"]),
                "query_id": str(record["query_id"]),
                "switch_label": bool(record["training_projection"]["switch_label"]),
                "features": {
                    arm: torch.as_tensor(record["native6_features"][arm])
                    .detach()
                    .cpu()
                    .to(torch.float64)
                    .contiguous()
                    for arm in ARMS
                },
            }
            for arm in ARMS:
                finite_tensor(projected["features"][arm], (1, 6), f"Pair64 {arm}")
            records.append(projected)
        bindings.append(
            {
                "shard": shard,
                "payload_sha256": seal["payload_sha256"],
                "validation_sha256": seal["validation_sha256"],
            }
        )
    require(
        [record["pair64_ordinal"] for record in records] == list(range(64))
        and len({record["query_id"] for record in records}) == 64
        and sum(record["switch_label"] for record in records) == 27,
        "Pair64 projected order/label population drift",
    )
    return records, bindings


def load_current_train() -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    validation = read_json(CURRENT_VALIDATION)
    require(
        validation.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATED"
        and validation.get("logical_sha256") == logical_sha256(validation)
        and all(validation.get("checks", {}).values())
        and validation.get("current64_query_count") == 64,
        "current64 feature authority drift",
    )
    seals = {int(item["shard"]): item for item in validation["shards"]}
    train = []
    bindings = []
    eval_skipped = 0
    for shard in range(8):
        directory = CURRENT_ROOT / f"shard{shard:02d}"
        payload_path = directory / "payload.pt"
        receipt_path = directory / "receipt.json"
        validation_path = directory / "validation.json"
        seal = seals[shard]
        require(
            seal["payload_sha256"] == sha256_file(payload_path)
            and seal["receipt_sha256"] == sha256_file(receipt_path)
            and seal["validation_sha256"] == sha256_file(validation_path),
            f"current64 shard {shard} seal drift",
        )
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        for record in payload["records"]:
            role = str(record["role_membership"])
            require(role in {"TRAIN", "EVAL"}, "current64 role drift")
            if role != "TRAIN":
                eval_skipped += 1
                continue
            projected = {
                "query_id": str(record["query_id"]),
                "execution_ordinal": int(record["execution_ordinal"]),
                "base_winner_position": int(record["base_winner_position"]),
                "challenger_positions": list(map(int, record["challenger_positions"])),
                "candidate_axis_sha256": str(record["candidate_axis_sha256"]),
                "feature_sha256": record["feature_sha256"],
                "features": {
                    arm: torch.as_tensor(record["real_native_features"][arm])
                    .detach()
                    .cpu()
                    .to(torch.float64)
                    .contiguous()
                    for arm in ARMS
                },
            }
            require(
                projected["base_winner_position"] == 0
                and projected["challenger_positions"] == list(range(1, 128)),
                "current64 base/challenger axis drift",
            )
            for arm in ARMS:
                finite_tensor(projected["features"][arm], (127, 6), f"TRAIN32 {arm}")
            train.append(projected)
        bindings.append(
            {
                "shard": shard,
                "payload_sha256": seal["payload_sha256"],
                "validation_sha256": seal["validation_sha256"],
            }
        )
    train.sort(key=lambda record: record["execution_ordinal"])
    require(
        len(train) == len({record["query_id"] for record in train}) == 32
        and eval_skipped == 32,
        "TRAIN32/EVAL semantic projection count drift",
    )
    return train, bindings, eval_skipped


def load_train_labels(train: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    public = read_json(J1_PUBLIC_VALIDATION)
    labels = read_json(J1_TRAIN)
    require(
        public.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_POSTSEAL_JOIN_VALIDATED"
        and public.get("logical_sha256") == logical_sha256(public)
        and all(public.get("checks", {}).values())
        and public.get("train32_ledger_sha256") == sha256_file(J1_TRAIN)
        and public.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING",
        "J1 public validation drift",
    )
    require(
        labels.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_J1_TRAIN32_LABEL_LEDGER_READY"
        and labels.get("logical_sha256") == logical_sha256(labels)
        and labels.get("role") == "TRAIN"
        and labels.get("population")
        == {
            "query_count": 32,
            "identity_count": 12,
            "supergroup_count": 12,
            "target_winner_count": 27,
            "target_nonwinner_count": 5,
            "target_absent_count": 0,
        }
        and labels.get("access_policy", {}).get("readable_by")
        == ["T0_NATIVE7_TRAINING"],
        "J1 TRAIN32 ledger drift",
    )
    by_query = {str(record["query_id"]): record for record in labels["records"]}
    require(len(by_query) == 32, "J1 TRAIN32 query population drift")
    joined = []
    winner_count = nonwinner_count = 0
    for record in train:
        label = by_query.get(record["query_id"])
        require(label is not None, "TRAIN32 feature/label join failed")
        target_position = label["target_candidate_position"]
        require(
            target_position is not None
            and int(target_position) in range(128)
            and label["targetfree_record_binding"]["candidate_axis_sha256"]
            == record["candidate_axis_sha256"]
            and label["targetfree_record_binding"]["feature_sha256"]
            == record["feature_sha256"],
            "TRAIN32 targetfree binding/position drift",
        )
        winner_count += int(int(target_position) == 0)
        nonwinner_count += int(int(target_position) != 0)
        joined.append({**record, "target_position": int(target_position)})
    require(winner_count == 27 and nonwinner_count == 5, "TRAIN32 state counts drift")
    return joined, {
        "train_ledger_sha256": sha256_file(J1_TRAIN),
        "train_ledger_logical_sha256": labels["logical_sha256"],
        "public_validation_sha256": sha256_file(J1_PUBLIC_VALIDATION),
        "public_validation_logical_sha256": public["logical_sha256"],
    }


def feature_ledger(pair: list[dict[str, Any]], train: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for arm in ARMS:
        matrix = torch.cat(
            [record["features"][arm] for record in pair]
            + [record["features"][arm] for record in train],
            dim=0,
        )
        finite_tensor(matrix, (4128, 6), f"{arm} design matrix")
        zero = [index for index in range(6) if int(torch.count_nonzero(matrix[:, index])) == 0]
        duplicate = [
            [left, right]
            for left in range(6)
            for right in range(left + 1, 6)
            if torch.equal(matrix[:, left], matrix[:, right])
        ]
        output[arm] = {
            "matrix_shape": [4128, 6],
            "feature_rank": int(torch.linalg.matrix_rank(matrix)),
            "design_with_bias_rank": int(
                torch.linalg.matrix_rank(
                    torch.cat([matrix, torch.ones((4128, 1), dtype=torch.float64)], dim=1)
                )
            ),
            "exact_zero_column_indices": zero,
            "exact_duplicate_column_pairs": duplicate,
            "all_finite": True,
        }
    return output


def train_arm(
    pair: list[dict[str, Any]],
    train: list[dict[str, Any]],
    arm: str,
    *,
    steps: int = STEPS,
) -> tuple[dict[str, Any], torch.Tensor, float]:
    pair_x = torch.cat([record["features"][arm] for record in pair], dim=0)
    pair_y = torch.tensor(
        [float(record["switch_label"]) for record in pair], dtype=torch.float64
    )
    pair_weight = torch.where(
        pair_y.eq(0), torch.full_like(pair_y, HOLD_WEIGHT), torch.ones_like(pair_y)
    )
    finite_tensor(pair_x, (64, 6), f"{arm} pair features")
    torch.manual_seed(SEED)
    head = nn.Linear(6, 1, dtype=torch.float64)
    head.weight.data.zero_()
    head.bias.data.zero_()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    final = None
    finite_loss = finite_gradient = finite_parameter = 0
    for _ in range(steps):
        optimizer.zero_grad()
        pair_logits = head(pair_x).squeeze(1)
        pair_loss = (
            F.binary_cross_entropy_with_logits(pair_logits, pair_y, reduction="none")
            * pair_weight
        ).mean()
        query_losses = []
        for record in train:
            logits = head(record["features"][arm]).squeeze(1)
            target = record["target_position"]
            if target == 0:
                query_losses.append(HOLD_WEIGHT * F.softplus(logits.max()))
            else:
                target_index = record["challenger_positions"].index(target)
                mask = torch.ones(127, dtype=torch.bool)
                mask[target_index] = False
                query_losses.append(
                    F.softplus(-logits[target_index])
                    + HOLD_WEIGHT * F.softplus(logits[mask].max())
                )
        full_loss = torch.stack(query_losses).mean()
        loss = pair_loss + full_loss
        require(bool(torch.isfinite(torch.stack([loss, pair_loss, full_loss])).all()), "nonfinite loss")
        finite_loss += 1
        loss.backward()
        require(
            all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in head.parameters()
            ),
            "nonfinite gradient",
        )
        finite_gradient += 1
        optimizer.step()
        require(all(bool(torch.isfinite(parameter).all()) for parameter in head.parameters()), "nonfinite parameter")
        finite_parameter += 1
        final = (
            float(loss.detach()),
            float(pair_loss.detach()),
            float(full_loss.detach()),
        )
    require(final is not None, "training performed zero updates")
    weight = head.weight.detach().cpu().flatten().contiguous()
    bias = float(head.bias.detach().cpu())
    return (
        {
            "weight": weight.tolist(),
            "bias": bias,
            "parameter_count": 7,
            "parameter_sha256": parameter_sha256(weight, bias),
            "loss_total": final[0],
            "loss_pair": final[1],
            "loss_fullnegative": final[2],
            "finite_loss_step_count": finite_loss,
            "finite_gradient_step_count": finite_gradient,
            "finite_parameter_step_count": finite_parameter,
            "optimizer_step_count": steps,
        },
        weight,
        bias,
    )


def synthetic_self_test() -> None:
    generator = torch.Generator().manual_seed(SEED)
    pair = [
        {
            "features": {arm: torch.randn(1, 6, generator=generator, dtype=torch.float64) for arm in ARMS},
            "switch_label": bool(index % 2),
        }
        for index in range(64)
    ]
    train = [
        {
            "features": {arm: torch.randn(127, 6, generator=generator, dtype=torch.float64) for arm in ARMS},
            "target_position": 0 if index else 3,
            "challenger_positions": list(range(1, 128)),
        }
        for index in range(2)
    ]
    first = train_arm(pair, train, "C_PAIRED", steps=3)[0]
    second = train_arm(pair, train, "C_PAIRED", steps=3)[0]
    require(first == second and first["parameter_count"] == 7, "synthetic deterministic replay failed")
    print(json.dumps({"status": "NATIVE7_T0_SYNTHETIC_SELF_TEST_PASS"}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-self-test", action="store_true")
    args = parser.parse_args()
    if args.synthetic_self_test:
        synthetic_self_test()
        return
    require(not OUT_ROOT.exists(), f"immutable T0 output exists: {OUT_ROOT}")
    for path, expected in EXPECTED_HASHES.items():
        require(sha256_file(path) == expected, f"fixed input hash drift: {path}")
    pair, pair_shards = load_pair64()
    current, current_shards, eval_skipped = load_current_train()
    train, j1_binding = load_train_labels(current)
    ledger = feature_ledger(pair, train)
    initial_weight = torch.zeros(6, dtype=torch.float64)
    initial_parameter_sha = parameter_sha256(initial_weight, 0.0)
    heads = {}
    tensor_heads = {}
    for arm in ARMS:
        head, weight, bias = train_arm(pair, train, arm)
        heads[arm] = head
        tensor_heads[arm] = {"weight": weight, "bias": bias}
    bindings = {
        "contract_sha256": sha256_file(CONTRACT),
        "current64_validation_sha256": sha256_file(CURRENT_VALIDATION),
        "current64_validation_logical_sha256": read_json(CURRENT_VALIDATION)["logical_sha256"],
        "current64_train_shards": current_shards,
        "pair64_validation_sha256": sha256_file(PAIR_VALIDATION),
        "pair64_validation_logical_sha256": read_json(PAIR_VALIDATION)["logical_sha256"],
        "pair64_shards": pair_shards,
        **j1_binding,
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
    }
    access = {
        "pair64_training_record_count": 64,
        "current_train_feature_record_count": 32,
        "current_train_fullnegative_row_count": 4064,
        "current_eval_record_skipped_before_feature_projection_count": eval_skipped,
        "current_eval_feature_semantic_consumption_count": 0,
        "current_eval_statistic_consumption_count": 0,
        "current_eval_label_file_open_count": 0,
        "train_label_ledger_open_count": 1,
        "j1_public_validation_open_count": 1,
        "train_target_identity_semantic_consumption_count": 0,
        "train_target_supergroup_semantic_consumption_count": 0,
        "train_target_position_semantic_consumption_count": 32,
        "model_update_count": 6000,
        "external_read_count": 0,
        "sealed_read_count": 0,
    }
    heads_payload = {
        "version": VERSION,
        "status": STATUS,
        "feature_names": list(FEATURE_NAMES),
        "arms": list(ARMS),
        "heads": tensor_heads,
        "recipe": {
            "seed": SEED,
            "dtype": "torch.float64",
            "zero_initialization": True,
            "initial_parameter_sha256": initial_parameter_sha,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "hold_weight": HOLD_WEIGHT,
            "updates_per_arm": STEPS,
            "switch_threshold": 0.0,
        },
        "bindings": bindings,
        "access": access,
    }
    result = {
        "version": VERSION,
        "status": STATUS,
        "claim_level": "INTERNAL_TRAIN_ONLY_NATIVE7_A_B_C_NO_EVAL_LABEL_OR_ACTION",
        "population": {
            "pair64_query_count": 64,
            "pair64_switch_count": 27,
            "pair64_hold_count": 37,
            "current_train_query_count": 32,
            "current_train_target_winner_count": 27,
            "current_train_target_nonwinner_count": 5,
            "current_train_target_absent_count": 0,
            "design_rows_per_arm": 4128,
        },
        "recipe": heads_payload["recipe"],
        "feature_degeneracy": ledger,
        "heads": heads,
        "bindings": bindings,
        "access": access,
        "heads_payload_sha256": "",
        "optimizer_constructed": True,
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": NEXT,
        "logical_sha256": "",
    }
    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".n2-native7-t0-", dir=OUT_ROOT.parent))
    try:
        torch.save(heads_payload, staging / "heads.pt")
        result["heads_payload_sha256"] = sha256_file(staging / "heads.pt")
        result["logical_sha256"] = logical_sha256(result)
        (staging / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        os.rename(staging, OUT_ROOT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"status": STATUS, "updates": 6000}, sort_keys=True))


if __name__ == "__main__":
    main()
