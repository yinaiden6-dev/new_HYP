#!/usr/bin/env python3
"""Independently retrain the three NATIVE7 arms without EVAL labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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
TRAINER = ROOT / "programs/run_routea_n2_fresh_d1_matched_three_arm_native7_training_v1.py"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_native7_training_v1"
HEADS = OUT_ROOT / "heads.pt"
RESULT = OUT_ROOT / "result.json"
OUT = OUT_ROOT / "independent_validation.json"

VERSION = "routea_n2_fresh_d1_matched_three_arm_native7_training_v1_20260904"
READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_COMPLETE"
VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_VALIDATED"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTION_SEAL"
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
LR = 0.03
WD = 1e-3
HOLD_WEIGHT = 4.0
EXPECTED = {
    CONTRACT: "1e129df85e71a14eefa814b477bf304771eeda631d14d4d68cc6c2a3b8342f79",
    CURRENT_VALIDATION: "24ea4c6e416fc9d5e0a57d22aaa957d5bb3f29e4f01022369a62dd7ff48b7859",
    PAIR_VALIDATION: "90e83f9598f3c9cfcf1a1d80282666be213a70abd98c4f8465ae01a3d6c3074e",
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"input absent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canon(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("logical_sha256", None)
    return canon(payload)


def thash(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_hash(weight: torch.Tensor, bias: float) -> str:
    return thash(
        torch.cat(
            [weight.detach().cpu().to(torch.float64).flatten(), torch.tensor([bias], dtype=torch.float64)]
        )
    )


def read(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"JSON absent: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "JSON root invalid")
    return value


def finite(value: torch.Tensor, shape: tuple[int, ...], name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().to(torch.float64).contiguous()
    require(tensor.shape == shape and bool(torch.isfinite(tensor).all()), f"{name} invalid")
    return tensor


def load_pair() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    authority = read(PAIR_VALIDATION)
    require(
        authority.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUTS_VALIDATED"
        and authority.get("logical_sha256") == logical(authority)
        and all(authority.get("checks", {}).values()),
        "Pair64 aggregate drift",
    )
    seals = {int(item["shard"]): item for item in authority["shards"]}
    rows = []
    bound = []
    for shard in range(8):
        directory = PAIR_ROOT / f"shard{shard:02d}"
        pp, rp, vp = directory / "payload.pt", directory / "receipt.json", directory / "validation.json"
        seal = seals[shard]
        require(
            seal["payload_sha256"] == sha(pp)
            and seal["receipt_sha256"] == sha(rp)
            and seal["validation_sha256"] == sha(vp),
            f"Pair64 shard {shard} seal drift",
        )
        payload = torch.load(pp, map_location="cpu", weights_only=False, mmap=True)
        for row in payload["records"]:
            projected = {
                "ordinal": int(row["pair64_ordinal"]),
                "query_id": str(row["query_id"]),
                "label": bool(row["training_projection"]["switch_label"]),
                "features": {
                    arm: finite(row["native6_features"][arm], (1, 6), f"Pair64 {arm}")
                    for arm in ARMS
                },
            }
            rows.append(projected)
        bound.append({"shard": shard, "payload_sha256": seal["payload_sha256"], "validation_sha256": seal["validation_sha256"]})
    require(
        [row["ordinal"] for row in rows] == list(range(64))
        and len({row["query_id"] for row in rows}) == 64
        and sum(row["label"] for row in rows) == 27,
        "Pair64 order/labels drift",
    )
    return rows, bound


def load_current_train() -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    authority = read(CURRENT_VALIDATION)
    require(
        authority.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATED"
        and authority.get("logical_sha256") == logical(authority)
        and all(authority.get("checks", {}).values()),
        "current64 aggregate drift",
    )
    seals = {int(item["shard"]): item for item in authority["shards"]}
    rows = []
    bound = []
    skipped = 0
    for shard in range(8):
        directory = CURRENT_ROOT / f"shard{shard:02d}"
        pp, rp, vp = directory / "payload.pt", directory / "receipt.json", directory / "validation.json"
        seal = seals[shard]
        require(
            seal["payload_sha256"] == sha(pp)
            and seal["receipt_sha256"] == sha(rp)
            and seal["validation_sha256"] == sha(vp),
            f"current shard {shard} seal drift",
        )
        payload = torch.load(pp, map_location="cpu", weights_only=False, mmap=True)
        for row in payload["records"]:
            role = str(row["role_membership"])
            require(role in {"TRAIN", "EVAL"}, "role drift")
            if role != "TRAIN":
                skipped += 1
                continue
            rows.append(
                {
                    "query_id": str(row["query_id"]),
                    "execution_ordinal": int(row["execution_ordinal"]),
                    "winner": int(row["base_winner_position"]),
                    "challengers": list(map(int, row["challenger_positions"])),
                    "candidate_axis_sha256": str(row["candidate_axis_sha256"]),
                    "feature_sha256": row["feature_sha256"],
                    "features": {
                        arm: finite(row["real_native_features"][arm], (127, 6), f"TRAIN {arm}")
                        for arm in ARMS
                    },
                }
            )
        bound.append({"shard": shard, "payload_sha256": seal["payload_sha256"], "validation_sha256": seal["validation_sha256"]})
    rows.sort(key=lambda row: row["execution_ordinal"])
    require(len(rows) == 32 and skipped == 32, "TRAIN/EVAL projection drift")
    return rows, bound, skipped


def join_train_labels(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    public = read(J1_PUBLIC_VALIDATION)
    ledger = read(J1_TRAIN)
    require(
        public.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_POSTSEAL_JOIN_VALIDATED"
        and public.get("logical_sha256") == logical(public)
        and all(public.get("checks", {}).values())
        and public.get("train32_ledger_sha256") == sha(J1_TRAIN)
        and public.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING",
        "J1 public validation drift",
    )
    require(
        ledger.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_J1_TRAIN32_LABEL_LEDGER_READY"
        and ledger.get("logical_sha256") == logical(ledger)
        and ledger.get("population", {}).get("target_winner_count") == 27
        and ledger.get("population", {}).get("target_nonwinner_count") == 5
        and ledger.get("population", {}).get("target_absent_count") == 0,
        "J1 TRAIN ledger drift",
    )
    labels = {str(item["query_id"]): item for item in ledger["records"]}
    joined = []
    for row in rows:
        item = labels.get(row["query_id"])
        require(item is not None, "TRAIN label join missing")
        position = item["target_candidate_position"]
        require(
            position is not None
            and item["targetfree_record_binding"]["candidate_axis_sha256"]
            == row["candidate_axis_sha256"]
            and item["targetfree_record_binding"]["feature_sha256"]
            == row["feature_sha256"],
            "TRAIN label binding drift",
        )
        joined.append({**row, "target_position": int(position)})
    require(
        sum(row["target_position"] == 0 for row in joined) == 27
        and sum(row["target_position"] != 0 for row in joined) == 5,
        "TRAIN target state counts drift",
    )
    return joined, {
        "train_ledger_sha256": sha(J1_TRAIN),
        "train_ledger_logical_sha256": ledger["logical_sha256"],
        "public_validation_sha256": sha(J1_PUBLIC_VALIDATION),
        "public_validation_logical_sha256": public["logical_sha256"],
    }


def train_one(pair: list[dict[str, Any]], current: list[dict[str, Any]], arm: str, steps: int = STEPS):
    px = torch.cat([row["features"][arm] for row in pair], dim=0)
    py = torch.tensor([float(row["label"]) for row in pair], dtype=torch.float64)
    weights = torch.where(py.eq(0), torch.full_like(py, HOLD_WEIGHT), torch.ones_like(py))
    torch.manual_seed(SEED)
    head = nn.Linear(6, 1, dtype=torch.float64)
    head.weight.data.zero_()
    head.bias.data.zero_()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    final = None
    for _ in range(steps):
        optimizer.zero_grad()
        pair_loss = (
            F.binary_cross_entropy_with_logits(head(px).squeeze(1), py, reduction="none")
            * weights
        ).mean()
        query_losses = []
        for row in current:
            logits = head(row["features"][arm]).squeeze(1)
            target = row["target_position"]
            if target == row["winner"]:
                query_losses.append(HOLD_WEIGHT * F.softplus(logits.max()))
            else:
                index = row["challengers"].index(target)
                mask = torch.ones(127, dtype=torch.bool)
                mask[index] = False
                query_losses.append(
                    F.softplus(-logits[index])
                    + HOLD_WEIGHT * F.softplus(logits[mask].max())
                )
        full_loss = torch.stack(query_losses).mean()
        loss = pair_loss + full_loss
        require(bool(torch.isfinite(loss)), "nonfinite replay loss")
        loss.backward()
        require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in head.parameters()), "nonfinite replay gradient")
        optimizer.step()
        require(all(bool(torch.isfinite(parameter).all()) for parameter in head.parameters()), "nonfinite replay parameter")
        final = (
            float(loss.detach()),
            float(pair_loss.detach()),
            float(full_loss.detach()),
        )
    weight = head.weight.detach().cpu().flatten().contiguous()
    bias = float(head.bias.detach().cpu())
    return {
        "weight": weight.tolist(),
        "bias": bias,
        "parameter_count": 7,
        "parameter_sha256": parameter_hash(weight, bias),
        "loss_total": final[0],
        "loss_pair": final[1],
        "loss_fullnegative": final[2],
        "finite_loss_step_count": steps,
        "finite_gradient_step_count": steps,
        "finite_parameter_step_count": steps,
        "optimizer_step_count": steps,
    }, weight, bias


def ledger(pair: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for arm in ARMS:
        matrix = torch.cat(
            [row["features"][arm] for row in pair]
            + [row["features"][arm] for row in current]
        )
        output[arm] = {
            "matrix_shape": [4128, 6],
            "feature_rank": int(torch.linalg.matrix_rank(matrix)),
            "design_with_bias_rank": int(
                torch.linalg.matrix_rank(
                    torch.cat([matrix, torch.ones((4128, 1), dtype=torch.float64)], dim=1)
                )
            ),
            "exact_zero_column_indices": [i for i in range(6) if int(torch.count_nonzero(matrix[:, i])) == 0],
            "exact_duplicate_column_pairs": [
                [i, j]
                for i in range(6)
                for j in range(i + 1, 6)
                if torch.equal(matrix[:, i], matrix[:, j])
            ],
            "all_finite": bool(torch.isfinite(matrix).all()),
        }
    return output


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), "immutable validation exists")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def synthetic_self_test() -> None:
    generator = torch.Generator().manual_seed(SEED)
    pair = [{"features": {arm: torch.randn(1, 6, generator=generator, dtype=torch.float64) for arm in ARMS}, "label": bool(i % 2)} for i in range(64)]
    current = [{"features": {arm: torch.randn(127, 6, generator=generator, dtype=torch.float64) for arm in ARMS}, "target_position": 0 if i else 2, "winner": 0, "challengers": list(range(1, 128))} for i in range(2)]
    require(train_one(pair, current, "C_PAIRED", 3)[0] == train_one(pair, current, "C_PAIRED", 3)[0], "independent synthetic replay failed")
    print(json.dumps({"status": "NATIVE7_T0_VALIDATOR_SYNTHETIC_SELF_TEST_PASS"}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-self-test", action="store_true")
    args = parser.parse_args()
    if args.synthetic_self_test:
        synthetic_self_test()
        return
    require(not OUT.exists(), "immutable T0 validation exists")
    for path, expected in EXPECTED.items():
        require(sha(path) == expected, f"fixed input hash drift: {path}")
    result = read(RESULT)
    heads_payload = torch.load(HEADS, map_location="cpu", weights_only=False)
    pair, pair_shards = load_pair()
    current, current_shards, skipped = load_current_train()
    current, j1_binding = join_train_labels(current)
    expected_heads = {}
    tensor_heads = {}
    for arm in ARMS:
        head, weight, bias = train_one(pair, current, arm)
        expected_heads[arm] = head
        tensor_heads[arm] = {"weight": weight, "bias": bias}
    expected_bindings = {
        "contract_sha256": sha(CONTRACT),
        "current64_validation_sha256": sha(CURRENT_VALIDATION),
        "current64_validation_logical_sha256": read(CURRENT_VALIDATION)["logical_sha256"],
        "current64_train_shards": current_shards,
        "pair64_validation_sha256": sha(PAIR_VALIDATION),
        "pair64_validation_logical_sha256": read(PAIR_VALIDATION)["logical_sha256"],
        "pair64_shards": pair_shards,
        **j1_binding,
        "trainer_sha256": sha(TRAINER),
    }
    expected_access = {
        "pair64_training_record_count": 64,
        "current_train_feature_record_count": 32,
        "current_train_fullnegative_row_count": 4064,
        "current_eval_record_skipped_before_feature_projection_count": skipped,
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
    recipe = {
        "seed": SEED,
        "dtype": "torch.float64",
        "zero_initialization": True,
        "initial_parameter_sha256": parameter_hash(torch.zeros(6, dtype=torch.float64), 0.0),
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WD,
        "hold_weight": HOLD_WEIGHT,
        "updates_per_arm": STEPS,
        "switch_threshold": 0.0,
    }
    checks = {
        "trainer_not_imported": True,
        "contract_and_input_hashes": result.get("bindings") == expected_bindings,
        "only_native7_three_arms": set(result.get("heads", {})) == set(ARMS)
        and result.get("recipe") == recipe,
        "independent_three_head_6000_update_replay": result.get("heads")
        == expected_heads,
        "heads_payload_exact": heads_payload.get("version") == VERSION
        and heads_payload.get("status") == READY
        and heads_payload.get("feature_names") == list(FEATURE_NAMES)
        and heads_payload.get("arms") == list(ARMS)
        and heads_payload.get("recipe") == recipe
        and heads_payload.get("bindings") == expected_bindings
        and heads_payload.get("access") == expected_access
        and all(
            torch.equal(heads_payload["heads"][arm]["weight"], tensor_heads[arm]["weight"])
            and heads_payload["heads"][arm]["bias"] == tensor_heads[arm]["bias"]
            for arm in ARMS
        ),
        "feature_degeneracy_replay": result.get("feature_degeneracy")
        == ledger(pair, current),
        "population_exact": result.get("population")
        == {
            "pair64_query_count": 64,
            "pair64_switch_count": 27,
            "pair64_hold_count": 37,
            "current_train_query_count": 32,
            "current_train_target_winner_count": 27,
            "current_train_target_nonwinner_count": 5,
            "current_train_target_absent_count": 0,
            "design_rows_per_arm": 4128,
        },
        "eval_label_and_feature_semantic_consumption_zero": result.get("access")
        == expected_access,
        "result_envelope": result.get("version") == VERSION
        and result.get("status") == READY
        and result.get("heads_payload_sha256") == sha(HEADS)
        and result.get("logical_sha256") == logical(result)
        and result.get("optimizer_constructed") is True
        and result.get("scientific_GO_or_NO_GO") is None
        and result.get("ownership_GO_or_NO_GO") is None
        and result.get("automatic_stage_advance") is False
        and result.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_INDEPENDENT_VALIDATION",
    }
    require(all(checks.values()), "T0 independent replay failed")
    value = {
        "version": VERSION,
        "status": VALIDATED,
        "claim_level": "INDEPENDENT_TRAIN_ONLY_NATIVE7_THREE_HEAD_REPLAY_NO_EVAL_LABEL_OR_ACTION",
        "checks": checks,
        "model_update_replay_count": 6000,
        "result_sha256": sha(RESULT),
        "result_logical_sha256": result["logical_sha256"],
        "heads_payload_sha256": sha(HEADS),
        "trainer_sha256": sha(TRAINER),
        "validator_sha256": sha(Path(__file__).resolve()),
        "access": expected_access,
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": NEXT,
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical(value)
    atomic(OUT, value)
    print(json.dumps({"status": VALIDATED, "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
