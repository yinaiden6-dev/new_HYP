#!/usr/bin/env python3
"""Independent validation and update-775 continuation replay for one N2 D1 fold."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
ROUTEA = ROOT.parent
sys.path[:0] = [str(ROOT / "src"), str(ROUTEA / "route_a_core")]

from rc_aslo_xf.n2_current_runtime_d1_training_v1 import (  # noqa: E402
    CORRECTED_ELIGIBLE_IDENTITY_COUNTS,
    EXPECTED_LABEL_SOURCE_HASHES,
    PARAMETER_COUNT,
    PHYSICAL_ROWS,
    NEGATIVE_COUNT,
    PREJOIN_AGGREGATE_VALID_STATUS,
    SEED,
    SHARD_COUNT,
    STEPS,
    TRAIN_STATUS,
    TRAIN_VALID_STATUS,
    VERSION,
    algorithm_dependency_hashes,
    label_join_dependency_hashes,
    raw_prejoin_dependency_hashes,
    atomic_json,
    canonical_sha256,
    corrected_mask_audit,
    fold_seed,
    load_gallery,
    load_token_shard,
    logical_sha256,
    parameter_manifest,
    require,
    sha256_file,
    state_dict_sha256,
    tensor_sha256,
    validate_prejoin_record,
    validate_token_authority,
)
from route_a import o1_c6direct_m1_d1 as d1  # noqa: E402
from route_a.o1_c6direct_m1_runtime import pad_reference_tokens  # noqa: E402


CONTRACT = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_N2_CURRENT_RUNTIME_FOLD_LOCAL_D1_TRAINING_V1_20260903.md"
RUNTIME = ROOT / "src/rc_aslo_xf/n2_current_runtime_d1_training_v1.py"
CORE = ROUTEA / "route_a_core/route_a/o1_c6direct_m1_d1.py"
TRAINER = ROOT / "programs/run_routea_matched_three_arm_n2_current_runtime_d1_fold_v1.py"
LABEL_ROOT = ROOT / "results/routea_matched_three_arm_n2_d1_label_join_authority_v1"
LABEL_VALIDATION = LABEL_ROOT / "independent_validation.json"
LABEL_PRODUCER = ROOT / "programs/materialize_routea_matched_three_arm_n2_d1_label_join_authority_v1.py"
LABEL_VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_d1_label_join_authority_v1.py"
PREJOIN_ROOT = ROOT / "results/routea_matched_three_arm_n2_d1_raw_prejoin_v1"
PREJOIN_VALIDATION = PREJOIN_ROOT / "independent_aggregate_validation.json"
PREJOIN_PRODUCER = ROOT / "programs/materialize_routea_matched_three_arm_n2_d1_raw_prejoin_shard_v1.py"
PREJOIN_SHARD_VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_d1_raw_prejoin_shard_v1.py"
PREJOIN_AGGREGATE_VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_d1_raw_prejoin_aggregate_v1.py"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_current_runtime_d1_folds_v1"
WORK_ROOT = ROOT / "tmp/routea_matched_three_arm_n2_current_runtime_d1_folds_v1"
TRACK_ORDER = ("outcome", "outcome", "difficult", "new_difficult_train")
TRACK_COUNTS = {"outcome": 2, "difficult": 1, "new_difficult_train": 1}
EXPECTED_TRAIN_COUNTS = {0: 775, 1: 782, 2: 798, 3: 799, 4: 794}


def state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous() for name, value in module.state_dict().items()}


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


REPLAY_ATOL = 1.0e-6
REPLAY_RTOL = 1.0e-6


def independent_select_corrected_top63(
    physical_scores: torch.Tensor,
    corrected_labels: list[str] | tuple[str, ...],
    legal: torch.Tensor,
    *,
    target_identity: str,
) -> torch.Tensor:
    """Independently reduce physical rows and select 63 negative identities.

    This deliberately does not call the producer's selection helper.  It first
    computes each corrected identity's maximum over legal physical rows, uses
    the lower physical row for an exact score tie, removes the complete target
    identity, and only then takes the top 63 identities.
    """

    require(
        isinstance(physical_scores, torch.Tensor)
        and physical_scores.dtype == torch.float32
        and physical_scores.shape == (PHYSICAL_ROWS,)
        and bool(torch.isfinite(physical_scores).all()),
        "independent negative selection requires finite float32 [5413] scores",
    )
    labels = tuple(map(str, corrected_labels))
    require(len(labels) == PHYSICAL_ROWS, "independent corrected label axis drift")
    require(
        isinstance(legal, torch.Tensor)
        and legal.dtype == torch.bool
        and legal.shape == (PHYSICAL_ROWS,),
        "independent legal mask schema drift",
    )
    best: dict[str, tuple[float, int]] = {}
    for physical_row, (identity, keep) in enumerate(zip(labels, legal.tolist())):
        if not keep:
            continue
        score = float(physical_scores[physical_row].item())
        previous = best.get(identity)
        if previous is None or score > previous[0] or (score == previous[0] and physical_row < previous[1]):
            best[identity] = (score, physical_row)
    require(target_identity in best, "independent training target absent or illegal")
    ranked = sorted(
        ((score, physical_row, identity) for identity, (score, physical_row) in best.items()),
        key=lambda item: (-item[0], item[1]),
    )
    selected = [physical_row for _score, physical_row, identity in ranked if identity != target_identity][
        :NEGATIVE_COUNT
    ]
    require(len(selected) == NEGATIVE_COUNT, "independent selection found fewer than 63 negatives")
    require(len({labels[row] for row in selected}) == NEGATIVE_COUNT, "independent negative identity duplication")
    require(target_identity not in {labels[row] for row in selected}, "independent target leakage into negatives")
    return torch.tensor(selected, dtype=torch.long)


def validate_optimizer_envelope(
    optimizer_state: Any,
    adapter_state: Mapping[str, torch.Tensor],
    *,
    completed_steps: int,
) -> None:
    """Validate the complete AdamW state envelope without changing it."""

    require(
        isinstance(optimizer_state, Mapping)
        and set(optimizer_state) == {"state", "param_groups"}
        and isinstance(optimizer_state["state"], Mapping)
        and isinstance(optimizer_state["param_groups"], list)
        and len(optimizer_state["param_groups"]) == 1,
        "optimizer envelope schema drift",
    )
    group = optimizer_state["param_groups"][0]
    require(isinstance(group, Mapping), "optimizer parameter group is not a mapping")
    params = group.get("params")
    expected_parameters = list(adapter_state.values())
    require(
        isinstance(params, list)
        and params == list(range(len(expected_parameters)))
        and set(optimizer_state["state"]) == set(params)
        and float(group.get("lr", -1.0)) == 3.0e-4
        and tuple(group.get("betas", ())) == (0.9, 0.999)
        and float(group.get("eps", -1.0)) == 1.0e-8
        and float(group.get("weight_decay", -1.0)) == 0.01
        and group.get("amsgrad") is False
        and group.get("maximize") is False,
        "optimizer parameter group or fixed AdamW recipe drift",
    )
    for parameter_id, parameter in enumerate(expected_parameters):
        slot = optimizer_state["state"][parameter_id]
        require(
            isinstance(slot, Mapping)
            and set(slot) == {"step", "exp_avg", "exp_avg_sq"},
            "optimizer per-parameter slot schema drift",
        )
        step = slot["step"]
        exp_avg = slot["exp_avg"]
        exp_avg_sq = slot["exp_avg_sq"]
        require(
            isinstance(step, torch.Tensor)
            and step.numel() == 1
            and bool(torch.isfinite(step).all())
            and float(step.item()) == float(completed_steps)
            and isinstance(exp_avg, torch.Tensor)
            and isinstance(exp_avg_sq, torch.Tensor)
            and exp_avg.shape == parameter.shape
            and exp_avg_sq.shape == parameter.shape
            and exp_avg.dtype == parameter.dtype
            and exp_avg_sq.dtype == parameter.dtype
            and bool(torch.isfinite(exp_avg).all())
            and bool(torch.isfinite(exp_avg_sq).all()),
            "optimizer moment/step envelope drift",
        )


def compare_nested(a: Any, b: Any) -> tuple[bool, float, float]:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape):
            return False, float("inf"), float("inf")
        aa, bb = a.detach().cpu(), b.detach().cpu()
        if not aa.is_floating_point():
            return torch.equal(aa, bb), 0.0, 0.0
        delta = (aa.double() - bb.double()).abs()
        max_abs = float(delta.max()) if delta.numel() else 0.0
        maximum_relative = float((delta / bb.double().abs().clamp_min(1.0e-12)).max()) if delta.numel() else 0.0
        return bool(torch.allclose(aa, bb, atol=REPLAY_ATOL, rtol=REPLAY_RTOL)), max_abs, maximum_relative
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not (isinstance(a, Mapping) and isinstance(b, Mapping) and set(a) == set(b)):
            return False, float("inf"), float("inf")
        comparisons = [compare_nested(a[key], b[key]) for key in a]
        return all(item[0] for item in comparisons), max((item[1] for item in comparisons), default=0.0), max((item[2] for item in comparisons), default=0.0)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (type(a) is type(b) and len(a) == len(b)):
            return False, float("inf"), float("inf")
        comparisons = [compare_nested(x, y) for x, y in zip(a, b)]
        return all(item[0] for item in comparisons), max((item[1] for item in comparisons), default=0.0), max((item[2] for item in comparisons), default=0.0)
    if isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            return False, float("inf"), float("inf")
        delta = abs(float(a) - float(b))
        relative = delta / max(abs(float(b)), 1.0e-12)
        return delta <= REPLAY_ATOL + REPLAY_RTOL * abs(float(b)), delta, relative
    return a == b, 0.0, 0.0


def independent_schedule(records: list[dict[str, Any]], fold: int) -> list[list[str]]:
    grouped: dict[str, dict[str, list[str]]] = {track: defaultdict(list) for track in TRACK_COUNTS}
    for row in records:
        grouped[row["track"]][row["target_identity"]].append(row["query_id"])
    canonical = {
        track: {identity: sorted(values) for identity, values in sorted(groups.items())}
        for track, groups in grouped.items()
    }
    rng = random.Random(fold_seed(fold))
    schedule = []
    for _ in range(STEPS):
        picked = {track: rng.sample(sorted(canonical[track]), count) for track, count in TRACK_COUNTS.items()}
        used: Counter[str] = Counter()
        step = []
        for track in TRACK_ORDER:
            identity = picked[track][used[track]]
            used[track] += 1
            step.append(rng.choice(canonical[track][identity]))
        schedule.append(step)
    return schedule


def expected_bindings(fold: int, label_validation: dict[str, Any], prejoin_validation: dict[str, Any]) -> dict[str, str]:
    token = validate_token_authority(ROOT)
    label_payload = LABEL_ROOT / f"fold_{fold}/payload.pt"
    return {
        **token,
        **algorithm_dependency_hashes(ROOT),
        "contract_sha256": sha256_file(CONTRACT),
        "runtime_sha256": sha256_file(RUNTIME),
        "trainer_sha256": sha256_file(TRAINER),
        "label_validation_sha256": sha256_file(LABEL_VALIDATION),
        "label_validation_logical_sha256": str(label_validation["logical_sha256"]),
        "fold_label_payload_sha256": sha256_file(label_payload),
        "prejoin_validation_sha256": sha256_file(PREJOIN_VALIDATION),
        "prejoin_validation_logical_sha256": str(prejoin_validation["logical_sha256"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    fold = args.fold
    require(fold in range(5), "fold must be in 0..4")
    require(torch.cuda.is_available(), "independent D1 continuation replay requires CUDA")
    require(LABEL_VALIDATION.is_file() and PREJOIN_VALIDATION.is_file(), "independent authority absent")
    label_validation = json.loads(LABEL_VALIDATION.read_text())
    prejoin_validation = json.loads(PREJOIN_VALIDATION.read_text())
    token_bindings = validate_token_authority(ROOT)
    expected_label_bindings = {
        **label_join_dependency_hashes(ROOT),
        "contract_sha256": sha256_file(CONTRACT),
        "runtime_sha256": sha256_file(RUNTIME),
        "producer_sha256": sha256_file(LABEL_PRODUCER),
        **EXPECTED_LABEL_SOURCE_HASHES,
        **token_bindings,
    }
    expected_prejoin_bindings = {
        **token_bindings,
        **raw_prejoin_dependency_hashes(ROOT),
        "contract_sha256": sha256_file(CONTRACT),
        "runtime_sha256": sha256_file(RUNTIME),
        "producer_sha256": sha256_file(PREJOIN_PRODUCER),
        "shard_validator_sha256": sha256_file(PREJOIN_SHARD_VALIDATOR),
        "aggregate_validator_sha256": sha256_file(PREJOIN_AGGREGATE_VALIDATOR),
    }
    require(
        label_validation.get("status") == "ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_VALIDATED"
        and bool(label_validation.get("checks"))
        and all(value is True for value in label_validation["checks"].values())
        and label_validation.get("logical_sha256") == logical_sha256(label_validation)
        and label_validation.get("bindings") == expected_label_bindings
        and label_validation.get("validator_sha256") == sha256_file(LABEL_VALIDATOR)
        and label_validation.get("next_authorized_stage")
        == "N2_D1_RAW_PREJOIN_AND_FOLD_LOCAL_TRAINING",
        "independent label authority drift",
    )
    require(
        prejoin_validation.get("status") == PREJOIN_AGGREGATE_VALID_STATUS
        and bool(prejoin_validation.get("checks"))
        and all(value is True for value in prejoin_validation["checks"].values())
        and prejoin_validation.get("logical_sha256") == logical_sha256(prejoin_validation)
        and prejoin_validation.get("bindings") == expected_prejoin_bindings
        and prejoin_validation.get("next_authorized_stage")
        == "N2_CURRENT_RUNTIME_D1_FOLD_LOCAL_TRAINING",
        "independent prejoin authority drift",
    )
    bindings = expected_bindings(fold, label_validation, prejoin_validation)
    label_payload = torch.load(LABEL_ROOT / f"fold_{fold}/payload.pt", map_location="cpu", weights_only=False)
    records = [dict(row) for row in label_payload["records"]]
    require(len(records) == EXPECTED_TRAIN_COUNTS[fold], "independent training population drift")
    allowed = {row["query_id"] for row in records}
    by_id = {row["query_id"]: row for row in records}
    schedule = independent_schedule(records, fold)

    tokens: dict[str, dict[str, Any]] = {}
    raw: dict[str, dict[str, Any]] = {}
    prejoin_seals = {int(item["shard"]): item for item in prejoin_validation["shards"]}
    for shard in range(SHARD_COUNT):
        token_payload = load_token_shard(ROOT, shard)
        prejoin_path = PREJOIN_ROOT / f"shard{shard:02d}/payload.pt"
        require(prejoin_seals[shard]["payload_sha256"] == sha256_file(prejoin_path), "independent prejoin seal drift")
        prejoin_payload = torch.load(prejoin_path, map_location="cpu", weights_only=False, mmap=True)
        prejoin_by_id = {row["query_id"]: row for row in prejoin_payload["records"]}
        for token in token_payload["records"]:
            query_id = token["query_id"]
            if query_id not in allowed:
                continue
            score = prejoin_by_id.get(query_id)
            require(score is not None, "independent train RAW score absent")
            validate_prejoin_record(score)
            tokens[query_id] = token
            raw[query_id] = score
    require(set(tokens) == set(raw) == allowed, "independent train-only cache coverage drift")

    reference_rows, corrected, _ = load_gallery(ROOT)
    legal = torch.as_tensor(label_payload["legal_physical_row_mask"]).bool()
    mask_audit = corrected_mask_audit(corrected, legal)
    require(mask_audit["eligible_corrected_identities"] == CORRECTED_ELIGIBLE_IDENTITY_COUNTS[fold], "independent corrected mask count")
    negatives: dict[str, torch.Tensor] = {}
    negative_manifest = []
    for row in records:
        selected = independent_select_corrected_top63(
            torch.as_tensor(raw[row["query_id"]]["physical_row_scores"]),
            corrected,
            legal,
            target_identity=row["target_identity"],
        )
        negatives[row["query_id"]] = selected
        negative_manifest.append([row["query_id"], tensor_sha256(selected), selected.tolist()])

    out = OUT_ROOT / f"fold_{fold}"
    result_path = out / "result.json"
    checkpoint_path = out / "checkpoint.pt"
    history_path = out / "history.pt"
    require(all(path.is_file() and not path.is_symlink() for path in (result_path, checkpoint_path, history_path)), "fold result artifacts absent")
    result = json.loads(result_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = torch.load(history_path, map_location="cpu", weights_only=False)
    require(
        result.get("status") == TRAIN_STATUS
        and result.get("fold") == fold
        and result.get("bindings") == bindings
        and result.get("checkpoint_sha256") == sha256_file(checkpoint_path)
        and result.get("history_sha256") == sha256_file(history_path)
        and result.get("final_state_dict_sha256")
        == state_dict_sha256(checkpoint.get("state_dict", {}))
        and result.get("receipt_sha256") == canonical_sha256(checkpoint.get("receipt", {}))
        and result.get("logical_sha256") == logical_sha256(result),
        "fold training result binding drift",
    )
    require(
        checkpoint.get("status") == TRAIN_STATUS
        and checkpoint.get("fold") == fold
        and checkpoint.get("receipt", {}).get("bindings") == bindings
        and history.get("status") == TRAIN_STATUS
        and history.get("fold") == fold
        and history.get("schedule") == schedule
        and history.get("negative_manifest") == negative_manifest
        and len(history.get("events", [])) == STEPS
        and [event["step"] for event in history["events"]] == list(range(1, STEPS + 1)),
        "checkpoint/history envelope drift",
    )
    require(
        all(
            torch.isfinite(torch.tensor([event["loss"], event["gradient_l1"], event["unclipped_gradient_norm"]])).all()
            and event["gradient_l1"] > 0.0
            for event in history["events"]
        ),
        "nonfinite or zero-gradient history",
    )

    d1.configure_deterministic_runtime()
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    initial = d1.build_fresh_d1_adapter()
    initial_manifest = parameter_manifest(initial)
    initial_hash = state_dict_sha256(state_cpu(initial))
    final = d1.build_fresh_d1_adapter()
    final.load_state_dict(checkpoint["state_dict"], strict=True)
    parameter_manifest(final)
    final_hash = state_dict_sha256(state_cpu(final))
    require(
        checkpoint["receipt"]["parameter_manifest"] == initial_manifest
        and checkpoint["receipt"]["parameter_count"] == PARAMETER_COUNT
        and checkpoint["receipt"]["initial_state_dict_sha256"] == initial_hash
        and checkpoint["receipt"]["final_state_dict_sha256"] == final_hash
        and initial_hash != final_hash,
        "independent parameter schema/state-change closure failed",
    )

    # Replay exactly the last 25 updates from the immutable update-775 state.
    resume_path = WORK_ROOT / f"fold_{fold}/step_0775.pt"
    final_resume_path = WORK_ROOT / f"fold_{fold}/step_0800.pt"
    require(resume_path.is_file() and final_resume_path.is_file(), "update-775/800 resume checkpoints absent")
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    final_resume = torch.load(final_resume_path, map_location="cpu", weights_only=False)
    schedule_sha = canonical_sha256(schedule)
    negative_sha = canonical_sha256(negative_manifest)
    binding_sha = canonical_sha256(bindings)
    require(
        resume.get("completed_steps") == 775
        and resume.get("bindings_sha256") == binding_sha
        and resume.get("schedule_sha256") == schedule_sha
        and resume.get("negative_manifest_sha256") == negative_sha
        and resume.get("initial_state_dict_sha256") == initial_hash
        and isinstance(resume.get("history"), list)
        and len(resume["history"]) == 775
        and [event.get("step") for event in resume["history"]] == list(range(1, 776)),
        "update-775 resume binding drift",
    )
    require(
        final_resume.get("version") == VERSION
        and final_resume.get("fold") == fold
        and final_resume.get("completed_steps") == STEPS
        and final_resume.get("bindings_sha256") == binding_sha
        and final_resume.get("schedule_sha256") == schedule_sha
        and final_resume.get("negative_manifest_sha256") == negative_sha
        and final_resume.get("initial_state_dict_sha256") == initial_hash
        and isinstance(final_resume.get("history"), list)
        and len(final_resume["history"]) == STEPS
        and [event.get("step") for event in final_resume["history"]] == list(range(1, STEPS + 1))
        and compare_nested(final_resume.get("adapter_state"), checkpoint["state_dict"])[0]
        and compare_nested(final_resume["history"], history["events"])[0],
        "update-800 complete resume envelope drift",
    )
    validate_optimizer_envelope(
        resume.get("optimizer_state"), resume.get("adapter_state", {}), completed_steps=775
    )
    validate_optimizer_envelope(
        final_resume.get("optimizer_state"), final_resume.get("adapter_state", {}), completed_steps=STEPS
    )
    adapter = d1.build_fresh_d1_adapter().to(device)
    adapter.load_state_dict(resume["adapter_state"], strict=True)
    optimizer = d1.build_fresh_d1_optimizer(adapter)
    optimizer.load_state_dict(resume["optimizer_state"])
    optimizer_to_device(optimizer, device)
    torch.set_rng_state(resume["cpu_rng_state"])
    torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
    padded, reference_mask = pad_reference_tokens([value.detach().to(torch.float32) for value in reference_rows])
    references = padded.to(device)
    reference_mask = reference_mask.to(device)
    replay_events = list(resume["history"])
    for step in range(775, STEPS):
        losses = []
        diagnostics = []
        for query_id in schedule[step]:
            row = by_id[query_id]
            token = tokens[query_id]
            candidate_rows = torch.cat((torch.tensor([row["target_physical_row"]]), negatives[query_id])).long().to(device)
            loss = d1.fresh_d1_loss(
                adapter,
                torch.as_tensor(token["image_tokens"]).float().to(device),
                torch.as_tensor(token["template_tokens"]).float().to(device),
                references.index_select(0, candidate_rows),
                reference_mask.index_select(0, candidate_rows),
                grid_h=int(token["grid_shape"][0]),
                grid_w=int(token["grid_shape"][1]),
            )
            losses.append(loss.total)
            diagnostics.append(
                {
                    "query_id": query_id,
                    "track": row["track"],
                    "raw_rank": int(loss.raw_ranking.target_rank),
                    "adapted_rank": int(loss.ranking.target_rank),
                    "ranking": float(loss.ranking.total.detach()),
                    "preservation": float(loss.margins.preservation.detach()),
                    "correction": float(loss.margins.correction.detach()),
                    "drift": float(loss.drift.detach()),
                }
            )
        update = d1.fresh_d1_optimizer_step(torch.stack(losses).mean(), adapter=adapter, optimizer=optimizer)
        replay_events.append({"step": step + 1, **update, "rows": diagnostics})
    replay_state = state_cpu(adapter)
    state_comparison = compare_nested(replay_state, checkpoint["state_dict"])
    optimizer_comparison = compare_nested(optimizer.state_dict(), final_resume["optimizer_state"])
    history_comparison = compare_nested(replay_events[775:], history["events"][775:])
    require(
        state_comparison[0]
        and optimizer_comparison[0]
        and history_comparison[0],
        "independent update-775 continuation replay exceeds frozen tolerance or structural exactness",
    )

    checks = {
        "token_label_prejoin_authorities_complete": True,
        "train_only_identity_supergroup_fold_payload": True,
        "corrected_5412_mask_and_top63_reconstructed": True,
        "exact_eight_key_49792_shared_parameter_schema": True,
        "fixed_seed17_initialization_and_fold_schedule": True,
        "finite_nonzero_gradient_800_step_history": True,
        "final_step_only_checkpoint_changed": True,
        "update800_complete_binding_schedule_negative_initial_optimizer_envelope": True,
        "update775_to800_structure_exact_float_tolerance_pass": True,
        "zero_heldout_label_path_external_sealed_access": checkpoint["receipt"]["access"]
        == {
            "train_target_label_join_count": len(records),
            "heldout_target_label_read_count": 0,
            "query_path_read_count": 0,
            "historical_query_token_read_count": 0,
            "historical_d1_checkpoint_load_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        },
    }
    require(bool(checks) and all(checks.values()), "independent D1 fold validation failed")
    value = {
        "version": VERSION,
        "status": TRAIN_VALID_STATUS,
        "claim_level": "INDEPENDENT_FOLD_LOCAL_TRAINING_VALIDATION_NO_SCIENTIFIC_CLAIM",
        "fold": fold,
        "checks": checks,
        "train_query_count": len(records),
        "parameter_count": PARAMETER_COUNT,
        "initial_state_dict_sha256": initial_hash,
        "final_state_dict_sha256": final_hash,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history_sha256": sha256_file(history_path),
        "result_sha256": sha256_file(result_path),
        "bindings": bindings,
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "continuation_replay_tolerance": {
            "absolute": REPLAY_ATOL,
            "relative": REPLAY_RTOL,
            "nonfloating_structure_and_values_exact": True,
            "parameter_state_max_abs": state_comparison[1],
            "parameter_state_max_rel": state_comparison[2],
            "optimizer_state_max_abs": optimizer_comparison[1],
            "optimizer_state_max_rel": optimizer_comparison[2],
            "history_float_max_abs": history_comparison[1],
            "history_float_max_rel": history_comparison[2],
        },
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": "N2_CURRENT_RUNTIME_D1_OOF_SCORING_AND_THREE_ARM_REMATERIALIZATION_CONTRACT",
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(out / "independent_validation.json", value)
    print(json.dumps({"status": value["status"], "fold": fold, "checks": checks}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
