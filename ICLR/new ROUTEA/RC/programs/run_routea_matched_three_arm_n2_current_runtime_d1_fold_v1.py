#!/usr/bin/env python3
"""Train one fresh current-runtime, fold-local N2 D1 adapter."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
ROUTEA = ROOT.parent
sys.path[:0] = [str(ROOT / "src"), str(ROUTEA / "route_a_core")]

from rc_aslo_xf.n2_current_runtime_d1_training_v1 import (  # noqa: E402
    CHECKPOINT_INTERVAL,
    CORRECTED_ELIGIBLE_IDENTITY_COUNTS,
    EXPECTED_LABEL_SOURCE_HASHES,
    PARAMETER_COUNT,
    PREJOIN_AGGREGATE_VALID_STATUS,
    SEED,
    SHARD_COUNT,
    STEPS,
    TRAIN_STATUS,
    VERSION,
    algorithm_dependency_hashes,
    label_join_dependency_hashes,
    raw_prejoin_dependency_hashes,
    atomic_json,
    atomic_torch,
    canonical_sha256,
    corrected_mask_audit,
    fold_seed,
    load_gallery,
    load_token_shard,
    logical_sha256,
    parameter_manifest,
    require,
    select_corrected_top63,
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
VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_current_runtime_d1_fold_v1.py"
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


def recipe(fold: int) -> dict[str, Any]:
    return {
        "algorithm_version": d1.VERSION,
        "fold": fold,
        "initialization_seed": SEED,
        "schedule_seed": fold_seed(fold),
        "steps": STEPS,
        "batch_size": 4,
        "track_order": list(TRACK_ORDER),
        "sampling": "identity_uniform_without_replacement_within_track_step",
        "candidate_count": 64,
        "negative_count": 63,
        "negative_mining": "sealed_RAW_scores_then_corrected_identity_max_lower_row_tie_then_target_exclusion_then_top63",
        "adapter": {
            "kind": "pointwise_query_only",
            "dim": 128,
            "hidden": 128,
            "max_ratio": 0.25,
            "parameter_count": PARAMETER_COUNT,
            "zero_initialized_delta_head": True,
        },
        "optimizer": {"kind": "AdamW", "learning_rate": 3.0e-4, "weight_decay": 0.01, "gradient_clip_norm": 1.0},
        "loss": {
            "temperature": 0.05,
            "preserve_cap": 0.03,
            "error_margin": 0.02,
            "preservation_weight": 10.0,
            "correction_weight": 2.0,
            "drift_weight": 0.05,
        },
        "checkpoint_selection": "final_step_800_only_no_heldout_selection",
    }


def build_schedule(records: Sequence[Mapping[str, Any]], fold: int) -> list[list[str]]:
    grouped: dict[str, dict[str, list[str]]] = {track: defaultdict(list) for track in TRACK_COUNTS}
    for row in records:
        grouped[str(row["track"])][str(row["target_identity"])].append(str(row["query_id"]))
    for track in grouped:
        grouped[track] = {
            identity: sorted(query_ids) for identity, query_ids in sorted(grouped[track].items())
        }
        require(len(grouped[track]) >= TRACK_COUNTS[track], f"fold {fold} lacks {track} identities")
    rng = random.Random(fold_seed(fold))
    output: list[list[str]] = []
    for _ in range(STEPS):
        chosen = {
            track: rng.sample(sorted(grouped[track]), count)
            for track, count in TRACK_COUNTS.items()
        }
        used: Counter[str] = Counter()
        step: list[str] = []
        for track in TRACK_ORDER:
            identity = chosen[track][used[track]]
            used[track] += 1
            step.append(rng.choice(grouped[track][identity]))
        output.append(step)
    require(len(output) == STEPS and all(len(step) == 4 for step in output), "schedule shape drift")
    return output


def validate_upstreams(fold: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    token_bindings = validate_token_authority(ROOT)
    require(LABEL_VALIDATION.is_file() and PREJOIN_VALIDATION.is_file(), "D1 training authorities absent")
    label_validation = json.loads(LABEL_VALIDATION.read_text())
    prejoin_validation = json.loads(PREJOIN_VALIDATION.read_text())
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
        "label-join authority is not independently valid",
    )
    require(
        prejoin_validation.get("status") == PREJOIN_AGGREGATE_VALID_STATUS
        and bool(prejoin_validation.get("checks"))
        and all(value is True for value in prejoin_validation["checks"].values())
        and prejoin_validation.get("query_count") == 987
        and prejoin_validation.get("logical_sha256") == logical_sha256(prejoin_validation)
        and prejoin_validation.get("bindings") == expected_prejoin_bindings
        and prejoin_validation.get("next_authorized_stage")
        == "N2_CURRENT_RUNTIME_D1_FOLD_LOCAL_TRAINING",
        "RAW prejoin aggregate is not independently valid",
    )
    fold_payload_path = LABEL_ROOT / f"fold_{fold}/payload.pt"
    require(fold_payload_path.is_file(), f"fold {fold} train-only label payload absent")
    seal = {int(item["fold"]): item for item in label_validation["folds"]}.get(fold)
    require(seal is not None and seal["payload_sha256"] == sha256_file(fold_payload_path), "fold label payload seal drift")
    fold_payload = torch.load(fold_payload_path, map_location="cpu", weights_only=False)
    bindings = {
        **token_bindings,
        **algorithm_dependency_hashes(ROOT),
        "contract_sha256": sha256_file(CONTRACT),
        "runtime_sha256": sha256_file(RUNTIME),
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "label_validation_sha256": sha256_file(LABEL_VALIDATION),
        "label_validation_logical_sha256": str(label_validation["logical_sha256"]),
        "fold_label_payload_sha256": sha256_file(fold_payload_path),
        "prejoin_validation_sha256": sha256_file(PREJOIN_VALIDATION),
        "prejoin_validation_logical_sha256": str(prejoin_validation["logical_sha256"]),
    }
    return fold_payload, prejoin_validation, bindings


def load_fold_data(fold: int) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]],
    list[torch.Tensor], tuple[str, ...], torch.Tensor, dict[str, str]
]:
    labels, prejoin_validation, bindings = validate_upstreams(fold)
    records = [dict(row) for row in labels["records"]]
    require(len(records) == EXPECTED_TRAIN_COUNTS[fold], "fold training query count drift")
    allowed = {row["query_id"] for row in records}
    require(len(allowed) == len(records), "fold training query IDs duplicate")

    tokens: dict[str, dict[str, Any]] = {}
    raw: dict[str, dict[str, Any]] = {}
    prejoin_seals = {int(item["shard"]): item for item in prejoin_validation["shards"]}
    for shard in range(SHARD_COUNT):
        token_payload = load_token_shard(ROOT, shard)
        prejoin_path = PREJOIN_ROOT / f"shard{shard:02d}/payload.pt"
        require(prejoin_seals[shard]["payload_sha256"] == sha256_file(prejoin_path), "prejoin payload seal drift")
        prejoin_payload = torch.load(prejoin_path, map_location="cpu", weights_only=False, mmap=True)
        prejoin_by_id = {row["query_id"]: row for row in prejoin_payload["records"]}
        for token in token_payload["records"]:
            query_id = token["query_id"]
            if query_id not in allowed:
                continue
            row = prejoin_by_id.get(query_id)
            require(row is not None, f"training query absent from RAW prejoin: {query_id}")
            validate_prejoin_record(row)
            require(
                row["image_tokens_sha256"] == token["image_tokens_sha256"]
                and row["template_tokens_sha256"] == token["template_tokens_sha256"],
                "training token/RAW prejoin hash drift",
            )
            tokens[query_id] = token
            raw[query_id] = row
    require(set(tokens) == set(raw) == allowed, "fold train-only token/prejoin coverage drift")

    references, corrected, _gallery_receipt = load_gallery(ROOT)
    legal = torch.as_tensor(labels["legal_physical_row_mask"]).bool().contiguous()
    mask_audit = corrected_mask_audit(corrected, legal)
    require(
        mask_audit == labels["mask_audit"]
        and mask_audit["eligible_corrected_identities"] == CORRECTED_ELIGIBLE_IDENTITY_COUNTS[fold],
        "fold corrected legal mask drift",
    )
    for row in records:
        require(
            corrected[int(row["target_physical_row"])] == row["target_identity"]
            and bool(legal[int(row["target_physical_row"])]),
            "fold training positive row/identity drift",
        )
    return records, tokens, raw, references, corrected, legal, bindings


def latest_checkpoint(directory: Path) -> Path | None:
    candidates = []
    if directory.exists():
        for path in directory.glob("step_*.pt"):
            suffix = path.stem.removeprefix("step_")
            require(path.is_file() and not path.is_symlink() and suffix.isdigit(), "malformed resume checkpoint")
            filename_step = int(suffix)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            require(
                isinstance(payload, Mapping)
                and type(payload.get("completed_steps")) is int
                and payload["completed_steps"] == filename_step,
                f"resume checkpoint filename/payload step mismatch: {path.name}",
            )
            candidates.append((filename_step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def run(fold: int) -> dict[str, Any]:
    require(fold in range(5), "fold must be in 0..4")
    require(torch.cuda.is_available(), "formal D1 training requires CUDA")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS workspace drift")
    d1.configure_deterministic_runtime()
    records, tokens, raw, reference_rows, corrected, legal, bindings = load_fold_data(fold)
    by_id = {row["query_id"]: row for row in records}
    schedule = build_schedule(records, fold)
    schedule_sha = canonical_sha256(schedule)

    negatives: dict[str, torch.Tensor] = {}
    negative_manifest = []
    for index, row in enumerate(records, start=1):
        query_id = row["query_id"]
        selected = select_corrected_top63(
            torch.as_tensor(raw[query_id]["physical_row_scores"]),
            corrected,
            legal,
            target_identity=row["target_identity"],
        )
        negatives[query_id] = selected
        negative_manifest.append([query_id, tensor_sha256(selected), selected.tolist()])
        if index == 1 or index % 64 == 0 or index == len(records):
            print(json.dumps({"event": "n2_d1_negative_selection", "fold": fold, "done": index, "total": len(records)}, sort_keys=True), flush=True)
    negative_sha = canonical_sha256(negative_manifest)

    device = torch.device("cuda")
    padded, reference_mask = pad_reference_tokens([value.detach().to(torch.float32) for value in reference_rows])
    references = padded.to(device)
    reference_mask = reference_mask.to(device)
    del padded, reference_rows

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    adapter = d1.build_fresh_d1_adapter().to(device)
    manifest = parameter_manifest(adapter)
    initial_hash = state_dict_sha256(state_cpu(adapter))
    optimizer = d1.build_fresh_d1_optimizer(adapter)
    history: list[dict[str, Any]] = []
    start = 0
    work = WORK_ROOT / f"fold_{fold}"
    work.mkdir(parents=True, exist_ok=True)
    binding_sha = canonical_sha256(bindings)
    resume = latest_checkpoint(work)
    if resume is not None:
        value = torch.load(resume, map_location="cpu", weights_only=False)
        require(
            value.get("version") == VERSION
            and value.get("fold") == fold
            and value.get("bindings_sha256") == binding_sha
            and value.get("schedule_sha256") == schedule_sha
            and value.get("negative_manifest_sha256") == negative_sha
            and value.get("initial_state_dict_sha256") == initial_hash
            and 0 < int(value.get("completed_steps", -1)) <= STEPS
            and len(value.get("history", [])) == int(value["completed_steps"]),
            "D1 resume checkpoint binding drift",
        )
        adapter.load_state_dict(value["adapter_state"], strict=True)
        optimizer.load_state_dict(value["optimizer_state"])
        optimizer_to_device(optimizer, device)
        torch.set_rng_state(value["cpu_rng_state"])
        torch.cuda.set_rng_state_all(value["cuda_rng_state"])
        history = list(value["history"])
        start = int(value["completed_steps"])

    def save_resume(completed: int) -> None:
        atomic_torch(
            work / f"step_{completed:04d}.pt",
            {
                "version": VERSION,
                "fold": fold,
                "completed_steps": completed,
                "adapter_state": state_cpu(adapter),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "cpu_rng_state": torch.get_rng_state().cpu(),
                "cuda_rng_state": [value.cpu() for value in torch.cuda.get_rng_state_all()],
                "bindings_sha256": binding_sha,
                "schedule_sha256": schedule_sha,
                "negative_manifest_sha256": negative_sha,
                "initial_state_dict_sha256": initial_hash,
            },
        )

    for step in range(start, STEPS):
        losses = []
        diagnostics = []
        for query_id in schedule[step]:
            row = by_id[query_id]
            token = tokens[query_id]
            candidate_rows = torch.cat(
                (torch.tensor([row["target_physical_row"]], dtype=torch.long), negatives[query_id])
            ).to(device)
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
        history.append({"step": step + 1, **update, "rows": diagnostics})
        if step == 0 or (step + 1) % CHECKPOINT_INTERVAL == 0 or step + 1 == STEPS:
            print(json.dumps({"event": "n2_d1_train", "fold": fold, "step": step + 1, "total": STEPS, "loss": update["loss"]}, sort_keys=True), flush=True)
            save_resume(step + 1)

    final_state = state_cpu(adapter)
    final_hash = state_dict_sha256(final_state)
    require(final_hash != initial_hash, "D1 optimizer did not change adapter")
    receipt = {
        "version": VERSION,
        "status": TRAIN_STATUS,
        "claim_level": "FOLD_LOCAL_TRAIN_ONLY_NO_HELDOUT_MODEL_SELECTION_NO_SCIENTIFIC_CLAIM",
        "fold": fold,
        "recipe": recipe(fold),
        "recipe_sha256": canonical_sha256(recipe(fold)),
        "parameter_manifest": manifest,
        "parameter_manifest_sha256": canonical_sha256(manifest),
        "parameter_count": PARAMETER_COUNT,
        "initial_state_dict_sha256": initial_hash,
        "final_state_dict_sha256": final_hash,
        "schedule_sha256": schedule_sha,
        "negative_manifest_sha256": negative_sha,
        "optimizer_steps": STEPS,
        "checkpoint_selection": "final_step_800_only_no_heldout_selection",
        "bindings": bindings,
        "access": {
            "train_target_label_join_count": len(records),
            "heldout_target_label_read_count": 0,
            "query_path_read_count": 0,
            "historical_query_token_read_count": 0,
            "historical_d1_checkpoint_load_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        },
    }
    checkpoint = {"version": VERSION, "status": TRAIN_STATUS, "fold": fold, "state_dict": final_state, "receipt": receipt}
    history_payload = {
        "version": VERSION,
        "status": TRAIN_STATUS,
        "fold": fold,
        "events": history,
        "schedule": schedule,
        "negative_manifest": negative_manifest,
        "bindings_sha256": binding_sha,
    }
    out = OUT_ROOT / f"fold_{fold}"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if out.exists():
        require(out.is_dir() and not out.is_symlink(), "existing final fold output is not a regular directory")
        required = (out / "checkpoint.pt", out / "history.pt", out / "result.json")
        require(all(path.is_file() and not path.is_symlink() for path in required), "partial final fold publication")
        existing_checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
        existing_history = torch.load(out / "history.pt", map_location="cpu", weights_only=False)
        existing_result = json.loads((out / "result.json").read_text())
        require(
            existing_checkpoint.get("receipt") == receipt
            and state_dict_sha256(existing_checkpoint.get("state_dict", {})) == final_hash
            and existing_history.get("bindings_sha256") == binding_sha
            and existing_history.get("events") == history
            and existing_result.get("bindings") == bindings
            and existing_result.get("final_state_dict_sha256") == final_hash
            and existing_result.get("checkpoint_sha256") == sha256_file(out / "checkpoint.pt")
            and existing_result.get("history_sha256") == sha256_file(out / "history.pt")
            and existing_result.get("receipt_sha256") == canonical_sha256(existing_checkpoint["receipt"])
            and existing_result.get("logical_sha256") == logical_sha256(existing_result),
            "existing final fold publication drift",
        )
        return existing_result

    temporary = Path(tempfile.mkdtemp(prefix=f".fold_{fold}.", suffix=".partial", dir=OUT_ROOT))
    try:
        torch.save(checkpoint, temporary / "checkpoint.pt")
        torch.save(history_payload, temporary / "history.pt")
        checkpoint_sha = sha256_file(temporary / "checkpoint.pt")
        history_sha = sha256_file(temporary / "history.pt")
        result = {
            "version": VERSION,
            "status": TRAIN_STATUS,
            "claim_level": receipt["claim_level"],
            "fold": fold,
            "train_query_count": len(records),
            "train_track_counts": dict(sorted(Counter(row["track"] for row in records).items())),
            "train_identity_count": len({row["target_identity"] for row in records}),
            "train_supergroup_count": len({row["supergroup"] for row in records}),
            "optimizer_steps": STEPS,
            "checkpoint_sha256": checkpoint_sha,
            "history_sha256": history_sha,
            "receipt_sha256": canonical_sha256(receipt),
            "final_state_dict_sha256": final_hash,
            "bindings": bindings,
            "scientific_GO_or_NO_GO": None,
            "automatic_stage_advance": False,
            "next_authorized_stage": "N2_CURRENT_RUNTIME_D1_FOLD_INDEPENDENT_VALIDATION",
            "logical_sha256": "",
        }
        result["logical_sha256"] = logical_sha256(result)
        with (temporary / "result.json").open("w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    value = run(args.fold)
    print(json.dumps({"status": value["status"], "fold": args.fold}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
