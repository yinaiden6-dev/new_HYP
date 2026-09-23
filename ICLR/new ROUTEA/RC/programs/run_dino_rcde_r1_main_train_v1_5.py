#!/usr/bin/env python3
"""Train one authorized R1 main arm/fold from the canonical FP16 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARAMETER_COUNT = 65_125
ALLOWED_ARMS = ("RCDE_BAG", "RCDE_CONTEXT")
UPDATE_COUNT = 2_048
EPISODES_PER_UPDATE = 4
SEED = 17
QUERY_TILE_ROWS = 8
REFERENCE_TILE_ROWS = 16
ORDER_NAMESPACE = "DINO_RCDE_R1_MAIN_QUERY_ORDER_V1_5"
DIRECTION_NAMESPACE = "DINO_RCDE_R1_PAIR_DIRECTION_V1_5"
BAG_QUERY_NAMESPACE = "DINO_RCDE_R1_BAG_QUERY_PERMUTATION_V1_5"
BAG_REFERENCE_NAMESPACE = "DINO_RCDE_R1_BAG_REFERENCE_PERMUTATION_V1_5"


class TrainingAbort(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def hash_parts(namespace: str, *parts: object) -> str:
    payload = "\0".join((namespace, *(str(part) for part in parts))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fold_number(value: object) -> int:
    text = str(value).strip().upper()
    if text.startswith("F"):
        text = text[1:]
    return int(text)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingAbort(f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_bound(path_text: str) -> Path:
    path = (ROOT / path_text).resolve()
    if ROOT not in path.parents:
        raise TrainingAbort(f"binding escapes RC root: {path}")
    lowered = {part.lower() for part in path.parts}
    if lowered.intersection({"c8", "opened", "sealed"}):
        raise TrainingAbort(f"protected path requested: {path}")
    return path


def validate_contract(protocol_path: Path, authority_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(protocol_path)
    authority = read_json(authority_path)
    if protocol.get("schema_version") != "rc_dino_rcde_r1_main_train_protocol_v1_5_20260813":
        raise TrainingAbort("R1 main training protocol schema drift")
    expected_authority = protocol.get("authority", {})
    if authority_path.resolve() != resolve_bound(str(expected_authority.get("path", ""))):
        raise TrainingAbort("authority path drift")
    if file_sha256(authority_path) != expected_authority.get("sha256"):
        raise TrainingAbort("authority digest drift")
    if authority.get("status") != expected_authority.get("required_status"):
        raise TrainingAbort("authority status does not authorize R1 main training")
    if authority.get("next_authorized_stage") != "R1_MAIN_TRAINING":
        raise TrainingAbort("authority next-stage drift")
    if authority.get("natural_training_authorized") is not True:
        raise TrainingAbort("natural training is not explicitly authorized")
    if protocol.get("automatic_stage_advance") is not False:
        raise TrainingAbort("automatic stage advancement must remain disabled")
    for name, binding in protocol.get("bindings", {}).items():
        path = resolve_bound(str(binding["path"]))
        if file_sha256(path) != binding["sha256"]:
            raise TrainingAbort(f"binding drift: {name}")
    cache_validation = read_json(resolve_bound(protocol["bindings"]["cache_validation"]["path"]))
    if cache_validation.get("status") != "DINO_RCDE_R1_CACHE_VALIDATION_PASS":
        raise TrainingAbort("canonical cache validation is not PASS")
    return protocol, authority


def seed_everything() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json_bytes(list(tensor.shape)) + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def cache_index(cache_root: Path) -> dict[tuple[str, int], Path]:
    result: dict[tuple[str, int], Path] = {}
    pattern = re.compile(r"^(query|reference)_(\d+)\.pt$")
    for path in cache_root.glob("*.pt"):
        match = pattern.match(path.name)
        if match is None:
            continue
        key = (match.group(1), int(match.group(2)))
        if key in result:
            raise TrainingAbort(f"duplicate cache key: {key}")
        result[key] = path
    if sum(kind == "query" for kind, _ in result) != 600:
        raise TrainingAbort("canonical query-cache cardinality drift")
    if sum(kind == "reference" for kind, _ in result) != 4_748:
        raise TrainingAbort("canonical reference-cache cardinality drift")
    return result


def load_payload(
    index: Mapping[tuple[str, int], Path],
    resident: dict[tuple[str, int], Mapping[str, Any]],
    kind: str,
    ordinal: int,
) -> Mapping[str, Any]:
    key = (kind, ordinal)
    if key not in resident:
        path = index.get(key)
        if path is None:
            raise TrainingAbort(f"missing canonical cache payload: {key}")
        resident[key] = torch.load(path, map_location="cpu", weights_only=False)
    return resident[key]


def permutation_shift(namespace: str, fold: int, item: object, valid_count: int) -> int:
    if valid_count < 2:
        raise TrainingAbort("BAG permutation requires at least two valid tokens")
    return 1 + int(hash_parts(namespace, fold, item), 16) % (valid_count - 1)


def core_inputs(
    payload: Mapping[str, Any],
    device: torch.device,
    *,
    bag_namespace: str | None,
    fold: int,
    item: object,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    tokens = payload["tokens_fp16"]
    mask = payload["valid_patch_mask"]
    if tokens.shape[0] != 1 or mask.shape[0] != 1:
        raise TrainingAbort("cache payload batch shape drift")
    tokens = tokens[0].to(device=device, dtype=torch.float32)
    mask = mask[0].to(device=device, dtype=torch.bool)
    grid = tuple(int(value) for value in mask.shape)
    if tuple(tokens.shape) != (4, math.prod(grid), 768):
        raise TrainingAbort("cache/core token shape drift")
    if bag_namespace is not None:
        valid = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        shift = permutation_shift(bag_namespace, fold, item, int(valid.numel()))
        source = valid.roll(shifts=shift)
        permuted = tokens.clone()
        permuted[:, valid, :] = tokens[:, source, :]
        tokens = permuted
    return tokens, mask, grid


def learning_rate(update_one_based: int) -> float:
    maximum = 3.0e-4
    minimum = 3.0e-5
    warmup = 128
    if update_one_based <= warmup:
        return maximum * update_one_based / warmup
    progress = (update_one_based - warmup) / (UPDATE_COUNT - warmup)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def ordered_pools(episode_ledger: Mapping[str, Any], roles: Mapping[str, Any], fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    role_rows = roles.get("rows", roles.get("records", []))
    role_by_query = {str(row["query_id"]): row for row in role_rows}
    correct: list[dict[str, Any]] = []
    wrong: list[dict[str, Any]] = []
    for episode in episode_ledger.get("episodes", []):
        if fold_number(episode.get("heldout_fold")) != fold or episode.get("pair_loss_eligible") is not True:
            continue
        query_id = str(episode["query_id"])
        role = role_by_query.get(query_id)
        if role is None or fold_number(role["inner_fold"]) == fold:
            raise TrainingAbort("outer-train identity restriction drift")
        if len(episode.get("selected_negatives", [])) < 1:
            raise TrainingAbort("eligible episode has no rivals")
        (correct if episode.get("full_gallery_RAW_correct") is True else wrong).append(episode)
    key = lambda row: hash_parts(ORDER_NAMESPACE, fold, row["query_id"])
    correct.sort(key=key)
    wrong.sort(key=key)
    if len(correct) < 2 or len(wrong) < 2:
        raise TrainingAbort("RAW-correct/RAW-wrong balancing pools are incomplete")
    return correct, wrong


def update_episodes(correct: Sequence[dict[str, Any]], wrong: Sequence[dict[str, Any]], update_zero_based: int) -> list[dict[str, Any]]:
    selected = [
        correct[(2 * update_zero_based) % len(correct)],
        correct[(2 * update_zero_based + 1) % len(correct)],
        wrong[(2 * update_zero_based) % len(wrong)],
        wrong[(2 * update_zero_based + 1) % len(wrong)],
    ]
    if len({str(row["query_id"]) for row in selected}) != EPISODES_PER_UPDATE:
        raise TrainingAbort("balanced update contains duplicate queries")
    return selected


def target_first(fold: int, query_id: str, rival_row: int) -> bool:
    return int(hash_parts(DIRECTION_NAMESPACE, fold, query_id, rival_row), 16) % 2 == 0


def schedule_event(fold: int, update_one_based: int, episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    queries = []
    for episode in episodes:
        rivals = [int(row["physical_row"]) for row in episode["selected_negatives"]]
        queries.append(
            {
                "query_id": str(episode["query_id"]),
                "query_ordinal": int(episode["query_ordinal"]),
                "target_physical_row": int(episode["target_physical_row"]),
                "rivals": rivals,
                "target_first": [target_first(fold, str(episode["query_id"]), row) for row in rivals],
            }
        )
    return {"update": update_one_based, "queries": queries}


def staged_episode_backward(
    model: DINO_RCDE_V1_2,
    query_payload: Mapping[str, Any],
    target_payload: Mapping[str, Any],
    rival_payloads: Sequence[Mapping[str, Any]],
    directions: Sequence[bool],
    *,
    denominator: int,
    device: torch.device,
    arm: str,
    fold: int,
    query_id: str,
    target_row: int,
    rival_rows: Sequence[int],
) -> list[float]:
    if len(rival_payloads) != len(directions) or len(rival_rows) != len(directions):
        raise TrainingAbort("rival/direction cardinality drift")
    bag = arm == "RCDE_BAG"
    q, qm, qg = core_inputs(
        query_payload,
        device,
        bag_namespace=BAG_QUERY_NAMESPACE if bag else None,
        fold=fold,
        item=query_id,
    )
    target_tokens, target_mask, target_grid = core_inputs(
        target_payload,
        device,
        bag_namespace=BAG_REFERENCE_NAMESPACE if bag else None,
        fold=fold,
        item=target_row,
    )
    target = model.decode_candidate_true_streaming_fast(
        q,
        target_tokens,
        qm,
        target_mask,
        qg,
        target_grid,
        query_tile_rows=QUERY_TILE_ROWS,
        reference_tile_rows=REFERENCE_TILE_ROWS,
    )
    target_proxy = target.relational.detach().requires_grad_(True)
    losses: list[float] = []
    for payload, is_target_first, rival_row in zip(rival_payloads, directions, rival_rows):
        rival_tokens, rival_mask, rival_grid = core_inputs(
            payload,
            device,
            bag_namespace=BAG_REFERENCE_NAMESPACE if bag else None,
            fold=fold,
            item=rival_row,
        )
        rival = model.decode_candidate_true_streaming_fast(
            q,
            rival_tokens,
            qm,
            rival_mask,
            qg,
            rival_grid,
            query_tile_rows=QUERY_TILE_ROWS,
            reference_tile_rows=REFERENCE_TILE_ROWS,
        )
        if is_target_first:
            pair = model.compare_relational(target_proxy, rival.relational, qm)
            signed_logit = pair.logit
        else:
            pair = model.compare_relational(rival.relational, target_proxy, qm)
            signed_logit = -pair.logit
        loss = F.softplus(-signed_logit)
        losses.append(float(loss.detach().cpu()))
        (loss / denominator).backward()
        del rival, pair, loss, signed_logit, rival_tokens, rival_mask
    gradient = target_proxy.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise TrainingAbort("target relational VJP is absent or nonfinite")
    target.relational.backward(gradient.detach())
    return losses


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ALLOWED_ARMS, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    authority_path = args.authority.resolve()
    cache_root = args.cache_root.resolve()
    output_dir = args.output_dir.resolve()
    if ROOT not in cache_root.parents or ROOT not in output_dir.parents:
        raise TrainingAbort("cache/output path escapes RC root")
    if output_dir.exists():
        raise TrainingAbort(f"immutable output namespace already exists: {output_dir}")
    protocol, _ = validate_contract(protocol_path, authority_path)
    scope = protocol["training"]
    if args.arm not in scope["arms"] or args.fold not in scope["outer_folds"]:
        raise TrainingAbort("requested arm/fold is outside frozen scope")

    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything()
    if not torch.cuda.is_available():
        raise TrainingAbort("R1 main training requires CUDA")
    device = torch.device("cuda")
    index = cache_index(cache_root)
    resident: dict[tuple[str, int], Mapping[str, Any]] = {}
    roles = read_json(resolve_bound(protocol["bindings"]["training_roles"]["path"]))
    ledger = read_json(resolve_bound(protocol["bindings"]["episode_ledger"]["path"]))
    correct, wrong = ordered_pools(ledger, roles, args.fold)

    model = DINO_RCDE_V1_2().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise TrainingAbort("RCDE parameter-count drift")
    initial_state_sha256 = state_dict_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    schedule_digest = hashlib.sha256()
    loss_trace: list[dict[str, Any]] = []
    total_pair_count = 0
    started = time.monotonic()

    model.train()
    for update_zero in range(UPDATE_COUNT):
        update_one = update_zero + 1
        episodes = update_episodes(correct, wrong, update_zero)
        event = schedule_event(args.fold, update_one, episodes)
        schedule_digest.update(json_bytes(event) + b"\n")
        denominator = sum(len(row["selected_negatives"]) for row in episodes)
        if denominator < EPISODES_PER_UPDATE:
            raise TrainingAbort("empty update denominator")
        lr = learning_rate(update_one)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        update_losses: list[float] = []
        for episode, query_event in zip(episodes, event["queries"]):
            query_ordinal = int(episode["query_ordinal"])
            target_row = int(episode["target_physical_row"])
            rival_rows = list(query_event["rivals"])
            update_losses.extend(
                staged_episode_backward(
                    model,
                    load_payload(index, resident, "query", query_ordinal),
                    load_payload(index, resident, "reference", target_row),
                    [load_payload(index, resident, "reference", row) for row in rival_rows],
                    list(query_event["target_first"]),
                    denominator=denominator,
                    device=device,
                    arm=args.arm,
                    fold=args.fold,
                    query_id=str(episode["query_id"]),
                    target_row=target_row,
                    rival_rows=rival_rows,
                )
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise TrainingAbort("nonfinite global gradient norm")
        optimizer.step()
        total_pair_count += denominator
        loss_trace.append(
            {
                "update": update_one,
                "learning_rate": lr,
                "mean_pair_loss": sum(update_losses) / len(update_losses),
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "pair_count": denominator,
            }
        )
        if update_one == 1 or update_one % 32 == 0:
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "fold": args.fold,
                        "update": update_one,
                        "mean_pair_loss": loss_trace[-1]["mean_pair_loss"],
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    torch.cuda.synchronize(device)
    final_state_sha256 = state_dict_sha256(model)
    checkpoint_path = output_dir / "checkpoint_update2048.pt"
    atomic_torch(
        checkpoint_path,
        {
            "schema_version": "rc_dino_rcde_r1_main_checkpoint_v1_5",
            "arm": args.arm,
            "outer_fold": args.fold,
            "update": UPDATE_COUNT,
            "seed": SEED,
            "protocol_sha256": file_sha256(protocol_path),
            "authority_sha256": file_sha256(authority_path),
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": final_state_sha256,
            "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        },
    )
    result = {
        "schema_version": "rc_dino_rcde_r1_main_train_result_v1_5",
        "status": "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAIN_COMPLETE",
        "claim_level": "engineering_training_complete_not_scientific_GO_or_NO_GO",
        "arm": args.arm,
        "outer_fold": args.fold,
        "authority_sha256": file_sha256(authority_path),
        "protocol_sha256": file_sha256(protocol_path),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "parameter_count": parameter_count,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
            "gradient_clip_l2": 1.0,
            "warmup_updates": 128,
            "cosine_final_learning_rate": 3.0e-5,
        },
        "training": {
            "seed": SEED,
            "update_count": UPDATE_COUNT,
            "episodes_per_update": EPISODES_PER_UPDATE,
            "raw_correct_per_update": 2,
            "raw_wrong_per_update": 2,
            "pair_count": total_pair_count,
            "correct_pool_count": len(correct),
            "wrong_pool_count": len(wrong),
            "schedule_sha256": schedule_digest.hexdigest(),
            "query_tile_rows": QUERY_TILE_ROWS,
            "reference_tile_rows": REFERENCE_TILE_ROWS,
            "loss_trace": loss_trace,
        },
        "bag_permutation": {
            "enabled": args.arm == "RCDE_BAG",
            "valid_tokens_only": True,
            "fixed_point_free_cyclic_shift": True,
            "query_binding_namespace": BAG_QUERY_NAMESPACE,
            "reference_binding_namespace": BAG_REFERENCE_NAMESPACE,
        },
        "runtime_audit": {
            "elapsed_seconds": time.monotonic() - started,
            "cuda_device": torch.cuda.get_device_name(device),
            "resident_cpu_cache_item_count": len(resident),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "protected_access_counts": {
            "C8_runtime_read_count": 0,
            "opened_runtime_read_count": 0,
            "sealed_runtime_read_count": 0,
            "unauthorized_natural_result_read_count": 0,
            "home_files_modified": 0,
        },
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_required_stage": "R1_MAIN_TRAINING_VALIDATION",
    }
    atomic_json(output_dir / "train_result.json", result)


if __name__ == "__main__":
    main()
