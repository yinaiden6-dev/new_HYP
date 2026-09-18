"""Frozen fresh-D1 primitives for Route A V3.1 M1.

This module deliberately contains no dataset paths, Slurm entry point, or
automatic stage transition.  It provides the small, independently testable
pieces that the later M1 runner needs:

* the inherited expanded-D1 adapter, loss, optimizer, and deterministic recipe;
* the fixed 2 outcome + 1 difficult + 1 new-difficult-train schedule;
* target-free, legal-gallery, label-unique top-63 negative mining;
* immutable checkpoint/state receipts; and
* strict exact-label full-gallery evaluation (including duplicate row labels).

Targets enter only after target-free scores have been produced: to exclude the
paired label from mined negatives, to form the D1 loss, or to compute offline
evaluation diagnostics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch import nn

if __package__:  # package import used by formal Route A runners
    from .domain_alignment import build_query_domain_adapter
    from .domain_safe import DomainSafeTrainingLoss, domain_safe_training_loss
else:  # standalone import used by the existing O1 test harness
    _CORE_ROOT = Path(__file__).resolve().parents[1]
    if str(_CORE_ROOT) not in sys.path:
        sys.path.insert(0, str(_CORE_ROOT))
    from route_a.domain_alignment import build_query_domain_adapter
    from route_a.domain_safe import DomainSafeTrainingLoss, domain_safe_training_loss


VERSION = "route_a_v3_1_m1_fresh_d1_core_v1"
CHECKPOINT_STATUS = "ROUTE_A_V3_1_M1_FRESH_D1_CHECKPOINT_READY"
CLAIM_LEVEL = "IMPLEMENTATION_ONLY_NO_SCIENTIFIC_CLAIM"

SEED = 17
FOLD = 0
FOLD_SEED_STRIDE = 1009
STEPS = 800
BATCH_SIZE = 4
TRACK_ORDER = (
    "outcome",
    "outcome",
    "difficult",
    "new_difficult_train",
)
TRACK_COUNTS = {
    "outcome": 2,
    "difficult": 1,
    "new_difficult_train": 1,
}

DIM = 128
HIDDEN = 128
MAX_RATIO = 0.25
ADAPTER_PARAMETER_COUNT = 49_792
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0

TEMPERATURE = 0.05
PRESERVE_CAP = 0.03
ERROR_MARGIN = 0.02
PRESERVATION_WEIGHT = 10.0
CORRECTION_WEIGHT = 2.0
DRIFT_WEIGHT = 0.05

NEGATIVE_COUNT = 63
CANDIDATE_COUNT = NEGATIVE_COUNT + 1
EXPECTED_FULL_GALLERY_ROWS = 5_413
EXPECTED_FULL_GALLERY_LABELS = 5_404


@dataclass(frozen=True)
class ExactLabelScores:
    """Max-over-row exact-label scores in deterministic label order."""

    label_values: tuple[Hashable, ...]
    label_scores: torch.Tensor
    winning_positions: torch.Tensor
    winning_row_ids: torch.Tensor


@dataclass(frozen=True)
class ExactLabelDiagnostic:
    target_label: Hashable
    target_score: float
    best_rival_label: Hashable
    best_rival_score: float
    margin: float
    rank: int


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using the repository's canonical form."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def effective_seed(*, seed: int = SEED, fold: int = FOLD) -> int:
    """Return the frozen M1 screen seed and reject a silent fold/seed scan."""

    if type(seed) is not int or type(fold) is not int:
        raise TypeError("seed and fold must be integers")
    if seed != SEED or fold != FOLD:
        raise ValueError("M1 fresh-D1 is frozen to seed 17 and fold 0")
    return seed + FOLD_SEED_STRIDE * fold


def frozen_recipe() -> dict[str, Any]:
    """Return the complete fixed M1 D1 recipe as a JSON-compatible object."""

    return {
        "version": VERSION,
        "seed": SEED,
        "fold": FOLD,
        "effective_seed": effective_seed(),
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "track_order": list(TRACK_ORDER),
        "track_counts_per_step": dict(TRACK_COUNTS),
        "sampling": "identity_uniform_without_replacement_within_track_step",
        "candidate_count": CANDIDATE_COUNT,
        "negative_count": NEGATIVE_COUNT,
        "negative_mining": (
            "target_free_scores_then_label_complete_legal_mask_then_"
            "paired_label_exclusion_then_stable_label_unique_top63"
        ),
        "adapter": {
            "kind": "pointwise_query_only",
            "dim": DIM,
            "hidden": HIDDEN,
            "max_ratio": MAX_RATIO,
            "parameter_count": ADAPTER_PARAMETER_COUNT,
            "zero_initialized_delta_head": True,
        },
        "optimizer": {
            "kind": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRAD_CLIP_NORM,
        },
        "loss": {
            "temperature": TEMPERATURE,
            "preserve_cap": PRESERVE_CAP,
            "error_margin": ERROR_MARGIN,
            "preservation_weight": PRESERVATION_WEIGHT,
            "correction_weight": CORRECTION_WEIGHT,
            "drift_weight": DRIFT_WEIGHT,
        },
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": ":4096:8",
        },
        "checkpoint_selection": "final_step_only_no_holdout_selection",
        "inference_identity_specific_parameters": False,
    }


def configure_deterministic_runtime() -> None:
    """Configure the inherited deterministic runtime before CUDA work starts."""

    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current not in {None, ":4096:8"}:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG conflicts with frozen D1")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_fresh_d1_adapter() -> nn.Module:
    """Build the exact fresh pointwise query adapter inherited from D1."""

    adapter = build_query_domain_adapter(
        "pointwise",
        dim=DIM,
        hidden=HIDDEN,
        max_ratio=MAX_RATIO,
    )
    count = sum(parameter.numel() for parameter in adapter.parameters())
    if count != ADAPTER_PARAMETER_COUNT:
        raise AssertionError("fresh-D1 adapter parameter count drift")
    return adapter


def build_fresh_d1_optimizer(adapter: nn.Module) -> torch.optim.AdamW:
    """Build the sole authorized optimizer for the M1 fresh-D1 screen."""

    count = sum(parameter.numel() for parameter in adapter.parameters())
    if count != ADAPTER_PARAMETER_COUNT:
        raise ValueError("optimizer received a non-frozen D1 adapter")
    return torch.optim.AdamW(
        adapter.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def fresh_d1_loss(
    adapter: nn.Module,
    raw_image_tokens: torch.Tensor,
    template_tokens: torch.Tensor,
    references: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
) -> DomainSafeTrainingLoss:
    """Apply the frozen expanded-D1 Domain-Safe objective without overrides."""

    return domain_safe_training_loss(
        adapter,
        raw_image_tokens,
        template_tokens,
        references,
        reference_mask,
        grid_h=grid_h,
        grid_w=grid_w,
        temperature=TEMPERATURE,
        preserve_cap=PRESERVE_CAP,
        error_margin=ERROR_MARGIN,
        preservation_weight=PRESERVATION_WEIGHT,
        correction_weight=CORRECTION_WEIGHT,
        drift_weight=DRIFT_WEIGHT,
    )


def fresh_d1_optimizer_step(
    loss: torch.Tensor,
    *,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Run one fixed, finite, clipped update after a caller forms a batch loss."""

    if loss.ndim != 0 or not bool(torch.isfinite(loss.detach())):
        raise ValueError("fresh-D1 batch loss must be one finite scalar")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in adapter.parameters()
        if parameter.grad is not None
    ]
    if not gradients or any(not bool(torch.isfinite(grad).all()) for grad in gradients):
        raise RuntimeError("fresh-D1 gradients are missing or non-finite")
    grad_l1 = float(sum(grad.detach().abs().sum() for grad in gradients))
    if not grad_l1 > 0.0:
        raise RuntimeError("no gradient reached the fresh-D1 adapter")
    unclipped_norm = torch.nn.utils.clip_grad_norm_(
        adapter.parameters(), GRAD_CLIP_NORM
    )
    if not bool(torch.isfinite(unclipped_norm)):
        raise RuntimeError("fresh-D1 gradient norm is non-finite")
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "gradient_l1": grad_l1,
        "unclipped_gradient_norm": float(unclipped_norm),
    }


def _require_schedule_population(
    rows_by_track: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    identity_key: str,
    row_key: str,
) -> dict[str, dict[str, tuple[str, ...]]]:
    if set(rows_by_track) != set(TRACK_COUNTS):
        raise ValueError("schedule requires exactly outcome/difficult/new_difficult_train")
    grouped: dict[str, dict[str, tuple[str, ...]]] = {}
    globally_seen_rows: set[str] = set()
    for track in TRACK_COUNTS:
        identity_rows: dict[str, list[str]] = defaultdict(list)
        for row in rows_by_track[track]:
            if not isinstance(row, Mapping):
                raise TypeError("schedule rows must be mappings")
            identity = row.get(identity_key)
            key = row.get(row_key)
            if not isinstance(identity, str) or not identity:
                raise ValueError("every schedule row needs a non-empty string identity")
            if not isinstance(key, str) or not key:
                raise ValueError("every schedule row needs a non-empty string row key")
            if key in globally_seen_rows:
                raise ValueError("schedule row keys must be globally role-disjoint")
            globally_seen_rows.add(key)
            identity_rows[identity].append(key)
        if len(identity_rows) < TRACK_COUNTS[track]:
            raise ValueError(f"track {track} has too few distinct identities")
        if not identity_rows:
            raise ValueError(f"track {track} is empty")
        grouped[track] = {
            identity: tuple(sorted(keys))
            for identity, keys in sorted(identity_rows.items())
        }
    return grouped


def build_three_track_schedule(
    rows_by_track: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    identity_key: str = "identity",
    row_key: str = "path",
    steps: int = STEPS,
    seed: int = SEED,
    fold: int = FOLD,
) -> list[list[dict[str, str]]]:
    """Build the frozen identity-uniform 2+1+1 schedule.

    Input order has no effect: identities and row keys are canonicalized before
    sampling.  Selection is without replacement among identities inside each
    track and step; replacement across steps is intentional.
    """

    if steps != STEPS:
        raise ValueError("formal M1 fresh-D1 schedule must contain 800 steps")
    grouped = _require_schedule_population(
        rows_by_track, identity_key=identity_key, row_key=row_key
    )
    rng = random.Random(effective_seed(seed=seed, fold=fold))
    schedule: list[list[dict[str, str]]] = []
    for _ in range(STEPS):
        selected: dict[str, list[str]] = {}
        for track, count in TRACK_COUNTS.items():
            selected[track] = rng.sample(sorted(grouped[track]), count)
        step: list[dict[str, str]] = []
        consumed: Counter[str] = Counter()
        for track in TRACK_ORDER:
            identity = selected[track][consumed[track]]
            consumed[track] += 1
            key = rng.choice(grouped[track][identity])
            step.append({"track": track, "identity": identity, "row_key": key})
        schedule.append(step)
    schedule_audit(schedule)
    return schedule


def schedule_audit(
    schedule: Sequence[Sequence[Mapping[str, str]]],
) -> dict[str, Any]:
    """Fail-closed validation and content receipt for a formal M1 schedule."""

    if len(schedule) != STEPS:
        raise ValueError("fresh-D1 schedule length drift")
    track_draws: Counter[str] = Counter()
    identity_draws: dict[str, Counter[str]] = {
        track: Counter() for track in TRACK_COUNTS
    }
    query_draws: Counter[tuple[str, str]] = Counter()
    canonical: list[list[dict[str, str]]] = []
    for step in schedule:
        if len(step) != BATCH_SIZE:
            raise ValueError("fresh-D1 step batch size drift")
        normalized: list[dict[str, str]] = []
        for expected_track, item in zip(TRACK_ORDER, step):
            if set(item) != {"track", "identity", "row_key"}:
                raise ValueError("fresh-D1 schedule item schema drift")
            track = item["track"]
            identity = item["identity"]
            key = item["row_key"]
            if track != expected_track or not identity or not key:
                raise ValueError("fresh-D1 schedule order or value drift")
            normalized.append(
                {"track": str(track), "identity": str(identity), "row_key": str(key)}
            )
            track_draws[track] += 1
            identity_draws[track][identity] += 1
            query_draws[(track, key)] += 1
        outcome_ids = [item["identity"] for item in step[:2]]
        if len(set(outcome_ids)) != 2:
            raise ValueError("outcome identities must be distinct within a step")
        canonical.append(normalized)
    expected_draws = {
        track: count * STEPS for track, count in TRACK_COUNTS.items()
    }
    if dict(track_draws) != expected_draws:
        raise ValueError("fresh-D1 per-track draw count drift")
    return {
        "version": VERSION,
        "logical_sha256": canonical_sha256(canonical),
        "step_count": STEPS,
        "batch_size": BATCH_SIZE,
        "track_order": list(TRACK_ORDER),
        "track_draw_counts": dict(sorted(track_draws.items())),
        "identity_draw_counts": {
            track: dict(sorted(counts.items()))
            for track, counts in sorted(identity_draws.items())
        },
        "unique_query_count_drawn": len(query_draws),
        "query_draw_count_min": min(query_draws.values()),
        "query_draw_count_max": max(query_draws.values()),
    }


def _labels_as_tuple(labels: Sequence[Hashable] | torch.Tensor) -> tuple[Hashable, ...]:
    if isinstance(labels, torch.Tensor):
        if labels.ndim != 1:
            raise ValueError("row labels must be one-dimensional")
        return tuple(value.item() for value in labels.detach().cpu())
    if isinstance(labels, (str, bytes)):
        raise TypeError("row labels must be a sequence, not one string")
    result = tuple(labels)
    if any(not isinstance(value, Hashable) for value in result):
        raise TypeError("every row label must be hashable")
    return result


def _boolean_mask(mask: torch.Tensor, *, length: int, device: torch.device) -> torch.Tensor:
    if mask.ndim != 1 or mask.numel() != length or mask.dtype != torch.bool:
        raise ValueError("legal_row_mask must be bool [gallery_rows]")
    return mask.to(device=device)


def validate_label_complete_legal_mask(
    row_labels: Sequence[Hashable] | torch.Tensor,
    legal_row_mask: torch.Tensor,
) -> dict[str, int]:
    """Require every exact label to be wholly legal or wholly excluded."""

    labels = _labels_as_tuple(row_labels)
    mask = _boolean_mask(legal_row_mask, length=len(labels), device=torch.device("cpu"))
    active = mask.detach().cpu().tolist()
    by_label: dict[Hashable, set[bool]] = defaultdict(set)
    for label, keep in zip(labels, active):
        by_label[label].add(bool(keep))
    partial = [label for label, values in by_label.items() if len(values) != 1]
    if partial:
        raise ValueError("legal gallery mask splits rows of an exact label")
    legal_labels = sum(next(iter(values)) for values in by_label.values())
    return {
        "row_count": len(labels),
        "label_count": len(by_label),
        "legal_row_count": int(mask.sum().item()),
        "legal_label_count": int(legal_labels),
        "excluded_row_count": len(labels) - int(mask.sum().item()),
        "excluded_label_count": len(by_label) - int(legal_labels),
    }


def select_legal_label_unique_topk(
    row_scores: torch.Tensor,
    row_labels: Sequence[Hashable] | torch.Tensor,
    legal_row_mask: torch.Tensor,
    *,
    target_label: Hashable,
    negative_count: int = NEGATIVE_COUNT,
) -> torch.Tensor:
    """Select stable natural negatives after a target-free scoring pass.

    Rows are ordered by score descending and then frozen gallery position
    ascending.  The target is used only after this ranking exists, to exclude
    every row carrying the paired exact label.  There is no fallback or fill.
    """

    if row_scores.ndim != 1 or not row_scores.is_floating_point():
        raise ValueError("row_scores must be one floating [gallery_rows] tensor")
    labels = _labels_as_tuple(row_labels)
    if len(labels) != row_scores.numel():
        raise ValueError("row score and label populations differ")
    if not isinstance(target_label, Hashable):
        raise TypeError("target label must be hashable")
    if type(negative_count) is not int or negative_count <= 0:
        raise ValueError("negative_count must be a positive integer")
    mask = _boolean_mask(
        legal_row_mask, length=len(labels), device=row_scores.device
    )
    validate_label_complete_legal_mask(labels, mask.detach().cpu())
    target_positions = [index for index, label in enumerate(labels) if label == target_label]
    if not target_positions or not any(bool(mask[index]) for index in target_positions):
        raise ValueError("paired target label is absent from the legal gallery")
    if not bool(torch.isfinite(row_scores[mask]).all()):
        raise ValueError("legal gallery scores must all be finite")

    source = row_scores.masked_fill(~mask, float("-inf"))
    ranked = torch.argsort(source, descending=True, stable=True).detach().cpu().tolist()
    chosen: list[int] = []
    seen_labels: set[Hashable] = set()
    for position in ranked:
        if not bool(mask[position]):
            continue
        label = labels[position]
        if label == target_label or label in seen_labels:
            continue
        chosen.append(int(position))
        seen_labels.add(label)
        if len(chosen) == negative_count:
            break
    if len(chosen) != negative_count:
        raise ValueError("fewer than the required legal label-unique negatives")
    return torch.tensor(chosen, dtype=torch.long, device=row_scores.device)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash dtype, shape, and exact contiguous tensor bytes."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_sha256 requires a tensor")
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash a state dict with sorted names and exact tensor bytes."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("state dict must map string names to tensors")
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(tensor_sha256(tensor).encode("ascii"))
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    return state_dict_sha256(module.state_dict())


def parameter_manifest(module: nn.Module) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": parameter.numel(),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in module.named_parameters()
    ]


def _require_sha256(value: str, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be one lowercase SHA-256")


def _checkpoint_contract() -> dict[str, bool]:
    return {
        "fresh_fold_local_d1": True,
        "query_only_adapter": True,
        "target_free_negative_scoring": True,
        "paired_target_used_only_after_scoring": True,
        "legal_label_unique_natural_top63": True,
        "fallback_or_gallery_order_fill": False,
        "full_gallery_evaluation_required": True,
        "runner_may_auto_advance_stage": False,
    }


def build_checkpoint_receipt(
    adapter: nn.Module,
    *,
    initial_state_dict_sha256: str,
    schedule_sha256: str,
    split_sha256: str,
    legal_gallery_mask_sha256: str,
    source_hashes: Mapping[str, str],
    optimizer_steps: int = STEPS,
) -> dict[str, Any]:
    """Bind a changed final adapter to its frozen schedule and input closure."""

    for field, value in (
        ("initial_state_dict_sha256", initial_state_dict_sha256),
        ("schedule_sha256", schedule_sha256),
        ("split_sha256", split_sha256),
        ("legal_gallery_mask_sha256", legal_gallery_mask_sha256),
    ):
        _require_sha256(value, field=field)
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("checkpoint receipt requires non-empty source hashes")
    normalized_sources: dict[str, str] = {}
    for role, value in sorted(source_hashes.items()):
        if not isinstance(role, str) or not role:
            raise ValueError("source hash roles must be non-empty strings")
        _require_sha256(value, field=f"source_hashes.{role}")
        normalized_sources[role] = value
    if optimizer_steps != STEPS:
        raise ValueError("M1 fresh-D1 checkpoint requires exactly 800 updates")

    manifest = parameter_manifest(adapter)
    parameter_count = sum(item["numel"] for item in manifest)
    if parameter_count != ADAPTER_PARAMETER_COUNT:
        raise ValueError("checkpoint adapter capacity drift")
    final_hash = module_state_sha256(adapter)
    if final_hash == initial_state_dict_sha256:
        raise ValueError("optimizer did not change the fresh-D1 adapter")
    recipe = frozen_recipe()
    receipt = {
        "version": VERSION,
        "status": CHECKPOINT_STATUS,
        "claim_level": CLAIM_LEVEL,
        "seed": SEED,
        "fold": FOLD,
        "effective_seed": effective_seed(),
        "optimizer_steps": optimizer_steps,
        "checkpoint_selection": "final_step_only_no_holdout_selection",
        "parameter_count": parameter_count,
        "parameter_manifest": manifest,
        "parameter_manifest_sha256": canonical_sha256(manifest),
        "initial_state_dict_sha256": initial_state_dict_sha256,
        "state_dict_sha256": final_hash,
        "recipe": recipe,
        "recipe_sha256": canonical_sha256(recipe),
        "schedule_sha256": schedule_sha256,
        "split_sha256": split_sha256,
        "legal_gallery_mask_sha256": legal_gallery_mask_sha256,
        "source_hashes": normalized_sources,
        "contract": _checkpoint_contract(),
    }
    receipt["logical_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_checkpoint_receipt(
    receipt: Mapping[str, Any], adapter: nn.Module
) -> None:
    """Reject receipt, recipe, manifest, or live model drift."""

    required = {
        "version",
        "status",
        "claim_level",
        "seed",
        "fold",
        "effective_seed",
        "optimizer_steps",
        "checkpoint_selection",
        "parameter_count",
        "parameter_manifest",
        "parameter_manifest_sha256",
        "initial_state_dict_sha256",
        "state_dict_sha256",
        "recipe",
        "recipe_sha256",
        "schedule_sha256",
        "split_sha256",
        "legal_gallery_mask_sha256",
        "source_hashes",
        "contract",
        "logical_sha256",
    }
    if set(receipt) != required:
        raise ValueError("checkpoint receipt schema drift")
    body = {key: value for key, value in receipt.items() if key != "logical_sha256"}
    if (
        receipt["version"] != VERSION
        or receipt["status"] != CHECKPOINT_STATUS
        or receipt["claim_level"] != CLAIM_LEVEL
        or receipt["seed"] != SEED
        or receipt["fold"] != FOLD
        or receipt["effective_seed"] != effective_seed()
        or receipt["optimizer_steps"] != STEPS
        or receipt["checkpoint_selection"]
        != "final_step_only_no_holdout_selection"
        or receipt["recipe"] != frozen_recipe()
        or receipt["recipe_sha256"] != canonical_sha256(frozen_recipe())
        or receipt["parameter_manifest"] != parameter_manifest(adapter)
        or receipt["parameter_manifest_sha256"]
        != canonical_sha256(parameter_manifest(adapter))
        or receipt["parameter_count"] != ADAPTER_PARAMETER_COUNT
        or receipt["state_dict_sha256"] != module_state_sha256(adapter)
        or receipt["contract"] != _checkpoint_contract()
        or receipt["logical_sha256"] != canonical_sha256(body)
    ):
        raise ValueError("checkpoint receipt or live state drift")
    for field in (
        "initial_state_dict_sha256",
        "state_dict_sha256",
        "schedule_sha256",
        "split_sha256",
        "legal_gallery_mask_sha256",
    ):
        _require_sha256(receipt[field], field=field)
    if receipt["initial_state_dict_sha256"] == receipt["state_dict_sha256"]:
        raise ValueError("checkpoint receipt records an unchanged adapter")
    sources = receipt["source_hashes"]
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("checkpoint receipt source closure is empty")
    for role, value in sources.items():
        if not isinstance(role, str) or not role:
            raise ValueError("checkpoint source role drift")
        _require_sha256(value, field=f"source_hashes.{role}")


def build_checkpoint_payload(
    adapter: nn.Module, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a primitive/tensor-only artifact safe for weights-only loading."""

    validate_checkpoint_receipt(receipt, adapter)
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in sorted(adapter.state_dict().items())
    }
    return {
        "version": VERSION,
        "status": CHECKPOINT_STATUS,
        "state_dict": state,
        "state_dict_sha256": state_dict_sha256(state),
        "receipt": dict(receipt),
    }


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    """Validate a checkpoint loaded with ``torch.load(weights_only=True)``."""

    if set(payload) != {
        "version",
        "status",
        "state_dict",
        "state_dict_sha256",
        "receipt",
    }:
        raise ValueError("checkpoint payload schema drift")
    state = payload["state_dict"]
    receipt = payload["receipt"]
    if (
        payload["version"] != VERSION
        or payload["status"] != CHECKPOINT_STATUS
        or not isinstance(state, Mapping)
        or not state
        or any(
            not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
            for name, tensor in state.items()
        )
        or not isinstance(receipt, Mapping)
    ):
        raise ValueError("checkpoint payload content drift")
    actual_hash = state_dict_sha256(state)
    if (
        payload["state_dict_sha256"] != actual_hash
        or receipt.get("state_dict_sha256") != actual_hash
    ):
        raise ValueError("checkpoint payload state hash drift")
    adapter = build_fresh_d1_adapter()
    try:
        adapter.load_state_dict(state, strict=True)
    except (KeyError, RuntimeError, TypeError) as error:
        raise ValueError("checkpoint payload state schema drift") from error
    validate_checkpoint_receipt(receipt, adapter)


def validate_full_gallery_population(
    row_labels: Sequence[Hashable] | torch.Tensor,
    *,
    expected_rows: int = EXPECTED_FULL_GALLERY_ROWS,
    expected_labels: int = EXPECTED_FULL_GALLERY_LABELS,
) -> dict[str, int]:
    """Enforce the registered 5,413-row / 5,404-label evaluation universe."""

    labels = _labels_as_tuple(row_labels)
    if len(labels) != expected_rows or len(set(labels)) != expected_labels:
        raise ValueError("full-gallery row or exact-label population drift")
    return {"row_count": len(labels), "label_count": len(set(labels))}


def _label_sort_key(label: Hashable) -> tuple[str, str]:
    return type(label).__qualname__, repr(label)


def aggregate_exact_label_scores(
    row_scores: torch.Tensor,
    row_labels: Sequence[Hashable] | torch.Tensor,
    *,
    legal_row_mask: torch.Tensor | None = None,
    canonical_row_ids: torch.Tensor | None = None,
) -> ExactLabelScores:
    """Max-pool rows into exact labels, resolving ties by canonical row ID."""

    if row_scores.ndim != 1 or not row_scores.is_floating_point():
        raise ValueError("row_scores must be one floating [gallery_rows] tensor")
    labels = _labels_as_tuple(row_labels)
    rows = row_scores.numel()
    if len(labels) != rows:
        raise ValueError("row score and label populations differ")
    if legal_row_mask is None:
        mask = torch.ones(rows, dtype=torch.bool, device=row_scores.device)
    else:
        mask = _boolean_mask(legal_row_mask, length=rows, device=row_scores.device)
        validate_label_complete_legal_mask(labels, mask.detach().cpu())
    if not bool(mask.any()) or not bool(torch.isfinite(row_scores[mask]).all()):
        raise ValueError("active evaluation scores must be non-empty and finite")
    if canonical_row_ids is None:
        row_ids = torch.arange(rows, dtype=torch.long, device=row_scores.device)
    else:
        if (
            canonical_row_ids.ndim != 1
            or canonical_row_ids.numel() != rows
            or canonical_row_ids.dtype != torch.long
            or torch.unique(canonical_row_ids).numel() != rows
        ):
            raise ValueError("canonical_row_ids must be unique int64 [gallery_rows]")
        row_ids = canonical_row_ids.to(row_scores.device)

    positions_by_label: dict[Hashable, list[int]] = defaultdict(list)
    active_cpu = mask.detach().cpu().tolist()
    for position, (label, active) in enumerate(zip(labels, active_cpu)):
        if active:
            positions_by_label[label].append(position)
    ordered_labels = tuple(sorted(positions_by_label, key=_label_sort_key))
    pooled_scores: list[torch.Tensor] = []
    winning_positions: list[torch.Tensor] = []
    winning_ids: list[torch.Tensor] = []
    for label in ordered_labels:
        positions = torch.tensor(
            positions_by_label[label], dtype=torch.long, device=row_scores.device
        )
        values = row_scores.index_select(0, positions)
        best_value = values.max()
        tied_positions = positions[values == best_value]
        tied_ids = row_ids.index_select(0, tied_positions)
        tie_choice = tied_ids.argmin()
        winner_position = tied_positions[tie_choice]
        pooled_scores.append(best_value)
        winning_positions.append(winner_position)
        winning_ids.append(row_ids[winner_position])
    return ExactLabelScores(
        label_values=ordered_labels,
        label_scores=torch.stack(pooled_scores),
        winning_positions=torch.stack(winning_positions),
        winning_row_ids=torch.stack(winning_ids),
    )


def aggregate_full_gallery_exact_label_scores(
    row_scores: torch.Tensor,
    row_labels: Sequence[Hashable] | torch.Tensor,
    *,
    canonical_row_ids: torch.Tensor | None = None,
) -> ExactLabelScores:
    """Validate and aggregate the unmasked 5,413-row evaluation gallery."""

    validate_full_gallery_population(row_labels)
    return aggregate_exact_label_scores(
        row_scores,
        row_labels,
        canonical_row_ids=canonical_row_ids,
    )


def strict_exact_label_rank(
    output: ExactLabelScores, target_label: Hashable
) -> int:
    """Pessimistic one-based exact-label rank; every non-target tie is ahead."""

    matches = [index for index, label in enumerate(output.label_values) if label == target_label]
    if len(matches) != 1:
        raise ValueError("target label must occur exactly once in evaluated labels")
    target_index = matches[0]
    target_score = output.label_scores[target_index]
    other = torch.ones(
        output.label_scores.numel(), dtype=torch.bool, device=output.label_scores.device
    )
    other[target_index] = False
    return 1 + int((output.label_scores[other] >= target_score).sum().item())


def exact_label_diagnostic(
    output: ExactLabelScores, target_label: Hashable
) -> ExactLabelDiagnostic:
    """Return target/rival score, margin, and strict rank for one query."""

    matches = [index for index, label in enumerate(output.label_values) if label == target_label]
    if len(matches) != 1 or len(output.label_values) < 2:
        raise ValueError("diagnostic needs one target and at least one rival label")
    target_index = matches[0]
    rival_indices = [
        index for index in range(len(output.label_values)) if index != target_index
    ]
    # label_values are canonical, so torch.argmax resolves score ties to the
    # lowest canonical exact label.
    rival_scores = output.label_scores[rival_indices]
    relative = int(torch.argmax(rival_scores).item())
    rival_index = rival_indices[relative]
    target_score = float(output.label_scores[target_index].detach())
    rival_score = float(output.label_scores[rival_index].detach())
    return ExactLabelDiagnostic(
        target_label=target_label,
        target_score=target_score,
        best_rival_label=output.label_values[rival_index],
        best_rival_score=rival_score,
        margin=target_score - rival_score,
        rank=strict_exact_label_rank(output, target_label),
    )


def exact_label_metrics(ranks: Sequence[int]) -> dict[str, float | int]:
    """Compute the registered strict R@1 and MRR summaries."""

    if not ranks or any(type(rank) is not int or rank < 1 for rank in ranks):
        raise ValueError("ranks must be a non-empty sequence of positive integers")
    count = len(ranks)
    return {
        "query_count": count,
        "R@1": sum(rank == 1 for rank in ranks) / count,
        "MRR": sum(1.0 / rank for rank in ranks) / count,
    }


def rescue_break_summary(
    baseline_ranks: Sequence[int], proposed_ranks: Sequence[int]
) -> dict[str, int]:
    """Count top-1 rescues, breaks, retained wins, and retained errors."""

    if (
        not baseline_ranks
        or len(baseline_ranks) != len(proposed_ranks)
        or any(type(rank) is not int or rank < 1 for rank in baseline_ranks)
        or any(type(rank) is not int or rank < 1 for rank in proposed_ranks)
    ):
        raise ValueError("rank ledgers must be equal non-empty positive-int sequences")
    baseline_correct = [rank == 1 for rank in baseline_ranks]
    proposed_correct = [rank == 1 for rank in proposed_ranks]
    rescues = sum(not before and after for before, after in zip(baseline_correct, proposed_correct))
    breaks = sum(before and not after for before, after in zip(baseline_correct, proposed_correct))
    return {
        "query_count": len(baseline_ranks),
        "rescues": rescues,
        "breaks": breaks,
        "net": rescues - breaks,
        "retained_correct": sum(before and after for before, after in zip(baseline_correct, proposed_correct)),
        "retained_wrong": sum(not before and not after for before, after in zip(baseline_correct, proposed_correct)),
    }


__all__ = [
    "ADAPTER_PARAMETER_COUNT",
    "BATCH_SIZE",
    "CANDIDATE_COUNT",
    "CHECKPOINT_STATUS",
    "CLAIM_LEVEL",
    "ExactLabelDiagnostic",
    "ExactLabelScores",
    "EXPECTED_FULL_GALLERY_LABELS",
    "EXPECTED_FULL_GALLERY_ROWS",
    "FOLD",
    "NEGATIVE_COUNT",
    "SEED",
    "STEPS",
    "TRACK_COUNTS",
    "TRACK_ORDER",
    "VERSION",
    "aggregate_exact_label_scores",
    "aggregate_full_gallery_exact_label_scores",
    "build_checkpoint_receipt",
    "build_checkpoint_payload",
    "build_fresh_d1_adapter",
    "build_fresh_d1_optimizer",
    "build_three_track_schedule",
    "canonical_sha256",
    "configure_deterministic_runtime",
    "effective_seed",
    "exact_label_diagnostic",
    "exact_label_metrics",
    "fresh_d1_loss",
    "fresh_d1_optimizer_step",
    "frozen_recipe",
    "module_state_sha256",
    "parameter_manifest",
    "rescue_break_summary",
    "schedule_audit",
    "select_legal_label_unique_topk",
    "state_dict_sha256",
    "strict_exact_label_rank",
    "tensor_sha256",
    "validate_checkpoint_receipt",
    "validate_checkpoint_payload",
    "validate_full_gallery_population",
    "validate_label_complete_legal_mask",
]
