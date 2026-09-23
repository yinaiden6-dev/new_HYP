from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import numpy as np
import torch

import rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 as v_runtime

from rc_aslo_xf.dino_rcde_cw1_multitile_superregion_v2 import (
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
)
from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (
    CandidatePLockV1,
    PAIR_COUNT,
    SealedDirectionPLockV1,
    SealedRootDinoScopeV1,
    SR0MTVContractError,
    assert_target_free_payload,
    canonical_unordered_pairs,
    decode_pair,
    decode_pair_training_selective_autograd,
    episode_schedule_key,
    ordered_training_pool,
    serialize_arm_evidence,
    target_first,
    train_updates,
    update_episodes,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (
    LOCK_READY,
    ROOT_READY,
    tensor_sha256 as p_tensor_sha256,
)
from rc_aslo_xf.dino_rcde_v1_2_resource_core import PairEvidence

from test_dino_rcde_cw1_multitile_vdecode_v1 import _inputs

PROGRAM_ROOT = Path(__file__).resolve().parents[1] / "programs"
sys.path.insert(0, str(PROGRAM_ROOT))
import materialize_dino_rcde_sr0_mt_prejoin_v1 as prejoin_aggregator  # noqa: E402
import materialize_dino_rcde_sr0_mt_postjoin_v1 as postjoin_materializer  # noqa: E402
import run_dino_rcde_sr0_mt_three_arm_v1 as three_arm_runner  # noqa: E402


class _SharedSyntheticV(torch.nn.Module):
    """One learned object used by all three synthetic arm decodes."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def decode_candidate(
        self,
        query_layers: torch.Tensor,
        reference_layers: torch.Tensor,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        **_: object,
    ) -> SimpleNamespace:
        qmask = torch.as_tensor(query_mask, dtype=torch.bool).flatten().to(query_layers.device)
        rmask = torch.as_tensor(reference_mask, dtype=torch.bool).flatten().to(reference_layers.device)
        assert qmask.shape == (math.prod(query_grid),)
        assert rmask.shape == (math.prod(reference_grid),)
        query = query_layers[0, :, 0]
        reference_mean = reference_layers[0, rmask, 0].mean()
        active = qmask.to(query.dtype)
        relational = torch.stack(
            (query * reference_mean, query.square() * reference_mean), dim=1
        ) * active[:, None] * self.scale
        return SimpleNamespace(relational=relational)

    def compare_relational(
        self,
        relational_g: torch.Tensor,
        relational_c: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> PairEvidence:
        mask = torch.as_tensor(query_mask, dtype=torch.bool).flatten().to(relational_g.device)
        signed = (relational_g[:, 0] - relational_c[:, 0]) * mask.to(relational_g.dtype)
        logit = signed.sum() / mask.sum().to(signed.dtype)
        return PairEvidence(
            signed_support=signed,
            modulation=mask.to(signed.dtype),
            evidence=signed,
            contributions=signed,
            logit=logit,
        )


def _locks():
    query, candidates, scopes = _inputs()
    locks = {}
    for candidate_key, candidate in candidates.items():
        direction_locks = {}
        for direction in ("a_to_b", "b_to_a"):
            scope = scopes[(candidate_key, direction)]
            population = scope.population
            row = scope.row
            all_roots = {
                binding.root_ordinal: SealedRootDinoScopeV1(
                    root_ordinal=binding.root_ordinal,
                    action_key_sha256=f"{binding.root_ordinal + 1:064x}",
                    query_mask=binding.dino_query_tile_mask,
                    reference_mask=binding.dino_reference_component_mask,
                    query_grid_shape=query.grid_shape,
                    reference_grid_shape=candidate.grid_shape,
                    query_mask_p_sha256=p_tensor_sha256(
                        binding.dino_query_tile_mask
                    ),
                    reference_mask_p_sha256=p_tensor_sha256(
                        binding.dino_reference_component_mask
                    ),
                    query_geometry_sha256=query.geometry_record_sha256,
                    reference_geometry_sha256=candidate.geometry_record_sha256,
                    binding_status=ROOT_READY,
                )
                for binding in population.root_bindings
            }
            direction_locks[direction] = SealedDirectionPLockV1(
                candidate_key=candidate_key,
                candidate_physical_row=candidate.physical_gallery_row,
                query_source_image_sha256=query.source_image_sha256,
                candidate_reference_source_sha256=candidate.source_image_sha256,
                direction=direction,
                status=LOCK_READY,
                selected_bank_ordinal=scope.family.bank_ordinal,
                selected_row_sha256=row.row_sha256,
                query_union_mask=row.dino_query_union_mask,
                ordered_root_ordinals=row.structural_region.contributing_root_ordinals,
                all_roots=all_roots,
                query_grid_shape=query.grid_shape,
                reference_grid_shape=candidate.grid_shape,
                query_geometry_sha256=query.geometry_record_sha256,
                reference_geometry_sha256=candidate.geometry_record_sha256,
                p_lock_record_sha256=(
                    ("8" if candidate_key == "candidate-g" else "6")
                    if direction == "a_to_b"
                    else ("9" if candidate_key == "candidate-g" else "7")
                )
                * 64,
            )
        locks[candidate_key] = CandidatePLockV1(
            candidate=candidate,
            direction_locks=direction_locks,
        )
    return query, locks


def test_runtime_redecodes_all_three_arms_with_one_lineage_and_shared_query_mask():
    query, locks = _locks()
    model = _SharedSyntheticV()
    result = decode_pair(model, query, locks, "candidate-g", "candidate-c")
    serialized = serialize_arm_evidence(result)
    assert result.model_checkpoint_sha256
    assert len({result.model_checkpoint_sha256 for _ in result.arms}) == 1
    full = serialized["arms"][ARM_QUERY_FULL_REFERENCE]
    paired = serialized["arms"][ARM_QUERY_LOCAL_COMPONENTS]
    assert full["forward_query_mask_sha256"] == paired["forward_query_mask_sha256"]
    assert full["reverse_query_mask_sha256"] == paired["reverse_query_mask_sha256"]
    swapped = decode_pair(model, query, locks, "candidate-c", "candidate-g")
    for name, first in result.by_name().items():
        assert torch.allclose(first.logit, -swapped.by_name()[name].logit, atol=1e-12, rtol=0)


def test_paired_direct_lock_addresses_owner_roots_when_opponent_selected_another_row():
    query, locks = _locks()
    rival = locks["candidate-c"]
    changed = {}
    for direction, sealed in rival.direction_locks.items():
        roots = (1, 2, 3)
        union = torch.stack([sealed.all_roots[root].query_mask for root in roots]).any(dim=0)
        changed[direction] = replace(
            sealed,
            selected_bank_ordinal=1,
            selected_row_sha256="5" * 64,
            ordered_root_ordinals=roots,
            query_union_mask=union,
            p_lock_record_sha256="4" * 64,
        )
    altered = dict(locks)
    altered["candidate-c"] = replace(rival, direction_locks=changed)
    result = decode_pair(
        _SharedSyntheticV(), query, altered, "candidate-g", "candidate-c"
    )
    paired = result.by_name()[ARM_QUERY_LOCAL_COMPONENTS]
    # The left/owner row contains root 0 although the rival selected row does
    # not.  Its forward term must still read the rival's sealed all-root scope
    # at root 0; selecting/falling back to the rival row would omit it.
    for term in paired.forward_by_direction:
        assert 0 in term.decoded_root_ordinals
        receipt = next(item for item in term.decode_receipts if item.root_ordinal == 0)
        assert receipt.opponent_component_source_root_ordinal == 0


def test_unordered_pair_enumeration_is_canonical_and_full_c128_is_8128():
    fake = {
        f"candidate-{index:03d}": SimpleNamespace(
            candidate=SimpleNamespace(
                candidate_key=f"candidate-{index:03d}",
                physical_gallery_row=2000 - index,
                source_image_sha256=f"{index:064x}",
            )
        )
        for index in range(128)
    }
    pairs = canonical_unordered_pairs(fake)  # type: ignore[arg-type]
    assert len(pairs) == PAIR_COUNT == 8128
    assert [item[0] for item in pairs] == list(range(PAIR_COUNT))
    assert all(left != right for _, left, right in pairs)


def test_target_free_prejoin_rejects_label_target_rival_and_d1_fields():
    for field in ("target", "identity", "rival", "D1_score", "winner", "slot"):
        with pytest.raises(SR0MTVContractError, match="forbidden target-free field"):
            assert_target_free_payload({"nested": [{field: "leak"}]})


def test_pair_projection_is_byte_schema_compatible_with_frozen_prejoin_aggregator():
    scientific_arms = (
        "ALL_PATCH_SAME_MODEL",
        "QUERY_REGION_FULL_REFERENCE_SAME_MODEL",
        "PAIRED_QUERY_REFERENCE_REGION",
    )
    runtime_arm = {
        "logit": 0.25,
        "scalar_contributions": torch.tensor([0.1, 0.15], dtype=torch.float64),
        "forward_query_mask_sha256": ["a" * 64, "b" * 64],
        "reverse_query_mask_sha256": ["c" * 64, "d" * 64],
        "reference_scope_sha256": "e" * 64,
        "dual_candidate_legal_h1": True,
    }
    serialized = {"arms": {name: dict(runtime_arm) for name in scientific_arms}}
    controls = {
        arm: {
            name: {
                "status": "READY",
                "margin": torch.tensor(0.25, dtype=torch.float64),
                "scalar_contributions": torch.tensor([0.1, 0.15], dtype=torch.float64),
                "only_registered_variable_changed": True,
            }
            for name in prejoin_aggregator.CONTROLS
        }
        for arm in scientific_arms
    }
    physical_rows = list(range(1000, 1128))
    candidate_axis_sha = prejoin_aggregator.canonical_sha256(physical_rows)
    bundle = SimpleNamespace(
        execution_ordinal=0,
        query_id="q0",
        candidate_axis_sha256=candidate_axis_sha,
        locks={
            "left": SimpleNamespace(candidate=SimpleNamespace(physical_gallery_row=1000)),
            "right": SimpleNamespace(candidate=SimpleNamespace(physical_gallery_row=1001)),
        },
    )
    payload = {}
    record = three_arm_runner.project_pair_record(
        serialized=serialized,
        control_record=controls,
        bundle=bundle,
        pair_ordinal=0,
        left_key="left",
        right_key="right",
        axis_position={"left": 0, "right": 1},
        contribution_payload=payload,
        candidate_reorder_exact=True,
        pair_swap_exact=True,
    )
    axes = [
        {
            "query_id": "q0" if index == 0 else f"unused-{index}",
            "candidate_axis_sha256": candidate_axis_sha,
            "candidate_physical_rows": physical_rows,
        }
        for index in range(600)
    ]
    assert prejoin_aggregator._validate_record(record, axes) == (0, 0)
    assert len(payload) == 3 * (1 + len(prejoin_aggregator.CONTROLS))


def test_signed_mass_fields_close_under_both_postjoin_orientations():
    contributions = torch.tensor([0.5, -0.25, -0.75, 0.0], dtype=torch.float64)
    masses = three_arm_runner._mass_fields(contributions)
    assert masses["positive_mass"] == 0.5
    assert masses["negative_mass"] == -1.0
    assert masses["max_positive_patch_fraction"] == 1.0
    assert masses["max_negative_patch_fraction"] == 0.75
    assert masses["cancellation"] == pytest.approx(2.0 / 3.0)

    arm = {
        "margin_f64_bits": three_arm_runner.f64_bits(float(contributions.sum())),
        **masses,
        "dual_candidate_legal_h1": True,
        "geometry_contract_pass": True,
        "query_mask_sha256": "a" * 64,
        "reference_scope_sha256": "b" * 64,
        "signed_contribution_payload_sha256": "c" * 64,
    }
    forward = postjoin_materializer.orient_arm(arm, 1.0)
    reverse = postjoin_materializer.orient_arm(arm, -1.0)
    assert forward["positive_mass"] == 0.5
    assert forward["negative_mass"] == -1.0
    assert forward["max_positive_patch_fraction"] == 1.0
    assert reverse["positive_mass"] == 1.0
    assert reverse["negative_mass"] == -0.5
    assert reverse["max_positive_patch_fraction"] == 0.75


@dataclass(frozen=True)
class _TinyQuery:
    source_image_sha256: str
    source_key: str
    cache_payload_sha256: str


@dataclass(frozen=True)
class _TinyEpisode:
    episode_id: str
    outer_fold: int
    query: _TinyQuery
    x: float
    y: float


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=True, dtype=torch.float64)


def _tiny_episodes() -> tuple[_TinyEpisode, ...]:
    return tuple(
        _TinyEpisode(
            episode_id=f"q{index}",
            outer_fold=1,
            query=_TinyQuery(
                source_image_sha256=f"{index:064x}",
                source_key=f"query_execution:{index}",
                cache_payload_sha256=f"{index + 100:064x}",
            ),
            x=float(index + 1),
            y=float((index % 3) - 1),
        )
        for index in range(8)
    )


def _tiny_loss(
    model: _TinyModel, episode: _TinyEpisode, target_first_value: bool
) -> torch.Tensor:
    prediction = model.linear(
        torch.tensor([[episode.x]], dtype=torch.float64)
    ).squeeze()
    # Exercise the schedule bit without changing the target-signed objective.
    assert isinstance(target_first_value, bool)
    return (prediction - episode.y).square()


def test_v_schedule_is_blind_to_episode_target_rival_identity() -> None:
    first = tuple(
        _TinyEpisode(
            episode_id=f"target-rival-A-{index}",
            outer_fold=1,
            query=_TinyQuery(
                source_image_sha256=f"{index:064x}",
                source_key=f"query_execution:{index}",
                cache_payload_sha256=f"{index + 100:064x}",
            ),
            x=float(index),
            y=1.0,
        )
        for index in range(8)
    )
    second = tuple(
        replace(item, episode_id=f"flipped-target-rival-B-{index}", y=-1.0)
        for index, item in enumerate(first)
    )
    first_order = ordered_training_pool(first, 1)  # type: ignore[arg-type]
    second_order = ordered_training_pool(second, 1)  # type: ignore[arg-type]
    first_keys = [episode_schedule_key(item) for item in first_order]  # type: ignore[arg-type]
    second_keys = [episode_schedule_key(item) for item in second_order]  # type: ignore[arg-type]
    assert first_keys == second_keys
    assert [target_first(1, 17, key) for key in first_keys] == [
        target_first(1, 17, key) for key in second_keys
    ]


def _tree_equal(first: object, second: object) -> bool:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, dict) and isinstance(second, dict):
        return set(first) == set(second) and all(_tree_equal(first[key], second[key]) for key in first)
    if isinstance(first, (tuple, list)) and isinstance(second, type(first)):
        return len(first) == len(second) and all(_tree_equal(a, b) for a, b in zip(first, second))
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        return np.array_equal(first, second)
    return first == second


def test_training_selective_autograd_preserves_all_three_arm_outputs_and_local_gradient():
    query, locks = _locks()
    full_model = _SharedSyntheticV()
    selective_model = _SharedSyntheticV()
    selective_model.load_state_dict(full_model.state_dict())

    full = decode_pair(
        full_model, query, locks, "candidate-g", "candidate-c"
    )
    selective = decode_pair_training_selective_autograd(
        selective_model, query, locks, "candidate-g", "candidate-c"
    )
    assert _tree_equal(
        serialize_arm_evidence(full), serialize_arm_evidence(selective)
    )
    assert tuple(full.by_name()) == tuple(selective.by_name())
    for arm_name in (ARM_ALL_PATCH, ARM_QUERY_FULL_REFERENCE):
        assert full.by_name()[arm_name].logit.requires_grad
        assert not selective.by_name()[arm_name].logit.requires_grad
        assert not selective.by_name()[arm_name].scalar_contributions.requires_grad
    assert full.by_name()[ARM_QUERY_LOCAL_COMPONENTS].logit.requires_grad
    assert selective.by_name()[ARM_QUERY_LOCAL_COMPONENTS].logit.requires_grad

    full.by_name()[ARM_QUERY_LOCAL_COMPONENTS].logit.backward()
    selective.by_name()[ARM_QUERY_LOCAL_COMPONENTS].logit.backward()
    torch.testing.assert_close(
        selective_model.scale.grad,
        full_model.scale.grad,
        rtol=0.0,
        atol=0.0,
    )


def test_streamed_four_query_update_matches_batched_mean_objective() -> None:
    """Streaming may change only floating reduction order, not the objective."""

    episodes = _tiny_episodes()
    torch.manual_seed(9127)
    base = _TinyModel().state_dict()

    reference = _TinyModel()
    reference.load_state_dict(base)
    reference_optimizer = torch.optim.AdamW(
        reference.parameters(),
        lr=v_runtime.MAX_LR,
        weight_decay=v_runtime.WEIGHT_DECAY,
    )
    ordered = ordered_training_pool(episodes, 1)  # type: ignore[arg-type]
    selected = update_episodes(ordered, 0)
    directions = tuple(
        target_first(1, 1, episode_schedule_key(item)) for item in selected
    )
    reference_optimizer.param_groups[0]["lr"] = v_runtime.learning_rate(1)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_losses = [
        _tiny_loss(reference, item, direction)
        for item, direction in zip(selected, directions, strict=True)
    ]
    reference_mean = torch.stack(reference_losses).mean()
    reference_mean.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), v_runtime.GRADIENT_CLIP_L2)
    reference_optimizer.step()

    streamed = _TinyModel()
    streamed.load_state_dict(base)
    state, streamed_optimizer = train_updates(
        streamed,
        episodes,  # type: ignore[arg-type]
        outer_fold=1,
        initialization_checkpoint_sha256="a" * 64,
        stop_after_updates=1,
        loss_fn=_tiny_loss,  # type: ignore[arg-type]
    )
    assert state["completed_updates"] == 1
    assert state["loss_trace"][0]["query_schedule_keys"] == [
        episode_schedule_key(item) for item in selected
    ]
    assert state["loss_trace"][0]["mean_pair_loss"] == pytest.approx(
        float(reference_mean.detach()), rel=1.0e-14, abs=1.0e-14
    )
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(
            streamed.state_dict()[name], expected, rtol=1.0e-12, atol=1.0e-12
        )
    for expected, observed in zip(
        reference_optimizer.state_dict()["state"].values(),
        streamed_optimizer.state_dict()["state"].values(),
        strict=True,
    ):
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                observed[key], expected[key], rtol=1.0e-12, atol=1.0e-12
            )


def test_natural_update_moves_only_four_selected_episodes(monkeypatch) -> None:
    episodes = _tiny_episodes()
    moved: list[str] = []

    def identity_move(item: _TinyEpisode, _device: torch.device) -> _TinyEpisode:
        moved.append(item.episode_id)
        return item

    def signed_logit(
        model: _TinyModel, item: _TinyEpisode, *, target_first: bool
    ) -> torch.Tensor:
        assert isinstance(target_first, bool)
        return model.linear(
            torch.tensor([[item.x]], dtype=torch.float64)
        ).squeeze() - item.y

    monkeypatch.setattr(v_runtime, "move_episode_to_device", identity_move)
    monkeypatch.setattr(v_runtime, "target_signed_paired_logit", signed_logit)
    model = _TinyModel()
    train_updates(
        model,
        episodes,  # type: ignore[arg-type]
        outer_fold=1,
        initialization_checkpoint_sha256="a" * 64,
        stop_after_updates=1,
    )
    selected = update_episodes(ordered_training_pool(episodes, 1), 0)  # type: ignore[arg-type]
    assert moved == [item.episode_id for item in selected]
    assert len(moved) == v_runtime.QUERIES_PER_UPDATE


def test_fresh_and_resumed_v_updates_are_bit_exact_including_optimizer_and_cursor():
    episodes = tuple(
        _TinyEpisode(
            episode_id=f"q{index}",
            outer_fold=1,
            query=_TinyQuery(
                source_image_sha256=f"{index:064x}",
                source_key=f"query_execution:{index}",
                cache_payload_sha256=f"{index + 100:064x}",
            ),
            x=float(index + 1),
            y=float((index % 3) - 1),
        )
        for index in range(8)
    )
    torch.manual_seed(123)
    base = _TinyModel().state_dict()

    def loss(model: _TinyModel, episode: _TinyEpisode, target_first: bool) -> torch.Tensor:
        prediction = model.linear(torch.tensor([[episode.x]], dtype=torch.float64)).squeeze()
        # The order bit is exercised in the schedule but cannot alter the
        # target-signed mathematical objective.
        signed = prediction - episode.y
        return signed.square()

    uninterrupted = _TinyModel()
    uninterrupted.load_state_dict(base)
    full, full_optimizer = train_updates(
        uninterrupted,
        episodes,  # type: ignore[arg-type]
        outer_fold=1,
        initialization_checkpoint_sha256="a" * 64,
        stop_after_updates=7,
        loss_fn=loss,  # type: ignore[arg-type]
    )

    resumed_model = _TinyModel()
    resumed_model.load_state_dict(base)
    prefix, _ = train_updates(
        resumed_model,
        episodes,  # type: ignore[arg-type]
        outer_fold=1,
        initialization_checkpoint_sha256="a" * 64,
        stop_after_updates=3,
        loss_fn=loss,  # type: ignore[arg-type]
    )
    resumed, resumed_optimizer = train_updates(
        resumed_model,
        episodes,  # type: ignore[arg-type]
        outer_fold=1,
        initialization_checkpoint_sha256="a" * 64,
        stop_after_updates=7,
        resume_state=prefix,
        loss_fn=loss,  # type: ignore[arg-type]
    )

    assert _tree_equal(full["model_state_dict"], resumed["model_state_dict"])
    assert _tree_equal(full_optimizer.state_dict(), resumed_optimizer.state_dict())
    assert full["loss_trace"] == resumed["loss_trace"]
    assert full["schedule_sha256"] == resumed["schedule_sha256"]
    assert full["next_sampler_query_keys"] == resumed["next_sampler_query_keys"]
    assert _tree_equal(full["rng_state"], resumed["rng_state"])
