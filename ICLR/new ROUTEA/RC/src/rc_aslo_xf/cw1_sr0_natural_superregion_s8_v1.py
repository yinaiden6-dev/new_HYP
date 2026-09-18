"""Target-free all-region V materialization for the natural S8 gate.

The frozen natural V2 bridge deliberately verifies only its single P-selected
connected super-region.  That is the correct deployed proposal/verification
contract, but it is not sufficient for the S8 representation gate: S8 needs a
feature row for every already-sealed r1--r4 P hypothesis without rerunning the
same root-level complementary evidence for every overlapping region.

This module leaves ``cw1_sr0_natural_superregion_v2`` unchanged.  It evaluates
the two complementary V banks of every frozen P root exactly once, then uses
the frozen structural aggregation weights to populate every region.  P is
never recomputed by V, and the first eight ledger columns are copied exactly
from the sealed preparation.  There is no label join, D1/rank/slot input,
target mask, endpoint access, policy, loss, or gradient-bearing training path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from . import cw1_sr0_natural_superregion_v2 as _v2
from .cw1_sr0_structure_v1 import enumerate_superregion_bank, superregion_sha256
from .geometry_hypothesis_v1 import grid_cell_centres


SCHEMA_VERSION = "rc_cw1_sr0_natural_superregion_s8_all_region_v1"
PROTECTED_ZERO: Mapping[str, int] = dict(_v2.PROTECTED_ZERO)
REGION_FEATURE_COLUMNS = _v2.REGION_FEATURE_COLUMNS
DIRECTION_COUNT = 2
DIRECTIONAL_P_FEATURE_COLUMNS = (
    "p_proposal_support",
    "p_action_margin",
    "p_eligible_mass",
    "p_root_count",
    "p_eligible_root_count",
    "p_affine_fit_quality",
)
DIRECTIONAL_V_FEATURE_COLUMNS = (
    "v_signed_score",
    "v_valid_fraction",
    "v_positive_mass",
    "v_observed",
)

CONTROL_BASE = _v2.CONTROL_BASE
CONTROL_V_C_BIND = _v2.CONTROL_V_C_BIND
CONTROL_V_REF_SPATIAL = _v2.CONTROL_V_REF_SPATIAL
CONTROL_MODES = _v2.CONTROL_MODES


@dataclass(frozen=True)
class NaturalAllRegionVerificationS8V1:
    """Root-once V evidence and its frozen aggregation over every P region."""

    schema_version: str
    preparation_seal_sha256: str
    control_mode: str
    verification_reference_sha256: str
    roots: tuple[_v2.NaturalVRootV2, ...]
    directional_root_v_features: torch.Tensor
    region_keys: tuple[tuple[int, int, str, int], ...]
    region_v_features: torch.Tensor
    directional_region_v_features: torch.Tensor
    v_read_query_mask: torch.Tensor
    seal_sha256: str
    protected_access_counts: Mapping[str, int]


@dataclass(frozen=True)
class NaturalAllRegionFeatureLedgerS8V1:
    """Finite [R,12] target-free ledger with P exact and V fully observed."""

    schema_version: str
    preparation_seal_sha256: str
    verification_seal_sha256: str
    region_keys: tuple[tuple[int, int, str, int], ...]
    feature_columns: tuple[str, ...]
    features: torch.Tensor
    directional_p_feature_columns: tuple[str, ...]
    directional_v_feature_columns: tuple[str, ...]
    directional_p_features: torch.Tensor
    directional_v_features: torch.Tensor
    logical_sha256: str
    protected_access_counts: Mapping[str, int]


def _region_keys(
    prepared: _v2.NaturalSuperregionPreparationV2,
) -> tuple[tuple[int, int, str, int], ...]:
    return tuple(
        (row.radius, row.cell_count, row.region_sha256, row.region_ordinal)
        for row in prepared.regions
    )


def _bind_verification_reference(
    prepared: _v2.NaturalSuperregionPreparationV2,
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    *,
    control_mode: str,
    reference_permutation: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Apply the frozen V2 control semantics without touching the P seal."""

    if control_mode not in CONTROL_MODES:
        raise ValueError("unknown natural S8 verification control")
    query = _v2._unit_tokens(  # noqa: SLF001 - exact frozen-V2 semantics
        query_tokens,
        prepared.query_grid_shape,
        name="S8 verification query tokens",
    )
    source_reference = _v2._unit_tokens(  # noqa: SLF001
        reference_tokens,
        prepared.reference_grid_shape,
        name="S8 verification reference tokens",
    ).to(query.device)
    if _v2.tensor_sha256(query) != prepared.query_tokens_sha256:
        raise ValueError("S8 V cannot change the query bound to P")
    source_hash = _v2.tensor_sha256(source_reference)

    if control_mode == CONTROL_BASE:
        if (
            reference_permutation is not None
            or source_hash != prepared.reference_tokens_sha256
        ):
            raise ValueError("BASE V must use the P-bound reference")
        reference = source_reference
    elif control_mode == CONTROL_V_C_BIND:
        if (
            reference_permutation is not None
            or source_hash == prepared.reference_tokens_sha256
        ):
            raise ValueError("V_C_BIND must use different reference content")
        reference = source_reference
    else:
        if (
            source_hash != prepared.reference_tokens_sha256
            or reference_permutation is None
        ):
            raise ValueError(
                "V_REF_SPATIAL requires P-bound content and a permutation"
            )
        permutation = torch.as_tensor(
            reference_permutation, dtype=torch.long
        ).detach().cpu().contiguous()
        count = math.prod(prepared.reference_grid_shape)
        identity = torch.arange(count)
        if (
            permutation.shape != (count,)
            or permutation.unique().numel() != count
            or int(permutation.min()) != 0
            or int(permutation.max()) != count - 1
            or bool(permutation.eq(identity).any())
        ):
            raise ValueError(
                "V_REF_SPATIAL requires a complete no-fixed-point permutation"
            )
        reference = source_reference[
            permutation.to(source_reference.device)
        ].contiguous()
        if _v2.tensor_sha256(reference) == prepared.reference_tokens_sha256:
            raise ValueError(
                "V_REF_SPATIAL permutation did not change reference content binding"
            )
    return query, reference, _v2.tensor_sha256(reference)


def _verify_root_once(
    prepared: _v2.NaturalSuperregionPreparationV2,
    root: _v2.NaturalPRootV2,
    query: torch.Tensor,
    reference: torch.Tensor,
    query_xy: torch.Tensor,
    actions: Sequence[_v2.NativeReferenceActionV2],
) -> tuple[_v2.NaturalVRootV2, torch.Tensor]:
    """Evaluate both complementary V banks for one P root exactly once."""

    direction_scores: list[float] = []
    valid_counts: list[int] = []
    total_counts: list[int] = []
    evidence_rows: list[torch.Tensor] = []
    read_mask = torch.zeros(
        math.prod(prepared.query_grid_shape), dtype=torch.bool
    )
    for direction in root.directions:
        v_indices = direction.v_query_indices
        total_counts.append(int(v_indices.numel()))
        predicted = _v2._apply_affine(  # noqa: SLF001
            direction.affine, query_xy[v_indices]
        )
        inside = predicted.gt(0.0).all(dim=1) & predicted.lt(1.0).all(dim=1)
        r_height, r_width = prepared.reference_grid_shape
        nearest_x = (
            torch.floor(predicted[:, 0] * r_width)
            .to(torch.long)
            .clamp(0, r_width - 1)
        )
        nearest_y = (
            torch.floor(predicted[:, 1] * r_height)
            .to(torch.long)
            .clamp(0, r_height - 1)
        )
        nearest = nearest_y * r_width + nearest_x
        action = actions[direction.selected_action]
        inside &= action.mask[nearest]
        valid_indices = v_indices[inside]
        valid_xy = predicted[inside]
        valid_counts.append(int(valid_indices.numel()))

        # Match the frozen selected-only implementation: the complete bank is
        # the denominator and invalid transports contribute exact zero.
        evidence = torch.zeros(
            v_indices.numel(), dtype=query.dtype, device=query.device
        )
        if valid_indices.numel() and direction.eligible:
            sampled = F.normalize(
                _v2._bilinear_sample(  # noqa: SLF001
                    reference, prepared.reference_grid_shape, valid_xy
                ),
                p=2,
                dim=1,
                eps=1.0e-12,
            )
            evidence[inside.to(evidence.device)] = (
                query[valid_indices.to(query.device)] * sampled
            ).sum(dim=1)
        score = (
            0.0
            if evidence.numel() == 0
            else float(evidence.sum().detach()) / float(evidence.numel())
        )
        direction_scores.append(score)
        evidence_rows.append(evidence)
        read_mask[v_indices] = True

    total = sum(total_counts)
    valid = sum(valid_counts)
    concatenated = (
        torch.cat(evidence_rows)
        if any(row.numel() for row in evidence_rows)
        else torch.empty(0, device=query.device)
    )
    eligible = all(
        direction.eligible and count >= _v2.MINIMUM_V_CELLS
        for direction, count in zip(
            root.directions, valid_counts, strict=True
        )
    )
    row = _v2.NaturalVRootV2(
        root_ordinal=root.root_ordinal,
        direction_scores=(direction_scores[0], direction_scores[1]),
        valid_v_patch_counts=(valid_counts[0], valid_counts[1]),
        total_v_patch_counts=(total_counts[0], total_counts[1]),
        score=0.5 * (direction_scores[0] + direction_scores[1]),
        valid_fraction=0.0 if total == 0 else float(valid) / float(total),
        positive_mass=(
            0.0
            if concatenated.numel() == 0
            else float(concatenated.clamp_min(0.0).sum().detach())
        ),
        eligible=eligible,
    )
    return row, read_mask


def _verify_all_roots_batched(
    prepared: _v2.NaturalSuperregionPreparationV2,
    query: torch.Tensor,
    reference: torch.Tensor,
    query_xy: torch.Tensor,
    actions: Sequence[_v2.NativeReferenceActionV2],
) -> tuple[tuple[_v2.NaturalVRootV2, ...], torch.Tensor, torch.Tensor]:
    """Exact root-once semantics with one reference sampling launch.

    A natural candidate has roughly 45--48 roots and two complementary V
    banks per root.  Launching ``grid_sample`` once per bank made the full-C128
    S8 path needlessly scheduler-bound.  The coordinates are fixed by the P
    seal, so concatenating them before sampling changes neither the evidence
    population nor its per-bank denominator.  Results are split back into the
    original banks before any reduction.
    """

    descriptors: list[dict[str, object]] = []
    all_valid_xy: list[torch.Tensor] = []
    read_mask = torch.zeros(
        math.prod(prepared.query_grid_shape), dtype=torch.bool
    )
    for root in prepared.roots:
        for direction_ordinal, direction in enumerate(root.directions):
            v_indices = direction.v_query_indices
            predicted = _v2._apply_affine(  # noqa: SLF001
                direction.affine, query_xy[v_indices]
            )
            inside = predicted.gt(0.0).all(dim=1) & predicted.lt(1.0).all(dim=1)
            r_height, r_width = prepared.reference_grid_shape
            nearest_x = (
                torch.floor(predicted[:, 0] * r_width)
                .to(torch.long)
                .clamp(0, r_width - 1)
            )
            nearest_y = (
                torch.floor(predicted[:, 1] * r_height)
                .to(torch.long)
                .clamp(0, r_height - 1)
            )
            nearest = nearest_y * r_width + nearest_x
            action = actions[direction.selected_action]
            inside &= action.mask[nearest]
            valid_indices = v_indices[inside]
            valid_xy = predicted[inside]
            descriptors.append(
                {
                    "root": root.root_ordinal,
                    "direction": direction_ordinal,
                    "direction_eligible": direction.eligible,
                    "v_indices": v_indices,
                    "inside": inside,
                    "valid_indices": valid_indices,
                    "valid_xy": valid_xy,
                }
            )
            if valid_xy.numel():
                all_valid_xy.append(valid_xy)
            read_mask[v_indices] = True

    if all_valid_xy:
        coordinates = torch.cat(all_valid_xy, dim=0)
        sampled_all = F.normalize(
            _v2._bilinear_sample(  # noqa: SLF001
                reference, prepared.reference_grid_shape, coordinates
            ),
            p=2,
            dim=1,
            eps=1.0e-12,
        )
    else:
        sampled_all = torch.empty(
            (0, _v2.DESCRIPTOR_DIMENSION),
            dtype=query.dtype,
            device=query.device,
        )

    cursor = 0
    by_root: dict[int, list[dict[str, object]]] = {
        root.root_ordinal: [] for root in prepared.roots
    }
    for descriptor in descriptors:
        v_indices = descriptor["v_indices"]
        inside = descriptor["inside"]
        valid_indices = descriptor["valid_indices"]
        count = int(valid_indices.numel())
        evidence = torch.zeros(
            int(v_indices.numel()), dtype=query.dtype, device=query.device
        )
        if count and bool(descriptor["direction_eligible"]):
            sampled = sampled_all[cursor : cursor + count]
            evidence[inside.to(evidence.device)] = (
                query[valid_indices.to(query.device)] * sampled
            ).sum(dim=1)
        cursor += count
        score = (
            0.0
            if evidence.numel() == 0
            else float(evidence.sum().detach()) / float(evidence.numel())
        )
        by_root[int(descriptor["root"])].append(
            {
                "direction": int(descriptor["direction"]),
                "score": score,
                "valid": count,
                "total": int(v_indices.numel()),
                "evidence": evidence,
                "eligible": bool(descriptor["direction_eligible"]),
            }
        )
    if cursor != sampled_all.shape[0]:
        raise RuntimeError("S8 batched V coordinate partition drift")

    output: list[_v2.NaturalVRootV2] = []
    directional = torch.empty(
        (len(prepared.roots), DIRECTION_COUNT, len(DIRECTIONAL_V_FEATURE_COLUMNS)),
        dtype=torch.float64,
    )
    for root in prepared.roots:
        rows = sorted(by_root[root.root_ordinal], key=lambda row: row["direction"])
        if len(rows) != 2 or [row["direction"] for row in rows] != [0, 1]:
            raise RuntimeError("S8 batched V direction population drift")
        total = sum(int(row["total"]) for row in rows)
        valid = sum(int(row["valid"]) for row in rows)
        evidence = torch.cat([row["evidence"] for row in rows])
        eligible = all(
            bool(row["eligible"]) and int(row["valid"]) >= _v2.MINIMUM_V_CELLS
            for row in rows
        )
        output.append(
            _v2.NaturalVRootV2(
                root_ordinal=root.root_ordinal,
                direction_scores=(float(rows[0]["score"]), float(rows[1]["score"])),
                valid_v_patch_counts=(int(rows[0]["valid"]), int(rows[1]["valid"])),
                total_v_patch_counts=(int(rows[0]["total"]), int(rows[1]["total"])),
                score=0.5 * (float(rows[0]["score"]) + float(rows[1]["score"])),
                valid_fraction=0.0 if total == 0 else float(valid) / float(total),
                positive_mass=(
                    0.0
                    if evidence.numel() == 0
                    else float(evidence.clamp_min(0.0).sum().detach())
                ),
                eligible=eligible,
            )
        )
        for direction, row in enumerate(rows):
            row_evidence = row["evidence"]
            directional[root.root_ordinal, direction] = torch.tensor(
                (
                    float(row["score"]),
                    0.0
                    if int(row["total"]) == 0
                    else float(int(row["valid"])) / float(int(row["total"])),
                    0.0
                    if row_evidence.numel() == 0
                    else float(row_evidence.clamp_min(0.0).sum().detach()),
                    1.0,
                ),
                dtype=torch.float64,
            )
    if not bool(torch.isfinite(directional).all()) or not bool(
        directional[..., 3].eq(1.0).all()
    ):
        raise RuntimeError("S8 directional root V features are invalid")
    return tuple(output), directional, read_mask


def _aggregate_regions(
    prepared: _v2.NaturalSuperregionPreparationV2,
    roots: Sequence[_v2.NaturalVRootV2],
) -> torch.Tensor:
    """Aggregate the root-once evidence with the sealed structural weights."""

    by_root = {item.root_ordinal: item for item in roots}
    if len(by_root) != len(prepared.roots) or set(by_root) != set(
        range(len(prepared.roots))
    ):
        raise RuntimeError("S8 all-region V root population is incomplete")
    structural = {
        superregion_sha256(item): item
        for item in enumerate_superregion_bank(
            prepared.query_grid_shape, include_r0_control=False
        )
    }
    output = torch.empty((len(prepared.regions), 4), dtype=torch.float64)
    for row in prepared.regions:
        try:
            region = structural[row.region_sha256]
        except KeyError as error:
            raise RuntimeError("S8 region hash is absent from frozen bank") from error
        output[row.region_ordinal] = torch.tensor(
            (
                sum(
                    float(region.aggregation_weights[root]) * by_root[root].score
                    for root in region.contributing_root_ordinals
                ),
                sum(
                    float(region.aggregation_weights[root])
                    * by_root[root].valid_fraction
                    for root in region.contributing_root_ordinals
                ),
                sum(
                    float(region.aggregation_weights[root])
                    * by_root[root].positive_mass
                    for root in region.contributing_root_ordinals
                ),
                1.0,
            ),
            dtype=torch.float64,
        )
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("S8 all-region V features are not finite")
    return output


def _structural_regions(
    prepared: _v2.NaturalSuperregionPreparationV2,
) -> dict[str, object]:
    return {
        superregion_sha256(item): item
        for item in enumerate_superregion_bank(
            prepared.query_grid_shape, include_r0_control=False
        )
    }


def _directional_p_regions(
    prepared: _v2.NaturalSuperregionPreparationV2,
) -> torch.Tensor:
    """Aggregate P banks separately so each direction keeps its held-out V."""

    structural = _structural_regions(prepared)
    r_height, r_width = prepared.reference_grid_shape
    cell_diagonal = math.sqrt(1.0 / r_height**2 + 1.0 / r_width**2)
    output = torch.empty(
        (
            DIRECTION_COUNT,
            len(prepared.regions),
            len(DIRECTIONAL_P_FEATURE_COLUMNS),
        ),
        dtype=torch.float64,
    )
    for row in prepared.regions:
        region = structural[row.region_sha256]
        roots = region.contributing_root_ordinals
        weights = region.aggregation_weights
        for direction in range(DIRECTION_COUNT):
            directions = [prepared.roots[root].directions[direction] for root in roots]
            support = sum(
                float(weights[root])
                * (
                    item.proposal_support
                    if math.isfinite(item.proposal_support)
                    else 0.0
                )
                for root, item in zip(roots, directions, strict=True)
            )
            margin = sum(
                float(weights[root]) * item.action_margin
                for root, item in zip(roots, directions, strict=True)
            )
            eligible_mass = sum(
                float(weights[root]) * float(item.eligible)
                for root, item in zip(roots, directions, strict=True)
            )
            fit_quality = sum(
                float(weights[root])
                * (
                    math.exp(-item.affine_rmse / cell_diagonal)
                    if math.isfinite(item.affine_rmse)
                    else 0.0
                )
                for root, item in zip(roots, directions, strict=True)
            )
            output[direction, row.region_ordinal] = torch.tensor(
                (
                    support,
                    margin,
                    eligible_mass,
                    float(len(roots)),
                    float(sum(int(item.eligible) for item in directions)),
                    fit_quality,
                ),
                dtype=torch.float64,
            )
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("S8 directional P features are not finite")
    return output


def _directional_v_regions(
    prepared: _v2.NaturalSuperregionPreparationV2,
    directional_roots: torch.Tensor,
) -> torch.Tensor:
    """Aggregate each complementary V bank without crossing directions."""

    expected = (
        len(prepared.roots),
        DIRECTION_COUNT,
        len(DIRECTIONAL_V_FEATURE_COLUMNS),
    )
    if directional_roots.shape != expected or not bool(
        torch.isfinite(directional_roots).all()
    ):
        raise RuntimeError("S8 directional root V population drift")
    structural = _structural_regions(prepared)
    output = torch.empty(
        (
            DIRECTION_COUNT,
            len(prepared.regions),
            len(DIRECTIONAL_V_FEATURE_COLUMNS),
        ),
        dtype=torch.float64,
    )
    for row in prepared.regions:
        region = structural[row.region_sha256]
        roots = region.contributing_root_ordinals
        for direction in range(DIRECTION_COUNT):
            value = torch.zeros(
                len(DIRECTIONAL_V_FEATURE_COLUMNS), dtype=torch.float64
            )
            for root in roots:
                value += (
                    float(region.aggregation_weights[root])
                    * directional_roots[root, direction]
                )
            # Observation is a contract bit, not a weighted statistic.
            value[3] = 1.0
            output[direction, row.region_ordinal] = value
    if not bool(torch.isfinite(output).all()) or not bool(
        output[..., 3].eq(1.0).all()
    ):
        raise RuntimeError("S8 directional region V features are invalid")
    return output


def _verification_seal(
    *,
    prepared: _v2.NaturalSuperregionPreparationV2,
    control_mode: str,
    verification_reference_hash: str,
    roots: Sequence[_v2.NaturalVRootV2],
    directional_root_v_features: torch.Tensor,
    region_keys: Sequence[tuple[int, int, str, int]],
    region_v_features: torch.Tensor,
    directional_region_v_features: torch.Tensor,
    v_read_query_mask: torch.Tensor,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "preparation_seal_sha256": prepared.seal_sha256,
        "control_mode": control_mode,
        "verification_reference_sha256": verification_reference_hash,
        "roots": [
            {
                "root": item.root_ordinal,
                "direction_scores": [
                    float(value).hex() for value in item.direction_scores
                ],
                "valid_v_patch_counts": list(item.valid_v_patch_counts),
                "total_v_patch_counts": list(item.total_v_patch_counts),
                "score": float(item.score).hex(),
                "valid_fraction": float(item.valid_fraction).hex(),
                "positive_mass": float(item.positive_mass).hex(),
                "eligible": item.eligible,
            }
            for item in roots
        ],
        "directional_root_v_features_sha256": _v2.tensor_sha256(
            directional_root_v_features
        ),
        "region_keys": list(region_keys),
        "region_v_features_sha256": _v2.tensor_sha256(region_v_features),
        "directional_region_v_features_sha256": _v2.tensor_sha256(
            directional_region_v_features
        ),
        "v_read_query_mask_sha256": _v2.tensor_sha256(v_read_query_mask),
        "protected_access_counts": dict(PROTECTED_ZERO),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_all_natural_superregions_s8_v1(
    prepared: _v2.NaturalSuperregionPreparationV2,
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    *,
    control_mode: str = CONTROL_BASE,
    reference_permutation: torch.Tensor | None = None,
) -> NaturalAllRegionVerificationS8V1:
    """Verify all sealed P regions while evaluating each root only once."""

    _v2._validate_preparation(prepared)  # noqa: SLF001
    query, reference, verification_hash = _bind_verification_reference(
        prepared,
        query_tokens,
        reference_tokens,
        control_mode=control_mode,
        reference_permutation=reference_permutation,
    )
    actions = _v2.enumerate_native_reference_actions_v2(
        prepared.reference_grid_shape
    )
    query_xy = grid_cell_centres(prepared.query_grid_shape, dtype=torch.float64)
    roots, directional_root_v_features, read_mask = _verify_all_roots_batched(
        prepared, query, reference, query_xy, actions
    )
    region_keys = _region_keys(prepared)
    region_v_features = _aggregate_regions(prepared, roots)
    directional_region_v_features = _directional_v_regions(
        prepared, directional_root_v_features
    )
    seal = _verification_seal(
        prepared=prepared,
        control_mode=control_mode,
        verification_reference_hash=verification_hash,
        roots=roots,
        directional_root_v_features=directional_root_v_features,
        region_keys=region_keys,
        region_v_features=region_v_features,
        directional_region_v_features=directional_region_v_features,
        v_read_query_mask=read_mask,
    )
    return NaturalAllRegionVerificationS8V1(
        schema_version=SCHEMA_VERSION,
        preparation_seal_sha256=prepared.seal_sha256,
        control_mode=control_mode,
        verification_reference_sha256=verification_hash,
        roots=tuple(roots),
        directional_root_v_features=directional_root_v_features,
        region_keys=region_keys,
        region_v_features=region_v_features,
        directional_region_v_features=directional_region_v_features,
        v_read_query_mask=read_mask,
        seal_sha256=seal,
        protected_access_counts=dict(PROTECTED_ZERO),
    )


def _validate_verification(
    prepared: _v2.NaturalSuperregionPreparationV2,
    verified: NaturalAllRegionVerificationS8V1,
) -> None:
    if not isinstance(verified, NaturalAllRegionVerificationS8V1):
        raise TypeError("expected an S8 all-region verification")
    if (
        verified.schema_version != SCHEMA_VERSION
        or verified.preparation_seal_sha256 != prepared.seal_sha256
        or verified.control_mode not in CONTROL_MODES
        or verified.protected_access_counts != PROTECTED_ZERO
        or verified.region_keys != _region_keys(prepared)
        or verified.region_v_features.shape != (len(prepared.regions), 4)
        or verified.directional_root_v_features.shape
        != (len(prepared.roots), DIRECTION_COUNT, 4)
        or verified.directional_region_v_features.shape
        != (DIRECTION_COUNT, len(prepared.regions), 4)
        or not bool(torch.isfinite(verified.region_v_features).all())
        or not bool(torch.isfinite(verified.directional_root_v_features).all())
        or not bool(torch.isfinite(verified.directional_region_v_features).all())
        or not bool(verified.region_v_features[:, 3].eq(1.0).all())
        or not bool(verified.directional_root_v_features[..., 3].eq(1.0).all())
        or not bool(verified.directional_region_v_features[..., 3].eq(1.0).all())
    ):
        raise ValueError("S8 all-region verification/preparation binding drift")
    expected = _verification_seal(
        prepared=prepared,
        control_mode=verified.control_mode,
        verification_reference_hash=verified.verification_reference_sha256,
        roots=verified.roots,
        directional_root_v_features=verified.directional_root_v_features,
        region_keys=verified.region_keys,
        region_v_features=verified.region_v_features,
        directional_region_v_features=verified.directional_region_v_features,
        v_read_query_mask=verified.v_read_query_mask,
    )
    if verified.seal_sha256 != expected:
        raise ValueError("S8 all-region verification seal drift")


def materialize_all_natural_superregion_feature_ledger_s8_v1(
    prepared: _v2.NaturalSuperregionPreparationV2,
    verified: NaturalAllRegionVerificationS8V1,
) -> NaturalAllRegionFeatureLedgerS8V1:
    """Join frozen P columns with all-region V columns, without labels."""

    _v2._validate_preparation(prepared)  # noqa: SLF001
    _validate_verification(prepared, verified)
    features = torch.empty(
        (len(prepared.regions), len(REGION_FEATURE_COLUMNS)), dtype=torch.float64
    )
    for row in prepared.regions:
        features[row.region_ordinal, :8] = torch.tensor(
            (
                row.p_score,
                row.proposal_support,
                row.action_margin,
                row.eligible_mass,
                float(row.root_count),
                float(row.eligible_root_count),
                row.reference_iou,
                row.affine_agreement,
            ),
            dtype=torch.float64,
        )
    features[:, 8:] = verified.region_v_features
    directional_p = _directional_p_regions(prepared)
    directional_v = verified.directional_region_v_features.clone()
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError("S8 all-region feature ledger is not finite")
    logical = hashlib.sha256(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "preparation_seal_sha256": prepared.seal_sha256,
                "verification_seal_sha256": verified.seal_sha256,
                "region_keys": verified.region_keys,
                "feature_columns": REGION_FEATURE_COLUMNS,
                "features_sha256": _v2.tensor_sha256(features),
                "directional_p_feature_columns": DIRECTIONAL_P_FEATURE_COLUMNS,
                "directional_v_feature_columns": DIRECTIONAL_V_FEATURE_COLUMNS,
                "directional_p_features_sha256": _v2.tensor_sha256(directional_p),
                "directional_v_features_sha256": _v2.tensor_sha256(directional_v),
                "protected_access_counts": dict(PROTECTED_ZERO),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return NaturalAllRegionFeatureLedgerS8V1(
        schema_version=SCHEMA_VERSION,
        preparation_seal_sha256=prepared.seal_sha256,
        verification_seal_sha256=verified.seal_sha256,
        region_keys=verified.region_keys,
        feature_columns=REGION_FEATURE_COLUMNS,
        features=features,
        directional_p_feature_columns=DIRECTIONAL_P_FEATURE_COLUMNS,
        directional_v_feature_columns=DIRECTIONAL_V_FEATURE_COLUMNS,
        directional_p_features=directional_p,
        directional_v_features=directional_v,
        logical_sha256=logical,
        protected_access_counts=dict(PROTECTED_ZERO),
    )


__all__ = [
    "SCHEMA_VERSION",
    "PROTECTED_ZERO",
    "REGION_FEATURE_COLUMNS",
    "DIRECTION_COUNT",
    "DIRECTIONAL_P_FEATURE_COLUMNS",
    "DIRECTIONAL_V_FEATURE_COLUMNS",
    "CONTROL_BASE",
    "CONTROL_V_C_BIND",
    "CONTROL_V_REF_SPATIAL",
    "NaturalAllRegionVerificationS8V1",
    "NaturalAllRegionFeatureLedgerS8V1",
    "verify_all_natural_superregions_s8_v1",
    "materialize_all_natural_superregion_feature_ledger_s8_v1",
]
