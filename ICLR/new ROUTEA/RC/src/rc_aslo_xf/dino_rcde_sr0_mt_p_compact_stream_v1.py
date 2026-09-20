"""Additive compact/batched SR0-MT P feature primitives.

The I1-sealed scalar core and V1 materializer are immutable.  This module is a
new implementation version that preserves their 16-coordinate equations while
removing two engineering pathologies from a future execution graph:

* one candidate normalizes its frozen ColNomic token banks and computes its
  complete float32 QxR cosine matrix exactly once; and
* a complete query-root x reference-action Cartesian population is evaluated
  in cardinality-matched tensor groups instead of thousands of Python calls.

The resulting coordinates have float32 provenance exactly like the sealed
core.  Compact storage therefore uses float32 and restores to float64 exactly
before the existing shared P head.  No target, label, rank, D1 quantity,
candidate pruning, learned parameter, or scientific reducer appears here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .dino_rcde_cw1_multitile_sr0_p_v1 import (
    COLNOMIC_LME_TEMPERATURE,
    EMPTY_ACTION_H0,
    FEATURE_DIM,
    MISSING_TOP2_GAP,
    NORMALIZED_ENTROPY_EPSILON,
    SharedPContractError,
)


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_p_compact_stream_v1_20260815"
COMPACT_CARTESIAN_LAYOUT = "QUERY_ROOT_X_REFERENCE_ACTION_COMPLETE_CARTESIAN"
COMPACT_FEATURE_STORAGE_DTYPE = torch.float32
RESTORED_HEAD_DTYPE = torch.float64


def _finite_floating(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise SharedPContractError(f"{name} must be finite floating point")
    return tensor


def _valid_mask(value: torch.Tensor, count: int, *, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.bool).detach().contiguous()
    if result.shape != (count,):
        raise SharedPContractError(f"{name} must be a one-dimensional patch mask")
    return result


def _unit_tokens(
    value: torch.Tensor, valid_patch_mask: torch.Tensor, *, name: str
) -> torch.Tensor:
    source = _finite_floating(value, name=name)
    if (
        source.ndim != 2
        or source.shape[0] != valid_patch_mask.numel()
        or source.shape[1] <= 0
    ):
        raise SharedPContractError(
            f"{name} must have shape [patch,positive-dimension]"
        )
    source = source.detach().to(torch.float32).contiguous()
    valid = valid_patch_mask.to(device=source.device)
    norms = torch.linalg.vector_norm(source, dim=1)
    if bool(norms[valid].le(0.0).any()):
        raise SharedPContractError(f"{name} valid patches must have nonzero norm")
    output = torch.zeros_like(source)
    output[valid] = source[valid] / norms[valid, None]
    return output


@dataclass(frozen=True)
class PreparedCandidateColNomicFeatures:
    """One target-free, normalized candidate-local similarity population."""

    full_similarity: torch.Tensor
    query_valid_patch_mask: torch.Tensor
    reference_valid_patch_mask: torch.Tensor

    def __post_init__(self) -> None:
        pair = torch.as_tensor(self.full_similarity)
        q_valid = torch.as_tensor(
            self.query_valid_patch_mask, dtype=torch.bool, device=pair.device
        ).detach().contiguous()
        r_valid = torch.as_tensor(
            self.reference_valid_patch_mask, dtype=torch.bool, device=pair.device
        ).detach().contiguous()
        if (
            pair.ndim != 2
            or pair.dtype != torch.float32
            or pair.shape != (q_valid.numel(), r_valid.numel())
            or not bool(torch.isfinite(pair).all())
            or not bool(q_valid.any())
            or not bool(r_valid.any())
        ):
            raise SharedPContractError(
                "prepared ColNomic similarity/mask population is invalid"
            )
        object.__setattr__(self, "full_similarity", pair.detach().contiguous())
        object.__setattr__(self, "query_valid_patch_mask", q_valid)
        object.__setattr__(self, "reference_valid_patch_mask", r_valid)


def _full_candidate_similarity(
    query_unit: torch.Tensor, reference_unit: torch.Tensor
) -> torch.Tensor:
    """Instrumentation boundary: exactly one full QxR matmul per prepare."""

    return (query_unit @ reference_unit.T).detach().contiguous()


def prepare_candidate_colnomic_features(
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    *,
    query_valid_patch_mask: torch.Tensor,
    reference_valid_patch_mask: torch.Tensor,
) -> PreparedCandidateColNomicFeatures:
    query_source = torch.as_tensor(query_tokens)
    reference_source = torch.as_tensor(reference_tokens)
    if query_source.device != reference_source.device:
        raise SharedPContractError("query/reference ColNomic devices differ")
    if query_source.ndim != 2 or reference_source.ndim != 2:
        raise SharedPContractError("query/reference ColNomic tokens must be matrices")
    if query_source.shape[1] != reference_source.shape[1]:
        raise SharedPContractError("query/reference ColNomic dimensions differ")
    q_valid = _valid_mask(
        query_valid_patch_mask, query_source.shape[0], name="query valid mask"
    ).to(query_source.device)
    r_valid = _valid_mask(
        reference_valid_patch_mask,
        reference_source.shape[0],
        name="reference valid mask",
    ).to(reference_source.device)
    if not bool(q_valid.any()) or not bool(r_valid.any()):
        raise SharedPContractError("valid ColNomic patch populations must be nonempty")
    query_unit = _unit_tokens(query_source, q_valid, name="query ColNomic tokens")
    reference_unit = _unit_tokens(
        reference_source, r_valid, name="reference ColNomic tokens"
    )
    return PreparedCandidateColNomicFeatures(
        full_similarity=_full_candidate_similarity(query_unit, reference_unit),
        query_valid_patch_mask=q_valid,
        reference_valid_patch_mask=r_valid,
    )


def _scalar_summary(pair: torch.Tensor, full_pair: torch.Tensor) -> torch.Tensor:
    """Exact equation/order replay of the I1-sealed scalar core."""

    tau = COLNOMIC_LME_TEMPERATURE
    q_lme = tau * (
        torch.logsumexp(pair / tau, dim=1) - math.log(pair.shape[1])
    )
    r_lme = tau * (
        torch.logsumexp(pair / tau, dim=0) - math.log(pair.shape[0])
    )
    q_best, q_argmax = pair.max(dim=1)
    r_best, r_argmax = pair.max(dim=0)

    def gap(value: torch.Tensor, *, dim: int) -> torch.Tensor:
        if value.shape[dim] < 2:
            return torch.full(
                (value.shape[1 - dim],),
                MISSING_TOP2_GAP,
                dtype=value.dtype,
                device=value.device,
            )
        best = torch.topk(value, 2, dim=dim, largest=True, sorted=True).values
        return best.select(dim, 0) - best.select(dim, 1)

    def entropy(value: torch.Tensor, *, dim: int) -> torch.Tensor:
        population = value.shape[dim]
        if population < 2:
            return torch.zeros(
                value.shape[1 - dim], dtype=value.dtype, device=value.device
            )
        probability = torch.softmax(value / tau, dim=dim)
        return -(
            probability
            * torch.log(probability.clamp_min(NORMALIZED_ENTROPY_EPSILON))
        ).sum(dim=dim) / math.log(population)

    q_gap = gap(pair, dim=1)
    r_gap = gap(pair, dim=0)
    q_entropy = entropy(pair, dim=1)
    r_entropy = entropy(pair, dim=0)
    q_ordinals = torch.arange(pair.shape[0], device=pair.device)
    r_ordinals = torch.arange(pair.shape[1], device=pair.device)
    mutual_q = r_argmax[q_argmax].eq(q_ordinals)
    mutual_r = q_argmax[r_argmax].eq(r_ordinals)
    mutual = q_best[mutual_q]
    mutual_mean = (
        pair.square().sum() * 0.0 if mutual.numel() == 0 else mutual.mean()
    )
    full_q_lme = tau * (
        torch.logsumexp(full_pair / tau, dim=1) - math.log(full_pair.shape[1])
    )
    value = torch.stack(
        (
            q_lme.mean(),
            q_best.mean(),
            q_best.std(unbiased=False),
            q_gap.mean(),
            q_entropy.mean(),
            r_lme.mean(),
            r_best.mean(),
            r_best.std(unbiased=False),
            r_gap.mean(),
            r_entropy.mean(),
            mutual_q.to(pair.dtype).mean(),
            mutual_r.to(pair.dtype).mean(),
            pair.new_tensor(torch.unique(q_argmax).numel() / float(pair.shape[1])),
            pair.new_tensor(torch.unique(r_argmax).numel() / float(pair.shape[0])),
            mutual_mean,
            q_lme.mean() - full_q_lme.mean(),
        )
    ).detach().to(torch.float64).contiguous()
    if value.shape != (FEATURE_DIM,) or not bool(torch.isfinite(value).all()):
        raise RuntimeError("compact scalar feature schema drift")
    return value


def candidate_action_colnomic_features_from_prepared(
    prepared: PreparedCandidateColNomicFeatures,
    *,
    query_action_mask: torch.Tensor,
    reference_action_mask: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(prepared, PreparedCandidateColNomicFeatures):
        raise SharedPContractError("prepared ColNomic extractor has wrong type")
    full = prepared.full_similarity
    q_mask = _valid_mask(
        query_action_mask, full.shape[0], name="query action mask"
    ).to(full.device)
    r_mask = _valid_mask(
        reference_action_mask, full.shape[1], name="reference action mask"
    ).to(full.device)
    if bool((q_mask & ~prepared.query_valid_patch_mask).any()) or bool(
        (r_mask & ~prepared.reference_valid_patch_mask).any()
    ):
        raise SharedPContractError("candidate/action mask escaped geometry-valid patches")
    if not bool(q_mask.any()) or not bool(r_mask.any()):
        return torch.full(
            (FEATURE_DIM,), EMPTY_ACTION_H0, dtype=torch.float64, device=full.device
        )
    return _scalar_summary(
        full[q_mask][:, r_mask],
        full[q_mask][:, prepared.reference_valid_patch_mask],
    )


def _batch_summary(pair: torch.Tensor, full_q_lme_mean: torch.Tensor) -> torch.Tensor:
    """Equal-cardinality batched summary with scalar-length reductions."""

    tau = COLNOMIC_LME_TEMPERATURE
    population, query_count, reference_count = pair.shape
    q_lme = tau * (
        torch.logsumexp(pair / tau, dim=2) - math.log(reference_count)
    )
    r_lme = tau * (
        torch.logsumexp(pair / tau, dim=1) - math.log(query_count)
    )
    q_best, q_argmax = pair.max(dim=2)
    r_best, r_argmax = pair.max(dim=1)
    if reference_count < 2:
        q_gap = torch.zeros_like(q_best)
        q_entropy = torch.zeros_like(q_best)
    else:
        best = torch.topk(pair, 2, dim=2, largest=True, sorted=True).values
        q_gap = best[:, :, 0] - best[:, :, 1]
        probability = torch.softmax(pair / tau, dim=2)
        q_entropy = -(
            probability
            * torch.log(probability.clamp_min(NORMALIZED_ENTROPY_EPSILON))
        ).sum(dim=2) / math.log(reference_count)
    if query_count < 2:
        r_gap = torch.zeros_like(r_best)
        r_entropy = torch.zeros_like(r_best)
    else:
        best = torch.topk(pair, 2, dim=1, largest=True, sorted=True).values
        r_gap = best[:, 0, :] - best[:, 1, :]
        probability = torch.softmax(pair / tau, dim=1)
        r_entropy = -(
            probability
            * torch.log(probability.clamp_min(NORMALIZED_ENTROPY_EPSILON))
        ).sum(dim=1) / math.log(query_count)
    q_ordinals = torch.arange(query_count, device=pair.device)[None, :]
    r_ordinals = torch.arange(reference_count, device=pair.device)[None, :]
    mutual_q = torch.gather(r_argmax, 1, q_argmax).eq(q_ordinals)
    mutual_r = torch.gather(q_argmax, 1, r_argmax).eq(r_ordinals)
    mutual_count = mutual_q.sum(dim=1)
    mutual_mean = pair.square().sum(dim=(1, 2)) * 0.0
    for count in sorted(set(int(item) for item in mutual_count.tolist()) - {0}):
        rows = torch.nonzero(mutual_count.eq(count), as_tuple=False).flatten()
        selected = q_best[rows][mutual_q[rows]].reshape(rows.numel(), count)
        mutual_mean[rows] = selected.mean(dim=1)
    unique_reference = F.one_hot(
        q_argmax, num_classes=reference_count
    ).any(dim=1).sum(dim=1)
    unique_query = F.one_hot(
        r_argmax, num_classes=query_count
    ).any(dim=1).sum(dim=1)
    value = torch.stack(
        (
            q_lme.mean(dim=1),
            q_best.mean(dim=1),
            q_best.std(dim=1, unbiased=False),
            q_gap.mean(dim=1),
            q_entropy.mean(dim=1),
            r_lme.mean(dim=1),
            r_best.mean(dim=1),
            r_best.std(dim=1, unbiased=False),
            r_gap.mean(dim=1),
            r_entropy.mean(dim=1),
            mutual_q.to(pair.dtype).mean(dim=1),
            mutual_r.to(pair.dtype).mean(dim=1),
            unique_reference.to(pair.dtype) / float(reference_count),
            unique_query.to(pair.dtype) / float(query_count),
            mutual_mean,
            q_lme.mean(dim=1) - full_q_lme_mean,
        ),
        dim=1,
    ).detach().to(torch.float64).contiguous()
    if value.shape != (population, FEATURE_DIM) or not bool(torch.isfinite(value).all()):
        raise RuntimeError("compact batched feature schema drift")
    return value


def candidate_action_colnomic_feature_table_from_prepared(
    prepared: PreparedCandidateColNomicFeatures,
    *,
    query_action_masks: torch.Tensor,
    reference_action_masks: torch.Tensor,
) -> torch.Tensor:
    """Return the complete R x A x 16 table without pruning either axis."""

    if not isinstance(prepared, PreparedCandidateColNomicFeatures):
        raise SharedPContractError("prepared ColNomic extractor has wrong type")
    full = prepared.full_similarity
    q_masks = torch.as_tensor(
        query_action_masks, dtype=torch.bool, device=full.device
    ).detach().contiguous()
    r_masks = torch.as_tensor(
        reference_action_masks, dtype=torch.bool, device=full.device
    ).detach().contiguous()
    if q_masks.ndim != 2 or q_masks.shape[1] != full.shape[0] or q_masks.shape[0] <= 0:
        raise SharedPContractError("query action-mask table shape drift")
    if r_masks.ndim != 2 or r_masks.shape[1] != full.shape[1] or r_masks.shape[0] <= 0:
        raise SharedPContractError("reference action-mask table shape drift")
    if bool((q_masks & ~prepared.query_valid_patch_mask[None]).any()) or bool(
        (r_masks & ~prepared.reference_valid_patch_mask[None]).any()
    ):
        raise SharedPContractError("candidate/action mask escaped geometry-valid patches")
    output = torch.zeros(
        (q_masks.shape[0], r_masks.shape[0], FEATURE_DIM),
        dtype=torch.float64,
        device=full.device,
    )
    q_counts = q_masks.sum(dim=1)
    r_counts = r_masks.sum(dim=1)

    # A root/action pair is only a Cartesian product of one query mask and one
    # reference mask.  Coordinates 0--4 depend on the reference mask before
    # the query-root reduction, while coordinates 5--9 depend on the query
    # mask before the reference-action reduction.  Materialize those two
    # one-sided tables once per cardinality group instead of rebuilding an
    # [root,action,query-patch,reference-patch] tensor for every group pair.
    q_groups: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for nq in sorted(set(int(item) for item in q_counts.tolist()) - {0}):
        q_rows = torch.nonzero(q_counts.eq(nq), as_tuple=False).flatten()
        q_indices = torch.stack(
            [torch.nonzero(q_masks[row], as_tuple=False).flatten() for row in q_rows]
        )
        q_groups.append((nq, q_rows, q_indices))
    r_groups: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for nr in sorted(set(int(item) for item in r_counts.tolist()) - {0}):
        r_rows = torch.nonzero(r_counts.eq(nr), as_tuple=False).flatten()
        r_indices = torch.stack(
            [torch.nonzero(r_masks[row], as_tuple=False).flatten() for row in r_rows]
        )
        r_groups.append((nr, r_rows, r_indices))

    tau = COLNOMIC_LME_TEMPERATURE
    full_r = torch.nonzero(
        prepared.reference_valid_patch_mask, as_tuple=False
    ).flatten()
    full_q_lme = tau * (
        torch.logsumexp(full[:, full_r] / tau, dim=1) - math.log(full_r.numel())
    )

    # Per reference-mask group, retain one q-side record for every physical
    # query patch.  Each record is independent of which query root later reads
    # it.  Shapes are [physical-query, reference-action-in-group].
    q_side: list[
        tuple[
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = []
    for nr, r_rows, r_indices in r_groups:
        pair = full[:, r_indices]
        q_lme = tau * (torch.logsumexp(pair / tau, dim=2) - math.log(nr))
        q_best, q_argmax = pair.max(dim=2)
        if nr < 2:
            q_gap = torch.zeros_like(q_best)
            q_entropy = torch.zeros_like(q_best)
        else:
            best = torch.topk(pair, 2, dim=2, largest=True, sorted=True).values
            q_gap = best[:, :, 0] - best[:, :, 1]
            probability = torch.softmax(pair / tau, dim=2)
            q_entropy = -(
                probability
                * torch.log(probability.clamp_min(NORMALIZED_ENTROPY_EPSILON))
            ).sum(dim=2) / math.log(nr)
        q_side.append(
            (
                nr,
                r_rows,
                r_indices,
                q_lme,
                q_best,
                q_gap,
                q_entropy,
                q_argmax,
            )
        )

    # Symmetrically, one r-side record is shared by every reference action
    # that later selects that physical reference patch.  Shapes are
    # [query-root-in-group, physical-reference].
    r_side: list[
        tuple[
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = []
    for nq, q_rows, q_indices in q_groups:
        pair = full[q_indices]
        r_best, r_argmax = pair.max(dim=1)
        if nq < 2:
            r_gap = torch.zeros_like(r_best)
        else:
            best = torch.topk(pair, 2, dim=1, largest=True, sorted=True).values
            r_gap = best[:, 0, :] - best[:, 1, :]
        baseline = full_q_lme[q_indices].mean(dim=1)
        r_side.append(
            (
                nq,
                q_rows,
                q_indices,
                baseline,
                r_best,
                r_gap,
                r_argmax,
            )
        )

    for (
        nq,
        q_rows,
        q_indices,
        baseline,
        r_best,
        r_gap,
        r_argmax,
    ) in r_side:
        for (
            nr,
            r_rows,
            r_indices,
            q_lme,
            q_best,
            q_gap,
            q_entropy,
            q_argmax,
        ) in q_side:
            population = q_rows.numel() * r_rows.numel()

            # Preserve the former _batch_summary scalar-length reduction
            # order by flattening root/action before reducing patch ordinals.
            flat_q_lme = (
                q_lme[q_indices].permute(0, 2, 1).contiguous().reshape(population, nq)
            )
            flat_q_best = (
                q_best[q_indices]
                .permute(0, 2, 1)
                .contiguous()
                .reshape(population, nq)
            )
            flat_q_gap = (
                q_gap[q_indices].permute(0, 2, 1).contiguous().reshape(population, nq)
            )
            flat_q_entropy = (
                q_entropy[q_indices]
                .permute(0, 2, 1)
                .contiguous()
                .reshape(population, nq)
            )
            flat_q_argmax = (
                q_argmax[q_indices]
                .permute(0, 2, 1)
                .contiguous()
                .reshape(population, nq)
            )
            flat_r_best = r_best[:, r_indices].contiguous().reshape(population, nr)
            flat_r_gap = r_gap[:, r_indices].contiguous().reshape(population, nr)
            flat_r_argmax = (
                r_argmax[:, r_indices].contiguous().reshape(population, nr)
            )

            # Coordinates 5 and 9 are mathematically one-sided, but computing
            # them first over every physical reference patch and selecting an
            # action afterwards changes the float32 reduction layout by a few
            # ULPs.  Recreate only the former equal-cardinality pair layout so
            # logsumexp/softmax see exactly [population,nq,nr], as in the
            # scalar-length batched kernel.  This remains one tensor operation
            # per cardinality pair; it does not restore a root x action Python
            # scalar loop or recompute any other coordinate.
            legacy_pair = full[
                q_indices[:, None, :, None], r_indices[None, :, None, :]
            ].contiguous().reshape(population, nq, nr)
            scaled_legacy_pair = legacy_pair / tau
            exact_r_lme_mean = (
                tau
                * (
                    torch.logsumexp(scaled_legacy_pair, dim=1) - math.log(nq)
                )
            ).mean(dim=1)
            if nq < 2:
                exact_r_entropy_mean = torch.zeros(
                    population, dtype=full.dtype, device=full.device
                )
            else:
                probability = torch.softmax(scaled_legacy_pair, dim=1)
                exact_r_entropy_mean = (
                    -(
                        probability
                        * torch.log(
                            probability.clamp_min(NORMALIZED_ENTROPY_EPSILON)
                        )
                    ).sum(dim=1)
                    / math.log(nq)
                ).mean(dim=1)

            # The only genuinely joint coordinates need the two argmax maps,
            # not the pairwise similarity matrix.  Gather reciprocal endpoints
            # exactly as the sealed scalar core, then count unique endpoints
            # from sorted argmax ordinals.
            q_ordinals = torch.arange(nq, device=full.device)[None, :]
            r_ordinals = torch.arange(nr, device=full.device)[None, :]
            mutual_q = torch.gather(flat_r_argmax, 1, flat_q_argmax).eq(q_ordinals)
            mutual_r = torch.gather(flat_q_argmax, 1, flat_r_argmax).eq(r_ordinals)
            unique_reference = torch.ones(
                population, dtype=torch.long, device=full.device
            )
            if nq > 1:
                ordered = flat_q_argmax.sort(dim=1).values
                unique_reference = unique_reference + ordered[:, 1:].ne(
                    ordered[:, :-1]
                ).sum(dim=1)
            unique_query = torch.ones(
                population, dtype=torch.long, device=full.device
            )
            if nr > 1:
                ordered = flat_r_argmax.sort(dim=1).values
                unique_query = unique_query + ordered[:, 1:].ne(
                    ordered[:, :-1]
                ).sum(dim=1)
            mutual_count = mutual_q.sum(dim=1)
            mutual_mean = flat_q_best.square().sum(dim=1) * 0.0
            for count in sorted(set(int(item) for item in mutual_count.tolist()) - {0}):
                rows = torch.nonzero(mutual_count.eq(count), as_tuple=False).flatten()
                selected = flat_q_best[rows][mutual_q[rows]].reshape(
                    rows.numel(), count
                )
                mutual_mean[rows] = selected.mean(dim=1)

            flat_baseline = baseline[:, None].expand(
                q_rows.numel(), r_rows.numel()
            ).reshape(-1)
            value = torch.stack(
                (
                    flat_q_lme.mean(dim=1),
                    flat_q_best.mean(dim=1),
                    flat_q_best.std(dim=1, unbiased=False),
                    flat_q_gap.mean(dim=1),
                    flat_q_entropy.mean(dim=1),
                    exact_r_lme_mean,
                    flat_r_best.mean(dim=1),
                    flat_r_best.std(dim=1, unbiased=False),
                    flat_r_gap.mean(dim=1),
                    exact_r_entropy_mean,
                    mutual_q.to(full.dtype).mean(dim=1),
                    mutual_r.to(full.dtype).mean(dim=1),
                    unique_reference.to(full.dtype) / float(nr),
                    unique_query.to(full.dtype) / float(nq),
                    mutual_mean,
                    flat_q_lme.mean(dim=1) - flat_baseline,
                ),
                dim=1,
            ).detach().to(torch.float64).contiguous()
            if value.shape != (population, FEATURE_DIM) or not bool(
                torch.isfinite(value).all()
            ):
                raise RuntimeError("compact factored feature schema drift")
            output[q_rows[:, None], r_rows[None, :]] = value.reshape(
                q_rows.numel(), r_rows.numel(), FEATURE_DIM
            )
    return output


@dataclass(frozen=True)
class CompactCartesianFeatureBlock:
    """Tensor-only storage for one complete candidate/direction population."""

    query_action_masks: torch.Tensor
    reference_action_masks: torch.Tensor
    eligibility: torch.Tensor
    features_float32: torch.Tensor

    def __post_init__(self) -> None:
        q = torch.as_tensor(self.query_action_masks, dtype=torch.bool).detach().cpu().contiguous()
        r = torch.as_tensor(self.reference_action_masks, dtype=torch.bool).detach().cpu().contiguous()
        eligible = torch.as_tensor(self.eligibility, dtype=torch.bool).detach().cpu().contiguous()
        features = torch.as_tensor(self.features_float32).detach().cpu().contiguous()
        if (
            q.ndim != 2
            or r.ndim != 2
            or eligible.shape != (q.shape[0], r.shape[0])
            or features.shape != (q.shape[0], r.shape[0], FEATURE_DIM)
            or features.dtype != COMPACT_FEATURE_STORAGE_DTYPE
            or not bool(torch.isfinite(features).all())
            or bool(features[~eligible].ne(0.0).any())
        ):
            raise SharedPContractError("compact Cartesian feature block drift")
        object.__setattr__(self, "query_action_masks", q)
        object.__setattr__(self, "reference_action_masks", r)
        object.__setattr__(self, "eligibility", eligible)
        object.__setattr__(self, "features_float32", features)

    def restored_features(self) -> torch.Tensor:
        return self.features_float32.to(RESTORED_HEAD_DTYPE)


def make_compact_cartesian_feature_block(
    features: torch.Tensor,
    *,
    query_action_masks: torch.Tensor,
    reference_action_masks: torch.Tensor,
    eligibility: torch.Tensor,
) -> CompactCartesianFeatureBlock:
    """Store float32-provenance coordinates losslessly and zero H0 rows."""

    value = torch.as_tensor(features, dtype=torch.float64).detach().cpu().contiguous()
    compact = value.to(COMPACT_FEATURE_STORAGE_DTYPE)
    if not torch.equal(value, compact.to(torch.float64)):
        raise SharedPContractError("feature table is not lossless float32 provenance")
    flags = torch.as_tensor(eligibility, dtype=torch.bool).detach().cpu().contiguous()
    if flags.shape != value.shape[:2]:
        raise SharedPContractError("compact eligibility/table shape drift")
    compact = compact.clone()
    compact[~flags] = 0.0
    return CompactCartesianFeatureBlock(
        query_action_masks=query_action_masks,
        reference_action_masks=reference_action_masks,
        eligibility=flags,
        features_float32=compact,
    )


__all__ = [
    "SCHEMA_VERSION",
    "COMPACT_CARTESIAN_LAYOUT",
    "COMPACT_FEATURE_STORAGE_DTYPE",
    "RESTORED_HEAD_DTYPE",
    "PreparedCandidateColNomicFeatures",
    "CompactCartesianFeatureBlock",
    "prepare_candidate_colnomic_features",
    "candidate_action_colnomic_features_from_prepared",
    "candidate_action_colnomic_feature_table_from_prepared",
    "make_compact_cartesian_feature_block",
]
