"""Target-free numerical core for the frozen RC-LTR--ASLO-XF contract.

This module intentionally contains no target identity, target mask, or target
insertion API.  It implements only the prejoin mathematical contracts needed by
E0: fixed H0/H1 marginalization, base-independent component marginalization,
OOF-style incremental margins, deterministic CAL atlas derangements,
move-to-front score projection, and Aumann--Shapley closure.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Hashable, Iterable, Sequence

import numpy as np
import torch


EPS = 1.0e-12
H1_SLOTS = 8
TOTAL_SLOTS = H1_SLOTS + 1
QUADRATURE_POINTS = 32
QUADRATURE_ABS_TOL = 1.0e-10
QUADRATURE_REL_TOL = 1.0e-8


def _finite(value: torch.Tensor, *, name: str, ndim: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a tensor with ndim={ndim}")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite floating point")


def _as_float64(value: torch.Tensor, *, name: str, ndim: int) -> torch.Tensor:
    _finite(value, name=name, ndim=ndim)
    return value.to(dtype=torch.float64)


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON subset used by CAL: scalar/list-only values, UTF-8, no spaces."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _hash_key(fields: Sequence[Any]) -> bytes:
    return hashlib.sha256(canonical_json_bytes(list(fields))).digest()


def component_evidence(
    patch_evidence: torch.Tensor,
    penalties: torch.Tensor,
    legal_h1: torch.Tensor,
) -> torch.Tensor:
    """Fixed 1xH0 + 8xH1 log marginal for one reference component.

    ``patch_evidence`` is [8, P] signed held-out evidence.  Invalid H1 slots
    are exact H0 slots (logit zero).  A legal H1 is never changed to H0 based
    on its held-out score, so negative evidence remains negative evidence.
    """

    z = _as_float64(patch_evidence, name="patch_evidence", ndim=2)
    p = _as_float64(penalties, name="penalties", ndim=1)
    if z.shape[0] != H1_SLOTS or p.shape != (H1_SLOTS,):
        raise ValueError("component requires exactly eight H1 slots")
    if legal_h1.dtype != torch.bool or legal_h1.shape != (H1_SLOTS,):
        raise ValueError("legal_h1 must be bool [8]")
    logits = torch.zeros(TOTAL_SLOTS, dtype=torch.float64, device=z.device)
    raw = z.sum(dim=1) - p
    logits[1:] = torch.where(legal_h1.to(z.device), raw, torch.zeros_like(raw))
    return torch.logsumexp(logits, dim=0) - math.log(float(TOTAL_SLOTS))


def uniform_component_marginal(component_values: torch.Tensor) -> torch.Tensor:
    """Base-independent log-mean-exp over unique reference components."""

    values = _as_float64(component_values, name="component_values", ndim=1)
    if values.numel() < 1:
        raise ValueError("at least one unique component is required")
    return torch.logsumexp(values, dim=0) - math.log(float(values.numel()))


def collapse_exact_components(
    component_hashes: Sequence[str],
    shapes: Sequence[Sequence[int]],
    dtypes: Sequence[str],
    byte_lengths: Sequence[int],
    values: torch.Tensor,
) -> tuple[tuple[int, ...], torch.Tensor]:
    """Collapse only byte-identical atlas components, fail closed on hash collision."""

    value64 = _as_float64(values, name="values", ndim=1)
    n = value64.numel()
    if not all(len(field) == n for field in (component_hashes, shapes, dtypes, byte_lengths)):
        raise ValueError("component metadata lengths must match values")
    seen: dict[str, tuple[tuple[int, ...], str, int, int]] = {}
    keep: list[int] = []
    for index, digest in enumerate(component_hashes):
        signature = (tuple(int(x) for x in shapes[index]), str(dtypes[index]), int(byte_lengths[index]))
        if digest in seen:
            prior_signature, _, _, _ = seen[digest]
            if signature != prior_signature:
                raise RuntimeError("canonical component hash collision")
            continue
        seen[digest] = (signature, str(digest), index, index)
        keep.append(index)
    indices = tuple(keep)
    return indices, value64[list(indices)]


def base_margin(base_difference: torch.Tensor | float, temperature: float) -> torch.Tensor:
    """Strictly monotone, sign-preserving f0(z)=z/T with f0(0)=0."""

    if not isinstance(temperature, float) or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    value = torch.as_tensor(base_difference, dtype=torch.float64)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("base_difference must be finite")
    return value / temperature


def incremental_margin(
    base_difference: torch.Tensor | float,
    local_difference: torch.Tensor | float,
    *,
    temperature: float,
    local_scale: float,
) -> torch.Tensor:
    if not isinstance(local_scale, float) or not math.isfinite(local_scale) or local_scale < 0.0:
        raise ValueError("local_scale must be finite and non-negative")
    local = torch.as_tensor(local_difference, dtype=torch.float64)
    if not bool(torch.isfinite(local).all()):
        raise ValueError("local_difference must be finite")
    return base_margin(base_difference, temperature) + local_scale * local


@dataclass(frozen=True)
class CalDerangement:
    assigned_atlas_ids: tuple[str, ...]
    movable_fraction: float
    distinct_strata_permutations: int
    singleton_slots: tuple[int, ...]


def cal_derangement(
    slot_labels: Sequence[str],
    atlas_ids: Sequence[str],
    stratum_keys: Sequence[Sequence[Any]],
    *,
    protocol_file_sha256: str,
    query_resource_sha256: str,
    draw_index: int,
    namespace: str = "CAL_NULL_V1",
) -> CalDerangement:
    """Deterministic whole-atlas derangement specified by the frozen contract."""

    if len(slot_labels) != len(atlas_ids) or len(slot_labels) != len(stratum_keys):
        raise ValueError("CAL slot, atlas, and stratum populations must agree")
    if len(set(slot_labels)) != len(slot_labels) or len(set(atlas_ids)) != len(atlas_ids):
        raise ValueError("CAL slots and atlas identifiers must be unique")
    if draw_index < 0 or namespace != "CAL_NULL_V1":
        raise ValueError("only registered non-negative CAL draws are accepted")
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, key in enumerate(stratum_keys):
        groups.setdefault(tuple(key), []).append(index)
    assignment = list(atlas_ids)
    singleton: list[int] = []
    moved = 0
    distinct = 0
    for key, indices in sorted(groups.items(), key=lambda item: repr(item[0])):
        ordered = sorted(indices, key=lambda index: slot_labels[index])
        if len(ordered) == 1:
            singleton.extend(ordered)
            continue
        for attempt in range(1024):
            keyed: list[tuple[bytes, str]] = []
            for index in ordered:
                atlas = atlas_ids[index]
                digest = _hash_key(
                    [
                        protocol_file_sha256,
                        namespace,
                        query_resource_sha256,
                        draw_index,
                        list(key),
                        attempt,
                        atlas,
                    ]
                )
                keyed.append((digest, atlas))
            if len({digest for digest, _ in keyed}) != len(keyed):
                raise RuntimeError("CAL digest collision")
            proposal = [atlas for _, atlas in sorted(keyed, key=lambda item: (item[0], item[1]))]
            if all(atlas != atlas_ids[index] for index, atlas in zip(ordered, proposal)):
                for index, atlas in zip(ordered, proposal):
                    assignment[index] = atlas
                moved += len(ordered)
                distinct += 1
                break
        else:
            raise RuntimeError("CAL derangement attempts exhausted")
    return CalDerangement(
        assigned_atlas_ids=tuple(assignment),
        movable_fraction=float(moved) / float(len(slot_labels)) if slot_labels else 0.0,
        distinct_strata_permutations=distinct,
        singleton_slots=tuple(sorted(singleton)),
    )


def empirical_cal_exceedance(observed: torch.Tensor | float, cal_maxima: torch.Tensor) -> float:
    maxima = _as_float64(cal_maxima, name="cal_maxima", ndim=1)
    value = torch.as_tensor(observed, dtype=torch.float64)
    if value.ndim != 0 or not bool(torch.isfinite(value)) or maxima.numel() < 1:
        raise ValueError("observed and CAL maxima must be finite")
    return float(1 + int((maxima >= value).sum().item())) / float(1 + maxima.numel())


def cal_orbit_eligible(
    draws: Sequence[CalDerangement],
    *,
    minimum_draws: int = 255,
    minimum_distinct_whole_query_permutations: int = 20,
    minimum_movable_candidate_fraction: float = 0.5,
) -> bool:
    """Return whether the registered A0 CAL orbit is usable for an action.

    This is deliberately an *action* qualification, not a claim that the
    atlas orbit is exchangeable.  A caller must retain the per-draw receipts;
    an insufficient orbit maps to exact HOLD rather than a smaller null test.
    """

    if minimum_draws != 255:
        raise ValueError("A0 CAL draw count is frozen at 255")
    if len(draws) != minimum_draws:
        return False
    assignments = [draw.assigned_atlas_ids for draw in draws]
    if any(len(value) == 0 for value in assignments):
        return False
    if len(set(assignments)) < minimum_distinct_whole_query_permutations:
        return False
    return all(draw.movable_fraction >= minimum_movable_candidate_fraction for draw in draws)


def aumann_shapley_closure(
    spatial: torch.Tensor,
    mdl: torch.Tensor,
    evidence: torch.Tensor,
) -> tuple[float, float, bool]:
    """Validate the frozen fixed-32 FP64 attribution closure.

    The quadrature rule is part of the scientific contract.  This helper never
    refines it: if the registered fixed rule is outside its tolerance, callers
    must stop rather than silently increase nodes or loosen the tolerance.
    """

    value = _as_float64(spatial, name="spatial", ndim=3).sum() + torch.as_tensor(mdl, dtype=torch.float64)
    target = torch.as_tensor(evidence, dtype=torch.float64)
    if target.ndim != 0 or not bool(torch.isfinite(target)):
        raise ValueError("evidence must be a finite scalar")
    absolute = float(torch.abs(value - target))
    tolerance = max(QUADRATURE_ABS_TOL, QUADRATURE_REL_TOL * abs(float(target)))
    return absolute, tolerance, absolute <= tolerance


def stable_label_scores(row_scores: torch.Tensor, row_labels: Sequence[Hashable]) -> dict[Hashable, torch.Tensor]:
    scores = _as_float64(row_scores, name="row_scores", ndim=1)
    if len(row_labels) != scores.numel():
        raise ValueError("row_labels must match row_scores")
    result: dict[Hashable, torch.Tensor] = {}
    for index, label in enumerate(row_labels):
        candidate = scores[index]
        if label not in result or bool(candidate > result[label]):
            result[label] = candidate
    return result


def stable_label_ranking(label_scores: dict[Hashable, torch.Tensor]) -> tuple[Hashable, ...]:
    return tuple(sorted(label_scores, key=lambda label: (-float(label_scores[label]), repr(label))))


def move_to_front_projection(
    row_scores: torch.Tensor,
    row_labels: Sequence[Hashable],
    *,
    challenger: Hashable,
) -> tuple[torch.Tensor, tuple[Hashable, ...]]:
    """Materialize the contract's stable challenger move-to-front row tensor."""

    original_scores = _as_float64(row_scores, name="row_scores", ndim=1)
    label_scores = stable_label_scores(original_scores, row_labels)
    original_order = stable_label_ranking(label_scores)
    if challenger not in label_scores:
        raise ValueError("challenger is not a full-gallery label")
    winner = original_order[0]
    if challenger == winner:
        raise ValueError("challenger must differ from base winner")
    new_order = (challenger, winner) + tuple(
        label for label in original_order if label not in {challenger, winner}
    )
    old_rank = original_order.index(challenger)
    target_label_scores = dict(label_scores)
    for rank in range(old_rank + 1):
        target_label_scores[new_order[rank]] = label_scores[original_order[rank]]
    final = original_scores.clone()
    for index, label in enumerate(row_labels):
        final[index] = final[index] + (target_label_scores[label] - label_scores[label])
    return final, new_order


@dataclass(frozen=True)
class SwitchDecision:
    base_winner: Hashable
    challenger: Hashable
    observed_margin: float
    cal_exceedance: float
    switched: bool
    final_row_scores: torch.Tensor
    final_order: tuple[Hashable, ...]


def decide_one_switch(
    row_scores: torch.Tensor,
    row_labels: Sequence[Hashable],
    candidate_labels: Sequence[Hashable],
    local_evidence: torch.Tensor,
    cal_local_evidence: torch.Tensor,
    direction_local_evidence: torch.Tensor,
    legal_h1_by_direction: torch.Tensor,
    *,
    temperature: float,
    local_scale: float,
    exceedance_threshold: float = 0.05,
) -> SwitchDecision:
    """Target-free one-action policy with empirical CAL exceedance and exact HOLD."""

    base_rows = _as_float64(row_scores, name="row_scores", ndim=1)
    candidate = tuple(candidate_labels)
    if len(candidate) < 2 or len(set(candidate)) != len(candidate):
        raise ValueError("candidate_labels must contain unique labels")
    label_scores = stable_label_scores(base_rows, row_labels)
    if any(label not in label_scores for label in candidate):
        raise ValueError("candidate label missing from full gallery")
    evidence = _as_float64(local_evidence, name="local_evidence", ndim=1)
    cal = _as_float64(cal_local_evidence, name="cal_local_evidence", ndim=2)
    directional = _as_float64(direction_local_evidence, name="direction_local_evidence", ndim=2)
    if evidence.shape != (len(candidate),) or cal.shape[1:] != (len(candidate),):
        raise ValueError("local evidence must align with candidates")
    if directional.shape != (2, len(candidate)):
        raise ValueError("two directional evidence rows are required")
    if cal.shape[0] != 255:
        raise ValueError("A0 requires exactly 255 registered CAL draws")
    if legal_h1_by_direction.dtype != torch.bool or legal_h1_by_direction.shape != (2, len(candidate)):
        raise ValueError("legal_h1_by_direction must be bool [2,C]")
    if not 0.0 < exceedance_threshold < 1.0:
        raise ValueError("exceedance_threshold must be in (0,1)")
    ranked_candidates = sorted(candidate, key=lambda label: (-float(label_scores[label]), repr(label)))
    winner = ranked_candidates[0]
    winner_index = candidate.index(winner)
    base_vector = torch.stack([label_scores[label] for label in candidate])
    base_differences = base_vector - base_vector[winner_index]
    margins = incremental_margin(
        base_differences, evidence - evidence[winner_index], temperature=temperature, local_scale=local_scale
    )
    margins[winner_index] = -torch.inf
    challenger_index = int(torch.argmax(margins).item())
    challenger = candidate[challenger_index]
    observed = margins[challenger_index]
    cal_margins = base_margin(base_differences.unsqueeze(0), temperature) + local_scale * (
        cal - cal[:, winner_index : winner_index + 1]
    )
    cal_margins[:, winner_index] = -torch.inf
    exceedance = empirical_cal_exceedance(observed, cal_margins.max(dim=1).values)
    directional_margins = base_margin(base_differences.unsqueeze(0), temperature) + local_scale * (
        directional - directional[:, winner_index : winner_index + 1]
    )
    direction_ok = bool((directional_margins[:, challenger_index] > 0.0).all())
    legal_ok = bool(legal_h1_by_direction[:, challenger_index].all()) and bool(
        (directional[:, challenger_index] > 0.0).all()
    )
    switched = bool(float(observed) > 0.0 and exceedance <= exceedance_threshold and direction_ok and legal_ok)
    if not switched:
        return SwitchDecision(
            base_winner=winner,
            challenger=challenger,
            observed_margin=float(observed),
            cal_exceedance=exceedance,
            switched=False,
            final_row_scores=row_scores,
            final_order=stable_label_ranking(label_scores),
        )
    final_rows, final_order = move_to_front_projection(base_rows, row_labels, challenger=challenger)
    return SwitchDecision(
        base_winner=winner,
        challenger=challenger,
        observed_margin=float(observed),
        cal_exceedance=exceedance,
        switched=True,
        final_row_scores=final_rows,
        final_order=final_order,
    )


def _gauss_legendre(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(QUADRATURE_POINTS)
    return (
        torch.tensor((nodes + 1.0) / 2.0, dtype=torch.float64, device=device),
        torch.tensor(weights / 2.0, dtype=torch.float64, device=device),
    )


def aumann_shapley(
    patch_evidence_by_component: torch.Tensor,
    penalties_by_component: torch.Tensor,
    legal_h1_by_component: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return spatial attribution, non-spatial MDL attribution, and E_g.

    Inputs are [U, 8, P], [U, 8], and bool [U, 8].  Scaling both signed patch
    evidence and MDL penalty from zero gives an H0/H1/component path whose
    spatial plus MDL attribution closes to the deployment local evidence.
    """

    z = _as_float64(patch_evidence_by_component, name="patch_evidence_by_component", ndim=3)
    penalties = _as_float64(penalties_by_component, name="penalties_by_component", ndim=2)
    if z.shape[:2] != penalties.shape or z.shape[1] != H1_SLOTS:
        raise ValueError("Aumann inputs require [U,8,P] and [U,8]")
    if legal_h1_by_component.dtype != torch.bool or legal_h1_by_component.shape != penalties.shape:
        raise ValueError("legal_h1_by_component must be bool [U,8]")
    nodes, weights = _gauss_legendre(z.device)
    spatial = torch.zeros_like(z)
    mdl = torch.zeros((), dtype=torch.float64, device=z.device)
    valid = legal_h1_by_component.to(z.device)
    slot_sum = z.sum(dim=2) - penalties
    for node, weight in zip(nodes, weights):
        logits = torch.zeros((z.shape[0], TOTAL_SLOTS), dtype=torch.float64, device=z.device)
        logits[:, 1:] = torch.where(valid, node * slot_sum, torch.zeros_like(slot_sum))
        slot_resp = torch.softmax(logits, dim=1)[:, 1:]
        component_e = torch.logsumexp(logits, dim=1) - math.log(float(TOTAL_SLOTS))
        component_resp = torch.softmax(component_e, dim=0)
        responsibility = component_resp[:, None] * slot_resp * valid.to(dtype=torch.float64)
        spatial = spatial + weight * responsibility[:, :, None] * z
        mdl = mdl + weight * (responsibility * (-penalties)).sum()
    component_values = torch.stack(
        [component_evidence(z[index], penalties[index], valid[index]) for index in range(z.shape[0])]
    )
    evidence = uniform_component_marginal(component_values)
    return spatial, mdl, evidence


def four_ledger(
    *,
    base_clean: float,
    base_corrupt: float,
    local_clean: float,
    local_corrupt_fixed: float,
    total_margin_clean: float,
    total_margin_corrupt: float,
    local_scale: float,
) -> dict[str, float]:
    """Continuous margin-ledger closure; actions/ranks remain outside this ledger."""

    values = (
        base_clean,
        base_corrupt,
        local_clean,
        local_corrupt_fixed,
        total_margin_clean,
        total_margin_corrupt,
        local_scale,
    )
    if any(not isinstance(value, (float, int)) or not math.isfinite(float(value)) for value in values):
        raise ValueError("four-ledger values must be finite")
    if local_scale < 0.0:
        raise ValueError("local_scale must be non-negative")
    d_base = float(base_clean - base_corrupt)
    d_local_fixed = float(local_clean - local_corrupt_fixed)
    d_total = float(total_margin_clean - total_margin_corrupt)
    d_rehyp = d_total - d_base - float(local_scale) * d_local_fixed
    return {
        "D_B": d_base,
        "D_L_fixed": d_local_fixed,
        "D_total_margin": d_total,
        "D_rehyp": d_rehyp,
    }


def fit_homography_dlt(reference_xy: torch.Tensor, query_xy: torch.Tensor) -> torch.Tensor:
    """Deterministic FP64 normalized DLT used only by the E0 geometry fixture."""

    reference = _as_float64(reference_xy, name="reference_xy", ndim=2)
    query = _as_float64(query_xy, name="query_xy", ndim=2)
    if reference.shape != query.shape or reference.shape[1:] != (2,) or reference.shape[0] < 4:
        raise ValueError("reference/query points must be equal [N,2], N>=4")

    def normalize(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        center = points.mean(dim=0)
        mean_distance = torch.linalg.vector_norm(points - center, dim=1).mean()
        if float(mean_distance) <= EPS:
            raise ValueError("degenerate point set")
        scale = math.sqrt(2.0) / float(mean_distance)
        transform = torch.tensor(
            [[scale, 0.0, -scale * float(center[0])], [0.0, scale, -scale * float(center[1])], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
            device=points.device,
        )
        homogeneous = torch.cat([points, torch.ones((points.shape[0], 1), dtype=torch.float64, device=points.device)], dim=1)
        return (transform @ homogeneous.T).T[:, :2], transform

    source, source_t = normalize(reference)
    target, target_t = normalize(query)
    rows: list[torch.Tensor] = []
    for (x, y), (u, v) in zip(source, target):
        zero = torch.zeros((), dtype=torch.float64, device=source.device)
        one = torch.ones((), dtype=torch.float64, device=source.device)
        rows.append(torch.stack([-x, -y, -one, zero, zero, zero, u * x, u * y, u]))
        rows.append(torch.stack([zero, zero, zero, -x, -y, -one, v * x, v * y, v]))
    design = torch.stack(rows)
    if int(torch.linalg.matrix_rank(design)) < 8:
        raise ValueError("degenerate homography design")
    _, _, vh = torch.linalg.svd(design, full_matrices=True)
    homography = torch.linalg.inv(target_t) @ vh[-1].reshape(3, 3) @ source_t
    if abs(float(homography[2, 2])) <= EPS:
        raise ValueError("invalid homography scale")
    return homography / homography[2, 2]


def apply_homography(homography: torch.Tensor, reference_xy: torch.Tensor) -> torch.Tensor:
    matrix = _as_float64(homography, name="homography", ndim=2)
    points = _as_float64(reference_xy, name="reference_xy", ndim=2)
    if matrix.shape != (3, 3) or points.shape[1:] != (2,):
        raise ValueError("homography must be [3,3] and points [N,2]")
    homogeneous = torch.cat([points, torch.ones((points.shape[0], 1), dtype=torch.float64, device=points.device)], dim=1)
    mapped = (matrix @ homogeneous.T).T
    if bool((mapped[:, 2].abs() <= EPS).any()):
        raise ValueError("homography maps point to infinity")
    return mapped[:, :2] / mapped[:, 2:3]
