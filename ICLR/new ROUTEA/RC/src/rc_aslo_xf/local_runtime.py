"""Target-free proposal/verification runtime for the frozen A0 local path.

The module deliberately has no label, rank, slot, target, or evaluator API.
It consumes only already-materialized P/V descriptor grids and coordinates.
The proposal and verification query subsets are separated by a two-cell halo:
with the frozen 16 px stride and at most 22 px receptive field, an A feature
cannot overlap a B feature that is used to verify its proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np
import torch

from .core import H1_SLOTS, apply_homography, component_evidence, fit_homography_dlt
from .verifier import JointDensityRatio, estimated_log_ratio


SIGMA = 0.08
PI_MATCH = 0.10
MAX_PROPOSAL_MATCHES = 32
NMS_RADIUS_GRID_CELLS = 2


@dataclass(frozen=True)
class CrossfitSplit:
    propose_a: torch.Tensor
    verify_b: torch.Tensor
    propose_b: torch.Tensor
    verify_a: torch.Tensor


@dataclass(frozen=True)
class HypothesisSet:
    matrices: torch.Tensor  # [8,3,3], invalid matrices are identity only as a placeholder
    legal: torch.Tensor  # [8]
    match_count: torch.Tensor  # [8]


def crossfit_split(coordinate: torch.Tensor) -> CrossfitSplit:
    """Build the frozen 4x4 chessboard A/B subsets with a two-cell halo."""

    if coordinate.ndim != 2 or coordinate.shape[1] != 2 or coordinate.shape[0] != 1024:
        raise ValueError("A0 expects a canonical 32x32 coordinate grid")
    x = torch.round(coordinate[:, 0] * 32.0 - 0.5).to(torch.long)
    y = torch.round(coordinate[:, 1] * 32.0 - 0.5).to(torch.long)
    # A-proposal positions and B-verification positions have Chebyshev distance
    # at least two grid cells.  Swap roles for the other direction.
    a = ((x.remainder(4) == 0) & (y.remainder(4) == 0)).nonzero().flatten()
    b = ((x.remainder(4) == 2) & (y.remainder(4) == 2)).nonzero().flatten()
    if a.numel() != 64 or b.numel() != 64 or bool(torch.isin(a, b).any()):
        raise RuntimeError("frozen cross-fit grid construction drift")
    return CrossfitSplit(propose_a=a, verify_b=b, propose_b=b, verify_a=a)


def _require_features(query_p: torch.Tensor, query_v: torch.Tensor, reference_p: torch.Tensor, reference_v: torch.Tensor, coordinate: torch.Tensor) -> None:
    tensors = (query_p, query_v, reference_p, reference_v, coordinate)
    if any(value.ndim != 2 or not value.is_floating_point() or not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("local runtime requires finite rank-2 tensors")
    if query_p.shape != (1024, 32) or query_v.shape != (1024, 32) or reference_p.shape != (1024, 32) or reference_v.shape != (1024, 32) or coordinate.shape != (1024, 2):
        raise ValueError("local runtime requires frozen 32x32 P/V grids")


def mutual_matches(query_p: torch.Tensor, reference_p: torch.Tensor, proposal_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Result-blind mutual P-channel matches for one candidate reference."""

    if proposal_indices.ndim != 1 or proposal_indices.numel() != 64:
        raise ValueError("proposal subset must contain 64 frozen cells")
    similarity = query_p[proposal_indices].float() @ reference_p.float().T
    q_to_r = similarity.argmax(dim=1)
    r_to_q = similarity.argmax(dim=0)
    local_q = torch.arange(proposal_indices.numel(), device=similarity.device)
    keep = r_to_q[q_to_r] == local_q
    q = proposal_indices[keep]
    r = q_to_r[keep]
    score = similarity[local_q[keep], r]
    order = torch.argsort(score, descending=True, stable=True)
    # Frozen grid-space NMS: mutual matches alone may consist of adjacent
    # texture cells and form numerically valid but physically uninformative H.
    chosen: list[int] = []
    for candidate in order.tolist():
        if all(int(torch.max(torch.abs(_grid_index(q[candidate]) - _grid_index(q[prior])))) >= NMS_RADIUS_GRID_CELLS for prior in chosen):
            chosen.append(candidate)
        if len(chosen) == MAX_PROPOSAL_MATCHES:
            break
    selected = torch.tensor(chosen, dtype=torch.long, device=q.device)
    return q[selected], r[selected], score[selected]


def _grid_index(index: torch.Tensor) -> torch.Tensor:
    """Canonical integer (x,y) grid index for NMS only."""

    value = index.to(torch.long)
    return torch.stack((value.remainder(32), torch.div(value, 32, rounding_mode="floor")))


def propose_hypotheses(query_p: torch.Tensor, reference_p: torch.Tensor, coordinate: torch.Tensor, proposal_indices: torch.Tensor) -> HypothesisSet:
    """Generate exactly eight pre-verification H1 slots; invalid slots are H0."""

    q_index, r_index, _ = mutual_matches(query_p, reference_p, proposal_indices)
    matrices = torch.eye(3, dtype=torch.float64, device=query_p.device).repeat(H1_SLOTS, 1, 1)
    legal = torch.zeros(H1_SLOTS, dtype=torch.bool, device=query_p.device)
    counts = torch.zeros(H1_SLOTS, dtype=torch.long, device=query_p.device)
    if q_index.numel() < 4:
        return HypothesisSet(matrices, legal, counts)
    # MAGSAC++ reads only P-channel proposal matches.  The eight slots are
    # deterministic leave-one-stride subsets, so every candidate has the same
    # fixed multiplicity and none can gain evidence by proposing more models.
    slot = 0
    cv2.setRNGSeed(481516234)
    for offset in range(H1_SLOTS):
        keep = torch.arange(q_index.numel(), device=q_index.device).remainder(H1_SLOTS) != offset
        if int(keep.sum()) < 4:
            continue
        source = coordinate[r_index[keep]].detach().cpu().numpy().astype(np.float64)
        target = coordinate[q_index[keep]].detach().cpu().numpy().astype(np.float64)
        try:
            raw, mask = cv2.findHomography(source, target, method=cv2.USAC_MAGSAC, ransacReprojThreshold=SIGMA, maxIters=10000, confidence=0.999)
            if raw is None or mask is None:
                continue
            matrix = torch.as_tensor(raw, dtype=torch.float64, device=query_p.device)
            if not bool(torch.isfinite(matrix).all()) or float(torch.linalg.cond(matrix)) > 1.0e6:
                continue
            projected = apply_homography(matrix, coordinate)
            if not bool(torch.isfinite(projected).all()):
                continue
            # Inlier test uses only proposal P matches and fixed sigma.
            mapped = apply_homography(matrix, coordinate[r_index])
            inliers = torch.linalg.vector_norm(mapped - coordinate[q_index], dim=1) <= SIGMA
            if int(inliers.sum()) < 4:
                continue
        except (cv2.error, RuntimeError, ValueError):
            continue
        matrices[slot] = matrix
        legal[slot] = True
        counts[slot] = int(inliers.sum())
        slot += 1
    return HypothesisSet(matrices, legal, counts)


def verification_features(query_v: torch.Tensor, reference_v: torch.Tensor, coordinate: torch.Tensor, verify_indices: torch.Tensor, hypotheses: HypothesisSet) -> torch.Tensor:
    """Return target-free [8,64,4] features for a fold-local joint verifier."""

    if verify_indices.ndim != 1 or verify_indices.numel() != 64:
        raise ValueError("verification subset must contain 64 frozen cells")
    output = torch.zeros((H1_SLOTS, verify_indices.numel(), 4), dtype=torch.float64, device=query_v.device)
    q = query_v[verify_indices].to(torch.float64)
    r = reference_v.to(torch.float64)
    similarity = q @ r.T
    reference_coordinate = coordinate.to(torch.float64)
    query_coordinate = coordinate[verify_indices].to(torch.float64)
    glyph_similarity = q[:, :16] @ r[:, :16].T
    for slot in range(H1_SLOTS):
        if not bool(hypotheses.legal[slot]):
            continue
        mapped = apply_homography(hypotheses.matrices[slot], reference_coordinate)
        distance2 = (query_coordinate[:, None, :] - mapped[None, :, :]).square().sum(dim=-1)
        geometry = -0.5 * distance2 / (SIGMA * SIGMA) - math.log(2.0 * math.pi * SIGMA * SIGMA)
        # All four are local visual/geometric observables.  The joint verifier
        # alone learns their correlated combination; no independent LLRs are
        # summed and labels are not available in this forward path.
        nearest = geometry.argmax(dim=1)
        appearance = similarity.gather(1, nearest[:, None]).squeeze(1)
        glyph_visual = glyph_similarity.gather(1, nearest[:, None]).squeeze(1)
        geometry_nearest = geometry.gather(1, nearest[:, None]).squeeze(1)
        output[slot] = torch.stack((appearance, geometry_nearest, glyph_visual, appearance * geometry_nearest), dim=-1)
    return output


def evidence_from_features(features: torch.Tensor, hypotheses: HypothesisSet, verifier: JointDensityRatio, *, sampling_log_odds: float) -> torch.Tensor:
    """Convert a frozen target-free feature ledger into signed patch evidence."""

    if features.shape != (H1_SLOTS, 64, 4):
        raise ValueError("verification feature ledger shape drift")
    log_ratio = estimated_log_ratio(verifier, features, sampling_log_odds=sampling_log_odds)
    log_mean_ratio = log_ratio  # one H1-consistent nearest reference pair per patch
    mixture = torch.logaddexp(
        torch.full_like(log_mean_ratio, math.log1p(-PI_MATCH)),
        log_mean_ratio + math.log(PI_MATCH),
    ) / 64.0
    return torch.where(hypotheses.legal[:, None], mixture, torch.zeros_like(mixture))


def candidate_directional_features(query_p: torch.Tensor, query_v: torch.Tensor, reference_p: torch.Tensor, reference_v: torch.Tensor, coordinate: torch.Tensor) -> tuple[HypothesisSet, HypothesisSet, torch.Tensor, torch.Tensor]:
    """Materialize both P/V-separated target-free verifier ledgers."""

    _require_features(query_p, query_v, reference_p, reference_v, coordinate)
    split = crossfit_split(coordinate)
    first = propose_hypotheses(query_p, reference_p, coordinate, split.propose_a)
    second = propose_hypotheses(query_p, reference_p, coordinate, split.propose_b)
    first_features = verification_features(query_v, reference_v, coordinate, split.verify_b, first)
    second_features = verification_features(query_v, reference_v, coordinate, split.verify_a, second)
    return first, second, first_features, second_features


def candidate_directional_evidence(query_p: torch.Tensor, query_v: torch.Tensor, reference_p: torch.Tensor, reference_v: torch.Tensor, coordinate: torch.Tensor, verifier: JointDensityRatio | None = None, *, sampling_log_odds: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute two cross-fit component evidences without any target join."""

    first, second, first_features, second_features = candidate_directional_features(query_p, query_v, reference_p, reference_v, coordinate)
    # E2 engineering smoke intentionally uses a zero-initialized verifier;
    # A0 replaces it with the fitted fold-local model after prejoin sealing.
    verifier = JointDensityRatio().to(query_p.device) if verifier is None else verifier
    first_z = evidence_from_features(first_features, first, verifier, sampling_log_odds=sampling_log_odds)
    second_z = evidence_from_features(second_features, second, verifier, sampling_log_odds=sampling_log_odds)
    penalty = torch.full((H1_SLOTS,), 8.0 * math.log(64.0) / (2.0 * 64.0), dtype=torch.float64, device=query_p.device)
    e_first = component_evidence(first_z, penalty, first.legal)
    e_second = component_evidence(second_z, penalty, second.legal)
    return torch.stack((e_first, e_second)), torch.stack((first.legal, second.legal)), first_z, second_z
