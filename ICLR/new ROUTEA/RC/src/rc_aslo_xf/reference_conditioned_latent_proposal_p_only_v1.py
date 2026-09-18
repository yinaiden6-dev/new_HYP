"""Minimal P-only RoMa/ColNomic atom-to-connected-hypothesis core.

This module has no retrieval-base, action, verification, or ownership import.
It implements only candidate-conditioned proposal generation and its matched
engineering comparators/controls.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


ATOM_FEATURE_DIM = 9
SELECTOR_PARAMETER_COUNT = 10
H1_SLOT_COUNT = 8
MIN_QUERY_PATCHES = 4
MIN_REFERENCE_ATOMS = 2
REFERENCE_CHEBYSHEV = 2
EPS = 1.0e-12


def tensor_sha256(value: torch.Tensor) -> str:
    item = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(item.dtype).encode("ascii"))
    digest.update(json.dumps(list(item.shape), separators=(",", ":")).encode("ascii"))
    digest.update(item.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def coordinate_multiset_sha256(value: torch.Tensor) -> str:
    rows = sorted(tuple(int(item) for item in row) for row in torch.as_tensor(value).detach().cpu().tolist())
    return tensor_sha256(torch.tensor(rows, dtype=torch.long))


def logical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def sparsemax(logits: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(logits)
    if value.ndim != 1 or value.numel() < 2 or not value.is_floating_point():
        raise ValueError("sparsemax expects one floating vector with at least two entries")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("sparsemax input must be finite")
    ordered, _ = torch.sort(value, descending=True)
    steps = torch.arange(1, value.numel() + 1, dtype=value.dtype, device=value.device)
    cumulative = ordered.cumsum(0)
    support = 1.0 + steps * ordered > cumulative
    count = int(support.sum().detach().cpu())
    if count <= 0:
        raise RuntimeError("empty sparsemax support")
    threshold = (cumulative[count - 1] - 1.0) / float(count)
    weights = (value - threshold).clamp_min(0.0)
    return weights / weights.sum()


@dataclass(frozen=True)
class ProposalAtomBank:
    query_resource_key: str
    candidate_resource_key: str
    reference_resource_key: str
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    features: torch.Tensor
    signed_evidence: torch.Tensor
    valid_mask: torch.Tensor
    query_valid_axis: torch.Tensor
    reference_valid_axis: torch.Tensor
    source_query_indices: torch.Tensor
    source_reference_indices: torch.Tensor
    query_indices: torch.Tensor
    reference_indices: torch.Tensor
    query_coordinate_axis: torch.Tensor
    reference_coordinate_axis: torch.Tensor
    query_centres_xy_axis: torch.Tensor
    reference_centres_xy_axis: torch.Tensor
    source_query_rc: torch.Tensor
    source_reference_rc: torch.Tensor
    query_rc: torch.Tensor
    reference_rc: torch.Tensor
    source_forward_reference_xy: torch.Tensor
    source_reverse_query_xy: torch.Tensor
    forward_reference_xy: torch.Tensor
    reverse_query_xy: torch.Tensor
    overlap_pair: torch.Tensor
    precision_quality: torch.Tensor
    cycle_error: torch.Tensor
    source_assignment_sha256: str
    assignment_sha256: str
    coordinate_binding_sha256: str
    source_binding_sha256: str
    control_namespace: str
    query_coordinate_permutation: torch.Tensor
    reference_coordinate_permutation: torch.Tensor


def _positive_shape(value: tuple[int, int], name: str) -> tuple[int, int]:
    shape = tuple(int(item) for item in value)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"{name} must be a positive two-dimensional shape")
    return shape


def validate_atom_bank(bank: ProposalAtomBank) -> ProposalAtomBank:
    qshape = _positive_shape(bank.query_grid_shape, "query grid")
    rshape = _positive_shape(bank.reference_grid_shape, "reference grid")
    features = torch.as_tensor(bank.features)
    count = features.shape[0] if features.ndim == 2 else -1
    qcount, rcount = qshape[0] * qshape[1], rshape[0] * rshape[1]
    if features.shape != (count, ATOM_FEATURE_DIM) or count != qcount:
        raise ValueError("atom features must have shape [N,9]")
    expected = {
        "signed evidence": (torch.as_tensor(bank.signed_evidence), (count,)),
        "valid mask": (torch.as_tensor(bank.valid_mask), (count,)),
        "query valid axis": (torch.as_tensor(bank.query_valid_axis), (qcount,)),
        "reference valid axis": (torch.as_tensor(bank.reference_valid_axis), (rcount,)),
        "source query indices": (torch.as_tensor(bank.source_query_indices), (count,)),
        "source reference indices": (torch.as_tensor(bank.source_reference_indices), (count,)),
        "query indices": (torch.as_tensor(bank.query_indices), (count,)),
        "reference indices": (torch.as_tensor(bank.reference_indices), (count,)),
        "query coordinate axis": (torch.as_tensor(bank.query_coordinate_axis), (qcount, 2)),
        "reference coordinate axis": (torch.as_tensor(bank.reference_coordinate_axis), (rcount, 2)),
        "query centre axis": (torch.as_tensor(bank.query_centres_xy_axis), (qcount, 2)),
        "reference centre axis": (torch.as_tensor(bank.reference_centres_xy_axis), (rcount, 2)),
        "source query coordinates": (torch.as_tensor(bank.source_query_rc), (count, 2)),
        "source reference coordinates": (torch.as_tensor(bank.source_reference_rc), (count, 2)),
        "query coordinates": (torch.as_tensor(bank.query_rc), (count, 2)),
        "reference coordinates": (torch.as_tensor(bank.reference_rc), (count, 2)),
        "source forward coordinates": (torch.as_tensor(bank.source_forward_reference_xy), (count, 2)),
        "source reverse coordinates": (torch.as_tensor(bank.source_reverse_query_xy), (count, 2)),
        "forward coordinates": (torch.as_tensor(bank.forward_reference_xy), (count, 2)),
        "reverse coordinates": (torch.as_tensor(bank.reverse_query_xy), (count, 2)),
        "overlap pair": (torch.as_tensor(bank.overlap_pair), (count, 2)),
        "precision quality": (torch.as_tensor(bank.precision_quality), (count,)),
        "cycle error": (torch.as_tensor(bank.cycle_error), (count,)),
        "query coordinate permutation": (torch.as_tensor(bank.query_coordinate_permutation), (qcount,)),
        "reference coordinate permutation": (torch.as_tensor(bank.reference_coordinate_permutation), (rcount,)),
    }
    for name, (value, shape) in expected.items():
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape drift")
    if not features.is_floating_point() or not torch.as_tensor(bank.signed_evidence).is_floating_point():
        raise ValueError("feature/evidence tensors must be floating")
    integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    for name in (
        "source_query_indices", "source_reference_indices", "query_indices", "reference_indices",
        "query_coordinate_axis", "reference_coordinate_axis", "source_query_rc", "source_reference_rc",
        "query_rc", "reference_rc", "query_coordinate_permutation", "reference_coordinate_permutation",
    ):
        if torch.as_tensor(getattr(bank, name)).dtype not in integer_dtypes:
            raise ValueError(f"{name} must have integer dtype")
    finite = (features, bank.signed_evidence, bank.query_centres_xy_axis,
              bank.reference_centres_xy_axis, bank.source_forward_reference_xy,
              bank.source_reverse_query_xy, bank.forward_reference_xy, bank.reverse_query_xy,
              bank.overlap_pair, bank.precision_quality, bank.cycle_error)
    if not all(bool(torch.isfinite(torch.as_tensor(item)).all()) for item in finite):
        raise ValueError("atom bank contains non-finite arithmetic")
    if any(
        torch.as_tensor(value).dtype != torch.bool
        for value in (bank.valid_mask, bank.query_valid_axis, bank.reference_valid_axis)
    ):
        raise ValueError("valid masks must be boolean")
    valid = torch.as_tensor(bank.valid_mask, dtype=torch.bool)
    qaxis = torch.as_tensor(bank.query_coordinate_axis, dtype=torch.long)
    raxis = torch.as_tensor(bank.reference_coordinate_axis, dtype=torch.long)
    expected_qaxis = torch.stack((torch.arange(qcount) // qshape[1], torch.arange(qcount) % qshape[1]), dim=1)
    expected_raxis = torch.stack((torch.arange(rcount) // rshape[1], torch.arange(rcount) % rshape[1]), dim=1)
    if not torch.equal(qaxis.cpu(), expected_qaxis) or not torch.equal(raxis.cpu(), expected_raxis):
        raise ValueError("canonical coordinate axis is not row-major")
    source_qindex = torch.as_tensor(bank.source_query_indices, dtype=torch.long)
    source_rindex = torch.as_tensor(bank.source_reference_indices, dtype=torch.long)
    active_qindex = torch.as_tensor(bank.query_indices, dtype=torch.long)
    active_rindex = torch.as_tensor(bank.reference_indices, dtype=torch.long)
    if not bool(((source_qindex >= 0) & (source_qindex < qcount) & (active_qindex >= 0) & (active_qindex < qcount)).all()):
        raise ValueError("query index outside canonical axis")
    if not bool(((source_rindex >= 0) & (source_rindex < rcount) & (active_rindex >= 0) & (active_rindex < rcount)).all()):
        raise ValueError("reference index outside canonical axis")
    if not torch.equal(torch.as_tensor(bank.source_query_rc, dtype=torch.long), qaxis[source_qindex]):
        raise ValueError("source query index/coordinate binding drift")
    if not torch.equal(torch.as_tensor(bank.source_reference_rc, dtype=torch.long), raxis[source_rindex]):
        raise ValueError("source reference index/coordinate binding drift")
    if not torch.equal(torch.as_tensor(bank.query_rc, dtype=torch.long), qaxis[active_qindex]):
        raise ValueError("active query index/coordinate binding drift")
    if not torch.equal(torch.as_tensor(bank.reference_rc, dtype=torch.long), raxis[active_rindex]):
        raise ValueError("active reference index/coordinate binding drift")
    qperm = torch.as_tensor(bank.query_coordinate_permutation, dtype=torch.long)
    rperm = torch.as_tensor(bank.reference_coordinate_permutation, dtype=torch.long)
    if not torch.equal(torch.sort(qperm).values.cpu(), torch.arange(qcount)) or not torch.equal(
        torch.sort(rperm).values.cpu(), torch.arange(rcount)
    ):
        raise ValueError("coordinate control is not a complete permutation")
    if not torch.equal(source_qindex.cpu(), torch.arange(qcount)):
        raise ValueError("source atom axis must contain one row per canonical query token")
    if not torch.equal(active_qindex.cpu(), qperm[source_qindex].cpu()) or not torch.equal(
        active_rindex.cpu(), rperm[source_rindex].cpu()
    ):
        raise ValueError("active index does not replay from source index and coordinate permutation")
    if bank.control_namespace != "REAL" and (
        not torch.equal(
            torch.as_tensor(bank.reverse_query_xy),
            torch.as_tensor(bank.query_centres_xy_axis)[active_qindex],
        )
        or not torch.equal(
            torch.as_tensor(bank.forward_reference_xy),
            torch.as_tensor(bank.reference_centres_xy_axis)[active_rindex],
        )
    ):
        raise ValueError("controlled continuous coordinate provenance does not replay from active indices")
    evidence = torch.as_tensor(bank.signed_evidence)
    if not torch.equal(evidence, features[:, -1]):
        raise ValueError("signed evidence must be feature column eight")
    if bool((~valid).any()) and (
        int(torch.count_nonzero(features[~valid])) != 0 or int(torch.count_nonzero(evidence[~valid])) != 0
    ):
        raise ValueError("invalid atom feature/evidence rows must be exact zero")
    overlap = torch.as_tensor(bank.overlap_pair)
    precision = torch.as_tensor(bank.precision_quality)
    cycle = torch.as_tensor(bank.cycle_error)
    if not bool(((overlap >= 0.0) & (overlap <= 1.0)).all()):
        raise ValueError("overlap values outside [0,1]")
    if not bool(((precision >= 0.0) & (precision <= 1.0)).all()) or not bool((cycle >= 0.0).all()):
        raise ValueError("precision/cycle domain drift")
    if bool(valid.any()) and not torch.equal(overlap[valid], features[valid, 2:4]):
        raise ValueError("overlap provenance does not equal selector feature columns")
    source_forward = torch.as_tensor(bank.source_forward_reference_xy)
    source_reverse = torch.as_tensor(bank.source_reverse_query_xy)
    reference_centres = torch.as_tensor(bank.reference_centres_xy_axis)
    reference_valid = torch.as_tensor(bank.reference_valid_axis, dtype=torch.bool)
    nearest_distance = torch.cdist(source_forward.to(torch.float32), reference_centres.to(torch.float32))
    nearest_distance[:, ~reference_valid] = torch.inf
    replay_reference_indices = nearest_distance.argmin(dim=1)
    if not torch.equal(source_rindex.cpu(), replay_reference_indices.cpu()):
        raise ValueError("source reference nearest-centre assignment does not replay")
    replay_cycle = torch.linalg.vector_norm(
        source_reverse.to(torch.float64)
        - torch.as_tensor(bank.query_centres_xy_axis, dtype=torch.float64)[source_qindex],
        dim=1,
    )
    if not torch.allclose(cycle.to(torch.float64), replay_cycle, rtol=1.0e-6, atol=1.0e-7):
        raise ValueError("source cycle-error provenance does not replay")
    replay_valid = (
        torch.as_tensor(bank.query_valid_axis, dtype=torch.bool)[source_qindex]
        & source_forward.abs().lt(1.0).all(dim=1)
        & reference_valid[source_rindex]
    )
    if not torch.equal(valid.cpu(), replay_valid.cpu()):
        raise ValueError("atom valid mask does not replay from source geometry")
    if bool(valid.any()):
        fv = features[valid]
        expected_reciprocal = torch.sqrt((fv[:, 2] * fv[:, 3]).clamp_min(0.0))
        expected_cycle_quality = torch.exp(-4.0 * cycle[valid])
        relations = (
            torch.allclose(fv[:, 4], expected_reciprocal, rtol=1.0e-6, atol=1.0e-7),
            torch.allclose(fv[:, 5], expected_cycle_quality, rtol=1.0e-6, atol=1.0e-7),
            torch.allclose(fv[:, 6], precision[valid], rtol=0.0, atol=0.0),
            torch.allclose(fv[:, 7], fv[:, 4] * fv[:, 1], rtol=1.0e-6, atol=1.0e-7),
            torch.allclose(fv[:, 8], fv[:, 7] * fv[:, 5], rtol=1.0e-6, atol=1.0e-7),
        )
        if not all(bool(item) for item in relations):
            raise ValueError("atom feature equations do not replay")
        if not bool(((fv[:, 0] >= -1.0) & (fv[:, 0] <= 1.0) & (fv[:, 1] >= -2.0) & (fv[:, 1] <= 2.0)).all()):
            raise ValueError("ColNomic cosine feature domain drift")
    _validate_canonical_centres(bank.query_centres_xy_axis, qshape, bank.query_valid_axis, "query")
    _validate_canonical_centres(bank.reference_centres_xy_axis, rshape, bank.reference_valid_axis, "reference")
    qrc = torch.as_tensor(bank.query_rc, dtype=torch.long)
    rrc = torch.as_tensor(bank.reference_rc, dtype=torch.long)
    if bool(valid.any()):
        for prefix, query_rows, reference_rows in (
            ("active", qrc[valid], rrc[valid]),
            ("source", torch.as_tensor(bank.source_query_rc, dtype=torch.long)[valid],
             torch.as_tensor(bank.source_reference_rc, dtype=torch.long)[valid]),
        ):
            if not bool(((query_rows[:, 0] >= 0) & (query_rows[:, 0] < qshape[0]) &
                         (query_rows[:, 1] >= 0) & (query_rows[:, 1] < qshape[1])).all()):
                raise ValueError(f"valid {prefix} query coordinate outside grid")
            if not bool(((reference_rows[:, 0] >= 0) & (reference_rows[:, 0] < rshape[0]) &
                         (reference_rows[:, 1] >= 0) & (reference_rows[:, 1] < rshape[1])).all()):
                raise ValueError(f"valid {prefix} reference coordinate outside grid")
    for name in (bank.query_resource_key, bank.candidate_resource_key, bank.reference_resource_key,
                 bank.source_assignment_sha256, bank.assignment_sha256,
                 bank.coordinate_binding_sha256, bank.source_binding_sha256, bank.control_namespace):
        if not isinstance(name, str) or not name:
            raise ValueError("atom bank binding is empty")
    if recompute_source_assignment_sha256(bank) != bank.source_assignment_sha256:
        raise ValueError("source assignment hash does not replay")
    if recompute_assignment_sha256(bank) != bank.assignment_sha256:
        raise ValueError("active assignment hash does not replay")
    if _coordinate_binding_hash(bank.query_indices, bank.reference_indices, bank.query_rc, bank.reference_rc) != bank.coordinate_binding_sha256:
        raise ValueError("coordinate binding hash does not replay")
    return bank


def _sample_bhwc(field: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(field)
    if value.ndim < 4 or value.shape[0] != 1:
        raise ValueError("dense field must have batch dimension one")
    channels = int(torch.tensor(value.shape[3:]).prod())
    flat = value.reshape(1, value.shape[1], value.shape[2], channels).permute(0, 3, 1, 2)
    grid = torch.as_tensor(xy, dtype=value.dtype, device=value.device).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled[0, :, :, 0].T.reshape(xy.shape[0], *value.shape[3:])


def _coordinate_binding_hash(query_indices: torch.Tensor, reference_indices: torch.Tensor,
                             query_rc: torch.Tensor, reference_rc: torch.Tensor) -> str:
    return logical_sha256(
        {
            "query_indices": tensor_sha256(query_indices),
            "reference_indices": tensor_sha256(reference_indices),
            "query_rc": tensor_sha256(query_rc),
            "reference_rc": tensor_sha256(reference_rc),
        }
    )


def recompute_source_assignment_sha256(bank: ProposalAtomBank) -> str:
    return logical_sha256(
        {
            "query_resource_key": bank.query_resource_key,
            "candidate_resource_key": bank.candidate_resource_key,
            "reference_resource_key": bank.reference_resource_key,
            "source_binding_sha256": bank.source_binding_sha256,
            "query_grid_shape": list(bank.query_grid_shape),
            "reference_grid_shape": list(bank.reference_grid_shape),
            "features": tensor_sha256(bank.features),
            "signed_evidence": tensor_sha256(bank.signed_evidence),
            "valid_mask": tensor_sha256(bank.valid_mask),
            "query_valid_axis": tensor_sha256(bank.query_valid_axis),
            "reference_valid_axis": tensor_sha256(bank.reference_valid_axis),
            "source_query_indices": tensor_sha256(bank.source_query_indices),
            "source_reference_indices": tensor_sha256(bank.source_reference_indices),
            "query_coordinate_axis": tensor_sha256(bank.query_coordinate_axis),
            "reference_coordinate_axis": tensor_sha256(bank.reference_coordinate_axis),
            "query_centres_xy_axis": tensor_sha256(bank.query_centres_xy_axis),
            "reference_centres_xy_axis": tensor_sha256(bank.reference_centres_xy_axis),
            "source_query_rc": tensor_sha256(bank.source_query_rc),
            "source_reference_rc": tensor_sha256(bank.source_reference_rc),
            "source_forward_reference_xy": tensor_sha256(bank.source_forward_reference_xy),
            "source_reverse_query_xy": tensor_sha256(bank.source_reverse_query_xy),
            "overlap_pair": tensor_sha256(bank.overlap_pair),
            "precision_quality": tensor_sha256(bank.precision_quality),
            "cycle_error": tensor_sha256(bank.cycle_error),
        }
    )


def recompute_assignment_sha256(bank: ProposalAtomBank) -> str:
    qcount = bank.query_grid_shape[0] * bank.query_grid_shape[1]
    rcount = bank.reference_grid_shape[0] * bank.reference_grid_shape[1]
    identity_q = torch.arange(qcount, dtype=torch.long)
    identity_r = torch.arange(rcount, dtype=torch.long)
    if bank.control_namespace == "REAL":
        if not torch.equal(bank.query_coordinate_permutation.cpu(), identity_q) or not torch.equal(
            bank.reference_coordinate_permutation.cpu(), identity_r
        ):
            raise ValueError("REAL coordinate permutations must be identity")
        if not (
            torch.equal(bank.query_indices, bank.source_query_indices)
            and torch.equal(bank.reference_indices, bank.source_reference_indices)
            and torch.equal(bank.query_rc, bank.source_query_rc)
            and torch.equal(bank.reference_rc, bank.source_reference_rc)
            and torch.equal(bank.forward_reference_xy, bank.source_forward_reference_xy)
            and torch.equal(bank.reverse_query_xy, bank.source_reverse_query_xy)
        ):
            raise ValueError("REAL active/source provenance drift")
        return bank.source_assignment_sha256
    return logical_sha256(
        {
            "control_namespace": bank.control_namespace,
            "source_assignment_sha256": bank.source_assignment_sha256,
            "query_coordinate_permutation": tensor_sha256(bank.query_coordinate_permutation),
            "reference_coordinate_permutation": tensor_sha256(bank.reference_coordinate_permutation),
            "query_indices": tensor_sha256(bank.query_indices),
            "reference_indices": tensor_sha256(bank.reference_indices),
            "query_rc": tensor_sha256(bank.query_rc),
            "reference_rc": tensor_sha256(bank.reference_rc),
            "forward_reference_xy": tensor_sha256(bank.forward_reference_xy),
            "reverse_query_xy": tensor_sha256(bank.reverse_query_xy),
        }
    )


def seal_atom_bank(bank: ProposalAtomBank) -> ProposalAtomBank:
    """Populate all tensor-derived hashes, then run the strict schema replay."""
    source_hash = recompute_source_assignment_sha256(bank)
    staged = replace(bank, source_assignment_sha256=source_hash)
    assignment_hash = recompute_assignment_sha256(staged)
    coordinate_hash = _coordinate_binding_hash(
        staged.query_indices, staged.reference_indices, staged.query_rc, staged.reference_rc
    )
    return validate_atom_bank(
        replace(
            staged,
            assignment_sha256=assignment_hash,
            coordinate_binding_sha256=coordinate_hash,
        )
    )


def _validate_canonical_centres(centres: torch.Tensor, shape: tuple[int, int],
                                valid_mask: torch.Tensor, name: str) -> None:
    value = torch.as_tensor(centres, dtype=torch.float64).reshape(shape[0], shape[1], 2)
    valid = torch.as_tensor(valid_mask, dtype=torch.bool).reshape(shape)
    if not bool(torch.isfinite(value).all()) or (bool(valid.any()) and not bool(value[valid].abs().lt(1.0).all())):
        raise ValueError(f"{name} centres must be finite and strictly inside normalized image")
    adjacent_valid = valid[:, 1:] & valid[:, :-1]
    if bool(adjacent_valid.any()) and not bool((value[:, 1:, 0][adjacent_valid] > value[:, :-1, 0][adjacent_valid]).all()):
        raise ValueError(f"{name} centres are not canonical row-major x order")
    if shape[0] > 1:
        present = valid.any(dim=1)
        row_y = torch.stack([value[row, valid[row], 1].mean() for row in range(shape[0]) if bool(present[row])])
        if row_y.numel() > 1 and not bool((row_y[1:] > row_y[:-1]).all()):
            raise ValueError(f"{name} centres are not canonical row-major y order")
    tolerance = 1.0e-7
    for row in range(shape[0]):
        if int(valid[row].sum()) > 1 and float(value[row, valid[row], 1].max() - value[row, valid[row], 1].min()) > tolerance:
            raise ValueError(f"{name} centres do not share one y coordinate per grid row")
    for column in range(shape[1]):
        present = valid[:, column]
        if int(present.sum()) > 1 and float(value[present, column, 0].max() - value[present, column, 0].min()) > tolerance:
            raise ValueError(f"{name} centres do not share one x coordinate per grid column")


def build_atom_bank_from_dense_match(
    *,
    query_resource_key: str,
    candidate_resource_key: str,
    reference_resource_key: str,
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    query_centres_xy: torch.Tensor,
    reference_centres_xy: torch.Tensor,
    query_valid_mask: torch.Tensor,
    reference_valid_mask: torch.Tensor,
    warp_ab: torch.Tensor,
    overlap_ab: torch.Tensor,
    precision_ab: torch.Tensor,
    warp_ba: torch.Tensor,
    overlap_ba: torch.Tensor,
    precision_ba: torch.Tensor,
    source_binding_sha256: str,
) -> ProposalAtomBank:
    """Compress one dense RoMa match into one atom per query patch."""
    qshape = _positive_shape(query_grid_shape, "query grid")
    rshape = _positive_shape(reference_grid_shape, "reference grid")
    qtokens = torch.as_tensor(query_tokens)
    rtokens = torch.as_tensor(reference_tokens)
    qxy = torch.as_tensor(query_centres_xy, dtype=warp_ab.dtype, device=warp_ab.device)
    rxy = torch.as_tensor(reference_centres_xy, dtype=warp_ab.dtype, device=warp_ab.device)
    qvalid = torch.as_tensor(query_valid_mask, dtype=torch.bool, device=warp_ab.device)
    rvalid = torch.as_tensor(reference_valid_mask, dtype=torch.bool, device=warp_ab.device)
    if qtokens.ndim != 2 or rtokens.ndim != 2 or qtokens.shape[1] != rtokens.shape[1]:
        raise ValueError("query/reference token dimension drift")
    if qtokens.shape[0] != qshape[0] * qshape[1] or rtokens.shape[0] != rshape[0] * rshape[1]:
        raise ValueError("token count does not close to the complete canonical grid")
    if qxy.shape != (qtokens.shape[0], 2) or rxy.shape != (rtokens.shape[0], 2):
        raise ValueError("canonical centre/token axis drift")
    if qvalid.shape != (qtokens.shape[0],) or rvalid.shape != (rtokens.shape[0],):
        raise ValueError("canonical valid-mask axis drift")
    if not bool(rvalid.any()):
        raise ValueError("reference has no valid canonical token")
    _validate_canonical_centres(qxy, qshape, qvalid, "query")
    _validate_canonical_centres(rxy, rshape, rvalid, "reference")

    mapped = _sample_bhwc(warp_ab, qxy).reshape(-1, 2)
    oq = _sample_bhwc(overlap_ab, qxy).reshape(-1).clamp(0.0, 1.0)
    reverse = _sample_bhwc(warp_ba, mapped).reshape(-1, 2)
    orev = _sample_bhwc(overlap_ba, mapped).reshape(-1).clamp(0.0, 1.0)
    pab = _sample_bhwc(precision_ab, qxy).reshape(-1, 2, 2)
    pba = _sample_bhwc(precision_ba, mapped).reshape(-1, 2, 2)

    distances = torch.cdist(mapped.to(torch.float32), rxy.to(torch.float32))
    distances[:, ~rvalid] = torch.inf
    ridx = distances.argmin(dim=1)
    in_bounds = mapped.abs().lt(1.0).all(dim=1)
    valid = qvalid & in_bounds & rvalid[ridx]

    qn = F.normalize(qtokens.to(device=warp_ab.device, dtype=torch.float32), dim=1)
    rn = F.normalize(rtokens.to(device=warp_ab.device, dtype=torch.float32), dim=1)
    assigned = (qn * rn[ridx]).sum(dim=1).clamp(-1.0, 1.0)
    mean_reference = rn[rvalid].mean(dim=0)
    baseline = (qn * mean_reference).sum(dim=1).clamp(-1.0, 1.0)
    centred = assigned - baseline
    cycle = torch.linalg.vector_norm(reverse - qxy, dim=1)
    cycle_quality = torch.exp(-4.0 * cycle)
    reciprocal = torch.sqrt((oq * orev).clamp_min(0.0))

    symmetric_ab = (pab + pab.transpose(1, 2)) * 0.5
    symmetric_ba = (pba + pba.transpose(1, 2)) * 0.5
    minimum_eigenvalue = torch.minimum(
        torch.linalg.eigvalsh(symmetric_ab).amin(dim=1),
        torch.linalg.eigvalsh(symmetric_ba).amin(dim=1),
    ).clamp_min(0.0)
    precision_quality = torch.tanh(torch.log1p(minimum_eigenvalue))
    evidence = centred * reciprocal * cycle_quality
    features = torch.stack(
        (assigned, centred, oq, orev, reciprocal, cycle_quality, precision_quality,
         reciprocal * centred, evidence),
        dim=1,
    )
    features = torch.where(valid[:, None], features, torch.zeros_like(features))
    evidence = torch.where(valid, evidence, torch.zeros_like(evidence))

    qindex = torch.arange(qtokens.shape[0], dtype=torch.long, device=warp_ab.device)
    raxis_index = torch.arange(rtokens.shape[0], dtype=torch.long, device=warp_ab.device)
    qaxis = torch.stack((qindex // qshape[1], qindex % qshape[1]), dim=1)
    raxis = torch.stack((raxis_index // rshape[1], raxis_index % rshape[1]), dim=1)
    qrc = qaxis[qindex]
    rrc = raxis[ridx]
    overlap_pair = torch.stack((oq, orev), dim=1)
    identity_q = torch.arange(qtokens.shape[0], dtype=torch.long)
    identity_r = torch.arange(rtokens.shape[0], dtype=torch.long)
    bank = ProposalAtomBank(
        query_resource_key=query_resource_key,
        candidate_resource_key=candidate_resource_key,
        reference_resource_key=reference_resource_key,
        query_grid_shape=qshape,
        reference_grid_shape=rshape,
        features=features.detach().cpu().contiguous(),
        signed_evidence=evidence.detach().cpu().contiguous(),
        valid_mask=valid.detach().cpu().contiguous(),
        query_valid_axis=qvalid.detach().cpu().contiguous(),
        reference_valid_axis=rvalid.detach().cpu().contiguous(),
        source_query_indices=qindex.detach().cpu().contiguous(),
        source_reference_indices=ridx.detach().cpu().contiguous(),
        query_indices=qindex.detach().cpu().contiguous(),
        reference_indices=ridx.detach().cpu().contiguous(),
        query_coordinate_axis=qaxis.detach().cpu().contiguous(),
        reference_coordinate_axis=raxis.detach().cpu().contiguous(),
        query_centres_xy_axis=qxy.detach().cpu().contiguous(),
        reference_centres_xy_axis=rxy.detach().cpu().contiguous(),
        source_query_rc=qrc.detach().cpu().contiguous(),
        source_reference_rc=rrc.detach().cpu().contiguous(),
        query_rc=qrc.detach().cpu().contiguous(),
        reference_rc=rrc.detach().cpu().contiguous(),
        source_forward_reference_xy=mapped.detach().cpu().contiguous(),
        source_reverse_query_xy=reverse.detach().cpu().contiguous(),
        forward_reference_xy=mapped.detach().cpu().contiguous(),
        reverse_query_xy=reverse.detach().cpu().contiguous(),
        overlap_pair=overlap_pair.detach().cpu().contiguous(),
        precision_quality=precision_quality.detach().cpu().contiguous(),
        cycle_error=cycle.detach().cpu().contiguous(),
        source_assignment_sha256="PENDING",
        assignment_sha256="PENDING",
        coordinate_binding_sha256=_coordinate_binding_hash(qindex, ridx, qrc, rrc),
        source_binding_sha256=source_binding_sha256,
        control_namespace="REAL",
        query_coordinate_permutation=identity_q,
        reference_coordinate_permutation=identity_r,
    )
    return seal_atom_bank(bank)


def _is_affine_permutation(coordinates: torch.Tensor, permutation: torch.Tensor) -> bool:
    xy = torch.as_tensor(coordinates, dtype=torch.float64)
    perm = torch.as_tensor(permutation, dtype=torch.long)
    if xy.shape[0] < 4 or xy.unique(dim=0).shape[0] < 4:
        raise ValueError("at least four unique coordinate rows are required to certify non-affinity")
    design = torch.cat((xy, torch.ones((xy.shape[0], 1), dtype=torch.float64)), dim=1)
    solution = torch.linalg.lstsq(design, xy[perm]).solution
    residual = torch.linalg.vector_norm(design @ solution - xy[perm], dim=1).amax()
    return bool(residual <= 1.0e-10)


def deterministic_derangement(count: int, namespace: str, resource_key: str,
                              coordinates: torch.Tensor | None = None) -> torch.Tensor:
    if count < 2 or not namespace or not resource_key:
        raise ValueError("derangement requires count>=2 and nonempty bindings")
    for attempt in range(128):
        ordered = sorted(
            range(count),
            key=lambda index: (
                hashlib.sha256(f"{namespace}|{resource_key}|{attempt}|{index}".encode()).digest(),
                index,
            ),
        )
        permutation = torch.empty(count, dtype=torch.long)
        for offset, source in enumerate(ordered):
            permutation[source] = ordered[(offset + 1) % count]
        if bool((permutation == torch.arange(count)).any()):
            continue
        if coordinates is not None and (
            torch.as_tensor(coordinates).shape != (count, 2)
            or _is_affine_permutation(coordinates, permutation)
        ):
            continue
        return permutation
    raise RuntimeError("could not construct a non-affine fixed-point-free permutation")


@dataclass(frozen=True)
class CoordinateControlReceipt:
    namespace: str
    query_permutation_sha256: str
    reference_permutation_sha256: str
    feature_sha256: str
    evidence_sha256: str
    query_coordinate_multiset_sha256: str
    reference_coordinate_multiset_sha256: str
    source_assignment_sha256: str
    controlled_assignment_sha256: str
    query_permutation: torch.Tensor
    reference_permutation: torch.Tensor
    query_non_affine: bool
    reference_non_affine: bool


def coordinate_destroy_atom_bank(bank: ProposalAtomBank, namespace: str) -> tuple[ProposalAtomBank, CoordinateControlReceipt]:
    validate_atom_bank(bank)
    qcount = bank.query_coordinate_axis.shape[0]
    rcount = bank.reference_coordinate_axis.shape[0]
    qperm = deterministic_derangement(
        qcount, f"{namespace}:QUERY", bank.query_resource_key, bank.query_coordinate_axis
    )
    rperm = deterministic_derangement(
        rcount, f"{namespace}:REFERENCE", bank.reference_resource_key, bank.reference_coordinate_axis
    )
    query_indices = qperm[bank.source_query_indices]
    reference_indices = rperm[bank.source_reference_indices]
    query_rc = bank.query_coordinate_axis[query_indices]
    reference_rc = bank.reference_coordinate_axis[reference_indices]
    reverse_query_xy = bank.query_centres_xy_axis[query_indices].to(bank.reverse_query_xy.dtype)
    forward_reference_xy = bank.reference_centres_xy_axis[reference_indices].to(bank.forward_reference_xy.dtype)
    controlled = seal_atom_bank(replace(
        bank,
        query_indices=query_indices,
        reference_indices=reference_indices,
        query_rc=query_rc,
        reference_rc=reference_rc,
        forward_reference_xy=forward_reference_xy,
        reverse_query_xy=reverse_query_xy,
        assignment_sha256="PENDING",
        coordinate_binding_sha256="PENDING",
        control_namespace=namespace,
        query_coordinate_permutation=qperm,
        reference_coordinate_permutation=rperm,
    ))
    receipt = CoordinateControlReceipt(
        namespace=namespace,
        query_permutation_sha256=tensor_sha256(qperm),
        reference_permutation_sha256=tensor_sha256(rperm),
        feature_sha256=tensor_sha256(bank.features),
        evidence_sha256=tensor_sha256(bank.signed_evidence),
        query_coordinate_multiset_sha256=coordinate_multiset_sha256(bank.query_coordinate_axis),
        reference_coordinate_multiset_sha256=coordinate_multiset_sha256(bank.reference_coordinate_axis),
        source_assignment_sha256=bank.source_assignment_sha256,
        controlled_assignment_sha256=controlled.assignment_sha256,
        query_permutation=qperm,
        reference_permutation=rperm,
        query_non_affine=not _is_affine_permutation(bank.query_coordinate_axis, qperm),
        reference_non_affine=not _is_affine_permutation(bank.reference_coordinate_axis, rperm),
    )
    return controlled, receipt


def _component_indices(selected: torch.Tensor, query_rc: torch.Tensor,
                       reference_rc: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    chosen = [int(index) for index in torch.nonzero(selected, as_tuple=False).flatten().tolist()]
    by_query = {tuple(int(x) for x in query_rc[index].tolist()): index for index in chosen}
    adjacency: dict[int, list[int]] = {index: [] for index in chosen}
    for index in chosen:
        row, column = (int(x) for x in query_rc[index].tolist())
        for neighbour_rc in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            neighbour = by_query.get(neighbour_rc)
            if neighbour is None:
                continue
            distance = (reference_rc[index] - reference_rc[neighbour]).abs().amax()
            if int(distance) <= REFERENCE_CHEBYSHEV:
                adjacency[index].append(neighbour)
    components: list[tuple[int, ...]] = []
    unseen = set(chosen)
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _query_four_connected(query_rc: torch.Tensor) -> bool:
    coordinates = {tuple(int(x) for x in row.tolist()) for row in query_rc}
    if not coordinates:
        return False
    unseen = set(coordinates)
    stack = [min(unseen)]
    unseen.remove(stack[0])
    while stack:
        row, column = stack.pop()
        for item in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            if item in unseen:
                unseen.remove(item)
                stack.append(item)
    return not unseen


@dataclass(frozen=True)
class ConnectedHypothesis:
    slot: int
    atom_indices: torch.Tensor
    source_query_indices: torch.Tensor
    source_reference_indices: torch.Tensor
    query_indices: torch.Tensor
    reference_indices: torch.Tensor
    query_rc: torch.Tensor
    reference_rc: torch.Tensor
    source_query_rc: torch.Tensor
    source_reference_rc: torch.Tensor
    atom_weights: torch.Tensor
    atom_evidence: torch.Tensor
    forward_reference_xy: torch.Tensor
    reverse_query_xy: torch.Tensor
    source_forward_reference_xy: torch.Tensor
    source_reverse_query_xy: torch.Tensor
    overlap_pair: torch.Tensor
    precision_quality: torch.Tensor
    cycle_error: torch.Tensor
    score: torch.Tensor
    logical_sha256: str


@dataclass(frozen=True)
class CandidateProposal:
    query_resource_key: str
    candidate_resource_key: str
    reference_resource_key: str
    h0_score: torch.Tensor
    h0_weight: torch.Tensor
    hypotheses: tuple[ConnectedHypothesis, ...]
    slot_states: tuple[str, ...]
    slot_scores: torch.Tensor
    candidate_score: torch.Tensor
    h1_decision: bool
    selected_atom_count: int
    valid_atom_count: int
    component_graph_sha256: str
    region_score_atom_indices: torch.Tensor
    region_score_sha256: str
    region_only_atom_read_count: int
    direct_all_atom_bypass_count: int
    source_assignment_sha256: str
    assignment_sha256: str
    coordinate_binding_sha256: str
    logical_sha256: str


def _hypothesis(bank: ProposalAtomBank, indices: tuple[int, ...], weights: torch.Tensor,
                score: torch.Tensor, slot: int) -> ConnectedHypothesis:
    rows = torch.tensor(indices, dtype=torch.long)
    detached_score = float(score.detach().cpu())
    signature = {
        "slot": slot,
        "query_resource_key": bank.query_resource_key,
        "candidate_resource_key": bank.candidate_resource_key,
        "reference_resource_key": bank.reference_resource_key,
        "source_binding_sha256": bank.source_binding_sha256,
        "source_assignment_sha256": bank.source_assignment_sha256,
        "assignment_sha256": bank.assignment_sha256,
        "coordinate_binding_sha256": bank.coordinate_binding_sha256,
        "atom_indices": list(indices),
        "source_query_indices": tensor_sha256(bank.source_query_indices[rows]),
        "source_reference_indices": tensor_sha256(bank.source_reference_indices[rows]),
        "query_indices": tensor_sha256(bank.query_indices[rows]),
        "reference_indices": tensor_sha256(bank.reference_indices[rows]),
        "query_rc": tensor_sha256(bank.query_rc[rows]),
        "reference_rc": tensor_sha256(bank.reference_rc[rows]),
        "source_query_rc": tensor_sha256(bank.source_query_rc[rows]),
        "source_reference_rc": tensor_sha256(bank.source_reference_rc[rows]),
        "atom_weights": tensor_sha256(weights.detach().cpu()),
        "atom_features": tensor_sha256(bank.features[rows]),
        "atom_evidence": tensor_sha256(bank.signed_evidence[rows]),
        "forward_reference_xy": tensor_sha256(bank.forward_reference_xy[rows]),
        "reverse_query_xy": tensor_sha256(bank.reverse_query_xy[rows]),
        "source_forward_reference_xy": tensor_sha256(bank.source_forward_reference_xy[rows]),
        "source_reverse_query_xy": tensor_sha256(bank.source_reverse_query_xy[rows]),
        "overlap_pair": tensor_sha256(bank.overlap_pair[rows]),
        "precision_quality": tensor_sha256(bank.precision_quality[rows]),
        "cycle_error": tensor_sha256(bank.cycle_error[rows]),
        "score": detached_score,
    }
    return ConnectedHypothesis(
        slot=slot,
        atom_indices=rows,
        source_query_indices=bank.source_query_indices[rows],
        source_reference_indices=bank.source_reference_indices[rows],
        query_indices=bank.query_indices[rows],
        reference_indices=bank.reference_indices[rows],
        query_rc=bank.query_rc[rows],
        reference_rc=bank.reference_rc[rows],
        source_query_rc=bank.source_query_rc[rows],
        source_reference_rc=bank.source_reference_rc[rows],
        atom_weights=weights,
        atom_evidence=bank.signed_evidence[rows],
        forward_reference_xy=bank.forward_reference_xy[rows],
        reverse_query_xy=bank.reverse_query_xy[rows],
        source_forward_reference_xy=bank.source_forward_reference_xy[rows],
        source_reverse_query_xy=bank.source_reverse_query_xy[rows],
        overlap_pair=bank.overlap_pair[rows],
        precision_quality=bank.precision_quality[rows],
        cycle_error=bank.cycle_error[rows],
        score=score,
        logical_sha256=logical_sha256(signature),
    )


class ConnectedProposalSelector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(ATOM_FEATURE_DIM, 1, bias=True, dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
        if sum(parameter.numel() for parameter in self.parameters()) != SELECTOR_PARAMETER_COUNT:
            raise RuntimeError("selector parameter-count drift")

    def forward(self, bank: ProposalAtomBank) -> CandidateProposal:
        validate_atom_bank(bank)
        device = self.head.weight.device
        features = bank.features.to(device=device, dtype=torch.float64)
        evidence = bank.signed_evidence.to(device=device, dtype=torch.float64)
        valid = bank.valid_mask.to(device=device, dtype=torch.bool)
        zero = self.head.weight.abs().sum() * 0.0 + self.head.bias.abs().sum() * 0.0
        valid_rows = torch.nonzero(valid, as_tuple=False).flatten()
        full_weights = torch.zeros_like(evidence)
        if valid_rows.numel() == 0:
            h0_weight = zero + 1.0
        else:
            logits = self.head(features[valid_rows]).squeeze(1)
            mixture = sparsemax(torch.cat((zero.reshape(1), logits)))
            h0_weight = mixture[0]
            full_weights = full_weights.scatter(0, valid_rows, mixture[1:])
        selected = full_weights.detach().gt(0.0).cpu()
        raw_components = _component_indices(selected, bank.query_rc, bank.reference_rc)
        component_graph_hash = logical_sha256(
            {
                "assignment_sha256": bank.assignment_sha256,
                "selected_indices": tensor_sha256(torch.nonzero(selected, as_tuple=False).flatten()),
                "components": [list(item) for item in raw_components],
            }
        )
        scored: list[tuple[torch.Tensor, tuple[int, ...], torch.Tensor]] = []
        for indices in raw_components:
            rows = torch.tensor(indices, dtype=torch.long, device=device)
            qrows = bank.query_rc[torch.tensor(indices, dtype=torch.long)]
            rrows = bank.reference_rc[torch.tensor(indices, dtype=torch.long)]
            if len(indices) < MIN_QUERY_PATCHES or torch.unique(rrows, dim=0).shape[0] < MIN_REFERENCE_ATOMS:
                continue
            if not _query_four_connected(qrows):
                continue
            local_weights = full_weights[rows]
            component_score = (local_weights * evidence[rows]).sum() / local_weights.sum().clamp_min(EPS)
            scored.append((component_score, indices, local_weights))
        scored.sort(key=lambda item: (-float(item[0].detach().cpu()), item[1]))
        scored = scored[:H1_SLOT_COUNT]
        hypotheses = tuple(
            _hypothesis(bank, indices, local_weights, score, slot)
            for slot, (score, indices, local_weights) in enumerate(scored, start=1)
        )
        slot_scores = torch.stack(
            [item.score for item in hypotheses] + [zero] * (H1_SLOT_COUNT - len(hypotheses))
        )
        if not hypotheses:
            candidate_score = zero
        else:
            candidate_score = torch.logsumexp(torch.cat((zero.reshape(1), slot_scores)), dim=0) - math.log(
                H1_SLOT_COUNT + 1
            )
        strongest = max((float(item.score.detach().cpu()) for item in hypotheses), default=0.0)
        slot_states = tuple(["H1"] * len(hypotheses) + ["H0_PADDING"] * (H1_SLOT_COUNT - len(hypotheses)))
        read_indices = torch.tensor(
            sorted(index for item in hypotheses for index in item.atom_indices.tolist()), dtype=torch.long
        )
        region_score_hash = logical_sha256(
            {
                "read_indices": tensor_sha256(read_indices),
                "read_weights": tensor_sha256(full_weights[read_indices.to(device)].detach().cpu()),
                "read_evidence": tensor_sha256(bank.signed_evidence[read_indices]),
                "slot_scores": tensor_sha256(slot_scores.detach().cpu()),
                "candidate_score": float(candidate_score.detach().cpu()),
            }
        )
        proposal_signature = {
            "query_resource_key": bank.query_resource_key,
            "candidate_resource_key": bank.candidate_resource_key,
            "reference_resource_key": bank.reference_resource_key,
            "source_assignment_sha256": bank.source_assignment_sha256,
            "assignment_sha256": bank.assignment_sha256,
            "coordinate_binding_sha256": bank.coordinate_binding_sha256,
            "hypothesis_sha256": [item.logical_sha256 for item in hypotheses],
            "slot_states": list(slot_states),
            "slot_scores": [float(item) for item in slot_scores.detach().cpu()],
            "candidate_score": float(candidate_score.detach().cpu()),
            "h1_decision": strongest > 0.0,
            "h0_weight": float(h0_weight.detach().cpu()),
            "selected_atom_count": int(selected.sum()),
            "valid_atom_count": int(valid.sum().detach().cpu()),
            "component_graph_sha256": component_graph_hash,
            "region_score_atom_indices": tensor_sha256(read_indices),
            "region_score_sha256": region_score_hash,
        }
        return CandidateProposal(
            query_resource_key=bank.query_resource_key,
            candidate_resource_key=bank.candidate_resource_key,
            reference_resource_key=bank.reference_resource_key,
            h0_score=zero,
            h0_weight=h0_weight,
            hypotheses=hypotheses,
            slot_states=slot_states,
            slot_scores=slot_scores,
            candidate_score=candidate_score,
            h1_decision=strongest > 0.0,
            selected_atom_count=int(selected.sum()),
            valid_atom_count=int(valid.sum().detach().cpu()),
            component_graph_sha256=component_graph_hash,
            region_score_atom_indices=read_indices,
            region_score_sha256=region_score_hash,
            region_only_atom_read_count=int(read_indices.numel()),
            direct_all_atom_bypass_count=0,
            assignment_sha256=bank.assignment_sha256,
            source_assignment_sha256=bank.source_assignment_sha256,
            coordinate_binding_sha256=bank.coordinate_binding_sha256,
            logical_sha256=logical_sha256(proposal_signature),
        )


class AllPatchNoHypComparator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(ATOM_FEATURE_DIM, 1, bias=True, dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
        if sum(parameter.numel() for parameter in self.parameters()) != SELECTOR_PARAMETER_COUNT:
            raise RuntimeError("all-patch capacity drift")

    def forward(self, bank: ProposalAtomBank) -> torch.Tensor:
        validate_atom_bank(bank)
        device = self.head.weight.device
        valid = torch.nonzero(bank.valid_mask, as_tuple=False).flatten().to(device)
        evidence = bank.signed_evidence.to(device=device, dtype=torch.float64)
        if valid.numel() == 0:
            return self.head.weight.abs().sum() * 0.0 + self.head.bias.abs().sum() * 0.0
        logits = self.head(bank.features.to(device=device, dtype=torch.float64)[valid]).squeeze(1)
        return (torch.softmax(logits, dim=0) * evidence[valid]).sum()


def query_only_features(tokens: torch.Tensor, grid_shape: tuple[int, int],
                        valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = _positive_shape(grid_shape, "query grid")
    value = torch.as_tensor(tokens, dtype=torch.float64)
    valid = torch.as_tensor(valid_mask, dtype=torch.bool)
    count = value.shape[0]
    if value.ndim != 2 or count != shape[0] * shape[1] or valid.shape != (count,):
        raise ValueError("query-only token/grid axis drift")
    normalized = F.normalize(value, dim=1)
    norms = torch.linalg.vector_norm(value, dim=1).clamp_min(EPS)
    median = norms[valid].median() if bool(valid.any()) else norms.median()
    norm_feature = torch.tanh(torch.log(norms / median.clamp_min(EPS)))
    contrast = torch.zeros(count, dtype=torch.float64)
    indices = torch.arange(count)
    rc = torch.stack((indices // shape[1], indices % shape[1]), dim=1)
    for index in range(count):
        row, column = (int(x) for x in rc[index].tolist())
        neighbours = []
        for rr, cc in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                other = rr * shape[1] + cc
                if bool(valid[other]):
                    neighbours.append(other)
        if neighbours:
            contrast[index] = 1.0 - (normalized[index] * normalized[neighbours]).sum(dim=1).mean()
    y = (rc[:, 0].to(torch.float64) + 0.5) * (2.0 / shape[0]) - 1.0
    x = (rc[:, 1].to(torch.float64) + 0.5) * (2.0 / shape[1]) - 1.0
    radius = torch.sqrt(x.square() + y.square()) / math.sqrt(2.0)
    features = torch.stack((norm_feature, contrast, x, y, x.square(), y.square(), x * y, radius, 1.0 - radius), dim=1)
    return features, rc


@dataclass(frozen=True)
class QueryOnlyRegionSelection:
    groups: tuple[tuple[int, ...], ...]
    weights: torch.Tensor
    h0_weight: torch.Tensor
    query_grid_shape: tuple[int, int]
    query_coordinate_axis_sha256: str
    query_resource_key: str
    query_feature_sha256: str
    query_valid_sha256: str


class QueryOnlyRegionComparator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(ATOM_FEATURE_DIM, 1, bias=True, dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
        if sum(parameter.numel() for parameter in self.parameters()) != SELECTOR_PARAMETER_COUNT:
            raise RuntimeError("query-only capacity drift")

    def select(self, query_resource_key: str, features: torch.Tensor, query_rc: torch.Tensor,
               valid_mask: torch.Tensor) -> QueryOnlyRegionSelection:
        device = self.head.weight.device
        feat = torch.as_tensor(features, dtype=torch.float64, device=device)
        valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
        if not query_resource_key:
            raise ValueError("query-only selection requires an opaque query resource key")
        zero = self.head.weight.abs().sum() * 0.0 + self.head.bias.abs().sum() * 0.0
        rows = torch.nonzero(valid, as_tuple=False).flatten()
        full = zero + torch.zeros(feat.shape[0], dtype=torch.float64, device=device)
        if rows.numel() == 0:
            return QueryOnlyRegionSelection(
                (), full, zero + 1.0,
                (int(query_rc[:, 0].max()) + 1, int(query_rc[:, 1].max()) + 1),
                tensor_sha256(query_rc),
                query_resource_key,
                tensor_sha256(torch.as_tensor(features)),
                tensor_sha256(torch.as_tensor(valid_mask)),
            )
        mixture = sparsemax(torch.cat((zero.reshape(1), self.head(feat[rows]).squeeze(1))))
        full = full.scatter(0, rows, mixture[1:])
        selected = full.detach().gt(0.0).cpu()
        # Reference coordinates equal query coordinates only to reuse the graph primitive.
        components = _component_indices(selected, query_rc, query_rc)
        legal = [item for item in components if len(item) >= MIN_QUERY_PATCHES and _query_four_connected(query_rc[list(item)])]
        legal.sort(
            key=lambda item: (
                -float(full[torch.tensor(item, device=device)].mean().detach().cpu()),
                item,
            )
        )
        shape = (int(query_rc[:, 0].max()) + 1, int(query_rc[:, 1].max()) + 1)
        expected = torch.stack(
            (torch.arange(query_rc.shape[0]) // shape[1], torch.arange(query_rc.shape[0]) % shape[1]), dim=1
        )
        if not torch.equal(torch.as_tensor(query_rc, dtype=torch.long).cpu(), expected):
            raise ValueError("query-only coordinate axis must be complete canonical row-major")
        return QueryOnlyRegionSelection(
            tuple(legal[:H1_SLOT_COUNT]), full, mixture[0], shape, tensor_sha256(query_rc),
            query_resource_key, tensor_sha256(torch.as_tensor(features)), tensor_sha256(torch.as_tensor(valid_mask))
        )

    def score(self, bank: ProposalAtomBank, selection: QueryOnlyRegionSelection) -> torch.Tensor:
        validate_atom_bank(bank)
        if (
            bank.query_grid_shape != selection.query_grid_shape
            or tensor_sha256(bank.query_coordinate_axis) != selection.query_coordinate_axis_sha256
            or bank.query_resource_key != selection.query_resource_key
            or tensor_sha256(bank.query_valid_axis) != selection.query_valid_sha256
        ):
            raise ValueError("query-only selection/bank coordinate binding drift")
        device = self.head.weight.device
        evidence = bank.signed_evidence.to(device=device, dtype=torch.float64)
        zero = selection.weights.abs().sum() * 0.0 + self.head.weight.abs().sum() * 0.0
        inverse = torch.empty(bank.query_indices.numel(), dtype=torch.long)
        inverse[bank.query_indices.to(torch.long)] = torch.arange(bank.query_indices.numel(), dtype=torch.long)
        scores = []
        for group in selection.groups:
            cells = torch.tensor(group, dtype=torch.long)
            content_rows = inverse[cells]
            content_rows = content_rows.to(device)
            cells = cells.to(device)
            weights = selection.weights[cells]
            scores.append((weights * evidence[content_rows]).sum() / weights.sum().clamp_min(EPS))
        if not scores:
            return zero
        slots = torch.stack(scores + [zero] * (H1_SLOT_COUNT - len(scores)))
        return torch.logsumexp(torch.cat((zero.reshape(1), slots)), dim=0) - math.log(H1_SLOT_COUNT + 1)


def whole_reference_derangement(keys: Sequence[str], namespace: str,
                                 query_resource_key: str) -> tuple[int, ...]:
    if len(keys) < 2 or len(set(keys)) != len(keys):
        raise ValueError("complete reference axis must contain distinct opaque keys")
    permutation = deterministic_derangement(len(keys), namespace, query_resource_key)
    return tuple(int(item) for item in permutation.tolist())


def score_reference_axis(selector: ConnectedProposalSelector,
                         banks: Sequence[ProposalAtomBank]) -> tuple[torch.Tensor, tuple[CandidateProposal, ...]]:
    if not banks:
        raise ValueError("empty reference axis")
    query_keys = {bank.query_resource_key for bank in banks}
    candidate_keys = [bank.candidate_resource_key for bank in banks]
    if len(query_keys) != 1 or len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("reference axis binding drift")
    proposals = tuple(selector(bank) for bank in banks)
    return torch.stack([item.candidate_score for item in proposals]), proposals


def proposal_replay_signature(proposal: CandidateProposal) -> dict[str, object]:
    return {
        "query_resource_key": proposal.query_resource_key,
        "candidate_resource_key": proposal.candidate_resource_key,
        "reference_resource_key": proposal.reference_resource_key,
        "h0_score": float(proposal.h0_score.detach().cpu()),
        "h0_weight": float(proposal.h0_weight.detach().cpu()),
        "hypothesis_sha256": [item.logical_sha256 for item in proposal.hypotheses],
        "slot_states": list(proposal.slot_states),
        "slot_scores": [float(item) for item in proposal.slot_scores.detach().cpu()],
        "candidate_score": float(proposal.candidate_score.detach().cpu()),
        "h1_decision": proposal.h1_decision,
        "selected_atom_count": proposal.selected_atom_count,
        "valid_atom_count": proposal.valid_atom_count,
        "component_graph_sha256": proposal.component_graph_sha256,
        "region_score_atom_indices": proposal.region_score_atom_indices.tolist(),
        "region_score_sha256": proposal.region_score_sha256,
        "region_only_atom_read_count": proposal.region_only_atom_read_count,
        "direct_all_atom_bypass_count": proposal.direct_all_atom_bypass_count,
        "assignment_sha256": proposal.assignment_sha256,
        "source_assignment_sha256": proposal.source_assignment_sha256,
        "coordinate_binding_sha256": proposal.coordinate_binding_sha256,
        "logical_sha256": proposal.logical_sha256,
    }


__all__ = [
    "ATOM_FEATURE_DIM",
    "SELECTOR_PARAMETER_COUNT",
    "H1_SLOT_COUNT",
    "MIN_QUERY_PATCHES",
    "MIN_REFERENCE_ATOMS",
    "REFERENCE_CHEBYSHEV",
    "ProposalAtomBank",
    "ConnectedHypothesis",
    "CandidateProposal",
    "ConnectedProposalSelector",
    "AllPatchNoHypComparator",
    "QueryOnlyRegionComparator",
    "QueryOnlyRegionSelection",
    "CoordinateControlReceipt",
    "build_atom_bank_from_dense_match",
    "coordinate_destroy_atom_bank",
    "coordinate_multiset_sha256",
    "deterministic_derangement",
    "logical_sha256",
    "proposal_replay_signature",
    "query_only_features",
    "score_reference_axis",
    "seal_atom_bank",
    "sparsemax",
    "tensor_sha256",
    "validate_atom_bank",
    "whole_reference_derangement",
]
