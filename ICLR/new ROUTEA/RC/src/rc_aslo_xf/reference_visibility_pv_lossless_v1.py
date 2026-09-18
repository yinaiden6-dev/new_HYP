"""Lossless interface split of the existing RAW/RoMa visibility score.

The proposal is a sealed full-grid visibility field, not an identity classifier
or a connected-region hypothesis. Verification preserves the original scalar
operation order, including the original whole-reference visibility mean.
This interface alone supplies no HYP or scientific GO qualification.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Mapping

import torch
from torch.nn import functional as F

SCHEMA = "REFERENCE_VISIBILITY_PV_LOSSLESS_V1"
FEATURE_DIMENSION = 128
REQUIRED_SOURCE_FIELDS = frozenset({"query_tokens_sha256", "reference_tokens_sha256", "source_maps_sha256"})


def _need(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def tensor_sha256(value: torch.Tensor) -> str:
    item = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(item.dtype).encode())
    digest.update(_canonical(list(item.shape)))
    digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and tensor_sha256(a) == tensor_sha256(b)


def _shape(value, name):
    _need(isinstance(value, (tuple, list)) and len(value) == 2
          and all(type(item) is int and item > 0 for item in value), name + "_INVALID")
    return tuple(value)


def _binding(value):
    _need(isinstance(value, Mapping) and REQUIRED_SOURCE_FIELDS.issubset(value), "SOURCE_BINDING_FIELDS_MISSING")
    _need(all(isinstance(key, str) and key.endswith("sha256") and isinstance(item, str)
              and len(item) == 64 and all(char in "0123456789abcdef" for char in item)
              for key, item in value.items()), "SOURCE_BINDING_NOT_SHA256_ONLY")
    return dict(sorted(value.items()))


def _visibility(value, count, name):
    _need(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.float64
          and value.shape == (count,) and value.is_contiguous() and bool(torch.isfinite(value).all())
          and bool(((value >= 0.0) & (value <= 1.0)).all()), name + "_FULL_FP64_GRID_INVALID")


@dataclass(frozen=True)
class VisibilityProposal:
    query_resource_key: str
    candidate_resource_key: str
    reference_resource_key: str
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    query_axis: torch.Tensor
    reference_axis: torch.Tensor
    query_visibility: torch.Tensor
    reference_visibility: torch.Tensor
    query_full_mean: torch.Tensor
    reference_full_mean: torch.Tensor
    visibility_mass: torch.Tensor
    source_binding: Mapping[str, str]
    logical_sha256: str


def _proposal_payload(value):
    return {"schema": SCHEMA, "query_resource_key": value.query_resource_key,
        "candidate_resource_key": value.candidate_resource_key, "reference_resource_key": value.reference_resource_key,
        "query_grid_shape": list(value.query_grid_shape), "reference_grid_shape": list(value.reference_grid_shape),
        "query_axis": value.query_axis.tolist(), "reference_axis": value.reference_axis.tolist(),
        "query_visibility_binary64": [float(x).hex() for x in value.query_visibility],
        "reference_visibility_binary64": [float(x).hex() for x in value.reference_visibility],
        "query_full_mean_binary64": float(value.query_full_mean).hex(),
        "reference_full_mean_binary64": float(value.reference_full_mean).hex(),
        "visibility_mass_binary64": float(value.visibility_mass).hex(),
        "source_binding": dict(value.source_binding)}


def _proposal_hash(value):
    return hashlib.sha256(_canonical(_proposal_payload(value))).hexdigest()


def validate_visibility_proposal(value: VisibilityProposal) -> VisibilityProposal:
    _need(isinstance(value, VisibilityProposal), "PROPOSAL_TYPE_INVALID")
    qshape = _shape(value.query_grid_shape, "QUERY_GRID")
    rshape = _shape(value.reference_grid_shape, "REFERENCE_GRID")
    _need(all(isinstance(key, str) and bool(key) for key in (value.query_resource_key,
              value.candidate_resource_key, value.reference_resource_key)), "RESOURCE_KEY_INVALID")
    _binding(value.source_binding)
    for axis, shape, name in ((value.query_axis, qshape, "QUERY"), (value.reference_axis, rshape, "REFERENCE")):
        _need(isinstance(axis, torch.Tensor) and axis.device.type == "cpu" and axis.dtype == torch.int64
              and axis.shape == (math.prod(shape),)
              and torch.equal(axis, torch.arange(math.prod(shape), dtype=torch.int64)), name + "_CANONICAL_FULL_AXIS_DRIFT")
    _visibility(value.query_visibility, math.prod(qshape), "QUERY_VISIBILITY")
    _visibility(value.reference_visibility, math.prod(rshape), "REFERENCE_VISIBILITY")
    for mean in (value.query_full_mean, value.reference_full_mean, value.visibility_mass):
        _need(isinstance(mean, torch.Tensor) and mean.dtype == torch.float64 and mean.device.type == "cpu"
              and mean.shape == () and bool(torch.isfinite(mean)), "FULL_MEAN_OR_MASS_SCALAR_INVALID")
    qmean = value.query_visibility.mean()
    rmean = value.reference_visibility.mean()
    mass = torch.sqrt(qmean * rmean)
    _need(_same_bits(qmean, value.query_full_mean) and _same_bits(rmean, value.reference_full_mean)
          and _same_bits(mass, value.visibility_mass), "FULL_GRID_MEAN_OR_MASS_NOT_EXACT_REPLAY")
    _need(value.logical_sha256 == _proposal_hash(value), "PROPOSAL_SOURCE_OR_FIELD_HASH_DRIFT")
    return value


def make_visibility_proposal(*, query_resource_key: str, candidate_resource_key: str,
                            reference_resource_key: str, query_grid_shape, reference_grid_shape,
                            query_visibility: torch.Tensor, reference_visibility: torch.Tensor,
                            source_binding: Mapping[str, str], query_axis=None, reference_axis=None) -> VisibilityProposal:
    qshape = _shape(query_grid_shape, "QUERY_GRID")
    rshape = _shape(reference_grid_shape, "REFERENCE_GRID")
    _visibility(query_visibility, math.prod(qshape), "QUERY_VISIBILITY")
    _visibility(reference_visibility, math.prod(rshape), "REFERENCE_VISIBILITY")
    qv, rv = query_visibility.detach().clone().contiguous(), reference_visibility.detach().clone().contiguous()
    qa = torch.arange(len(qv), dtype=torch.int64) if query_axis is None else query_axis.detach().clone()
    ra = torch.arange(len(rv), dtype=torch.int64) if reference_axis is None else reference_axis.detach().clone()
    qmean, rmean = qv.mean(), rv.mean()
    provisional = VisibilityProposal(query_resource_key, candidate_resource_key, reference_resource_key,
        qshape, rshape, qa, ra, qv, rv, qmean, rmean, torch.sqrt(qmean * rmean), _binding(source_binding), "PENDING")
    return validate_visibility_proposal(replace(provisional, logical_sha256=_proposal_hash(provisional)))


def serialize_visibility_proposal(value: VisibilityProposal) -> dict:
    value = validate_visibility_proposal(value)
    return {**_proposal_payload(value), "logical_sha256": value.logical_sha256}


def deserialize_visibility_proposal(value: Mapping) -> VisibilityProposal:
    expected = {"schema", "query_resource_key", "candidate_resource_key", "reference_resource_key",
        "query_grid_shape", "reference_grid_shape", "query_axis", "reference_axis", "query_visibility_binary64",
        "reference_visibility_binary64", "query_full_mean_binary64", "reference_full_mean_binary64",
        "visibility_mass_binary64", "source_binding", "logical_sha256"}
    _need(isinstance(value, Mapping) and set(value) == expected and value["schema"] == SCHEMA, "SERIALIZED_PROPOSAL_SCHEMA_DRIFT")
    for field in ("query_axis", "reference_axis"):
        _need(isinstance(value[field], list) and all(type(x) is int for x in value[field]), "SERIALIZED_AXIS_NOT_INTEGERS")
    def vector(field):
        items = value[field]
        _need(isinstance(items, list) and all(isinstance(x, str) for x in items), "SERIALIZED_VISIBILITY_NOT_BINARY64_STRINGS")
        return torch.tensor([float.fromhex(x) for x in items], dtype=torch.float64)
    def scalar(field):
        _need(isinstance(value[field], str), "SERIALIZED_SCALAR_NOT_BINARY64_STRING")
        return torch.tensor(float.fromhex(value[field]), dtype=torch.float64)
    proposal = VisibilityProposal(value["query_resource_key"], value["candidate_resource_key"], value["reference_resource_key"],
        _shape(value["query_grid_shape"], "QUERY_GRID"), _shape(value["reference_grid_shape"], "REFERENCE_GRID"),
        torch.tensor(value["query_axis"], dtype=torch.int64), torch.tensor(value["reference_axis"], dtype=torch.int64),
        vector("query_visibility_binary64"), vector("reference_visibility_binary64"), scalar("query_full_mean_binary64"),
        scalar("reference_full_mean_binary64"), scalar("visibility_mass_binary64"), _binding(value["source_binding"]), value["logical_sha256"])
    return validate_visibility_proposal(proposal)


def _legacy_score(q, r, wq, wr):
    # Keep this exact scalar sequence identical to original score(), including
    # normalization defaults, whole-reference mean, and denominator clamp.
    qn = F.normalize(q.to(torch.float64), dim=1)
    rn = F.normalize(r.to(torch.float64), dim=1)
    sim = qn @ rn.T
    local = (sim * wr[None]).max(1).values
    mass = torch.sqrt(wq.mean() * wr.mean())
    return mass * (wq * local).sum() / wq.sum().clamp_min(1e-12), mass, local


@dataclass(frozen=True)
class VisibilityEvidence:
    real_score: torch.Tensor
    visibility_mass: torch.Tensor
    query_control_score: torch.Tensor
    reference_control_score: torch.Tensor
    query_control_mass: torch.Tensor
    reference_control_mass: torch.Tensor
    real_local_values: torch.Tensor
    query_control_local_values: torch.Tensor
    reference_control_local_values: torch.Tensor
    proposal_logical_sha256: str

    def as_old_feature_record(self):
        return {field: float(getattr(self, field).detach()) for field in (
            "real_score", "visibility_mass", "query_control_score", "reference_control_score")}


def verify_visibility(proposal: VisibilityProposal, query_tokens: torch.Tensor, reference_tokens: torch.Tensor,
                      *, expected_source_binding: Mapping[str, str] | None = None) -> VisibilityEvidence:
    proposal = validate_visibility_proposal(proposal)
    if expected_source_binding is not None:
        _need(_binding(expected_source_binding) == dict(proposal.source_binding), "EXPECTED_SOURCE_BINDING_MISMATCH")
    for tokens, grid, key in ((query_tokens, proposal.query_grid_shape, "query_tokens_sha256"),
                              (reference_tokens, proposal.reference_grid_shape, "reference_tokens_sha256")):
        _need(isinstance(tokens, torch.Tensor) and tokens.device.type == "cpu" and tokens.is_floating_point()
              and tokens.shape == (math.prod(grid), FEATURE_DIMENSION) and bool(torch.isfinite(tokens).all()), "RAW_TOKEN_FULL_GRID_INVALID")
        _need(tensor_sha256(tokens) == proposal.source_binding[key], "RAW_TOKEN_SOURCE_BINDING_MISMATCH:" + key)
    wq, wr = proposal.query_visibility, proposal.reference_visibility
    real, mass, local = _legacy_score(query_tokens, reference_tokens, wq, wr)
    qc, qm, qlocal = _legacy_score(query_tokens, reference_tokens, wq.roll(max(1, wq.numel() // 2)), wr)
    rc, rm, rlocal = _legacy_score(query_tokens, reference_tokens, wq, wr.roll(max(1, wr.numel() // 2)))
    _need(_same_bits(mass, proposal.visibility_mass), "VERIFIER_MASS_DIFFERS_FROM_FULL_PROPOSAL_MASS")
    _need(all(bool(torch.isfinite(value).all()) for value in (real, mass, qc, rc, qm, rm, local, qlocal, rlocal)), "NONFINITE_VERIFICATION")
    return VisibilityEvidence(real, mass, qc, rc, qm, rm, local, qlocal, rlocal, proposal.logical_sha256)
