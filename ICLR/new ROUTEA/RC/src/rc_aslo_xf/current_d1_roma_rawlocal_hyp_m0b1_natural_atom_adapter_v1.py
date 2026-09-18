"""Natural RoMa dense-output to five-coordinate Proposal atom adapter.

This module is deliberately narrower than the historical nine-coordinate
RoMa/ColNomic atom builder.  It reads only the six dense tensors emitted by
RoMa and geometry that has already been frozen in the EXIF-oriented image
frame.  In particular, it does not read local image tokens or a retrieval
base.

The frozen cell-box order is authoritative.  A box centre is used to sample
the RoMa fields at the same token ordinal; boxes are never sorted or inferred
from a nominal grid.  This matters for EXIF rotations, for which token ordinal
zero may legitimately be located at the right edge of the oriented image.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
from typing import Any, Mapping

import torch
from torch.nn import functional as F


SCHEMA_VERSION = "current_d1_roma_rawlocal_hyp_m0b1_natural_atom_payload_v1_20260907"
FEATURE_NAMES = (
    "overlap_q",
    "overlap_r",
    "sqrt_overlap_product",
    "cycle_quality",
    "precision_quality",
)
FEATURE_DIM = 5
COORDINATE_FRAME = "EXIF_ORIENTED_FULL_IMAGE_NORMALIZED_XYXY_V1"
NULL_SHA256 = "0" * 64
DENSE_OUTPUT_NAMES = (
    "warp_AB",
    "overlap_AB",
    "precision_AB",
    "warp_BA",
    "overlap_BA",
    "precision_BA",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _logical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_grid(value: tuple[int, int], name: str) -> tuple[int, int]:
    shape = tuple(int(item) for item in value)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"{name} must be a positive two-dimensional grid")
    return shape


def _as_boxes(
    value: torch.Tensor | list[list[float]],
    count: int,
    name: str,
) -> torch.Tensor:
    boxes = torch.as_tensor(value, dtype=torch.float64).detach().cpu().contiguous()
    if tuple(boxes.shape) != (count, 4) or not bool(torch.isfinite(boxes).all()):
        raise ValueError(f"{name} cell boxes must be finite FP64 [N,4]")
    if (
        bool(((boxes < 0.0) | (boxes > 1.0)).any())
        or not bool((boxes[:, 2] > boxes[:, 0]).all())
        or not bool((boxes[:, 3] > boxes[:, 1]).all())
    ):
        raise ValueError(f"{name} cell boxes are outside the oriented unit image")
    return boxes


def oriented_box_centres_xy(value: torch.Tensor) -> torch.Tensor:
    """Return normalized RoMa centres without changing the supplied order."""
    boxes = torch.as_tensor(value, dtype=torch.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("oriented cell boxes must have shape [N,4]")
    # 2 * ((lo + hi) / 2) - 1, deliberately in the input ordinal order.
    return torch.stack(
        (boxes[:, 0] + boxes[:, 2] - 1.0, boxes[:, 1] + boxes[:, 3] - 1.0),
        dim=1,
    ).contiguous()


def _validate_dense_outputs(outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if set(outputs) != set(DENSE_OUTPUT_NAMES):
        raise ValueError("exactly six canonical RoMa dense outputs are required")
    result = {name: torch.as_tensor(outputs[name]) for name in DENSE_OUTPUT_NAMES}
    dtypes = {item.dtype for item in result.values()}
    devices = {item.device for item in result.values()}
    if len(dtypes) != 1 or next(iter(dtypes)) not in (torch.float32, torch.float64):
        raise ValueError("RoMa dense outputs must share FP32 or FP64 arithmetic")
    if len(devices) != 1 or not all(bool(torch.isfinite(item).all()) for item in result.values()):
        raise ValueError("RoMa dense outputs must share a device and be finite")
    wab, oab, pab = result["warp_AB"], result["overlap_AB"], result["precision_AB"]
    wba, oba, pba = result["warp_BA"], result["overlap_BA"], result["precision_BA"]
    if (
        wab.ndim != 4
        or wab.shape[0] != 1
        or wab.shape[-1] != 2
        or tuple(oab.shape) != (*wab.shape[:3], 1)
        or tuple(pab.shape) != (*wab.shape[:3], 2, 2)
        or wba.ndim != 4
        or wba.shape[0] != 1
        or wba.shape[-1] != 2
        or tuple(oba.shape) != (*wba.shape[:3], 1)
        or tuple(pba.shape) != (*wba.shape[:3], 2, 2)
        or min(int(wab.shape[1]), int(wab.shape[2]), int(wba.shape[1]), int(wba.shape[2])) <= 0
    ):
        raise ValueError("RoMa directional dense-output shape drift")
    return result


def _sample_bhwc(field: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(field)
    channels = int(torch.tensor(value.shape[3:]).prod())
    flat = value.reshape(1, value.shape[1], value.shape[2], channels).permute(0, 3, 1, 2)
    grid = torch.as_tensor(xy, dtype=value.dtype, device=value.device).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        flat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[0, :, :, 0].T.reshape(xy.shape[0], *value.shape[3:])


@dataclass(frozen=True)
class NaturalRoMaAtomPayloadV1:
    """Sealed primitive payload for one query/candidate endpoint."""

    schema_version: str
    feature_names: tuple[str, ...]
    query_resource_key: str
    candidate_resource_key: str
    reference_resource_key: str
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    coordinate_frame: str
    control_namespace: str
    source_binding_sha256: str
    parent_real_binding_sha256: str
    control_plan_sha256: str
    control_realization_sha256: str
    arithmetic_dtype: str
    features: torch.Tensor
    valid_mask: torch.Tensor
    query_valid_axis: torch.Tensor
    reference_valid_axis: torch.Tensor
    query_cell_boxes_xyxy: torch.Tensor
    reference_cell_boxes_xyxy: torch.Tensor
    query_centres_xy: torch.Tensor
    reference_centres_xy: torch.Tensor
    query_indices: torch.Tensor
    reference_indices: torch.Tensor
    query_coordinates: torch.Tensor
    reference_coordinates: torch.Tensor
    forward_reference_xy: torch.Tensor
    reverse_query_xy: torch.Tensor
    sampled_overlap_q: torch.Tensor
    sampled_overlap_r: torch.Tensor
    sampled_precision_ab: torch.Tensor
    sampled_precision_ba: torch.Tensor
    cycle_error: torch.Tensor
    dense_output_sha256: tuple[tuple[str, str], ...]
    primitive_tensor_sha256: tuple[tuple[str, str], ...]
    logical_sha256: str


_PRIMITIVE_TENSOR_FIELDS = tuple(
    item.name
    for item in fields(NaturalRoMaAtomPayloadV1)
    if item.name
    in {
        "features",
        "valid_mask",
        "query_valid_axis",
        "reference_valid_axis",
        "query_cell_boxes_xyxy",
        "reference_cell_boxes_xyxy",
        "query_centres_xy",
        "reference_centres_xy",
        "query_indices",
        "reference_indices",
        "query_coordinates",
        "reference_coordinates",
        "forward_reference_xy",
        "reverse_query_xy",
        "sampled_overlap_q",
        "sampled_overlap_r",
        "sampled_precision_ab",
        "sampled_precision_ba",
        "cycle_error",
    }
)


def _primitive_hashes(value: NaturalRoMaAtomPayloadV1) -> tuple[tuple[str, str], ...]:
    return tuple((name, tensor_sha256(getattr(value, name))) for name in _PRIMITIVE_TENSOR_FIELDS)


def _payload_logical(value: NaturalRoMaAtomPayloadV1) -> str:
    return _logical_sha256(
        {
            "schema_version": value.schema_version,
            "feature_names": list(value.feature_names),
            "query_resource_key": value.query_resource_key,
            "candidate_resource_key": value.candidate_resource_key,
            "reference_resource_key": value.reference_resource_key,
            "query_grid_shape": list(value.query_grid_shape),
            "reference_grid_shape": list(value.reference_grid_shape),
            "coordinate_frame": value.coordinate_frame,
            "control_namespace": value.control_namespace,
            "source_binding_sha256": value.source_binding_sha256,
            "parent_real_binding_sha256": value.parent_real_binding_sha256,
            "control_plan_sha256": value.control_plan_sha256,
            "control_realization_sha256": value.control_realization_sha256,
            "arithmetic_dtype": value.arithmetic_dtype,
            "dense_output_sha256": [list(item) for item in value.dense_output_sha256],
            "primitive_tensor_sha256": [list(item) for item in value.primitive_tensor_sha256],
        }
    )


def _seal(value: NaturalRoMaAtomPayloadV1) -> NaturalRoMaAtomPayloadV1:
    staged = replace(value, primitive_tensor_sha256=_primitive_hashes(value), logical_sha256="PENDING")
    return replace(staged, logical_sha256=_payload_logical(staged))


def build_natural_roma_atom_payload(
    *,
    query_resource_key: str,
    candidate_resource_key: str,
    reference_resource_key: str,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    query_cell_boxes_xyxy: torch.Tensor | list[list[float]],
    reference_cell_boxes_xyxy: torch.Tensor | list[list[float]],
    query_valid_mask: torch.Tensor | list[bool],
    reference_valid_mask: torch.Tensor | list[bool],
    warp_ab: torch.Tensor,
    overlap_ab: torch.Tensor,
    precision_ab: torch.Tensor,
    warp_ba: torch.Tensor,
    overlap_ba: torch.Tensor,
    precision_ba: torch.Tensor,
    source_binding_sha256: str,
    control_namespace: str = "REAL",
    parent_real_binding_sha256: str = NULL_SHA256,
    control_plan_sha256: str = NULL_SHA256,
    control_realization_sha256: str = NULL_SHA256,
) -> NaturalRoMaAtomPayloadV1:
    """Build one complete query-cell atom axis from natural RoMa outputs."""
    qshape = _positive_grid(query_grid_shape, "query grid")
    rshape = _positive_grid(reference_grid_shape, "reference grid")
    qcount, rcount = qshape[0] * qshape[1], rshape[0] * rshape[1]
    qboxes = _as_boxes(query_cell_boxes_xyxy, qcount, "query")
    rboxes = _as_boxes(reference_cell_boxes_xyxy, rcount, "reference")
    qvalid_cpu = torch.as_tensor(query_valid_mask, dtype=torch.bool).detach().cpu().contiguous()
    rvalid_cpu = torch.as_tensor(reference_valid_mask, dtype=torch.bool).detach().cpu().contiguous()
    if tuple(qvalid_cpu.shape) != (qcount,) or tuple(rvalid_cpu.shape) != (rcount,):
        raise ValueError("canonical valid-mask axis drift")
    if not bool(rvalid_cpu.any()):
        raise ValueError("reference has no valid canonical cell")
    if any(not isinstance(item, str) or not item for item in (
        query_resource_key,
        candidate_resource_key,
        reference_resource_key,
        control_namespace,
    )) or not _is_sha256(source_binding_sha256):
        raise ValueError("resource or source binding is invalid")
    control_hashes = (parent_real_binding_sha256, control_plan_sha256, control_realization_sha256)
    if not all(_is_sha256(item) for item in control_hashes):
        raise ValueError("control provenance hash is invalid")
    if control_namespace == "REAL":
        if candidate_resource_key != reference_resource_key or any(item != NULL_SHA256 for item in control_hashes):
            raise ValueError("REAL endpoint binding is invalid")
    elif any(item == NULL_SHA256 for item in control_hashes):
        raise ValueError("controlled endpoint lacks parent/plan provenance")

    outputs = _validate_dense_outputs(
        {
            "warp_AB": warp_ab,
            "overlap_AB": overlap_ab,
            "precision_AB": precision_ab,
            "warp_BA": warp_ba,
            "overlap_BA": overlap_ba,
            "precision_BA": precision_ba,
        }
    )
    device = outputs["warp_AB"].device
    arithmetic_dtype = outputs["warp_AB"].dtype
    qcentres_cpu = oriented_box_centres_xy(qboxes)
    rcentres_cpu = oriented_box_centres_xy(rboxes)
    qcentres = qcentres_cpu.to(device=device, dtype=arithmetic_dtype)
    rcentres = rcentres_cpu.to(device=device, dtype=arithmetic_dtype)

    mapped_device = _sample_bhwc(outputs["warp_AB"], qcentres).reshape(-1, 2)
    overlap_q_device = _sample_bhwc(outputs["overlap_AB"], qcentres).reshape(-1).clamp(0.0, 1.0)
    reverse_device = _sample_bhwc(outputs["warp_BA"], mapped_device).reshape(-1, 2)
    overlap_r_device = _sample_bhwc(outputs["overlap_BA"], mapped_device).reshape(-1).clamp(0.0, 1.0)
    precision_forward_device = _sample_bhwc(outputs["precision_AB"], qcentres).reshape(-1, 2, 2)
    precision_reverse_device = _sample_bhwc(outputs["precision_BA"], mapped_device).reshape(-1, 2, 2)

    # Freeze sampled primitives first, then perform all reductions and nearest
    # assignment on CPU in the original arithmetic dtype.  The artifact is
    # therefore internally exactly replayable without asserting that a second
    # GPU/kernel must reproduce RoMa's dense bytes.
    mapped = mapped_device.detach().cpu().contiguous()
    reverse = reverse_device.detach().cpu().contiguous()
    overlap_q = overlap_q_device.detach().cpu().contiguous()
    overlap_r = overlap_r_device.detach().cpu().contiguous()
    precision_forward = precision_forward_device.detach().cpu().contiguous()
    precision_reverse = precision_reverse_device.detach().cpu().contiguous()
    qcentres_native = qcentres.detach().cpu().contiguous()
    rcentres_native = rcentres.detach().cpu().contiguous()
    distance = torch.cdist(mapped.to(torch.float32), rcentres_native.to(torch.float32))
    distance[:, ~rvalid_cpu] = torch.inf
    reference_indices = distance.argmin(dim=1)
    valid = qvalid_cpu & mapped.abs().lt(1.0).all(dim=1) & rvalid_cpu[reference_indices]
    cycle = torch.linalg.vector_norm(reverse - qcentres_native, dim=1)
    reciprocal = torch.sqrt((overlap_q * overlap_r).clamp_min(0.0))
    cycle_quality = torch.exp(-4.0 * cycle)
    symmetric_forward = (precision_forward + precision_forward.transpose(1, 2)) * 0.5
    symmetric_reverse = (precision_reverse + precision_reverse.transpose(1, 2)) * 0.5
    minimum_eigenvalue = torch.minimum(
        torch.linalg.eigvalsh(symmetric_forward).amin(dim=1),
        torch.linalg.eigvalsh(symmetric_reverse).amin(dim=1),
    ).clamp_min(0.0)
    precision_quality = torch.tanh(torch.log1p(minimum_eigenvalue))
    feature_native = torch.stack(
        (overlap_q, overlap_r, reciprocal, cycle_quality, precision_quality),
        dim=1,
    )
    feature_native = torch.where(valid[:, None], feature_native, torch.zeros_like(feature_native))

    query_indices = torch.arange(qcount, dtype=torch.int64)
    reference_axis = torch.arange(rcount, dtype=torch.int64)
    query_axis = torch.stack((query_indices // qshape[1], query_indices % qshape[1]), dim=1)
    reference_axis_rc = torch.stack(
        (reference_axis // rshape[1], reference_axis % rshape[1]),
        dim=1,
    )
    dense_hashes = tuple((name, tensor_sha256(outputs[name])) for name in DENSE_OUTPUT_NAMES)

    payload = NaturalRoMaAtomPayloadV1(
        schema_version=SCHEMA_VERSION,
        feature_names=FEATURE_NAMES,
        query_resource_key=query_resource_key,
        candidate_resource_key=candidate_resource_key,
        reference_resource_key=reference_resource_key,
        query_grid_shape=qshape,
        reference_grid_shape=rshape,
        coordinate_frame=COORDINATE_FRAME,
        control_namespace=control_namespace,
        source_binding_sha256=source_binding_sha256,
        parent_real_binding_sha256=parent_real_binding_sha256,
        control_plan_sha256=control_plan_sha256,
        control_realization_sha256=control_realization_sha256,
        arithmetic_dtype=str(arithmetic_dtype),
        features=feature_native.to(torch.float64).contiguous(),
        valid_mask=valid.contiguous(),
        query_valid_axis=qvalid_cpu,
        reference_valid_axis=rvalid_cpu,
        query_cell_boxes_xyxy=qboxes,
        reference_cell_boxes_xyxy=rboxes,
        query_centres_xy=qcentres_cpu,
        reference_centres_xy=rcentres_cpu,
        query_indices=query_indices.contiguous(),
        reference_indices=reference_indices.to(torch.int64).contiguous(),
        query_coordinates=query_axis.to(torch.int64).contiguous(),
        reference_coordinates=reference_axis_rc[reference_indices].to(torch.int64).contiguous(),
        forward_reference_xy=mapped.to(torch.float64).contiguous(),
        reverse_query_xy=reverse.to(torch.float64).contiguous(),
        sampled_overlap_q=overlap_q.to(torch.float64).contiguous(),
        sampled_overlap_r=overlap_r.to(torch.float64).contiguous(),
        sampled_precision_ab=precision_forward.to(torch.float64).contiguous(),
        sampled_precision_ba=precision_reverse.to(torch.float64).contiguous(),
        cycle_error=cycle.to(torch.float64).contiguous(),
        dense_output_sha256=dense_hashes,
        primitive_tensor_sha256=(),
        logical_sha256="PENDING",
    )
    return validate_natural_roma_atom_payload(_seal(payload))


def _native_dtype(value: str) -> torch.dtype:
    if value == "torch.float32":
        return torch.float32
    if value == "torch.float64":
        return torch.float64
    raise ValueError("unsupported RoMa arithmetic dtype")


def validate_natural_roma_atom_payload(
    value: NaturalRoMaAtomPayloadV1,
) -> NaturalRoMaAtomPayloadV1:
    """Replay schema, geometry, equations, and all primitive tensor seals."""
    qshape = _positive_grid(value.query_grid_shape, "query grid")
    rshape = _positive_grid(value.reference_grid_shape, "reference grid")
    qcount, rcount = qshape[0] * qshape[1], rshape[0] * rshape[1]
    if (
        value.schema_version != SCHEMA_VERSION
        or value.feature_names != FEATURE_NAMES
        or value.coordinate_frame != COORDINATE_FRAME
        or not all(isinstance(item, str) and item for item in (
            value.query_resource_key,
            value.candidate_resource_key,
            value.reference_resource_key,
            value.control_namespace,
        ))
        or not _is_sha256(value.source_binding_sha256)
    ):
        raise ValueError("natural RoMa atom metadata drift")
    control_hashes = (value.parent_real_binding_sha256, value.control_plan_sha256, value.control_realization_sha256)
    if not all(_is_sha256(item) for item in control_hashes):
        raise ValueError("natural RoMa control provenance drift")
    if value.control_namespace == "REAL":
        if value.candidate_resource_key != value.reference_resource_key or any(item != NULL_SHA256 for item in control_hashes):
            raise ValueError("REAL natural RoMa atom binding drift")
    elif any(item == NULL_SHA256 for item in control_hashes):
        raise ValueError("controlled natural RoMa atom provenance drift")
    dtype = _native_dtype(value.arithmetic_dtype)
    expected = {
        "features": (torch.float64, (qcount, FEATURE_DIM)),
        "valid_mask": (torch.bool, (qcount,)),
        "query_valid_axis": (torch.bool, (qcount,)),
        "reference_valid_axis": (torch.bool, (rcount,)),
        "query_cell_boxes_xyxy": (torch.float64, (qcount, 4)),
        "reference_cell_boxes_xyxy": (torch.float64, (rcount, 4)),
        "query_centres_xy": (torch.float64, (qcount, 2)),
        "reference_centres_xy": (torch.float64, (rcount, 2)),
        "query_indices": (torch.int64, (qcount,)),
        "reference_indices": (torch.int64, (qcount,)),
        "query_coordinates": (torch.int64, (qcount, 2)),
        "reference_coordinates": (torch.int64, (qcount, 2)),
        "forward_reference_xy": (torch.float64, (qcount, 2)),
        "reverse_query_xy": (torch.float64, (qcount, 2)),
        "sampled_overlap_q": (torch.float64, (qcount,)),
        "sampled_overlap_r": (torch.float64, (qcount,)),
        "sampled_precision_ab": (torch.float64, (qcount, 2, 2)),
        "sampled_precision_ba": (torch.float64, (qcount, 2, 2)),
        "cycle_error": (torch.float64, (qcount,)),
    }
    for name, (expected_dtype, expected_shape) in expected.items():
        item = torch.as_tensor(getattr(value, name))
        if item.dtype != expected_dtype or tuple(item.shape) != expected_shape or not item.is_contiguous():
            raise ValueError(f"{name} dtype/shape/contiguity drift")
        if item.is_floating_point() and not bool(torch.isfinite(item).all()):
            raise ValueError(f"{name} contains non-finite values")
    _as_boxes(value.query_cell_boxes_xyxy, qcount, "query")
    _as_boxes(value.reference_cell_boxes_xyxy, rcount, "reference")
    if not bool(value.reference_valid_axis.any()):
        raise ValueError("reference has no valid canonical cell")
    if not torch.equal(value.query_centres_xy, oriented_box_centres_xy(value.query_cell_boxes_xyxy)) or not torch.equal(
        value.reference_centres_xy, oriented_box_centres_xy(value.reference_cell_boxes_xyxy)
    ):
        raise ValueError("oriented box-centre provenance drift")
    expected_query_indices = torch.arange(qcount, dtype=torch.int64)
    expected_query_coordinates = torch.stack(
        (expected_query_indices // qshape[1], expected_query_indices % qshape[1]),
        dim=1,
    )
    if not torch.equal(value.query_indices, expected_query_indices) or not torch.equal(
        value.query_coordinates, expected_query_coordinates
    ):
        raise ValueError("query ordinal was reordered")
    if not bool(((value.reference_indices >= 0) & (value.reference_indices < rcount)).all()):
        raise ValueError("nearest reference index outside canonical axis")
    reference_coordinates = torch.stack(
        (value.reference_indices // rshape[1], value.reference_indices % rshape[1]),
        dim=1,
    )
    if not torch.equal(value.reference_coordinates, reference_coordinates):
        raise ValueError("reference index/coordinate binding drift")
    distance = torch.cdist(
        value.forward_reference_xy.to(torch.float32),
        value.reference_centres_xy.to(torch.float32),
    )
    distance[:, ~value.reference_valid_axis] = torch.inf
    if not torch.equal(value.reference_indices, distance.argmin(dim=1)):
        raise ValueError("nearest reference-cell assignment does not replay")
    replay_valid = (
        value.query_valid_axis
        & value.forward_reference_xy.abs().lt(1.0).all(dim=1)
        & value.reference_valid_axis[value.reference_indices]
    )
    if not torch.equal(value.valid_mask, replay_valid):
        raise ValueError("natural RoMa atom validity does not replay")

    # Recompute in the original RoMa arithmetic dtype, then widen exactly as
    # the adapter does.  This avoids silently changing an FP32 model output by
    # recomputing its equations in FP64.
    oq = value.sampled_overlap_q.to(dtype)
    orev = value.sampled_overlap_r.to(dtype)
    cycle = value.cycle_error.to(dtype)
    pf = value.sampled_precision_ab.to(dtype)
    pr = value.sampled_precision_ba.to(dtype)
    reciprocal = torch.sqrt((oq * orev).clamp_min(0.0))
    cycle_quality = torch.exp(-4.0 * cycle)
    sf = (pf + pf.transpose(1, 2)) * 0.5
    sr = (pr + pr.transpose(1, 2)) * 0.5
    minimum_eigenvalue = torch.minimum(
        torch.linalg.eigvalsh(sf).amin(dim=1),
        torch.linalg.eigvalsh(sr).amin(dim=1),
    ).clamp_min(0.0)
    precision_quality = torch.tanh(torch.log1p(minimum_eigenvalue))
    replay_feature = torch.stack((oq, orev, reciprocal, cycle_quality, precision_quality), dim=1)
    replay_feature = torch.where(value.valid_mask[:, None], replay_feature, torch.zeros_like(replay_feature))
    if not torch.equal(value.features, replay_feature.to(torch.float64)):
        raise ValueError("five-coordinate RoMa feature equations do not replay")
    if bool(((value.features < 0.0) | (value.features > 1.0)).any()):
        raise ValueError("five-coordinate RoMa feature domain drift")
    if bool((~value.valid_mask).any()) and int(torch.count_nonzero(value.features[~value.valid_mask])) != 0:
        raise ValueError("invalid atom rows must be exact zero")
    if tuple(name for name, _digest in value.dense_output_sha256) != DENSE_OUTPUT_NAMES or not all(
        _is_sha256(digest) for _name, digest in value.dense_output_sha256
    ):
        raise ValueError("dense-output seal axis drift")
    if value.primitive_tensor_sha256 != _primitive_hashes(value):
        raise ValueError("primitive tensor seal drift")
    if value.logical_sha256 != _payload_logical(replace(value, logical_sha256="PENDING")):
        raise ValueError("natural RoMa atom logical seal drift")
    return value


def payload_as_mapping(value: NaturalRoMaAtomPayloadV1) -> dict[str, Any]:
    """Return a torch-save-ready mapping after strict replay validation."""
    validate_natural_roma_atom_payload(value)
    return {item.name: getattr(value, item.name) for item in fields(value)}


def payload_from_mapping(value: Mapping[str, Any]) -> NaturalRoMaAtomPayloadV1:
    expected = {item.name for item in fields(NaturalRoMaAtomPayloadV1)}
    if set(value) != expected:
        raise ValueError("natural RoMa atom payload field axis drift")
    return validate_natural_roma_atom_payload(NaturalRoMaAtomPayloadV1(**dict(value)))


def to_proposal_input(value: NaturalRoMaAtomPayloadV1):
    """Adapt a validated natural payload to the isolated five-D P core."""
    validate_natural_roma_atom_payload(value)
    # This is the current five-D P core, not the historical nine-D builder.
    from .current_d1_roma_rawlocal_hyp_e0_v1 import RoMaProposalInput, seal_proposal_input

    return seal_proposal_input(
        RoMaProposalInput(
            query_resource_key=value.query_resource_key,
            candidate_resource_key=value.candidate_resource_key,
            reference_resource_key=value.reference_resource_key,
            query_grid_shape=value.query_grid_shape,
            reference_grid_shape=value.reference_grid_shape,
            features=value.features,
            valid_mask=value.valid_mask,
            query_indices=value.query_indices,
            reference_indices=value.reference_indices,
            query_coordinates=value.query_coordinates,
            reference_coordinates=value.reference_coordinates,
            roma_payload_sha256="PENDING",
            coordinate_binding_sha256="PENDING",
            control_namespace=value.control_namespace,
            parent_real_binding_sha256=value.parent_real_binding_sha256,
            control_plan_sha256=value.control_plan_sha256,
            control_realization_sha256=value.control_realization_sha256,
        )
    )


__all__ = [
    "COORDINATE_FRAME",
    "DENSE_OUTPUT_NAMES",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "NULL_SHA256",
    "NaturalRoMaAtomPayloadV1",
    "build_natural_roma_atom_payload",
    "oriented_box_centres_xy",
    "payload_as_mapping",
    "payload_from_mapping",
    "tensor_sha256",
    "to_proposal_input",
    "validate_natural_roma_atom_payload",
]
