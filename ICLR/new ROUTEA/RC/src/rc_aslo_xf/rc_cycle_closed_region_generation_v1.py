"""Score-free, source-token cycle-closed connected region generation.

This module accepts the geometry fields of an existing ProposalAtomBank by duck
typing.  The caller owns source-bank validation and candidate-axis provenance.
No feature, token, rank, label, confidence, score, or learned parameter is read.
All valid input atoms remain present in the bank.  The only eligibility test is
whether the actual source reverse warp lands in its original query token cell.

Cycle eligibility uses source geometry, including under P_COORD.  Connectivity
uses CURRENT token-coordinate bindings.  Every maximal qualifying component is
returned, without a best-component decision, score pruning, or a top-K limit.
An empty component family is structural H0; semantic no-match is not inferred.
"""

from __future__ import annotations

import json
import math
import numbers
from typing import Any


SCHEMA = "RC_CYCLE_CLOSED_REGION_GENERATION_V1"
MIN_QUERY_ATOMS = 4
MIN_DISTINCT_REFERENCE_ATOMS = 2
REFERENCE_CHEBYSHEV = 2


def _list(value: Any, name: str) -> list:
    """Read array storage, supporting CPU/GPU tensors and numpy-like arrays."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _integers(value: Any, name: str, length: int) -> list[int]:
    values = _list(value, name)
    if len(values) != length or any(
        isinstance(item, bool) or not isinstance(item, numbers.Integral)
        for item in values
    ):
        raise ValueError(f"{name} must contain {length} integers")
    return [int(item) for item in values]


def _shape(value: Any, name: str) -> tuple[int, int]:
    shape = _integers(value, name, 2)
    if any(item <= 0 for item in shape):
        raise ValueError(f"{name} must be positive")
    return shape[0], shape[1]


def _rc(value: Any, name: str, count: int) -> list[list[int]]:
    rows = _list(value, name)
    if len(rows) != count:
        raise ValueError(f"{name} must have shape [{count}, 2]")
    return [_integers(row, name, 2) for row in rows]


def _xy(value: Any, name: str, count: int) -> list[list[float]]:
    rows = _list(value, name)
    if len(rows) != count:
        raise ValueError(f"{name} must have shape [{count}, 2]")
    result = []
    for row in rows:
        row = _list(row, name)
        if len(row) != 2 or any(
            isinstance(item, bool) or not isinstance(item, numbers.Real)
            or not math.isfinite(float(item)) for item in row
        ):
            raise ValueError(f"{name} must contain finite [x, y] pairs")
        result.append([float(item) for item in row])
    return result


def _geometry(bank: Any) -> dict:
    """Validate only geometry essential to interpreting the proposal."""
    qshape = _shape(bank.query_grid_shape, "query_grid_shape")
    rshape = _shape(bank.reference_grid_shape, "reference_grid_shape")
    count = qshape[0] * qshape[1]
    rcount = rshape[0] * rshape[1]
    valid = _list(bank.valid_mask, "valid_mask")
    if len(valid) != count or any(not isinstance(item, bool) for item in valid):
        raise ValueError("valid_mask must be boolean on the full query-token axis")
    source_q = _integers(bank.source_query_indices, "source_query_indices", count)
    source_r = _integers(bank.source_reference_indices, "source_reference_indices", count)
    current_q = _integers(bank.query_indices, "query_indices", count)
    current_r = _integers(bank.reference_indices, "reference_indices", count)
    qperm = _integers(bank.query_coordinate_permutation, "query_coordinate_permutation", count)
    rperm = _integers(bank.reference_coordinate_permutation, "reference_coordinate_permutation", rcount)
    if source_q != list(range(count)):
        raise ValueError("source atom axis must contain every canonical query token once")
    if sorted(qperm) != list(range(count)) or sorted(rperm) != list(range(rcount)):
        raise ValueError("coordinate controls must be complete permutations")
    if any(index < 0 or index >= rcount for index in source_r):
        raise ValueError("source reference index outside reference grid")
    if current_q != [qperm[index] for index in source_q]:
        raise ValueError("query indices do not replay from the source permutation")
    if current_r != [rperm[index] for index in source_r]:
        raise ValueError("reference indices do not replay from the source permutation")
    qrc = _rc(bank.query_rc, "query_rc", count)
    rrc = _rc(bank.reference_rc, "reference_rc", count)
    if qrc != [[index // qshape[1], index % qshape[1]] for index in current_q]:
        raise ValueError("query index and current coordinate binding disagree")
    if rrc != [[index // rshape[1], index % rshape[1]] for index in current_r]:
        raise ValueError("reference index and current coordinate binding disagree")
    reverse_xy = _xy(bank.source_reverse_query_xy, "source_reverse_query_xy", count)
    return dict(qshape=qshape, rshape=rshape, count=count, valid=valid,
                source_q=source_q, source_r=source_r, current_q=current_q,
                current_r=current_r, qrc=qrc, rrc=rrc, reverse_xy=reverse_xy)


def enumerate_cycle_closed_regions(bank: Any) -> dict:
    """Return every cycle-closed maximal connected multi-patch component.

    For source reverse coordinates (x, y) in [-1, 1)^2, the returned cell is
    (floor((y+1)*height/2), floor((x+1)*width/2)).  Values outside the domain
    are ineligible and never clamped.  Eligibility requires a valid atom whose
    returned row-major query cell equals its ORIGINAL source query index.

    Edges join CURRENT four-neighbour query cells only when their CURRENT
    reference cells have Chebyshev distance at most two.  A component must have
    at least four query atoms and at least two distinct assigned reference
    indices.  Members and components are canonically ordered by current query
    index; original atom indices preserve the source assignment provenance.
    """
    g = _geometry(bank)
    height, width = g["qshape"]
    returned_indices: list[int | None] = []
    for x, y in g["reverse_xy"]:
        if -1.0 <= x < 1.0 and -1.0 <= y < 1.0:
            # Evaluate the specified floor exactly for the represented binary
            # coordinate.  A floating x+1 can round to 2 for x immediately below
            # 1, spuriously mapping an in-domain point outside the last cell.
            # Integer ratios implement the same expression without clamping.
            xn, xd = x.as_integer_ratio()
            yn, yd = y.as_integer_ratio()
            column = ((xn + xd) * width) // (2 * xd)
            row = ((yn + yd) * height) // (2 * yd)
            returned_indices.append(row * width + column)
        else:
            returned_indices.append(None)
    eligible = [index for index in range(g["count"])
                if g["valid"][index] and returned_indices[index] == g["source_q"][index]]
    by_cell = {tuple(g["qrc"][index]): index for index in eligible}
    adjacency: dict[int, list[int]] = {index: [] for index in eligible}
    for index in eligible:
        row, column = g["qrc"][index]
        for cell in ((row - 1, column), (row + 1, column),
                     (row, column - 1), (row, column + 1)):
            neighbour = by_cell.get(cell)
            if neighbour is not None and max(
                abs(a - b) for a, b in zip(g["rrc"][index], g["rrc"][neighbour])
            ) <= REFERENCE_CHEBYSHEV:
                adjacency[index].append(neighbour)
    unseen = set(eligible)
    raw_components = []
    while unseen:
        start = min(unseen, key=lambda index: g["current_q"][index])
        unseen.remove(start)
        pending = [start]
        members = []
        while pending:
            index = pending.pop()
            members.append(index)
            for neighbour in adjacency[index]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    pending.append(neighbour)
        raw_components.append(sorted(members, key=lambda index: g["current_q"][index]))
    raw_components.sort(key=lambda members: tuple(g["current_q"][index] for index in members))
    components = []
    rejected_small = 0
    rejected_reference = 0
    for members in raw_components:
        if len(members) < MIN_QUERY_ATOMS:
            rejected_small += 1
            continue
        if len({g["current_r"][index] for index in members}) < MIN_DISTINCT_REFERENCE_ATOMS:
            rejected_reference += 1
            continue
        components.append({
            "component_ordinal": len(components),
            "atom_indices": members,
            "query_indices": [g["current_q"][index] for index in members],
            "query_rc": [g["qrc"][index] for index in members],
            "reference_indices": [g["current_r"][index] for index in members],
            "reference_rc": [g["rrc"][index] for index in members],
            "source_query_indices": [g["source_q"][index] for index in members],
            "source_reference_indices": [g["source_r"][index] for index in members],
            "source_reverse_query_xy": [g["reverse_xy"][index] for index in members],
            "source_returned_query_indices": [returned_indices[index] for index in members],
        })
    if components:
        reason = None
    elif not any(g["valid"]):
        reason = "NO_VALID_ATOMS"
    elif not eligible:
        reason = "NO_TOKEN_CYCLE_CLOSED_ATOMS"
    else:
        reason = "NO_QUALIFYING_CONNECTED_COMPONENTS"
    result = {
        "schema": SCHEMA,
        "query_grid_shape": list(g["qshape"]),
        "reference_grid_shape": list(g["rshape"]),
        "components": components,
        "structural_h0": not bool(components),
        "h0_reason": "CYCLE_CLOSED_REGION_FAMILY_EMPTY" if reason else None,
        "h0_detail": reason,
        "cycle_closed_atom_indices": eligible,
        "source_returned_query_indices": returned_indices,
        "counts": {
            "input_atoms": g["count"],
            "valid_atoms": sum(g["valid"]),
            "valid_in_domain_reverse_atoms": sum(
                g["valid"][index] and returned_indices[index] is not None
                for index in range(g["count"])),
            "cycle_closed_atoms": len(eligible),
            "maximal_components": len(raw_components),
            "rejected_fewer_than_four_query_atoms": rejected_small,
            "rejected_fewer_than_two_reference_atoms": rejected_reference,
            "qualifying_components": len(components),
            "atoms_in_qualifying_components": sum(len(item["atom_indices"]) for item in components),
        },
    }
    # Resource keys are copied after generation; they never affect eligibility,
    # component enumeration, order, or H0.  No filename parsing is performed.
    for name in ("query_resource_key", "candidate_resource_key", "reference_resource_key"):
        value = getattr(bank, name, None)
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{name} metadata must be a string")
            result[name] = value
    return result


def serialize_cycle_closed_regions(result: dict) -> str:
    """Serialize exact primitive replay evidence; no intermediate hashes."""
    return json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)


def replay_cycle_closed_regions(bank: Any, serialized: str) -> dict:
    """Regenerate through the sole enumerator and require complete equality."""
    original = json.loads(serialized)
    regenerated = enumerate_cycle_closed_regions(bank)
    if serialize_cycle_closed_regions(original) != serialize_cycle_closed_regions(regenerated):
        raise ValueError("cycle-closed region serialization replay mismatch")
    return regenerated
