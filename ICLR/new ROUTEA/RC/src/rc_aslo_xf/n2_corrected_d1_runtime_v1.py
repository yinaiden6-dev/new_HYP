"""Target-free corrected-identity boundary for Route A N2 D1.

The historical D1 implementation remains the algorithm authority, but its
5,404 filename-derived label assertion is not a valid N2 gallery interface.
This module supplies the small, explicit successor boundary: corrected 5,412
identity reduction, deterministic natural-C128 construction, and query-id-only
fold joining.  It contains no query target interface and no training loop.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

from rc_aslo_xf.gallery_identity_repair import (
    CORRECTED_IDENTITY_COUNT,
    PHYSICAL_ROW_COUNT,
    build_identity_map,
)


NATURAL_CANDIDATE_COUNT = 128
CANONICAL_QUERY_COUNT = 987
CANONICAL_FOLD_COUNTS = {0: 212, 1: 205, 2: 189, 3: 188, 4: 193}
CANONICAL_TRACKS = {"outcome", "difficult", "new_difficult_train"}
LEGACY_LABEL_COUNT = 5404


class N2CorrectedD1RuntimeError(RuntimeError):
    """A fail-closed corrected-gallery or canonical-fold contract failure."""


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CorrectedIdentityReduction:
    identity_values: tuple[str, ...]
    identity_scores: torch.Tensor
    representative_physical_rows: torch.Tensor
    ranked_identity_slots: torch.Tensor

    @property
    def ranked_identities(self) -> tuple[str, ...]:
        return tuple(self.identity_values[int(slot)] for slot in self.ranked_identity_slots)

    @property
    def ranked_representative_rows(self) -> tuple[int, ...]:
        return tuple(
            int(self.representative_physical_rows[int(slot)])
            for slot in self.ranked_identity_slots
        )


@dataclass(frozen=True)
class NaturalC128:
    identities: tuple[str, ...]
    representative_physical_rows: tuple[int, ...]
    scores: torch.Tensor


@dataclass(frozen=True)
class CanonicalFoldAssignment:
    query_id: str
    query_ordinal: int
    heldout_fold: int
    track: str
    grid_h: int
    grid_w: int
    source_image_sha256: str


def corrected_labels_from_legacy(legacy_labels: Sequence[str]) -> tuple[str, ...]:
    """Apply only the frozen, independently registered gallery correction."""

    try:
        identity_map = build_identity_map(tuple(map(str, legacy_labels)))
    except Exception as exc:  # preserve one stable public failure type
        raise N2CorrectedD1RuntimeError("corrected gallery mapping is not valid") from exc
    labels = tuple(identity_map.labels)
    validate_corrected_identity_axis(labels)
    return labels


def validate_corrected_identity_axis(labels: Sequence[str]) -> dict[str, Any]:
    values = tuple(map(str, labels))
    if len(values) != PHYSICAL_ROW_COUNT:
        raise N2CorrectedD1RuntimeError("corrected gallery must have 5,413 physical rows")
    population = len(set(values))
    if population == LEGACY_LABEL_COUNT:
        raise N2CorrectedD1RuntimeError("legacy 5,404-label gallery is forbidden in N2")
    if population != CORRECTED_IDENTITY_COUNT:
        raise N2CorrectedD1RuntimeError("corrected gallery must have 5,412 identities")
    positions: dict[str, list[int]] = defaultdict(list)
    for row, identity in enumerate(values):
        if not identity:
            raise N2CorrectedD1RuntimeError("corrected identity is empty")
        positions[identity].append(row)
    repeated = {identity: rows for identity, rows in positions.items() if len(rows) > 1}
    if repeated != {"Biogen_21": [714, 715]}:
        raise N2CorrectedD1RuntimeError("corrected duplicate component drift")
    return {
        "physical_row_count": len(values),
        "corrected_identity_count": population,
        "duplicate_components": repeated,
        "corrected_axis_sha256": canonical_sha256(list(enumerate(values))),
    }


def reduce_corrected_full_gallery_scores(
    physical_row_scores: torch.Tensor,
    corrected_identities: Sequence[str],
) -> CorrectedIdentityReduction:
    """Reduce 5,413 rows by max, resolving every tie by lower physical row."""

    labels = tuple(map(str, corrected_identities))
    validate_corrected_identity_axis(labels)
    if (
        not isinstance(physical_row_scores, torch.Tensor)
        or physical_row_scores.ndim != 1
        or physical_row_scores.numel() != PHYSICAL_ROW_COUNT
        or not physical_row_scores.is_floating_point()
        or not bool(torch.isfinite(physical_row_scores).all())
    ):
        raise N2CorrectedD1RuntimeError("full-gallery scores must be finite floating [5413]")

    positions: dict[str, list[int]] = defaultdict(list)
    for row, identity in enumerate(labels):
        positions[identity].append(row)
    # Identity slot order is itself target-free and deterministic.
    identity_values = tuple(sorted(positions))
    scores: list[torch.Tensor] = []
    representatives: list[int] = []
    for identity in identity_values:
        rows = positions[identity]
        row_index = torch.tensor(rows, dtype=torch.long, device=physical_row_scores.device)
        values = physical_row_scores.index_select(0, row_index)
        maximum = values.max()
        tied_rows = row_index[values == maximum]
        representative = int(tied_rows.min().item())
        scores.append(maximum)
        representatives.append(representative)
    identity_scores = torch.stack(scores)
    representative_rows = torch.tensor(
        representatives, dtype=torch.long, device=physical_row_scores.device
    )
    ranked = sorted(
        range(len(identity_values)),
        key=lambda slot: (-float(identity_scores[slot].detach().cpu()), representatives[slot]),
    )
    ranked_slots = torch.tensor(ranked, dtype=torch.long, device=physical_row_scores.device)
    return CorrectedIdentityReduction(
        identity_values=identity_values,
        identity_scores=identity_scores,
        representative_physical_rows=representative_rows,
        ranked_identity_slots=ranked_slots,
    )


def natural_c128(reduction: CorrectedIdentityReduction) -> NaturalC128:
    if len(reduction.identity_values) != CORRECTED_IDENTITY_COUNT:
        raise N2CorrectedD1RuntimeError("natural C128 requires the complete corrected gallery")
    selected = reduction.ranked_identity_slots[:NATURAL_CANDIDATE_COUNT]
    identities = tuple(reduction.identity_values[int(slot)] for slot in selected)
    rows = tuple(int(reduction.representative_physical_rows[int(slot)]) for slot in selected)
    scores = reduction.identity_scores.index_select(0, selected).detach().clone()
    if len(identities) != len(set(identities)) or len(rows) != len(set(rows)):
        raise N2CorrectedD1RuntimeError("natural C128 is not identity/row unique")
    if len(identities) != NATURAL_CANDIDATE_COUNT:
        raise N2CorrectedD1RuntimeError("natural C128 population drift")
    return NaturalC128(identities=identities, representative_physical_rows=rows, scores=scores)


def _field(record: object, name: str) -> Any:
    if isinstance(record, Mapping):
        if name not in record:
            raise N2CorrectedD1RuntimeError(f"canonical ledger field absent: {name}")
        return record[name]
    if not hasattr(record, name):
        raise N2CorrectedD1RuntimeError(f"canonical ledger field absent: {name}")
    return getattr(record, name)


def validate_canonical_987_ledger(ledger: Sequence[object]) -> dict[str, Any]:
    if len(ledger) != CANONICAL_QUERY_COUNT:
        raise N2CorrectedD1RuntimeError("canonical query ledger must contain 987 rows")
    query_ids: list[str] = []
    ordinals: list[int] = []
    folds: list[int] = []
    tracks: list[str] = []
    for position, record in enumerate(ledger):
        query_id = str(_field(record, "query_id"))
        ordinal = int(_field(record, "query_ordinal"))
        fold = int(_field(record, "heldout_fold"))
        track = str(_field(record, "track"))
        grid_h = int(_field(record, "grid_h"))
        grid_w = int(_field(record, "grid_w"))
        source_sha = str(_field(record, "source_image_sha256"))
        if (
            not query_id
            or ordinal != position
            or fold not in range(5)
            or track not in CANONICAL_TRACKS
            or grid_h < 1
            or grid_w < 1
            or len(source_sha) != 64
        ):
            raise N2CorrectedD1RuntimeError("canonical query ledger value drift")
        query_ids.append(query_id)
        ordinals.append(ordinal)
        folds.append(fold)
        tracks.append(track)
    if len(set(query_ids)) != CANONICAL_QUERY_COUNT or len(set(ordinals)) != CANONICAL_QUERY_COUNT:
        raise N2CorrectedD1RuntimeError("canonical query ids or ordinals are not unique")
    fold_counts = dict(sorted(Counter(folds).items()))
    if fold_counts != CANONICAL_FOLD_COUNTS:
        raise N2CorrectedD1RuntimeError("canonical heldout-fold population drift")
    return {
        "query_count": len(query_ids),
        "fold_counts": fold_counts,
        "track_counts": dict(sorted(Counter(tracks).items())),
        "query_id_to_fold_sha256": canonical_sha256(
            [[query_ids[index], ordinals[index], folds[index]] for index in range(len(ledger))]
        ),
    }


def join_query_ids_to_canonical_folds(
    query_ids: Sequence[str],
    canonical_ledger: Sequence[object],
) -> tuple[CanonicalFoldAssignment, ...]:
    """Join only opaque query IDs; no execution or Pair64 inner-fold input exists."""

    validate_canonical_987_ledger(canonical_ledger)
    requested = tuple(map(str, query_ids))
    if not requested or len(requested) != len(set(requested)):
        raise N2CorrectedD1RuntimeError("query-id join input must be nonempty and unique")
    by_query_id = {str(_field(record, "query_id")): record for record in canonical_ledger}
    if any(query_id not in by_query_id for query_id in requested):
        raise N2CorrectedD1RuntimeError("query-id join member absent from canonical ledger")
    output = []
    for query_id in requested:
        record = by_query_id[query_id]
        output.append(
            CanonicalFoldAssignment(
                query_id=query_id,
                query_ordinal=int(_field(record, "query_ordinal")),
                heldout_fold=int(_field(record, "heldout_fold")),
                track=str(_field(record, "track")),
                grid_h=int(_field(record, "grid_h")),
                grid_w=int(_field(record, "grid_w")),
                source_image_sha256=str(_field(record, "source_image_sha256")),
            )
        )
    return tuple(output)


def fold_assignment_sha256(assignments: Sequence[CanonicalFoldAssignment]) -> str:
    return canonical_sha256(
        [
            {
                "query_id": item.query_id,
                "query_ordinal": item.query_ordinal,
                "heldout_fold": item.heldout_fold,
                "track": item.track,
                "grid_h": item.grid_h,
                "grid_w": item.grid_w,
                "source_image_sha256": item.source_image_sha256,
            }
            for item in assignments
        ]
    )


__all__ = [
    "CANONICAL_FOLD_COUNTS",
    "CANONICAL_QUERY_COUNT",
    "CanonicalFoldAssignment",
    "CorrectedIdentityReduction",
    "LEGACY_LABEL_COUNT",
    "N2CorrectedD1RuntimeError",
    "NATURAL_CANDIDATE_COUNT",
    "NaturalC128",
    "corrected_labels_from_legacy",
    "fold_assignment_sha256",
    "join_query_ids_to_canonical_folds",
    "natural_c128",
    "reduce_corrected_full_gallery_scores",
    "validate_canonical_987_ledger",
    "validate_corrected_identity_axis",
]
