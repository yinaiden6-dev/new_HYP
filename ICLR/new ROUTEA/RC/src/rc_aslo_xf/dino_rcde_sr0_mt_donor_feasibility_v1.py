"""Metadata-only feasibility audit for future RCDE verification-null donors.

This module deliberately has no geometry-lock input and never emits a donor
pairing.  It answers only the earlier capacity question: given a planned scope,
can records *in principle* be paired to a coherent-reference non-match from the
same track without sharing query, source, identity, or supergroup?

Final donor construction is forbidden until fold-local P locks exist and their
geometry signatures have been independently sealed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import statistics
from typing import Mapping, Sequence


MAX_SCOPES = 16
REQUIRED_FIELDS = ("query_id", "source_sha", "identity", "supergroup", "track")


class DonorFeasibilityError(RuntimeError):
    """The supplied metadata cannot define a deterministic feasibility graph."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise DonorFeasibilityError(message)


@dataclass(frozen=True, order=True)
class ScopeRecord:
    """Private metadata used to build the feasibility graph; never returned."""

    query_id: str
    source_sha: str
    identity: str
    supergroup: str
    track: str

    @property
    def source_key(self) -> tuple[str, str]:
        return (self.query_id, self.source_sha)


@dataclass(frozen=True)
class DonorFeasibilitySummary:
    """Non-sensitive capacity summary.  Raw identity/source values are omitted."""

    schema: str
    scope: str
    record_count: int
    candidate_edge_count: int
    min_degree: int
    median_degree: float
    max_degree: int
    zero_degree_count: int
    ineligible_count: int
    matched_count: int
    matching_deficit_count: int
    hall_deficiency: bool
    identity_count: int
    supergroup_count: int
    track_count: int
    identity_supergroup_track_cell_count: int
    scope_metadata_sha256: str
    geometry_lock_checked: bool
    final_donor_matching_authorized: bool
    donor_pairing_emitted: bool

    def as_dict(self) -> dict[str, object]:
        """Return an explicitly redacted, JSON-ready summary."""

        return asdict(self)


def _coerce_record(raw: ScopeRecord | Mapping[str, object]) -> ScopeRecord:
    if isinstance(raw, ScopeRecord):
        record = raw
    else:
        _require(isinstance(raw, Mapping), "scope record is not a mapping")
        _require(set(raw) == set(REQUIRED_FIELDS), "scope record fields are not exact")
        values = []
        for field in REQUIRED_FIELDS:
            value = raw[field]
            _require(isinstance(value, str) and value, f"scope record {field} is empty")
            values.append(value)
        record = ScopeRecord(*values)
    _require(all(isinstance(getattr(record, field), str) and getattr(record, field) for field in REQUIRED_FIELDS), "ScopeRecord fields are empty")
    return record


def _canonical_records(records: Sequence[ScopeRecord | Mapping[str, object]]) -> tuple[ScopeRecord, ...]:
    normalized = tuple(sorted((_coerce_record(raw) for raw in records)))
    source_keys = [record.source_key for record in normalized]
    _require(len(source_keys) == len(set(source_keys)), "duplicate query/source record")
    return normalized


def _metadata_sha256(records: tuple[ScopeRecord, ...]) -> str:
    # The digest binds all donor-eligibility metadata but reveals no raw label.
    payload = [
        {
            "query_id": record.query_id,
            "source_sha": record.source_sha,
            "identity": record.identity,
            "supergroup": record.supergroup,
            "track": record.track,
        }
        for record in records
    ]
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _eligible(recipient: ScopeRecord, donor: ScopeRecord) -> bool:
    """The fixed cross-query coherent-reference non-match edge contract."""

    return (
        recipient.track == donor.track
        and recipient.query_id != donor.query_id
        and recipient.source_sha != donor.source_sha
        and recipient.identity != donor.identity
        and recipient.supergroup != donor.supergroup
    )


def _maximum_bipartite_cardinality(adjacency: tuple[tuple[int, ...], ...]) -> int:
    """Deterministic Kuhn matching over sorted vertices; returns only cardinality."""

    right_match = [-1] * len(adjacency)

    def augment(left: int, seen: list[bool]) -> bool:
        for right in adjacency[left]:
            if seen[right]:
                continue
            seen[right] = True
            if right_match[right] < 0 or augment(right_match[right], seen):
                right_match[right] = left
                return True
        return False

    matched = 0
    for left in range(len(adjacency)):
        if augment(left, [False] * len(adjacency)):
            matched += 1
    return matched


def analyze_scope(
    scope: str,
    records: Sequence[ScopeRecord | Mapping[str, object]],
) -> DonorFeasibilitySummary:
    """Return a metadata-only donor-capacity audit for one planned scope.

    The bipartite graph has one recipient and one donor copy of every record.
    A maximum matching is computed solely to diagnose Hall/capacity deficiency;
    its edges are intentionally discarded before returning.
    """

    _require(isinstance(scope, str) and scope, "scope is empty")
    rows = _canonical_records(records)
    adjacency = tuple(
        tuple(index for index, donor in enumerate(rows) if _eligible(recipient, donor))
        for recipient in rows
    )
    degrees = tuple(len(edges) for edges in adjacency)
    matched = _maximum_bipartite_cardinality(adjacency)
    record_count = len(rows)
    zero_degree = sum(degree == 0 for degree in degrees)
    return DonorFeasibilitySummary(
        schema="RCDE_SR0_MT_DONOR_METADATA_FEASIBILITY_V1",
        scope=scope,
        record_count=record_count,
        candidate_edge_count=sum(degrees),
        min_degree=min(degrees, default=0),
        median_degree=float(statistics.median(degrees)) if degrees else 0.0,
        max_degree=max(degrees, default=0),
        zero_degree_count=zero_degree,
        ineligible_count=zero_degree,
        matched_count=matched,
        matching_deficit_count=record_count - matched,
        hall_deficiency=matched < record_count,
        identity_count=len({record.identity for record in rows}),
        supergroup_count=len({record.supergroup for record in rows}),
        track_count=len({record.track for record in rows}),
        identity_supergroup_track_cell_count=len(
            {(record.identity, record.supergroup, record.track) for record in rows}
        ),
        scope_metadata_sha256=_metadata_sha256(rows),
        geometry_lock_checked=False,
        final_donor_matching_authorized=False,
        donor_pairing_emitted=False,
    )


def analyze_scopes(
    scopes: Mapping[str, Sequence[ScopeRecord | Mapping[str, object]]],
) -> dict[str, DonorFeasibilitySummary]:
    """Analyze up to the 16 pre-registered planned scopes independently."""

    _require(isinstance(scopes, Mapping) and scopes, "scope population is empty")
    _require(len(scopes) <= MAX_SCOPES, f"at most {MAX_SCOPES} scopes are supported")
    _require(all(isinstance(scope, str) and scope for scope in scopes), "scope key is empty")
    return {scope: analyze_scope(scope, scopes[scope]) for scope in sorted(scopes)}


__all__ = [
    "MAX_SCOPES",
    "REQUIRED_FIELDS",
    "DonorFeasibilityError",
    "ScopeRecord",
    "DonorFeasibilitySummary",
    "analyze_scope",
    "analyze_scopes",
]
