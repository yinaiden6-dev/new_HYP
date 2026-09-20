"""Metadata-only balanced full-reference donor assignment for Track H Q0."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


SCHEMA_VERSION = "rc_dino_rcde_h0_balanced_donor_q0_core_v1_20260821"
DUMMY_COST = 1.0e6
FORBIDDEN_COST = 1.0e9
REUSE_PENALTY = 4.0
CAPACITY_SLACK = 2
EPSILON = 1.0e-12


class H0DonorQ0Error(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise H0DonorQ0Error(message)


def _sha(value: object, *, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} SHA256 drift",
    )
    return str(value)


def _tie(*parts: object) -> float:
    value = hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big") / 2**64 * 1.0e-9


@dataclass(frozen=True)
class DonorNodeV1:
    fit_id: str
    namespace: str
    outer_fold: int
    execution_ordinal: int
    query_id: str
    query_source_sha256: str
    identity: str
    supergroup: str
    track: str
    candidate_key: str
    physical_row: int
    reference_source_sha256: str
    geometry: tuple[float, float, float]
    orientation_class: str
    c128_identities: frozenset[str]

    def __post_init__(self) -> None:
        require(bool(self.fit_id) and bool(self.namespace), "node scope absent")
        require(self.outer_fold in (1, 2, 3, 4), "node outer fold drift")
        require(self.execution_ordinal >= 0 and bool(self.query_id), "node query address drift")
        _sha(self.query_source_sha256, name="query source")
        _sha(self.candidate_key, name="candidate key")
        _sha(self.reference_source_sha256, name="reference source")
        require(self.physical_row >= 0, "node physical row drift")
        require(bool(self.identity) and bool(self.supergroup) and bool(self.track), "node role metadata absent")
        require(
            len(self.geometry) == 3
            and all(math.isfinite(float(value)) for value in self.geometry),
            "node nuisance geometry drift",
        )
        require(self.orientation_class in {"PORTRAIT", "SQUARE", "LANDSCAPE"}, "orientation class drift")

    @property
    def payload_key(self) -> tuple[str, int, str]:
        return self.candidate_key, self.physical_row, self.reference_source_sha256


@dataclass(frozen=True)
class DonorPayloadV1:
    payload_key: tuple[str, int, str]
    identity: str
    supergroup: str
    track: str
    geometry: tuple[float, float, float]
    orientation_class: str
    providers: tuple[DonorNodeV1, ...]


def robust_scales(nodes: Sequence[DonorNodeV1]) -> tuple[float, float, float]:
    require(bool(nodes), "scale population empty")
    values = np.asarray([node.geometry for node in nodes], dtype=np.float64)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median[None, :]), axis=0)
    scales = np.where(mad > EPSILON, mad, 1.0)
    return tuple(float(value) for value in scales)


def payloads_from_nodes(nodes: Sequence[DonorNodeV1]) -> tuple[DonorPayloadV1, ...]:
    grouped: dict[tuple[str, int, str], list[DonorNodeV1]] = {}
    for node in nodes:
        grouped.setdefault(node.payload_key, []).append(node)
    output = []
    for key in sorted(grouped):
        providers = tuple(
            sorted(grouped[key], key=lambda item: (item.query_source_sha256, item.execution_ordinal))
        )
        first = providers[0]
        require(
            all(
                item.identity == first.identity
                and item.supergroup == first.supergroup
                and item.track == first.track
                and item.geometry == first.geometry
                and item.orientation_class == first.orientation_class
                for item in providers
            ),
            "one payload has inconsistent role/geometry metadata",
        )
        output.append(
            DonorPayloadV1(
                payload_key=key,
                identity=first.identity,
                supergroup=first.supergroup,
                track=first.track,
                geometry=first.geometry,
                orientation_class=first.orientation_class,
                providers=providers,
            )
        )
    return tuple(output)


def eligible_provider(
    recipient: DonorNodeV1, payload: DonorPayloadV1
) -> DonorNodeV1 | None:
    if (
        payload.payload_key == recipient.payload_key
        or payload.identity == recipient.identity
        or payload.supergroup == recipient.supergroup
        or payload.track != recipient.track
        or payload.orientation_class != recipient.orientation_class
        or payload.identity in recipient.c128_identities
    ):
        return None
    providers = [
        item
        for item in payload.providers
        if item.fit_id == recipient.fit_id
        and item.namespace == recipient.namespace
        and item.execution_ordinal != recipient.execution_ordinal
        and item.query_id != recipient.query_id
        and item.query_source_sha256 != recipient.query_source_sha256
    ]
    if not providers:
        return None
    return min(
        providers,
        key=lambda item: hashlib.sha256(
            f"{recipient.execution_ordinal}\0{item.query_source_sha256}".encode("utf-8")
        ).hexdigest(),
    )


def geometry_cost(
    recipient: DonorNodeV1,
    payload: DonorPayloadV1,
    scales: tuple[float, float, float],
) -> float:
    return float(
        sum(
            abs(float(left) - float(right)) / float(scale)
            for left, right, scale in zip(
                recipient.geometry, payload.geometry, scales, strict=True
            )
        )
    )


def _assignment(
    recipients: Sequence[DonorNodeV1],
    payloads: Sequence[DonorPayloadV1],
    *,
    scales: tuple[float, float, float],
    capacity: int,
    reuse_penalty: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered_recipients = tuple(sorted(recipients, key=lambda item: item.execution_ordinal))
    ordered_payloads = tuple(sorted(payloads, key=lambda item: item.payload_key))
    slots: list[tuple[int, int]] = [
        (payload_index, slot_index)
        for payload_index in range(len(ordered_payloads))
        for slot_index in range(capacity)
    ]
    columns = len(slots) + len(ordered_recipients)
    costs = np.full((len(ordered_recipients), columns), FORBIDDEN_COST, dtype=np.float64)
    providers: dict[tuple[int, int], DonorNodeV1] = {}
    eligible_counts = []
    for row, recipient in enumerate(ordered_recipients):
        eligible_payloads = 0
        for slot_column, (payload_index, slot_index) in enumerate(slots):
            payload = ordered_payloads[payload_index]
            provider = eligible_provider(recipient, payload)
            if provider is None:
                continue
            if slot_index == 0:
                eligible_payloads += 1
            providers[(row, slot_column)] = provider
            costs[row, slot_column] = (
                geometry_cost(recipient, payload, scales)
                + reuse_penalty * slot_index
                + _tie(recipient.execution_ordinal, payload.payload_key, slot_index)
            )
        eligible_counts.append(eligible_payloads)
        costs[row, len(slots) + row] = DUMMY_COST + _tie("dummy", recipient.execution_ordinal)
    finite_valid = costs[:, : len(slots)][costs[:, : len(slots)] < DUMMY_COST]
    if finite_valid.size:
        require(
            float(finite_valid.max()) * max(1, len(ordered_recipients)) < DUMMY_COST,
            "valid assignment costs no longer guarantee cardinality-first optimization",
        )
    row_index, column_index = linear_sum_assignment(costs)
    require(len(row_index) == len(ordered_recipients), "linear assignment row coverage drift")
    result = []
    reuse: dict[tuple[str, int, str], int] = {}
    for row, column in zip(row_index.tolist(), column_index.tolist(), strict=True):
        recipient = ordered_recipients[row]
        if column >= len(slots) or costs[row, column] >= DUMMY_COST:
            result.append(
                {
                    "recipient_execution_ordinal": recipient.execution_ordinal,
                    "recipient_query_id": recipient.query_id,
                    "status": "UNMATCHED",
                    "eligible_payload_count": eligible_counts[row],
                    "donor": None,
                }
            )
            continue
        payload_index, slot_index = slots[column]
        payload = ordered_payloads[payload_index]
        provider = providers[(row, column)]
        reuse[payload.payload_key] = reuse.get(payload.payload_key, 0) + 1
        result.append(
            {
                "recipient_execution_ordinal": recipient.execution_ordinal,
                "recipient_query_id": recipient.query_id,
                "status": "MATCHED",
                "eligible_payload_count": eligible_counts[row],
                "geometry_cost": geometry_cost(recipient, payload, scales),
                "reuse_slot": slot_index,
                "donor": {
                    "payload_key": list(payload.payload_key),
                    "identity_sha256": hashlib.sha256(payload.identity.encode()).hexdigest(),
                    "supergroup_sha256": hashlib.sha256(payload.supergroup.encode()).hexdigest(),
                    "provider_execution_ordinal": provider.execution_ordinal,
                    "provider_query_id": provider.query_id,
                    "provider_query_source_sha256": provider.query_source_sha256,
                    "geometry": list(payload.geometry),
                },
            }
        )
    matched = sum(item["status"] == "MATCHED" for item in result)
    zero_degree = sum(count == 0 for count in eligible_counts)
    summary = {
        "recipient_count": len(ordered_recipients),
        "payload_count": len(ordered_payloads),
        "matched_count": matched,
        "coverage": matched / len(ordered_recipients) if ordered_recipients else 0.0,
        "zero_degree_count": zero_degree,
        "zero_degree_rate": zero_degree / len(ordered_recipients) if ordered_recipients else 0.0,
        "distinct_payloads_used": len(reuse),
        "reuse": {"|".join(map(str, key)): value for key, value in sorted(reuse.items())},
        "maximum_reuse": max(reuse.values(), default=0),
        "maximum_payload_share": max(reuse.values(), default=0) / matched if matched else 0.0,
    }
    return result, summary


def assign_scope(
    nodes: Sequence[DonorNodeV1],
    *,
    scales: tuple[float, float, float],
) -> dict[str, Any]:
    require(bool(nodes), "scope node population empty")
    fit_ids = {node.fit_id for node in nodes}
    namespaces = {node.namespace for node in nodes}
    require(len(fit_ids) == len(namespaces) == 1, "scope mixes fit/namespace")
    strict_rows = []
    main_rows = []
    strict_summary = {"recipient_count": 0, "matched_count": 0}
    main_summary = {"recipient_count": 0, "matched_count": 0}
    block_summaries = []
    for track in sorted({node.track for node in nodes}):
        recipients = [node for node in nodes if node.track == track]
        payloads = payloads_from_nodes(recipients)
        strict, strict_block = _assignment(
            recipients,
            payloads,
            scales=scales,
            capacity=1,
            reuse_penalty=0.0,
        )
        capacity = (math.ceil(len(recipients) / len(payloads)) + CAPACITY_SLACK) if payloads else 1
        main, main_block = _assignment(
            recipients,
            payloads,
            scales=scales,
            capacity=capacity,
            reuse_penalty=REUSE_PENALTY,
        )
        for row in strict:
            row["track"] = track
        for row in main:
            row["track"] = track
            row["registered_capacity"] = capacity
        strict_rows.extend(strict)
        main_rows.extend(main)
        strict_summary["recipient_count"] += strict_block["recipient_count"]
        strict_summary["matched_count"] += strict_block["matched_count"]
        main_summary["recipient_count"] += main_block["recipient_count"]
        main_summary["matched_count"] += main_block["matched_count"]
        block_summaries.append(
            {
                "track": track,
                "registered_capacity": capacity,
                "strict": strict_block,
                "balanced": main_block,
            }
        )
    recipient_count = int(main_summary["recipient_count"])
    matched_count = int(main_summary["matched_count"])
    zero_degree = sum(item["eligible_payload_count"] == 0 for item in main_rows)
    reuse: dict[str, int] = {}
    for item in main_rows:
        if item["status"] == "MATCHED":
            key = "|".join(map(str, item["donor"]["payload_key"]))
            reuse[key] = reuse.get(key, 0) + 1
    main_summary.update(
        {
            "coverage": matched_count / recipient_count,
            "zero_degree_count": zero_degree,
            "zero_degree_rate": zero_degree / recipient_count,
            "distinct_payloads_used": len(reuse),
            "maximum_reuse": max(reuse.values(), default=0),
            "maximum_payload_share": max(reuse.values(), default=0) / matched_count if matched_count else 0.0,
        }
    )
    strict_summary["coverage"] = strict_summary["matched_count"] / strict_summary["recipient_count"]
    return {
        "fit_id": next(iter(fit_ids)),
        "namespace": next(iter(namespaces)),
        "strict_rows": strict_rows,
        "balanced_rows": main_rows,
        "strict_summary": strict_summary,
        "balanced_summary": main_summary,
        "track_blocks": block_summaries,
    }


__all__ = [
    "SCHEMA_VERSION",
    "DUMMY_COST",
    "FORBIDDEN_COST",
    "REUSE_PENALTY",
    "CAPACITY_SLACK",
    "H0DonorQ0Error",
    "DonorNodeV1",
    "DonorPayloadV1",
    "robust_scales",
    "payloads_from_nodes",
    "eligible_provider",
    "geometry_cost",
    "assign_scope",
]
