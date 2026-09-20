"""Target-free full-C128 planning with pair-only C_DINO materialization.

The historical Track-R scorer constructed only the two anonymous pair locks
before calling the otherwise full-axis C_DINO helper.  That reduced the
control to a deterministic two-way swap.  This append-only module separates
the complete natural-C128 donor plan from runtime materialization:

* all 128 destination addresses participate in one identity-disjoint perfect
  matching on CPU metadata;
* destination P-lock record hashes remain attached to their destinations;
* only the two anonymous pair destinations and their two donor DINO payloads
  are materialized;
* the device-transfer boundary rejects anything other than that two-member
  extraction.

No target/rival label, score, correctness flag, model, or scientific result is
accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from .dino_rcde_cw1_multitile_vdecode_v1 import (
    CandidateReferenceFieldV1,
    FIXED_DIRECTIONS,
    PHYSICAL_GALLERY_ROWS,
)
from .dino_rcde_sr0_mt_controls_v1 import CDinoCandidateBindingV1
from .dino_rcde_sr0_mt_v_runtime_v1 import (
    CandidatePLockV1,
    SR0MTVContractError,
    hash_parts,
    tensor_sha256,
)


SCHEMA_VERSION = "rc_dino_rcde_track_r_full_c128_c_dino_pair_v1_20260824"
CONTROL_NAME = "C_DINO_V"
CANDIDATE_COUNT = 128
PAIR_MEMBER_COUNT = 2
FORBIDDEN_RESULT_FIELDS = frozenset(
    {
        "target",
        "target_key",
        "target_label",
        "rival",
        "rival_key",
        "strongest_rival",
        "correctness",
        "winner",
        "raw_score",
        "raw_margin",
        "d1_score",
        "d1_rank",
        "scientific_result",
    }
)


class FullC128CDinoContractError(SR0MTVContractError):
    """The full-C128 pair-only C_DINO contract was violated."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise FullC128CDinoContractError(message)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: object, *, name: str) -> str:
    _require(_is_sha256(value), f"{name} SHA256 drift")
    return str(value)


def corrected_identity_sha256(value: str) -> str:
    _require(isinstance(value, str) and bool(value), "corrected identity is absent")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reject_result_fields(value: object) -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_RESULT_FIELDS.intersection(map(str, value.keys()))
        _require(not forbidden, f"target/result field leaked into C128 control: {sorted(forbidden)}")
        for item in value.values():
            _reject_result_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_result_fields(item)


@dataclass(frozen=True)
class FullC128CandidateAxisRecordV1:
    """One score-erased candidate address used by the donor matcher."""

    candidate_position: int
    candidate_key: str
    candidate_physical_row: int
    candidate_reference_source_sha256: str
    corrected_identity_sha256: str
    p_lock_record_sha256_by_direction: Mapping[str, str]
    candidate_native_content_sha256_by_direction: Mapping[str, str]
    dino_content_binding_sha256: str

    def __post_init__(self) -> None:
        _require(
            type(self.candidate_position) is int
            and 0 <= self.candidate_position < CANDIDATE_COUNT,
            "C128 candidate position drift",
        )
        _require(
            isinstance(self.candidate_key, str) and bool(self.candidate_key),
            "C128 candidate key is absent",
        )
        _require(
            type(self.candidate_physical_row) is int
            and 0 <= self.candidate_physical_row < PHYSICAL_GALLERY_ROWS,
            "C128 physical row drift",
        )
        _sha256(
            self.candidate_reference_source_sha256,
            name="C128 candidate reference source",
        )
        _sha256(self.corrected_identity_sha256, name="C128 corrected identity")
        p_locks = dict(self.p_lock_record_sha256_by_direction)
        native = dict(self.candidate_native_content_sha256_by_direction)
        _require(set(p_locks) == set(FIXED_DIRECTIONS), "C128 P-lock direction axis drift")
        _require(
            set(native) == set(FIXED_DIRECTIONS),
            "C128 native-content direction axis drift",
        )
        for direction in FIXED_DIRECTIONS:
            _sha256(p_locks[direction], name=f"C128 {direction} P lock")
            _sha256(native[direction], name=f"C128 {direction} native content")
        _sha256(self.dino_content_binding_sha256, name="C128 DINO content binding")
        object.__setattr__(
            self,
            "p_lock_record_sha256_by_direction",
            MappingProxyType(p_locks),
        )
        object.__setattr__(
            self,
            "candidate_native_content_sha256_by_direction",
            MappingProxyType(native),
        )

    def logical_record(self, *, axis_index: int | None = None) -> dict[str, object]:
        value: dict[str, object] = {
            "candidate_position": self.candidate_position,
            "candidate_key": self.candidate_key,
            "candidate_physical_row": self.candidate_physical_row,
            "candidate_reference_source_sha256": self.candidate_reference_source_sha256,
            "corrected_identity_sha256": self.corrected_identity_sha256,
            "p_lock_record_sha256_by_direction": {
                direction: self.p_lock_record_sha256_by_direction[direction]
                for direction in FIXED_DIRECTIONS
            },
            "candidate_native_content_sha256_by_direction": {
                direction: self.candidate_native_content_sha256_by_direction[direction]
                for direction in FIXED_DIRECTIONS
            },
            "dino_content_binding_sha256": self.dino_content_binding_sha256,
        }
        if axis_index is not None:
            value["axis_index"] = axis_index
        return value


@dataclass(frozen=True)
class FullC128ControlAxisV1:
    """Complete natural C128 metadata axis for one target-free query."""

    query_id: str
    candidate_axis_sha256: str
    records: tuple[FullC128CandidateAxisRecordV1, ...]
    target_free: bool = True

    def __post_init__(self) -> None:
        _require(isinstance(self.query_id, str) and bool(self.query_id), "query ID is absent")
        _sha256(self.candidate_axis_sha256, name="natural candidate axis")
        records = tuple(self.records)
        _require(len(records) == CANDIDATE_COUNT, "C_DINO axis is not complete C128")
        by_position = {item.candidate_position: item for item in records}
        _require(
            set(by_position) == set(range(CANDIDATE_COUNT)),
            "C_DINO candidate positions are not 0..127",
        )
        position_order = tuple(by_position[index] for index in range(CANDIDATE_COUNT))
        rows = [item.candidate_physical_row for item in position_order]
        _require(rows == sorted(rows) and len(set(rows)) == CANDIDATE_COUNT, "C128 physical-row axis drift")
        _require(
            len({item.candidate_key for item in records}) == CANDIDATE_COUNT,
            "C128 candidate-key collision",
        )
        _require(
            self.candidate_axis_sha256 == _canonical_sha256(rows),
            "natural C128 candidate-axis hash drift",
        )
        _require(self.target_free is True, "C128 control axis is not target-free")
        object.__setattr__(self, "records", position_order)

    def canonical_matching_axis(self) -> tuple[FullC128CandidateAxisRecordV1, ...]:
        return tuple(
            sorted(
                self.records,
                key=lambda item: (
                    item.candidate_physical_row,
                    item.candidate_reference_source_sha256,
                    item.candidate_key,
                ),
            )
        )


@dataclass(frozen=True)
class FullC128CDinoPairPlanV1:
    """A complete C128 derangement plus two anonymous extraction addresses."""

    axis: FullC128ControlAxisV1
    namespace: str
    pair_member_keys: tuple[str, str]
    destination_to_source: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(isinstance(self.namespace, str) and bool(self.namespace), "control namespace is absent")
        pair = tuple(self.pair_member_keys)
        _require(
            len(pair) == PAIR_MEMBER_COUNT and pair[0] != pair[1],
            "C_DINO pair must contain two distinct anonymous members",
        )
        canonical = self.axis.canonical_matching_axis()
        by_key = {item.candidate_key: item for item in canonical}
        _require(set(pair).issubset(by_key), "C_DINO pair is outside natural C128")
        order = tuple(int(item) for item in self.destination_to_source)
        _require(
            len(order) == CANDIDATE_COUNT
            and sorted(order) == list(range(CANDIDATE_COUNT)),
            "C_DINO donor mapping is not a C128 permutation",
        )
        _require(
            all(index != source for index, source in enumerate(order)),
            "C_DINO donor mapping contains a fixed point",
        )
        _require(
            all(
                canonical[index].corrected_identity_sha256
                != canonical[source].corrected_identity_sha256
                for index, source in enumerate(order)
            ),
            "C_DINO donor mapping is not corrected-identity-disjoint",
        )
        index_by_key = {
            item.candidate_key: index for index, item in enumerate(canonical)
        }
        pair_donors = {
            canonical[order[index_by_key[key]]].candidate_key for key in pair
        }
        _require(
            pair_donors.isdisjoint(pair),
            "scored pair received a donor from inside the scored pair",
        )
        object.__setattr__(self, "pair_member_keys", pair)
        object.__setattr__(self, "destination_to_source", order)

    @property
    def donor_candidate_keys(self) -> tuple[str, str]:
        canonical = self.axis.canonical_matching_axis()
        index_by_key = {item.candidate_key: index for index, item in enumerate(canonical)}
        return tuple(
            canonical[self.destination_to_source[index_by_key[key]]].candidate_key
            for key in self.pair_member_keys
        )  # type: ignore[return-value]


def full_c128_axis_from_selected_head_records(
    records: Sequence[Mapping[str, Any]],
    *,
    corrected_identity_by_physical_row: Mapping[int, str],
    dino_content_binding_sha256_by_physical_row: Mapping[int, str],
    expected_candidate_axis_sha256: str | None = None,
) -> FullC128ControlAxisV1:
    """Extract a lightweight full-C128 axis from a selected target-free head.

    ``records`` must already be the one outer-refit head: 128 candidates times
    the two fixed directions.  No DINO tensor is opened here.
    """

    selected = tuple(records)
    _require(
        len(selected) == CANDIDATE_COUNT * len(FIXED_DIRECTIONS),
        "selected head is not 128 candidates x two directions",
    )
    for value in selected:
        _require(isinstance(value, Mapping), "selected head record is not a mapping")
        _reject_result_fields(value)
    query_ids = {
        str(value.get("query", {}).get("query_id"))
        for value in selected
        if isinstance(value.get("query"), Mapping)
    }
    _require(len(query_ids) == 1 and "None" not in query_ids, "selected head mixes query IDs")
    query_id = next(iter(query_ids))
    buckets: dict[int, dict[str, Mapping[str, Any]]] = {}
    addresses: dict[int, tuple[str, int, str]] = {}
    for value in selected:
        query = value.get("query")
        candidate = value.get("candidate")
        _require(isinstance(query, Mapping) and isinstance(candidate, Mapping), "selected head address is absent")
        _require(query.get("query_id") == query_id, "selected head query address drift")
        position = candidate.get("candidate_position")
        row = candidate.get("candidate_physical_row")
        key = candidate.get("candidate_key")
        source = candidate.get("candidate_reference_source_sha256")
        direction = value.get("direction")
        _require(
            type(position) is int and 0 <= position < CANDIDATE_COUNT,
            "selected head candidate position drift",
        )
        _require(
            type(row) is int and 0 <= row < PHYSICAL_GALLERY_ROWS,
            "selected head physical row drift",
        )
        _require(isinstance(key, str) and bool(key), "selected head candidate key is absent")
        _sha256(source, name="selected head reference source")
        _require(direction in FIXED_DIRECTIONS, "selected head direction drift")
        address = (key, row, str(source))
        previous = addresses.setdefault(position, address)
        _require(previous == address, "selected head candidate binding differs across directions")
        bucket = buckets.setdefault(position, {})
        _require(direction not in bucket, "duplicate selected candidate direction")
        bucket[str(direction)] = value
    _require(set(buckets) == set(range(CANDIDATE_COUNT)), "selected head candidate population drift")
    output: list[FullC128CandidateAxisRecordV1] = []
    for position in range(CANDIDATE_COUNT):
        bucket = buckets[position]
        _require(set(bucket) == set(FIXED_DIRECTIONS), "selected head direction population drift")
        key, row, source = addresses[position]
        identity = corrected_identity_by_physical_row.get(row)
        _require(isinstance(identity, str) and bool(identity), "corrected identity is absent from C128 axis")
        dino_binding = dino_content_binding_sha256_by_physical_row.get(row)
        _sha256(dino_binding, name="selected-head DINO content binding")
        output.append(
            FullC128CandidateAxisRecordV1(
                candidate_position=position,
                candidate_key=key,
                candidate_physical_row=row,
                candidate_reference_source_sha256=source,
                corrected_identity_sha256=corrected_identity_sha256(identity),
                p_lock_record_sha256_by_direction={
                    direction: _sha256(
                        bucket[direction].get("record_sha256"),
                        name=f"selected {direction} Natural P lock",
                    )
                    for direction in FIXED_DIRECTIONS
                },
                candidate_native_content_sha256_by_direction={
                    direction: _sha256(
                        bucket[direction]["candidate"].get(
                            "candidate_native_content_sha256"
                        ),
                        name=f"selected {direction} native content",
                    )
                    for direction in FIXED_DIRECTIONS
                },
                dino_content_binding_sha256=str(dino_binding),
            )
        )
    candidate_axis_sha256 = _canonical_sha256(
        [item.candidate_physical_row for item in output]
    )
    if expected_candidate_axis_sha256 is not None:
        _require(
            candidate_axis_sha256 == expected_candidate_axis_sha256,
            "selected head/role-free pair candidate-axis hash drift",
        )
    return FullC128ControlAxisV1(
        query_id=query_id,
        candidate_axis_sha256=candidate_axis_sha256,
        records=tuple(output),
    )


def full_c128_axis_from_runtime_locks(
    locks: Mapping[str, CandidatePLockV1],
    *,
    query_id: str,
    corrected_identity_by_physical_row: Mapping[int, str],
) -> FullC128ControlAxisV1:
    """Convenience adapter for callers that already materialized all locks."""

    _require(len(locks) == CANDIDATE_COUNT, "runtime lock mapping is not C128")
    ordered = tuple(
        sorted(
            locks.values(),
            key=lambda lock: (
                lock.candidate.physical_gallery_row,
                lock.candidate.source_image_sha256,
                lock.candidate.candidate_key,
            ),
        )
    )
    rows = [lock.candidate.physical_gallery_row for lock in ordered]
    records: list[FullC128CandidateAxisRecordV1] = []
    for position, lock in enumerate(ordered):
        candidate = lock.candidate
        identity = corrected_identity_by_physical_row.get(candidate.physical_gallery_row)
        _require(isinstance(identity, str) and bool(identity), "runtime C128 corrected identity is absent")
        records.append(
            FullC128CandidateAxisRecordV1(
                candidate_position=position,
                candidate_key=candidate.candidate_key,
                candidate_physical_row=candidate.physical_gallery_row,
                candidate_reference_source_sha256=candidate.source_image_sha256,
                corrected_identity_sha256=corrected_identity_sha256(identity),
                p_lock_record_sha256_by_direction={
                    direction: lock.p_lock_record_sha256_by_direction[direction]
                    for direction in FIXED_DIRECTIONS
                },
                candidate_native_content_sha256_by_direction={
                    direction: candidate.tokens_sha256 for direction in FIXED_DIRECTIONS
                },
                dino_content_binding_sha256=_canonical_sha256(
                    _candidate_content_record(candidate)
                ),
            )
        )
    return FullC128ControlAxisV1(
        query_id=query_id,
        candidate_axis_sha256=_canonical_sha256(rows),
        records=tuple(records),
    )


def _deterministic_identity_disjoint_matching(
    axis: Sequence[FullC128CandidateAxisRecordV1],
    *,
    namespace: str,
    query_id: str,
    protected_pair_keys: Sequence[str],
) -> tuple[int, ...]:
    keys = tuple(item.candidate_key for item in axis)
    identity_by_key = {
        item.candidate_key: item.corrected_identity_sha256 for item in axis
    }
    protected = frozenset(protected_pair_keys)
    _require(
        len(protected) == PAIR_MEMBER_COUNT and protected.issubset(keys),
        "protected pair axis drift",
    )
    donor_for: dict[str, str] = {}
    destination_for_donor: dict[str, str] = {}

    def assign(destination: str, seen: set[str]) -> bool:
        donors = sorted(
            (
                donor
                for donor in keys
                if donor != destination
                and identity_by_key[donor] != identity_by_key[destination]
                and not (destination in protected and donor in protected)
            ),
            key=lambda donor: hash_parts(
                namespace,
                query_id,
                "FULL_C128_DONOR",
                destination,
                donor,
            ),
        )
        for donor in donors:
            if donor in seen:
                continue
            seen.add(donor)
            prior = destination_for_donor.get(donor)
            if prior is None or assign(prior, seen):
                destination_for_donor[donor] = destination
                donor_for[destination] = donor
                return True
        return False

    destination_order = sorted(
        keys,
        key=lambda key: hash_parts(namespace, query_id, "FULL_C128_DEST", key),
    )
    _require(
        all(assign(destination, set()) for destination in destination_order),
        "C_DINO full-C128 axis has no identity-disjoint perfect matching",
    )
    index_by_key = {key: index for index, key in enumerate(keys)}
    order = tuple(index_by_key[donor_for[key]] for key in keys)
    _require(
        sorted(order) == list(range(CANDIDATE_COUNT))
        and all(index != source for index, source in enumerate(order)),
        "C_DINO full-C128 matching is not a derangement",
    )
    return order


def plan_full_c128_c_dino_pair(
    axis: FullC128ControlAxisV1,
    left_candidate_key: str,
    right_candidate_key: str,
    *,
    namespace: str,
) -> FullC128CDinoPairPlanV1:
    """Plan the complete C128 control before opening pair donor tensors."""

    canonical = axis.canonical_matching_axis()
    order = _deterministic_identity_disjoint_matching(
        canonical,
        namespace=namespace,
        query_id=axis.query_id,
        protected_pair_keys=(left_candidate_key, right_candidate_key),
    )
    return FullC128CDinoPairPlanV1(
        axis=axis,
        namespace=namespace,
        pair_member_keys=(left_candidate_key, right_candidate_key),
        destination_to_source=order,
    )


def _candidate_content_record(candidate: CandidateReferenceFieldV1) -> dict[str, object]:
    return {
        "source_image_sha256": candidate.source_image_sha256,
        "source_key": candidate.source_key,
        "cache_payload_sha256": candidate.cache_payload_sha256,
        "geometry_record_sha256": candidate.geometry_record_sha256,
        "tokens_sha256": candidate.tokens_sha256,
        "grid_shape": list(candidate.grid_shape),
        "valid_patch_mask_sha256": tensor_sha256(
            candidate.valid_patch_mask.to(torch.uint8)
        ),
    }


def dino_content_binding_sha256_from_cache_index_record(
    record: Mapping[str, Any],
) -> str:
    """Bind one target-free DINO cache-index content record without loading tokens."""

    _reject_result_fields(record)
    fields = (
        "kind",
        "cache_key",
        "source_image_sha256",
        "tokens_fp16_payload_sha256",
        "cache_logical_sha256",
        "valid_patch_mask_sha256",
        "geometry_logical_sha256",
    )
    _require(set(record) == set(fields), "DINO cache-index content field drift")
    _require(record.get("kind") == "reference", "DINO content record is not a reference")
    _require(
        type(record.get("cache_key")) is int
        and 0 <= int(record["cache_key"]) < PHYSICAL_GALLERY_ROWS,
        "DINO content cache key drift",
    )
    for name in fields[2:]:
        _sha256(record.get(name), name=f"DINO cache-index {name}")
    return _canonical_sha256({name: record[name] for name in fields})


def materialize_full_c128_c_dino_pair(
    plan: FullC128CDinoPairPlanV1,
    *,
    destination_pair_locks: Mapping[str, CandidatePLockV1],
    donor_candidates: Mapping[str, CandidateReferenceFieldV1],
    donor_dino_content_binding_sha256_by_candidate_key: Mapping[str, str],
) -> tuple[Mapping[str, CDinoCandidateBindingV1], Mapping[str, object]]:
    """Materialize only the two pair bindings selected by a full-C128 plan."""

    pair = plan.pair_member_keys
    _require(
        set(destination_pair_locks) == set(pair)
        and len(destination_pair_locks) == PAIR_MEMBER_COUNT,
        "destination runtime locks are not exactly the anonymous pair",
    )
    expected_donors = plan.donor_candidate_keys
    _require(
        set(donor_candidates) == set(expected_donors)
        and len(donor_candidates) == PAIR_MEMBER_COUNT,
        "donor DINO payloads are not exactly the planned pair donors",
    )
    _require(
        set(donor_dino_content_binding_sha256_by_candidate_key)
        == set(expected_donors),
        "donor DINO content-binding receipts are not exactly the planned donors",
    )
    canonical = plan.axis.canonical_matching_axis()
    index_by_key = {
        item.candidate_key: index for index, item in enumerate(canonical)
    }
    by_key = {item.candidate_key: item for item in canonical}
    bindings: dict[str, CDinoCandidateBindingV1] = {}
    pair_receipts: list[dict[str, object]] = []
    for destination_key, donor_key in zip(pair, expected_donors, strict=True):
        destination_axis = by_key[destination_key]
        donor_axis = by_key[donor_key]
        destination_lock = destination_pair_locks[destination_key]
        destination = destination_lock.candidate
        donor = donor_candidates[donor_key]
        _require(
            destination.candidate_key == destination_axis.candidate_key
            and destination.physical_gallery_row
            == destination_axis.candidate_physical_row
            and destination.source_image_sha256
            == destination_axis.candidate_reference_source_sha256,
            "destination pair lock/full-C128 address drift",
        )
        _require(
            all(
                destination_lock.p_lock_record_sha256_by_direction[direction]
                == destination_axis.p_lock_record_sha256_by_direction[direction]
                for direction in FIXED_DIRECTIONS
            ),
            "destination pair P-lock/full-C128 receipt drift",
        )
        _require(
            donor.candidate_key == donor_axis.candidate_key
            and donor.physical_gallery_row == donor_axis.candidate_physical_row
            and donor.source_image_sha256
            == donor_axis.candidate_reference_source_sha256,
            "donor DINO payload/full-C128 address drift",
        )
        donor_binding_sha = donor_dino_content_binding_sha256_by_candidate_key[
            donor_key
        ]
        _require(
            donor_binding_sha == donor_axis.dino_content_binding_sha256,
            "donor DINO content/full-C128 binding drift",
        )
        controlled_candidate = CandidateReferenceFieldV1(
            candidate_key=destination.candidate_key,
            physical_gallery_row=destination.physical_gallery_row,
            layers=donor.layers.clone(),
            grid_shape=donor.grid_shape,
            valid_patch_mask=donor.valid_patch_mask.clone(),
            source_image_sha256=donor.source_image_sha256,
            source_key=donor.source_key,
            cache_payload_sha256=donor.cache_payload_sha256,
            geometry_record_sha256=donor.geometry_record_sha256,
            tokens_sha256=donor.tokens_sha256,
        )
        bindings[destination_key] = CDinoCandidateBindingV1(
            candidate=controlled_candidate,
            destination_p_lock=destination_lock,
            donor_candidate_key=donor_key,
            donor_physical_gallery_row=donor.physical_gallery_row,
            donor_corrected_identity_sha256=donor_axis.corrected_identity_sha256,
            destination_corrected_identity_sha256=destination_axis.corrected_identity_sha256,
        )
        pair_receipts.append(
            {
                "destination_axis_index": index_by_key[destination_key],
                "destination_candidate_key": destination_key,
                "destination_physical_gallery_row": destination.physical_gallery_row,
                "destination_corrected_identity_sha256": destination_axis.corrected_identity_sha256,
                "donor_axis_index": index_by_key[donor_key],
                "donor_candidate_key": donor_key,
                "donor_physical_gallery_row": donor.physical_gallery_row,
                "donor_corrected_identity_sha256": donor_axis.corrected_identity_sha256,
                "donor_dino_content_binding_sha256": donor_binding_sha,
                "destination_p_lock_input_sha256_by_direction": {
                    direction: destination_axis.p_lock_record_sha256_by_direction[
                        direction
                    ]
                    for direction in FIXED_DIRECTIONS
                },
                "destination_p_lock_output_sha256_by_direction": {
                    direction: bindings[
                        destination_key
                    ].p_lock_record_sha256_by_direction[direction]
                    for direction in FIXED_DIRECTIONS
                },
                "donor_content": _candidate_content_record(donor),
                "controlled_content": _candidate_content_record(
                    controlled_candidate
                ),
            }
        )
    axis_records = [
        item.logical_record(axis_index=index) for index, item in enumerate(canonical)
    ]
    input_p_locks = [
        item["p_lock_record_sha256_by_direction"] for item in axis_records
    ]
    native_axis = [
        {
            "candidate_key": item["candidate_key"],
            "candidate_physical_row": item["candidate_physical_row"],
            "candidate_reference_source_sha256": item[
                "candidate_reference_source_sha256"
            ],
            "candidate_native_content_sha256_by_direction": item[
                "candidate_native_content_sha256_by_direction"
            ],
            "dino_content_binding_sha256": item["dino_content_binding_sha256"],
        }
        for item in axis_records
    ]
    output_native_axis = [
        native_axis[source] for source in plan.destination_to_source
    ]
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "control_name": CONTROL_NAME,
        "namespace": plan.namespace,
        "query_id": plan.axis.query_id,
        "target_free": True,
        "label_target_rival_read_count": 0,
        "scientific_reduction_count": 0,
        "candidate_count": CANDIDATE_COUNT,
        "physical_gallery_row_count": PHYSICAL_GALLERY_ROWS,
        "pair_member_count": PAIR_MEMBER_COUNT,
        "pair_member_keys": list(pair),
        "candidate_axis_sha256": plan.axis.candidate_axis_sha256,
        "axis_records": axis_records,
        "destination_to_source": list(plan.destination_to_source),
        "pair_bindings": pair_receipts,
        "changed_variable": "DINO_REFERENCE_NATIVE_CONTENT_BINDING_ACROSS_FULL_C128",
        "fixed_point_free": True,
        "corrected_identity_disjoint": True,
        "pair_member_donor_exclusion": True,
        "candidate_axis_preserved": True,
        "p_lock_preserved": True,
        "full_native_content_binding_permutation": True,
        "input_p_lock_population_sha256": _canonical_sha256(input_p_locks),
        "output_p_lock_population_sha256": _canonical_sha256(input_p_locks),
        "input_native_content_axis_sha256": _canonical_sha256(native_axis),
        "output_native_content_multiset_sha256": _canonical_sha256(
            sorted(
                (_canonical_sha256(item) for item in output_native_axis)
            )
        ),
        "input_native_content_multiset_sha256": _canonical_sha256(
            sorted((_canonical_sha256(item) for item in native_axis))
        ),
        "full_axis_planning_candidate_count": CANDIDATE_COUNT,
        "materialized_control_candidate_count": PAIR_MEMBER_COUNT,
        "maximum_device_transfer_candidate_count": PAIR_MEMBER_COUNT,
        "model_load_count": 0,
        "model_forward_count": 0,
        "model_backward_count": 0,
        "model_update_count": 0,
    }
    _require(
        payload["input_native_content_multiset_sha256"]
        == payload["output_native_content_multiset_sha256"],
        "full-C128 native-content multiset drift",
    )
    payload["logical_sha256"] = _canonical_sha256(payload)
    return MappingProxyType(bindings), MappingProxyType(payload)


def move_full_c128_c_dino_pair_to_device(
    pair_bindings: Mapping[str, CDinoCandidateBindingV1],
    device: torch.device | str,
    *,
    expected_pair_member_keys: Sequence[str] | None = None,
) -> Mapping[str, CDinoCandidateBindingV1]:
    """Move exactly two planned donor payloads, never a full C128, to device."""

    _require(
        len(pair_bindings) == PAIR_MEMBER_COUNT,
        "C_DINO device boundary accepts exactly two pair donor payloads",
    )
    if expected_pair_member_keys is not None:
        expected = tuple(expected_pair_member_keys)
        _require(
            len(expected) == PAIR_MEMBER_COUNT
            and set(pair_bindings) == set(expected),
            "C_DINO device pair-member address drift",
        )
    target_device = torch.device(device)
    output: dict[str, CDinoCandidateBindingV1] = {}
    for key, binding in pair_bindings.items():
        source = binding.candidate
        candidate = CandidateReferenceFieldV1(
            candidate_key=source.candidate_key,
            layers=source.layers.to(target_device),
            grid_shape=source.grid_shape,
            valid_patch_mask=source.valid_patch_mask,
            physical_gallery_row=source.physical_gallery_row,
            source_image_sha256=source.source_image_sha256,
            source_key=source.source_key,
            cache_payload_sha256=source.cache_payload_sha256,
            geometry_record_sha256=source.geometry_record_sha256,
            tokens_sha256=source.tokens_sha256,
        )
        output[key] = replace(binding, candidate=candidate)
    return MappingProxyType(output)


def extract_full_c128_c_dino_pair_from_runtime_locks(
    locks: Mapping[str, CandidatePLockV1],
    left_candidate_key: str,
    right_candidate_key: str,
    *,
    namespace: str,
    query_id: str,
    corrected_identity_by_physical_row: Mapping[int, str],
) -> tuple[Mapping[str, CDinoCandidateBindingV1], Mapping[str, object]]:
    """Compatibility convenience path for an already materialized CPU C128."""

    axis = full_c128_axis_from_runtime_locks(
        locks,
        query_id=query_id,
        corrected_identity_by_physical_row=corrected_identity_by_physical_row,
    )
    plan = plan_full_c128_c_dino_pair(
        axis,
        left_candidate_key,
        right_candidate_key,
        namespace=namespace,
    )
    pair = plan.pair_member_keys
    donors = plan.donor_candidate_keys
    return materialize_full_c128_c_dino_pair(
        plan,
        destination_pair_locks={key: locks[key] for key in pair},
        donor_candidates={key: locks[key].candidate for key in donors},
        donor_dino_content_binding_sha256_by_candidate_key={
            key: _canonical_sha256(_candidate_content_record(locks[key].candidate))
            for key in donors
        },
    )


__all__ = [
    "SCHEMA_VERSION",
    "CONTROL_NAME",
    "CANDIDATE_COUNT",
    "PAIR_MEMBER_COUNT",
    "FullC128CDinoContractError",
    "FullC128CandidateAxisRecordV1",
    "FullC128ControlAxisV1",
    "FullC128CDinoPairPlanV1",
    "corrected_identity_sha256",
    "full_c128_axis_from_selected_head_records",
    "full_c128_axis_from_runtime_locks",
    "plan_full_c128_c_dino_pair",
    "materialize_full_c128_c_dino_pair",
    "dino_content_binding_sha256_from_cache_index_record",
    "move_full_c128_c_dino_pair_to_device",
    "extract_full_c128_c_dino_pair_from_runtime_locks",
]
