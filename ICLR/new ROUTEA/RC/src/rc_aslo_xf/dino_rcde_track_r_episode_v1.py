"""Compact Track-R inner-OOF episode descriptors over Natural/Core P-lock V2.

The descriptor stores only immutable addresses, two candidates' four selected
Natural/Core V2 lock records, and validated loss-role receipt hashes.  DINO tensors are
not copied into the ledger.  A training consumer must reopen the registered
query/reference cache payloads and project these records through the qualified
V2 three-arm adapter.

Labels are allowed only to select the exact target and the already frozen
natural strongest rival after the complete P-lock artifact has been sealed.
No score, rank, winner, gap, outcome, mask supervision, donor, or action is
serialized here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256
from .dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1 import (
    HeadSelectionSpecV1,
    inner_oof_head_spec,
    outer_refit_head_spec,
    select_natural_v2_head_records,
)


SCHEMA_VERSION = "rc_dino_rcde_track_r_compact_episode_v1_20260821"
FIXED_DIRECTIONS = ("a_to_b", "b_to_a")
SOURCE_FOLDS = (1, 2, 3, 4)
EXPECTED_HEAD_COUNT = 4
EXPECTED_CANDIDATE_COUNT = 128
EXPECTED_RECORD_COUNT = 1024


class TrackREpisodeError(ValueError):
    """A V2 artifact, fold/head, or compact loss-role invariant failed."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackREpisodeError(message)


def _sha(value: object, *, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} SHA256 drift",
    )
    return value


_VALIDATED_HEAD_TOKEN = object()


@dataclass(frozen=True)
class ValidatedHeadRecordsV1(Sequence[Mapping[str, Any]]):
    """One fully validated Natural/Core head, reusable without revalidation."""

    spec: HeadSelectionSpecV1
    expected_candidate_count: int
    records: tuple[Mapping[str, Any], ...] = field(repr=False, compare=False)
    record_sequence_sha256: str
    candidate_axis_sha256: str
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        require(self._token is _VALIDATED_HEAD_TOKEN, "validated-head constructor token drift")
        require(
            type(self.expected_candidate_count) is int
            and self.expected_candidate_count >= 2
            and len(self.records) == self.expected_candidate_count * len(FIXED_DIRECTIONS),
            "validated-head record population drift",
        )
        sequence = [str(item["record_sha256"]) for item in self.records]
        axis = [
            (
                item["candidate"]["candidate_position"],
                item["candidate"]["candidate_key"],
                item["candidate"]["candidate_physical_row"],
                item["candidate"]["candidate_native_content_sha256"],
                item["candidate"]["candidate_reference_source_sha256"],
            )
            for item in self.records
            if item["direction"] == FIXED_DIRECTIONS[0]
        ]
        require(
            self.record_sequence_sha256 == canonical_sha256(sequence)
            and self.candidate_axis_sha256 == canonical_sha256(axis),
            "validated-head receipt drift",
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def expected_head_specs(source_fold: int) -> tuple[HeadSelectionSpecV1, ...]:
    require(source_fold in SOURCE_FOLDS, "source fold drift")
    return (
        outer_refit_head_spec(source_fold=source_fold),
        *(
            inner_oof_head_spec(outer_fold=outer_fold, source_fold=source_fold)
            for outer_fold in SOURCE_FOLDS
            if outer_fold != source_fold
        ),
    )


def index_complete_artifact_records(
    records: Sequence[Mapping[str, Any]],
    *,
    source_fold: int,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
) -> Mapping[str, ValidatedHeadRecordsV1]:
    """Validate and index the exact four-head x C128 x two-direction artifact."""

    require(
        len(records) == EXPECTED_HEAD_COUNT * expected_candidate_count * 2,
        "P-V2 artifact record count drift",
    )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        require(isinstance(record, Mapping), "P-V2 artifact record is not a mapping")
        query = record.get("query")
        require(isinstance(query, Mapping), "P-V2 record query address absent")
        fit_id = query.get("fit_id")
        require(isinstance(fit_id, str) and bool(fit_id), "P-V2 fit ID absent")
        grouped.setdefault(fit_id, []).append(record)
    specs = expected_head_specs(source_fold)
    expected_fit_ids = {spec.fit_id for spec in specs}
    require(set(grouped) == expected_fit_ids, "P-V2 artifact four-head population drift")

    output: dict[str, ValidatedHeadRecordsV1] = {}
    canonical_axis: tuple[tuple[object, ...], ...] | None = None
    common_query: tuple[object, ...] | None = None
    for spec in specs:
        try:
            selected = select_natural_v2_head_records(
                grouped[spec.fit_id],
                spec=spec,
                expected_candidate_count=expected_candidate_count,
            )
        except Exception as error:
            raise TrackREpisodeError(
                f"Natural/Core V2 head validation failed: {error}"
            ) from error
        axis = tuple(
            (
                record["candidate"]["candidate_position"],
                record["candidate"]["candidate_key"],
                record["candidate"]["candidate_physical_row"],
                record["candidate"]["candidate_native_content_sha256"],
                record["candidate"]["candidate_reference_source_sha256"],
            )
            for record in selected
            if record["direction"] == FIXED_DIRECTIONS[0]
        )
        query = selected[0]["query"]
        query_receipt = (
            query["query_id"],
            query["historical_query_ordinal"],
            query["execution_ordinal"],
            query["query_source_image_sha256"],
            query["track"],
        )
        if canonical_axis is None:
            canonical_axis = axis
            common_query = query_receipt
        else:
            require(axis == canonical_axis, "candidate axis differs across P heads")
            require(query_receipt == common_query, "query provenance differs across P heads")
        output[spec.fit_id] = ValidatedHeadRecordsV1(
            spec=spec,
            expected_candidate_count=expected_candidate_count,
            records=tuple(selected),
            record_sequence_sha256=canonical_sha256(
                [str(item["record_sha256"]) for item in selected]
            ),
            candidate_axis_sha256=canonical_sha256(axis),
            _token=_VALIDATED_HEAD_TOKEN,
        )
    return output


def _records_by_position(
    selected: Sequence[Mapping[str, Any]],
) -> Mapping[int, Mapping[str, Mapping[str, Any]]]:
    output: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in selected:
        position = record["candidate"]["candidate_position"]
        direction = record["direction"]
        require(type(position) is int, "candidate position drift")
        require(direction in FIXED_DIRECTIONS, "direction drift")
        bucket = output.setdefault(position, {})
        require(direction not in bucket, "candidate direction duplicate")
        bucket[direction] = record
    require(
        all(set(bucket) == set(FIXED_DIRECTIONS) for bucket in output.values()),
        "candidate direction pair incomplete",
    )
    return output


def build_compact_episode(
    *,
    selected_head_records: Sequence[Mapping[str, Any]],
    source_fold: int,
    outer_fold: int,
    query_id: str,
    execution_ordinal: int,
    query_source_image_sha256: str,
    track: str,
    target_candidate_key: str,
    strongest_rival_candidate_key: str,
    loss_role_record_sha256: str,
    loss_join_sha256: str,
    lock_artifact_path: str,
    lock_artifact_sha256: str,
) -> dict[str, Any]:
    """Select one target/rival episode after an immutable inner-OOF head exists."""

    require(source_fold in SOURCE_FOLDS, "episode source fold drift")
    require(outer_fold in SOURCE_FOLDS and outer_fold != source_fold, "episode outer fold drift")
    spec = inner_oof_head_spec(outer_fold=outer_fold, source_fold=source_fold)
    if isinstance(selected_head_records, ValidatedHeadRecordsV1):
        require(
            selected_head_records._token is _VALIDATED_HEAD_TOKEN
            and selected_head_records.spec == spec
            and selected_head_records.expected_candidate_count
            == EXPECTED_CANDIDATE_COUNT,
            "validated episode-head receipt/spec drift",
        )
        selected = selected_head_records.records
    else:
        try:
            selected = select_natural_v2_head_records(
                selected_head_records,
                spec=spec,
                expected_candidate_count=EXPECTED_CANDIDATE_COUNT,
            )
        except Exception as error:
            raise TrackREpisodeError(
                f"Natural/Core V2 episode-head validation failed: {error}"
            ) from error
    query = selected[0]["query"]
    require(
        query["query_id"] == query_id
        and query["execution_ordinal"] == execution_ordinal
        and query["query_source_image_sha256"] == query_source_image_sha256
        and query["track"] == track,
        "episode query/source role join drift",
    )
    by_position = _records_by_position(selected)
    key_to_position: dict[str, int] = {}
    for position, directions in by_position.items():
        key = str(directions[FIXED_DIRECTIONS[0]]["candidate"]["candidate_key"])
        require(key not in key_to_position, "candidate key duplicate")
        key_to_position[key] = position
    require(
        target_candidate_key != strongest_rival_candidate_key
        and target_candidate_key in key_to_position
        and strongest_rival_candidate_key in key_to_position,
        "target/rival is absent or duplicated in natural C128",
    )

    def member(role: str, candidate_key: str) -> dict[str, Any]:
        position = key_to_position[candidate_key]
        records = by_position[position]
        first = records[FIXED_DIRECTIONS[0]]
        candidate = first["candidate"]
        return {
            "role": role,
            "candidate_position": position,
            "candidate_key": candidate["candidate_key"],
            "candidate_physical_row": candidate["candidate_physical_row"],
            "candidate_native_content_sha256": candidate[
                "candidate_native_content_sha256"
            ],
            "candidate_reference_source_sha256": candidate[
                "candidate_reference_source_sha256"
            ],
            "direction_records": {
                direction: copy.deepcopy(records[direction])
                for direction in FIXED_DIRECTIONS
            },
            "direction_record_sha256": [
                records[direction]["record_sha256"] for direction in FIXED_DIRECTIONS
            ],
        }

    target = member("TARGET", target_candidate_key)
    rival = member("STRONGEST_NATURAL_RIVAL", strongest_rival_candidate_key)
    episode_id = canonical_sha256(
        {
            "namespace": "DINO_RCDE_TRACK_R_EPISODE_V1",
            "query_source_image_sha256": query_source_image_sha256,
            "outer_fold": outer_fold,
            "fit_id": spec.fit_id,
        }
    )
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "query": {
            "query_id": query_id,
            "execution_ordinal": execution_ordinal,
            "query_source_image_sha256": query_source_image_sha256,
            "source_fold": source_fold,
            "outer_fold": outer_fold,
            "fit_id": spec.fit_id,
            "crossfit_role": spec.crossfit_role,
            "track": track,
        },
        "target": target,
        "strongest_rival": rival,
        "loss_role_record_sha256": _sha(
            loss_role_record_sha256, name="loss role record"
        ),
        "loss_join_sha256": _sha(loss_join_sha256, name="loss join"),
        "lock_artifact_path": lock_artifact_path,
        "lock_artifact_sha256": _sha(
            lock_artifact_sha256, name="lock artifact"
        ),
        "label_use": "LOSS_ONLY_AFTER_INNER_OOF_P_LOCK_SEAL",
        "model_visible_role_fields": ["target.candidate_key", "strongest_rival.candidate_key"],
        "model_visible_colnomic_score_rank_winner_gap_fields": [],
        "donor_null_h0_hold_switch_fields": [],
    }
    value["record_sha256"] = canonical_sha256(value)
    validate_compact_episode(value)
    return value


def validate_compact_episode(value: Mapping[str, Any]) -> None:
    expected_top = {
        "schema_version",
        "episode_id",
        "query",
        "target",
        "strongest_rival",
        "loss_role_record_sha256",
        "loss_join_sha256",
        "lock_artifact_path",
        "lock_artifact_sha256",
        "label_use",
        "model_visible_role_fields",
        "model_visible_colnomic_score_rank_winner_gap_fields",
        "donor_null_h0_hold_switch_fields",
        "record_sha256",
    }
    require(set(value) == expected_top, "compact episode field-set drift")
    require(value.get("schema_version") == SCHEMA_VERSION, "compact episode schema drift")
    unsigned = {key: copy.deepcopy(item) for key, item in value.items() if key != "record_sha256"}
    require(value.get("record_sha256") == canonical_sha256(unsigned), "compact episode hash drift")
    query = value.get("query")
    require(
        isinstance(query, Mapping)
        and query.get("source_fold") in SOURCE_FOLDS
        and query.get("outer_fold") in SOURCE_FOLDS
        and query.get("source_fold") != query.get("outer_fold")
        and query.get("fit_id")
        == f"P_OUTER{query.get('outer_fold')}_INNER{query.get('source_fold')}_FIT"
        and query.get("crossfit_role") == "INNER_HELDOUT_DEPLOYMENT",
        "compact episode fold/head drift",
    )
    _sha(query.get("query_source_image_sha256"), name="query source")
    _sha(value.get("loss_role_record_sha256"), name="loss role record")
    _sha(value.get("loss_join_sha256"), name="loss join")
    _sha(value.get("lock_artifact_sha256"), name="lock artifact")
    require(
        value.get("label_use") == "LOSS_ONLY_AFTER_INNER_OOF_P_LOCK_SEAL"
        and value.get("model_visible_role_fields")
        == ["target.candidate_key", "strongest_rival.candidate_key"]
        and value.get("model_visible_colnomic_score_rank_winner_gap_fields") == []
        and value.get("donor_null_h0_hold_switch_fields") == [],
        "compact episode role/visibility boundary drift",
    )
    members = (value.get("target"), value.get("strongest_rival"))
    require(all(isinstance(member, Mapping) for member in members), "episode member absent")
    require(
        members[0]["candidate_key"] != members[1]["candidate_key"]
        and members[0]["candidate_physical_row"]
        != members[1]["candidate_physical_row"],
        "episode target/rival collision",
    )
    for member in members:
        records = member.get("direction_records")
        require(
            isinstance(records, Mapping)
            and tuple(records) == FIXED_DIRECTIONS
            and member.get("direction_record_sha256")
            == [records[direction]["record_sha256"] for direction in FIXED_DIRECTIONS],
            "episode direction-record receipt drift",
        )
        for direction in FIXED_DIRECTIONS:
            record = records[direction]
            require(
                record["record_sha256"]
                == member["direction_record_sha256"][FIXED_DIRECTIONS.index(direction)]
                and record["query"]["fit_id"] == query["fit_id"]
                and record["query"]["query_id"] == query["query_id"]
                and record["query"]["execution_ordinal"] == query["execution_ordinal"]
                and record["direction"] == direction
                and record["candidate"]["candidate_key"] == member["candidate_key"]
                and record["candidate"]["candidate_physical_row"]
                == member["candidate_physical_row"],
                "episode member/Natural V2 record binding drift",
            )


__all__ = [
    "SCHEMA_VERSION",
    "FIXED_DIRECTIONS",
    "SOURCE_FOLDS",
    "EXPECTED_HEAD_COUNT",
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_RECORD_COUNT",
    "TrackREpisodeError",
    "ValidatedHeadRecordsV1",
    "expected_head_specs",
    "index_complete_artifact_records",
    "build_compact_episode",
    "validate_compact_episode",
]
