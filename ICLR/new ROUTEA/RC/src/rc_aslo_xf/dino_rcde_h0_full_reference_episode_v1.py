"""Additive donor descriptors over immutable Track-R compact episodes."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from .dino_rcde_sr0_mt_p_natural_adapter_v2 import validate_natural_p_lock_v2
from .dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256
from .dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1 import inner_oof_head_spec
from .dino_rcde_track_r_episode_v1 import FIXED_DIRECTIONS, validate_compact_episode


SCHEMA_VERSION = "rc_dino_rcde_h0_full_reference_episode_augmentation_v1_20260821"
MATCHED = "H0_FULL_REFERENCE_DONOR_MATCHED"
PAIR_ONLY = "H0_Q0_UNMATCHED_PAIR_ONLY"


class H0EpisodeError(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise H0EpisodeError(message)


def sha(value: object, *, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} SHA256 drift",
    )
    return str(value)


def build_relative_from_selected_records(
    records: Sequence[Mapping[str, Any]],
    *,
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
    """Build the canonical relative descriptor from only 2x2 consumed rows.

    V111 already independently validated all 608,256 records. U1 rehashes the
    artifact and revalidates only these four records instead of repeating the
    full C128 scan for every consumer.
    """

    spec = inner_oof_head_spec(outer_fold=outer_fold, source_fold=source_fold)
    require(len(records) == 4, "selected target/rival direction population drift")
    by_key: dict[str, dict[str, Mapping[str, Any]]] = {}
    checkpoint = set()
    for record in records:
        validate_natural_p_lock_v2(record)
        query = record["query"]
        candidate = record["candidate"]
        require(
            query["fit_id"] == spec.fit_id
            and query["crossfit_role"] == spec.crossfit_role
            and query["outer_fold"] == spec.outer_fold
            and query["inner_heldout_fold"] == spec.inner_heldout_fold
            and query["query_id"] == query_id
            and int(query["execution_ordinal"]) == execution_ordinal
            and query["query_source_image_sha256"] == query_source_image_sha256
            and query["track"] == track,
            "selected record query/head drift",
        )
        key = str(candidate["candidate_key"])
        direction = str(record["direction"])
        require(
            key in {target_candidate_key, strongest_rival_candidate_key}
            and direction in FIXED_DIRECTIONS,
            "selected record role/direction drift",
        )
        bucket = by_key.setdefault(key, {})
        require(direction not in bucket, "selected record duplicate")
        bucket[direction] = record
        checkpoint.add((record["p_checkpoint_sha256"], record["p_training_manifest_sha256"]))
    require(
        set(by_key) == {target_candidate_key, strongest_rival_candidate_key}
        and all(tuple(bucket) == FIXED_DIRECTIONS for bucket in by_key.values())
        and len(checkpoint) == 1,
        "selected target/rival closure drift",
    )

    def member(role: str, key: str) -> dict[str, Any]:
        directions = by_key[key]
        first = directions[FIXED_DIRECTIONS[0]]
        candidate = first["candidate"]
        require(
            all(
                directions[direction]["candidate"]["candidate_key"] == key
                and directions[direction]["candidate"]["candidate_physical_row"]
                == candidate["candidate_physical_row"]
                and directions[direction]["candidate"]["candidate_reference_source_sha256"]
                == candidate["candidate_reference_source_sha256"]
                for direction in FIXED_DIRECTIONS
            )
            and len(
                {
                    directions[direction]["candidate"]["candidate_native_content_sha256"]
                    for direction in FIXED_DIRECTIONS
                }
            )
            == 2,
            "selected candidate direction binding drift",
        )
        return {
            "role": role,
            "candidate_position": candidate["candidate_position"],
            "candidate_key": key,
            "candidate_physical_row": candidate["candidate_physical_row"],
            "candidate_native_content_sha256": candidate[
                "candidate_native_content_sha256"
            ],
            "candidate_reference_source_sha256": candidate[
                "candidate_reference_source_sha256"
            ],
            "direction_records": {
                direction: copy.deepcopy(directions[direction])
                for direction in FIXED_DIRECTIONS
            },
            "direction_record_sha256": [
                directions[direction]["record_sha256"]
                for direction in FIXED_DIRECTIONS
            ],
        }

    target = member("TARGET", target_candidate_key)
    rival = member("STRONGEST_NATURAL_RIVAL", strongest_rival_candidate_key)
    require(
        target["candidate_physical_row"] != rival["candidate_physical_row"]
        and target["candidate_position"] != rival["candidate_position"],
        "selected target/rival physical collision",
    )
    value: dict[str, Any] = {
        "schema_version": "rc_dino_rcde_track_r_compact_episode_v1_20260821",
        "episode_id": canonical_sha256(
            {
                "namespace": "DINO_RCDE_TRACK_R_EPISODE_V1",
                "query_source_image_sha256": query_source_image_sha256,
                "outer_fold": outer_fold,
                "fit_id": spec.fit_id,
            }
        ),
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
        "loss_role_record_sha256": sha(
            loss_role_record_sha256, name="loss role record"
        ),
        "loss_join_sha256": sha(loss_join_sha256, name="loss join"),
        "lock_artifact_path": lock_artifact_path,
        "lock_artifact_sha256": sha(lock_artifact_sha256, name="lock artifact"),
        "label_use": "LOSS_ONLY_AFTER_INNER_OOF_P_LOCK_SEAL",
        "model_visible_role_fields": [
            "target.candidate_key",
            "strongest_rival.candidate_key",
        ],
        "model_visible_colnomic_score_rank_winner_gap_fields": [],
        "donor_null_h0_hold_switch_fields": [],
    }
    value["record_sha256"] = canonical_sha256(value)
    validate_compact_episode(value)
    return value


def inner_q0_index(q0_result: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    require(
        q0_result.get("status") == "FULL_REFERENCE_BALANCED_DONOR_Q0_GO"
        and q0_result.get("q0_qualified") is True
        and q0_result.get("scope_node_count") == 2376,
        "Q0 result envelope drift",
    )
    output = {}
    for scope in q0_result["scope_results"]:
        if scope["namespace"] != "TRAIN_INNER_OOF_NULL_FEASIBILITY":
            continue
        fit_id = str(scope["fit_id"])
        for row in scope["balanced_rows"]:
            key = (fit_id, int(row["recipient_execution_ordinal"]))
            require(key not in output, "Q0 inner assignment alias")
            output[key] = row
    require(len(output) == 1782, "Q0 inner assignment population drift")
    return output


def build_augmentation(
    relative_episode: Mapping[str, Any],
    q0_row: Mapping[str, Any],
    *,
    q0_result_logical_sha256: str,
) -> dict[str, Any]:
    validate_compact_episode(relative_episode)
    query = relative_episode["query"]
    target = relative_episode["target"]
    require(
        int(q0_row["recipient_execution_ordinal"]) == int(query["execution_ordinal"])
        and q0_row["recipient_query_id"] == query["query_id"]
        and q0_row["recipient_query_source_sha256"] == query["query_source_image_sha256"]
        and q0_row["track"] == query["track"]
        and q0_row["recipient_candidate_key"] == target["candidate_key"]
        and int(q0_row["recipient_physical_row"]) == int(target["candidate_physical_row"])
        and q0_row["recipient_reference_source_sha256"]
        == target["candidate_reference_source_sha256"],
        "relative episode/Q0 recipient closure drift",
    )
    donor = q0_row.get("donor")
    status = str(q0_row["status"])
    if status == "MATCHED":
        require(isinstance(donor, Mapping), "matched Q0 donor absent")
        payload = donor["payload_key"]
        require(
            isinstance(payload, list)
            and len(payload) == 3
            and payload[0] not in {
                target["candidate_key"],
                relative_episode["strongest_rival"]["candidate_key"],
            }
            and int(payload[1]) not in {
                int(target["candidate_physical_row"]),
                int(relative_episode["strongest_rival"]["candidate_physical_row"]),
            },
            "donor collides with natural target/rival",
        )
        donor_value = {
            "candidate_key": sha(payload[0], name="donor candidate key"),
            "candidate_physical_row": int(payload[1]),
            "candidate_reference_source_sha256": sha(payload[2], name="donor reference source"),
            "provider_execution_ordinal": int(donor["provider_execution_ordinal"]),
            "provider_query_id": str(donor["provider_query_id"]),
            "provider_query_source_sha256": sha(
                donor["provider_query_source_sha256"], name="donor provider query source"
            ),
            "identity_sha256": sha(donor["identity_sha256"], name="donor identity hash"),
            "supergroup_sha256": sha(donor["supergroup_sha256"], name="donor supergroup hash"),
            "reference_scope": "COMPLETE_NATIVE_FULL_REFERENCE_VALID_MASK",
            "donor_query_or_p_lock_consumed": False,
        }
        augmentation_status = MATCHED
    elif status == "UNMATCHED":
        require(donor is None, "unmatched Q0 row contains donor")
        donor_value = None
        augmentation_status = PAIR_ONLY
    else:
        raise H0EpisodeError("Q0 assignment status drift")
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": augmentation_status,
        "episode_id": relative_episode["episode_id"],
        "relative_episode_record_sha256": relative_episode["record_sha256"],
        "query": copy.deepcopy(query),
        "positive": {
            "candidate_key": target["candidate_key"],
            "candidate_physical_row": target["candidate_physical_row"],
            "candidate_reference_source_sha256": target[
                "candidate_reference_source_sha256"
            ],
            "direction_record_sha256": list(target["direction_record_sha256"]),
            "query_mask_source": "RECIPIENT_EXACT_CLEAN_P_LOCK_QUERY_UNION_PER_DIRECTION",
            "reference_scope": "COMPLETE_NATIVE_FULL_REFERENCE_VALID_MASK",
        },
        "donor": donor_value,
        "loss_contract": (
            "PAIR_PLUS_POSITIVE_UNARY_PLUS_DONOR_UNARY"
            if augmentation_status == MATCHED
            else "PAIR_ONLY_NO_DONOR_FALLBACK"
        ),
        "q0_assignment_sha256": canonical_sha256(q0_row),
        "q0_result_logical_sha256": sha(
            q0_result_logical_sha256, name="Q0 result logical"
        ),
        "model_visible_colnomic_score_rank_winner_gap_fields": [],
        "donor_query_mask_or_local_component_fields": [],
    }
    value["record_sha256"] = canonical_sha256(value)
    validate_augmentation(value)
    return value


def validate_augmentation(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "status",
        "episode_id",
        "relative_episode_record_sha256",
        "query",
        "positive",
        "donor",
        "loss_contract",
        "q0_assignment_sha256",
        "q0_result_logical_sha256",
        "model_visible_colnomic_score_rank_winner_gap_fields",
        "donor_query_mask_or_local_component_fields",
        "record_sha256",
    }
    require(set(value) == expected, "augmentation field-set drift")
    unsigned = {key: copy.deepcopy(item) for key, item in value.items() if key != "record_sha256"}
    require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("record_sha256") == canonical_sha256(unsigned)
        and value.get("status") in {MATCHED, PAIR_ONLY}
        and value.get("model_visible_colnomic_score_rank_winner_gap_fields") == []
        and value.get("donor_query_mask_or_local_component_fields") == [],
        "augmentation envelope drift",
    )
    sha(value.get("relative_episode_record_sha256"), name="relative episode record")
    sha(value.get("q0_assignment_sha256"), name="Q0 assignment")
    sha(value.get("q0_result_logical_sha256"), name="Q0 result logical")
    donor = value.get("donor")
    if value["status"] == MATCHED:
        require(
            isinstance(donor, Mapping)
            and donor.get("reference_scope")
            == "COMPLETE_NATIVE_FULL_REFERENCE_VALID_MASK"
            and donor.get("donor_query_or_p_lock_consumed") is False
            and value.get("loss_contract")
            == "PAIR_PLUS_POSITIVE_UNARY_PLUS_DONOR_UNARY",
            "matched augmentation semantics drift",
        )
    else:
        require(
            donor is None
            and value.get("loss_contract") == "PAIR_ONLY_NO_DONOR_FALLBACK",
            "pair-only augmentation semantics drift",
        )


__all__ = [
    "SCHEMA_VERSION",
    "MATCHED",
    "PAIR_ONLY",
    "H0EpisodeError",
    "build_relative_from_selected_records",
    "inner_q0_index",
    "build_augmentation",
    "validate_augmentation",
]
