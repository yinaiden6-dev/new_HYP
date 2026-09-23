"""Runtime construction of VPairEpisode objects from compact Track-R ledgers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RC_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from rc_aslo_xf.dino_rcde_cw1_multitile_vdecode_v1 import (  # noqa: E402
    CandidateReferenceFieldV1,
    QueryTokenFieldV1,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402
from rc_aslo_xf.dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1 import (  # noqa: E402
    GeometryNamespaceBindingV1,
    inner_oof_head_spec,
    project_natural_v2_candidate_to_v1_runtime,
)
from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (  # noqa: E402
    VPairEpisode,
    episode_schedule_key,
    ordered_training_pool,
)
from rc_aslo_xf.dino_rcde_track_r_episode_v1 import (  # noqa: E402
    validate_compact_episode,
)

import dino_rcde_sr0_mt_v_input_common_v1 as v_input  # noqa: E402


LEDGER_SCHEMA = "rc_dino_rcde_track_r_relative_e1_episode_ledger_v1_20260821"
LEDGER_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_LEDGER_READY"


class TrackRInputError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRInputError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def safe_path(path: Path, *, file: bool = True) -> Path:
    value = path.resolve()
    require(RC_ROOT == value or RC_ROOT in value.parents, f"path escapes RC root: {value}")
    require(
        not {"c8", "s8", "opened", "sealed"}.intersection(
            part.lower() for part in value.parts
        ),
        f"protected path requested: {value}",
    )
    require((value.is_file() if file else value.is_dir()) and not value.is_symlink(), f"input absent/unsafe: {value}")
    return value


def load_compact_ledger(
    path: Path,
    *,
    expected_outer_fold: int,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    ledger_path = safe_path(path)
    if expected_sha256 is not None:
        require(file_sha256(ledger_path) == expected_sha256, "compact ledger physical hash drift")
    value = torch.load(ledger_path, map_location="cpu", weights_only=True, mmap=True)
    require(
        isinstance(value, Mapping)
        and value.get("schema_version") == LEDGER_SCHEMA
        and value.get("status") == LEDGER_STATUS
        and value.get("outer_fold") == expected_outer_fold
        and value.get("input_role") == "INNER_OOF_P_V2_LOCKS_ONLY"
        and value.get("outer_heldout_record_count") == 0
        and value.get("logical_sha256") == logical_sha256(value)
        and value.get("training_authorized") is False
        and value.get("scientific_GO_or_NO_GO") is None,
        "compact ledger envelope drift",
    )
    episodes = value.get("episodes")
    require(isinstance(episodes, list) and len(episodes) == value.get("episode_count"), "compact episode population drift")
    for episode in episodes:
        validate_compact_episode(episode)
        require(episode["query"]["outer_fold"] == expected_outer_fold, "compact ledger mixed outer folds")
    return value


def load_geometry_payload(path: Path) -> tuple[dict[int, Mapping[str, Any]], dict[int, Mapping[str, Any]], str]:
    geometry_path = safe_path(path)
    value = torch.load(geometry_path, map_location="cpu", weights_only=True, mmap=True)
    require(
        isinstance(value, Mapping)
        and value.get("schema_version")
        == "rc_dino_rcde_colnomic_sr_full600_geometry_payload_v2_20260815"
        and value.get("status") == "RCDE_SR_FULL600_CANONICAL_GEOMETRY_V2_READY"
        and len(value.get("query_records", ())) == 600
        and len(value.get("reference_records", ())) == 4748,
        "full600 canonical geometry payload drift",
    )
    queries = {int(item["execution_ordinal"]): item for item in value["query_records"]}
    references = {int(item["physical_row"]): item for item in value["reference_records"]}
    require(len(queries) == 600 and len(references) == 4748, "full600 geometry address collision")
    return queries, references, file_sha256(geometry_path)


def materialize_vpair_episodes(
    ledger: Mapping[str, Any],
    *,
    cache_root: Path,
    cache_index: Mapping[tuple[str, int], Mapping[str, object]],
    expected_cache_model_sha256: str,
    query_geometry_by_execution: Mapping[int, Mapping[str, Any]],
    reference_geometry_by_row: Mapping[int, Mapping[str, Any]],
) -> tuple[tuple[VPairEpisode, ...], dict[str, Any]]:
    """Reopen each unique token payload once and construct two-lock episodes."""

    cache = safe_path(cache_root, file=False)
    outer_fold = int(ledger["outer_fold"])
    query_cache: dict[int, QueryTokenFieldV1] = {}
    reference_cache: dict[int, CandidateReferenceFieldV1] = {}
    output: list[VPairEpisode] = []
    projection_count = 0
    for descriptor in ledger["episodes"]:
        validate_compact_episode(descriptor)
        query_address = descriptor["query"]
        execution = int(query_address["execution_ordinal"])
        source_fold = int(query_address["source_fold"])
        require(int(query_address["outer_fold"]) == outer_fold, "descriptor outer fold drift")
        query_geometry = query_geometry_by_execution.get(execution)
        require(isinstance(query_geometry, Mapping), "query canonical geometry absent")
        query = query_cache.get(execution)
        if query is None:
            query = v_input.make_query_field(
                {
                    "execution_ordinal": execution,
                    "query_source_image_sha256": query_address[
                        "query_source_image_sha256"
                    ],
                },
                cache_root=cache,
                index=cache_index,
                expected_model_sha256=expected_cache_model_sha256,
            )
            require(
                query.source_image_sha256 == query_geometry["source_image_sha256"],
                "query token/canonical geometry source drift",
            )
            query_cache[execution] = query
        locks = {}
        keys = []
        for member_name in ("target", "strongest_rival"):
            member = descriptor[member_name]
            row = int(member["candidate_physical_row"])
            reference_geometry = reference_geometry_by_row.get(row)
            require(isinstance(reference_geometry, Mapping), "reference canonical geometry absent")
            candidate = reference_cache.get(row)
            if candidate is None:
                candidate = v_input.make_candidate_field(
                    row,
                    str(member["candidate_reference_source_sha256"]),
                    cache_root=cache,
                    index=cache_index,
                    expected_model_sha256=expected_cache_model_sha256,
                )
                require(
                    candidate.source_image_sha256
                    == reference_geometry["source_image_sha256"],
                    "reference token/canonical geometry source drift",
                )
                reference_cache[row] = candidate
            require(candidate.candidate_key == member["candidate_key"], "candidate key/cache drift")
            spec = inner_oof_head_spec(
                outer_fold=outer_fold, source_fold=source_fold
            )
            projection = project_natural_v2_candidate_to_v1_runtime(
                query=query,
                candidate=candidate,
                direction_records=member["direction_records"],
                spec=spec,
                query_geometry_binding=GeometryNamespaceBindingV1(
                    source_image_sha256=query.source_image_sha256,
                    grid_shape=query.grid_shape,
                    canonical_geometry_sha256=query_geometry["dino_geometry"]
                    ["geometry_sha256"],
                    cache_geometry_record_sha256=query.geometry_record_sha256,
                ),
                reference_geometry_binding=GeometryNamespaceBindingV1(
                    source_image_sha256=candidate.source_image_sha256,
                    grid_shape=candidate.grid_shape,
                    canonical_geometry_sha256=reference_geometry["dino_geometry"]
                    ["geometry_sha256"],
                    cache_geometry_record_sha256=candidate.geometry_record_sha256,
                ),
            )
            locks[candidate.candidate_key] = projection.runtime_lock
            keys.append(candidate.candidate_key)
            projection_count += 1
        output.append(
            VPairEpisode(
                episode_id=str(descriptor["episode_id"]),
                query=query,
                locks=locks,
                target_key=keys[0],
                strongest_rival_key=keys[1],
                outer_fold=outer_fold,
                identity_key=str(descriptor["loss_role_record_sha256"]),
                supergroup_key=str(descriptor["loss_join_sha256"]),
            )
        )
    require(len(output) == int(ledger["episode_count"]), "runtime episode count drift")
    require(len({item.episode_id for item in output}) == len(output), "runtime episode ID collision")
    ordered = ordered_training_pool(output, outer_fold)
    receipt = {
        "outer_fold": outer_fold,
        "episode_count": len(output),
        "unique_query_token_count": len(query_cache),
        "unique_reference_token_count": len(reference_cache),
        "candidate_projection_count": projection_count,
        "query_token_cache_reuse_count": len(output) - len(query_cache),
        "reference_token_cache_reuse_count": 2 * len(output) - len(reference_cache),
        "episode_id_sequence_sha256": canonical_sha256(
            [item.episode_id for item in output]
        ),
        "query_schedule_key_sequence_sha256": canonical_sha256(
            [episode_schedule_key(item) for item in output]
        ),
        "ordered_query_schedule_keys": [
            episode_schedule_key(item) for item in ordered
        ],
        "ordered_query_schedule_keys_sha256": canonical_sha256(
            [episode_schedule_key(item) for item in ordered]
        ),
        "target_rival_key_sequence_sha256": canonical_sha256(
            [(item.target_key, item.strongest_rival_key) for item in output]
        ),
        "score_rank_winner_gap_outcome_read_count": 0,
        "donor_null_h0_hold_switch_read_count": 0,
        "protected_access_count": 0,
    }
    receipt["logical_sha256"] = canonical_sha256(receipt)
    return tuple(output), receipt


__all__ = [
    "LEDGER_SCHEMA",
    "LEDGER_STATUS",
    "TrackRInputError",
    "file_sha256",
    "logical_sha256",
    "safe_path",
    "load_compact_ledger",
    "load_geometry_payload",
    "materialize_vpair_episodes",
]
