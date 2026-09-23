#!/usr/bin/env python3
"""Independent V124-E1 V2 replay with full 128/256 C_COL validation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import validate_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_independent_v1 as IV1  # noqa: E402
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import (  # noqa: E402
    candidate_p_lock_v2_from_record,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_natural_adapter_v2 import (  # noqa: E402
    validate_natural_p_lock_v2,
)
from rc_aslo_xf.dino_rcde_track_r_c_col_p_phase_a_schema_v1 import (  # noqa: E402
    CANDIDATE_FIELDS,
    CANDIDATE_SCHEMA,
    candidate_logical_sha256,
)
from rc_aslo_xf.dino_rcde_track_r_v124_e1_atomic_family_v2 import (  # noqa: E402
    file_sha256,
    logical_sha256,
    publish_validation_family,
    validate_family,
)
from rc_aslo_xf.dino_rcde_track_r_v124_e1_authority_v2 import (  # noqa: E402
    CANDIDATE_AXIS_SHA256,
    PAIR_SHA256,
    read_authority,
)


AUTHORITY = ROOT / "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_authority_v2_20260826.json"
PRODUCER_DIR = ROOT / "results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v2_committed"
PUBLIC_DIR = ROOT / "results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_validation_v2_committed"
ARTIFACT_SCHEMA = "rc_dino_rcde_track_r_v124_e1_v2_prejoin_artifact_20260826"
PRODUCER_STATUS = "DINO_RCDE_TRACK_R_V124_E1_V2_FIXED_NATURAL_OPENED_CANARY_PREJOIN_PASS"
RESULT_SCHEMA = "rc_dino_rcde_track_r_v124_e1_v2_independent_validation_20260826"
STATUS = "DINO_RCDE_TRACK_R_V124_E1_V2_FIXED_NATURAL_OPENED_CANARY_PREJOIN_INDEPENDENT_VALIDATION_PASS"
FORBIDDEN = frozenset(
    {
        "target",
        "target_key",
        "target_label",
        "rival",
        "rival_key",
        "role",
        "correctness",
        "score",
        "raw_score",
        "rank",
        "winner",
        "outcome",
        "scientific_result",
        "join",
        "postjoin",
    }
)


class E1V2IndependentError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise E1V2IndependentError(message)


def _no_protected(value: object) -> None:
    if isinstance(value, Mapping):
        require(
            not FORBIDDEN.intersection(map(str, value.keys())),
            "protected key in independent C_COL source",
        )
        for item in value.values():
            _no_protected(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _no_protected(item)


def deep_validate_c_col_independent(
    artifact: Mapping[str, Any],
    *,
    canonical_geometry_path: Path,
) -> dict[str, Any]:
    real = artifact.get("real_bundle")
    wrappers = artifact.get("control_candidates")
    require(
        isinstance(real, IV1.VQueryBundle)
        and isinstance(wrappers, (list, tuple))
        and len(wrappers) == 128,
        "independent C_COL wrapper population drift",
    )
    query_geometry_sha, reference_geometry = IV1.canonical_geometry(
        canonical_geometry_path, real
    )
    by_coordinate = {}
    wrapper_hash_by_position = {}
    for raw_wrapper in wrappers:
        require(
            isinstance(raw_wrapper, Mapping)
            and set(raw_wrapper) == set(CANDIDATE_FIELDS)
            and raw_wrapper.get("schema_version") == CANDIDATE_SCHEMA
            and raw_wrapper.get("logical_sha256")
            == candidate_logical_sha256(raw_wrapper),
            "independent wrapper schema/logical drift",
        )
        position = raw_wrapper.get("candidate_position")
        require(
            type(position) is int
            and 0 <= position < 128
            and position not in wrapper_hash_by_position,
            "independent wrapper position duplicate/drift",
        )
        candidate = raw_wrapper["candidate"]
        key = raw_wrapper["candidate_key"]
        require(
            key in real.locks
            and candidate is real.locks[key].candidate
            and candidate.physical_gallery_row
            == raw_wrapper["candidate_physical_row"]
            and candidate.source_image_sha256
            == raw_wrapper["candidate_reference_source_sha256"],
            "independent wrapper/REAL DINO drift",
        )
        records = raw_wrapper["direction_records"]
        require(
            isinstance(records, Mapping)
            and set(records) == set(IV1.FIXED_DIRECTIONS),
            "independent wrapper direction set drift",
        )
        for direction in IV1.FIXED_DIRECTIONS:
            coordinate = (position, direction)
            require(
                coordinate not in by_coordinate,
                "independent C_COL coordinate duplicated",
            )
            record = records[direction]
            _no_protected(record)
            validate_natural_p_lock_v2(record)
            core = candidate_p_lock_v2_from_record(
                record["core_lock_record"]
            )
            query = record["query"]
            address = record["candidate"]
            direction_index = IV1.FIXED_DIRECTIONS.index(direction)
            require(
                record["direction"] == direction
                and record["record_sha256"]
                == raw_wrapper["direction_record_sha256_sequence"][
                    direction_index
                ]
                and query
                == {
                    "query_id": artifact["query_id"],
                    "historical_query_ordinal": artifact[
                        "historical_query_ordinal"
                    ],
                    "execution_ordinal": 0,
                    "query_source_image_sha256": artifact[
                        "query_source_image_sha256"
                    ],
                    "outer_fold": 2,
                    "inner_heldout_fold": None,
                    "fit_id": "P_OUTER2_OUTER_REFIT",
                    "crossfit_role": "OUTER_HELDOUT_DEPLOYMENT",
                    "track": "difficult",
                }
                and address["candidate_position"] == position
                and address["candidate_key"] == key
                and address["candidate_physical_row"]
                == candidate.physical_gallery_row
                and address["candidate_reference_source_sha256"]
                == candidate.source_image_sha256
                and record["p_checkpoint_sha256"]
                == artifact["p_checkpoint_sha256"]
                and record["p_training_manifest_sha256"]
                == artifact["p_training_manifest_sha256"]
                and core.candidate_key == key
                and core.candidate_physical_row
                == candidate.physical_gallery_row
                and torch.equal(
                    core.query_valid_mask, real.query.valid_patch_mask
                )
                and torch.equal(
                    core.reference_valid_mask, candidate.valid_patch_mask
                )
                and core.geometry_signature.query_geometry_sha256
                == record["query_geometry"]["geometry_sha256"]
                == query_geometry_sha
                and core.geometry_signature.reference_geometry_sha256
                == record["reference_geometry"]["geometry_sha256"]
                == reference_geometry[candidate.physical_gallery_row],
                "independent Natural/Core fold/checkpoint/provenance drift",
            )
            by_coordinate[coordinate] = record["record_sha256"]
            del core
        wrapper_hash_by_position[position] = raw_wrapper["logical_sha256"]
        gc.collect()
    expected_coordinates = tuple(
        (position, direction)
        for position in range(128)
        for direction in IV1.FIXED_DIRECTIONS
    )
    require(
        tuple(sorted(by_coordinate, key=lambda item: (item[0], IV1.FIXED_DIRECTIONS.index(item[1]))))
        == expected_coordinates
        and len(by_coordinate) == 256
        and [by_coordinate[item] for item in expected_coordinates]
        == artifact["control_record_sha256_sequence"]
        and len(set(by_coordinate.values())) == 256
        and [wrapper_hash_by_position[index] for index in range(128)]
        == [row["logical_sha256"] for row in wrappers],
        "independent C_COL 128/256 loss/duplicate/reorder drift",
    )
    result = {
        "schema_version": "rc_dino_rcde_track_r_v124_e1_v2_c_col_independent_deep_validation_20260826",
        "status": "DINO_RCDE_TRACK_R_V124_E1_V2_C_COL_128_256_INDEPENDENT_DEEP_VALIDATION_PASS",
        "wrapper_count": 128,
        "record_count": 256,
        "coordinate_sequence_sha256": IV1.canonical_sha256(
            [list(item) for item in expected_coordinates]
        ),
        "record_sequence_sha256": IV1.canonical_sha256(
            [by_coordinate[item] for item in expected_coordinates]
        ),
        "missing_count": 0,
        "duplicate_count": 0,
        "reordered_count": 0,
        "foreign_count": 0,
        "protected_key_count": 0,
    }
    result["logical_sha256"] = logical_sha256(result)
    return result


def _bound(authority: Mapping[str, Any], name: str) -> Path:
    row = authority["bindings"][name]
    raw = Path(row["path"])
    return (raw if raw.is_absolute() else ROOT / raw).resolve()


def run(
    *,
    authority_path: Path,
    authority_sha256: str,
    crash_after: str | None = None,
) -> Mapping[str, Any]:
    require(Path.cwd().resolve() == ROOT.resolve(), "E1 V2 validator cwd drift")
    authority = read_authority(authority_path, authority_sha256)
    if PUBLIC_DIR.exists() or PUBLIC_DIR.is_symlink():
        _, reused = validate_family(
            PUBLIC_DIR,
            authority_sha256=authority_sha256,
            family_role="E1_V2_VALIDATION_FAMILY",
            producer=False,
        )
        require(
            reused.get("schema_version") == RESULT_SCHEMA
            and reused.get("status") == STATUS
            and reused.get("authority_sha256") == authority_sha256
            and reused.get("logical_sha256") == logical_sha256(reused),
            "reused E1 V2 independent result drift",
        )
        return reused
    observed, producer_result = validate_family(
        PRODUCER_DIR,
        authority_sha256=authority_sha256,
        family_role="E1_V2_PRODUCER_FAMILY",
        producer=True,
    )
    require(
        observed is not None
        and observed.get("schema_version") == ARTIFACT_SCHEMA
        and observed.get("status") == PRODUCER_STATUS
        and observed.get("authority_sha256") == authority_sha256
        and observed.get("logical_sha256")
        == IV1.canonical_sha256(
            {
                key: item
                for key, item in observed.items()
                if key not in {"logical_sha256", "tensor_payloads"}
            }
        )
        and producer_result.get("status") == PRODUCER_STATUS
        and producer_result.get("authority_sha256") == authority_sha256
        and producer_result.get("logical_sha256")
        == logical_sha256(producer_result),
        "E1 V2 producer committed family semantic drift",
    )
    phase_a = torch.load(
        _bound(authority, "phase_a_artifact"),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    require(isinstance(phase_a, Mapping), "independent Phase-A root drift")
    deep = deep_validate_c_col_independent(
        phase_a,
        canonical_geometry_path=_bound(
            authority, "canonical_geometry_payload"
        ),
    )
    real = phase_a["real_bundle"]
    pair_row, pair = IV1.execution0_pair(
        _bound(authority, "role_free_pair_manifest")
    )
    query_geometry_sha, reference_geometry = IV1.canonical_geometry(
        _bound(authority, "canonical_geometry_payload"), real
    )
    c_col = IV1.reconstruct_c_col(
        phase_a,
        pair,
        canonical_query_geometry_sha256=query_geometry_sha,
        canonical_reference_geometry_by_row=reference_geometry,
    )
    c_col_view = IV1.c_col_axis_view(phase_a)
    c_dino, c_dino_receipt = IV1.reconstruct_c_dino(
        real,
        pair,
        IV1.identities(_bound(authority, "gallery_cache"), real),
    )
    model, state = IV1.model(_bound(authority, "fold2_v_checkpoint"))
    require(
        state == authority["v_checkpoint_state_sha256"]
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 1,
        "independent E1 V2 checkpoint/GPU drift",
    )
    with torch.no_grad():
        real_all = IV1.all_patch_only(model, real.query, real.locks, pair)
        c_dino_all = IV1.all_patch_only(model, real.query, c_dino, pair)
        c_col_all = IV1.all_patch_only(
            model, real.query, c_col_view.locks, pair
        )
    gpu = torch.device("cuda:0")
    model.to(gpu)
    query_gpu = IV1.query_to_device(real.query, gpu)
    real_gpu = IV1.move_locks_to_device(
        {key: real.locks[key] for key in pair}, gpu
    )
    c_dino_gpu = IV1.move_c_dino_locks_to_device(c_dino, gpu)
    c_col_gpu = IV1.move_locks_to_device(c_col, gpu)
    with torch.no_grad():
        real_regional = IV1.ordinary_regional(
            model, query_gpu, real_gpu, pair
        )
        c_dino_regional, transports = IV1.c_dino_regional(
            model, query_gpu, c_dino_gpu, pair
        )
        c_col_regional = IV1.ordinary_regional(
            model, query_gpu, c_col_gpu, pair
        )
    evidence = {
        "REAL": IV1.assemble(model, pair, real_all, real_regional),
        "C_DINO_V": IV1.assemble(
            model, pair, c_dino_all, c_dino_regional
        ),
        "C_COL_P": IV1.assemble(model, pair, c_col_all, c_col_regional),
    }
    require(IV1.state_dict_sha256(model) == state, "independent V2 model mutation")
    replay_records = IV1.records(evidence, transports)
    replay_tensors, replay_tensor_hashes = IV1.tensor_payloads(evidence)
    replay_phase_b = IV1.phase_b_receipt(
        model_value=model,
        real=real,
        c_col=c_col_view,
        pair=pair,
        real_evidence=evidence["REAL"],
        c_col_evidence=evidence["C_COL_P"],
    )
    require(
        observed.get("records") == replay_records
        and observed.get("tensor_payload_hashes") == replay_tensor_hashes
        and IV1._tensor_payloads_equal(
            observed["tensor_payloads"], replay_tensors
        )
        and observed.get("c_dino_receipt") == dict(c_dino_receipt)
        and observed.get("phase_b_invariance_receipt") == replay_phase_b
        and observed.get("c_col_deep_validation", {}).get("record_count")
        == deep["record_count"]
        and observed.get("pair_member_keys") == list(pair)
        and observed.get("pair_address_sha256_sequence")
        == [row["address_sha256"] for row in pair_row["members"]],
        "E1 V2 producer/independent deep replay drift",
    )
    decode_calls = sum(
        term["decoder_candidate_call_count"]
        for record in replay_records
        for term in record["direction_terms"]
    )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": STATUS,
        "authority_sha256": authority_sha256,
        "claim_level": "FIXED_NATURAL_OPENED_CANARY_TARGET_FREE_PREJOIN_ONLY",
        "producer_family_commit_sha256": file_sha256(
            PRODUCER_DIR / "family_commit.json"
        ),
        "execution_ordinal": 0,
        "query_id": "DIFFICULT-0000",
        "candidate_axis_sha256": CANDIDATE_AXIS_SHA256,
        "pair_sha256": PAIR_SHA256,
        "pair_arm_record_count": 9,
        "directional_term_count": 36,
        "c_col_deep_validated_wrapper_count": 128,
        "c_col_deep_validated_record_count": 256,
        "c_col_runtime_materialized_candidate_count": 2,
        "independent_checked_record_count": 9,
        "independent_checked_directional_term_count": 36,
        "unchecked_record_count": 0,
        "producer_independent_exact_match": True,
        "v_decode_candidate_call_count": decode_calls,
        "target_rival_read_count": 0,
        "postjoin_read_count": 0,
        "scientific_reduction_count": 0,
        "automatic_submit_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "validation_pass": True,
    }
    result["logical_sha256"] = logical_sha256(result)
    published, _ = publish_validation_family(
        PUBLIC_DIR,
        authority_sha256=authority_sha256,
        result=result,
        crash_after=crash_after,
    )
    return published


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, default=AUTHORITY)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--crash-after")
    args = parser.parse_args()
    result = run(
        authority_path=args.authority,
        authority_sha256=args.authority_sha256,
        crash_after=args.crash_after,
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
