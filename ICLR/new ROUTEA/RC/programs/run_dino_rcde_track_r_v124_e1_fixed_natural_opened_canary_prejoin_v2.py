#!/usr/bin/env python3
"""V124-E1 V2 producer with deep C_COL validation and atomic publication."""

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

import run_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v1 as V1  # noqa: E402
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
    publish_producer_family,
    validate_family,
)
from rc_aslo_xf.dino_rcde_track_r_v124_e1_authority_v2 import (  # noqa: E402
    CANDIDATE_AXIS_SHA256,
    PAIR_SHA256,
    read_authority,
)


AUTHORITY = ROOT / "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_authority_v2_20260826.json"
PUBLIC_DIR = ROOT / "results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v2_committed"
ARTIFACT_SCHEMA = "rc_dino_rcde_track_r_v124_e1_v2_prejoin_artifact_20260826"
RESULT_SCHEMA = "rc_dino_rcde_track_r_v124_e1_v2_producer_result_20260826"
STATUS = "DINO_RCDE_TRACK_R_V124_E1_V2_FIXED_NATURAL_OPENED_CANARY_PREJOIN_PASS"
FORBIDDEN_KEYS = frozenset(
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
        "postjoin",
        "join",
    }
)


class E1V2ProducerError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise E1V2ProducerError(message)


def _reject_protected_keys(value: object, path: str = "root") -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_KEYS.intersection(map(str, value.keys()))
        require(not forbidden, f"protected C_COL key at {path}: {sorted(forbidden)}")
        for key, item in value.items():
            _reject_protected_keys(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_protected_keys(item, f"{path}[{index}]")


def deep_validate_c_col_population(
    artifact: Mapping[str, Any],
    *,
    canonical_geometry_path: Path,
) -> dict[str, Any]:
    real = artifact.get("real_bundle")
    wrappers = artifact.get("control_candidates")
    require(
        isinstance(real, V1.VQueryBundle)
        and isinstance(wrappers, (list, tuple))
        and len(wrappers) == 128
        and artifact.get("record_count") == 256
        and artifact.get("candidate_axis_sha256") == CANDIDATE_AXIS_SHA256,
        "C_COL deep-validation artifact population drift",
    )
    query_geometry_sha, reference_geometry_by_row = V1.canonical_geometry_axis(
        canonical_geometry_path, real
    )
    expected_sequence = []
    wrapper_hashes = []
    coordinates = []
    query_structure_hashes = set()
    for position, wrapper in enumerate(wrappers):
        require(
            isinstance(wrapper, Mapping)
            and set(wrapper) == set(CANDIDATE_FIELDS)
            and wrapper.get("schema_version") == CANDIDATE_SCHEMA
            and wrapper.get("candidate_position") == position
            and wrapper.get("logical_sha256")
            == candidate_logical_sha256(wrapper),
            "C_COL wrapper schema/order/logical drift",
        )
        candidate = wrapper["candidate"]
        key = wrapper["candidate_key"]
        require(
            key in real.locks
            and candidate is real.locks[key].candidate
            and candidate.candidate_key == key
            and candidate.physical_gallery_row
            == wrapper["candidate_physical_row"]
            and candidate.source_image_sha256
            == wrapper["candidate_reference_source_sha256"],
            "C_COL wrapper/REAL DINO candidate drift",
        )
        records = wrapper.get("direction_records")
        require(
            isinstance(records, Mapping)
            and tuple(records) == tuple(V1.FIXED_DIRECTIONS)
            and len(wrapper["direction_record_sha256_sequence"]) == 2,
            "C_COL wrapper direction field/order drift",
        )
        for direction_index, direction in enumerate(V1.FIXED_DIRECTIONS):
            record = records[direction]
            _reject_protected_keys(record)
            validate_natural_p_lock_v2(record)
            core = candidate_p_lock_v2_from_record(
                record["core_lock_record"]
            )
            query = record["query"]
            address = record["candidate"]
            require(
                record["direction"] == direction
                and record["record_sha256"]
                == wrapper["direction_record_sha256_sequence"][
                    direction_index
                ]
                and query["query_id"] == artifact["query_id"]
                and query["historical_query_ordinal"]
                == artifact["historical_query_ordinal"]
                and query["execution_ordinal"] == 0
                and query["query_source_image_sha256"]
                == artifact["query_source_image_sha256"]
                and query["outer_fold"] == 2
                and query["inner_heldout_fold"] is None
                and query["fit_id"] == "P_OUTER2_OUTER_REFIT"
                and query["crossfit_role"]
                == "OUTER_HELDOUT_DEPLOYMENT"
                and query["track"] == "difficult"
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
                == reference_geometry_by_row[candidate.physical_gallery_row],
                "C_COL Natural/Core fold/checkpoint/address/provenance drift",
            )
            coordinates.append([position, direction])
            expected_sequence.append(record["record_sha256"])
            query_structure_hashes.add(core.query_structure_sha256)
            del core
        wrapper_hashes.append(wrapper["logical_sha256"])
        gc.collect()
    expected_coordinates = [
        [position, direction]
        for position in range(128)
        for direction in V1.FIXED_DIRECTIONS
    ]
    require(
        coordinates == expected_coordinates
        and len(set(tuple(item) for item in coordinates)) == 256
        and expected_sequence == artifact["control_record_sha256_sequence"]
        and len(set(expected_sequence)) == 256
        and V1.canonical_sha256(wrapper_hashes)
        == artifact["control_candidate_wrapper_sequence_sha256"],
        "C_COL 128/256 loss/duplicate/reorder/global sequence drift",
    )
    result = {
        "schema_version": "rc_dino_rcde_track_r_v124_e1_v2_c_col_deep_validation_20260826",
        "status": "DINO_RCDE_TRACK_R_V124_E1_V2_C_COL_128_256_DEEP_VALIDATION_PASS",
        "wrapper_count": 128,
        "record_count": 256,
        "coordinate_sequence_sha256": V1.canonical_sha256(coordinates),
        "record_sequence_sha256": V1.canonical_sha256(expected_sequence),
        "wrapper_sequence_sha256": V1.canonical_sha256(wrapper_hashes),
        "query_structure_sha256_count": len(query_structure_hashes),
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


def _reuse_if_committed(
    authority_sha256: str,
) -> Mapping[str, Any] | None:
    if not PUBLIC_DIR.exists() and not PUBLIC_DIR.is_symlink():
        return None
    artifact, result = validate_family(
        PUBLIC_DIR,
        authority_sha256=authority_sha256,
        family_role="E1_V2_PRODUCER_FAMILY",
        producer=True,
    )
    require(
        artifact is not None
        and artifact.get("schema_version") == ARTIFACT_SCHEMA
        and artifact.get("status") == STATUS
        and artifact.get("authority_sha256") == authority_sha256
        and artifact.get("logical_sha256")
        == V1.canonical_sha256(
            {
                key: item
                for key, item in artifact.items()
                if key not in {"logical_sha256", "tensor_payloads"}
            }
        )
        and result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") == STATUS
        and result.get("authority_sha256") == authority_sha256
        and result.get("logical_sha256") == logical_sha256(result),
        "committed E1 V2 family semantic drift",
    )
    return result


def run(
    *,
    authority_path: Path,
    authority_sha256: str,
    crash_after: str | None = None,
) -> Mapping[str, Any]:
    require(Path.cwd().resolve() == ROOT.resolve(), "E1 V2 cwd drift")
    authority = read_authority(authority_path, authority_sha256)
    reused = _reuse_if_committed(authority_sha256)
    if reused is not None:
        return reused
    phase_a_path = _bound(authority, "phase_a_artifact")
    phase_a = torch.load(
        phase_a_path, map_location="cpu", weights_only=False, mmap=True
    )
    require(isinstance(phase_a, Mapping), "Phase-A artifact root drift")
    deep = deep_validate_c_col_population(
        phase_a,
        canonical_geometry_path=_bound(
            authority, "canonical_geometry_payload"
        ),
    )
    real = phase_a["real_bundle"]
    pair_row, pair = V1.execution0_pair(
        _bound(authority, "role_free_pair_manifest")
    )
    query_geometry_sha, reference_geometry_by_row = V1.canonical_geometry_axis(
        _bound(authority, "canonical_geometry_payload"), real
    )
    c_col_pair = V1.build_c_col_pair_locks(
        phase_a,
        pair,
        canonical_query_geometry_sha256=query_geometry_sha,
        canonical_reference_geometry_by_row=reference_geometry_by_row,
    )
    c_col_view = V1.build_c_col_axis_view(phase_a)
    identities = V1.corrected_identity_by_row(
        _bound(authority, "gallery_cache"), real
    )
    c_dino, c_dino_receipt = (
        V1.extract_full_c128_c_dino_pair_from_runtime_locks(
            real.locks,
            pair[0],
            pair[1],
            namespace=V1.C_DINO_NAMESPACE,
            query_id=real.query_id,
            corrected_identity_by_physical_row=identities,
        )
    )
    model, state_sha = V1.load_model(_bound(authority, "fold2_v_checkpoint"))
    require(
        state_sha == authority["v_checkpoint_state_sha256"]
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 1,
        "E1 V2 checkpoint/GPU drift",
    )
    with torch.no_grad():
        real_all = V1._all_patch_only(model, real.query, real.locks, pair)
        c_dino_all = V1._all_patch_only(model, real.query, c_dino, pair)
        c_col_all = V1._all_patch_only(
            model, real.query, c_col_view.locks, pair
        )
    gpu = torch.device("cuda:0")
    model.to(gpu)
    query_gpu = V1._query_to_device(real.query, gpu)
    real_gpu = V1.move_locks_to_device(
        {key: real.locks[key] for key in pair}, gpu
    )
    c_dino_gpu = V1.move_full_c128_c_dino_pair_to_device(
        c_dino, gpu, expected_pair_member_keys=pair
    )
    c_col_gpu = V1.move_locks_to_device(c_col_pair, gpu)
    with torch.no_grad():
        real_regional = V1._ordinary_regional_arms(
            model, query_gpu, real_gpu, pair
        )
        c_dino_regional, transports = V1._c_dino_regional_arms(
            model, query_gpu, c_dino_gpu, pair
        )
        c_col_regional = V1._ordinary_regional_arms(
            model, query_gpu, c_col_gpu, pair
        )
    evidence = {
        "REAL": V1._assemble_three_arm(
            model, pair, real_all, real_regional
        ),
        "C_DINO_V": V1._assemble_three_arm(
            model, pair, c_dino_all, c_dino_regional
        ),
        "C_COL_P": V1._assemble_three_arm(
            model, pair, c_col_all, c_col_regional
        ),
    }
    require(V1.state_dict_sha256(model) == state_sha, "E1 V2 model mutation")
    records = V1.evidence_records(evidence, c_dino_transport=transports)
    tensors, tensor_hashes = V1.evidence_tensor_payloads(evidence)
    decode_calls = sum(
        term["decoder_candidate_call_count"]
        for record in records
        for term in record["direction_terms"]
    )
    phase_b_receipt = V1.phase_b_receipt_from_evidence(
        model=model,
        real_bundle=real,
        c_col_bundle=c_col_view,
        pair=pair,
        real_evidence=evidence["REAL"],
        c_col_evidence=evidence["C_COL_P"],
    )
    bound_phase_b = json.loads(
        _bound(authority, "phase_b_result").read_text(encoding="ascii")
    )
    require(
        all(
            bound_phase_b[key] == phase_b_receipt[key]
            for key in V1.PHASE_B_FIELDS
        ),
        "E1 V2 Phase-B byte replay drift",
    )
    artifact: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA,
        "status": STATUS,
        "authority_sha256": authority_sha256,
        "claim_level": "FIXED_NATURAL_OPENED_CANARY_TARGET_FREE_PREJOIN_ONLY",
        "execution_ordinal": 0,
        "query_id": "DIFFICULT-0000",
        "source_fold": 2,
        "candidate_count": 128,
        "candidate_axis_sha256": CANDIDATE_AXIS_SHA256,
        "pair_sha256": PAIR_SHA256,
        "pair_member_keys": list(pair),
        "pair_address_sha256_sequence": [
            row["address_sha256"] for row in pair_row["members"]
        ],
        "opened594_role": "DIAGNOSTIC_ONLY_NOT_CONFIRMATION",
        "family_sequence": list(V1.FAMILY_SEQUENCE),
        "arm_sequence": list(V1.ARM_SEQUENCE),
        "pair_arm_record_count": 9,
        "directional_term_count": 36,
        "records": records,
        "record_sha256_sequence": [row["record_sha256"] for row in records],
        "tensor_payloads": tensors,
        "tensor_payload_hashes": tensor_hashes,
        "c_col_deep_validation": deep,
        "c_dino_receipt": dict(c_dino_receipt),
        "phase_b_invariance_receipt": phase_b_receipt,
        "phase_a_artifact_sha256": file_sha256(phase_a_path),
        "v_checkpoint_state_sha256": state_sha,
        "device_schedule": authority["device_schedule"],
        "v_decode_candidate_call_count": decode_calls,
        "target_rival_read_count": 0,
        "postjoin_read_count": 0,
        "scientific_reduction_count": 0,
        "automatic_submit_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    artifact["logical_sha256"] = V1.canonical_sha256(
        {
            key: item
            for key, item in artifact.items()
            if key not in {"logical_sha256", "tensor_payloads"}
        }
    )

    public_artifact = PUBLIC_DIR / "prejoin_artifact.pt"

    def result_builder(staged_artifact: Path) -> Mapping[str, Any]:
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA,
            "status": STATUS,
            "authority_sha256": authority_sha256,
            "claim_level": "FIXED_NATURAL_OPENED_CANARY_TARGET_FREE_PREJOIN_ONLY",
            "artifact_path": public_artifact.relative_to(ROOT).as_posix(),
            "artifact_sha256": file_sha256(staged_artifact),
            "artifact_logical_sha256": artifact["logical_sha256"],
            "execution_ordinal": 0,
            "query_id": "DIFFICULT-0000",
            "candidate_axis_sha256": CANDIDATE_AXIS_SHA256,
            "pair_sha256": PAIR_SHA256,
            "pair_arm_record_count": 9,
            "directional_term_count": 36,
            "c_col_deep_validated_wrapper_count": 128,
            "c_col_deep_validated_record_count": 256,
            "c_col_runtime_materialized_candidate_count": 2,
            "v_decode_candidate_call_count": decode_calls,
            "target_rival_read_count": 0,
            "postjoin_read_count": 0,
            "scientific_reduction_count": 0,
            "automatic_submit_count": 0,
            "scientific_GO_or_NO_GO": None,
            "automatic_stage_advance": False,
            "next_authorized_stage": None,
        }
        result["logical_sha256"] = logical_sha256(result)
        return result

    _, result, _ = publish_producer_family(
        PUBLIC_DIR,
        authority_sha256=authority_sha256,
        artifact=artifact,
        result_builder=result_builder,
        crash_after=crash_after,
    )
    return result


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
