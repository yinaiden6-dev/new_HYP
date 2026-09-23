#!/usr/bin/env python3
"""Produce the V124-E1 fixed natural opened-canary target-free prejoin.

The executable entry point accepts only a later independently reviewed E1
execution authority.  Pure construction/serialization helpers are exposed for
poison and synthetic tests; this file never reads target/rival or joined data.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.dino_rcde_cw1_multitile_superregion_v2 import (  # noqa: E402
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
)
from rc_aslo_xf.dino_rcde_cw1_multitile_vdecode_v1 import (  # noqa: E402
    BINDING_REAL,
    FIXED_DIRECTIONS,
    ThreeArmFixedDenominatorEvidenceV1,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1 import (  # noqa: E402
    GeometryNamespaceBindingV1,
    outer_refit_head_spec,
    project_natural_v2_candidate_to_v1_runtime,
)
from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (  # noqa: E402
    VQueryBundle,
    move_locks_to_device,
    state_dict_sha256,
    tensor_sha256,
)
from rc_aslo_xf.dino_rcde_track_r_full_c128_c_dino_v1 import (  # noqa: E402
    extract_full_c128_c_dino_pair_from_runtime_locks,
    move_full_c128_c_dino_pair_to_device,
)
from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2  # noqa: E402
from rc_aslo_xf.gallery_identity_repair import build_identity_map  # noqa: E402
from rc_aslo_xf import dino_rcde_sr0_mt_v_runtime_v2 as runtime_v2  # noqa: E402
from rc_aslo_xf import dino_rcde_sr0_mt_controls_v1 as controls_v1  # noqa: E402
from rc_aslo_xf import dino_rcde_track_r_control_decoder_v2 as decoder_v2  # noqa: E402
from rc_aslo_xf import dino_rcde_track_r_phase_b_all_patch_v1 as phase_b  # noqa: E402


AUTHORITY = ROOT / "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_authority_v1_20260826.json"
AUTHORITY_SCHEMA = "rc_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_authority_v1_20260826"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_V124_E1_FIXED_NATURAL_OPENED_CANARY_PREJOIN_EXECUTION_AUTHORIZED"
OUTPUT_ROOT = ROOT / "results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v1"
ARTIFACT = OUTPUT_ROOT / "prejoin_artifact.pt"
RESULT = OUTPUT_ROOT / "producer_result.json"
ARTIFACT_SCHEMA = "rc_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_artifact_v1_20260826"
RESULT_SCHEMA = "rc_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_result_v1_20260826"
STATUS = "DINO_RCDE_TRACK_R_V124_E1_FIXED_NATURAL_OPENED_CANARY_PREJOIN_PASS"
ABORT = "DINO_RCDE_TRACK_R_V124_E1_ENGINEERING_ABORT"
PAIR_SHA256 = "6c761ca6e98b5191f3b6e526d1f0d4e3b5e0d7f31e839356d79a0340dba3a7c5"
CANDIDATE_AXIS_SHA256 = "bfdd876c39964380302e0cc73507def00ca767ceec6b60886d51d775c709ade6"
FAMILY_SEQUENCE = ("REAL", "C_DINO_V", "C_COL_P")
ARM_SEQUENCE = (
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
)
C_DINO_NAMESPACE = "DINO_RCDE_TRACK_R_V124_E1_C_DINO_V1"
PHASE_B_FIELDS = (
    "real_request_sha256",
    "control_request_sha256",
    "real_patch_tensor_sha256",
    "control_patch_tensor_sha256",
    "real_direction_sequence_sha256",
    "control_direction_sequence_sha256",
    "real_pair_score_sha256",
    "control_pair_score_sha256",
)


class V124E1ProducerError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise V124E1ProducerError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def _bound_path(authority: Mapping[str, Any], name: str) -> Path:
    row = authority.get("bindings", {}).get(name)
    require(isinstance(row, Mapping), f"authority binding absent: {name}")
    raw = Path(str(row["path"]))
    path = (raw if raw.is_absolute() else ROOT / raw).resolve()
    require(path.is_file() and not path.is_symlink(), f"bound file absent/unsafe: {name}")
    require(file_sha256(path) == row.get("sha256"), f"bound file hash drift: {name}")
    return path


def read_authority(path: Path, expected_sha256: str) -> dict[str, Any]:
    authority_path = path.resolve()
    require(authority_path.is_file() and not authority_path.is_symlink(), "E1 authority absent/unsafe")
    require(file_sha256(authority_path) == expected_sha256, "E1 authority physical hash drift")
    value = json.loads(authority_path.read_text(encoding="ascii"))
    require(
        value.get("schema_version") == AUTHORITY_SCHEMA
        and value.get("status") == AUTHORITY_STATUS
        and value.get("logical_sha256") == logical_sha256(value)
        and value.get("independent_review_status") == "PASS"
        and value.get("execution_ordinal") == 0
        and value.get("query_id") == "DIFFICULT-0000"
        and value.get("source_fold") == 2
        and value.get("candidate_count") == 128
        and value.get("candidate_axis_sha256") == CANDIDATE_AXIS_SHA256
        and value.get("pair_sha256") == PAIR_SHA256
        and tuple(value.get("family_sequence", ())) == FAMILY_SEQUENCE
        and tuple(value.get("arm_sequence", ())) == ARM_SEQUENCE
        and value.get("expected_pair_arm_record_count") == 9
        and value.get("expected_directional_term_count") == 36
        and value.get("model_load_authorized") is True
        and value.get("model_forward_authorized") is True
        and value.get("p_training_authorized") is False
        and value.get("v_training_authorized") is False
        and value.get("device_schedule")
        == {
            "all_patch_device": "cpu",
            "regional_device": "cuda:0",
            "model_load_count": 1,
            "model_device_transition_count": 1,
        }
        and value.get("resource_contract", {}).get("allowed_partitions")
        == ["accelerated"]
        and value.get("resource_contract", {}).get("gpu_count") == 1
        and value.get("resource_contract", {}).get("cpus_per_task") == 8
        and value.get("resource_contract", {}).get("memory_megabytes")
        == 128000
        and value.get("resource_contract", {}).get("walltime_seconds")
        == 7200
        and value.get("target_rival_join_authorized") is False
        and value.get("postjoin_authorized") is False
        and value.get("scientific_reduction_authorized") is False
        and value.get("automatic_submit_authorized") is False
        and value.get("automatic_stage_advance") is False
        and value.get("next_authorized_stage") is None,
        "E1 authority envelope/capability drift",
    )
    for name in value["bindings"]:
        bound = _bound_path(value, name)
        require(
            not {"c8", "s8", "opened", "sealed", "track-h", "rgh_xf", "h0", "gx"}.intersection(
                part.lower() for part in bound.parts
            ),
            f"protected/isolated authority binding forbidden: {name}",
        )
    return value


def execution0_pair(path: Path) -> tuple[Mapping[str, Any], tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="ascii"))
    rows = [
        row
        for row in payload.get("pair_manifest", {}).get("records", [])
        if row.get("execution_ordinal") == 0
    ]
    require(len(rows) == 1, "execution-0 anonymous pair absent/duplicated")
    pair = rows[0]
    members = pair.get("members")
    require(
        pair.get("query_id") == "DIFFICULT-0000"
        and pair.get("natural_axis_count") == 128
        and pair.get("candidate_axis_sha256") == CANDIDATE_AXIS_SHA256
        and pair.get("pair_sha256") == PAIR_SHA256
        and isinstance(members, list)
        and len(members) == 2,
        "execution-0 anonymous pair drift",
    )
    keys = tuple(str(item["candidate_key"]) for item in members)
    require(len(set(keys)) == 2, "anonymous pair member drift")
    return pair, keys  # type: ignore[return-value]


def _c_col_wrapper_axis(
    artifact: Mapping[str, Any],
) -> tuple[VQueryBundle, Mapping[str, Mapping[str, Any]]]:
    """Validate all 128/256 sealed records without projecting large locks."""

    real = artifact.get("real_bundle")
    wrappers = artifact.get("control_candidates")
    require(isinstance(real, VQueryBundle), "Phase-A REAL bundle type drift")
    require(
        artifact.get("candidate_count") == 128
        and artifact.get("record_count") == 256
        and artifact.get("dino_axis_byte_identical") is True
        and isinstance(wrappers, (list, tuple))
        and len(wrappers) == 128,
        "Phase-A C_COL envelope/population drift",
    )
    by_key = {}
    for position, wrapper in enumerate(wrappers):
        require(
            isinstance(wrapper, Mapping)
            and wrapper.get("candidate_position") == position
            and set(wrapper.get("direction_records", {})) == set(FIXED_DIRECTIONS),
            "ordered C_COL wrapper/direction drift",
        )
        candidate = wrapper["candidate"]
        records = wrapper["direction_records"]
        require(
            candidate.candidate_key == wrapper.get("candidate_key")
            and candidate.physical_gallery_row
            == wrapper.get("candidate_physical_row")
            and candidate.candidate_key in real.locks
            and candidate is real.locks[candidate.candidate_key].candidate
            and all(
                records[direction].get("record_sha256")
                == wrapper["direction_record_sha256_sequence"][index]
                for index, direction in enumerate(FIXED_DIRECTIONS)
            ),
            "C_COL wrapper/REAL DINO/record receipt drift",
        )
        by_key[candidate.candidate_key] = wrapper
    require(len(by_key) == 128, "C_COL wrapper key population drift")
    return real, by_key


def canonical_geometry_axis(
    path: Path, real: VQueryBundle
) -> tuple[str, Mapping[int, str]]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(isinstance(payload, Mapping), "canonical geometry payload drift")
    queries = [
        row
        for row in payload.get("query_records", [])
        if row.get("execution_ordinal") == 0
    ]
    require(
        len(queries) == 1
        and queries[0].get("query_id") == real.query_id
        and queries[0].get("source_image_sha256")
        == real.query.source_image_sha256
        and queries[0].get("candidate_axis_sha256")
        == real.candidate_axis_sha256,
        "canonical execution-0 query geometry drift",
    )
    references = {
        row["physical_row"]: row
        for row in payload.get("reference_records", [])
    }
    output = {}
    for lock in real.locks.values():
        candidate = lock.candidate
        row = references.get(candidate.physical_gallery_row)
        require(
            isinstance(row, Mapping)
            and row.get("source_image_sha256")
            == candidate.source_image_sha256
            and tuple(row["dino_geometry"]["grid_shape"])
            == candidate.grid_shape,
            "canonical candidate geometry drift",
        )
        output[candidate.physical_gallery_row] = row["dino_geometry"][
            "geometry_sha256"
        ]
    require(len(output) == 128, "canonical geometry C128 count drift")
    return queries[0]["dino_geometry"]["geometry_sha256"], output


def build_c_col_pair_locks(
    artifact: Mapping[str, Any],
    pair: tuple[str, str],
    *,
    canonical_query_geometry_sha256: str,
    canonical_reference_geometry_by_row: Mapping[int, str],
) -> Mapping[str, object]:
    """Project only the scored pair after validating all 256 metadata rows."""

    real, wrappers = _c_col_wrapper_axis(artifact)
    require(set(pair).issubset(wrappers), "C_COL pair outside wrapper axis")
    locks = {}
    spec = outer_refit_head_spec(source_fold=2)
    for key in pair:
        wrapper = wrappers[key]
        candidate = wrapper["candidate"]
        records = wrapper["direction_records"]
        first = records[FIXED_DIRECTIONS[0]]
        query_geometry = first["query_geometry"]
        reference_geometry = first["reference_geometry"]
        reference_sha = canonical_reference_geometry_by_row.get(
            candidate.physical_gallery_row
        )
        require(
            query_geometry["geometry_sha256"]
            == canonical_query_geometry_sha256
            and reference_geometry["geometry_sha256"] == reference_sha,
            "C_COL record/canonical geometry provenance drift",
        )
        projection = project_natural_v2_candidate_to_v1_runtime(
            query=real.query,
            candidate=candidate,
            direction_records=records,
            spec=spec,
            query_geometry_binding=GeometryNamespaceBindingV1(
                source_image_sha256=real.query.source_image_sha256,
                grid_shape=real.query.grid_shape,
                canonical_geometry_sha256=canonical_query_geometry_sha256,
                cache_geometry_record_sha256=real.query.geometry_record_sha256,
            ),
            reference_geometry_binding=GeometryNamespaceBindingV1(
                source_image_sha256=candidate.source_image_sha256,
                grid_shape=candidate.grid_shape,
                canonical_geometry_sha256=str(reference_sha),
                cache_geometry_record_sha256=candidate.geometry_record_sha256,
            ),
        )
        require(key not in locks, "duplicate C_COL runtime candidate")
        locks[key] = projection.runtime_lock
    require(len(locks) == 2, "C_COL runtime materialization is not pair-only")
    return locks


def build_c_col_axis_view(artifact: Mapping[str, Any]) -> VQueryBundle:
    """Complete candidate-only DINO view; P access is poisoned."""

    real, wrappers = _c_col_wrapper_axis(artifact)
    locks = {
        key: _CandidateOnlyPoison(wrapper["candidate"])
        for key, wrapper in wrappers.items()
    }
    return VQueryBundle(
        real.query_id,
        real.execution_ordinal,
        real.outer_fold,
        real.query,
        locks,  # type: ignore[arg-type]
        real.candidate_axis_sha256,
    )


def corrected_identity_by_row(
    gallery_cache_path: Path,
    bundle: VQueryBundle,
) -> Mapping[int, str]:
    gallery = torch.load(
        gallery_cache_path, map_location="cpu", weights_only=True, mmap=True
    )
    legacy = gallery.get("setids") if isinstance(gallery, Mapping) else None
    require(
        isinstance(legacy, (list, tuple)) and len(legacy) == 5413,
        "gallery corrected-identity source drift",
    )
    labels = build_identity_map(tuple(map(str, legacy))).labels
    rows = {
        lock.candidate.physical_gallery_row: labels[
            lock.candidate.physical_gallery_row
        ]
        for lock in bundle.locks.values()
    }
    require(len(rows) == 128, "C128 corrected-identity population drift")
    return rows


def load_model(path: Path) -> tuple[DINO_RCDE_V1_2, str]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(
        isinstance(payload, Mapping)
        and payload.get("schema_version") == "rc_dino_rcde_sr0_mt_v_checkpoint_v1"
        and payload.get("status") == "DINO_RCDE_TRACK_R_RELATIVE_CHECKPOINT_COMPLETE"
        and payload.get("outer_fold") == 2
        and payload.get("update") == 2048
        and payload.get("relative_pair_loss_only") is True,
        "historical fold-2 V checkpoint drift",
    )
    model = DINO_RCDE_V1_2()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    state = state_dict_sha256(model)
    require(state == payload.get("final_state_sha256"), "fold-2 V state drift")
    return model, state


def _query_to_device(query: object, device: torch.device) -> object:
    return replace(
        query,
        layers=query.layers.to(device),
        valid_patch_mask=query.valid_patch_mask.to(device),
    )


class _CandidateOnlyPoison:
    def __init__(self, candidate: object) -> None:
        self.candidate = candidate

    @property
    def direction_locks(self):
        raise V124E1ProducerError("ALL_PATCH attempted P/regional access")


def _all_patch_only(
    model: torch.nn.Module,
    query: object,
    bindings: Mapping[str, object],
    pair: tuple[str, str],
):
    poison = {
        key: _CandidateOnlyPoison(bindings[key].candidate) for key in pair  # type: ignore[attr-defined]
    }
    return decoder_v2.decode_all_patch_pair_v2(
        model, query, poison, pair[0], pair[1]  # type: ignore[arg-type]
    )


def _ordinary_regional_arms(
    model: torch.nn.Module,
    query: object,
    locks: Mapping[str, object],
    pair: tuple[str, str],
) -> tuple[object, object]:
    left, right = runtime_v2._v1._validate_decode_inputs(
        query, locks, pair[0], pair[1]  # type: ignore[arg-type]
    )
    model_sha, comparator_sha, _ = runtime_v2.bind_runtime_lineage(model)
    adapter = runtime_v2.DeviceMaskRuntimeAdapter(
        model, model_sha, comparator_sha
    )
    output = []
    for arm_name in (
        ARM_QUERY_FULL_REFERENCE,
        ARM_QUERY_LOCAL_COMPONENTS,
    ):
        forward = tuple(
            runtime_v2.decode_direct_arm_term(
                adapter,
                query,
                left,
                right,
                direction=direction,
                arm_name=arm_name,
            )
            for direction in FIXED_DIRECTIONS
        )
        reverse = tuple(
            runtime_v2.decode_direct_arm_term(
                adapter,
                query,
                right,
                left,
                direction=direction,
                arm_name=arm_name,
            )
            for direction in FIXED_DIRECTIONS
        )
        output.append(
            runtime_v2._reduce_arm(
                arm_name, pair[0], pair[1], forward, reverse
            )
        )
    return tuple(output)  # type: ignore[return-value]


def _c_dino_regional_arms(
    model: torch.nn.Module,
    query: object,
    locks: Mapping[str, object],
    pair: tuple[str, str],
) -> tuple[tuple[object, object], Mapping[str, str]]:
    left, right = decoder_v2._validate_pair_only_c_dino_inputs(
        query, locks, pair[0], pair[1]  # type: ignore[arg-type]
    )
    model_sha, comparator_sha, _ = runtime_v2.bind_runtime_lineage(model)
    adapter = runtime_v2.DeviceMaskRuntimeAdapter(
        model, model_sha, comparator_sha
    )
    output = []
    transport = {
        ARM_ALL_PATCH: decoder_v2._transport_receipt_sha256(
            left_key=pair[0],
            right_key=pair[1],
            arm_name=ARM_ALL_PATCH,
            receipts=(),
        )
    }
    for arm_name in (
        ARM_QUERY_FULL_REFERENCE,
        ARM_QUERY_LOCAL_COMPONENTS,
    ):
        forward = []
        reverse = []
        receipts = []
        for direction in FIXED_DIRECTIONS:
            if arm_name == ARM_QUERY_FULL_REFERENCE:
                forward.append(
                    runtime_v2.decode_direct_arm_term(
                        adapter,
                        query,
                        left,
                        right,
                        direction=direction,
                        arm_name=arm_name,
                    )
                )
                reverse.append(
                    runtime_v2.decode_direct_arm_term(
                        adapter,
                        query,
                        right,
                        left,
                        direction=direction,
                        arm_name=arm_name,
                    )
                )
            else:
                term, local_receipts = controls_v1._c_dino_paired_term(
                    adapter, query, left, right, direction=direction
                )
                decoder_v2._validate_local_transport_term(
                    left,
                    right,
                    direction=direction,
                    term=term,
                    receipts=local_receipts,
                )
                forward.append(term)
                receipts.extend(local_receipts)
                term, local_receipts = controls_v1._c_dino_paired_term(
                    adapter, query, right, left, direction=direction
                )
                decoder_v2._validate_local_transport_term(
                    right,
                    left,
                    direction=direction,
                    term=term,
                    receipts=local_receipts,
                )
                reverse.append(term)
                receipts.extend(local_receipts)
        output.append(
            runtime_v2._reduce_arm(
                arm_name,
                pair[0],
                pair[1],
                tuple(forward),
                tuple(reverse),
            )
        )
        transport[arm_name] = decoder_v2._transport_receipt_sha256(
            left_key=pair[0],
            right_key=pair[1],
            arm_name=arm_name,
            receipts=tuple(receipts),
        )
    return tuple(output), transport  # type: ignore[return-value]


def _assemble_three_arm(
    model: torch.nn.Module,
    pair: tuple[str, str],
    all_patch: object,
    regional: tuple[object, object],
) -> ThreeArmFixedDenominatorEvidenceV1:
    model_sha, comparator_sha, reducer_sha = runtime_v2.bind_runtime_lineage(
        model
    )
    evidence = ThreeArmFixedDenominatorEvidenceV1(
        left_candidate_key=pair[0],
        right_candidate_key=pair[1],
        model_checkpoint_sha256=model_sha,
        pair_comparator_sha256=comparator_sha,
        reducer_sha256=reducer_sha,
        binding_mode=BINDING_REAL,
        arms=(all_patch, *regional),  # type: ignore[arg-type]
    )
    runtime_v2._assert_matched_regional_structure(evidence)
    return evidence


def _term_record(orientation: str, term: object) -> dict[str, object]:
    receipts = [
        {
            "root_ordinal": item.root_ordinal,
            "owner_component_source_root_ordinal": (
                item.owner_component_source_root_ordinal
            ),
            "opponent_component_source_root_ordinal": (
                item.opponent_component_source_root_ordinal
            ),
            "binding_mode": item.binding_mode,
            "query_mask_sha256": item.query_mask_sha256,
            "owner_reference_mask_sha256": (
                item.owner_reference_mask_sha256
            ),
            "opponent_reference_mask_sha256": (
                item.opponent_reference_mask_sha256
            ),
        }
        for item in term.decode_receipts
    ]
    return {
        "orientation": orientation,
        "direction": term.direction,
        "owner_candidate_key": term.owner_candidate_key,
        "opponent_candidate_key": term.opponent_candidate_key,
        "status": term.status,
        "query_mask_sha256": tensor_sha256(term.query_mask),
        "patch_evidence_sha256": tensor_sha256(term.patch_evidence),
        "scalar_contributions_sha256": tensor_sha256(
            term.scalar_contributions
        ),
        "logit_sha256": tensor_sha256(term.logit),
        "decoded_root_ordinals": list(term.decoded_root_ordinals),
        "decode_receipts": receipts,
        "decoder_candidate_call_count": 2 * len(receipts),
    }


def evidence_records(
    evidence_by_family: Mapping[str, object],
    *,
    c_dino_transport: Mapping[str, str],
) -> list[dict[str, object]]:
    records = []
    for family in FAMILY_SEQUENCE:
        evidence = evidence_by_family[family]
        by_arm = evidence.by_name()
        require(tuple(by_arm) == ARM_SEQUENCE, f"{family} arm order drift")
        for arm_name in ARM_SEQUENCE:
            arm = by_arm[arm_name]
            terms = [
                *(
                    _term_record("FORWARD", term)
                    for term in arm.forward_by_direction
                ),
                *(
                    _term_record("REVERSE", term)
                    for term in arm.reverse_by_direction
                ),
            ]
            require(
                len(terms) == 4
                and [row["direction"] for row in terms]
                == [*FIXED_DIRECTIONS, *FIXED_DIRECTIONS],
                f"{family}/{arm_name} direction-term order drift",
            )
            record: dict[str, object] = {
                "family": family,
                "arm_name": arm_name,
                "left_candidate_key": arm.left_candidate_key,
                "right_candidate_key": arm.right_candidate_key,
                "model_checkpoint_sha256": evidence.model_checkpoint_sha256,
                "pair_comparator_sha256": evidence.pair_comparator_sha256,
                "reducer_sha256": evidence.reducer_sha256,
                "binding_mode": evidence.binding_mode,
                "scalar_contributions_sha256": tensor_sha256(
                    arm.scalar_contributions
                ),
                "pair_score_sha256": tensor_sha256(arm.logit),
                "direction_terms": terms,
                "direction_sequence_sha256": canonical_sha256(terms),
                "c_dino_transport_receipt_sha256": (
                    c_dino_transport[arm_name]
                    if family == "C_DINO_V"
                    else None
                ),
            }
            record["record_sha256"] = canonical_sha256(record)
            records.append(record)
    require(len(records) == 9, "E1 pair-arm record population drift")
    require(
        sum(len(row["direction_terms"]) for row in records) == 36,
        "E1 directional-term population drift",
    )
    return records


def evidence_tensor_payloads(
    evidence_by_family: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    payloads: dict[str, object] = {}
    hashes: dict[str, object] = {}
    for family in FAMILY_SEQUENCE:
        arms = evidence_by_family[family].by_name()
        for arm_name in ARM_SEQUENCE:
            arm = arms[arm_name]
            terms = (*arm.forward_by_direction, *arm.reverse_by_direction)
            key = f"{family}/{arm_name}"
            payload = {
                "scalar_contributions": arm.scalar_contributions.detach()
                .cpu()
                .contiguous(),
                "pair_score": arm.logit.detach().cpu().contiguous(),
                "signed_term_logits": tuple(
                    item.detach().cpu().contiguous()
                    for item in arm.signed_term_logits
                ),
                "term_query_masks": tuple(
                    item.query_mask.detach().cpu().contiguous()
                    for item in terms
                ),
                "term_patch_evidence": tuple(
                    item.patch_evidence.detach().cpu().contiguous()
                    for item in terms
                ),
                "term_scalar_contributions": tuple(
                    item.scalar_contributions.detach().cpu().contiguous()
                    for item in terms
                ),
                "term_logits": tuple(
                    item.logit.detach().cpu().contiguous() for item in terms
                ),
            }
            payloads[key] = payload
            hashes[key] = {
                name: (
                    [tensor_sha256(item) for item in value]
                    if isinstance(value, tuple)
                    else tensor_sha256(value)
                )
                for name, value in payload.items()
            }
    require(len(payloads) == len(hashes) == 9, "E1 tensor payload population drift")
    return payloads, hashes


def phase_b_receipt_from_evidence(
    *,
    model: torch.nn.Module,
    real_bundle: VQueryBundle,
    c_col_bundle: VQueryBundle,
    pair: tuple[str, str],
    real_evidence: object,
    c_col_evidence: object,
) -> dict[str, str]:
    model_sha, comparator_sha, reducer_sha = runtime_v2.bind_runtime_lineage(
        model
    )
    real_axis, control_axis = phase_b._assert_dino_axis_exact(
        real_bundle, c_col_bundle
    )
    real_request = phase_b._request_record(
        axis_record=real_axis,
        bundle=real_bundle,
        pair_member_keys=pair,
        model_checkpoint_sha256=model_sha,
        comparator_sha256=comparator_sha,
        reducer_sha256=reducer_sha,
    )
    control_request = phase_b._request_record(
        axis_record=control_axis,
        bundle=c_col_bundle,
        pair_member_keys=pair,
        model_checkpoint_sha256=model_sha,
        comparator_sha256=comparator_sha,
        reducer_sha256=reducer_sha,
    )
    real_arm = real_evidence.by_name()[ARM_ALL_PATCH]
    control_arm = c_col_evidence.by_name()[ARM_ALL_PATCH]
    phase_b._assert_arm_exact(real_arm, control_arm)
    receipt = {
        "real_request_sha256": phase_b._payload_sha256(real_request),
        "control_request_sha256": phase_b._payload_sha256(control_request),
        "real_patch_tensor_sha256": tensor_sha256(
            real_arm.scalar_contributions
        ),
        "control_patch_tensor_sha256": tensor_sha256(
            control_arm.scalar_contributions
        ),
        "real_direction_sequence_sha256": phase_b._payload_sha256(
            phase_b._direction_sequence_record(real_arm)
        ),
        "control_direction_sequence_sha256": phase_b._payload_sha256(
            phase_b._direction_sequence_record(control_arm)
        ),
        "real_pair_score_sha256": tensor_sha256(real_arm.logit),
        "control_pair_score_sha256": tensor_sha256(control_arm.logit),
    }
    require(
        receipt["real_request_sha256"] == receipt["control_request_sha256"]
        and receipt["real_patch_tensor_sha256"]
        == receipt["control_patch_tensor_sha256"]
        and receipt["real_direction_sequence_sha256"]
        == receipt["control_direction_sequence_sha256"]
        and receipt["real_pair_score_sha256"]
        == receipt["control_pair_score_sha256"],
        "E1 REAL/C_COL ALL_PATCH exact invariance drift",
    )
    return receipt


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.chmod(0o444)
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _seal_payload(
    path: Path,
    value: Mapping[str, Any],
    *,
    role: str,
    authority_sha256: str,
    torch_payload: bool,
) -> None:
    require(not path.exists() and not path.is_symlink(), f"immutable payload exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if torch_payload:
        torch.save(dict(value), temporary)
    else:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n",
            encoding="ascii",
        )
    temporary.chmod(0o444)
    os.link(temporary, path)
    temporary.unlink()
    receipt = {
        "schema_version": "rc_dino_rcde_track_r_v124_e1_prejoin_receipt_v1_20260826",
        "status": "DINO_RCDE_TRACK_R_V124_E1_PREJOIN_RECEIPT_READY",
        "role": role,
        "payload_path": path.name,
        "payload_bytes": path.stat().st_size,
        "payload_sha256": file_sha256(path),
        "payload_logical_sha256": value["logical_sha256"],
    }
    receipt["logical_sha256"] = logical_sha256(receipt)
    receipt_path = path.with_name(path.name + ".receipt.json")
    _exclusive_json(receipt_path, receipt)
    commit = {
        "schema_version": "rc_dino_rcde_track_r_v124_e1_prejoin_commit_v1_20260826",
        "status": "DINO_RCDE_TRACK_R_V124_E1_PREJOIN_PAYLOAD_COMMITTED",
        "authority_sha256": authority_sha256,
        "role": role,
        "payload_path": path.name,
        "payload_sha256": file_sha256(path),
        "receipt_path": receipt_path.name,
        "receipt_sha256": file_sha256(receipt_path),
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    commit["logical_sha256"] = logical_sha256(commit)
    _exclusive_json(path.with_name(path.name + ".commit.json"), commit)


def run(
    *,
    authority_path: Path,
    authority_sha256: str,
    artifact_output: Path = ARTIFACT,
    result_output: Path = RESULT,
) -> dict[str, Any]:
    require(Path.cwd().resolve() == ROOT.resolve(), "E1 producer cwd must equal RC root")
    authority = read_authority(authority_path, authority_sha256)
    phase_a_path = _bound_path(authority, "phase_a_artifact")
    phase_a_artifact = torch.load(
        phase_a_path, map_location="cpu", weights_only=False, mmap=True
    )
    require(isinstance(phase_a_artifact, Mapping), "Phase-A artifact root drift")
    real_bundle = phase_a_artifact.get("real_bundle")
    require(
        isinstance(real_bundle, VQueryBundle)
        and real_bundle.query_id == "DIFFICULT-0000"
        and real_bundle.execution_ordinal == 0
        and real_bundle.outer_fold == 2
        and real_bundle.candidate_axis_sha256 == CANDIDATE_AXIS_SHA256
        and len(real_bundle.locks) == 128,
        "REAL execution-0 bundle drift",
    )
    pair_record, pair = execution0_pair(
        _bound_path(authority, "role_free_pair_manifest")
    )
    require(set(pair).issubset(real_bundle.locks), "pair absent from REAL C128")
    query_geometry_sha, reference_geometry_by_row = canonical_geometry_axis(
        _bound_path(authority, "canonical_geometry_payload"), real_bundle
    )
    c_col_pair_locks = build_c_col_pair_locks(
        phase_a_artifact,
        pair,
        canonical_query_geometry_sha256=query_geometry_sha,
        canonical_reference_geometry_by_row=reference_geometry_by_row,
    )
    c_col_axis_view = build_c_col_axis_view(phase_a_artifact)
    identities = corrected_identity_by_row(
        _bound_path(authority, "gallery_cache"), real_bundle
    )
    c_dino_pair, c_dino_receipt = (
        extract_full_c128_c_dino_pair_from_runtime_locks(
            real_bundle.locks,
            pair[0],
            pair[1],
            namespace=C_DINO_NAMESPACE,
            query_id=real_bundle.query_id,
            corrected_identity_by_physical_row=identities,
        )
    )
    require(
        c_dino_receipt.get("candidate_count") == 128
        and c_dino_receipt.get("fixed_point_free") is True
        and c_dino_receipt.get("corrected_identity_disjoint") is True
        and c_dino_receipt.get("pair_member_donor_exclusion") is True
        and c_dino_receipt.get("candidate_axis_preserved") is True
        and c_dino_receipt.get("p_lock_preserved") is True
        and c_dino_receipt.get("full_native_content_binding_permutation") is True
        and c_dino_receipt.get("materialized_control_candidate_count") == 2
        and c_dino_receipt.get("model_load_count") == 0
        and c_dino_receipt.get("model_forward_count") == 0,
        "corrected C_DINO full-C128 receipt drift",
    )
    model, state_sha = load_model(_bound_path(authority, "fold2_v_checkpoint"))
    require(
        state_sha == authority.get("v_checkpoint_state_sha256"),
        "authority/fold-2 V state drift",
    )
    require(
        torch.cuda.is_available() and torch.cuda.device_count() >= 1,
        "E1 accelerated execution requires one visible GPU",
    )
    with torch.no_grad():
        real_all = _all_patch_only(
            model, real_bundle.query, real_bundle.locks, pair
        )
        c_dino_all = _all_patch_only(
            model, real_bundle.query, c_dino_pair, pair
        )
        c_col_all = _all_patch_only(
            model, real_bundle.query, c_col_axis_view.locks, pair
        )
    gpu = torch.device("cuda:0")
    model.to(gpu)
    query_gpu = _query_to_device(real_bundle.query, gpu)
    real_pair_gpu = move_locks_to_device(
        {key: real_bundle.locks[key] for key in pair}, gpu
    )
    c_dino_pair_gpu = move_full_c128_c_dino_pair_to_device(
        c_dino_pair, gpu, expected_pair_member_keys=pair
    )
    c_col_pair_gpu = move_locks_to_device(c_col_pair_locks, gpu)
    with torch.no_grad():
        real_regional = _ordinary_regional_arms(
            model, query_gpu, real_pair_gpu, pair
        )
        c_dino_regional, c_dino_transport = _c_dino_regional_arms(
            model, query_gpu, c_dino_pair_gpu, pair
        )
        c_col_regional = _ordinary_regional_arms(
            model, query_gpu, c_col_pair_gpu, pair
        )
    real_evidence = _assemble_three_arm(
        model, pair, real_all, real_regional
    )
    c_dino_evidence = _assemble_three_arm(
        model, pair, c_dino_all, c_dino_regional
    )
    c_col_evidence = _assemble_three_arm(
        model, pair, c_col_all, c_col_regional
    )
    require(state_dict_sha256(model) == state_sha, "E1 scoring mutated V checkpoint")
    evidence = {
        "REAL": real_evidence,
        "C_DINO_V": c_dino_evidence,
        "C_COL_P": c_col_evidence,
    }
    records = evidence_records(evidence, c_dino_transport=c_dino_transport)
    tensor_payloads, tensor_payload_hashes = evidence_tensor_payloads(evidence)
    decode_candidate_call_count = sum(
        term["decoder_candidate_call_count"]
        for record in records
        for term in record["direction_terms"]
    )
    phase_b_receipt = phase_b_receipt_from_evidence(
        model=model,
        real_bundle=real_bundle,
        c_col_bundle=c_col_axis_view,
        pair=pair,
        real_evidence=real_evidence,
        c_col_evidence=c_col_evidence,
    )
    bound_phase_b = json.loads(
        _bound_path(authority, "phase_b_result").read_text(encoding="ascii")
    )
    require(
        all(bound_phase_b.get(key) == phase_b_receipt[key] for key in PHASE_B_FIELDS),
        "E1 ALL_PATCH does not replay the independently passed Phase-B bytes",
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
            item["address_sha256"] for item in pair_record["members"]
        ],
        "opened594_role": "DIAGNOSTIC_ONLY_NOT_CONFIRMATION",
        "family_sequence": list(FAMILY_SEQUENCE),
        "arm_sequence": list(ARM_SEQUENCE),
        "pair_arm_record_count": 9,
        "directional_term_count": 36,
        "records": records,
        "record_sha256_sequence": [row["record_sha256"] for row in records],
        "tensor_payloads": tensor_payloads,
        "tensor_payload_hashes": tensor_payload_hashes,
        "c_dino_receipt": dict(c_dino_receipt),
        "c_dino_receipt_logical_sha256": c_dino_receipt["logical_sha256"],
        "c_col_record_metadata_checked_count": 256,
        "c_col_runtime_materialized_candidate_count": 2,
        "phase_a_artifact_sha256": file_sha256(phase_a_path),
        "phase_b_result_sha256": file_sha256(
            _bound_path(authority, "phase_b_result")
        ),
        "phase_b_invariance_receipt": phase_b_receipt,
        "v_checkpoint_state_sha256": state_sha,
        "device_schedule": {
            "all_patch_device": "cpu",
            "regional_device": "cuda:0",
            "model_load_count": 1,
            "model_device_transition_count": 1,
        },
        "access_audit": {
            "v_model_load_count": 1,
            "model_device_transition_count": 1,
            "v_decode_candidate_call_count": decode_candidate_call_count,
            "p_model_load_count": 0,
            "p_model_forward_count": 0,
            "training_backward_update_count": 0,
            "target_label_read_count": 0,
            "target_rival_read_count": 0,
            "postjoin_read_count": 0,
            "protected_source_read_count": 0,
            "h0_track_h_rgh_gx_read_count": 0,
            "scientific_reduction_count": 0,
            "automatic_submit_count": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    artifact["logical_sha256"] = canonical_sha256(
        {
            key: item
            for key, item in artifact.items()
            if key not in {"logical_sha256", "tensor_payloads"}
        }
    )
    _seal_payload(
        artifact_output,
        artifact,
        role="E1_PREJOIN_ARTIFACT",
        authority_sha256=authority_sha256,
        torch_payload=True,
    )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": STATUS,
        "abort_status": ABORT,
        "claim_level": "FIXED_NATURAL_OPENED_CANARY_TARGET_FREE_PREJOIN_ONLY",
        "authority_sha256": authority_sha256,
        "artifact_path": artifact_output.relative_to(ROOT).as_posix(),
        "artifact_sha256": file_sha256(artifact_output),
        "artifact_logical_sha256": artifact["logical_sha256"],
        "execution_ordinal": 0,
        "query_id": "DIFFICULT-0000",
        "source_fold": 2,
        "candidate_count": 128,
        "candidate_axis_sha256": CANDIDATE_AXIS_SHA256,
        "pair_sha256": PAIR_SHA256,
        "opened594_role": "DIAGNOSTIC_ONLY_NOT_CONFIRMATION",
        "family_sequence": list(FAMILY_SEQUENCE),
        "arm_sequence": list(ARM_SEQUENCE),
        "pair_arm_record_count": 9,
        "directional_term_count": 36,
        "record_sha256_sequence_sha256": canonical_sha256(
            artifact["record_sha256_sequence"]
        ),
        "c_dino_full_c128_pass": True,
        "c_col_phase_a_retry1_pass": True,
        "c_col_record_metadata_checked_count": 256,
        "c_col_runtime_materialized_candidate_count": 2,
        "real_c_col_all_patch_exact_phase_b_match": True,
        "all_patch_device": "cpu",
        "regional_device": "cuda:0",
        "v_model_load_count": 1,
        "model_device_transition_count": 1,
        "v_decode_candidate_call_count": decode_candidate_call_count,
        "p_model_load_count": 0,
        "p_model_forward_count": 0,
        "training_backward_update_count": 0,
        "target_label_read_count": 0,
        "target_rival_read_count": 0,
        "postjoin_read_count": 0,
        "protected_source_read_count": 0,
        "h0_track_h_rgh_gx_read_count": 0,
        "scientific_reduction_count": 0,
        "automatic_submit_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    result["logical_sha256"] = logical_sha256(result)
    _seal_payload(
        result_output,
        result,
        role="E1_PRODUCER_RESULT",
        authority_sha256=authority_sha256,
        torch_payload=False,
    )
    print(
        json.dumps(
            {"status": STATUS, "logical_sha256": result["logical_sha256"]},
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, default=AUTHORITY)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--artifact-output", type=Path, default=ARTIFACT)
    parser.add_argument("--result-output", type=Path, default=RESULT)
    args = parser.parse_args()
    run(
        authority_path=args.authority,
        authority_sha256=args.authority_sha256,
        artifact_output=args.artifact_output,
        result_output=args.result_output,
    )


if __name__ == "__main__":
    main()
