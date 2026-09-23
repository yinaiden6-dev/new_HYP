#!/usr/bin/env python3
"""Source-level independent V124-E1 natural-canary prejoin replay.

This validator does not import the E1 producer.  It independently rebuilds all
three mechanism families, replays all three corrected arms, and compares the
complete ordered 9-record/36-term artifact before emitting a validation result.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.dino_rcde_cw1_multitile_superregion_v2 import (  # noqa: E402
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
)
from rc_aslo_xf.dino_rcde_cw1_multitile_vdecode_v1 import FIXED_DIRECTIONS  # noqa: E402
from rc_aslo_xf.dino_rcde_cw1_multitile_vdecode_v1 import (  # noqa: E402
    BINDING_REAL,
    ThreeArmFixedDenominatorEvidenceV1,
)
from rc_aslo_xf.dino_rcde_cw1_multitile_vdecode_v1 import (  # noqa: E402
    CandidateReferenceFieldV1,
    token_tensor_sha256,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import (  # noqa: E402
    LOCK_PROPOSAL_READY as V2_LOCK_READY,
    LOCK_STRUCTURAL_PROPOSAL_UNAVAILABLE as V2_LOCK_UNAVAILABLE,
    ROOT_QUERY_UNMAPPABLE as V2_ROOT_QUERY_UNMAPPABLE,
    ROOT_READY as V2_ROOT_READY,
    ROOT_REFERENCE_MISSING as V2_ROOT_REFERENCE_MISSING,
    candidate_p_lock_v2_from_record,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_natural_adapter_v2 import (  # noqa: E402
    validate_natural_p_lock_v2,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (  # noqa: E402
    LOCK_H0 as V1_LOCK_H0,
    LOCK_READY as V1_LOCK_READY,
    ROOT_H0 as V1_ROOT_H0,
    ROOT_READY as V1_ROOT_READY,
    tensor_sha256 as p_tensor_sha256,
)
from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (  # noqa: E402
    CandidatePLockV1,
    SealedDirectionPLockV1,
    SealedRootDinoScopeV1,
    VQueryBundle,
    canonical_candidate_axis,
    hash_parts,
    move_locks_to_device,
    state_dict_sha256,
    tensor_sha256,
)
from rc_aslo_xf.dino_rcde_sr0_mt_controls_v1 import (  # noqa: E402
    CDinoCandidateBindingV1,
    move_c_dino_locks_to_device,
)
from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2  # noqa: E402
from rc_aslo_xf.gallery_identity_repair import build_identity_map  # noqa: E402
from rc_aslo_xf import dino_rcde_sr0_mt_v_runtime_v2 as runtime_v2  # noqa: E402
from rc_aslo_xf import dino_rcde_sr0_mt_controls_v1 as controls_v1  # noqa: E402
from rc_aslo_xf import dino_rcde_track_r_control_decoder_v2 as decoder_v2  # noqa: E402


AUTHORITY = ROOT / "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_authority_v1_20260826.json"
ARTIFACT = ROOT / "results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v1/prejoin_artifact.pt"
PRODUCER_RESULT = ROOT / "results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v1/producer_result.json"
OUTPUT = ROOT / "results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_validation_v1/result.json"
AUTHORITY_SCHEMA = "rc_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_authority_v1_20260826"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_V124_E1_FIXED_NATURAL_OPENED_CANARY_PREJOIN_EXECUTION_AUTHORIZED"
ARTIFACT_SCHEMA = "rc_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_artifact_v1_20260826"
PRODUCER_STATUS = "DINO_RCDE_TRACK_R_V124_E1_FIXED_NATURAL_OPENED_CANARY_PREJOIN_PASS"
SCHEMA = "rc_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_independent_validation_v1_20260826"
STATUS = "DINO_RCDE_TRACK_R_V124_E1_FIXED_NATURAL_OPENED_CANARY_PREJOIN_INDEPENDENT_VALIDATION_PASS"
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


class V124E1IndependentError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise V124E1IndependentError(message)


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
    require(path.is_file() and not path.is_symlink(), f"bound file absent: {name}")
    require(file_sha256(path) == row.get("sha256"), f"bound hash drift: {name}")
    return path


def read_authority(path: Path, expected_sha256: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(
        file_sha256(path) == expected_sha256
        and value.get("schema_version") == AUTHORITY_SCHEMA
        and value.get("status") == AUTHORITY_STATUS
        and value.get("logical_sha256") == logical_sha256(value)
        and value.get("independent_review_status") == "PASS"
        and value.get("execution_ordinal") == 0
        and value.get("query_id") == "DIFFICULT-0000"
        and value.get("candidate_axis_sha256") == CANDIDATE_AXIS_SHA256
        and value.get("pair_sha256") == PAIR_SHA256
        and tuple(value.get("family_sequence", ())) == FAMILY_SEQUENCE
        and tuple(value.get("arm_sequence", ())) == ARM_SEQUENCE
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
        and value.get("target_rival_join_authorized") is False
        and value.get("postjoin_authorized") is False
        and value.get("scientific_reduction_authorized") is False
        and value.get("automatic_submit_authorized") is False,
        "independent E1 authority drift",
    )
    for name in value["bindings"]:
        _bound_path(value, name)
    return value


def _validate_triple(
    path: Path,
    value: Mapping[str, Any],
    *,
    role: str,
    authority_sha256: str,
) -> None:
    receipt_path = path.with_name(path.name + ".receipt.json")
    commit_path = path.with_name(path.name + ".commit.json")
    require(
        path.is_file()
        and receipt_path.is_file()
        and commit_path.is_file()
        and all(
            item.stat().st_mode & 0o777 == 0o444
            for item in (path, receipt_path, commit_path)
        ),
        f"{role} complete immutable triple absent",
    )
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    commit = json.loads(commit_path.read_text(encoding="ascii"))
    require(
        receipt.get("schema_version")
        == "rc_dino_rcde_track_r_v124_e1_prejoin_receipt_v1_20260826"
        and receipt.get("status")
        == "DINO_RCDE_TRACK_R_V124_E1_PREJOIN_RECEIPT_READY"
        and receipt.get("role") == role
        and receipt.get("payload_path") == path.name
        and receipt.get("payload_sha256") == file_sha256(path)
        and receipt.get("payload_bytes") == path.stat().st_size
        and receipt.get("payload_logical_sha256") == value.get("logical_sha256")
        and receipt.get("logical_sha256") == logical_sha256(receipt)
        and commit.get("schema_version")
        == "rc_dino_rcde_track_r_v124_e1_prejoin_commit_v1_20260826"
        and commit.get("status")
        == "DINO_RCDE_TRACK_R_V124_E1_PREJOIN_PAYLOAD_COMMITTED"
        and commit.get("authority_sha256") == authority_sha256
        and commit.get("role") == role
        and commit.get("payload_path") == path.name
        and commit.get("payload_sha256") == file_sha256(path)
        and commit.get("receipt_path") == receipt_path.name
        and commit.get("receipt_sha256") == file_sha256(receipt_path)
        and commit.get("scientific_GO_or_NO_GO") is None
        and commit.get("automatic_stage_advance") is False
        and commit.get("next_authorized_stage") is None
        and commit.get("logical_sha256") == logical_sha256(commit),
        f"{role} receipt/commit drift",
    )


def execution0_pair(path: Path) -> tuple[Mapping[str, Any], tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="ascii"))
    rows = [
        row
        for row in payload["pair_manifest"]["records"]
        if row.get("execution_ordinal") == 0
    ]
    require(len(rows) == 1 and rows[0].get("pair_sha256") == PAIR_SHA256, "pair drift")
    pair = rows[0]
    require(
        pair.get("candidate_axis_sha256") == CANDIDATE_AXIS_SHA256
        and pair.get("query_id") == "DIFFICULT-0000"
        and len(pair.get("members", [])) == 2,
        "pair address drift",
    )
    return pair, tuple(str(row["candidate_key"]) for row in pair["members"])  # type: ignore[return-value]


def canonical_geometry(
    path: Path, real: VQueryBundle
) -> tuple[str, Mapping[int, str]]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    query_rows = {}
    for row in payload.get("query_records", []):
        if row.get("execution_ordinal") == 0:
            require(0 not in query_rows, "duplicate canonical query geometry")
            query_rows[0] = row
    require(
        0 in query_rows
        and query_rows[0].get("query_id") == real.query_id
        and query_rows[0].get("source_image_sha256")
        == real.query.source_image_sha256
        and query_rows[0].get("candidate_axis_sha256")
        == real.candidate_axis_sha256,
        "independent canonical query geometry drift",
    )
    reference_rows = {}
    for row in payload.get("reference_records", []):
        physical = row.get("physical_row")
        require(
            physical not in reference_rows,
            "duplicate canonical reference geometry row",
        )
        reference_rows[physical] = row
    output = {}
    for lock in real.locks.values():
        candidate = lock.candidate
        row = reference_rows.get(candidate.physical_gallery_row)
        require(
            isinstance(row, Mapping)
            and row.get("source_image_sha256")
            == candidate.source_image_sha256
            and tuple(row["dino_geometry"]["grid_shape"])
            == candidate.grid_shape,
            "independent canonical reference geometry drift",
        )
        output[candidate.physical_gallery_row] = row["dino_geometry"][
            "geometry_sha256"
        ]
    require(len(output) == 128, "independent canonical C128 geometry drift")
    return query_rows[0]["dino_geometry"]["geometry_sha256"], output


def reconstruct_c_col(
    artifact: Mapping[str, Any],
    pair: tuple[str, str],
    *,
    canonical_query_geometry_sha256: str,
    canonical_reference_geometry_by_row: Mapping[int, str],
) -> Mapping[str, CandidatePLockV1]:
    """Independent loop; no producer constructor or reducer is imported."""

    real = artifact.get("real_bundle")
    wrappers = artifact.get("control_candidates")
    require(isinstance(real, VQueryBundle), "REAL bundle type drift")
    require(isinstance(wrappers, (list, tuple)) and len(wrappers) == 128, "C_COL wrappers drift")
    wrapper_by_key = {}
    for expected_position in range(128):
        matches = [
            row
            for row in wrappers
            if row.get("candidate_position") == expected_position
        ]
        require(len(matches) == 1, "C_COL position absent/duplicated")
        wrapper = matches[0]
        candidate = wrapper["candidate"]
        records = wrapper["direction_records"]
        require(
            set(records) == set(FIXED_DIRECTIONS)
            and candidate.candidate_key == wrapper.get("candidate_key")
            and candidate.candidate_key in real.locks
            and candidate is real.locks[candidate.candidate_key].candidate
            and all(
                records[direction]["record_sha256"]
                == wrapper["direction_record_sha256_sequence"][index]
                for index, direction in enumerate(FIXED_DIRECTIONS)
            ),
            "C_COL direction/REAL DINO/record receipt drift",
        )
        wrapper_by_key[candidate.candidate_key] = wrapper
    require(
        len(wrapper_by_key) == 128 and set(pair).issubset(wrapper_by_key),
        "C_COL wrapper key/pair population drift",
    )
    locks = {}
    for key in pair:
        wrapper = wrapper_by_key[key]
        candidate = wrapper["candidate"]
        records = wrapper["direction_records"]
        projected = {}
        for direction in FIXED_DIRECTIONS:
            value = records[direction]
            validate_natural_p_lock_v2(value)
            core = candidate_p_lock_v2_from_record(value["core_lock_record"])
            query_address = value["query"]
            candidate_address = value["candidate"]
            selected = value["selected"]
            fixed = value["fixed_denominator"]
            require(
                value["direction"] == direction
                and query_address["fit_id"] == "P_OUTER2_OUTER_REFIT"
                and query_address["crossfit_role"]
                == "OUTER_HELDOUT_DEPLOYMENT"
                and query_address["outer_fold"] == 2
                and query_address["inner_heldout_fold"] is None
                and query_address["query_source_image_sha256"]
                == real.query.source_image_sha256
                and candidate_address["candidate_key"]
                == candidate.candidate_key
                and candidate_address["candidate_physical_row"]
                == candidate.physical_gallery_row
                and candidate_address["candidate_reference_source_sha256"]
                == candidate.source_image_sha256
                and core.candidate_key == candidate.candidate_key
                and core.candidate_physical_row
                == candidate.physical_gallery_row
                and torch.equal(
                    core.query_valid_mask, real.query.valid_patch_mask
                )
                and torch.equal(
                    core.reference_valid_mask, candidate.valid_patch_mask
                )
                and core.geometry_signature.query_geometry_sha256
                == value["query_geometry"]["geometry_sha256"]
                == canonical_query_geometry_sha256
                and core.geometry_signature.reference_geometry_sha256
                == value["reference_geometry"]["geometry_sha256"],
                "independent C_COL address/geometry/head drift",
            )
            require(
                value["reference_geometry"]["geometry_sha256"]
                == canonical_reference_geometry_by_row.get(
                    candidate.physical_gallery_row
                ),
                "independent C_COL canonical reference geometry drift",
            )
            all_roots = {}
            for root in core.roots:
                require(
                    root.status != V2_ROOT_REFERENCE_MISSING,
                    "C_COL needs a V2-native bridge for reference missingness",
                )
                if root.status == V2_ROOT_READY:
                    binding_status = V1_ROOT_READY
                elif root.status == V2_ROOT_QUERY_UNMAPPABLE:
                    binding_status = V1_ROOT_H0
                else:
                    raise V124E1IndependentError(
                        "unknown independent C_COL root status"
                    )
                all_roots[root.root_ordinal] = SealedRootDinoScopeV1(
                    root_ordinal=root.root_ordinal,
                    action_key_sha256=root.action_key_sha256,
                    query_mask=root.query_mask,
                    reference_mask=root.reference_mask,
                    query_grid_shape=real.query.grid_shape,
                    reference_grid_shape=candidate.grid_shape,
                    query_mask_p_sha256=p_tensor_sha256(root.query_mask),
                    reference_mask_p_sha256=p_tensor_sha256(
                        root.reference_mask
                    ),
                    query_geometry_sha256=real.query.geometry_record_sha256,
                    reference_geometry_sha256=(
                        candidate.geometry_record_sha256
                    ),
                    binding_status=binding_status,
                )
            union = torch.as_tensor(
                fixed["query_union"], dtype=torch.bool
            ).flatten()
            require(
                torch.equal(union, core.query_union_mask),
                "independent C_COL fixed denominator drift",
            )
            if core.lock_state == V2_LOCK_READY:
                lock_status = V1_LOCK_READY
                selected_bank = selected["selected_bank_ordinal"]
                selected_row = selected["selected_bank_row_sha256"]
            elif core.lock_state == V2_LOCK_UNAVAILABLE:
                lock_status = V1_LOCK_H0
                selected_bank = None
                selected_row = None
                require(not bool(union.any()), "unavailable C_COL union drift")
            else:
                raise V124E1IndependentError(
                    "unknown independent C_COL lock state"
                )
            projected[direction] = SealedDirectionPLockV1(
                candidate_key=candidate.candidate_key,
                candidate_physical_row=candidate.physical_gallery_row,
                query_source_image_sha256=real.query.source_image_sha256,
                candidate_reference_source_sha256=(
                    candidate.source_image_sha256
                ),
                direction=direction,
                status=lock_status,
                selected_bank_ordinal=selected_bank,
                selected_row_sha256=selected_row,
                query_union_mask=union,
                ordered_root_ordinals=core.selected_root_ordinals,
                all_roots=all_roots,
                query_grid_shape=real.query.grid_shape,
                reference_grid_shape=candidate.grid_shape,
                query_geometry_sha256=real.query.geometry_record_sha256,
                reference_geometry_sha256=candidate.geometry_record_sha256,
                p_lock_record_sha256=value["record_sha256"],
            )
        require(key not in locks, "duplicate C_COL candidate")
        locks[key] = CandidatePLockV1(
            candidate=candidate, direction_locks=projected
        )
    require(len(locks) == 2, "independent C_COL materialization not pair-only")
    return locks


def c_col_axis_view(artifact: Mapping[str, Any]) -> VQueryBundle:
    real = artifact["real_bundle"]
    wrappers = artifact["control_candidates"]
    by_key = {
        wrapper["candidate_key"]: _PRegionalPoison(wrapper["candidate"])
        for wrapper in wrappers
    }
    require(len(by_key) == 128, "independent C_COL axis view drift")
    return VQueryBundle(
        real.query_id,
        real.execution_ordinal,
        real.outer_fold,
        real.query,
        by_key,  # type: ignore[arg-type]
        real.candidate_axis_sha256,
    )


def identities(path: Path, bundle: VQueryBundle) -> Mapping[int, str]:
    gallery = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    legacy = gallery.get("setids") if isinstance(gallery, Mapping) else None
    require(isinstance(legacy, (list, tuple)) and len(legacy) == 5413, "identity axis drift")
    labels = build_identity_map(tuple(str(item) for item in legacy)).labels
    output = {}
    for lock in sorted(
        bundle.locks.values(), key=lambda row: row.candidate.physical_gallery_row
    ):
        row = lock.candidate.physical_gallery_row
        require(row not in output, "duplicate physical row")
        output[row] = labels[row]
    require(len(output) == 128, "identity C128 count drift")
    return output


def _identity_sha(value: str) -> str:
    require(isinstance(value, str) and bool(value), "corrected identity absent")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_content(candidate: CandidateReferenceFieldV1) -> dict[str, object]:
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


def reconstruct_c_dino(
    bundle: VQueryBundle,
    pair: tuple[str, str],
    identity_by_row: Mapping[int, str],
) -> tuple[Mapping[str, CDinoCandidateBindingV1], Mapping[str, object]]:
    """Independent complete matching and pair-only materialization."""

    canonical = tuple(
        sorted(
            bundle.locks.values(),
            key=lambda lock: (
                lock.candidate.physical_gallery_row,
                lock.candidate.source_image_sha256,
                lock.candidate.candidate_key,
            ),
        )
    )
    require(len(canonical) == 128, "independent C_DINO axis is not C128")
    keys = tuple(lock.candidate.candidate_key for lock in canonical)
    require(len(set(keys)) == 128 and set(pair).issubset(keys), "independent C_DINO key axis drift")
    identity_sha_by_key = {
        lock.candidate.candidate_key: _identity_sha(
            identity_by_row[lock.candidate.physical_gallery_row]
        )
        for lock in canonical
    }
    protected = frozenset(pair)
    donor_for: dict[str, str] = {}
    destination_for_donor: dict[str, str] = {}

    def assign(destination: str, seen: set[str]) -> bool:
        donors = sorted(
            (
                donor
                for donor in keys
                if donor != destination
                and identity_sha_by_key[donor]
                != identity_sha_by_key[destination]
                and not (destination in protected and donor in protected)
            ),
            key=lambda donor: hash_parts(
                C_DINO_NAMESPACE,
                bundle.query_id,
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
        key=lambda key: hash_parts(
            C_DINO_NAMESPACE, bundle.query_id, "FULL_C128_DEST", key
        ),
    )
    require(
        all(assign(destination, set()) for destination in destination_order),
        "independent C_DINO perfect matching absent",
    )
    index_by_key = {key: index for index, key in enumerate(keys)}
    order = tuple(index_by_key[donor_for[key]] for key in keys)
    require(
        sorted(order) == list(range(128))
        and all(index != source for index, source in enumerate(order))
        and all(
            identity_sha_by_key[keys[index]]
            != identity_sha_by_key[keys[source]]
            for index, source in enumerate(order)
        ),
        "independent C_DINO derangement/identity drift",
    )
    pair_donors = tuple(keys[order[index_by_key[key]]] for key in pair)
    require(set(pair_donors).isdisjoint(pair), "independent pair donor exclusion drift")

    axis_records = []
    native_axis = []
    for axis_index, lock in enumerate(canonical):
        candidate = lock.candidate
        content_binding = canonical_sha256(_candidate_content(candidate))
        row = {
            "candidate_position": axis_index,
            "candidate_key": candidate.candidate_key,
            "candidate_physical_row": candidate.physical_gallery_row,
            "candidate_reference_source_sha256": candidate.source_image_sha256,
            "corrected_identity_sha256": identity_sha_by_key[
                candidate.candidate_key
            ],
            "p_lock_record_sha256_by_direction": {
                direction: lock.p_lock_record_sha256_by_direction[direction]
                for direction in FIXED_DIRECTIONS
            },
            "candidate_native_content_sha256_by_direction": {
                direction: candidate.tokens_sha256
                for direction in FIXED_DIRECTIONS
            },
            "dino_content_binding_sha256": content_binding,
            "axis_index": axis_index,
        }
        axis_records.append(row)
        native_axis.append(
            {
                "candidate_key": candidate.candidate_key,
                "candidate_physical_row": candidate.physical_gallery_row,
                "candidate_reference_source_sha256": candidate.source_image_sha256,
                "candidate_native_content_sha256_by_direction": row[
                    "candidate_native_content_sha256_by_direction"
                ],
                "dino_content_binding_sha256": content_binding,
            }
        )
    by_key = {lock.candidate.candidate_key: lock for lock in canonical}
    bindings = {}
    pair_receipts = []
    for destination_key, donor_key in zip(pair, pair_donors, strict=True):
        destination_lock = by_key[destination_key]
        destination = destination_lock.candidate
        donor = by_key[donor_key].candidate
        controlled = CandidateReferenceFieldV1(
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
        binding = CDinoCandidateBindingV1(
            candidate=controlled,
            destination_p_lock=destination_lock,
            donor_candidate_key=donor_key,
            donor_physical_gallery_row=donor.physical_gallery_row,
            donor_corrected_identity_sha256=identity_sha_by_key[donor_key],
            destination_corrected_identity_sha256=identity_sha_by_key[
                destination_key
            ],
        )
        bindings[destination_key] = binding
        destination_index = index_by_key[destination_key]
        donor_index = index_by_key[donor_key]
        donor_binding_sha = axis_records[donor_index][
            "dino_content_binding_sha256"
        ]
        pair_receipts.append(
            {
                "destination_axis_index": destination_index,
                "destination_candidate_key": destination_key,
                "destination_physical_gallery_row": (
                    destination.physical_gallery_row
                ),
                "destination_corrected_identity_sha256": (
                    identity_sha_by_key[destination_key]
                ),
                "donor_axis_index": donor_index,
                "donor_candidate_key": donor_key,
                "donor_physical_gallery_row": donor.physical_gallery_row,
                "donor_corrected_identity_sha256": identity_sha_by_key[
                    donor_key
                ],
                "donor_dino_content_binding_sha256": donor_binding_sha,
                "destination_p_lock_input_sha256_by_direction": {
                    direction: destination_lock.p_lock_record_sha256_by_direction[
                        direction
                    ]
                    for direction in FIXED_DIRECTIONS
                },
                "destination_p_lock_output_sha256_by_direction": {
                    direction: binding.p_lock_record_sha256_by_direction[
                        direction
                    ]
                    for direction in FIXED_DIRECTIONS
                },
                "donor_content": _candidate_content(donor),
                "controlled_content": _candidate_content(controlled),
            }
        )
    input_p_locks = [
        row["p_lock_record_sha256_by_direction"] for row in axis_records
    ]
    output_native_axis = [native_axis[source] for source in order]
    payload: dict[str, object] = {
        "schema_version": "rc_dino_rcde_track_r_full_c128_c_dino_pair_v1_20260824",
        "control_name": "C_DINO_V",
        "namespace": C_DINO_NAMESPACE,
        "query_id": bundle.query_id,
        "target_free": True,
        "label_target_rival_read_count": 0,
        "scientific_reduction_count": 0,
        "candidate_count": 128,
        "physical_gallery_row_count": 5413,
        "pair_member_count": 2,
        "pair_member_keys": list(pair),
        "candidate_axis_sha256": canonical_sha256(
            [lock.candidate.physical_gallery_row for lock in canonical]
        ),
        "axis_records": axis_records,
        "destination_to_source": list(order),
        "pair_bindings": pair_receipts,
        "changed_variable": "DINO_REFERENCE_NATIVE_CONTENT_BINDING_ACROSS_FULL_C128",
        "fixed_point_free": True,
        "corrected_identity_disjoint": True,
        "pair_member_donor_exclusion": True,
        "candidate_axis_preserved": True,
        "p_lock_preserved": True,
        "full_native_content_binding_permutation": True,
        "input_p_lock_population_sha256": canonical_sha256(input_p_locks),
        "output_p_lock_population_sha256": canonical_sha256(input_p_locks),
        "input_native_content_axis_sha256": canonical_sha256(native_axis),
        "output_native_content_multiset_sha256": canonical_sha256(
            sorted(canonical_sha256(item) for item in output_native_axis)
        ),
        "input_native_content_multiset_sha256": canonical_sha256(
            sorted(canonical_sha256(item) for item in native_axis)
        ),
        "full_axis_planning_candidate_count": 128,
        "materialized_control_candidate_count": 2,
        "maximum_device_transfer_candidate_count": 2,
        "model_load_count": 0,
        "model_forward_count": 0,
        "model_backward_count": 0,
        "model_update_count": 0,
    }
    require(
        payload["input_native_content_multiset_sha256"]
        == payload["output_native_content_multiset_sha256"],
        "independent C_DINO content multiset drift",
    )
    payload["logical_sha256"] = canonical_sha256(payload)
    return bindings, payload


def model(path: Path) -> tuple[DINO_RCDE_V1_2, str]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(
        payload.get("schema_version") == "rc_dino_rcde_sr0_mt_v_checkpoint_v1"
        and payload.get("status") == "DINO_RCDE_TRACK_R_RELATIVE_CHECKPOINT_COMPLETE"
        and payload.get("outer_fold") == 2
        and payload.get("update") == 2048
        and payload.get("relative_pair_loss_only") is True,
        "V checkpoint drift",
    )
    value = DINO_RCDE_V1_2()
    value.load_state_dict(payload["model_state_dict"], strict=True)
    value.eval()
    state = state_dict_sha256(value)
    require(state == payload.get("final_state_sha256"), "V state drift")
    return value, state


def query_to_device(query: object, device: torch.device) -> object:
    return replace(
        query,
        layers=query.layers.to(device),
        valid_patch_mask=query.valid_patch_mask.to(device),
    )


class _PRegionalPoison:
    def __init__(self, candidate: object) -> None:
        self.candidate = candidate

    @property
    def direction_locks(self):
        raise V124E1IndependentError(
            "independent ALL_PATCH attempted P/regional access"
        )


def all_patch_only(
    model_value: torch.nn.Module,
    query: object,
    bindings: Mapping[str, object],
    pair: tuple[str, str],
):
    poison = {
        key: _PRegionalPoison(bindings[key].candidate) for key in pair  # type: ignore[attr-defined]
    }
    return decoder_v2.decode_all_patch_pair_v2(
        model_value, query, poison, pair[0], pair[1]  # type: ignore[arg-type]
    )


def ordinary_regional(
    model_value: torch.nn.Module,
    query: object,
    locks: Mapping[str, object],
    pair: tuple[str, str],
) -> tuple[object, object]:
    left, right = runtime_v2._v1._validate_decode_inputs(
        query, locks, pair[0], pair[1]  # type: ignore[arg-type]
    )
    model_sha, comparator_sha, _ = runtime_v2.bind_runtime_lineage(model_value)
    adapter = runtime_v2.DeviceMaskRuntimeAdapter(
        model_value, model_sha, comparator_sha
    )
    result = []
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
        result.append(
            runtime_v2._reduce_arm(
                arm_name, pair[0], pair[1], forward, reverse
            )
        )
    return tuple(result)  # type: ignore[return-value]


def c_dino_regional(
    model_value: torch.nn.Module,
    query: object,
    locks: Mapping[str, object],
    pair: tuple[str, str],
) -> tuple[tuple[object, object], Mapping[str, str]]:
    require(set(locks) == set(pair) and len(locks) == 2, "C_DINO pair binding drift")
    left, right = locks[pair[0]], locks[pair[1]]
    model_sha, comparator_sha, _ = runtime_v2.bind_runtime_lineage(model_value)
    adapter = runtime_v2.DeviceMaskRuntimeAdapter(
        model_value, model_sha, comparator_sha
    )
    transport = {
        ARM_ALL_PATCH: decoder_v2._transport_receipt_sha256(
            left_key=pair[0],
            right_key=pair[1],
            arm_name=ARM_ALL_PATCH,
            receipts=(),
        )
    }
    result = []
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
                term, local = controls_v1._c_dino_paired_term(
                    adapter, query, left, right, direction=direction
                )
                decoder_v2._validate_local_transport_term(
                    left,
                    right,
                    direction=direction,
                    term=term,
                    receipts=local,
                )
                forward.append(term)
                receipts.extend(local)
                term, local = controls_v1._c_dino_paired_term(
                    adapter, query, right, left, direction=direction
                )
                decoder_v2._validate_local_transport_term(
                    right,
                    left,
                    direction=direction,
                    term=term,
                    receipts=local,
                )
                reverse.append(term)
                receipts.extend(local)
        result.append(
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
    return tuple(result), transport  # type: ignore[return-value]


def assemble(
    model_value: torch.nn.Module,
    pair: tuple[str, str],
    all_patch: object,
    regional: tuple[object, object],
) -> ThreeArmFixedDenominatorEvidenceV1:
    model_sha, comparator_sha, reducer_sha = runtime_v2.bind_runtime_lineage(
        model_value
    )
    value = ThreeArmFixedDenominatorEvidenceV1(
        left_candidate_key=pair[0],
        right_candidate_key=pair[1],
        model_checkpoint_sha256=model_sha,
        pair_comparator_sha256=comparator_sha,
        reducer_sha256=reducer_sha,
        binding_mode=BINDING_REAL,
        arms=(all_patch, *regional),  # type: ignore[arg-type]
    )
    runtime_v2._assert_matched_regional_structure(value)
    return value


def _term(orientation: str, value: object) -> dict[str, object]:
    receipts = [
        {
            "root_ordinal": receipt.root_ordinal,
            "owner_component_source_root_ordinal": receipt.owner_component_source_root_ordinal,
            "opponent_component_source_root_ordinal": receipt.opponent_component_source_root_ordinal,
            "binding_mode": receipt.binding_mode,
            "query_mask_sha256": receipt.query_mask_sha256,
            "owner_reference_mask_sha256": receipt.owner_reference_mask_sha256,
            "opponent_reference_mask_sha256": receipt.opponent_reference_mask_sha256,
        }
        for receipt in value.decode_receipts
    ]
    return {
        "orientation": orientation,
        "direction": value.direction,
        "owner_candidate_key": value.owner_candidate_key,
        "opponent_candidate_key": value.opponent_candidate_key,
        "status": value.status,
        "query_mask_sha256": tensor_sha256(value.query_mask),
        "patch_evidence_sha256": tensor_sha256(value.patch_evidence),
        "scalar_contributions_sha256": tensor_sha256(value.scalar_contributions),
        "logit_sha256": tensor_sha256(value.logit),
        "decoded_root_ordinals": list(value.decoded_root_ordinals),
        "decode_receipts": receipts,
        "decoder_candidate_call_count": 2 * len(receipts),
    }


def records(
    family_evidence: Mapping[str, object],
    transports: Mapping[str, str],
) -> list[dict[str, object]]:
    output = []
    for family in FAMILY_SEQUENCE:
        evidence = family_evidence[family]
        arms = evidence.by_name()
        require(tuple(arms) == ARM_SEQUENCE, "independent arm order drift")
        for arm_name in ARM_SEQUENCE:
            arm = arms[arm_name]
            terms = [
                *(_term("FORWARD", row) for row in arm.forward_by_direction),
                *(_term("REVERSE", row) for row in arm.reverse_by_direction),
            ]
            require(len(terms) == 4, "independent direction count drift")
            row: dict[str, object] = {
                "family": family,
                "arm_name": arm_name,
                "left_candidate_key": arm.left_candidate_key,
                "right_candidate_key": arm.right_candidate_key,
                "model_checkpoint_sha256": evidence.model_checkpoint_sha256,
                "pair_comparator_sha256": evidence.pair_comparator_sha256,
                "reducer_sha256": evidence.reducer_sha256,
                "binding_mode": evidence.binding_mode,
                "scalar_contributions_sha256": tensor_sha256(arm.scalar_contributions),
                "pair_score_sha256": tensor_sha256(arm.logit),
                "direction_terms": terms,
                "direction_sequence_sha256": canonical_sha256(terms),
                "c_dino_transport_receipt_sha256": (
                    transports[arm_name] if family == "C_DINO_V" else None
                ),
            }
            row["record_sha256"] = canonical_sha256(row)
            output.append(row)
    require(len(output) == 9 and sum(len(row["direction_terms"]) for row in output) == 36, "independent 9/36 coverage drift")
    return output


def tensor_payloads(
    family_evidence: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    payloads = {}
    hashes = {}
    for family in FAMILY_SEQUENCE:
        arms = family_evidence[family].by_name()
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
                    value.detach().cpu().contiguous()
                    for value in arm.signed_term_logits
                ),
                "term_query_masks": tuple(
                    value.query_mask.detach().cpu().contiguous()
                    for value in terms
                ),
                "term_patch_evidence": tuple(
                    value.patch_evidence.detach().cpu().contiguous()
                    for value in terms
                ),
                "term_scalar_contributions": tuple(
                    value.scalar_contributions.detach().cpu().contiguous()
                    for value in terms
                ),
                "term_logits": tuple(
                    value.logit.detach().cpu().contiguous() for value in terms
                ),
            }
            payloads[key] = payload
            hashes[key] = {
                name: (
                    [tensor_sha256(item) for item in item_or_items]
                    if isinstance(item_or_items, tuple)
                    else tensor_sha256(item_or_items)
                )
                for name, item_or_items in payload.items()
            }
    require(len(payloads) == len(hashes) == 9, "independent tensor payload drift")
    return payloads, hashes


def _query_request_record(bundle: VQueryBundle) -> dict[str, object]:
    query = bundle.query
    require(token_tensor_sha256(query.layers) == query.tokens_sha256, "query token drift")
    return {
        "source_image_sha256": query.source_image_sha256,
        "source_key": query.source_key,
        "cache_payload_sha256": query.cache_payload_sha256,
        "geometry_record_sha256": query.geometry_record_sha256,
        "tokens_sha256": query.tokens_sha256,
        "layers_exact_sha256": tensor_sha256(query.layers),
        "valid_patch_mask_exact_sha256": tensor_sha256(
            query.valid_patch_mask
        ),
        "grid_shape": list(query.grid_shape),
        "dtype": str(query.layers.dtype),
        "device": str(query.layers.device),
    }


def _candidate_request_record(binding: object) -> dict[str, object]:
    candidate = binding.candidate  # type: ignore[attr-defined]
    require(
        token_tensor_sha256(candidate.layers) == candidate.tokens_sha256,
        "candidate token drift",
    )
    return {
        "candidate_key": candidate.candidate_key,
        "physical_gallery_row": candidate.physical_gallery_row,
        "source_image_sha256": candidate.source_image_sha256,
        "source_key": candidate.source_key,
        "cache_payload_sha256": candidate.cache_payload_sha256,
        "geometry_record_sha256": candidate.geometry_record_sha256,
        "tokens_sha256": candidate.tokens_sha256,
        "layers_exact_sha256": tensor_sha256(candidate.layers),
        "valid_patch_mask_exact_sha256": tensor_sha256(
            candidate.valid_patch_mask
        ),
        "grid_shape": list(candidate.grid_shape),
        "dtype": str(candidate.layers.dtype),
        "device": str(candidate.layers.device),
    }


def _axis_request_record(bundle: VQueryBundle) -> dict[str, object]:
    axis = canonical_candidate_axis(bundle.locks)
    return {
        "query_id": bundle.query_id,
        "execution_ordinal": bundle.execution_ordinal,
        "outer_fold": bundle.outer_fold,
        "candidate_axis_sha256": bundle.candidate_axis_sha256,
        "candidate_keys": list(axis),
        "query": _query_request_record(bundle),
        "candidates": [
            _candidate_request_record(bundle.locks[key]) for key in axis
        ],
    }


def _request_record(
    bundle: VQueryBundle,
    axis_record: Mapping[str, object],
    pair: tuple[str, str],
    model_sha: str,
    comparator_sha: str,
    reducer_sha: str,
) -> dict[str, object]:
    axis = canonical_candidate_axis(bundle.locks)
    positions = {key: index for index, key in enumerate(axis)}
    return {
        "schema_version": "rc_dino_rcde_track_r_phase_b_all_patch_v1_20260824",
        "arm_name": ARM_ALL_PATCH,
        "pair_member_keys": list(pair),
        "pair_member_axis_positions": [positions[key] for key in pair],
        "directions": list(FIXED_DIRECTIONS),
        "dino_axis_sha256": canonical_sha256(axis_record),
        "query": _query_request_record(bundle),
        "left_candidate": _candidate_request_record(bundle.locks[pair[0]]),
        "right_candidate": _candidate_request_record(bundle.locks[pair[1]]),
        "model_checkpoint_sha256": model_sha,
        "pair_comparator_sha256": comparator_sha,
        "reducer_sha256": reducer_sha,
        "adapter_source_sha256": hashlib.sha256(
            inspect.getsource(runtime_v2.DeviceMaskRuntimeAdapter).encode()
        ).hexdigest(),
        "single_arm_entry_source_sha256": hashlib.sha256(
            inspect.getsource(decoder_v2.decode_all_patch_pair_v2).encode()
        ).hexdigest(),
        "direct_decoder_source_sha256": hashlib.sha256(
            inspect.getsource(runtime_v2.decode_direct_arm_term).encode()
        ).hexdigest(),
        "all_patch_term_source_sha256": hashlib.sha256(
            inspect.getsource(runtime_v2._v1._all_patch_term).encode()
        ).hexdigest(),
        "numerical_policy": "SAME_DEVICE_DTYPE_EXACT_BYTES_ZERO_TOLERANCE_V1",
    }


def _phase_b_direction(arm: object) -> list[dict[str, object]]:
    output = []
    terms = (
        *(("FORWARD", item) for item in arm.forward_by_direction),
        *(("REVERSE", item) for item in arm.reverse_by_direction),
    )
    for orientation, term in terms:
        require(
            term.arm_name == ARM_ALL_PATCH
            and term.decoded_root_ordinals == (-1,),
            "non-ALL_PATCH Phase-B term",
        )
        output.append(
            {
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
                "decode_receipts": [
                    {
                        "root_ordinal": item.root_ordinal,
                        "owner_component_source_root_ordinal": item.owner_component_source_root_ordinal,
                        "opponent_component_source_root_ordinal": item.opponent_component_source_root_ordinal,
                        "binding_mode": item.binding_mode,
                        "query_mask_sha256": item.query_mask_sha256,
                        "owner_reference_mask_sha256": item.owner_reference_mask_sha256,
                        "opponent_reference_mask_sha256": item.opponent_reference_mask_sha256,
                    }
                    for item in term.decode_receipts
                ],
            }
        )
    return output


def phase_b_receipt(
    *,
    model_value: torch.nn.Module,
    real: VQueryBundle,
    c_col: VQueryBundle,
    pair: tuple[str, str],
    real_evidence: object,
    c_col_evidence: object,
) -> dict[str, str]:
    model_sha, comparator_sha, reducer_sha = runtime_v2.bind_runtime_lineage(
        model_value
    )
    real_axis = _axis_request_record(real)
    control_axis = _axis_request_record(c_col)
    require(real_axis == control_axis, "independent Phase-B DINO axis drift")
    require(
        torch.equal(real.query.layers, c_col.query.layers)
        and torch.equal(
            real.query.valid_patch_mask, c_col.query.valid_patch_mask
        ),
        "independent Phase-B query bytes drift",
    )
    for key in canonical_candidate_axis(real.locks):
        first = real.locks[key].candidate
        second = c_col.locks[key].candidate
        require(
            torch.equal(first.layers, second.layers)
            and torch.equal(first.valid_patch_mask, second.valid_patch_mask),
            "independent Phase-B candidate bytes drift",
        )
    real_request = _request_record(
        real, real_axis, pair, model_sha, comparator_sha, reducer_sha
    )
    control_request = _request_record(
        c_col, control_axis, pair, model_sha, comparator_sha, reducer_sha
    )
    real_arm = real_evidence.by_name()[ARM_ALL_PATCH]
    control_arm = c_col_evidence.by_name()[ARM_ALL_PATCH]
    real_terms = (*real_arm.forward_by_direction, *real_arm.reverse_by_direction)
    control_terms = (
        *control_arm.forward_by_direction,
        *control_arm.reverse_by_direction,
    )
    require(
        all(
            first.status == second.status
            and first.direction == second.direction
            and first.owner_candidate_key == second.owner_candidate_key
            and first.opponent_candidate_key == second.opponent_candidate_key
            and first.decoded_root_ordinals == second.decoded_root_ordinals
            and first.decode_receipts == second.decode_receipts
            and torch.equal(first.query_mask, second.query_mask)
            and torch.equal(first.patch_evidence, second.patch_evidence)
            and torch.equal(
                first.scalar_contributions, second.scalar_contributions
            )
            and torch.equal(first.logit, second.logit)
            for first, second in zip(
                real_terms, control_terms, strict=True
            )
        )
        and torch.equal(
            real_arm.scalar_contributions, control_arm.scalar_contributions
        )
        and torch.equal(real_arm.logit, control_arm.logit),
        "independent Phase-B ALL_PATCH bytes drift",
    )
    return {
        "real_request_sha256": canonical_sha256(real_request),
        "control_request_sha256": canonical_sha256(control_request),
        "real_patch_tensor_sha256": tensor_sha256(real_arm.scalar_contributions),
        "control_patch_tensor_sha256": tensor_sha256(control_arm.scalar_contributions),
        "real_direction_sequence_sha256": canonical_sha256(
            _phase_b_direction(real_arm)
        ),
        "control_direction_sequence_sha256": canonical_sha256(
            _phase_b_direction(control_arm)
        ),
        "real_pair_score_sha256": tensor_sha256(real_arm.logit),
        "control_pair_score_sha256": tensor_sha256(control_arm.logit),
    }


def _exclusive(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "immutable validation output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
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


def _seal_result(path: Path, value: Mapping[str, Any], authority_sha256: str) -> None:
    _exclusive(path, value)
    receipt = {
        "schema_version": "rc_dino_rcde_track_r_v124_e1_prejoin_receipt_v1_20260826",
        "status": "DINO_RCDE_TRACK_R_V124_E1_PREJOIN_RECEIPT_READY",
        "role": "E1_INDEPENDENT_VALIDATION_RESULT",
        "payload_path": path.name,
        "payload_bytes": path.stat().st_size,
        "payload_sha256": file_sha256(path),
        "payload_logical_sha256": value["logical_sha256"],
    }
    receipt["logical_sha256"] = logical_sha256(receipt)
    receipt_path = path.with_name(path.name + ".receipt.json")
    _exclusive(receipt_path, receipt)
    commit = {
        "schema_version": "rc_dino_rcde_track_r_v124_e1_prejoin_commit_v1_20260826",
        "status": "DINO_RCDE_TRACK_R_V124_E1_PREJOIN_PAYLOAD_COMMITTED",
        "authority_sha256": authority_sha256,
        "role": "E1_INDEPENDENT_VALIDATION_RESULT",
        "payload_path": path.name,
        "payload_sha256": file_sha256(path),
        "receipt_path": receipt_path.name,
        "receipt_sha256": file_sha256(receipt_path),
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    commit["logical_sha256"] = logical_sha256(commit)
    _exclusive(path.with_name(path.name + ".commit.json"), commit)


def _tensor_payloads_equal(
    observed: Mapping[str, object], replay: Mapping[str, object]
) -> bool:
    if set(observed) != set(replay):
        return False
    for key in replay:
        first = observed[key]
        second = replay[key]
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            return False
        if set(first) != set(second):
            return False
        for name in second:
            left = first[name]
            right = second[name]
            if isinstance(right, tuple):
                if not isinstance(left, tuple) or len(left) != len(right):
                    return False
                if not all(
                    torch.equal(torch.as_tensor(a), torch.as_tensor(b))
                    for a, b in zip(left, right, strict=True)
                ):
                    return False
            elif not torch.equal(
                torch.as_tensor(left), torch.as_tensor(right)
            ):
                return False
    return True


def run(
    *,
    authority_path: Path,
    authority_sha256: str,
    artifact_path: Path,
    producer_result_path: Path,
    output: Path = OUTPUT,
) -> dict[str, Any]:
    require(Path.cwd().resolve() == ROOT.resolve(), "independent cwd must equal RC root")
    authority = read_authority(authority_path, authority_sha256)
    observed = torch.load(artifact_path, map_location="cpu", weights_only=False, mmap=True)
    producer = json.loads(producer_result_path.read_text(encoding="ascii"))
    require(isinstance(observed, Mapping), "producer artifact mapping absent")
    _validate_triple(artifact_path, observed, role="E1_PREJOIN_ARTIFACT", authority_sha256=authority_sha256)
    _validate_triple(producer_result_path, producer, role="E1_PRODUCER_RESULT", authority_sha256=authority_sha256)
    require(
        observed.get("schema_version") == ARTIFACT_SCHEMA
        and observed.get("status") == PRODUCER_STATUS
        and observed.get("authority_sha256") == authority_sha256
        and observed.get("logical_sha256")
        == canonical_sha256(
            {
                key: item
                for key, item in observed.items()
                if key not in {"logical_sha256", "tensor_payloads"}
            }
        )
        and producer.get("status") == PRODUCER_STATUS
        and producer.get("authority_sha256") == authority_sha256
        and producer.get("artifact_sha256") == file_sha256(artifact_path)
        and producer.get("logical_sha256") == logical_sha256(producer),
        "producer artifact/result envelope drift",
    )
    phase_a = torch.load(_bound_path(authority, "phase_a_artifact"), map_location="cpu", weights_only=False, mmap=True)
    require(isinstance(phase_a, Mapping) and isinstance(phase_a.get("real_bundle"), VQueryBundle), "Phase-A REAL source drift")
    real = phase_a["real_bundle"]
    pair_row, pair = execution0_pair(_bound_path(authority, "role_free_pair_manifest"))
    query_geometry_sha, reference_geometry_by_row = canonical_geometry(
        _bound_path(authority, "canonical_geometry_payload"), real
    )
    c_col_locks = reconstruct_c_col(
        phase_a,
        pair,
        canonical_query_geometry_sha256=query_geometry_sha,
        canonical_reference_geometry_by_row=reference_geometry_by_row,
    )
    c_col_view = c_col_axis_view(phase_a)
    c_dino, c_dino_receipt = reconstruct_c_dino(
        real,
        pair,
        identities(_bound_path(authority, "gallery_cache"), real),
    )
    model_value, state = model(_bound_path(authority, "fold2_v_checkpoint"))
    require(state == authority.get("v_checkpoint_state_sha256"), "authority V state drift")
    require(
        torch.cuda.is_available() and torch.cuda.device_count() >= 1,
        "independent E1 replay requires one visible accelerated GPU",
    )
    with torch.no_grad():
        real_all = all_patch_only(model_value, real.query, real.locks, pair)
        c_dino_all = all_patch_only(model_value, real.query, c_dino, pair)
        c_col_all = all_patch_only(
            model_value, real.query, c_col_view.locks, pair
        )
    gpu = torch.device("cuda:0")
    model_value.to(gpu)
    query_gpu = query_to_device(real.query, gpu)
    real_pair_gpu = move_locks_to_device(
        {key: real.locks[key] for key in pair}, gpu
    )
    c_dino_gpu = move_c_dino_locks_to_device(c_dino, gpu)
    c_col_gpu = move_locks_to_device(c_col_locks, gpu)
    with torch.no_grad():
        real_regional = ordinary_regional(
            model_value, query_gpu, real_pair_gpu, pair
        )
        c_dino_regional_arms, transports = c_dino_regional(
            model_value, query_gpu, c_dino_gpu, pair
        )
        c_col_regional = ordinary_regional(
            model_value, query_gpu, c_col_gpu, pair
        )
    real_evidence = assemble(
        model_value, pair, real_all, real_regional
    )
    c_dino_evidence = assemble(
        model_value, pair, c_dino_all, c_dino_regional_arms
    )
    c_col_evidence = assemble(
        model_value, pair, c_col_all, c_col_regional
    )
    require(state_dict_sha256(model_value) == state, "independent replay mutated V checkpoint")
    replay_records = records(
        {"REAL": real_evidence, "C_DINO_V": c_dino_evidence, "C_COL_P": c_col_evidence},
        transports,
    )
    replay_tensor_payloads, replay_tensor_hashes = tensor_payloads(
        {
            "REAL": real_evidence,
            "C_DINO_V": c_dino_evidence,
            "C_COL_P": c_col_evidence,
        }
    )
    decode_candidate_call_count = sum(
        term["decoder_candidate_call_count"]
        for record in replay_records
        for term in record["direction_terms"]
    )
    replay_phase_b = phase_b_receipt(
        model_value=model_value,
        real=real,
        c_col=c_col_view,
        pair=pair,
        real_evidence=real_evidence,
        c_col_evidence=c_col_evidence,
    )
    bound_phase_b = json.loads(_bound_path(authority, "phase_b_result").read_text(encoding="ascii"))
    require(
        observed.get("records") == replay_records
        and observed.get("record_sha256_sequence")
        == [row["record_sha256"] for row in replay_records]
        and observed.get("tensor_payload_hashes") == replay_tensor_hashes
        and isinstance(observed.get("tensor_payloads"), Mapping)
        and _tensor_payloads_equal(
            observed["tensor_payloads"], replay_tensor_payloads
        )
        and observed.get("c_dino_receipt") == dict(c_dino_receipt)
        and observed.get("c_dino_receipt_logical_sha256")
        == c_dino_receipt["logical_sha256"]
        and observed.get("phase_b_invariance_receipt") == replay_phase_b
        and all(bound_phase_b.get(key) == replay_phase_b[key] for key in PHASE_B_FIELDS)
        and observed.get("pair_member_keys") == list(pair)
        and observed.get("pair_address_sha256_sequence")
        == [row["address_sha256"] for row in pair_row["members"]]
        and observed.get("pair_arm_record_count") == 9
        and observed.get("directional_term_count") == 36,
        "producer/independent E1 ordered replay mismatch",
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_level": "FIXED_NATURAL_OPENED_CANARY_TARGET_FREE_PREJOIN_ONLY",
        "authority_sha256": authority_sha256,
        "producer_artifact_sha256": file_sha256(artifact_path),
        "producer_result_sha256": file_sha256(producer_result_path),
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
        "independent_checked_record_count": 9,
        "independent_checked_directional_term_count": 36,
        "unchecked_record_count": 0,
        "producer_independent_exact_match": True,
        "c_dino_full_c128_reconstructed": True,
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
        "validation_pass": True,
    }
    result["logical_sha256"] = logical_sha256(result)
    _seal_result(output, result, authority_sha256)
    print(json.dumps({"status": STATUS, "logical_sha256": result["logical_sha256"]}, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, default=AUTHORITY)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--producer-result", type=Path, default=PRODUCER_RESULT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(
        authority_path=args.authority,
        authority_sha256=args.authority_sha256,
        artifact_path=args.artifact,
        producer_result_path=args.producer_result,
        output=args.output,
    )


if __name__ == "__main__":
    main()
