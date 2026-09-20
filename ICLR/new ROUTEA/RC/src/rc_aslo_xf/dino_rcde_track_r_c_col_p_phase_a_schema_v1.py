"""Frozen serialization schema for the V124 C_COL_P Phase-A artifact.

The module contains no C_COL transform, P scorer, model loader, or file I/O.
Both producer and independent validator may consume these field sets without
sharing an implementation of the intervention being tested.
"""

from __future__ import annotations

from typing import Any, Mapping

from .dino_rcde_sr0_mt_p_runtime_v1 import canonical_sha256


SCHEMA_VERSION = (
    "rc_dino_rcde_track_r_v124_c_col_p_p_v2_phase_a_artifact_v1_20260824"
)
STATUS = "DINO_RCDE_TRACK_R_V124_C_COL_P_P_V2_PHASE_A_READY"
PHASE = "PHASE_A_P_V2_LOCK_REBUILD_ONLY"
OPENED594_ROLE = "DIAGNOSTIC_ONLY_NOT_CONFIRMATION"
CANDIDATE_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_p_v2_phase_a_candidate_v1_20260824"
)
VALIDATION_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_p_v2_phase_a_independent_validation_v1_20260824"
)
VALIDATION_STATUS = (
    "DINO_RCDE_TRACK_R_V124_C_COL_P_P_V2_PHASE_A_INDEPENDENT_VALIDATION_PASS"
)

HEAVY_FIELDS = frozenset(
    {"real_bundle", "real_outer_p_v2_records", "control_candidates"}
)

ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "phase",
        "target_free",
        "opened594_role",
        "query_id",
        "historical_query_ordinal",
        "execution_ordinal",
        "query_source_image_sha256",
        "source_fold",
        "candidate_count",
        "direction_count",
        "selected_p_head_count",
        "record_count",
        "candidate_axis_sha256",
        "p_fit_id",
        "p_checkpoint_sha256",
        "p_training_manifest_sha256",
        "real_bundle",
        "real_bundle_fingerprint",
        "real_dino_axis_fingerprint",
        "control_dino_axis_fingerprint",
        "dino_axis_byte_identical",
        "real_outer_p_v2_records",
        "real_record_sha256_sequence",
        "control_candidates",
        "control_record_sha256_sequence",
        "control_candidate_wrapper_sequence_sha256",
        "strict_core_receipt",
        "source_bindings",
        "access_audit",
        "scientific_GO_or_NO_GO",
        "automatic_stage_advance",
        "next_authorized_stage",
        "logical_sha256",
    }
)

CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_position",
        "candidate_key",
        "candidate_physical_row",
        "candidate_reference_source_sha256",
        "candidate",
        "candidate_dino_fingerprint",
        "donor_position",
        "donor_candidate_key",
        "donor_physical_row",
        "donor_source_sha256",
        "destination_colnomic_tokens_sha256",
        "donor_colnomic_tokens_sha256",
        "destination_colnomic_valid_mask_sha256",
        "donor_colnomic_valid_mask_sha256",
        "destination_colnomic_grid",
        "donor_colnomic_grid",
        "direction_records",
        "direction_record_sha256_sequence",
        "logical_sha256",
    }
)

SOURCE_BINDING_FIELDS = frozenset(
    {
        "source_manifest_sha256",
        "source_manifest_validation_sha256",
        "source_artifact_sha256",
        "source_artifact_validation_sha256",
        "p_v2_lock_artifact_sha256",
        "p_v2_lock_strict_validation_sha256",
        "p_checkpoint_sha256",
        "p_training_manifest_sha256",
        "membership_sha256",
        "fold_schedule_sha256",
        "geometry_payload_sha256",
        "geometry_receipt_sha256",
        "redacted_schedule_sha256",
        "redacted_cache_index_sha256",
        "dino_model_sha256",
        "pair_manifest_sha256",
        "pair_manifest_validation_sha256",
        "gallery_cache_sha256",
        "identity_registry_sha256",
        "identity_contract_sha256",
        "phase_a_authority_sha256",
        "preflight_logical_sha256",
        "pair_sha256",
    }
)

ACCESS_AUDIT_FIELDS = frozenset(
    {
        "corrected_identity_axis_read_count",
        "corrected_identity_consumed_count",
        "corrected_identity_use",
        "target_label_read_count",
        "target_rival_read_count",
        "rank_score_outcome_read_count",
        "training_backward_update_count",
        "v_model_load_count",
        "v_model_forward_count",
        "all_patch_forward_count",
        "h0_track_h_read_count",
        "scientific_reduction_count",
        "automatic_submit_count",
    }
)


def artifact_logical_sha256(value: Mapping[str, Any]) -> str:
    """Hash the metadata envelope while heavy objects use sealed fingerprints."""

    return canonical_sha256(
        {
            key: item
            for key, item in value.items()
            if key not in HEAVY_FIELDS and key != "logical_sha256"
        }
    )


def candidate_logical_sha256(value: Mapping[str, Any]) -> str:
    """Hash one wrapper through its DINO and two P-V2 record fingerprints."""

    return canonical_sha256(
        {
            key: item
            for key, item in value.items()
            if key not in {"candidate", "direction_records", "logical_sha256"}
        }
    )


__all__ = [
    "SCHEMA_VERSION",
    "STATUS",
    "PHASE",
    "OPENED594_ROLE",
    "CANDIDATE_SCHEMA",
    "VALIDATION_SCHEMA",
    "VALIDATION_STATUS",
    "HEAVY_FIELDS",
    "ARTIFACT_FIELDS",
    "CANDIDATE_FIELDS",
    "SOURCE_BINDING_FIELDS",
    "ACCESS_AUDIT_FIELDS",
    "artifact_logical_sha256",
    "candidate_logical_sha256",
]
