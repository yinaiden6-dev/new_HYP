#!/usr/bin/env python3
"""Target-free exhaustive-C128 three-arm SR0-MT inference shard.

One invocation handles one immutable query bundle and a contiguous range of
canonical unordered candidate pairs.  It never accepts a target/rival argument.
PAIR_SWAP is derived as exact negation of each canonical pair and candidate
reorder is a separately validated container operation, not a binding shuffle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RC_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from rc_aslo_xf.dino_rcde_cw1_multitile_superregion_v2 import (  # noqa: E402
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
)
from rc_aslo_xf.dino_rcde_sr0_mt_controls_v1 import (  # noqa: E402
    MANDATORY_CONTROLS,
    candidate_reorder,
    pair_swap_record,
)
from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    PAIR_COUNT,
    VPairEpisode,
    VQueryBundle,
    assert_target_free_payload,
    canonical_candidate_axis,
    canonical_unordered_pairs,
    decode_pair,
    execution_manifest_arguments,
    execution_manifest_argv,
    serialize_arm_evidence,
    state_dict_sha256,
    tensor_sha256,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (  # noqa: E402
    validate_natural_authority,
)
from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2  # noqa: E402


INPUT_SCHEMA = "rc_dino_rcde_sr0_mt_three_arm_query_input_v1"
OUTPUT_SCHEMA = "rc_dino_rcde_sr0_mt_pair_shard_v1_20260815"
OUTPUT_STATUS = "RCDE_SR0_MT_PAIR_SHARD_COMPLETE"
CONTROL_INPUT_SCHEMA = "rc_dino_rcde_sr0_mt_control_score_input_v1"
RUNTIME_TO_SCIENTIFIC = {
    ARM_ALL_PATCH: "ALL_PATCH_SAME_MODEL",
    ARM_QUERY_FULL_REFERENCE: "QUERY_REGION_FULL_REFERENCE_SAME_MODEL",
    ARM_QUERY_LOCAL_COMPONENTS: "PAIRED_QUERY_REFERENCE_REGION",
}
SCIENTIFIC_ARMS = tuple(RUNTIME_TO_SCIENTIFIC.values())
ZERO_AUDIT = {
    "target_or_label_read_count": 0,
    "target_insertions": 0,
    "candidate_mutation_count": 0,
    "postjoin_model_forward_count": 0,
    "target_selected_pair_forward_count": 0,
    "target_spatial_supervision_read_count": 0,
    "D1_score_rank_slot_winner_gap_read_count": 0,
    "opened_runtime_read_count": 0,
    "sealed_runtime_read_count": 0,
    "C8_runtime_read_count": 0,
    "S8_runtime_read_count": 0,
    "home_files_modified": 0,
    "prior_artifact_files_modified": 0,
}


class ThreeArmRunnerError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ThreeArmRunnerError("JSON artifact is not an object")
    return value


def safe_path(path: Path, *, must_exist: bool) -> Path:
    resolved = path.resolve()
    if RC_ROOT != resolved and RC_ROOT not in resolved.parents:
        raise ThreeArmRunnerError(f"path escapes RC root: {resolved}")
    if {"c8", "s8", "opened", "sealed"}.intersection(
        part.lower() for part in resolved.parts
    ):
        raise ThreeArmRunnerError(f"protected path requested: {resolved}")
    if must_exist and not resolved.exists():
        raise ThreeArmRunnerError(f"missing input: {resolved}")
    return resolved


def atomic_torch(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def f64_bits(value: float) -> str:
    number = float(value)
    if not torch.isfinite(torch.tensor(number, dtype=torch.float64)):
        raise ThreeArmRunnerError("attempted to serialize a nonfinite margin")
    return struct.pack(">d", number).hex()


def validate_authority(
    protocol_path: Path, authority_path: Path, authority_sha256: str
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    authority = read_json(authority_path)
    validate_natural_authority(
        authority_path,
        expected_file_sha256=authority_sha256,
        required_scope="natural_three_arm_scoring",
        execution_protocol_sha256=file_sha256(protocol_path),
    )
    required = (
        authority.get("model_load_authorized") is True
        and authority.get("model_forward_authorized") is True
        and authority.get("outer_heldout_target_free_forward_authorized") is True
        and authority.get("checkpoint_deserialization_authorized") is True
        and authority.get("automatic_stage_advance") is False
    )
    if not required:
        raise ThreeArmRunnerError("authority does not authorize target-free outer forward")
    if protocol.get("automatic_stage_advance") is not False:
        raise ThreeArmRunnerError("three-arm protocol permits automatic advancement")
    aliases = protocol.get("arm_alias_contract", {}).get("runtime_to_scientific")
    if aliases is not None and aliases != RUNTIME_TO_SCIENTIFIC:
        raise ThreeArmRunnerError("three-arm alias contract drift")
    return protocol


def validate_execution_manifest(
    path: Path,
    *,
    protocol: Path,
    authority: Path,
    authority_sha256: str,
    query_input: Path,
    checkpoint: Path,
    output: Path,
    contribution_payload: Path,
    control_input: Path,
    pair_start: int,
    pair_stop: int,
    device: str,
    streaming_chunk_size: int | None,
) -> str:
    expected = {
        "protocol": str(protocol),
        "authority": str(authority),
        "authority_sha256": authority_sha256,
        "query_input": str(query_input),
        "checkpoint": str(checkpoint),
        "output": str(output),
        "contribution_payload": str(contribution_payload),
        "control_input": str(control_input),
        "pair_start": pair_start,
        "pair_stop": pair_stop,
        "device": device,
        "streaming_chunk_size": streaming_chunk_size,
    }
    observed = dict(
        execution_manifest_arguments(path, expected_program=Path(__file__).name)
    )
    if observed != expected:
        raise ThreeArmRunnerError("execution manifest binding drift")
    return file_sha256(path)


def load_control_input(
    path: Path,
    *,
    bundle: VQueryBundle,
    checkpoint_sha256: str,
    pair_start: int,
    pair_stop: int,
) -> Mapping[int, Mapping[str, Mapping[str, Mapping[str, object]]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != CONTROL_INPUT_SCHEMA:
        raise ThreeArmRunnerError("control-score input schema drift")
    assert_target_free_payload(
        {key: value for key, value in payload.items() if key != "records"}
    )
    if (
        payload.get("query_id") != bundle.query_id
        or payload.get("candidate_axis_sha256") != bundle.candidate_axis_sha256
        or payload.get("checkpoint_sha256") != checkpoint_sha256
        or int(payload.get("pair_start", -1)) != pair_start
        or int(payload.get("pair_stop", -1)) != pair_stop
        or payload.get("inference_only") is not True
        or payload.get("used_in_training") is not False
    ):
        raise ThreeArmRunnerError("control-score input binding drift")
    raw = payload.get("records")
    if not isinstance(raw, Mapping):
        raise ThreeArmRunnerError("control-score records are absent")
    records: dict[int, Mapping[str, Mapping[str, Mapping[str, object]]]] = {}
    for ordinal in range(pair_start, pair_stop):
        value = raw.get(ordinal, raw.get(str(ordinal)))
        if not isinstance(value, Mapping) or tuple(value) != SCIENTIFIC_ARMS:
            raise ThreeArmRunnerError("control-score arm key/order drift")
        for arm in SCIENTIFIC_ARMS:
            controls = value[arm]
            if not isinstance(controls, Mapping) or tuple(controls) != MANDATORY_CONTROLS[:8]:
                raise ThreeArmRunnerError("control-score key/order drift")
            for name in MANDATORY_CONTROLS[:8]:
                item = controls[name]
                if not isinstance(item, Mapping) or item.get("status") not in {
                    "READY",
                    "NOT_APPLICABLE_CONTRACT",
                }:
                    raise ThreeArmRunnerError("control-score applicability drift")
                if item["status"] == "READY":
                    contribution = torch.as_tensor(item.get("scalar_contributions"))
                    margin = torch.as_tensor(item.get("margin"), dtype=torch.float64)
                    if (
                        contribution.ndim != 1
                        or not contribution.is_floating_point()
                        or not bool(torch.isfinite(contribution).all())
                        or margin.ndim != 0
                        or not bool(torch.isfinite(margin))
                        or not torch.allclose(
                            contribution.to(torch.float64).sum(), margin,
                            atol=1e-7, rtol=0,
                        )
                        or item.get("only_registered_variable_changed") is not True
                    ):
                        raise ThreeArmRunnerError("control-score numeric/isolation closure failed")
        records[ordinal] = value  # type: ignore[assignment]
    return records


def load_query_bundle(path: Path, device: torch.device) -> VQueryBundle:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != INPUT_SCHEMA:
        raise ThreeArmRunnerError("three-arm query input schema drift")
    assert_target_free_payload(
        {key: value for key, value in payload.items() if key != "bundle"}
    )
    bundle = payload.get("bundle")
    if isinstance(bundle, VPairEpisode) or not isinstance(bundle, VQueryBundle):
        raise ThreeArmRunnerError("training pair or noncanonical bundle entered prejoin")
    # Explicitly reconstruct token fields on device through a local, target-free
    # adapter.  P populations/masks stay canonical CPU.
    from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (  # noqa: PLC0415
        CandidatePLockV1,
    )
    from rc_aslo_xf.dino_rcde_cw1_multitile_vdecode_v1 import (  # noqa: PLC0415
        CandidateReferenceFieldV1,
        QueryTokenFieldV1,
    )

    query = QueryTokenFieldV1(
        layers=bundle.query.layers.to(device),
        grid_shape=bundle.query.grid_shape,
        valid_patch_mask=bundle.query.valid_patch_mask,
        source_image_sha256=bundle.query.source_image_sha256,
        source_key=bundle.query.source_key,
        cache_payload_sha256=bundle.query.cache_payload_sha256,
        geometry_record_sha256=bundle.query.geometry_record_sha256,
        tokens_sha256=bundle.query.tokens_sha256,
    )
    locks = {}
    for key, lock in bundle.locks.items():
        source = lock.candidate
        candidate = CandidateReferenceFieldV1(
            candidate_key=source.candidate_key,
            layers=source.layers.to(device),
            grid_shape=source.grid_shape,
            valid_patch_mask=source.valid_patch_mask,
            physical_gallery_row=source.physical_gallery_row,
            source_image_sha256=source.source_image_sha256,
            source_key=source.source_key,
            cache_payload_sha256=source.cache_payload_sha256,
            geometry_record_sha256=source.geometry_record_sha256,
            tokens_sha256=source.tokens_sha256,
        )
        locks[key] = CandidatePLockV1(
            candidate=candidate,
            direction_locks=lock.direction_locks,
        )
    return VQueryBundle(
        query_id=bundle.query_id,
        execution_ordinal=bundle.execution_ordinal,
        outer_fold=bundle.outer_fold,
        query=query,
        locks=locks,
        candidate_axis_sha256=bundle.candidate_axis_sha256,
    )


def load_model(path: Path, outer_fold: int, device: torch.device) -> tuple[DINO_RCDE_V1_2, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA
        or int(payload.get("outer_fold", -1)) != outer_fold
        or payload.get("same_checkpoint_all_three_arms_and_controls") is not True
    ):
        raise ThreeArmRunnerError("final fold V checkpoint contract drift")
    model = DINO_RCDE_V1_2().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if state_dict_sha256(model) != payload.get("final_state_sha256"):
        raise ThreeArmRunnerError("final fold V state hash drift")
    model.eval()
    return model, file_sha256(path)


def scientific_record(serialized: Mapping[str, object]) -> dict[str, object]:
    runtime_arms = serialized["arms"]
    if not isinstance(runtime_arms, Mapping) or set(runtime_arms) != set(
        RUNTIME_TO_SCIENTIFIC
    ):
        raise ThreeArmRunnerError("runtime did not emit exactly three arms")
    scientific = {
        RUNTIME_TO_SCIENTIFIC[name]: runtime_arms[name]
        for name in RUNTIME_TO_SCIENTIFIC
    }
    query_full = scientific["QUERY_REGION_FULL_REFERENCE_SAME_MODEL"]
    paired = scientific["PAIRED_QUERY_REFERENCE_REGION"]
    if not isinstance(query_full, Mapping) or not isinstance(paired, Mapping):
        raise ThreeArmRunnerError("regional arm record schema drift")
    if (
        query_full["forward_query_mask_sha256"]
        != paired["forward_query_mask_sha256"]
        or query_full["reverse_query_mask_sha256"]
        != paired["reverse_query_mask_sha256"]
    ):
        raise ThreeArmRunnerError("query-full and paired query masks are not byte-identical")
    output = dict(serialized)
    output["arms"] = scientific
    return output


def _mass_fields(contributions: torch.Tensor) -> dict[str, float]:
    values = torch.as_tensor(contributions, dtype=torch.float64).flatten()
    positive_values = values.clamp_min(0)
    negative_values = values.clamp_max(0)
    positive = float(positive_values.sum())
    negative = float(negative_values.sum())
    negative_magnitude = -negative
    total = positive + negative_magnitude
    cancellation = (
        0.0
        if total == 0
        else (total - abs(positive - negative_magnitude)) / total
    )
    return {
        "positive_mass": positive,
        "negative_mass": negative,
        "cancellation": cancellation,
        "max_positive_patch_fraction": 0.0
        if positive == 0
        else float(positive_values.max()) / positive,
        "max_negative_patch_fraction": 0.0
        if negative_magnitude == 0
        else float((-negative_values).max()) / negative_magnitude,
    }


def _query_mask_digest(value: Mapping[str, object]) -> str:
    payload = {
        "forward": value["forward_query_mask_sha256"],
        "reverse": value["reverse_query_mask_sha256"],
    }
    return canonical_sha256(payload)


def project_pair_record(
    *,
    serialized: Mapping[str, object],
    control_record: Mapping[str, Mapping[str, Mapping[str, object]]],
    bundle: VQueryBundle,
    pair_ordinal: int,
    left_key: str,
    right_key: str,
    axis_position: Mapping[str, int],
    contribution_payload: dict[str, object],
    candidate_reorder_exact: bool,
    pair_swap_exact: bool,
) -> dict[str, object]:
    arms_in = serialized["arms"]
    if not isinstance(arms_in, Mapping) or tuple(arms_in) != SCIENTIFIC_ARMS:
        raise ThreeArmRunnerError("scientific arm projection drift")
    arms: dict[str, object] = {}
    controls: dict[str, object] = {}
    for arm_name in SCIENTIFIC_ARMS:
        value = arms_in[arm_name]
        if not isinstance(value, Mapping):
            raise ThreeArmRunnerError("arm evidence is not a mapping")
        contributions = torch.as_tensor(value["scalar_contributions"], dtype=torch.float64)
        key = f"{bundle.execution_ordinal}:{pair_ordinal}:{arm_name}:REAL"
        contribution_payload[key] = contributions
        margin = float(value["logit"])
        if not torch.allclose(
            contributions.sum(), torch.tensor(margin, dtype=torch.float64), atol=1e-7, rtol=0
        ):
            raise ThreeArmRunnerError("real signed-contribution reconstruction failed")
        arms[arm_name] = {
            "margin_f64_bits": f64_bits(margin),
            "fixed_denominator": 4,
            "query_mask_sha256": _query_mask_digest(value),
            "reference_scope_sha256": value["reference_scope_sha256"],
            "signed_contribution_payload_sha256": tensor_sha256(contributions),
            **_mass_fields(contributions),
            "dual_candidate_legal_h1": bool(value["dual_candidate_legal_h1"]),
            "geometry_contract_pass": True,
            "scalar_reconstruction_pass": True,
        }
        control_output: dict[str, object] = {}
        for control_name in MANDATORY_CONTROLS[:8]:
            item = control_record[arm_name][control_name]
            if item["status"] == "NOT_APPLICABLE_CONTRACT":
                control_output[control_name] = {
                    "status": "NOT_APPLICABLE_CONTRACT"
                }
                continue
            control_contribution = torch.as_tensor(
                item["scalar_contributions"], dtype=torch.float64
            ).flatten()
            control_key = (
                f"{bundle.execution_ordinal}:{pair_ordinal}:{arm_name}:{control_name}"
            )
            contribution_payload[control_key] = control_contribution
            control_output[control_name] = {
                "status": "READY",
                "margin_f64_bits": f64_bits(float(torch.as_tensor(item["margin"]))),
                "only_registered_variable_changed": True,
                "signed_contribution_payload_sha256": tensor_sha256(control_contribution),
            }
        controls[arm_name] = control_output

    # Digesting the same forward/reverse query-mask ledger proves the only
    # allowed regional-arm input difference is reference scope.
    if (
        arms["QUERY_REGION_FULL_REFERENCE_SAME_MODEL"]["query_mask_sha256"]
        != arms["PAIRED_QUERY_REFERENCE_REGION"]["query_mask_sha256"]
    ):
        raise ThreeArmRunnerError("regional query-mask digest mismatch")
    left_ordinal = axis_position[left_key]
    right_ordinal = axis_position[right_key]
    if left_ordinal >= right_ordinal:
        raise ThreeArmRunnerError("candidate pair is not in canonical axis order")
    record: dict[str, object] = {
        "record_type": "PAIR",
        "execution_ordinal": bundle.execution_ordinal,
        "query_id": bundle.query_id,
        "candidate_axis_sha256": bundle.candidate_axis_sha256,
        "left_candidate_ordinal": left_ordinal,
        "right_candidate_ordinal": right_ordinal,
        "left_physical_row": bundle.locks[left_key].candidate.physical_gallery_row,
        "right_physical_row": bundle.locks[right_key].candidate.physical_gallery_row,
        "pair_ordinal": pair_ordinal,
        "arms": arms,
        "controls": controls,
        "candidate_reorder_exact": bool(candidate_reorder_exact),
        "pair_swap_exact": bool(pair_swap_exact),
    }
    if not record["candidate_reorder_exact"] or not record["pair_swap_exact"]:
        raise ThreeArmRunnerError("candidate-reorder/pair-swap replay failed")
    record["record_sha256"] = canonical_sha256(record)
    return record


def main() -> None:
    expanded_argv, execution_manifest = execution_manifest_argv(
        sys.argv[1:], expected_program=Path(__file__).name
    )
    if execution_manifest is None:
        raise ThreeArmRunnerError(
            "formal three-arm scoring requires one JSON execution manifest"
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--query-input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contribution-payload", type=Path, required=True)
    parser.add_argument("--control-input", type=Path, required=True)
    parser.add_argument("--pair-start", type=int, required=True)
    parser.add_argument("--pair-stop", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--streaming-chunk-size", type=int)
    args = parser.parse_args(expanded_argv)

    protocol_path = safe_path(args.protocol, must_exist=True)
    authority_path = safe_path(args.authority, must_exist=True)
    manifest_path = safe_path(execution_manifest, must_exist=True)
    query_input = safe_path(args.query_input, must_exist=True)
    checkpoint = safe_path(args.checkpoint, must_exist=True)
    output = safe_path(args.output, must_exist=False)
    contribution_output = safe_path(args.contribution_payload, must_exist=False)
    control_input = safe_path(args.control_input, must_exist=True)
    if output.exists() or contribution_output.exists():
        raise ThreeArmRunnerError("immutable prejoin shard output already exists")
    if not 0 <= args.pair_start < args.pair_stop <= PAIR_COUNT:
        raise ThreeArmRunnerError("pair shard range is outside canonical C128 pairs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ThreeArmRunnerError("CUDA was requested but unavailable")
    protocol = validate_authority(
        protocol_path, authority_path, args.authority_sha256
    )
    manifest_sha = validate_execution_manifest(
        manifest_path,
        protocol=protocol_path,
        authority=authority_path,
        authority_sha256=args.authority_sha256,
        query_input=query_input,
        checkpoint=checkpoint,
        output=output,
        contribution_payload=contribution_output,
        control_input=control_input,
        pair_start=args.pair_start,
        pair_stop=args.pair_stop,
        device=args.device,
        streaming_chunk_size=args.streaming_chunk_size,
    )
    device = torch.device(args.device)
    bundle = load_query_bundle(query_input, device)
    model, checkpoint_sha = load_model(checkpoint, bundle.outer_fold, device)
    control_records = load_control_input(
        control_input,
        bundle=bundle,
        checkpoint_sha256=checkpoint_sha,
        pair_start=args.pair_start,
        pair_stop=args.pair_stop,
    )
    pairs = canonical_unordered_pairs(bundle.locks)
    axis = canonical_candidate_axis(bundle.locks)
    reordered_axis, reorder_receipt = candidate_reorder(
        axis,
        namespace="RCDE_SR0_MT_CANDIDATE_REORDER_V1",
        query_id=bundle.query_id,
    )
    if set(reordered_axis) != set(axis):
        raise ThreeArmRunnerError("candidate reorder changed candidate membership")
    reordered_locks = {key: bundle.locks[key] for key in reordered_axis}

    records = []
    contribution_payload: dict[str, object] = {}
    axis_position = {key: index for index, key in enumerate(axis)}
    with torch.inference_mode():
        for ordinal, left, right in pairs[args.pair_start : args.pair_stop]:
            evidence = decode_pair(
                model,
                bundle.query,
                bundle.locks,
                left,
                right,
                streaming_chunk_size=args.streaming_chunk_size,
            )
            reorder_replay = decode_pair(
                model,
                bundle.query,
                reordered_locks,
                left,
                right,
                streaming_chunk_size=args.streaming_chunk_size,
            )
            swap_replay = decode_pair(
                model,
                bundle.query,
                bundle.locks,
                right,
                left,
                streaming_chunk_size=args.streaming_chunk_size,
            )
            candidate_reorder_exact = all(
                torch.equal(first.logit, second.logit)
                and torch.equal(
                    first.scalar_contributions, second.scalar_contributions
                )
                for first, second in zip(
                    evidence.arms, reorder_replay.arms, strict=True
                )
            )
            pair_swap_exact = all(
                torch.equal(first.logit, -second.logit)
                and torch.equal(
                    first.scalar_contributions, -second.scalar_contributions
                )
                for first, second in zip(
                    evidence.arms, swap_replay.arms, strict=True
                )
            )
            serialized = scientific_record(serialize_arm_evidence(evidence))
            paired = serialized["arms"]["PAIRED_QUERY_REFERENCE_REGION"]
            contribution_sha = tensor_sha256(paired["scalar_contributions"])
            swap = pair_swap_record(left, right, float(paired["logit"]), contribution_sha)
            if swap["derived_logit"] != -float(paired["logit"]):
                raise ThreeArmRunnerError("pair swap derivation failed")
            records.append(
                project_pair_record(
                    serialized=serialized,
                    control_record=control_records[ordinal],
                    bundle=bundle,
                    pair_ordinal=ordinal,
                    left_key=left,
                    right_key=right,
                    axis_position=axis_position,
                    contribution_payload=contribution_payload,
                    candidate_reorder_exact=candidate_reorder_exact,
                    pair_swap_exact=pair_swap_exact,
                )
            )

    contribution_payload_value = {
        "schema_version": "rc_dino_rcde_sr0_mt_signed_contribution_payload_v1",
        "query_id": bundle.query_id,
        "execution_ordinal": bundle.execution_ordinal,
        "pair_start": args.pair_start,
        "pair_stop": args.pair_stop,
        "checkpoint_sha256": checkpoint_sha,
        "values": contribution_payload,
    }
    contribution_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch(contribution_output, contribution_payload_value)
    contribution_sha = file_sha256(contribution_output)
    header = {
        "schema_version": OUTPUT_SCHEMA,
        "status": OUTPUT_STATUS,
        "target_free": True,
        "query_id": bundle.query_id,
        "execution_ordinal": bundle.execution_ordinal,
        "outer_fold": bundle.outer_fold,
        "candidate_count": len(axis),
        "candidate_axis_sha256": bundle.candidate_axis_sha256,
        "canonical_pair_count": len(pairs),
        "pair_start": args.pair_start,
        "pair_stop": args.pair_stop,
        "record_count": len(records),
        "checkpoint_sha256": checkpoint_sha,
        "same_checkpoint_all_three_arms_and_controls": True,
        "runtime_to_scientific_alias": RUNTIME_TO_SCIENTIFIC,
        "candidate_reorder_receipt_sha256": canonical_sha256(reorder_receipt.__dict__),
        "candidate_reorder_is_container_only": True,
        "pair_swap_second_forward_count": 0,
        "execution_manifest_sha256": manifest_sha,
        "query_input_sha256": file_sha256(query_input),
        "control_input_sha256": file_sha256(control_input),
        "contribution_payload_sha256": contribution_sha,
        "protocol_sha256": file_sha256(protocol_path),
        "authority_sha256": file_sha256(authority_path),
        "access_audit": ZERO_AUDIT,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
    }
    assert_target_free_payload(header)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
        for record in records:
            # Nested arm/control insertion order is part of the frozen shard
            # schema.  The logical hash remains canonical/sorted separately.
            handle.write(json.dumps(record, sort_keys=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "status": header["status"],
                "query_id": bundle.query_id,
                "pair_start": args.pair_start,
                "pair_stop": args.pair_stop,
                "record_count": len(records),
                "output_sha256": file_sha256(output),
                "contribution_payload_sha256": contribution_sha,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
