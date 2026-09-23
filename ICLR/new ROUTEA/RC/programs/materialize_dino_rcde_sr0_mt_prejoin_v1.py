#!/usr/bin/env python3
"""Seal exhaustive target-free SR0-MT pair shards into one prejoin ledger.

The model runner must enumerate every unordered pair on each immutable natural
C128 axis.  This materializer performs no model computation; it verifies shard
content, sidecar hashes, exact pair coverage, target-free fields, and immutable
candidate axes before publishing the concatenated JSONL ledger and receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import sys
import tempfile
from typing import Any, Iterable, Mapping

import torch

RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import argv_from_execution_manifest  # noqa: E402


SHARD_SCHEMA = "rc_dino_rcde_sr0_mt_pair_shard_v1_20260815"
SHARD_STATUS = "RCDE_SR0_MT_PAIR_SHARD_COMPLETE"
MANIFEST_SCHEMA = "rc_dino_rcde_sr0_mt_pair_shard_manifest_v1_20260815"
MANIFEST_STATUS = "RCDE_SR0_MT_PAIR_SHARD_MANIFEST_COMPLETE"
LEDGER_SCHEMA = "rc_dino_rcde_sr0_mt_prejoin_ledger_v1_20260815"
LEDGER_STATUS = "RCDE_SR0_MT_PREJOIN_LEDGER_SEALED"
RECEIPT_SCHEMA = "rc_dino_rcde_sr0_mt_prejoin_receipt_v1_20260815"
RECEIPT_STATUS = "RCDE_SR0_MT_PREJOIN_MATERIALIZATION_PASS"
CONTRIBUTION_SCHEMA = "rc_dino_rcde_sr0_mt_signed_contribution_payload_v1"
AUTHORITY_STATUS = "RCDE_SR0_MT_FORMAL_TRAINING_EXECUTION_AUTHORIZED"
QUERY_COUNT = 600
CANDIDATES = 128
PAIRS_PER_QUERY = CANDIDATES * (CANDIDATES - 1) // 2
PAIR_COUNT = QUERY_COUNT * PAIRS_PER_QUERY
ARMS = (
    "ALL_PATCH_SAME_MODEL",
    "QUERY_REGION_FULL_REFERENCE_SAME_MODEL",
    "PAIRED_QUERY_REFERENCE_REGION",
)
QUERY_FULL = ARMS[1]
PAIRED = ARMS[2]
CONTROLS = (
    "C_DINO_V",
    "C_COL_P",
    "P_QUERY",
    "P_REFERENCE",
    "RANDOM_CONNECTED_MATCHED_SHAPE",
    "DISCONNECTED_MATCHED_AREA",
    "MAGNITUDE_ONLY",
    "SINGLE_SEED",
)
FORBIDDEN_KEYS = {
    "identity",
    "supergroup",
    "target",
    "target_row",
    "target_identity",
    "target_hit",
    "rival",
    "rival_row",
    "rival_identity",
    "correctness",
    "rank",
    "slot",
    "winner",
    "gap",
    "d1_score",
    "d1_rank",
}
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


class PrejoinMaterializerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PrejoinMaterializerError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"required file absent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pair_ordinal(left: int, right: int) -> int:
    require(0 <= left < right < CANDIDATES, "candidate pair is not canonical")
    return left * (2 * CANDIDATES - left - 1) // 2 + (right - left - 1)


def f64_from_bits(value: Any) -> float:
    require(
        isinstance(value, str)
        and len(value) == 16
        and all(character in "0123456789abcdef" for character in value),
        "margin must be lowercase float64 bits",
    )
    number = struct.unpack(">d", bytes.fromhex(value))[0]
    require(math.isfinite(number), "margin is nonfinite")
    return number


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(
        json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8")
        + b"\0"
    )
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _mass_fields(value: torch.Tensor) -> dict[str, float]:
    tensor = torch.as_tensor(value, dtype=torch.float64).flatten()
    positive_values = tensor.clamp_min(0)
    negative_values = tensor.clamp_max(0)
    positive = float(positive_values.sum())
    negative = float(negative_values.sum())
    negative_magnitude = -negative
    total = positive + negative_magnitude
    return {
        "positive_mass": positive,
        "negative_mass": negative,
        "cancellation": (
            0.0
            if total == 0
            else (total - abs(positive - negative_magnitude)) / total
        ),
        "max_positive_patch_fraction": (
            0.0 if positive == 0 else float(positive_values.max()) / positive
        ),
        "max_negative_patch_fraction": (
            0.0
            if negative_magnitude == 0
            else float((-negative_values).max()) / negative_magnitude
        ),
    }


def _contribution_tensor(
    values: Mapping[str, Any],
    remaining: set[str],
    *,
    key: str,
    expected_sha256: str,
    expected_margin: float,
) -> torch.Tensor:
    require(key in remaining and key in values, f"contribution key absent/duplicated: {key}")
    remaining.remove(key)
    tensor = torch.as_tensor(values[key]).detach().cpu().contiguous()
    require(
        tensor.dtype == torch.float64
        and tensor.ndim == 1
        and tensor.numel() > 0
        and bool(torch.isfinite(tensor).all()),
        f"invalid signed contribution tensor: {key}",
    )
    require(tensor_sha256(tensor) == expected_sha256, f"contribution tensor hash drift: {key}")
    require(
        math.isclose(float(tensor.sum()), expected_margin, rel_tol=0.0, abs_tol=1.0e-7),
        f"signed contribution margin reconstruction drift: {key}",
    )
    return tensor


def _validate_record_contributions(
    record: Mapping[str, Any],
    *,
    values: Mapping[str, Any],
    remaining: set[str],
) -> None:
    execution = int(record["execution_ordinal"])
    pair = int(record["pair_ordinal"])
    arms = record["arms"]
    controls = record["controls"]
    for arm in ARMS:
        arm_payload = arms[arm]
        real_key = f"{execution}:{pair}:{arm}:REAL"
        real = _contribution_tensor(
            values,
            remaining,
            key=real_key,
            expected_sha256=str(arm_payload["signed_contribution_payload_sha256"]),
            expected_margin=f64_from_bits(arm_payload["margin_f64_bits"]),
        )
        replay = _mass_fields(real)
        for field, expected in replay.items():
            require(
                math.isclose(
                    float(arm_payload[field]), expected, rel_tol=0.0, abs_tol=1.0e-12
                ),
                f"arm signed-mass diagnostic drift: {real_key}:{field}",
            )
        for control in CONTROLS:
            item = controls[arm][control]
            if item["status"] == "NOT_APPLICABLE_CONTRACT":
                require(
                    f"{execution}:{pair}:{arm}:{control}" not in values,
                    "not-applicable control unexpectedly has contribution payload",
                )
                continue
            control_key = f"{execution}:{pair}:{arm}:{control}"
            _contribution_tensor(
                values,
                remaining,
                key=control_key,
                expected_sha256=str(item["signed_contribution_payload_sha256"]),
                expected_margin=f64_from_bits(item["margin_f64_bits"]),
            )


def _load_contribution_payload(
    path: Path,
    *,
    header: Mapping[str, Any],
) -> tuple[Mapping[str, Any], set[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(payload, Mapping), "contribution payload is not a mapping")
    values = payload.get("values")
    require(
        payload.get("schema_version") == CONTRIBUTION_SCHEMA
        and payload.get("query_id") == header.get("query_id")
        and payload.get("execution_ordinal") == header.get("execution_ordinal")
        and payload.get("pair_start") == header.get("pair_start")
        and payload.get("pair_stop") == header.get("pair_stop")
        and payload.get("checkpoint_sha256") == header.get("checkpoint_sha256")
        and isinstance(values, Mapping)
        and all(isinstance(key, str) and key for key in values),
        "contribution payload header/binding drift",
    )
    return values, set(values)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key).lower()
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _validate_authority(path: Path) -> dict[str, Any]:
    value = _json(path)
    require(value.get("status") == AUTHORITY_STATUS, "formal training authority status drift")
    require(value.get("natural_training_authorized") is True, "natural execution is not authorized")
    require(value.get("model_forward_authorized") is True, "outer-heldout forward is not authorized")
    require(value.get("automatic_stage_advance") is False, "automatic advance must remain false")
    require(value.get("scientific_GO_or_NO_GO") is None, "authority already contains a result")
    return value


def _load_axes(path: Path) -> list[Mapping[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(value, Mapping), "I0 target-free ledger is not a mapping")
    records = value.get("records")
    require(isinstance(records, list) and len(records) == QUERY_COUNT, "I0 axis population drift")
    ordered = sorted(records, key=lambda item: int(item["execution_ordinal"]))
    require([int(item["execution_ordinal"]) for item in ordered] == list(range(QUERY_COUNT)), "execution ordinals drift")
    for record in ordered:
        require(len(record.get("candidate_physical_rows", [])) == CANDIDATES, "C128 axis width drift")
    return ordered


def _validate_arm_payload(value: Any, *, name: str) -> None:
    require(isinstance(value, Mapping), f"missing arm payload: {name}")
    f64_from_bits(value.get("margin_f64_bits"))
    require(value.get("fixed_denominator") == 4, f"{name} denominator drift")
    require(isinstance(value.get("query_mask_sha256"), str), f"{name} query mask hash absent")
    require(isinstance(value.get("reference_scope_sha256"), str), f"{name} reference scope hash absent")
    require(isinstance(value.get("signed_contribution_payload_sha256"), str), f"{name} contribution hash absent")
    require(isinstance(value.get("positive_mass"), (int, float)), f"{name} positive mass absent")
    require(isinstance(value.get("negative_mass"), (int, float)), f"{name} negative mass absent")
    require(isinstance(value.get("cancellation"), (int, float)), f"{name} cancellation absent")
    require(isinstance(value.get("max_positive_patch_fraction"), (int, float)), f"{name} concentration absent")
    require(isinstance(value.get("max_negative_patch_fraction"), (int, float)), f"{name} reverse concentration absent")
    require(isinstance(value.get("dual_candidate_legal_h1"), bool), f"{name} H1 legality absent")
    require(isinstance(value.get("geometry_contract_pass"), bool), f"{name} geometry contract absent")
    require(value.get("scalar_reconstruction_pass") is True, f"{name} scalar reconstruction failed")


def _validate_record(record: Mapping[str, Any], axes: list[Mapping[str, Any]]) -> tuple[int, int]:
    require(record.get("record_type") == "PAIR", "non-pair record in shard body")
    forbidden = FORBIDDEN_KEYS.intersection(_walk_keys(record))
    require(not forbidden, f"protected postjoin field in prejoin record: {sorted(forbidden)}")
    execution = int(record.get("execution_ordinal", -1))
    require(0 <= execution < QUERY_COUNT, "execution ordinal out of range")
    axis = axes[execution]
    require(record.get("query_id") == axis.get("query_id"), "query ID/axis drift")
    require(record.get("candidate_axis_sha256") == axis.get("candidate_axis_sha256"), "candidate axis hash drift")
    left = int(record.get("left_candidate_ordinal", -1))
    right = int(record.get("right_candidate_ordinal", -1))
    ordinal = pair_ordinal(left, right)
    require(int(record.get("pair_ordinal", -1)) == ordinal, "pair ordinal drift")
    rows = [int(item) for item in axis["candidate_physical_rows"]]
    require(int(record.get("left_physical_row", -1)) == rows[left], "left physical row drift")
    require(int(record.get("right_physical_row", -1)) == rows[right], "right physical row drift")
    arms = record.get("arms")
    require(isinstance(arms, Mapping) and tuple(arms) == ARMS, "three-arm key/order drift")
    for name in ARMS:
        _validate_arm_payload(arms[name], name=name)
    require(
        arms[QUERY_FULL]["query_mask_sha256"]
        == arms[PAIRED]["query_mask_sha256"],
        "regional query masks are not byte-identical",
    )
    controls = record.get("controls")
    require(isinstance(controls, Mapping) and tuple(controls) == ARMS, "control arm key/order drift")
    for arm in ARMS:
        values = controls[arm]
        require(isinstance(values, Mapping) and tuple(values) == CONTROLS, "control key/order drift")
        for control in CONTROLS:
            item = values[control]
            require(isinstance(item, Mapping), "control payload absent")
            status = item.get("status")
            require(status in {"READY", "NOT_APPLICABLE_CONTRACT"}, "control applicability drift")
            if status == "READY":
                f64_from_bits(item.get("margin_f64_bits"))
                require(item.get("only_registered_variable_changed") is True, "control isolation failed")
    require(record.get("candidate_reorder_exact") is True, "candidate reorder contract failed")
    require(record.get("pair_swap_exact") is True, "pair swap contract failed")
    expected_sha = record.get("record_sha256")
    body = dict(record)
    body.pop("record_sha256", None)
    require(expected_sha == canonical_sha256(body), "pair record logical hash drift")
    return execution, ordinal


def materialize(
    *, rc_root: Path, authority: Path, base_prejoin: Path, shard_manifest: Path, output_dir: Path
) -> dict[str, Any]:
    rc_root = rc_root.resolve(strict=True)
    require(_inside(rc_root, authority) and _inside(rc_root, base_prejoin) and _inside(rc_root, shard_manifest), "input escapes RC root")
    authority_value = _validate_authority(authority)
    axes = _load_axes(base_prejoin)
    manifest = _json(shard_manifest)
    require(manifest.get("schema_version") == MANIFEST_SCHEMA, "shard manifest schema drift")
    require(manifest.get("status") == MANIFEST_STATUS, "shard manifest incomplete")
    shards = manifest.get("shards")
    require(isinstance(shards, list) and shards, "no pair shards registered")
    require(not output_dir.exists(), f"immutable output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    ledger = output_dir / "prejoin_ledger.jsonl"
    seen = bytearray(PAIR_COUNT)
    count = 0
    record_hashes = hashlib.sha256()
    shard_receipts = []
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".prejoin_ledger.", suffix=".partial", dir=output_dir)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            header = {
                "record_type": "HEADER",
                "schema_version": LEDGER_SCHEMA,
                "status": LEDGER_STATUS,
                "query_count": QUERY_COUNT,
                "candidate_count": CANDIDATES,
                "pairs_per_query": PAIRS_PER_QUERY,
                "pair_count": PAIR_COUNT,
                "target_free": True,
                "base_prejoin_sha256": sha256_file(base_prejoin),
                "shard_manifest_sha256": sha256_file(shard_manifest),
            }
            output.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
            for shard in shards:
                require(isinstance(shard, Mapping), "invalid shard manifest item")
                relative = Path(str(shard.get("path", "")))
                path = relative if relative.is_absolute() else rc_root / relative
                require(_inside(rc_root, path), "shard escapes RC root")
                require(sha256_file(path) == shard.get("sha256"), "shard file hash drift")
                payload_relative = Path(str(shard.get("contribution_payload_path", "")))
                payload_path = payload_relative if payload_relative.is_absolute() else rc_root / payload_relative
                require(_inside(rc_root, payload_path), "contribution payload escapes RC root")
                require(sha256_file(payload_path) == shard.get("contribution_payload_sha256"), "contribution payload hash drift")
                shard_count = 0
                with path.open("r", encoding="utf-8") as handle:
                    first = json.loads(next(handle))
                    require(first.get("schema_version") == SHARD_SCHEMA and first.get("status") == SHARD_STATUS, "pair shard header drift")
                    require(first.get("target_free") is True and first.get("access_audit") == ZERO_AUDIT, "pair shard access audit drift")
                    require(first.get("contribution_payload_sha256") == shard.get("contribution_payload_sha256"), "pair shard payload binding drift")
                    contribution_values, remaining_contribution_keys = _load_contribution_payload(
                        payload_path,
                        header=first,
                    )
                    contribution_key_count = len(remaining_contribution_keys)
                    for line in handle:
                        record = json.loads(line)
                        require(isinstance(record, Mapping), "pair shard line is not an object")
                        execution, ordinal = _validate_record(record, axes)
                        _validate_record_contributions(
                            record,
                            values=contribution_values,
                            remaining=remaining_contribution_keys,
                        )
                        index = execution * PAIRS_PER_QUERY + ordinal
                        require(seen[index] == 0, "duplicate candidate pair")
                        seen[index] = 1
                        rendered = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
                        output.write(rendered + "\n")
                        record_hashes.update(bytes.fromhex(str(record["record_sha256"])))
                        shard_count += 1
                        count += 1
                require(
                    not remaining_contribution_keys,
                    "unconsumed signed contribution tensors remain in shard payload",
                )
                require(shard_count == int(first.get("record_count", -1)), "shard record count drift")
                shard_receipts.append({"path": str(shard.get("path")), "sha256": str(shard.get("sha256")), "record_count": shard_count, "contribution_payload_path": str(shard.get("contribution_payload_path")), "contribution_payload_sha256": str(shard.get("contribution_payload_sha256")), "contribution_payload_key_count": contribution_key_count})
            output.flush()
            os.fsync(output.fileno())
        require(count == PAIR_COUNT and all(seen), "exhaustive natural-C128 pair coverage failed")
        os.link(temporary_name, ledger)
        os.chmod(ledger, 0o440)
        Path(temporary_name).unlink(missing_ok=True)
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": RECEIPT_STATUS,
            "prejoin_ledger_sha256": sha256_file(ledger),
            "record_sha_sequence_sha256": record_hashes.hexdigest(),
            "pair_count": count,
            "query_count": QUERY_COUNT,
            "pairs_per_query": PAIRS_PER_QUERY,
            "target_free": True,
            "candidate_reorder_exact_count": count,
            "pair_swap_exact_count": count,
            "authority_path": str(authority.relative_to(rc_root)),
            "authority_sha256": sha256_file(authority),
            "base_prejoin_sha256": sha256_file(base_prejoin),
            "shard_manifest_sha256": sha256_file(shard_manifest),
            "shards": shard_receipts,
            "access_audit": ZERO_AUDIT,
            "scientific_GO_or_NO_GO": None,
            "next_authorized_stage": None,
            "automatic_stage_advance": False,
        }
        receipt["logical_sha256"] = canonical_sha256(receipt)
        receipt_path = output_dir / "prejoin_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(receipt_path, 0o440)
        return receipt
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc-root", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--base-prejoin", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(
        argv_from_execution_manifest(sys.argv[1:], expected_program=Path(__file__).name)
    )
    if sha256_file(args.authority) != args.authority_sha256:
        raise SystemExit("prejoin authority file hash drift")
    materialize(
        rc_root=args.rc_root,
        authority=args.authority,
        base_prejoin=args.base_prejoin,
        shard_manifest=args.shard_manifest,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
