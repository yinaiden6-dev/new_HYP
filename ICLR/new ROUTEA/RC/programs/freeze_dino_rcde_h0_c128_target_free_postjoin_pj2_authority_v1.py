#!/usr/bin/env python3
"""Freeze the registered PJ2 reducer only after independent PJ1 PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
NS = "h0_c128_target_free_postjoin"
STATUS = "H0_C128_TARGET_FREE_POSTJOIN_PJ2_REDUCTION_AUTHORIZED"
AUTH = ROOT / "registry/h0/h0_c128_target_free_postjoin_pj2_authority_v1_20260824.json"
PJ1AUTH = ROOT / "registry/h0/h0_c128_target_free_postjoin_pj1_authority_v1_20260824.json"
PJ1 = ROOT / "results/dino_rcde_h0_c128_target_free_postjoin_pj1_v1/result.json"
PJ1V = ROOT / "results/dino_rcde_h0_c128_target_free_postjoin_pj1_validation_v1/result.json"


class E(RuntimeError):
    pass


def req(condition: Any, message: str) -> None:
    if not condition:
        raise E(message)


def canon(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return canon({key: item for key, item in value.items() if key != "logical_sha256"})


def fsha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    req(path.is_file() and not path.is_symlink(), f"JSON absent/unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    req(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def bind(path: Path, logical_hash: bool = False) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    req(resolved.is_file() and not resolved.is_symlink(), f"unsafe binding: {path}")
    req(resolved.is_relative_to(ROOT), f"binding escaped RC: {path}")
    value: dict[str, Any] = {
        "path": str(resolved.relative_to(ROOT)),
        "bytes": resolved.stat().st_size,
        "sha256": fsha(resolved),
    }
    if logical_hash:
        payload = read(resolved)
        req(payload.get("logical_sha256") == logical(payload), f"logical binding drift: {path}")
        value["logical_sha256"] = payload["logical_sha256"]
    return value


def validate_pj1_lineage() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path in (PJ1AUTH, PJ1, PJ1V):
        req(path.is_file() and not path.is_symlink(), f"PJ1 lineage absent/unsafe: {path}")
        req(path.stat().st_mode & 0o777 == 0o444, f"mutable PJ1 lineage: {path}")
    authority, result, validation = read(PJ1AUTH), read(PJ1), read(PJ1V)
    req(
        authority.get("namespace") == NS
        and authority.get("status") == "H0_C128_TARGET_FREE_POSTJOIN_PJ1_EXECUTION_AUTHORIZED"
        and authority.get("logical_sha256") == logical(authority)
        and authority.get("scientific_reduction_authorized") is False
        and authority.get("scientific_GO_or_NO_GO") is None,
        "PJ1 authority drift",
    )
    req(
        result.get("namespace") == NS
        and result.get("status") == "H0_C128_TARGET_FREE_POSTJOIN_PJ1_READY"
        and result.get("logical_sha256") == logical(result)
        and result.get("authority_sha256") == fsha(PJ1AUTH)
        and result.get("authority_logical_sha256") == authority["logical_sha256"]
        and result.get("query_count") == 594
        and result.get("candidate_count") == 76_032
        and result.get("raw_directional_arm_logit_count") == 304_128
        and result.get("target_insertions") == 0
        and result.get("candidate_mutations") == 0
        and result.get("direction_reduction_count") == 0
        and result.get("winner_rank_margin_count") == 0
        and result.get("scientific_GO_or_NO_GO") is None,
        "PJ1 result drift",
    )
    req(
        validation.get("namespace") == NS
        and validation.get("status")
        == "H0_C128_TARGET_FREE_POSTJOIN_PJ1_INDEPENDENT_VALIDATION_PASS"
        and validation.get("validation_pass") is True
        and validation.get("logical_sha256") == logical(validation)
        and validation.get("authority_sha256") == fsha(PJ1AUTH)
        and validation.get("authority_logical_sha256") == authority["logical_sha256"]
        and validation.get("result_sha256") == fsha(PJ1)
        and validation.get("result_logical_sha256") == result["logical_sha256"]
        and validation.get("query_count") == 594
        and validation.get("candidate_count") == 76_032
        and validation.get("raw_directional_arm_logit_count") == 304_128
        and validation.get("independent_full_shard_replay") is True
        and validation.get("independent_label_join_replay") is True
        and validation.get("scientific_metric_count") == 0
        and validation.get("scientific_GO_or_NO_GO") is None,
        "PJ1 validation drift",
    )
    return authority, result, validation


def build() -> dict[str, Any]:
    validate_pj1_lineage()
    paths = {
        "contract": ROOT / "plan/DINO_RCDE_H0_C128_TARGET_FREE_POSTJOIN_AUTOMATIC_HANDOFF_CONTRACT_V1_20260824.md",
        "freezer": Path(__file__),
        "reducer": ROOT / "programs/reduce_dino_rcde_h0_c128_target_free_postjoin_pj2_v1.py",
        "validator": ROOT / "programs/validate_dino_rcde_h0_c128_target_free_postjoin_pj2_v1.py",
        "launcher": ROOT / "slurm/dino_rcde_h0_c128_target_free_postjoin_pj2_v1.sbatch",
        "handoff_authority": ROOT / "registry/h0/h0_c128_target_free_postjoin_handoff_authority_v1_20260824.json",
        "test": ROOT / "tests/test_dino_rcde_h0_c128_target_free_postjoin_pj2_v1.py",
        "authority_test": ROOT / "tests/test_freeze_dino_rcde_h0_c128_target_free_postjoin_pj2_authority_v1.py",
    }
    bindings = {name: bind(path, name == "handoff_authority") for name, path in paths.items()}
    bindings.update(
        {
            "pj1_authority": bind(PJ1AUTH, True),
            "pj1_result": bind(PJ1, True),
            "pj1_validation": bind(PJ1V, True),
        }
    )
    output: dict[str, Any] = {
        "schema_version": "rc_h0_c128_target_free_postjoin_pj2_authority_v1_20260824",
        "namespace": NS,
        "status": STATUS,
        "claim_level": "RELATIVE_C128_RANK_INCREMENT_ONLY",
        "scientific_reduction_authorized": True,
        "training_authorized": False,
        "model_forward_authorized": False,
        "rescue_break_analysis_authorized": False,
        "c_p_ownership_or_full_gallery_retrieval_claim_authorized": False,
        "formula_contract": {
            "candidate_score": "binary64(0.5*(a_to_b+b_to_a))",
            "same_identity_reduce": "max_then_smaller_physical_row",
            "strongest_wrong_order": "(-score,physical_row,UTF8_label)",
            "margin": "target_score-own_strongest_wrong_score",
            "strict_top1": "margin>0",
            "rank": "1+count(wrong_score>=target_score)",
            "reciprocal_rank": "1/rank",
            "forbidden_cross_arm_margin_difference": True,
        },
        "statistical_contract": {
            "aggregation": "supergroup_mean_then_equal_supergroup_mean",
            "bootstrap_replicates": 9999,
            "bootstrap_seed": 17,
            "bootstrap_rng": "numpy.PCG64",
            "bootstrap_quantile_method": "linear",
            "signflip_replicates": 9999,
            "signflip_seed": 17,
            "signflip_rng": "numpy.PCG64",
            "signflip_p": "(1+count(null>=observed))/10000",
        },
        "gate_contract": {
            "minimum_group_balanced_top1_gain": 0.03,
            "top1_ci_lower_strictly_positive": True,
            "top1_one_sided_p_below": 0.05,
            "mrr_gain_strictly_positive": True,
            "mrr_ci_lower_strictly_positive": True,
            "mrr_one_sided_p_below": 0.05,
            "minimum_positive_top1_folds": 3,
            "minimum_positive_mrr_folds": 3,
            "zero_to_one_strictly_exceeds_one_to_zero": True,
        },
        "output": "results/dino_rcde_h0_c128_target_free_postjoin_pj2_v1/result.json",
        "validation_output": "results/dino_rcde_h0_c128_target_free_postjoin_pj2_validation_v1/result.json",
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "resource_contract": {
            "partitions": ["cpuonly", "dev_cpuonly"],
            "cpus_per_task": 4,
            "memory_megabytes": 32_768,
            "walltime_seconds": 3600,
        },
        "bindings": bindings,
    }
    output["logical_sha256"] = logical(output)
    return output


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    req(not path.exists() and not path.is_symlink(), "authority exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o444)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=AUTH)
    args = parser.parse_args()
    req(args.output.resolve(strict=False) == AUTH.resolve(strict=False), "output drift")
    value = build()
    if args.output.exists():
        req(
            read(args.output) == value and args.output.stat().st_mode & 0o777 == 0o444,
            "existing authority drift",
        )
    else:
        atomic(args.output, value)
    print(json.dumps({"status": STATUS, "logical_sha256": value["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
