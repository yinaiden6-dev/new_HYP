#!/usr/bin/env python3
"""Independent replay for ColNomic plus unit component residual confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

import reduce_dino_rcde_colnomic_component_residual_confirmation_v1 as R


ROOT = Path(__file__).resolve().parents[1]
PASS = "COLNOMIC_COMPONENT_RESIDUAL_INDEPENDENT_VALIDATION_PASS"


class ResidualValidationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ResidualValidationError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def bound_path(authority: Mapping[str, Any], name: str) -> Path:
    binding = authority.get("bindings", {}).get(name)
    require(isinstance(binding, Mapping), f"binding absent: {name}")
    path = (ROOT / str(binding.get("path", ""))).resolve()
    require(
        ROOT in path.parents
        and path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == int(binding.get("bytes", -1))
        and file_sha(path) == binding.get("sha256"),
        f"binding drift: {name}",
    )
    return path


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), "immutable residual validation exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authority = read_json(args.authority)
    result = read_json(args.result)
    require(
        authority.get("status") == R.AUTHORITY_STATUS
        and authority.get("logical_sha256") == R.logical(authority)
        and result.get("status") in {R.GO, R.NO_GO}
        and result.get("logical_sha256") == R.logical(result)
        and result.get("authority_sha256") == file_sha(args.authority)
        and args.result.resolve() == (ROOT / authority["output"]).resolve()
        and args.output.resolve(strict=False)
        == (ROOT / authority["validation_output"]).resolve(strict=False),
        "authority/result envelope drift",
    )
    raw_path = bound_path(authority, "raw_prejoin_ledger")
    pj1_path = bound_path(authority, "pj1_result")
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    pj1 = read_json(pj1_path)
    recomputed = R.reduce_rows(pj1["rows"], R.validate_raw_prejoin(raw))
    for key in (
        "row_population_sha256",
        "arm_summaries",
        "fused_vs_raw",
        "gates",
        "all_gates_pass",
    ):
        require(result[key] == recomputed[key], f"recomputed residual drift: {key}")
    require(
        result["rows"] == recomputed["rows"]
        and len(result["rows"]) == 582
        and all(
            int(row["execution_ordinal"]) not in R.EXCLUDED
            and row["record_sha256"] == R.record_sha(row)
            for row in result["rows"]
        ),
        "residual row/exclusion drift",
    )
    go = bool(recomputed["all_gates_pass"])
    require(
        result["status"] == (R.GO if go else R.NO_GO)
        and result["scientific_GO_or_NO_GO"] == ("GO" if go else "NO_GO")
        and result["residual_weight"] == 1.0
        and result["development_execution_ordinals_excluded"] == list(range(12))
        and result["alternative_weight_count"] == 0
        and all(
            result[key] == 0
            for key in (
                "model_load_count",
                "model_forward_count",
                "model_backward_count",
                "model_update_count",
            )
        )
        and result["full_gallery_retrieval_or_ownership_claim_authorized"] is False
        and result["next_authorized_stage"] is None,
        "decision/access boundary drift",
    )
    value: dict[str, Any] = {
        "schema_version": "rc_colnomic_component_residual_confirmation_validation_v1_20260824",
        "status": PASS,
        "validation_pass": True,
        "screen_status": result["status"],
        "all_gates_recomputed_pass": go,
        "authority_sha256": file_sha(args.authority),
        "authority_logical_sha256": authority["logical_sha256"],
        "result_sha256": file_sha(args.result),
        "result_logical_sha256": result["logical_sha256"],
        "raw_prejoin_ledger_sha256": file_sha(raw_path),
        "pj1_result_sha256": file_sha(pj1_path),
        "query_count": 582,
        "candidate_count": 74496,
        "residual_weight": 1.0,
        "development_execution_ordinals_excluded": list(range(12)),
        "transition_counts": recomputed["fused_vs_raw"]["transition_counts"],
        "model_forward_count": 0,
        "scientific_GO_or_NO_GO": result["scientific_GO_or_NO_GO"],
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = R.logical(value)
    atomic_write(args.output, value)
    print(json.dumps({"status": PASS, "screen_status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
