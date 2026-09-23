#!/usr/bin/env python3
"""Independently validate the complete V122 execution authority."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))
from dino_rcde_track_r_p_v2_utility_runtime_v1 import (  # noqa: E402
    fsha,
    logical,
    validate_authority,
)


OUTPUT = "results/dino_rcde_track_r_p_v2_utility_authority_validation_v1/result.json"
STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_AUTHORITY_INDEPENDENT_VALIDATION_PASS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    roles = (
        ("p_v2_utility_materialization_authorized", "producer", "programs/materialize_dino_rcde_track_r_p_v2_utility_shard_v1.py"),
        ("p_v2_utility_validation_authorized", "validator", "programs/validate_dino_rcde_track_r_p_v2_utility_shard_v1.py"),
        ("p_v2_utility_reduction_authorized", "reducer", "programs/reduce_dino_rcde_track_r_p_v2_utility_v1.py"),
        ("p_v2_utility_aggregate_validation_authorized", "aggregate_validator", "programs/validate_dino_rcde_track_r_p_v2_utility_comparator_v1.py"),
    )
    authority = None
    authority_sha = None
    for flag, key, path in roles:
        observed, observed_sha = validate_authority(
            args.authority,
            required_flag=flag,
            runtime_binding=key,
            runtime_path=path,
        )
        authority = observed if authority is None else authority
        authority_sha = observed_sha if authority_sha is None else authority_sha
        if observed != authority or observed_sha != authority_sha:
            raise RuntimeError("V122 independent role-authority replay drift")
    result = {
        "schema_version": "rc_dino_rcde_track_r_p_v2_utility_authority_validation_v1_20260822",
        "status": STATUS,
        "validation_pass": True,
        "authority_sha256": authority_sha,
        "authority_logical_sha256": authority["logical_sha256"],
        "runtime_role_count": 4,
        "runtime_module_count": authority["bindings"]["runtime_import_closure"]["module_count"],
        "source_shard_manifest_count": 50,
        "target_rival_read_authorized": False,
        "label_identity_supergroup_read_authorized": False,
        "dino_token_read_authorized": False,
        "scientific_GO_or_NO_GO": None,
    }
    result["logical_sha256"] = logical(result)
    output = args.output.resolve()
    if output != (ROOT / OUTPUT).resolve() or output.exists():
        raise RuntimeError("V122 authority validation output drift")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".partial")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temp, output)
    output.chmod(0o444)
    print(json.dumps({"status": STATUS, "authority_sha256": fsha(args.authority)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

