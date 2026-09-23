#!/usr/bin/env python3
"""Validate one V121 runtime entry against the frozen authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from dino_rcde_track_r_v121_lineage_v1 import (
    read_authority,
    validate_runtime_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = "registry/current_authority_v121_20260821.json"
SCHEMA = "rc_current_authority_v121_20260821"
STATUS = "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARDS_AUTHORIZED"
REQUIRED_RUNTIME_BINDINGS = (
    "producer",
    "validator",
    "finalizer",
    "lineage_runtime",
    "runtime_entry_validator",
    "package_initializer",
    "launcher",
    "post_vfit_controller",
    "post_oof_controller",
)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--binding-name", choices=REQUIRED_RUNTIME_BINDINGS, required=True)
    parser.add_argument("--runtime-file", type=Path, required=True)
    args = parser.parse_args()

    authority, authority_sha256 = read_authority(
        args.authority,
        root=ROOT,
        expected_relative=AUTHORITY,
        expected_schema=SCHEMA,
        expected_status=STATUS,
    )
    validate_runtime_bindings(
        authority,
        root=ROOT,
        required_names=REQUIRED_RUNTIME_BINDINGS,
    )
    binding = authority.get("bindings", {}).get(args.binding_name)
    require(isinstance(binding, Mapping), "selected runtime binding absent")
    expected = (ROOT / str(binding["path"])).resolve(strict=True)
    require(
        args.runtime_file.resolve(strict=True) == expected,
        "selected runtime entry path drift",
    )
    print(
        json.dumps(
            {
                "status": "DINO_RCDE_TRACK_R_V121_RUNTIME_BINDING_PASS",
                "authority_sha256": authority_sha256,
                "binding_name": args.binding_name,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
