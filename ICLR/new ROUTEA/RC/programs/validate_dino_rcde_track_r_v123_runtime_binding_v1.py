#!/usr/bin/env python3
"""Validate a V123 runtime entry and the complete frozen local closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dino_rcde_track_r_v121_lineage_v1 import read_authority, validate_runtime_bindings


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = "registry/current_authority_v123_20260822.json"
SCHEMA = "rc_current_authority_v123_20260822"
STATUS = "DINO_RCDE_TRACK_R_SCIENTIFIC_REDUCTION_AUTHORIZED"
NAMES = (
    "metadata_sealer",
    "metadata_validator",
    "statistics",
    "independent_statistics",
    "reducer",
    "validator",
    "runtime_entry_validator",
    "lineage_runtime",
    "package_initializer",
    "launcher",
    "post_p_controller",
    "automatic_continuation_addendum",
    "v123_promotion_controller",
    "v123_autochain_registrar",
    "v123_autochain_registration_validator",
    "v123_autochain_registration",
    "v123_autochain_registration_validation",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--binding-name", choices=NAMES, required=True)
    parser.add_argument("--runtime-file", type=Path, required=True)
    args = parser.parse_args()
    authority, authority_sha256 = read_authority(
        args.authority,
        root=ROOT,
        expected_relative=AUTHORITY,
        expected_schema=SCHEMA,
        expected_status=STATUS,
    )
    validate_runtime_bindings(authority, root=ROOT, required_names=NAMES)
    binding = authority["bindings"][args.binding_name]
    if args.runtime_file.resolve(strict=True) != (ROOT / binding["path"]).resolve(strict=True):
        raise RuntimeError("V123 selected runtime path drift")
    print(json.dumps({"status": "DINO_RCDE_TRACK_R_V123_RUNTIME_BINDING_PASS", "authority_sha256": authority_sha256, "binding_name": args.binding_name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
