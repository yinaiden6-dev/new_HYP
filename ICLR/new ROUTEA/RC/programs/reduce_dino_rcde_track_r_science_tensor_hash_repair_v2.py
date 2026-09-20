#!/usr/bin/env python3
"""V123 production reducer wrapper restoring the frozen tensor-hash contract."""

from __future__ import annotations

import reduce_dino_rcde_track_r_science_v1 as base
import dino_rcde_track_r_oof_tensor_canonical_v2 as tensor_canonical
import numpy as np
import torch


REPAIR_AUTHORITY_PATH = (
    "registry/dino_rcde_track_r_v123_tensor_hash_repair_authority_v1_20260824.json"
)
REPAIR_AUTHORITY_SCHEMA = (
    "rc_dino_rcde_track_r_v123_tensor_hash_repair_authority_v1_20260824"
)
REPAIR_AUTHORITY_STATUS = (
    "DINO_RCDE_TRACK_R_V123_TENSOR_HASH_REPAIR_AUTHORIZED"
)
ORIGINAL_JSON_LOGICAL_SHA256 = base.logical_sha256


def _contains_array(value: object) -> bool:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return True
    if isinstance(value, dict):
        return any(_contains_array(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_array(item) for item in value)
    return False


def repaired_logical_sha256(value):
    if _contains_array(value):
        return tensor_canonical.logical_sha256(value)
    return ORIGINAL_JSON_LOGICAL_SHA256(value)


def configure() -> None:
    base.AUTHORITY_PATH = REPAIR_AUTHORITY_PATH
    base.AUTHORITY_SCHEMA = REPAIR_AUTHORITY_SCHEMA
    base.AUTHORITY_STATUS = REPAIR_AUTHORITY_STATUS
    base.logical_sha256 = repaired_logical_sha256


def main() -> int:
    configure()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
