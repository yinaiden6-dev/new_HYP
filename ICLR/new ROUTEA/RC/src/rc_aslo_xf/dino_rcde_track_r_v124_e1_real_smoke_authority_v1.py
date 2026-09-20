"""Exact V3 base closure plus non-promotable real-smoke overlay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from . import dino_rcde_track_r_v124_e1_authority_v3 as BASE


OVERLAY_BINDINGS = frozenset(
    {
        "smoke_producer",
        "smoke_independent_validator",
        "smoke_launcher",
        "smoke_authority_runtime",
        "smoke_tests",
    }
)
PRODUCER_NAMESPACE = "results/dino_rcde_track_r_v124_e1_v3_accelerated_real_smoke_v1_committed"
VALIDATION_NAMESPACE = "results/dino_rcde_track_r_v124_e1_v3_accelerated_real_smoke_validation_v1_committed"


class E1RealSmokeAuthorityError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise E1RealSmokeAuthorityError(message)


def required_bindings() -> frozenset[str]:
    return BASE.required_bindings() | OVERLAY_BINDINGS


def validate_authority_envelope(value: Mapping[str, Any]) -> None:
    bindings = value.get("bindings")
    require(
        isinstance(bindings, Mapping)
        and set(bindings) == required_bindings(),
        "real-smoke base/overlay binding whitelist drift",
    )
    projected = dict(value)
    projected["bindings"] = {
        key: bindings[key] for key in BASE.required_bindings()
    }
    projected["logical_sha256"] = BASE.V2.logical_sha256(projected)
    BASE.validate_authority_envelope(projected)
    require(
        value.get("logical_sha256") == BASE.V2.logical_sha256(value)
        and value.get("execution_scope")
        == "EXECUTION0_ACCELERATED_REAL_SMOKE_ONLY_NONPROMOTABLE"
        and value.get("real_smoke") is True
        and value.get("eligible_as_e1_result") is False
        and value.get("formal_or_full594_execution_authorized") is False
        and value.get("submission_review_status") == "PENDING"
        and value.get("submission_authorized") is False
        and value.get("manual_submission_requires_independent_review") is True
        and value.get("output_contract")
        == {
            "producer_family": PRODUCER_NAMESPACE,
            "validation_family": VALIDATION_NAMESPACE,
            "append_only": True,
            "atomic_family_required": True,
            "exact_committed_reuse": True,
            "overwrite_or_repair_authorized": False,
            "eligible_as_e1_result": False,
        }
        and value.get("base_v3_import_closure_count") == 65
        and value.get("base_v3_binding_count") == 115
        and value.get("smoke_overlay_binding_count") == len(OVERLAY_BINDINGS)
        and value.get("target_rival_join_authorized") is False
        and value.get("postjoin_authorized") is False
        and value.get("scientific_reduction_authorized") is False
        and value.get("automatic_submit_authorized") is False
        and value.get("scientific_GO_or_NO_GO") is None
        and value.get("automatic_stage_advance") is False
        and value.get("next_authorized_stage") is None,
        "real-smoke non-promotable/no-science envelope drift",
    )
    for key in OVERLAY_BINDINGS:
        row = bindings[key]
        require(
            isinstance(row, Mapping)
            and set(row) == {"path", "bytes", "sha256"},
            f"smoke overlay binding field drift: {key}",
        )


def read_authority(path: Path, expected_sha256: str) -> dict[str, Any]:
    authority_path = path.resolve()
    require(
        authority_path.is_file()
        and not authority_path.is_symlink()
        and BASE.V2.file_sha256(authority_path) == expected_sha256,
        "real-smoke authority physical drift",
    )
    value = json.loads(authority_path.read_text(encoding="ascii"))
    validate_authority_envelope(value)
    for name, row in value["bindings"].items():
        raw = Path(row["path"])
        bound = (
            raw if raw.is_absolute() else BASE.RC_ROOT / raw
        ).resolve()
        require(
            bound.is_file()
            and not bound.is_symlink()
            and bound.stat().st_size == row["bytes"]
            and BASE.V2.file_sha256(bound) == row["sha256"],
            f"real-smoke bound file drift: {name}",
        )
    return value


__all__ = [
    "E1RealSmokeAuthorityError",
    "OVERLAY_BINDINGS",
    "PRODUCER_NAMESPACE",
    "VALIDATION_NAMESPACE",
    "read_authority",
    "required_bindings",
    "validate_authority_envelope",
]
