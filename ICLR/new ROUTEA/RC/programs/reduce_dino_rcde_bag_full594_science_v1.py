#!/usr/bin/env python3
"""Primary BAG-only full-594 scientific reducer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))
sys.path.insert(0, str(RC_ROOT / "programs"))

from rc_aslo_xf.dino_rcde_bag_completion_v1 import (  # noqa: E402
    PERMUTATIONS,
    SEED,
    VALID_QUERY_COUNT,
    canonical_sha256,
    evaluate_bag_completion,
)
from dino_rcde_bag_completion_io_v1 import (  # noqa: E402
    atomic_json,
    binding_path,
    file_sha256,
    read_authority,
    read_json,
    require,
)
from join_dino_rcde_bag_full594_labels_v1 import SCHEMA as JOIN_SCHEMA, STATUS as JOIN_STATUS  # noqa: E402


SCHEMA = "rc_dino_rcde_bag_full594_scientific_result_v1_20260827"
STATUS = "DINO_RCDE_BAG_FULL594_SCIENTIFIC_REDUCTION_COMPLETE"


def reduce(authority_path: Path, join_path: Path, output_path: Path) -> dict[str, Any]:
    authority, authority_sha = read_authority(authority_path)
    require(
        authority.get("scientific_reduction_after_label_join_authorized") is True
        and authority.get("training_authorized") is False
        and authority.get("model_forward_after_label_join_authorized") is False,
        "scientific reducer authority drift",
    )
    require(binding_path(authority, "scientific_reducer") == Path(__file__).resolve(), "scientific reducer runtime binding drift")
    join = read_json(join_path)
    require(
        join.get("schema_version") == JOIN_SCHEMA
        and join.get("status") == JOIN_STATUS
        and join.get("query_count") == VALID_QUERY_COUNT
        and join.get("target_rival_read_count") == VALID_QUERY_COUNT
        and join.get("scientific_reduction_count") == 0
        and join.get("source_bindings", {}).get("authority_sha256") == authority_sha,
        "scientific reducer join predecessor drift",
    )
    statistics = evaluate_bag_completion(
        join["rows"], repetitions=PERMUTATIONS, seed=SEED
    )
    decision = statistics.get("decision")
    if statistics.get("status") == "DINO_RCDE_BAG_COMPLETION_STATISTICALLY_INELIGIBLE":
        decision = "DINO_RCDE_BAG_COMPLETION_STATISTICALLY_INELIGIBLE"
    output: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_level": "GIVEN_C128_FIXED_BAG_DECODER_NONSPATIAL_REPRESENTATION_QUALIFICATION_ONLY",
        "query_count": VALID_QUERY_COUNT,
        "statistics": statistics,
        "decision": decision,
        "scientific_GO_or_NO_GO": statistics.get("scientific_GO_or_NO_GO"),
        "candidate_reorder_gate": True,
        "claim_boundaries": {
            "fixed_bag_decoder_only": True,
            "dispersed_candidate_bound_information_if_go": True,
            "deployable_multi_component_model": False,
            "spatial_correspondence": False,
            "connected_region": False,
            "full_gallery_retrieval": False,
            "target_absence": False,
            "hold_switch": False,
            "ownership": False,
            "opened_or_sealed_generalization": False,
        },
        "no_go_scope": "ONLY_THIS_FROZEN_BAG_DECODER_NOT_ALL_DISPERSED_PATCH_MECHANISMS",
        "source_bindings": {
            "authority_sha256": authority_sha,
            "label_join_sha256": file_sha256(join_path),
            "label_join_logical_sha256": join["logical_sha256"],
        },
        "model_load_count": 0,
        "model_forward_count": 0,
        "model_update_count": 0,
        "scientific_reduction_count": 1,
        "protected_access_count": 0,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    output["logical_sha256"] = canonical_sha256(output)
    atomic_json(output_path, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--join", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reduce(args.authority, args.join, args.output)
    print({"status": result["status"], "decision": result["decision"], "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
