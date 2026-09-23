#!/usr/bin/env python3
"""Independent replay of a full-C128, pair-only C_DINO receipt.

This validator intentionally does not import the production control helper or
the legacy control implementation.  It reconstructs the deterministic
identity-disjoint matching and every population hash from JSON fields alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


INPUT_SCHEMA = "rc_dino_rcde_track_r_full_c128_c_dino_pair_v1_20260824"
OUTPUT_SCHEMA = (
    "rc_dino_rcde_track_r_full_c128_c_dino_pair_independent_replay_v1_20260824"
)
OUTPUT_STATUS = "FULL_C128_C_DINO_PAIR_INDEPENDENT_REPLAY_PASS"
CANDIDATE_COUNT = 128
PAIR_MEMBER_COUNT = 2
PHYSICAL_GALLERY_ROWS = 5413
DIRECTIONS = ("a_to_b", "b_to_a")


class IndependentReplayError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise IndependentReplayError(message)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def hash_parts(namespace: str, *parts: object) -> str:
    payload = b"\0".join(
        (namespace.encode("utf-8"), *(str(item).encode("utf-8") for item in parts))
    )
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def exact_mapping(
    value: object, fields: Sequence[str], *, name: str
) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{name} is not a mapping")
    require(set(value) == set(fields), f"{name} field-set drift")
    return value


def deterministic_identity_disjoint_matching(
    axis: Sequence[Mapping[str, Any]],
    *,
    namespace: str,
    query_id: str,
    protected_pair_keys: Sequence[str],
) -> tuple[int, ...]:
    keys = tuple(str(item["candidate_key"]) for item in axis)
    identities = {
        str(item["candidate_key"]): str(item["corrected_identity_sha256"])
        for item in axis
    }
    protected = frozenset(protected_pair_keys)
    require(
        len(protected) == PAIR_MEMBER_COUNT and protected.issubset(keys),
        "protected pair axis drift",
    )
    donor_for: dict[str, str] = {}
    destination_for_donor: dict[str, str] = {}

    def assign(destination: str, seen: set[str]) -> bool:
        donors = sorted(
            (
                donor
                for donor in keys
                if donor != destination
                and identities[donor] != identities[destination]
                and not (destination in protected and donor in protected)
            ),
            key=lambda donor: hash_parts(
                namespace,
                query_id,
                "FULL_C128_DONOR",
                destination,
                donor,
            ),
        )
        for donor in donors:
            if donor in seen:
                continue
            seen.add(donor)
            prior = destination_for_donor.get(donor)
            if prior is None or assign(prior, seen):
                destination_for_donor[donor] = destination
                donor_for[destination] = donor
                return True
        return False

    destinations = sorted(
        keys,
        key=lambda key: hash_parts(namespace, query_id, "FULL_C128_DEST", key),
    )
    require(
        all(assign(destination, set()) for destination in destinations),
        "independent replay found no identity-disjoint perfect matching",
    )
    index_by_key = {key: index for index, key in enumerate(keys)}
    return tuple(index_by_key[donor_for[key]] for key in keys)


def replay_receipt(receipt: Mapping[str, Any]) -> dict[str, object]:
    fields = (
        "schema_version",
        "control_name",
        "namespace",
        "query_id",
        "target_free",
        "label_target_rival_read_count",
        "scientific_reduction_count",
        "candidate_count",
        "physical_gallery_row_count",
        "pair_member_count",
        "pair_member_keys",
        "candidate_axis_sha256",
        "axis_records",
        "destination_to_source",
        "pair_bindings",
        "changed_variable",
        "fixed_point_free",
        "corrected_identity_disjoint",
        "pair_member_donor_exclusion",
        "candidate_axis_preserved",
        "p_lock_preserved",
        "full_native_content_binding_permutation",
        "input_p_lock_population_sha256",
        "output_p_lock_population_sha256",
        "input_native_content_axis_sha256",
        "output_native_content_multiset_sha256",
        "input_native_content_multiset_sha256",
        "full_axis_planning_candidate_count",
        "materialized_control_candidate_count",
        "maximum_device_transfer_candidate_count",
        "model_load_count",
        "model_forward_count",
        "model_backward_count",
        "model_update_count",
        "logical_sha256",
    )
    value = exact_mapping(receipt, fields, name="C_DINO receipt")
    require(value["schema_version"] == INPUT_SCHEMA, "C_DINO receipt schema drift")
    require(value["control_name"] == "C_DINO_V", "control name drift")
    require(
        isinstance(value["namespace"], str)
        and bool(value["namespace"])
        and isinstance(value["query_id"], str)
        and bool(value["query_id"]),
        "control namespace/query ID drift",
    )
    require(
        value["target_free"] is True
        and value["label_target_rival_read_count"] == 0
        and value["scientific_reduction_count"] == 0,
        "target-free boundary drift",
    )
    require(
        value["candidate_count"] == CANDIDATE_COUNT
        and value["physical_gallery_row_count"] == PHYSICAL_GALLERY_ROWS
        and value["full_axis_planning_candidate_count"] == CANDIDATE_COUNT
        and value["pair_member_count"] == PAIR_MEMBER_COUNT
        and value["materialized_control_candidate_count"] == PAIR_MEMBER_COUNT
        and value["maximum_device_transfer_candidate_count"] == PAIR_MEMBER_COUNT,
        "full-axis/pair-only cardinality drift",
    )
    require(
        value["changed_variable"]
        == "DINO_REFERENCE_NATIVE_CONTENT_BINDING_ACROSS_FULL_C128"
        and value["fixed_point_free"] is True
        and value["corrected_identity_disjoint"] is True
        and value["pair_member_donor_exclusion"] is True
        and value["candidate_axis_preserved"] is True
        and value["p_lock_preserved"] is True
        and value["full_native_content_binding_permutation"] is True,
        "control invariant declaration drift",
    )
    require(
        all(
            value[name] == 0
            for name in (
                "model_load_count",
                "model_forward_count",
                "model_backward_count",
                "model_update_count",
            )
        ),
        "receipt reports model execution",
    )
    require(
        is_sha256(value["logical_sha256"])
        and value["logical_sha256"]
        == canonical_sha256(
            {key: item for key, item in value.items() if key != "logical_sha256"}
        ),
        "receipt logical hash drift",
    )

    axis_value = value["axis_records"]
    require(
        isinstance(axis_value, list) and len(axis_value) == CANDIDATE_COUNT,
        "receipt axis is not C128",
    )
    axis_fields = (
        "candidate_position",
        "candidate_key",
        "candidate_physical_row",
        "candidate_reference_source_sha256",
        "corrected_identity_sha256",
        "p_lock_record_sha256_by_direction",
        "candidate_native_content_sha256_by_direction",
        "dino_content_binding_sha256",
        "axis_index",
    )
    axis: list[Mapping[str, Any]] = []
    for index, raw in enumerate(axis_value):
        item = exact_mapping(raw, axis_fields, name="C128 axis record")
        require(item["axis_index"] == index, "C128 canonical axis index drift")
        require(
            type(item["candidate_position"]) is int
            and 0 <= item["candidate_position"] < CANDIDATE_COUNT
            and type(item["candidate_physical_row"]) is int
            and 0 <= item["candidate_physical_row"] < PHYSICAL_GALLERY_ROWS,
            "C128 candidate numeric address drift",
        )
        require(
            isinstance(item["candidate_key"], str)
            and bool(item["candidate_key"])
            and is_sha256(item["candidate_reference_source_sha256"])
            and is_sha256(item["corrected_identity_sha256"]),
            "C128 candidate string address drift",
        )
        require(
            is_sha256(item["dino_content_binding_sha256"]),
            "C128 DINO content binding drift",
        )
        for field in (
            "p_lock_record_sha256_by_direction",
            "candidate_native_content_sha256_by_direction",
        ):
            directional = exact_mapping(item[field], DIRECTIONS, name=field)
            require(
                all(is_sha256(directional[direction]) for direction in DIRECTIONS),
                f"{field} digest drift",
            )
        axis.append(item)
    require(
        len({item["candidate_position"] for item in axis}) == CANDIDATE_COUNT
        and len({item["candidate_key"] for item in axis}) == CANDIDATE_COUNT
        and len({item["candidate_physical_row"] for item in axis})
        == CANDIDATE_COUNT,
        "C128 axis collision",
    )
    require(
        axis
        == sorted(
            axis,
            key=lambda item: (
                item["candidate_physical_row"],
                item["candidate_reference_source_sha256"],
                item["candidate_key"],
            ),
        ),
        "receipt axis is not in canonical matching order",
    )
    by_position = sorted(axis, key=lambda item: item["candidate_position"])
    rows = [item["candidate_physical_row"] for item in by_position]
    require(rows == sorted(rows), "natural candidate-position row order drift")
    require(
        is_sha256(value["candidate_axis_sha256"])
        and value["candidate_axis_sha256"] == canonical_sha256(rows),
        "candidate-axis hash replay drift",
    )

    pair_keys = value["pair_member_keys"]
    require(
        isinstance(pair_keys, list)
        and len(pair_keys) == PAIR_MEMBER_COUNT
        and len(set(pair_keys)) == PAIR_MEMBER_COUNT
        and set(pair_keys).issubset({item["candidate_key"] for item in axis}),
        "anonymous pair address drift",
    )
    observed_order = value["destination_to_source"]
    require(
        isinstance(observed_order, list)
        and len(observed_order) == CANDIDATE_COUNT
        and all(type(item) is int for item in observed_order)
        and sorted(observed_order) == list(range(CANDIDATE_COUNT))
        and all(index != source for index, source in enumerate(observed_order)),
        "destination-to-source is not a fixed-point-free C128 permutation",
    )
    recomputed = deterministic_identity_disjoint_matching(
        axis,
        namespace=str(value["namespace"]),
        query_id=str(value["query_id"]),
        protected_pair_keys=pair_keys,
    )
    require(list(recomputed) == observed_order, "full-C128 donor mapping replay drift")
    require(
        all(
            axis[index]["corrected_identity_sha256"]
            != axis[source]["corrected_identity_sha256"]
            for index, source in enumerate(observed_order)
        ),
        "same corrected identity crossed the C_DINO mapping",
    )
    index_by_key = {
        str(item["candidate_key"]): index for index, item in enumerate(axis)
    }
    pair_donors = {
        str(axis[observed_order[index_by_key[str(key)]]]["candidate_key"])
        for key in pair_keys
    }
    require(
        pair_donors.isdisjoint(map(str, pair_keys)),
        "scored pair received a donor from inside the scored pair",
    )

    p_population = [item["p_lock_record_sha256_by_direction"] for item in axis]
    p_sha = canonical_sha256(p_population)
    require(
        value["input_p_lock_population_sha256"]
        == value["output_p_lock_population_sha256"]
        == p_sha,
        "destination P-lock population changed",
    )
    native_axis = [
        {
            "candidate_key": item["candidate_key"],
            "candidate_physical_row": item["candidate_physical_row"],
            "candidate_reference_source_sha256": item[
                "candidate_reference_source_sha256"
            ],
            "candidate_native_content_sha256_by_direction": item[
                "candidate_native_content_sha256_by_direction"
            ],
            "dino_content_binding_sha256": item[
                "dino_content_binding_sha256"
            ],
        }
        for item in axis
    ]
    output_native = [native_axis[source] for source in observed_order]
    input_multiset_sha = canonical_sha256(
        sorted(canonical_sha256(item) for item in native_axis)
    )
    output_multiset_sha = canonical_sha256(
        sorted(canonical_sha256(item) for item in output_native)
    )
    require(
        value["input_native_content_axis_sha256"] == canonical_sha256(native_axis)
        and value["input_native_content_multiset_sha256"] == input_multiset_sha
        and value["output_native_content_multiset_sha256"] == output_multiset_sha
        and input_multiset_sha == output_multiset_sha,
        "native-content permutation replay drift",
    )

    pair_value = value["pair_bindings"]
    require(
        isinstance(pair_value, list) and len(pair_value) == PAIR_MEMBER_COUNT,
        "pair binding receipt count drift",
    )
    pair_fields = (
        "destination_axis_index",
        "destination_candidate_key",
        "destination_physical_gallery_row",
        "destination_corrected_identity_sha256",
        "donor_axis_index",
        "donor_candidate_key",
        "donor_physical_gallery_row",
        "donor_corrected_identity_sha256",
        "donor_dino_content_binding_sha256",
        "destination_p_lock_input_sha256_by_direction",
        "destination_p_lock_output_sha256_by_direction",
        "donor_content",
        "controlled_content",
    )
    for pair_index, raw in enumerate(pair_value):
        item = exact_mapping(raw, pair_fields, name="pair binding")
        destination_index = item["destination_axis_index"]
        donor_index = item["donor_axis_index"]
        require(
            type(destination_index) is int
            and type(donor_index) is int
            and 0 <= destination_index < CANDIDATE_COUNT
            and 0 <= donor_index < CANDIDATE_COUNT,
            "pair binding axis index drift",
        )
        destination = axis[destination_index]
        donor = axis[donor_index]
        require(
            item["destination_candidate_key"] == pair_keys[pair_index]
            == destination["candidate_key"]
            and item["destination_physical_gallery_row"]
            == destination["candidate_physical_row"]
            and item["destination_corrected_identity_sha256"]
            == destination["corrected_identity_sha256"],
            "pair destination replay drift",
        )
        require(
            donor_index == observed_order[destination_index]
            and item["donor_candidate_key"] == donor["candidate_key"]
            and item["donor_physical_gallery_row"]
            == donor["candidate_physical_row"]
            and item["donor_corrected_identity_sha256"]
            == donor["corrected_identity_sha256"],
            "pair donor replay drift",
        )
        require(
            item["donor_dino_content_binding_sha256"]
            == donor["dino_content_binding_sha256"],
            "pair donor DINO content-binding replay drift",
        )
        p_input = exact_mapping(
            item["destination_p_lock_input_sha256_by_direction"],
            DIRECTIONS,
            name="pair input P lock",
        )
        p_output = exact_mapping(
            item["destination_p_lock_output_sha256_by_direction"],
            DIRECTIONS,
            name="pair output P lock",
        )
        require(
            dict(p_input)
            == dict(p_output)
            == dict(destination["p_lock_record_sha256_by_direction"]),
            "pair destination P lock changed",
        )
        content_fields = (
            "source_image_sha256",
            "source_key",
            "cache_payload_sha256",
            "geometry_record_sha256",
            "tokens_sha256",
            "grid_shape",
            "valid_patch_mask_sha256",
        )
        donor_content = exact_mapping(
            item["donor_content"], content_fields, name="pair donor content"
        )
        controlled_content = exact_mapping(
            item["controlled_content"],
            content_fields,
            name="pair controlled content",
        )
        require(
            dict(donor_content) == dict(controlled_content),
            "pair donor content was changed during rebinding",
        )
        require(
            donor_content["source_image_sha256"]
            == donor["candidate_reference_source_sha256"]
            and all(
                is_sha256(donor_content[name])
                for name in (
                    "source_image_sha256",
                    "cache_payload_sha256",
                    "geometry_record_sha256",
                    "tokens_sha256",
                    "valid_patch_mask_sha256",
                )
            ),
            "pair donor tensor provenance drift",
        )

    result: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA,
        "status": OUTPUT_STATUS,
        "input_receipt_logical_sha256": value["logical_sha256"],
        "query_id": value["query_id"],
        "candidate_axis_sha256": value["candidate_axis_sha256"],
        "validated_candidate_count": CANDIDATE_COUNT,
        "validated_pair_member_count": PAIR_MEMBER_COUNT,
        "mapping_recomputed_independently": True,
        "fixed_point_count": 0,
        "same_corrected_identity_edge_count": 0,
        "p_lock_drift_count": 0,
        "target_or_rival_read_count": 0,
        "model_execution_count": 0,
        "scientific_GO_or_NO_GO": None,
    }
    result["logical_sha256"] = canonical_sha256(result)
    return result


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), "receipt JSON is not a mapping")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    output = path.resolve()
    require(not output.exists(), f"immutable replay output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    output.chmod(0o444)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = replay_receipt(read_json(args.receipt.resolve()))
    if args.output is not None:
        atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
