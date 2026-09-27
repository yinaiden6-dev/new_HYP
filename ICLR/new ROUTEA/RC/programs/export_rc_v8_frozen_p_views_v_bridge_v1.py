#!/usr/bin/env python3
"""Export saved V8 geometry to FrozenPView; no V scores or scientific GO.

This bridge reads only existing decision/provenance artifacts. It neither
generates proposals nor reads target-role joins, raw tokens, or model weights.
The old V8 NO-GO remains in force. A structurally valid view is not a qualified
natural P seal and does not authorize natural V execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from rc_aslo_xf.frozen_p_raw_colnomic_three_arm_v1 import (
    FrozenPView, seal_frozen_p_view, validate_frozen_p_view,
)

SOURCE_PINS = {
    "score_records": ("results/rc_coherent_residual_development_v8/score_records.jsonl", "b43d04a37a50d45d65c550baab4e4a4d60e402f7f51f92fa2b2013943e2c5ac4"),
    "result": ("results/rc_coherent_residual_development_v8/result.json", "09374961f1c4d62f381bc0e519e73851c04ad78dec346fa1dd0314471c7f734e"),
    "validation": ("results/rc_coherent_residual_development_v8_validation/result.json", "b83ebcc18f76eaf6327ec5aef6b6206574d06accc6e86cd54dafad65521f6f21"),
    "authority": ("registry/rc_coherent_residual_development_authority_v8_20260908.json", "4a6d24014f7e2e8b13b3de9b41faf76e1d1ad7a8c625ee47db442f6755f34380"),
    "final_checkpoint": ("results/rc_coherent_residual_development_v8/final_checkpoint.json", "8ccf6b9815025ffe494ed9d9db1a2c504ea14d267496d59f9707c847b22283ef"),
    "generation_prejoin_seal": ("results/rc_cycle_closed_region_generation_v1/prejoin_seal.json", "e3a7808fa77f208a703c85a0f6e1e2e6fae18051aa552cb9dbc9676003b93896"),
    "generation_authority": ("registry/rc_cycle_closed_region_generation_authority_v1_20260908.json", "5472d55fde3839d668c4cbb94917d9ec2611131a1f7840e1fa988e748a40b85c"),
    "v8_core": ("src/rc_aslo_xf/reference_conditioned_coherent_residual_p_only_v8.py", "19806530ba94c4fbdfac050cf437326e9790b446633f2b5d4b001dbb371d6fd2"),
}
CONTROLS = {"REAL": "decisions", "C_BIND": "c_bind_decisions", "P_COORD": "p_coord_decisions"}
GEOMETRY_FIELDS = ("atom_indices", "query_indices", "query_rc", "reference_indices", "reference_rc", "source_query_indices", "source_reference_indices")
TENSOR_FORMAT = {"dtype": "torch.int64", "device": "cpu", "encoding": "JSON_INTEGER_LIST", "rank": 1}
CLAIM = "ENGINEERING_STRUCTURAL_P_VIEW_COMPATIBILITY_ONLY"
DEFAULT_OUT = ROOT / "results/rc_v8_frozen_p_views_v_bridge_v1"


def need(value, message):
    if not bool(value):
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_new(path, value):
    with path.open("xb") as handle:
        handle.write(canonical(value) + b"\n")


def load_frozen_view(value: dict) -> FrozenPView:
    """Decode and validate a standard CPU int64 FrozenPView JSON record."""
    expected = set(FrozenPView.__dataclass_fields__) | {"tensor_format"}
    need(set(value) == expected and value["tensor_format"] == TENSOR_FORMAT, "VIEW_JSON_ENVELOPE_DRIFT")
    item = {key: value[key] for key in FrozenPView.__dataclass_fields__}
    for field in ("query_indices", "reference_indices"):
        need(isinstance(item[field], list) and all(type(x) is int for x in item[field]), "VIEW_INDEX_NOT_INTEGER_LIST")
        item[field] = torch.tensor(item[field], dtype=torch.int64)
    for field in ("query_grid_shape", "reference_grid_shape"):
        need(isinstance(item[field], list) and len(item[field]) == 2 and all(type(x) is int for x in item[field]), "VIEW_GRID_NOT_INTEGER_PAIR")
        item[field] = tuple(item[field])
    return validate_frozen_p_view(FrozenPView(**item))


def dump_view(value):
    result = {key: getattr(value, key) for key in FrozenPView.__dataclass_fields__}
    for key in ("query_indices", "reference_indices"):
        result[key] = result[key].tolist()
    for key in ("query_grid_shape", "reference_grid_shape"):
        result[key] = list(result[key])
    result["tensor_format"] = TENSOR_FORMAT
    return result


def make_view(decision, family, binding):
    for field in ("query_resource_key", "candidate_resource_key", "reference_resource_key"):
        need(decision[field] == family[field], "DECISION_GENERATION_RESOURCE_AXIS_MISMATCH:" + field)
    need(decision["arm"] == "REAL" and decision["control"] in CONTROLS, "SOURCE_ARM_OR_CONTROL_DRIFT")
    state, ordinal, selected = decision["state"], decision["map_seed_ordinal"], decision["selected_H"]
    components = family["components"]
    if state == "H0":
        need(not components and ordinal is None and selected is None and family["structural_h0"], "H0_NOT_STRUCTURALLY_EMPTY")
        geometry = None
        qi, ri = [], []
    else:
        need(state == "H1" and type(ordinal) is int and 0 <= ordinal < len(components) and not family["structural_h0"], "H1_MAP_OUTSIDE_COMPLETE_FAMILY")
        geometry = {key: components[ordinal][key] for key in GEOMETRY_FIELDS}
        need(set(selected) == set(GEOMETRY_FIELDS) | {"pooling_witness"}, "SELECTED_SUPPORT_SCHEMA_DRIFT")
        need({key: selected[key] for key in GEOMETRY_FIELDS} == geometry, "SELECTED_H_NOT_ORIGINAL_GENERATED_COMPONENT")
        qi, ri = geometry["query_indices"], geometry["reference_indices"]
        need(len(qi) == len(set(qi)) and len(ri) == len(qi), "NONUNIQUE_QUERY_OR_UNPAIRED_REFERENCE_AXIS")
        for indices, rc, shape in ((qi, geometry["query_rc"], family["query_grid_shape"]), (ri, geometry["reference_rc"], family["reference_grid_shape"])):
            need(rc == [[x // shape[1], x % shape[1]] for x in indices], "INDEX_COORDINATE_BINDING_DRIFT")
    payload = dict(state=state, query_resource_key=decision["query_resource_key"],
        candidate_resource_key=decision["candidate_resource_key"], reference_resource_key=decision["reference_resource_key"],
        query_grid_shape=tuple(family["query_grid_shape"]), reference_grid_shape=tuple(family["reference_grid_shape"]),
        query_indices=torch.tensor(qi, dtype=torch.int64), reference_indices=torch.tensor(ri, dtype=torch.int64),
        source_p_logical_sha256=logical({"source_bindings": binding, "source_control": decision["control"],
            "state": state, "candidate_resource_key": decision["candidate_resource_key"],
            "reference_resource_key": decision["reference_resource_key"], "map_seed_ordinal": ordinal,
            "query_grid_shape": family["query_grid_shape"], "reference_grid_shape": family["reference_grid_shape"],
            "selected_geometry": geometry}),
        source_p_authority_sha256=SOURCE_PINS["authority"][1], control_namespace=decision["control"], logical_sha256="PENDING")
    return validate_frozen_p_view(seal_frozen_p_view(FrozenPView(**payload)))


def export(out):
    torch.set_num_threads(1)
    need(not out.exists(), "APPEND_ONLY_OUTPUT_ALREADY_EXISTS")
    initial = {name: {"path": rel, "sha256": sha_file(ROOT / rel)} for name, (rel, _) in SOURCE_PINS.items()}
    for name, (_, expected) in SOURCE_PINS.items():
        need(initial[name]["sha256"] == expected, "FROZEN_SOURCE_HASH_DRIFT:" + name)
    # Checkpoint is hashed as opaque bytes. Its parameters are never decoded.
    result = json.loads((ROOT / SOURCE_PINS["result"][0]).read_text())
    validation = json.loads((ROOT / SOURCE_PINS["validation"][0]).read_text())
    scores = [json.loads(line) for line in (ROOT / SOURCE_PINS["score_records"][0]).read_text().splitlines()]
    need(result["gate_go"] is False and validation["gate_go"] is False and result["formal_panel_consumed"] is False, "SOURCE_SCIENTIFIC_STATUS_DRIFT")
    need(validation["status"] == "RC_COHERENT_RESIDUAL_V8_DEVELOPMENT_INDEPENDENT_VALIDATION_PASS", "SOURCE_INDEPENDENT_VALIDATION_MISSING")
    # The source runner seals the parsed row array logically, in addition to
    # this bridge's separate byte hash of the JSONL file.
    need(result["score_records_sha256"] == logical(scores) and validation["result_sha256"] == initial["result"]["sha256"], "SOURCE_RESULT_CROSS_BINDING_DRIFT")
    need(result["final_checkpoint_sha256"] == validation["checkpoint_sha256"] == initial["final_checkpoint"]["sha256"], "SOURCE_CHECKPOINT_CROSS_BINDING_DRIFT")
    need(result["authority_sha256"] == validation["authority_sha256"] == initial["authority"]["sha256"], "SOURCE_AUTHORITY_CROSS_BINDING_DRIFT")
    seal = json.loads((ROOT / SOURCE_PINS["generation_prejoin_seal"][0]).read_text())
    need(seal["prejoin_label_reads"] == 0 and seal["authority_sha256"] == initial["generation_authority"]["sha256"], "GENERATION_PREJOIN_PROVENANCE_DRIFT")
    generation = {entry["query_resource_key"]: entry for entry in seal["records"]}
    need(len(scores) == len(generation) == len(seal["records"]) == 32, "FULL_TRAIN32_AXIS_MISSING")
    need([record["query_resource_key"] for record in scores] == sorted(generation), "QUERY_AXIS_NOT_CANONICAL_OR_COMPLETE")
    support_validator = ROOT / "src/rc_aslo_xf/frozen_p_raw_colnomic_three_arm_v1.py"
    type_validator = ROOT / "src/rc_aslo_xf/current_d1_roma_rawlocal_hyp_e0_v1.py"
    implementation = {"exporter": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha_file(Path(__file__))},
        "existing_frozen_view_validator": {"path": str(support_validator.relative_to(ROOT)), "sha256": sha_file(support_validator)},
        "existing_type_definitions": {"path": str(type_validator.relative_to(ROOT)), "sha256": sha_file(type_validator)}}
    out.mkdir(); (out / "prejoin").mkdir()
    outputs, source_generation_files, counts = [], [], Counter()
    for score in scores:
        query = score["query_resource_key"]
        entry = generation[query]
        path = ROOT / "results/rc_cycle_closed_region_generation_v1" / entry["file"]
        need(sha_file(path) == entry["sha256"], "GENERATION_QUERY_HASH_DRIFT:" + query)
        gen = json.loads(path.read_text())
        keys = score["candidate_keys"]
        need(keys == sorted(set(keys)) and len(keys) == 128 and keys == gen["candidate_keys"] and gen["query_resource_key"] == query, "FULL_C128_JOIN_MISMATCH")
        need(gen["authority_sha256"] == initial["generation_authority"]["sha256"], "GENERATION_AUTHORITY_DRIFT")
        source_generation_files.append({"path": str(path.relative_to(ROOT)), "sha256": entry["sha256"]})
        binding = {"sources": initial, "generation_query": source_generation_files[-1],
            "query_resource_key": query, "source_context_sha256": score["context_sha256"],
            "source_atom_payload_sha256": gen["source_payload_sha256"]}
        record = {"schema": "rc_v8_frozen_p_view_query_v_bridge_v1", "query_resource_key": query,
            "candidate_keys": keys, "source_bindings": binding, "views": {}, "claim_level": CLAIM}
        for control, decision_field in CONTROLS.items():
            decisions, families = score["arms"]["REAL"][decision_field], gen["families"][control]
            need(len(decisions) == len(families) == 128, "CONTROL_C128_INCOMPLETE")
            need([d["candidate_resource_key"] for d in decisions] == keys and [f["candidate_resource_key"] for f in families] == keys, "CONTROL_CANDIDATE_AXIS_DRIFT")
            need(all(d["control"] == control for d in decisions), "CONTROL_NAMESPACE_MISMATCH")
            record["views"][control] = []
            for decision, family in zip(decisions, families, strict=True):
                view = make_view(decision, family, binding)
                encoded = dump_view(view)
                replay = load_frozen_view(json.loads(json.dumps(encoded)))
                need(replay.logical_sha256 == view.logical_sha256, "EXISTING_VIEW_VALIDATOR_JSON_RELOAD_DRIFT")
                record["views"][control].append(encoded)
                counts[(control, view.state)] += 1
        dest = out / "prejoin" / (query + ".json")
        write_new(dest, record)
        outputs.append({"query_resource_key": query, "file": str(dest.relative_to(out)), "sha256": sha_file(dest), "candidate_count": 128, "view_count": 384})
    need(sum(counts.values()) == 12288, "COMPLETE_32_C128_THREE_CONTROL_VIEW_COUNT_FAILED")
    need(counts[("P_COORD", "H1")] == 0, "FROZEN_V8_PC_STRUCTURAL_STATUS_CHANGED")
    for entry in list(initial.values()) + source_generation_files + list(implementation.values()):
        need(sha_file(ROOT / entry["path"]) == entry["sha256"], "SOURCE_CHANGED_DURING_EXPORT:" + entry["path"])
    counts_json = {control: {state: counts[(control, state)] for state in ("H0", "H1")} for control in CONTROLS}
    manifest = {"schema": "rc_v8_frozen_p_views_v_bridge_v1", "status": "RC_V8_FROZEN_P_VIEWS_ENGINEERING_EXPORT_COMPLETE",
        "claim_level": CLAIM, "sources": initial, "implementation": implementation, "records": outputs,
        "query_count": 32, "candidates_per_query": 128, "controls": list(CONTROLS), "view_count": 12288,
        "state_counts": counts_json, "tensor_format": TENSOR_FORMAT,
        "loader": "programs/export_rc_v8_frozen_p_views_v_bridge_v1.py:load_frozen_view",
        "source_V8_P_GO": False, "old_26_P_gate_passed": False, "old_26_P_gate_status": "NOT_PASSED",
        "natural_V_execution_authorized": False, "automatic_stage_advance": False,
        "formal_392_consumed": False, "target_role_join_read_count": 0, "raw_token_read_count": 0,
        "training_update_count": 0, "V_score_evaluation_count": 0, "checkpoint_parameters_decoded": False,
        "P_scores_witnesses_or_head_state_exported": False,
        "source_geometry_recomputed": False, "geometry_equals_original_V8_MAP_component": True}
    write_new(out / "manifest.json", manifest)
    write_new(out / "prejoin_seal.json", {"schema": "rc_v8_frozen_p_view_prejoin_seal_v1", "records": outputs,
        "manifest_sha256": sha_file(out / "manifest.json"), "target_role_join_read_count": 0,
        "claim_level": CLAIM, "natural_V_execution_authorized": False})
    for entry in outputs:
        need(sha_file(out / entry["file"]) == entry["sha256"], "EXPORTED_QUERY_HASH_DRIFT")
    receipt = {"status": "RC_V8_FROZEN_P_VIEWS_EXISTING_STRUCTURAL_VALIDATION_PASS", "claim_level": CLAIM,
        "manifest_sha256": sha_file(out / "manifest.json"), "prejoin_seal_sha256": sha_file(out / "prejoin_seal.json"),
        "view_count": 12288, "state_counts": counts_json, "source_V8_P_GO": False,
        "natural_V_execution_authorized": False, "scientific_GO_claimed": False,
        "checks": {"complete_32_C128_three_control_axes": True, "source_hashes_verified_unchanged": True,
            "original_MAP_geometry_exact": True, "existing_seal_and_validate_frozen_p_view_all_views": True,
            "JSON_integer_tensor_roundtrip_all_views": True, "control_reference_binding_preserved": True,
            "source_NO_GO_preserved": True},
        "validation_scope": "EXISTING_STRUCTURAL_VIEW_VALIDATOR_AND_ARTIFACT_BINDINGS_NO_RAW_GEOMETRY_OR_SCORE_REPLAY"}
    write_new(out / "validation.json", receipt)
    return {**receipt, "output_path": str(out), "validation_sha256": sha_file(out / "validation.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(export(args.out.resolve()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
