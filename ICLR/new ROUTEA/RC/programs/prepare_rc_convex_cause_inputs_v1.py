#!/usr/bin/env python3
"""Stage the frozen TRAIN objective, then six quarantined historical cones.

This entry point performs no optimization, training, prediction, or job action.
TRAIN is durably sealed before any opened-EVAL diagnostic is read. Every file
is created exclusively; an existing artifact must be exactly equivalent.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "results/rc_convex_train_loss_cause_v1"
CONES = ROOT / "results/rc_opened_convex_loss_rescue_cones_v1/inputs"
ORIGINAL = ROOT / "results/rc_original_mixed96_group_risk_v1"
HEAD = ROOT / "registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json"
DESIGN_SHA = "16739684a5902783b0793750998ed86d7e5cc89464fd9d424de5d288521ab6ee"
PARAMETER_SHA = "ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263"
INITIAL_PINS = {
    "train_inputs.pt": "01ebc493e325fb15fce26cec62ec96a731bed3b149a239bac8f5206e88abd42b",
    "train_input_seal.json": "79ebf43494267b144bb66fc60a90c111f9b7f0b42090d951d509d63e8bee54ba",
    "head": "42c8e503874cb807ca7085a39c802c016de2b1b8c6124bb17a56ed6f06bd174b",
}
_train_sealed = False


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def checked(binding):
    path = Path(binding["path"])
    if not path.is_absolute():
        path = ROOT / path
    need(sha(path) == binding["sha256"], "SOURCE_SHA_MISMATCH: " + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text())


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encode(value) + b"\n"
    if path.exists():
        need(path.read_bytes() == data, "EXISTING_OUTPUT_NOT_EQUIVALENT: " + str(path))
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def access_guard(event, arguments):
    if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
        return
    path = str(Path(os.fsdecode(arguments[0])).absolute()).lower()
    need(not any(token in path for token in (
        "d1-mi", "d1_mi", "grozi", "gisc_prerecall_universe", "/target_join/", "formal_private"
    )), "PROTECTED_SOURCE_FORBIDDEN")
    if not _train_sealed:
        need("/results/rc_opened_" not in path, "TRAIN_MUST_BE_SEALED_BEFORE_EVAL_OPEN")


def tensor_sha(tensor):
    import torch
    x = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str(x.dtype).encode() + encode(list(x.shape)) +
                          x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def stage_train():
    global _train_sealed
    import numpy as np
    import torch
    from rc_convex_loss_cause_core_v1 import build_terms, decode_terms

    for name in ("train_inputs.pt", "train_input_seal.json"):
        need(sha(ORIGINAL / name) == INITIAL_PINS[name], "FROZEN_TRAIN_PIN_" + name)
    need(sha(HEAD) == INITIAL_PINS["head"], "FROZEN_HEAD_PIN")
    seal = read(ORIGINAL / "train_input_seal.json")
    head = read(HEAD)
    need(seal["status"] == "ORIGINAL_MIXED96_INPUTS_FROZEN", "ORIGINAL_TRAIN_STATUS")
    need(seal["counts"] == {"FULL": 32, "PAIR": 64, "identities": 32, "unique_images": 96}, "ORIGINAL_COUNTS")
    need(seal["original_head_training_updates"] == 0, "ORIGINAL_HEAD_UNCHANGED")
    data = torch.load(checked(seal["payload"]), weights_only=True, map_location="cpu")
    pair, full = data["pair"]["records"], data["full"]
    need(len(pair) == 64 and len(full) == 32, "PAIR64_FULL32")
    need(len({r["query_id"] for r in pair + full}) == 96, "UNIQUE_TRAIN96")
    need([r["execution_ordinal"] for r in pair] == [r[2] for r in seal["original_pair_order"]], "ORIGINAL_PAIR_ORDER")
    need([r["execution_ordinal"] for r in full] == seal["original_training_order"], "ORIGINAL_FULL_ORDER")
    for pool, rows, count in (("PAIR", pair, 1), ("FULL", full, 127)):
        for row in rows:
            x = row["real_native_features"]["C_PAIRED"]
            need(x.dtype == torch.float64 and tuple(x.shape) == (count, 6), pool + "_FEATURE_SHAPE")
            need(bool(torch.isfinite(x).all()), "FINITE_TRAIN_ENDPOINTS")
            challengers = row["challenger_positions"]
            need(len(challengers) == len(set(challengers)) == count, "CHALLENGER_COUNT")
            need(len(row["candidate_physical_rows"]) == len(set(row["candidate_physical_rows"])) == 128, "RAW_C128")
            need(row["base_winner_position"] not in challengers, "BASE_EXCLUDED")
            need(row["target_position"] in [row["base_winner_position"]] + challengers, "NATURAL_TARGET")
            if pool == "FULL":
                need(sorted(challengers + [row["base_winner_position"]]) == list(range(128)), "COMPLETE_FULL127")
            else:
                need(bool(row["switch_label"]) == (row["target_position"] != row["base_winner_position"]), "PAIR_SWITCH_LABEL")
    design = torch.cat([r["real_native_features"]["C_PAIRED"] for r in pair + full])
    need(tuple(design.shape) == (4128, 6), "ORIGINAL_4128X6")
    need(tensor_sha(design) == seal["feature_sha256"] == DESIGN_SHA, "ORIGINAL_TRAIN_FEATURE_BITS")
    px = np.concatenate((design[:64].numpy(), np.ones((64, 1))), axis=1)
    fx = np.concatenate((design[64:].numpy().reshape(32, 127, 6), np.ones((32, 127, 1))), axis=2)
    targets = [-1 if r["target_position"] == r["base_winner_position"] else
               r["challenger_positions"].index(r["target_position"]) for r in full]
    labels = [int(r["switch_label"]) for r in pair]
    terms = build_terms(px, labels, fx, targets)
    encoded_terms = [{"weight": str(t["weight"]), "vectors": [[float(v).hex() for v in row]
                     for row in t["vectors"]]} for t in terms]
    need(decode_terms(encoded_terms) == terms, "LOSS_TERMS_EXACT_ROUNDTRIP")
    theta_hex = head["weight_binary64"] + [head["bias_binary64"]]
    theta = [float.fromhex(x) for x in theta_hex]
    need(len(theta) == 7 and all(math.isfinite(x) for x in theta), "OLD_THETA7_FINITE")
    need(hashlib.sha256(encode({"weight": theta[:6], "bias": theta[6]})).hexdigest() ==
         head["parameter_sha256"] == PARAMETER_SHA, "OLD_EC7_PARAMETERS")
    row_order = []
    offset = 0
    for pool, rows in (("PAIR", pair), ("FULL", full)):
        for index, row in enumerate(rows):
            item = {key: row[key] for key in ("query_id", "execution_ordinal", "target_position",
                                             "base_winner_position", "challenger_positions")}
            item.update(pool=pool, pool_index=index, feature_row_start=offset,
                        feature_row_count=len(row["challenger_positions"]))
            if pool == "PAIR":
                item["switch_label"] = bool(row["switch_label"])
                item["original_pair_order"] = seal["original_pair_order"][index]
            offset += item["feature_row_count"]
            row_order.append(item)
    counts = {**seal["counts"], "feature_rows": 4128, "feature_columns": 6, "parameters": 7,
              "PAIR_positive": sum(labels), "PAIR_negative": 64 - sum(labels),
              "FULL_RAW_true": targets.count(-1), "FULL_RAW_wrong": 32 - targets.count(-1),
              "terms": len(terms), "term_vectors": sum(len(t["vectors"]) for t in terms)}
    sources = {"original_train_pack": bind(ORIGINAL / "train_inputs.pt"),
               "original_train_input_seal": bind(ORIGINAL / "train_input_seal.json"),
               "old_head_parameter_seal": bind(HEAD)}
    problem = {"schema_version": 1, "status": "TRAIN_ONLY_ORIGINAL_SIGN_CONVEX_OBJECTIVE_FROZEN",
               "terms": encoded_terms, "old_theta_binary64": theta_hex,
               "old_parameter_sha256": PARAMETER_SHA, "train_design_sha256": DESIGN_SHA,
               "original_row_order": row_order, "counts": counts, "sources": sources,
               "objective": "sum weight*softplus(max(v dot theta)); original SIGN loss; PAIR64 mean + FULL32 mean",
               "group_weights_used": False, "regularization": None,
               "evaluation_labels_included": False, "training_updates": 0}
    exclusive_json(TRAIN / "train_problem.json", problem)
    output_seal = {"schema_version": 1, "status": "TRAIN_INPUTS_SEALED_BEFORE_EVAL_CONES",
                   "train_problem": bind(TRAIN / "train_problem.json"), "sources": sources,
                   "train_design_sha256": DESIGN_SHA, "old_parameter_sha256": PARAMETER_SHA,
                   "counts": counts, "evaluation_labels_included": False,
                   "source_pins_and_order_verified": True, "training_updates": 0,
                   "evaluation_endpoint_label_or_prediction_reads_before_seal": 0,
                   "upstream_pins_preserved_from_original_seal": {
                       "sources": seal["sources"], "full_shards": seal["full_shards"],
                       "training_role_sources": seal["training_role_sources"]}}
    exclusive_json(TRAIN / "input_seal.json", output_seal)
    need(read(TRAIN / "input_seal.json")["train_problem"] == bind(TRAIN / "train_problem.json"), "DURABLE_TRAIN_SEAL")
    _train_sealed = True
    print(json.dumps({"event": "TRAIN_SEALED", "counts": counts,
                      "input_seal": bind(TRAIN / "input_seal.json")}), flush=True)


def stage_cones():
    need(_train_sealed, "TRAIN_SEAL_REQUIRED")
    import torch

    label = "NON_DEPLOYABLE_LABEL_AWARE_EVAL_DIAGNOSTIC"
    feasible = "EXACT_RATIONAL_STRICT_ACTION_FEASIBLE"
    joint = ROOT / "results/rc_opened_joint_eval_linear_retention_capacity_v1"
    old = ROOT / "results/rc_opened_eval_strict_cardinality_capacity_v1"
    pins = {
        "joint_result": (joint / "result.json", "dcce41a66b93ab6e70c181a742436cbd955c6f6196d21360c0263bac63465c07"),
        "joint_validation": (joint / "independent_validation.json", "573dbfe0c9609576b89beaa4020c723e8931615e3f523238108c9f384f5f966e"),
        "joint_endpoints": (joint / "source_feature_endpoints.json", "da8c6d30313cbb86eee71d08b1085c2938cfa581ce9bdfd7561bb526cae73241"),
        "old32_result": (old / "result.json", "42bab0902d0725d7fc4ae53017c1e2f67a618a2deff1961024959fe14357c3a7"),
        "old32_validation": (old / "independent_validation.json", "225a184777825c6b37496d4aeaade580d7f8d72af0a601483eca01eef6a2f2a1"),
        "old32_endpoints": (old / "source_feature_endpoints.json", "b1a722f64059ac4c816b7cf38aa15c94f889403d278d00f6347eb40a05502268"),
        "old32_witness": (old / "existential_certificate.json", "4c9241ba50faa53fa0219ee79c9f3cff4018cf3a0eb989fbb39b466358ddaea9"),
    }
    sources = {}
    for key, (path, expected) in pins.items():
        need(sha(path) == expected, "FROZEN_CONE_SOURCE_PIN: " + key)
        sources[key] = bind(path)
    result = read(joint / "result.json")
    validation = read(joint / "independent_validation.json")
    endpoints = read(joint / "source_feature_endpoints.json")
    old_result = read(old / "result.json")
    old_validation = read(old / "independent_validation.json")
    old_endpoints = read(old / "source_feature_endpoints.json")
    old_witness = read(old / "existential_certificate.json")
    for document in (result, validation, endpoints, old_result, old_validation, old_endpoints, old_witness):
        need(document["diagnostic_label"] == label, "HISTORICAL_DIAGNOSTIC_QUARANTINE")
    need(validation["result_sha256"] == sources["joint_result"]["sha256"], "JOINT_VALIDATOR_RESULT_SHA")
    need(old_validation["result_sha256"] == sources["old32_result"]["sha256"], "OLD32_VALIDATOR_RESULT_SHA")
    checked(validation["validator"])
    checked(old_validation["validator"])
    need(checked(result["source_feature_endpoints"]) == joint / "source_feature_endpoints.json", "JOINT_ENDPOINT_BINDING")
    need(checked(old_result["artifacts"]["existential_certificate.json"]) == old / "existential_certificate.json", "OLD32_WITNESS_BINDING")
    need(checked(old_result["artifacts"]["source_feature_endpoints.json"]) == old / "source_feature_endpoints.json", "OLD32_ENDPOINT_BINDING")
    need(validation["exact_feasible_system_count"] == result["exact_feasible_cases"] == 5, "FIVE_FIXED_FEASIBLE_SYSTEMS")
    need(validation["all22_system_roles_and_constraints_independently_verified"] is True and
         validation["exact_original_FP64_endpoints"] is True, "JOINT_ENDPOINT_ROLE_VALIDATION")
    need(validation["status"] in (
        "OPENED_JOINT_EVAL_LINEAR_RETENTION_CAPACITY_INDEPENDENT_EXACT_VALIDATION_PASS",
        "OPENED_JOINT_EVAL_LINEAR_RETENTION_CAPACITY_INDEPENDENT_VALIDATION_UNRESOLVED"
    ), "JOINT_PRIOR_VALIDATION_STATUS")
    need(old_validation["status"] == "OPENED_EVAL_STRICT_CARDINALITY_INDEPENDENT_EXACT_CERTIFICATE_VALIDATION_PASS"
         and old_validation["exact_feasible_witness_verified"] is True, "OLD32_PRIOR_EXACT_WITNESS_VALIDATED")
    need(old_witness["not_a_deployable_or_trained_model"] is True and
         old_witness["no_predictions_on_omitted_queries"] is True, "OLD32_WITNESS_SCOPE")
    rows = endpoints["rows"]
    need(len(rows) == 160 and len({row["query_id"] for row in rows}) == 160, "FIXED_EVAL160_ENDPOINTS")
    by_query = {row["query_id"]: row for row in rows}
    old_by_query = {row["query_id"]: row for row in old_endpoints["rows"]}
    need(len(old_by_query) == 32, "OLD32_ENDPOINT_COUNT")
    for row in rows:
        need(row["query_id"] == row["panel"] + "|" + row["source_query_id"], "QUERY_ID_PANEL_SOURCE_MAPPING")
        x = [[float.fromhex(value) for value in values] for values in row["native6_binary64"]]
        need(len(x) == 127 and all(len(values) == 6 and all(math.isfinite(v) for v in values) for values in x), "ENDPOINT127X6_FINITE")
        need(tensor_sha(torch.tensor(x, dtype=torch.float64)) == row["source_feature_tensor_sha256"], "ENDPOINT_TENSOR_BITS")
        if row["panel"] == "EVAL32":
            prior = old_by_query[row["source_query_id"]]
            for key in ("native6_binary64", "target_position", "base_winner_position", "challenger_positions", "candidate_physical_rows"):
                need(row[key] == prior[key], "OLD32_SHARED_ENDPOINT_EXACT_PARITY: " + key)

    def equations(row):
        challengers = row["challenger_positions"]
        target, winner = row["target_position"], row["base_winner_position"]
        need(target is not None and sorted(challengers + [winner]) == list(range(128)), "CONE_TARGET_AND_FULL_AXIS")
        vectors = {candidate: [Fraction.from_float(float.fromhex(value)) for value in values] + [Fraction(1)]
                   for candidate, values in zip(challengers, row["native6_binary64"])}
        values, metadata = [], []
        if target == winner:
            for candidate in challengers:
                values.append([-v for v in vectors[candidate]])
                metadata.append({"query_id": row["query_id"], "kind": "BASE_CORRECT_WRONG_BELOW_ZERO", "candidate_position": candidate})
        else:
            need(target in vectors, "TARGET_CHALLENGER_PRESENT")
            values.append(vectors[target])
            metadata.append({"query_id": row["query_id"], "kind": "TRUE_CHALLENGER_ABOVE_HOLD", "candidate_position": target})
            for candidate in challengers:
                if candidate != target:
                    values.append([left - right for left, right in zip(vectors[target], vectors[candidate])])
                    metadata.append({"query_id": row["query_id"], "kind": "TRUE_CHALLENGER_ABOVE_WRONG", "candidate_position": candidate})
        need(len(values) == len(metadata) == 127, "EXACT_CONSTRAINT_COUNT_PER_QUERY")
        return values, metadata

    equation_cache = {row["query_id"]: equations(row) for row in rows if row["target_position"] is not None}
    checks = {item["query_id"]: item for item in validation["systems"]}
    keys = [key for key in result["eligible_error_query_ids"] if result["systems"][key]["status"] == feasible]
    need(len(keys) == 5 and len(set(keys)) == 5, "FIVE_ORIGINAL_ORDER_CONES")
    specifications = []
    protected = set(result["protected_query_ids"])
    for key in keys:
        certificate = result["systems"][key]
        need(checks[key]["exact_certificate_valid"] is True and checks[key]["certificate_status"] == feasible,
             "INDIVIDUAL_PRIMAL_PROOF_VALIDATED")
        required = certificate["required_query_ids"]
        need(required == [row["query_id"] for row in rows if row["query_id"] in protected or row["query_id"] == key],
             "FIXED_PROTECTED127_PLUS_RESCUE_ORDER")
        need(certificate["exact_constraint_validation"] is True and certificate["parameter_count"] == 7,
             "JOINT_CERTIFICATE_SCHEMA")
        specifications.append((key, required, certificate["rational_unit_margin_theta"],
                               certificate["constraint_order_sha256"],
                               "Preserve the historical ec7 28 EVAL32 and 99 EVAL128 successes; rescue " + key + "."))
    old_required = ["EVAL32|" + query for query in old_witness["required_query_ids"]]
    need(len(old_required) == len(set(old_required)) == 29 and old_witness["constraint_count"] == 3683
         and old_witness["parameter_count"] == 7, "OLD32_FIXED29_WITNESS_DIMENSIONS")
    specifications.append(("EVAL32_AT_LEAST29_FIXED_EXISTENTIAL_WITNESS", old_required,
                           old_witness["rational_unit_margin_coefficients"], None,
                           "The historical EVAL32 fixed 29-query existential cone; net-gain witness that does not require preserving RAW25."))
    entries = []
    for index, (name, required, witness, order_sha, description) in enumerate(specifications):
        values = [value for query in required for value in equation_cache[query][0]]
        metadata = [value for query in required for value in equation_cache[query][1]]
        if order_sha is not None:
            need(hashlib.sha256(encode(metadata)).hexdigest() == order_sha, "ORIGINAL_CONSTRAINT_ORDER_SHA")
        rational_witness = [Fraction(value) for value in witness]
        need(len(rational_witness) == 7 and len(values) == 127 * len(required), "CONE_DIMENSIONS")
        minimum = min(sum((a * t for a, t in zip(row, rational_witness)), Fraction(0)) for row in values)
        need(minimum >= 1, "EXACT_EXISTING_WITNESS_ALL_UNIT_MARGINS")
        cone_id = "cone" + str(index).zfill(2)
        cone = {"schema_version": 1, "cone_id": cone_id, "index": index,
                "source_system_id": name, "A_exact": [[str(v) for v in row] for row in values],
                "strict_witness": [str(v) for v in rational_witness], "required_query_ids": required,
                "label_aware": True, "status": label, "diagnostic_label": label,
                "sourcebindings": sources, "description": description,
                "constraint_count": len(values), "parameter_count": 7,
                "existing_strict_witness_exact_minimum_margin": str(minimum),
                "exact_endpoint_subtraction": True, "new_predictions_computed": False,
                "model_selection_or_deployment_allowed": False,
                "closed_cone": "A_exact theta >= 0; existing strict witness satisfies A_exact theta >= 1"}
        path = CONES / (cone_id + ".json")
        exclusive_json(path, cone)
        entries.append({"cone_id": cone_id, "index": index, "source_system_id": name,
                        "input": bind(path), "constraint_count": len(values),
                        "required_query_count": len(required), "existing_witness_exact_minimum_margin": str(minimum)})
        print(json.dumps({"event": "CONE_SEALED", **entries[-1]}), flush=True)
    for binding in sources.values():
        checked(binding)
    manifest = {"schema_version": 1, "status": label, "diagnostic_label": label,
                "label_aware": True, "cones": entries, "cone_count": 6, "sourcebindings": sources,
                "train_input_seal": bind(TRAIN / "input_seal.json"),
                "train_problem": bind(TRAIN / "train_problem.json"),
                "train_sealed_before_eval_cone_reads": True,
                "existing_witnesses_all_exactly_revalidated": True,
                "fixed_cone_order": "five prior exact feasible systems in eligible_error_query_ids order, then prior EVAL32 existential witness",
                "unresolved_joint_systems_are_not_treated_as_infeasible": True,
                "new_predictions_computed": False, "optimization_performed": False,
                "training_updates": 0, "model_selection_or_deployment_allowed": False}
    exclusive_json(CONES / "manifest.json", manifest)
    print(json.dumps({"event": "SIX_CONES_SEALED", "manifest": bind(CONES / "manifest.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "cones", "all"), default="all")
    args = parser.parse_args()
    sys.addaudithook(access_guard)
    stage_train()
    if args.stage in ("cones", "all"):
        stage_cones()


if __name__ == "__main__":
    main()
