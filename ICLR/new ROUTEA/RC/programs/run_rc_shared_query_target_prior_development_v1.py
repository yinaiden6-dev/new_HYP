#!/usr/bin/env python3
"""One query-only prior over sealed NATIVE7/C; delayed EVAL role postjoin."""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PREFIX = "rc_shared_query_target_prior_development_v1"
OUT = ROOT / "results" / PREFIX
AUTH = ROOT / "registry/rc_shared_query_target_prior_development_authority_v1_20260909.json"
PLAN = ROOT / "plan/RC_SHARED_QUERY_TARGET_PRIOR_DEVELOPMENT_V1_20260909.md"
LAUNCH = ROOT / "slurm/rc_shared_query_target_prior_development_v1_dev_cpuonly_59m.sbatch"
CACHE = ROOT / "results/rc_shared_query_target_prior_cache_v1"
HEAD = ROOT / "registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json"
FULL = ROOT / "results/routea_matched_three_arm_fullnegative_features_v1"
ROLE_ROOT = ROOT / "results/cw0_rgh_xf_v2_p0_a0_manifest_v2"
MODES = ("REAL", "UNIFORM", "PRIOR_HALF_ROLL", "CBIND")
PINS = {
    "old_runner": ("programs/run_routea_matched_three_arm_common3_native7_crossfit_v1.py", "546e2bc7b3df6c67bcac41e079ba1b00f15c7e65c52a8ab994cd5b2c38d81e22"),
    "frozen_head": ("src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py", "96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7"),
    "full_validation": (str((FULL / "validation.json").relative_to(ROOT)), "4f6c514b7e7d619d32cf312e06ee9a39494eb7ef4113aaebc45e10b38ce1c827"),
    "role_manifest": (str((ROLE_ROOT / "role_manifest.json").relative_to(ROOT)), "2f104f4fbf71bada1b6186fa3d0915fa7f8059c65d6798414ab00043e5835454"),
    "role_validation": (str((ROLE_ROOT / "independent_validation.json").relative_to(ROOT)), "ae735624176e5e400ff5c16874b7f73b0505ce71f6814d0c4ed515d94455f8ac"),
    "old_result": ("results/routea_matched_three_arm_common3_native7_crossfit_v1/result.json", "591b9787403417efa774e7bdb264291f7e8e4a827bfd4f479256d4e7a8e134c3"),
    "old_validation": ("results/routea_matched_three_arm_common3_native7_crossfit_v1/independent_validation.json", "1a7b08edfad7823db62f86c5f0b213106daead6f2ddba2d299102b69e546d8df"),
    "identity_manifest": ("registry/gallery_identity_repair_v1.json", "9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f"),
    "head_parameter_seal": (str(HEAD.relative_to(ROOT)), "42c8e503874cb807ca7085a39c802c016de2b1b8c6124bb17a56ed6f06bd174b"),
    "split_qualification": ("registry/rc_shared_query_target_prior_split_qualification_v1_20260909.json", "4f1e242170681e2509f1ed70c0b3a0dacbd0df3bd575f3fe624f39b04d74f642"),
    "split_validator": ("programs/validate_rc_shared_query_target_prior_split_v1.py", "cd3c76fe25986bc2419211d4690d851ee80e264e89ad3ef47c2c51f1e2084045"),
}
CONTRACT = {
    "primary_head": "SEALED_EXISTING_NATIVE7_C_PAIRED", "head_training": False,
    "new_method_count": 1, "trainable_parameter_count": 133,
    "prior": "exp([normalize_FP64(original_q_token128),x,y,x*x,y*y,x*y]@theta-max_query_score)",
    "coordinates": "ROW_MAJOR_NORMALIZED_CELL_CENTERS_MINUS1_PLUS1", "theta_initialization": "ALL_ZERO_FP64",
    "query_prior_candidate_input": False, "effective_query_visibility": "alpha(q,p)*original_wq(q,g,p)",
    "reference_identity": "UNCHANGED_FULL_REFERENCE_WR_WEIGHTED_MAXSIM",
    "Q_control": "ROLL_EFFECTIVE_ALPHA_TIMES_WQ_BY_ORIGINAL_COHORT_SHIFT",
    "R_control": "ORIGINAL_WR_ROLL_WITH_ACTUAL_ROLLED_MEAN_AND_FULL_REFERENCE_MAXSIM",
    "prior_control": "HALF_ROLL_ALPHA_ONLY_THEN_RECOMPUTE_ALL_FOUR_SCALARS_AND_ACTION",
    "CBIND": "FULL_CANDIDATE_EVIDENCE_BUNDLE_SHIFT64_WITH_RAW_AXIS_FIXED",
    "updates": 2000, "seed": 17, "optimizer": "AdamW", "lr": .03, "weight_decay": .001,
    "betas": [.9, .999], "eps": 1e-8, "amsgrad": False, "foreach": None, "fused": None,
    "train_order": "PAIR64_ORIGINAL_CACHE_MANIFEST_ORDER_THEN_TRAIN32_EXECUTION_ASCENDING_FULL_BATCH",
    "pair_loss": "MEAN_BCE_WITH_LOGITS_LABEL0_WEIGHT4_LABEL1_WEIGHT1",
    "query_loss": "BASE_CORRECT:4softplus(max127);BASE_WRONG:softplus(-target)+4softplus(max_other126)",
    "total_loss": "PAIR_LOSS_PLUS_MEAN_32_QUERY_LOSSES", "cpu_threads": 8,
    "action_arithmetic": "ORIGINAL_FEATURE_MATRIX_MATMUL_FIXED_WEIGHT_PLUS_BIAS",
    "baseline": "ALL128_EPISODES_LITERAL_AND_BATCHED_SCALARS_AND_NATIVE_FEATURES_AND_LOGITS_BIT_EXACT_OR_ABORT",
    "eval_modes": list(MODES), "eval_queries": 32, "candidate_count": 128,
    "postjoin": "ALL_PARAMETERS_AND_ALL_EVAL_MODE_LOGITS_SEALED_BEFORE_EVAL_ROLES_OR_OLD_OUTCOME_SEMANTIC_READ",
    "checkpoint_selection": False, "early_stopping": False, "hyperparameter_search": False,
    "validator": "REPLAY_ALL_FROZEN_OUTPUTS_ALL_ADAMW_TRANSITIONS_AND_ENDPOINT_SOURCE_LOSS_GRADIENTS_NO_RETRAIN",
    "validator_evaluation_oracle": "ALL_FINAL_THETA_MODES_1D_LITERAL_SCALARS_OLD_PUBLIC_FEATURES_OLD_MATMUL",
    "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None, "formal_panel_authorized": False,
    "deployment_replacement_authorized": False, "automatic_stage_advance": False,
}
BARRIER = None


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def safe(path):
    path = Path(path).resolve()
    need(path.is_relative_to(ROOT), "SOURCE_OUTSIDE_ROOT")
    need(not any(x in str(path).lower() for x in ("d1_mi", "d1-mi", "d1_minimal_intervention", "grozi", "gisc_prerecall_universe")), "PROTECTED_SOURCE")
    return path


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    path = safe(path)
    digest = hashlib.sha256()
    if BARRIER is not None:
        BARRIER.hash_only += 1
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    finally:
        if BARRIER is not None:
            BARRIER.hash_only -= 1
    return digest.hexdigest()


def read(path):
    return json.loads(safe(path).read_text())


def binding(path):
    return {"path": str(safe(path).relative_to(ROOT)), "sha256": sha(path)}


def checked(item):
    path = safe(ROOT / item["path"])
    need(sha(path) == item["sha256"], "SOURCE_HASH_DRIFT:" + item["path"])
    return path


def atomic(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encode(value) + b"\n"
    if path.exists():
        need(path.read_bytes() == data, "IMMUTABLE_REPLAY_DRIFT:" + str(path))
        return
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def hx(value):
    return float(value).hex()


def tensor_sha(value):
    import torch
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode() + encode(list(value.shape))
                          + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def exact_tensor(a, b, message):
    left, right = tensor_sha(a), tensor_sha(b)
    if left != right:
        details = {"left_sha256": left, "right_sha256": right,
                   "left_shape": list(a.shape), "right_shape": list(b.shape)}
        if a.dtype == b.dtype and a.shape == b.shape:
            import torch
            av, bv = a.detach().contiguous().flatten(), b.detach().contiguous().flatten()
            differences = av.view(torch.uint8).reshape(av.numel(), av.element_size()).ne(
                bv.view(torch.uint8).reshape(bv.numel(), bv.element_size())).any(dim=1)
            indices = differences.nonzero().flatten()[:3].tolist()
            details["different_values"] = int(differences.sum())
            details["first_differences"] = [{"flat_index": i, "left": hx(av[i]), "right": hx(bv[i])} for i in indices]
        raise RuntimeError(message + ":" + encode(details).decode())


def save_arrays(path, arrays):
    import numpy as np
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **{k: v.detach().cpu().numpy() for k, v in arrays.items()})
        stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)
    return binding(path)


class ReadBarrier:
    def __init__(self):
        self.train_paths = set(); self.eval_paths = set()
        self.outcome_paths = {str((ROOT / PINS[k][0]).resolve()) for k in ("old_result", "old_validation")}
        self.hash_only = 0; self.released = False; self.blocked = 0

    def hook(self, event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = str(Path(os.fsdecode(args[0])).resolve())
        if path in self.outcome_paths and not self.released and not self.hash_only:
            self.blocked += 1
            raise RuntimeError("OLD_OUTCOME_READ_BEFORE_EVAL_SEAL")
        if str(ROLE_ROOT / "role_shards") in path and path not in self.train_paths:
            if not self.released or path not in self.eval_paths:
                self.blocked += 1
                raise RuntimeError("EVAL_OR_UNRELATED_ROLE_READ_BEFORE_SEAL")


def install_barrier():
    global BARRIER
    need(BARRIER is None, "BARRIER_ALREADY_INSTALLED")
    BARRIER = ReadBarrier()
    sys.addaudithook(BARRIER.hook)


def modules():
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    core = importlib.import_module("rc_aslo_xf.shared_query_target_prior_v1")
    frozen = importlib.import_module("rc_aslo_xf.romav2_colnomic_frozen_gate_v1")
    source = ROOT / PINS["old_runner"][0]
    need(sha(source) == PINS["old_runner"][1], "OLD_PURE_SOURCE_PIN")
    names = {"finite_tensor", "finite_scalar", "parameter_sha", "actions", "summary", "retention"}
    selected = [node for node in ast.parse(source.read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    need({node.name for node in selected} == names, "OLD_PURE_FUNCTIONS_MISSING")
    namespace = {"torch": torch, "math": math, "hashlib": hashlib, "json": json, "CrossfitContractError": RuntimeError,
                 "FAMILIES": {"NATIVE7": ("real_native_features", "cbind_native_features", frozen.FEATURE_NAMES)}}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])), str(source), "exec"), namespace)
    return core, frozen, namespace


def source_pins():
    sources = {name: {"path": path, "sha256": digest} for name, (path, digest) in PINS.items()}
    for item in sources.values():
        checked(item)
    validate_split_qualification(sources)
    for name, path in (("program", Path(__file__)), ("core", ROOT / "src/rc_aslo_xf/shared_query_target_prior_v1.py"),
                       ("core_tests", ROOT / "tests/test_shared_query_target_prior_v1.py"), ("plan", PLAN), ("launcher", LAUNCH),
                       ("cache_manifest", CACHE / "manifest.json"), ("cache_validation", CACHE / "validation.json"),
                       ("cache_authority", ROOT / "registry/rc_shared_query_target_prior_cache_authority_v1_20260909.json"),
                       ("head_parameter_seal", HEAD), ("identity_core", ROOT / "src/rc_aslo_xf/gallery_identity_repair.py"),
                       ("identity_contract", ROOT / "protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json")):
        sources[name] = binding(path)
    return sources


def validate_split_qualification(sources):
    value = read(checked(sources["split_qualification"]))
    need(value["status"] == "RC_SHARED_QUERY_TARGET_PRIOR_SPLIT_QUALIFICATION_V1_PASS"
         and value["PAIR_actual_supervision_target_role_identity_matches"] == 64
         and value["PAIR_actual_supervision_target_role_identity_mismatches"] == 0
         and value["episodes_dropped_or_reassigned"] == 0
         and value["evaluation_identity_or_target_details_exported"] == 0
         and value["role_shards_physically_and_logically_verified"] == 128, "SPLIT_QUALIFICATION_NOT_CLOSED")
    expected_pairs = {"FULL_TRAIN32__FULL_EVAL32", "PAIR64__FULL_EVAL32", "PAIR64__FULL_TRAIN32"}
    need(set(value["overlap_counts"]) == expected_pairs and all(
        set(counts) == {"execution_ordinal", "identity", "query_id", "supergroup"}
        and all(count == 0 for count in counts.values()) for counts in value["overlap_counts"].values()), "IDENTITY_OR_GROUP_SPLIT_OVERLAP")
    need(value["sources"]["validator"] == sources["split_validator"]
         and value["sources"]["role_manifest"] == sources["role_manifest"], "SPLIT_SOURCE_BINDING")
    return value


def load_head(sources, core, pure):
    import torch
    seal = read(checked(sources["head_parameter_seal"]))
    need(set(seal) == {"status", "family", "arm", "weight_binary64", "bias_binary64", "parameter_sha256",
                       "source_result_sha256", "source_result_path", "trainable_in_query_prior_branch"}
         and seal["family"] == "NATIVE7" and seal["arm"] == "C_PAIRED"
         and seal["status"] == "FROZEN_NATIVE7_C_PARAMETERS_FOR_QUERY_PRIOR_ONLY"
         and seal["trainable_in_query_prior_branch"] is False, "PRIMARY_HEAD_IDENTITY")
    need(seal["source_result_sha256"] == PINS["old_result"][1], "PRIMARY_HEAD_RESULT_PIN")
    weight = torch.tensor([float.fromhex(x) for x in seal["weight_binary64"]], dtype=torch.float64)
    bias = float.fromhex(seal["bias_binary64"])
    need(pure["parameter_sha"](weight, bias) == seal["parameter_sha256"], "PRIMARY_HEAD_PARAMETER_SHA")
    head = core.FixedActionHead(weight, bias)
    need(not list(head.parameters()), "HEAD_IS_TRAINABLE")
    return head, seal


def native_logits(features, head):
    # Match old deployed actions(): do not replace GEMV by elementwise sum.
    return features @ head.weight + float(head.bias)


def make_episode(meta, arrays, full_sources, core):
    import torch
    positions = [int(x) for x in arrays["candidate_positions"]]
    q = torch.from_numpy(arrays["q_tokens"].copy())
    count = len(q)
    sources = {}
    for index, position in enumerate(positions):
        def scalar(key):
            return torch.tensor(float(arrays[key][index]), dtype=torch.float64)
        sources[position] = core.CandidateEvidenceSource(
            torch.from_numpy(arrays["b"][index].copy()), torch.from_numpy(arrays["br"][index].copy()),
            torch.from_numpy(arrays["wq"][index].copy()), scalar("wr_mean"), scalar("wr_rolled_mean"),
            torch.arange(count, dtype=torch.int64), int(arrays["qshift"][index]))
    packed = core.pack_candidate_sources(sources)
    raw = [float(x) for x in arrays["raw_scores"]]
    winner = int(meta["winner"])
    need(len(raw) == len(meta["axis"]) == 128 and winner in positions, "RAW_OR_SOURCE_AXIS")
    need(winner == max(range(128), key=lambda i: (raw[i], -meta["axis"][i])), "ORIGINAL_RAW_WINNER_DRIFT")
    gaps = torch.stack([core.standardized_raw_gap(raw, p, winner) for p in positions])
    if meta["kind"] == "FULL":
        old = full_sources[int(meta["execution_ordinal"])]
        need(old["query_id"] == meta["query_id"] and list(old["candidate_physical_rows"]) == meta["axis"], "FULL_CACHE_LINEAGE")
        exact_tensor(torch.tensor(raw, dtype=torch.float64), old["base_scores"].to(torch.float64), "FULL_RAW_SCORE_DRIFT")
        challengers = [int(x) for x in old["challenger_positions"]]
        expected_features = old["real_native_features"]["C_PAIRED"]
        need(positions == list(range(128)) and list(old["cbind_source_positions"]) == [(i + 64) % 128 for i in range(128)], "FULL_CANDIDATE_OR_CBIND_AXIS")
        action_row = old
    else:
        challengers = [int(x) for x in meta["challenger_positions"]]
        expected_features = torch.tensor(meta["old_native_features"]["C_PAIRED"], dtype=torch.float64)
        need(len(positions) == 2 and len(challengers) == 1 and set(positions) == {winner, challengers[0]}, "PAIR2_AXES")
        action_row = None
    return {"meta": meta, "query_tokens": q, "features": core.query_features(q, meta["q_grid_shape"]),
            "sources": sources, "packed": packed, "raw": raw, "raw_gaps": gaps,
            "winner": winner, "winner_index": positions.index(winner), "positions": positions,
            "challengers": challengers, "challenger_indices": torch.tensor([positions.index(p) for p in challengers]),
            "expected_features": expected_features, "action_row": action_row}


def prepare(sources, core, frozen, head):
    import numpy as np
    import torch
    manifest = validate_cache_qualification(sources)
    split = validate_split_qualification(sources)
    need(split["sources"]["full_manifest"] == manifest["sources"]["full_manifest"]
         and split["sources"]["pair_payload"] == manifest["sources"]["pair_payload"], "SPLIT_CACHE_POPULATION_BINDING")
    full_validation = read(checked(sources["full_validation"]))
    need(full_validation["status"] == "ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURES_VALIDATED"
         and all(value is True for value in full_validation["checks"].values()), "ORIGINAL_FULL_SOURCE_INVALID")
    full_sources = {}
    for item in full_validation["shards"]:
        path = FULL / f"shard{int(item['shard']):02d}/payload.pt"
        need(sha(path) == item["payload_sha256"], "FULL_PAYLOAD_SHA")
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        for row in payload["records"]:
            need("target_position" not in row and "target_identity" not in row, "FULL_INPUT_CONTAINS_TARGET")
            need(int(row["execution_ordinal"]) not in full_sources, "DUPLICATE_FULL_EXECUTION")
            full_sources[int(row["execution_ordinal"])] = row
    rolem = read(checked(sources["role_manifest"]))
    entries = {int(entry["execution_ordinal"]): entry for entry in rolem["shards"]}
    for entry in manifest["records"]:
        if entry["kind"] == "FULL":
            path = str(safe(entries[int(entry["execution_ordinal"])]["path"]))
            (BARRIER.train_paths if entry["role"] == "TRAIN" else BARRIER.eval_paths).add(path)
    need(len(BARRIER.train_paths) == len(BARRIER.eval_paths) == 32, "TRAIN_EVAL_ROLE_PATH_AXIS")
    episodes = []
    for entry in manifest["records"]:
        meta = read(checked(entry))
        need(meta["episode_key"] == entry["episode_key"] and meta["kind"] == entry["kind"]
             and meta["role"] == entry["role"] and meta["query_id"] == entry["query_id"], "CACHE_EPISODE_BINDING")
        with np.load(checked(meta["arrays"]), allow_pickle=False) as bundle:
            arrays = {key: bundle[key].copy() for key in bundle.files}
        for key, value in arrays.items():
            need(list(value.shape) == meta["array_schema"][key]["shape"] and str(value.dtype) == meta["array_schema"][key]["dtype"], "CACHE_ARRAY_SCHEMA")
        episodes.append(make_episode(meta, arrays, full_sources, core))
    pair = [row for row in episodes if row["meta"]["kind"] == "PAIR"]
    train = sorted((row for row in episodes if row["meta"]["kind"] == "FULL" and row["meta"]["role"] == "TRAIN"), key=lambda row: row["meta"]["execution_ordinal"])
    evals = sorted((row for row in episodes if row["meta"]["kind"] == "FULL" and row["meta"]["role"] == "EVAL"), key=lambda row: row["meta"]["execution_ordinal"])
    need((len(pair), len(train), len(evals)) == (64, 32, 32), "EPISODE_POPULATIONS")
    for field in ("query_id", "execution_ordinal", "query_source_image_sha256"):
        need(not ({r["meta"][field] for r in pair + train} & {r["meta"][field] for r in evals}), "TRAIN_EVAL_QUERY_LEAKAGE:" + field)
    baseline = baseline_check(episodes, core, frozen, head)
    closure = {"status": "SHARED_QUERY_PRIOR_INPUTS_AND_UNIFORM_BASELINE_EXACT", "cache_manifest_sha256": sources["cache_manifest"]["sha256"],
               "split_qualification_sha256": sources["split_qualification"]["sha256"],
               "pair_order": [row["meta"]["episode_key"] for row in pair],
               "train_order": [row["meta"]["episode_key"] for row in train], "eval_order": [row["meta"]["episode_key"] for row in evals],
               "baseline": baseline, "EVAL_role_reads": 0, "old_outcome_semantic_reads": 0,
               "all_query_source_and_image_disjoint": True, "thread_count": torch.get_num_threads()}
    return pair, train, evals, entries, closure


def validate_cache_qualification(sources):
    manifest = read(checked(sources["cache_manifest"]))
    validation = read(checked(sources["cache_validation"]))
    need(manifest["status"] == "RC_SHARED_QUERY_TARGET_PRIOR_CACHE_V1_COMPLETE"
         and validation["status"] == "RC_SHARED_QUERY_TARGET_PRIOR_CACHE_V1_SOURCE_REPLAY_VALIDATION_PASS"
         and validation["manifest_sha256"] == sources["cache_manifest"]["sha256"]
         and validation["checks"] and all(value is True for value in validation["checks"].values()), "CACHE_NOT_QUALIFIED")
    need(manifest["authority_sha256"] == validation["authority_sha256"] == sources["cache_authority"]["sha256"]
         and manifest["counts"]["candidate_occurrences"] == 8320 and len(manifest["records"]) == 128,
         "CACHE_POPULATION_OR_AUTHORITY")
    need(validation["counts"]["episodes"] == 128 and validation["counts"]["candidate_occurrences"] == 8320
         and validation["counts"]["raw_local_vectors_bit_exact"] == 24960
         and validation["counts"]["original_scalar_hex_checks"] == 33280, "CACHE_SOURCE_REPLAY_COUNTS")
    return manifest


def episode_outputs(row, prior, core, head, mode="REAL"):
    import torch
    need(mode in MODES, "UNKNOWN_MODE")
    alpha = torch.ones(row["features"].shape[0], dtype=torch.float64) if mode == "UNIFORM" else prior.from_features(row["features"])
    if mode == "PRIOR_HALF_ROLL":
        alpha = alpha.roll(max(1, alpha.numel() // 2))
    source = row["packed"]
    if mode == "CBIND":
        need(row["meta"]["kind"] == "FULL" and len(source.candidate_keys) == 128, "CBIND_REQUIRES_FULL128")
        source = replace(source, **{name: getattr(source, name).roll(64, dims=0) for name in
            ("local_values", "reference_control_local_values", "query_visibility", "reference_mean", "reference_control_mean")})
    scores = core.score_candidates_batched(alpha, source, validate_source=False)
    all_features = core.frozen_action_features_batched(row["raw_gaps"], scores, row["winner_index"])
    features = all_features[row["challenger_indices"]]
    logits = native_logits(features, head)
    need(bool(torch.isfinite(logits.detach()).all()), "NONFINITE_LOGITS")
    return {"alpha": alpha, "scalars": torch.stack([getattr(scores, field) for field in core.SCORE_FIELDS], dim=1),
            "features": features, "logits": logits}


def literal_episode_outputs(row, prior, core, frozen, head, mode):
    """Independent final-theta readout: 1-D source scoring and old public V.

    This is intentionally distinct from the batched producer.  Query features
    are rebuilt from the original query tokens.  Each control remaps only its
    declared input, then every candidate is scored with the literal 1-D path;
    all six pair features use the original public scalar function and logits
    use the original action's matrix multiplication.
    """
    import torch
    need(mode in MODES, "UNKNOWN_LITERAL_MODE")
    features = core.query_features(row["query_tokens"], row["meta"]["q_grid_shape"])
    exact_tensor(features, row["features"], "FINAL_QUERY_FEATURE_SOURCE_REPLAY")
    alpha = torch.ones(len(row["query_tokens"]), dtype=torch.float64) if mode == "UNIFORM" else prior.from_features(features)
    if mode == "PRIOR_HALF_ROLL":
        alpha = alpha.roll(max(1, alpha.numel() // 2))
    if mode == "CBIND":
        need(row["positions"] == list(range(128)), "LITERAL_CBIND_FULL128_AXIS")
        sources = {destination: row["sources"][(destination + 64) % 128] for destination in row["positions"]}
    else:
        sources = row["sources"]
    literal = {position: core.score_candidate(alpha, sources[position]) for position in row["positions"]}
    plain = {position: result.as_old_feature_record() for position, result in literal.items()}
    native = torch.stack([frozen.candidate_feature(row["raw"], plain, challenger, row["winner"])
                          for challenger in row["challengers"]])
    scalars = torch.stack([torch.stack([getattr(literal[position], field) for field in core.SCORE_FIELDS])
                           for position in row["positions"]])
    return {"alpha": alpha, "scalars": scalars, "features": native, "logits": native_logits(native, head)}


def baseline_check(episodes, core, frozen, head):
    import torch
    prior = core.SharedQueryTargetPrior()
    checks = {"episode_count": 0, "candidate_occurrences": 0, "scalar_hex_checks": 0, "feature_value_checks": 0,
              "logit_value_checks": 0, "literal_to_batch_bit_differences": 0}
    ledgers = []
    with torch.no_grad():
        for row in episodes:
            qcount = len(row["query_tokens"])
            alpha = prior.from_features(row["features"])
            exact_tensor(alpha, torch.ones(qcount, dtype=torch.float64), "UNIFORM_ALPHA_NOT_EXACT_ONE")
            literal = {key: core.score_candidate(alpha, source) for key, source in row["sources"].items()}
            out = episode_outputs(row, prior, core, head, "REAL")
            for index, position in enumerate(row["positions"]):
                expected = row["meta"]["candidates"][index]
                need(expected["candidate_position"] == position, "CACHE_POSITION_LEDGER")
                for field_index, field in enumerate(core.SCORE_FIELDS):
                    value = getattr(literal[position], field)
                    need(hx(value) == expected["old_scalars_binary64"][field],
                         "LITERAL_BASELINE_SCALAR_BIT_DRIFT:" + row["meta"]["episode_key"] + ":" + str(position) + ":" + field)
                    exact_tensor(value, out["scalars"][index, field_index],
                                 "BATCH_BASELINE_SCALAR_BIT_DRIFT:" + row["meta"]["episode_key"] + ":" + str(position) + ":" + field)
                    checks["scalar_hex_checks"] += 1
            plain = {key: value.as_old_feature_record() for key, value in literal.items()}
            public = torch.stack([frozen.candidate_feature(row["raw"], plain, challenger, row["winner"]) for challenger in row["challengers"]])
            exact_tensor(public, row["expected_features"], "ORIGINAL_PUBLIC_FEATURE_REPLAY_DRIFT:" + row["meta"]["episode_key"])
            exact_tensor(out["features"], public, "BATCH_FEATURE_REPLAY_DRIFT:" + row["meta"]["episode_key"])
            expected_logits = native_logits(row["expected_features"], head)
            exact_tensor(out["logits"], expected_logits, "BASELINE_NATIVE_LOGIT_DRIFT")
            checks["episode_count"] += 1; checks["candidate_occurrences"] += len(row["positions"])
            checks["feature_value_checks"] += public.numel(); checks["logit_value_checks"] += expected_logits.numel()
            ledgers.append({"episode_key": row["meta"]["episode_key"], "scalars_sha256": tensor_sha(out["scalars"]),
                            "features_sha256": tensor_sha(out["features"]), "logits_sha256": tensor_sha(out["logits"])})
    need(checks["episode_count"] == 128 and checks["candidate_occurrences"] == 8320, "BASELINE_POPULATION")
    return {"checks": checks, "episodes": ledgers, "all_bits_exact": True, "old_result_semantic_reads": 0}


def corrected_labels():
    from rc_aslo_xf.gallery_identity_repair import build_identity_map, PHYSICAL_ROW_COUNT
    gallery = ROOT.parents[2] / "dailymed/data/box_flat_20000_images/data/raw_images"
    paths = []
    for directory, subdirs, names in os.walk(gallery):
        subdirs.sort()
        paths.extend(Path(directory) / name for name in sorted(names) if (Path(directory) / name).is_file()
                     and Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
    need(len(paths) == PHYSICAL_ROW_COUNT, "GALLERY_CATALOGUE_COUNT")
    return build_identity_map([path.stem.strip() for path in paths]).labels


def join_roles(rows, entries, labels):
    joined = []
    for row in rows:
        source = row["action_row"]
        entry = entries[int(row["meta"]["execution_ordinal"])]
        path = safe(entry["path"])
        need(sha(path) == entry["sha256"], "ROLE_SHA")
        role = read(path)
        need(role["query_id"] == row["meta"]["query_id"] and role["track"] == source["track"]
             and role["target_insertion_count"] == role["raw_d1_field_count"] == role["target_spatial_supervision_count"] == 0,
             "ROLE_BINDING")
        targets = [i for i, physical in enumerate(row["meta"]["axis"]) if labels[physical] == role["identity"]]
        need(len(targets) == 1, "TARGET_NOT_UNIQUE_IN_ORIGINAL_RAW128")
        joined.append({**row, "target_position": targets[0], "target_identity": role["identity"], "supergroup": role["supergroup"]})
    return joined


def losses(prior, head, core, pair, train):
    import torch
    from torch.nn import functional as F
    pair_logits = torch.cat([episode_outputs(row, prior, core, head)["logits"] for row in pair])
    labels = torch.tensor([float(row["meta"]["switch_label"]) for row in pair], dtype=torch.float64)
    weights = torch.where(labels.eq(0), torch.full_like(labels, 4.), torch.ones_like(labels))
    pair_loss = (F.binary_cross_entropy_with_logits(pair_logits, labels, reduction="none") * weights).mean()
    query_losses = []
    for row in train:
        logits = episode_outputs(row, prior, core, head)["logits"]
        target = int(row["target_position"])
        if target == row["winner"]:
            query_losses.append(4 * F.softplus(logits.max()))
        else:
            index = row["challengers"].index(target)
            mask = torch.ones(len(row["challengers"]), dtype=torch.bool); mask[index] = False
            query_losses.append(F.softplus(-logits[index]) + 4 * F.softplus(logits[mask].max()))
    full_loss = torch.stack(query_losses).mean()
    values = torch.stack((pair_loss + full_loss, pair_loss, full_loss))
    need(bool(torch.isfinite(values.detach()).all()), "NONFINITE_LOSS")
    return values


def make_optimizer(prior):
    import torch
    return torch.optim.AdamW(prior.parameters(), lr=.03, weight_decay=.001)


def train_prior(core, head, pair, train):
    import torch
    torch.manual_seed(17)
    prior = core.SharedQueryTargetPrior()
    optimizer = make_optimizer(prior)
    states = [prior.theta.detach().clone()]; gradients = []; history = []
    started = time.monotonic()
    for update in range(CONTRACT["updates"]):
        optimizer.zero_grad()
        value = losses(prior, head, core, pair, train)
        value[0].backward()
        need(prior.theta.grad is not None and bool(torch.isfinite(prior.theta.grad).all()), "NONFINITE_PRIOR_GRADIENT")
        gradients.append(prior.theta.grad.detach().clone()); history.append(value.detach().clone())
        optimizer.step()
        need(bool(torch.isfinite(prior.theta.detach()).all()), "NONFINITE_PRIOR_PARAMETER")
        states.append(prior.theta.detach().clone())
        if update == 0 or (update + 1) % 100 == 0:
            print(json.dumps({"event": "SHARED_QUERY_PRIOR_UPDATE", "updates": update + 1,
                              "loss": float(value[0].detach()), "elapsed_seconds": time.monotonic() - started}), flush=True)
    optimizer.zero_grad()
    final_loss = losses(prior, head, core, pair, train)
    final_loss[0].backward()
    need(bool(torch.isfinite(prior.theta.grad).all()), "NONFINITE_FINAL_SOURCE_GRADIENT")
    trace = {"theta_states": torch.stack(states), "gradients": torch.stack(gradients), "losses": torch.stack(history),
             "final_source_loss": final_loss.detach(), "final_source_gradient": prior.theta.grad.detach().clone(),
             "final_exp_avg": optimizer.state[prior.theta]["exp_avg"].detach().clone(),
             "final_exp_avg_sq": optimizer.state[prior.theta]["exp_avg_sq"].detach().clone()}
    prior.zero_grad(set_to_none=True)
    prior.theta.requires_grad_(False)
    return prior, trace


def validate_training_trace(core, head, pair, train, trace, parameters):
    import torch
    steps = CONTRACT["updates"]
    need(trace["theta_states"].shape == (steps + 1, 133) and trace["gradients"].shape == (steps, 133)
         and trace["losses"].shape == (steps, 3), "TRAIN_TRACE_SHAPE")
    need(all(bool(torch.isfinite(value).all()) for value in trace.values()), "TRAIN_TRACE_NONFINITE")
    exact_tensor(trace["theta_states"][0], torch.zeros(133, dtype=torch.float64), "TRAIN_INITIALIZATION")
    prior, optimizer = replay_adamw_transitions(core, trace, steps)
    # Source-grounded endpoint gradients bind the cached trace to this actual
    # loss. Intermediate gradients are checked for finiteness and transitions;
    # they are not independently recomputed, and no full re-training is claimed.
    for index in (0, steps - 1, steps):
        with torch.no_grad(): prior.theta.copy_(trace["theta_states"][index])
        prior.zero_grad(set_to_none=True)
        actual = losses(prior, head, core, pair, train); actual[0].backward()
        expected_loss = trace["final_source_loss"] if index == steps else trace["losses"][index]
        expected_grad = trace["final_source_gradient"] if index == steps else trace["gradients"][index]
        exact_tensor(actual, expected_loss, "ENDPOINT_SOURCE_LOSS_DRIFT")
        exact_tensor(prior.theta.grad, expected_grad, "ENDPOINT_SOURCE_GRADIENT_DRIFT")
    need([hx(x) for x in prior.theta.detach()] == parameters["theta_binary64"], "FINAL_THETA_FILE_DRIFT")
    need([hx(x) for x in trace["losses"][-1]] == parameters["final_preupdate_loss_binary64"]
         and [hx(x) for x in trace["final_source_loss"]] == parameters["final_frozen_source_loss_binary64"], "FINAL_LOSS_PARAMETER_LEDGER_DRIFT")
    prior.zero_grad(set_to_none=True); prior.theta.requires_grad_(False)
    return prior


def replay_adamw_transitions(core, trace, steps):
    """Arithmetic replay using recorded gradients, not an optimization rerun."""
    prior = core.SharedQueryTargetPrior(); optimizer = make_optimizer(prior)
    for update in range(steps):
        exact_tensor(prior.theta, trace["theta_states"][update], "ADAMW_PRESTEP_STATE_DRIFT")
        prior.theta.grad = trace["gradients"][update].clone()
        optimizer.step()
        exact_tensor(prior.theta, trace["theta_states"][update + 1], "ADAMW_POSTSTEP_STATE_DRIFT")
    exact_tensor(optimizer.state[prior.theta]["exp_avg"], trace["final_exp_avg"], "FINAL_ADAMW_FIRST_MOMENT")
    exact_tensor(optimizer.state[prior.theta]["exp_avg_sq"], trace["final_exp_avg_sq"], "FINAL_ADAMW_SECOND_MOMENT")
    return prior, optimizer


def predict(row, logits):
    challengers = row["challengers"]; axis = row["meta"]["axis"]
    index = max(range(len(challengers)), key=lambda i: (float(logits[i]), -axis[challengers[i]]))
    best = challengers[index]; switch = float(logits[index]) > 0.
    return {"base_winner_position": row["winner"], "proposed_challenger_position": best,
            "final_position": best if switch else row["winner"], "decision": "SWITCH" if switch else "HOLD",
            "switch_logit_binary64": hx(logits[index]), "all127_logits_sha256": tensor_sha(logits)}


def eval_prejoin(evals, prior, core, frozen, head, validate):
    import numpy as np
    import torch
    records = []
    with torch.no_grad():
        for row in evals:
            arrays = {}; predictions = {}
            for mode in MODES:
                output = episode_outputs(row, prior, core, head, mode)
                if validate:
                    oracle = literal_episode_outputs(row, prior, core, frozen, head, mode)
                    for field in output:
                        exact_tensor(output[field], oracle[field],
                                     "FINAL_LITERAL_VS_BATCH_REPLAY:" + row["meta"]["episode_key"] + ":" + mode + ":" + field)
                arrays.update({mode + "__" + name: value for name, value in output.items()})
                predictions[mode] = predict(row, output["logits"])
            path = OUT / "eval_arrays" / (row["meta"]["episode_key"] + ".npz")
            if validate:
                with np.load(path, allow_pickle=False) as saved:
                    need(set(saved.files) == set(arrays), "PREJOIN_ARRAY_FIELDS")
                    for name, value in arrays.items():
                        exact_tensor(value, torch.from_numpy(saved[name].copy()), "FROZEN_EVAL_OUTPUT_REPLAY:" + name)
                array_binding = binding(path)
            else:
                array_binding = save_arrays(path, arrays)
            records.append({"execution_ordinal": row["meta"]["execution_ordinal"], "query_id": row["meta"]["query_id"],
                            "candidate_physical_rows": row["meta"]["axis"], "arrays": array_binding,
                            "predictions": predictions, "EVAL_target_reads": 0})
    return records


def postjoin(sources, evals, train, entries, labels, prejoin, head, pure):
    import numpy as np
    import torch
    seal = read(OUT / "eval_prejoin_seal.json")
    need(seal["parameters_sha256"] == sha(OUT / "parameters.json")
         and seal["eval_prejoin_sha256"] == sha(OUT / "eval_prejoin.json")
         and seal["eval_query_count"] == 32 and BARRIER.blocked == 0, "POSTJOIN_WITHOUT_COMPLETE_SEAL")
    BARRIER.released = True
    joined = join_roles(evals, entries, labels)
    disjoint = {field: not bool({r[field] for r in train} & {r[field] for r in joined}) for field in ("target_identity", "supergroup")}
    need(all(disjoint.values()), "TRAIN_EVAL_IDENTITY_OR_GROUP_OVERLAP")
    old = read(checked(sources["old_result"])); old_validation = read(checked(sources["old_validation"]))
    need(old_validation["status"] == "ROUTEA_MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_INDEPENDENT_VALIDATION_PASS"
         and old_validation["producer_result_sha256"] == PINS["old_result"][1]
         and all(value is True for value in old_validation["checks"].values()), "OLD_RESULT_NOT_VALIDATED")
    old_head = old["heads"]["NATIVE7"]["C_PAIRED"]
    need([hx(x) for x in head.weight] == [hx(x) for x in old_head["weight"]]
         and hx(head.bias) == hx(old_head["bias"]), "NATIVE7_C_HEAD_POSTJOIN_REPLAY")
    by_ex = {record["execution_ordinal"]: record for record in prejoin}
    actions = {}
    for mode in MODES:
        rows = []
        for row in joined:
            saved = by_ex[row["meta"]["execution_ordinal"]]
            with np.load(checked(saved["arrays"]), allow_pickle=False) as arrays:
                features = torch.from_numpy(arrays[mode + "__features"].copy())
            rows.append({**row["action_row"], "target_position": row["target_position"],
                         "real_native_features": {"C_PAIRED": features}})
        actions[mode] = pure["actions"](head.weight, float(head.bias), rows, "NATIVE7", "C_PAIRED")
        for action in actions[mode]:
            pred = by_ex[action["execution_ordinal"]]["predictions"][mode]
            need(pred["final_position"] == action["final_position"] and pred["decision"] == action["decision"]
                 and pred["switch_logit_binary64"] == hx(action["switch_logit"]), "POSTJOIN_TARGET_FREE_ACTION_DRIFT")
    baseline_pass = encode(actions["UNIFORM"]) == encode(old["evaluations"]["NATIVE7"]["C_PAIRED"]["actions"])
    need(baseline_pass, "NATIVE7_C_UNIFORM_ACTION_REGRESSION_ABORT")
    comparisons = {}
    real = {row["execution_ordinal"]: row for row in actions["REAL"]}
    for mode in MODES[1:]:
        rescue = sum(real[row["execution_ordinal"]]["final_correct"] and not row["final_correct"] for row in actions[mode])
        breaks = sum(row["final_correct"] and not real[row["execution_ordinal"]]["final_correct"] for row in actions[mode])
        comparisons[mode] = {"rescue": rescue, "break": breaks, "paired_net": rescue - breaks}
    return {"status": "RC_SHARED_QUERY_TARGET_PRIOR_DEVELOPMENT_COMPLETE", "claim_level": "OPENED_INTERNAL_EVAL32_FIXED_NATIVE7_C_PLUS_SHARED_QUERY_PRIOR",
            "authority_sha256": sha(AUTH), "input_closure_sha256": sha(OUT / "input_closure.json"),
            "parameters_sha256": sha(OUT / "parameters.json"), "eval_prejoin_seal_sha256": sha(OUT / "eval_prejoin_seal.json"),
            "baseline_regression_pass": True, "primary_head": "NATIVE7/C_PAIRED", "metrics": {mode: pure["summary"](rows) for mode, rows in actions.items()},
            "REAL_paired_comparisons": comparisons, "CBIND_rescue_retention": pure["retention"](actions["REAL"], actions["CBIND"]),
            "actions": actions, "train_eval_disjoint": disjoint, "training_updates": 2000, "trainable_parameter_count": 133,
            "candidate_count": 128, "PAIR_training_queries": 64, "FULL_training_queries": 32, "EVAL_queries": 32,
            "old_head_parameters_updated": 0, "new_encoder_or_RoMa_forwards": 0, "early_blocked_read_count": BARRIER.blocked,
            "EVAL_targets_read_only_after_all_parameters_and_all_mode_logits_sealed": True,
            "limits": ["Previously opened internal EVAL32; no untouched generalization claim.",
                       "Prior is a shared soft query field, not a certified connected hypothesis or pixel intervention.",
                       "No formal HYP or P-only gate is introduced or weakened by this development result."],
            "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None, "formal_panel_consumed": False,
            "deployment_replacement_authorized": False, "automatic_stage_advance": False}


def synthetic_e0():
    import torch
    core, frozen, pure = modules()
    rng = torch.Generator().manual_seed(17)
    sources = {i: core.CandidateEvidenceSource(torch.rand(15, generator=rng, dtype=torch.float64),
               torch.rand(15, generator=rng, dtype=torch.float64), torch.rand(15, generator=rng, dtype=torch.float64),
               torch.tensor(.5 + i / 512, dtype=torch.float64), torch.tensor(.5 + i / 512, dtype=torch.float64),
               torch.arange(15), 7) for i in range(128)}
    q = torch.randn((15, 128), generator=rng, dtype=torch.float32)
    raw = [float(128 - i) for i in range(128)]
    row = {"meta": {"kind": "FULL", "axis": list(range(128)), "q_grid_shape": (3, 5)}, "query_tokens": q,
           "features": core.query_features(q, (3, 5)), "sources": sources, "packed": core.pack_candidate_sources(sources),
           "raw": raw, "raw_gaps": torch.stack([core.standardized_raw_gap(raw, i, 0) for i in range(128)]),
           "winner": 0, "winner_index": 0, "positions": list(range(128)), "challengers": list(range(1, 128)),
           "challenger_indices": torch.arange(1, 128)}
    prior, head = core.SharedQueryTargetPrior(), core.FixedActionHead()
    real = episode_outputs(row, prior, core, head)
    uniform = episode_outputs(row, prior, core, head, "UNIFORM")
    for key in real:
        exact_tensor(real[key], uniform[key], "SYNTHETIC_UNIFORM_REPLAY")
    with torch.no_grad(): prior.theta[128] = .75
    control = episode_outputs(row, prior, core, head, "PRIOR_HALF_ROLL")
    exact_tensor(control["alpha"], prior.from_features(row["features"]).roll(7), "PRIOR_CONTROL_ALPHA")
    cbind = episode_outputs(row, prior, core, head, "CBIND")
    real = episode_outputs(row, prior, core, head)
    exact_tensor(cbind["scalars"], real["scalars"].roll(64, dims=0), "COMPLETE_CANDIDATE_BUNDLE_CONTROL")
    for mode in MODES:
        batch = episode_outputs(row, prior, core, head, mode)
        oracle = literal_episode_outputs(row, prior, core, frozen, head, mode)
        for field in batch:
            exact_tensor(batch[field], oracle[field], "SYNTHETIC_NONZERO_THETA_LITERAL_CONTROL_REPLAY")
    # Exercise the actual PAIR weighted BCE plus both FULL query-loss cases.
    pair_source = {i: sources[i] for i in (0, 1)}
    pair_row = {**row, "meta": {"kind": "PAIR", "axis": list(range(128)), "switch_label": False},
                "sources": pair_source, "packed": core.pack_candidate_sources(pair_source),
                "positions": [0, 1], "raw_gaps": row["raw_gaps"][:2], "challengers": [1],
                "challenger_indices": torch.tensor([1])}
    pair = [pair_row, {**pair_row, "meta": {**pair_row["meta"], "switch_label": True}}]
    train = [{**row, "target_position": 0}, {**row, "target_position": 1}]
    from torch.nn import functional as F
    pair_values = torch.cat([episode_outputs(r, prior, core, head)["logits"] for r in pair])
    y = torch.tensor([0., 1.], dtype=torch.float64)
    want_pair = (F.binary_cross_entropy_with_logits(pair_values, y, reduction="none") * torch.tensor([4., 1.], dtype=torch.float64)).mean()
    values = episode_outputs(row, prior, core, head)["logits"]
    want_full = torch.stack([4 * F.softplus(values.max()), F.softplus(-values[0]) + 4 * F.softplus(values[1:].max())]).mean()
    exact_tensor(losses(prior, head, core, pair, train), torch.stack((want_pair + want_full, want_pair, want_full)), "ORIGINAL_LOSS_FORMULA")
    # A two-step synthetic trace exercises the same saved-gradient AdamW
    # transition verifier used for the production 2000-step trace.
    with torch.no_grad(): prior.theta.zero_()
    optimizer = make_optimizer(prior)
    states = [prior.theta.detach().clone()]; grads = []
    for _ in range(2):
        optimizer.zero_grad()
        value = losses(prior, head, core, pair, train)[0]
        value.backward()
        need(bool(torch.isfinite(prior.theta.grad).all()), "SYNTHETIC_GRADIENT")
        grads.append(prior.theta.grad.detach().clone())
        optimizer.step()
        states.append(prior.theta.detach().clone())
    trace = {"theta_states": torch.stack(states), "gradients": torch.stack(grads),
             "final_exp_avg": optimizer.state[prior.theta]["exp_avg"].detach().clone(),
             "final_exp_avg_sq": optimizer.state[prior.theta]["exp_avg_sq"].detach().clone()}
    replayed, _ = replay_adamw_transitions(core, trace, 2)
    exact_tensor(replayed.theta, prior.theta, "SYNTHETIC_ADAMW_TRACE_REPLAY")
    corrupt = {**trace, "gradients": trace["gradients"].clone()}; corrupt["gradients"][0, 0] += 1.
    try: replay_adamw_transitions(core, corrupt, 2)
    except RuntimeError: pass
    else: raise RuntimeError("CORRUPTED_GRADIENT_TRACE_ACCEPTED")
    need(not list(head.parameters()) and sum(p.numel() for p in prior.parameters()) == 133, "ONLY_PRIOR_TRAINABLE")
    barrier = ReadBarrier()
    train_path = str(ROLE_ROOT / "role_shards/role_exec998.json")
    eval_path = str(ROLE_ROOT / "role_shards/role_exec999.json")
    barrier.train_paths.add(train_path); barrier.eval_paths.add(eval_path)
    barrier.hook("open", (train_path, "r"))
    for path in (eval_path, str(ROOT / PINS["old_result"][0])):
        try: barrier.hook("open", (path, "r"))
        except RuntimeError: pass
        else: raise RuntimeError("SYNTHETIC_BARRIER_ACCEPTED_EARLY_READ")
    barrier.hash_only = 1; barrier.hook("open", (str(ROOT / PINS["old_result"][0]), "rb")); barrier.hash_only = 0
    barrier.released = True; barrier.hook("open", (eval_path, "r"))
    return {"status": "RC_SHARED_QUERY_TARGET_PRIOR_RUNNER_E0_PASS", "checks": {
        "uniform_source_computation_replays_without_cached_constants": True, "complete_bundle_CBIND_and_raw_axis_fixed": True,
        "prior_half_roll_only_alpha_then_all_fields_recomputed": True, "only_theta133_trainable_fixed_head_buffers": True,
        "two_synthetic_optimizer_updates_finite": True, "EVAL_and_old_outcome_barriers_fail_closed": True,
        "original_PAIR_weighted_BCE_and_both_FULL_loss_cases_exact": True,
        "recorded_AdamW_transitions_replay_and_corrupt_gradient_rejected": True,
        "nonzero_theta_all_four_modes_literal_scores_old_public_features_old_matmul_replay": True,
        "hash_only_access_separate_from_semantic_read": True, "original_pure_action_and_summary_functions_reused": True,
        "original_matmul_action_arithmetic": True}, "synthetic_only": True, "natural_data_reads": 0,
        "natural_training_updates": 0, "scientific_GO_or_NO_GO": None}


def freeze():
    need(not AUTH.exists() and not OUT.exists(), "APPEND_ONLY_AUTHORITY_OR_OUTPUT_EXISTS")
    sources = source_pins()
    validate_cache_qualification(sources)
    core, _, pure = modules(); load_head(sources, core, pure)
    e0_path = ROOT / "results" / (PREFIX + "_e0") / "result.json"
    atomic(e0_path, synthetic_e0()); sources["e0"] = binding(e0_path)
    value = {"status": "RC_SHARED_QUERY_TARGET_PRIOR_DEVELOPMENT_AUTHORIZED", "sources": sources,
             "contract": CONTRACT, "output_rel": str(OUT.relative_to(ROOT)), "scientific_GO_or_NO_GO": None}
    atomic(AUTH, value)
    print(json.dumps({"status": value["status"], "authority_sha256": sha(AUTH)}), flush=True)


def authority():
    value = read(AUTH)
    need(value["status"] == "RC_SHARED_QUERY_TARGET_PRIOR_DEVELOPMENT_AUTHORIZED" and value["contract"] == CONTRACT
         and value["output_rel"] == str(OUT.relative_to(ROOT)) and value["scientific_GO_or_NO_GO"] is None, "AUTHORITY_SCOPE")
    for item in value["sources"].values(): checked(item)
    need({k: v for k, v in value["sources"].items() if k != "e0"} == source_pins(), "AUTHORITY_SOURCE_DRIFT")
    e0 = read(checked(value["sources"]["e0"]))
    need(e0["status"] == "RC_SHARED_QUERY_TARGET_PRIOR_RUNNER_E0_PASS" and all(v is True for v in e0["checks"].values()), "E0_NOT_CLOSED")
    return value


def execute(validate=False):
    import numpy as np
    import torch
    auth = authority()
    core, frozen, pure = modules(); head, head_seal = load_head(auth["sources"], core, pure)
    pair, unjoined_train, evals, entries, closure = prepare(auth["sources"], core, frozen, head)
    labels = corrected_labels(); train = join_roles(unjoined_train, entries, labels)
    if validate:
        need(OUT.is_dir() and read(OUT / "input_closure.json") == closure, "INPUT_CLOSURE_REPLAY")
        parameters = read(OUT / "parameters.json")
        with np.load(checked(parameters["training_trace"]), allow_pickle=False) as arrays:
            trace = {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}
        prior = validate_training_trace(core, head, pair, train, trace, parameters)
    else:
        need(not OUT.exists(), "APPEND_ONLY_OUTPUT_EXISTS"); OUT.mkdir(parents=True)
        atomic(OUT / "input_closure.json", closure)
        prior, trace = train_prior(core, head, pair, train)
        trace_binding = save_arrays(OUT / "training_trace.npz", trace)
        parameters = {"status": "SHARED_QUERY_PRIOR_FINAL_2000_UPDATES_FROZEN", "theta_binary64": [hx(x) for x in prior.theta],
                      "theta_sha256": tensor_sha(prior.theta), "head_parameter_sha256": head_seal["parameter_sha256"],
                      "head_parameter_seal": auth["sources"]["head_parameter_seal"], "training_trace": trace_binding,
                      "final_preupdate_loss_binary64": [hx(x) for x in trace["losses"][-1]],
                      "final_frozen_source_loss_binary64": [hx(x) for x in trace["final_source_loss"]],
                      "updates": 2000, "trainable_parameter_count": 133, "checkpoint_selection": False,
                      "EVAL_role_reads": 0, "old_outcome_semantic_reads": 0}
        atomic(OUT / "parameters.json", parameters)
    need(parameters["theta_sha256"] == tensor_sha(prior.theta) and parameters["head_parameter_sha256"] == head_seal["parameter_sha256"]
         and parameters["updates"] == 2000 and not prior.theta.requires_grad, "FINAL_PARAMETER_SCOPE")
    prejoin = eval_prejoin(evals, prior, core, frozen, head, validate)
    if validate:
        need(read(OUT / "eval_prejoin.json") == prejoin, "EVAL_PREJOIN_REPLAY")
    else:
        atomic(OUT / "eval_prejoin.json", prejoin)
        atomic(OUT / "eval_prejoin_seal.json", {"status": "SHARED_QUERY_PRIOR_ALL_EVAL_MODES_PREJOIN_SEALED",
            "authority_sha256": sha(AUTH), "parameters_sha256": sha(OUT / "parameters.json"),
            "eval_prejoin_sha256": sha(OUT / "eval_prejoin.json"), "eval_query_count": 32,
            "candidate_count": 128, "modes": list(MODES), "EVAL_role_reads": 0, "old_outcome_semantic_reads": 0})
    result = postjoin(auth["sources"], evals, train, entries, labels, prejoin, head, pure)
    if validate:
        need(read(OUT / "result.json") == result, "RESULT_REPLAY")
        receipt = {"status": "RC_SHARED_QUERY_TARGET_PRIOR_INDEPENDENT_FROZEN_REPLAY_VALIDATION_PASS",
            "authority_sha256": sha(AUTH), "result_sha256": sha(OUT / "result.json"), "checks": {
                "all128_episode_uniform_scalar_feature_logit_baselines_bit_exact": True,
                "all2000_saved_gradient_AdamW_transitions_replayed_exactly": True,
                "initial_last_preupdate_and_final_source_losses_and_gradients_recomputed": True,
                "all_frozen_theta_EVAL_modes_full128_scores_features_logits_replayed": True,
                "final_theta_all_modes_independent_1D_candidate_score_old_public_features_old_matmul_exact": True,
                "all_EVAL_roles_delayed_until_complete_prejoin": True,
                "existing_NATIVE7_C_actions_exactly_preserved_under_uniform": True,
                "all_actions_metrics_and_comparisons_recomputed": True},
            "full_training_repeated": False, "all_intermediate_source_gradients_recomputed": False,
            "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}
        atomic(OUT / "independent_validation.json", receipt)
        print(json.dumps(receipt), flush=True)
    else:
        atomic(OUT / "result.json", result)
        print(json.dumps({"status": result["status"], "metrics": result["metrics"],
                          "REAL_paired_comparisons": result["REAL_paired_comparisons"], "scientific_GO_or_NO_GO": None}), flush=True)


def preflight():
    """Natural alpha=1 engineering check only; no role joins or optimizer."""
    auth = authority()
    core, frozen, pure = modules()
    head, _ = load_head(auth["sources"], core, pure)
    _, _, _, _, closure = prepare(auth["sources"], core, frozen, head)
    receipt = {"status": "RC_SHARED_QUERY_TARGET_PRIOR_NATURAL_UNIFORM_PREFLIGHT_PASS",
               "authority_sha256": sha(AUTH), "input_closure": closure,
               "FULL_TRAIN_role_reads": 0, "EVAL_role_reads": 0, "old_outcome_semantic_reads": 0,
               "training_updates": 0, "blocked_read_count": BARRIER.blocked,
               "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}
    path = ROOT / "results" / (PREFIX + "_preflight") / "result.json"
    atomic(path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt_sha256": sha(path),
                      "baseline_checks": closure["baseline"]["checks"], "training_updates": 0}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("e0", "freeze", "preflight", "run", "validate"), required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(CONTRACT["cpu_threads"]); torch.set_num_interop_threads(1)
    if args.phase == "e0":
        print(json.dumps(synthetic_e0(), sort_keys=True), flush=True)
        return
    install_barrier()
    if args.phase == "freeze": freeze()
    elif args.phase == "preflight": preflight()
    else: execute(validate=args.phase == "validate")


if __name__ == "__main__":
    main()
