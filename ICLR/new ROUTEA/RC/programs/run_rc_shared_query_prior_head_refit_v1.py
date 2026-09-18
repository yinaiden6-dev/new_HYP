#!/usr/bin/env python3
"""Refit only the original seven-parameter action on a frozen query prior."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PREFIX = "rc_shared_query_prior_head_refit_v1"
OUT = ROOT / "results" / PREFIX
AUTH = ROOT / "registry/rc_shared_query_prior_head_refit_authority_v1_20260909.json"
PLAN = ROOT / "plan/RC_SHARED_QUERY_PRIOR_HEAD_REFIT_V1_20260909.md"
LAUNCH = ROOT / "slurm/rc_shared_query_prior_head_refit_v1_dev_cpuonly_59m.sbatch"
PARENT = ROOT / "results/rc_shared_query_target_prior_development_v1"
MODES = ("REAL", "UNIFORM", "PRIOR_HALF_ROLL", "CBIND")
PINS = {
    "prior_runner": ("programs/run_rc_shared_query_target_prior_development_v1.py", "b004d6b3c518773d9f31c65f8e9a40f0d43593fe8e06b39b99bd4f33bb951307"),
    "prior_authority": ("registry/rc_shared_query_target_prior_development_authority_v1_20260909.json", "2b087beece2ebd88f706e3f672a7af39d274e44baf262e1e5bb9b5c43e2a23d9"),
    "prior_result": (str((PARENT / "result.json").relative_to(ROOT)), "977ffe3dcd76deb62bb0dde880ee4e9c5694ec25ab0323bd8acfa9a64e05d2f0"),
    "prior_validation": (str((PARENT / "independent_validation.json").relative_to(ROOT)), "a71bea2cf6af73d1aa4610c8b2e773b05820b124f224a585b04aff58db78d6bd"),
    "prior_parameters": (str((PARENT / "parameters.json").relative_to(ROOT)), "06b927f5d531f20712ec5e4875bad13b8bcd1d82260a31d8609d80719ffd1958"),
    "prior_input_closure": (str((PARENT / "input_closure.json").relative_to(ROOT)), "fe14d954cf6ae26d07ff43c3b61efe3321b87ebfd2f14a905ac7cd2323f59fbd"),
    "prior_eval_prejoin": (str((PARENT / "eval_prejoin.json").relative_to(ROOT)), "11f5fb9fbd172a413e52cc8977fd5a40c50aa93187ea6f01c2febb8ceeb40595"),
    "prior_eval_seal": (str((PARENT / "eval_prejoin_seal.json").relative_to(ROOT)), "050314fca6eba50e7b8b421e2e9979881c48a6960a98946c2b45ded33b9b9279"),
    "split_qualification": ("registry/rc_shared_query_target_prior_split_qualification_v1_20260909.json", "4f1e242170681e2509f1ed70c0b3a0dacbd0df3bd575f3fe624f39b04d74f642"),
    "old_training_source": ("programs/run_routea_matched_three_arm_common3_native7_crossfit_v1.py", "546e2bc7b3df6c67bcac41e079ba1b00f15c7e65c52a8ab994cd5b2c38d81e22"),
}
CONTRACT = {
    "new_head_count": 1, "trainable_parameter_count": 7, "prior_training_updates": 0,
    "prior": "EXACT_FROZEN_PARENT_THETA2000", "feature_count": 6, "feature_family": "NATIVE7", "arm": "C_PAIRED",
    "training_function": "EXACT_AST_ORIGINAL_TRAIN_HEAD", "updates": 2000, "seed": 17,
    "initialization": "FP64_LINEAR_ALL_ZERO", "optimizer": "AdamW", "lr": .03, "weight_decay": .001,
    "betas": [.9, .999], "eps": 1e-8, "amsgrad": False, "foreach": None, "fused": None,
    "train_order": "PAIR64_ORIGINAL_ORDER_PLUS_FULL_TRAIN32_EXECUTION_ASCENDING_FULL_BATCH",
    "pair_loss": "MEAN_BCE_LABEL0_WEIGHT4_LABEL1_WEIGHT1",
    "query_loss": "BASE_CORRECT_4SOFTPLUS_MAX;BASE_WRONG_SOFTPLUS_NEG_TARGET_PLUS4SOFTPLUS_MAX_OTHER",
    "cpu_threads": 8, "EVAL_modes": list(MODES), "EVAL_queries": 32, "candidate_count": 128,
    "EVAL_features": "PARENT_ALL_FOUR_MODES_ALREADY_SEALED_AND_INDEPENDENT_LITERAL_VALIDATED",
    "postjoin": "NEW_HEAD_AND_ALL_EVAL_LOGITS_SEALED_BEFORE_PARENT_RESULT_TARGET_READ",
    "controls": "ALL_FOUR_MODES_WITH_NEW_HEAD_PLUS_OLD_HEAD_REAL_AND_UNIFORM_CROSSED_COMPARISONS",
    "validator": "INDEPENDENT_2000_UPDATE_SEVEN_PARAMETER_HEAD_ONLY_RETRAIN_AND_COMPLETE_FROZEN_OUTPUT_REPLAY",
    "early_stopping": False, "checkpoint_selection": False, "hyperparameter_search": False,
    "new_prior_or_backbone": False, "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None,
    "formal_panel_consumed": False, "deployment_replacement_authorized": False, "automatic_stage_advance": False,
}


def need(value, message):
    if not bool(value): raise RuntimeError(message)


def load_runner():
    path = ROOT / PINS["prior_runner"][0]
    need(hashlib.sha256(path.read_bytes()).hexdigest() == PINS["prior_runner"][1], "PRIOR_RUNNER_PIN")
    spec = importlib.util.spec_from_file_location("frozen_query_prior_for_action_refit", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def barrier(runner):
    runner.install_barrier()
    runner.BARRIER.outcome_paths.add(str((PARENT / "result.json").resolve()))


def sources(runner):
    result = {name: {"path": path, "sha256": digest} for name, (path, digest) in PINS.items()}
    for item in result.values(): runner.checked(item)
    result.update(program=runner.binding(Path(__file__)), plan=runner.binding(PLAN), launcher=runner.binding(LAUNCH))
    return result


def modules(runner):
    import torch
    from torch import nn
    from torch.nn import functional as F
    core, frozen, pure = runner.modules()
    source = ROOT / PINS["old_training_source"][0]
    need(runner.sha(source) == PINS["old_training_source"][1], "TRAIN_HEAD_SOURCE_PIN")
    nodes = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "train_head"]
    need(len(nodes) == 1, "TRAIN_HEAD_AST_NOT_UNIQUE")
    pure.update(nn=nn, F=F, STEPS=2000)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, nodes[0]], type_ignores=[])), str(source), "exec"), pure)
    return core, frozen, pure


def verify_parent(runner, source_bindings):
    validation = runner.read(runner.checked(source_bindings["prior_validation"]))
    parameters = runner.read(runner.checked(source_bindings["prior_parameters"]))
    seal = runner.read(runner.checked(source_bindings["prior_eval_seal"]))
    need(validation["status"] == "RC_SHARED_QUERY_TARGET_PRIOR_INDEPENDENT_FROZEN_REPLAY_VALIDATION_PASS"
         and validation["result_sha256"] == source_bindings["prior_result"]["sha256"]
         and validation["checks"] and all(value is True for value in validation["checks"].values()), "PRIOR_NOT_VALIDATED")
    need(validation["authority_sha256"] == source_bindings["prior_authority"]["sha256"]
         and seal["parameters_sha256"] == source_bindings["prior_parameters"]["sha256"]
         and seal["eval_prejoin_sha256"] == source_bindings["prior_eval_prejoin"]["sha256"]
         and parameters["updates"] == 2000, "PRIOR_PARAMETER_PREJOIN_BINDING")
    return parameters


def prepare(runner, a, core, frozen, pure):
    import numpy as np
    import torch
    from torch.nn import functional as F
    parent_parameters = verify_parent(runner, a["sources"])
    parent_authority = runner.authority()
    need(parent_authority["sources"]["split_qualification"] == a["sources"]["split_qualification"], "IDENTITY_SPLIT_PIN")
    old_head, _ = runner.load_head(parent_authority["sources"], core, pure)
    pair_inputs, train_inputs, eval_inputs, entries, closure = runner.prepare(parent_authority["sources"], core, frozen, old_head)
    need(closure == runner.read(runner.checked(a["sources"]["prior_input_closure"])), "PARENT_INPUT_CLOSURE_REPLAY")
    train_inputs = runner.join_roles(train_inputs, entries, runner.corrected_labels())
    prior = core.SharedQueryTargetPrior()
    with torch.no_grad(): prior.theta.copy_(torch.tensor([float.fromhex(x) for x in parent_parameters["theta_binary64"]], dtype=torch.float64))
    prior.theta.requires_grad_(False)
    need(runner.tensor_sha(prior.theta) == parent_parameters["theta_sha256"], "FROZEN_PRIOR_SHA")
    pair = {"records": []}; train = []; feature_ledger = []
    with torch.no_grad():
        for kind, rows in (("PAIR", pair_inputs), ("TRAIN", train_inputs)):
            for row in rows:
                output = runner.episode_outputs(row, prior, core, old_head, "REAL")
                literal = runner.literal_episode_outputs(row, prior, core, frozen, old_head, "REAL")
                for key in output: runner.exact_tensor(output[key], literal[key], "FROZEN_TRAIN_LITERAL_REPLAY:" + key)
                matrix = output["features"].detach().clone()
                if kind == "PAIR":
                    pair["records"].append({"switch_label": row["meta"]["switch_label"], "real_native_features": {"C_PAIRED": matrix}})
                else:
                    train.append({**row["action_row"], "target_position": row["target_position"],
                                  "real_native_features": {"C_PAIRED": matrix}})
                feature_ledger.append({"kind": kind, "episode_key": row["meta"]["episode_key"],
                                       "feature_sha256": runner.tensor_sha(matrix), "old_head_logits_sha256": runner.tensor_sha(output["logits"])})
    # Cached-feature replay of the actual final prior objective, under the old
    # frozen head, must exactly match the parent trace endpoint before refit.
    pair_logits = torch.cat([runner.native_logits(row["real_native_features"]["C_PAIRED"], old_head) for row in pair["records"]])
    y = torch.tensor([float(row["switch_label"]) for row in pair["records"]], dtype=torch.float64)
    weights = torch.where(y.eq(0), torch.full_like(y, 4.), torch.ones_like(y))
    pair_loss = (F.binary_cross_entropy_with_logits(pair_logits, y, reduction="none") * weights).mean()
    query_losses = []
    for row in train:
        logits = runner.native_logits(row["real_native_features"]["C_PAIRED"], old_head)
        if row["target_position"] == row["base_winner_position"]: query_losses.append(4 * F.softplus(logits.max()))
        else:
            index = row["challenger_positions"].index(row["target_position"])
            mask = torch.ones(127, dtype=torch.bool); mask[index] = False
            query_losses.append(F.softplus(-logits[index]) + 4 * F.softplus(logits[mask].max()))
    full_loss = torch.stack(query_losses).mean()
    prior_loss = torch.stack((pair_loss + full_loss, pair_loss, full_loss))
    need([runner.hx(x) for x in prior_loss] == parent_parameters["final_frozen_source_loss_binary64"], "PRIOR_FINAL_LOSS_TRACE_REPLAY")
    parent_prejoin = runner.read(runner.checked(a["sources"]["prior_eval_prejoin"]))
    by_execution = {row["execution_ordinal"]: row for row in parent_prejoin}
    evals = []
    for row in eval_inputs:
        saved = by_execution[row["meta"]["execution_ordinal"]]
        need(saved["query_id"] == row["meta"]["query_id"] and saved["candidate_physical_rows"] == row["meta"]["axis"], "EVAL_FEATURE_BINDING")
        mode_features = {}
        with np.load(runner.checked(saved["arrays"]), allow_pickle=False) as arrays:
            for mode in MODES:
                matrix = torch.from_numpy(arrays[mode + "__features"].copy())
                need(matrix.dtype == torch.float64 and matrix.shape == (127, 6) and bool(torch.isfinite(matrix).all()), "SEALED_EVAL_FEATURE_MATRIX")
                old_logits = runner.native_logits(matrix, old_head)
                runner.exact_tensor(old_logits, torch.from_numpy(arrays[mode + "__logits"].copy()), "OLD_HEAD_EVAL_LOGIT_REPLAY")
                mode_features[mode] = matrix
        evals.append({**row["action_row"], "mode_features": mode_features, "parent_array_binding": saved["arrays"]})
    need(len(pair["records"]) == 64 and len(train) == len(evals) == 32, "TRAIN_EVAL_COUNTS")
    input_closure = {"status": "FROZEN_PRIOR_HEAD_REFIT_INPUTS_EXACT", "prior_theta_sha256": parent_parameters["theta_sha256"],
        "pair_order": closure["pair_order"], "train_order": closure["train_order"], "eval_order": closure["eval_order"],
        "training_feature_ledger": feature_ledger, "prior_final_loss_binary64": [runner.hx(x) for x in prior_loss],
        "eval_feature_sources": [{"execution_ordinal": row["execution_ordinal"], "source": row["parent_array_binding"],
            "feature_sha256": {mode: runner.tensor_sha(row["mode_features"][mode]) for mode in MODES}} for row in evals],
        "split_qualification": a["sources"]["split_qualification"], "EVAL_target_reads": 0, "prior_updates": 0}
    return pair, train, evals, input_closure, old_head


def fit_head(pair, train, pure, runner):
    need(pure["STEPS"] == 2000, "PRODUCTION_HEAD_UPDATES_DRIFT")
    head, final_losses, finite = pure["train_head"](pair, train, "NATIVE7", "C_PAIRED")
    weight = head.weight.detach().flatten(); bias = float(head.bias.detach())
    need(finite["all_finite"] and finite["finite_loss_step_count"] == finite["finite_gradient_step_count"] == 2000
         and finite["finite_parameter_state_count"] == 2001, "HEAD_TRAINING_FINITE_COUNTS")
    return {"weight_binary64": [runner.hx(x) for x in weight], "bias_binary64": runner.hx(bias),
            "parameter_sha256": pure["parameter_sha"](weight, bias), "parameter_count": 7, "head_updates": 2000,
            "last_preupdate_loss": list(final_losses), "finite_training": finite, "prior_updates": 0}, weight, bias


def predict(row, logits, runner):
    axis = row["candidate_physical_rows"]; challengers = row["challenger_positions"]; winner = row["base_winner_position"]
    index = max(range(127), key=lambda i: (float(logits[i]), -axis[challengers[i]]))
    best = challengers[index]; switch = float(logits[index]) > 0.
    return {"all127_logits_binary64": [runner.hx(x) for x in logits], "base_winner_position": winner,
            "proposed_challenger_position": best, "decision": "SWITCH" if switch else "HOLD",
            "final_position": best if switch else winner, "switch_logit_binary64": runner.hx(logits[index])}


def prejoin(evals, weight, bias, runner):
    return [{"execution_ordinal": row["execution_ordinal"], "query_id": row["query_id"],
             "candidate_physical_rows": row["candidate_physical_rows"],
             "predictions": {mode: predict(row, row["mode_features"][mode] @ weight + bias, runner) for mode in MODES},
             "EVAL_target_reads": 0} for row in evals]


def compare(real, baseline):
    base = {row["execution_ordinal"]: row for row in baseline}
    need(set(base) == {row["execution_ordinal"] for row in real}, "PAIRED_EXECUTION_AXIS")
    rescue = [row["execution_ordinal"] for row in real if row["final_correct"] and not base[row["execution_ordinal"]]["final_correct"]]
    broken = [row["execution_ordinal"] for row in real if not row["final_correct"] and base[row["execution_ordinal"]]["final_correct"]]
    return {"rescue": len(rescue), "break": len(broken), "paired_net": len(rescue) - len(broken),
            "rescued_execution_ordinals": rescue, "broken_execution_ordinals": broken}


def postjoin(a, runner, pure, evals, predictions, weight, bias):
    seal = runner.read(OUT / "eval_prejoin_seal.json")
    need(seal["parameters_sha256"] == runner.sha(OUT / "parameters.json")
         and seal["eval_prejoin_sha256"] == runner.sha(OUT / "eval_prejoin.json")
         and seal["eval_query_count"] == 32 and seal["modes"] == list(MODES)
         and runner.BARRIER.blocked == 0, "NEW_HEAD_LOGITS_NOT_FULLY_SEALED")
    runner.BARRIER.released = True
    parent = runner.read(runner.checked(a["sources"]["prior_result"]))
    need(parent["authority_sha256"] == a["sources"]["prior_authority"]["sha256"], "PARENT_OUTCOME_LINEAGE")
    targets = {row["execution_ordinal"]: row for row in parent["actions"]["REAL"]}
    by_execution = {row["execution_ordinal"]: row for row in predictions}
    actions = {}
    for mode in MODES:
        rows = []
        for row in evals:
            target = targets[row["execution_ordinal"]]
            need(target["query_id"] == row["query_id"] and target["base_winner"] == row["base_winner_position"]
                 and target["target_physical_row"] == row["candidate_physical_rows"][target["target_position"]], "PARENT_TARGET_CANDIDATE_BINDING")
            rows.append({**row, "target_position": target["target_position"], "real_native_features": {"C_PAIRED": row["mode_features"][mode]}})
        actions[mode] = pure["actions"](weight, bias, rows, "NATIVE7", "C_PAIRED")
        for action in actions[mode]:
            prediction = by_execution[action["execution_ordinal"]]["predictions"][mode]
            need(prediction["final_position"] == action["final_position"] and prediction["decision"] == action["decision"]
                 and prediction["switch_logit_binary64"] == runner.hx(action["switch_logit"]), "NEW_HEAD_POSTJOIN_ACTION_DRIFT")
    comparisons = {"NEW_HEAD_" + mode: compare(actions["REAL"], actions[mode]) for mode in MODES[1:]}
    comparisons.update({"OLD_HEAD_" + mode: compare(actions["REAL"], parent["actions"][mode]) for mode in ("REAL", "UNIFORM")})
    return {"status": "RC_SHARED_QUERY_PRIOR_HEAD_REFIT_V1_COMPLETE", "authority_sha256": runner.sha(AUTH),
            "claim_level": "OPENED_INTERNAL_EVAL32_FROZEN_PRIOR_ORIGINAL_ACTION_HEAD_INTERFACE_REFIT",
            "input_closure_sha256": runner.sha(OUT / "input_closure.json"), "parameters_sha256": runner.sha(OUT / "parameters.json"),
            "eval_prejoin_seal_sha256": runner.sha(OUT / "eval_prejoin_seal.json"),
            "new_head_metrics": {mode: pure["summary"](rows) for mode, rows in actions.items()},
            "old_head_metrics": parent["metrics"], "new_head_REAL_paired_comparisons": comparisons,
            "new_head_CBIND_rescue_retention": pure["retention"](actions["REAL"], actions["CBIND"]), "new_head_actions": actions,
            "head_training_updates": 2000, "new_head_count": 1, "trainable_parameter_count": 7, "prior_updates": 0,
            "candidate_source": "ORIGINAL_RAW_C128", "frozen_prior_source": a["sources"]["prior_parameters"],
            "PAIR_training_count": 64, "FULL_training_count": 32, "EVAL_count": 32, "candidate_count": 128,
            "EVAL_targets_only_after_all_new_head_logits_sealed": True, "early_blocked_read_count": runner.BARRIER.blocked,
            "limits": ["This is one refit on the unchanged training population with the prior frozen; no architecture or threshold search.",
                       "New-head REAL and new-head UNIFORM must both be reported, together with old-head REAL and UNIFORM.",
                       "Previously opened internal EVAL32 does not supply untouched confirmation or pixel ownership evidence."],
            "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None, "formal_panel_consumed": False,
            "deployment_replacement_authorized": False, "automatic_stage_advance": False}


def e0(runner):
    import torch
    core, frozen, pure = modules(runner)
    rng = torch.Generator().manual_seed(17)
    rows = []
    for index in (0, 1):
        rows.append({"execution_ordinal": index, "query_id": "synthetic" + str(index), "track": "synthetic", "heldout_fold": 1,
            "candidate_physical_rows": list(range(128)), "base_scores": torch.arange(128., 0., -1., dtype=torch.float64),
            "base_winner_position": 0, "challenger_positions": list(range(1, 128)), "target_position": index,
            "real_native_features": {"C_PAIRED": torch.randn((127, 6), generator=rng, dtype=torch.float64)}})
    pair = {"records": [{"switch_label": bool(index), "real_native_features": {"C_PAIRED": row["real_native_features"]["C_PAIRED"][:1]}}
                         for index, row in enumerate(rows)]}
    pure["STEPS"] = 2
    head1, loss1, finite1 = pure["train_head"](pair, rows, "NATIVE7", "C_PAIRED")
    head2, loss2, finite2 = pure["train_head"](pair, rows, "NATIVE7", "C_PAIRED")
    runner.exact_tensor(head1.weight, head2.weight, "SYNTHETIC_HEAD_RETRAIN_WEIGHT")
    runner.exact_tensor(head1.bias, head2.bias, "SYNTHETIC_HEAD_RETRAIN_BIAS")
    need(loss1 == loss2 and finite1 == finite2 and finite1["all_finite"], "SYNTHETIC_HEAD_REPLAY")
    pure["STEPS"] = 2000
    actions = pure["actions"](head1.weight.detach().flatten(), float(head1.bias.detach()), rows, "NATIVE7", "C_PAIRED")
    for row, action in zip(rows, actions):
        pred = predict(row, row["real_native_features"]["C_PAIRED"] @ head1.weight.detach().flatten() + float(head1.bias.detach()), runner)
        need(pred["final_position"] == action["final_position"] and pred["switch_logit_binary64"] == runner.hx(action["switch_logit"]), "SYNTHETIC_TARGET_FREE_ACTION")
    test_barrier = runner.ReadBarrier(); result_path = str((PARENT / "result.json").resolve())
    test_barrier.outcome_paths.add(result_path)
    try: test_barrier.hook("open", (result_path, "r"))
    except RuntimeError: pass
    else: raise RuntimeError("PARENT_TARGET_BARRIER_ACCEPTED")
    test_barrier.hash_only = 1; test_barrier.hook("open", (result_path, "rb")); test_barrier.hash_only = 0
    test_barrier.released = True; test_barrier.hook("open", (result_path, "r"))
    return {"status": "RC_SHARED_QUERY_PRIOR_HEAD_REFIT_V1_E0_PASS", "checks": {
        "original_train_head_AST_exact_source": True, "synthetic_original_head_retraining_bit_exact": True,
        "original_127_challenger_target_free_action_parity": True, "parent_EVAL_outcomes_blocked_until_new_prejoin": True,
        "parent_hash_only_read_separate": True, "production_steps_restored_2000": pure["STEPS"] == 2000},
        "natural_training_updates": 0, "prior_updates": 0, "scientific_GO_or_NO_GO": None}


def freeze(runner):
    need(not AUTH.exists() and not OUT.exists(), "APPEND_ONLY_AUTHORITY_OR_OUTPUT_EXISTS")
    source_bindings = sources(runner)
    verify_parent(runner, source_bindings)
    parent_authority = runner.authority()
    need(parent_authority["sources"]["split_qualification"] == source_bindings["split_qualification"], "SPLIT_AUTHORITY_BINDING")
    e0path = ROOT / "results" / (PREFIX + "_e0") / "result.json"
    runner.atomic(e0path, e0(runner)); source_bindings["e0"] = runner.binding(e0path)
    value = {"status": "RC_SHARED_QUERY_PRIOR_HEAD_REFIT_V1_AUTHORIZED", "sources": source_bindings,
             "contract": CONTRACT, "output_rel": str(OUT.relative_to(ROOT)), "scientific_GO_or_NO_GO": None}
    runner.atomic(AUTH, value)
    print(json.dumps({"status": value["status"], "authority_sha256": runner.sha(AUTH)}), flush=True)


def authority(runner):
    value = runner.read(AUTH)
    need(value["status"] == "RC_SHARED_QUERY_PRIOR_HEAD_REFIT_V1_AUTHORIZED" and value["contract"] == CONTRACT
         and value["output_rel"] == str(OUT.relative_to(ROOT)), "AUTHORITY_SCOPE")
    for source in value["sources"].values(): runner.checked(source)
    need({k: v for k, v in value["sources"].items() if k != "e0"} == sources(runner), "IMPLEMENTATION_SOURCE_DRIFT")
    check = runner.read(runner.checked(value["sources"]["e0"]))
    need(check["status"] == "RC_SHARED_QUERY_PRIOR_HEAD_REFIT_V1_E0_PASS" and all(v is True for v in check["checks"].values()), "E0_NOT_CLOSED")
    return value


def execute(runner, validate=False):
    a = authority(runner)
    core, frozen, pure = modules(runner)
    pair, train, evals, closure, old_head = prepare(runner, a, core, frozen, pure)
    if validate: need(OUT.is_dir() and runner.read(OUT / "input_closure.json") == closure, "INPUT_CLOSURE_REPLAY")
    else:
        need(not OUT.exists(), "APPEND_ONLY_OUTPUT_EXISTS"); OUT.mkdir(parents=True)
        runner.atomic(OUT / "input_closure.json", closure)
    params, weight, bias = fit_head(pair, train, pure, runner)
    params.update(prior_theta_sha256=closure["prior_theta_sha256"], source_parameters=a["sources"]["prior_parameters"],
                  EVAL_target_reads=0, checkpoint_selection=False)
    if validate: need(runner.read(OUT / "parameters.json") == params, "INDEPENDENT_2000_STEP_HEAD_RETRAIN_DRIFT")
    else: runner.atomic(OUT / "parameters.json", params)
    predictions = prejoin(evals, weight, bias, runner)
    if validate: need(runner.read(OUT / "eval_prejoin.json") == predictions, "ALL_EVAL_MODE_LOGITS_REPLAY")
    else:
        runner.atomic(OUT / "eval_prejoin.json", predictions)
        runner.atomic(OUT / "eval_prejoin_seal.json", {"status": "NEW_HEAD_ALL_EVAL_MODE_LOGITS_SEALED",
            "authority_sha256": runner.sha(AUTH), "parameters_sha256": runner.sha(OUT / "parameters.json"),
            "eval_prejoin_sha256": runner.sha(OUT / "eval_prejoin.json"), "eval_query_count": 32,
            "candidate_count": 128, "modes": list(MODES), "EVAL_target_reads": 0})
    result = postjoin(a, runner, pure, evals, predictions, weight, bias)
    result["new_head_FULL_TRAIN32_metrics"] = pure["summary"](pure["actions"](weight, bias, train, "NATIVE7", "C_PAIRED"))
    if validate:
        need(runner.read(OUT / "result.json") == result, "INDEPENDENT_RESULT_REPLAY")
        receipt = {"status": "RC_SHARED_QUERY_PRIOR_HEAD_REFIT_V1_INDEPENDENT_VALIDATION_PASS",
            "authority_sha256": runner.sha(AUTH), "result_sha256": runner.sha(OUT / "result.json"), "checks": {
                "same_frozen_prior_training_features_reconstructed_and_literal_checked": True,
                "prior_final_training_loss_matches_original_trace_exactly": True,
                "single_original_seven_parameter_head_2000_updates_independently_retrained": True,
                "all_head_parameters_losses_and_finite_counts_exact": True,
                "all_four_EVAL_mode_logits_actions_metrics_exact": True,
                "EVAL_target_outcome_read_only_after_new_head_and_all_logits_sealed": True,
                "old_head_and_new_head_crossed_controls_all_reported": True,
                "prior_never_retrained": True}, "head_training_repeated_for_validation": True,
            "prior_updates": 0, "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}
        runner.atomic(OUT / "independent_validation.json", receipt)
        print(json.dumps(receipt), flush=True)
    else:
        runner.atomic(OUT / "result.json", result)
        print(json.dumps({"status": result["status"], "new_head_metrics": result["new_head_metrics"],
                          "new_head_REAL_paired_comparisons": result["new_head_REAL_paired_comparisons"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("e0", "freeze", "run", "validate"), required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    runner = load_runner()
    if args.phase == "e0": print(json.dumps(e0(runner), sort_keys=True), flush=True); return
    barrier(runner)
    if args.phase == "freeze": freeze(runner)
    else: execute(runner, validate=args.phase == "validate")


if __name__ == "__main__": main()
