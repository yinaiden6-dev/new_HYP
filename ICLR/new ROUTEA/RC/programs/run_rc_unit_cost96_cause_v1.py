#!/usr/bin/env python3
"""Matched original mixed96 control: change only three negative-cost constants."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "programs"), str(ROOT / "src")]
sys.dont_write_bytecode = True
import torch
from torch import nn
import torch.nn.functional as F
import materialize_rc_new_hyp593_inputs_v1 as M

TRAIN = ROOT / "results/rc_convex_train_loss_cause_v1"
OUT = TRAIN / "unit_cost96"
PREV = ROOT / "results/rc_fixed_panels_train269_group_risk_v1"
AUTH = ROOT / "registry/rc_convex_cause_isolation_authority_v1_20260912.json"
LEGACY = ROOT / "programs/run_routea_matched_three_arm_common3_native7_crossfit_v1.py"
LEGACY_SHA = "546e2bc7b3df6c67bcac41e079ba1b00f15c7e65c52a8ab994cd5b2c38d81e22"
STEPS = 2000
MODEL = "TRAIN_UNIT_COST7"


def checked(binding):
    path = Path(binding["path"])
    path = path if path.is_absolute() else ROOT / path
    M.need(M.sha(path) == binding["sha256"], "SOURCE_DRIFT:" + str(path))
    return path


def configure_threads():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)


def sourcecheck(stage):
    authority = M.read(AUTH)
    for binding in authority["sources"].values():
        checked(binding)
    M.need(authority["sources"]["unit_cost_program"] == M.bind(__file__), "UNIT_COST_SOURCE_BINDING")
    M.need(M.sha(LEGACY) == LEGACY_SHA, "ORIGINAL_TRAIN_FUNCTION_SOURCE_PIN")
    if stage != "preflight":
        M.need(bool(os.environ.get("SLURM_JOB_ID")), "SLURM_REQUIRED")

    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        lower = str(path).lower()
        M.need(not any(x in lower for x in ("rc_opened_", "curator_roles", "/reports/", "d1-mi",
                   "d1_mi", "grozi", "gisc_prerecall_universe", "/target_join/")), "TRAIN_ONLY_SOURCE_BARRIER")
        parent = ROOT / "results/rc_new_hyp593_oof5_v1"
        M.need(parent / "fits" not in path.parents and parent / "roles" not in path.parents,
               "NO_OTHER_FOLD_FITS_OR_LABELS")
        M.need(path.name not in ("result.json", "result_validation.json") or TRAIN in path.parents,
               "NO_EVAL_RESULTS")
        if stage in ("fit", "replay", "preflight"):
            M.need(PREV / "fits" not in path.parents, "NO_PREVIOUS_PARAMETERS_IN_TRAINING")

    sys.addaudithook(audit)
    if stage != "preflight":
        pf = M.read(OUT / "preflight.json")
        M.need(pf["status"] == "UNIT_COST96_OBJECTIVE_PREFLIGHT_PASS" and
               pf["authority"] == M.bind(AUTH) and pf["input_seal"] == M.bind(TRAIN / "input_seal.json"),
               "UNIT_COST_PREFLIGHT_BINDINGS")
    return authority


def training_function(steps=STEPS):
    M.need(M.sha(LEGACY) == LEGACY_SHA, "ORIGINAL_TRAIN_SOURCE_PIN")
    text = LEGACY.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "train_head")
    original = ast.get_source_segment(text, node)
    replacements = [
        ("torch.full_like(py, 4.0)", "torch.full_like(py, 1.0)"),
        ("query_losses.append(4 * F.softplus(values.max()))",
         "query_losses.append(1 * F.softplus(values.max()))"),
        ("query_losses.append(F.softplus(-values[position]) + 4 * F.softplus(values[mask].max()))",
         "query_losses.append(F.softplus(-values[position]) + 1 * F.softplus(values[mask].max()))"),
    ]
    body = original
    for before, after in replacements:
        M.need(body.count(before) == 1, "EXACTLY_THREE_COST_EDITS")
        body = body.replace(before, after)
    reverted = body
    for before, after in replacements:
        M.need(reverted.count(after) == 1, "UNIQUE_CHANGED_COST_SITE")
        reverted = reverted.replace(after, before)
    M.need(reverted == original and "loss = pair_loss + full_loss" in body, "ONLY_COST_CONSTANTS_CHANGED")

    def finite_tensor(value, name):
        M.need(bool(torch.isfinite(value).all()), "FINITE_" + name)

    ns = dict(torch=torch, nn=nn, F=F, STEPS=steps, finite_tensor=finite_tensor,
              CrossfitContractError=RuntimeError,
              FAMILIES={"NATIVE7": ("real_native_features", "cbind_native_features", list(range(6)))})
    exec(compile("from __future__ import annotations\n" + body, str(LEGACY) + "#UNIT_COST_3_EDITS", "exec"), ns)
    change = dict(original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  changed_function_sha256=hashlib.sha256(body.encode()).hexdigest(), replacement_count=3,
                  replacements=[dict(before=a, after=b) for a, b in replacements],
                  PAIR_FULL_mixture=[1, 1], group_weights_used=False, negative_cost_before=4,
                  negative_cost_after=1, positive_cost=1)
    return ns["train_head"], change


def synthetic_preflight():
    # Exercise all three sites without opening any natural training/evaluation input.
    pair = {"records": [dict(real_native_features={"C_PAIRED": torch.tensor([row], dtype=torch.float64)},
                              switch_label=label) for row, label in
                             [([1., -.5, .2, 0., 1., -1.], False),
                              ([-.3, 1., -.1, .8, .2, 1.], True)]]}
    full = [dict(real_native_features={"C_PAIRED": torch.tensor(rows, dtype=torch.float64)},
                 base_winner_position=0, target_position=target, challenger_positions=[1, 2])
            for rows, target in [([[.4, .7, -.2, 1., 0., -.5], [-.2, .3, .8, -.5, 1., .4]], 0),
                                 ([[.2, -.4, .6, .8, -.1, 1.], [-1., .2, -.8, 0., .3, .7]], 1)]]
    fn, change = training_function(steps=1)
    actual, losses, finite = fn(pair, full, "NATIVE7", "C_PAIRED")
    torch.manual_seed(17)
    literal = nn.Linear(6, 1, dtype=torch.float64)
    literal.weight.data.zero_()
    literal.bias.data.zero_()
    opt = torch.optim.AdamW(literal.parameters(), lr=.03, weight_decay=.001)
    px = torch.cat([row["real_native_features"]["C_PAIRED"] for row in pair["records"]])
    py = torch.tensor([0., 1.], dtype=torch.float64)
    pair_loss = F.binary_cross_entropy_with_logits(literal(px).squeeze(1), py, reduction="none").mean()
    z0 = literal(full[0]["real_native_features"]["C_PAIRED"]).squeeze(1)
    z1 = literal(full[1]["real_native_features"]["C_PAIRED"]).squeeze(1)
    full_loss = torch.stack([F.softplus(z0.max()), F.softplus(-z1[0]) + F.softplus(z1[1])]).mean()
    total = pair_loss + full_loss
    total.backward()
    M.need(losses == tuple(float(v.detach()) for v in (total, pair_loss, full_loss)), "SYNTHETIC_UNIT_COST_LOSS")
    M.need(all(torch.equal(a.grad, b.grad) for a, b in zip(actual.parameters(), literal.parameters())),
           "SYNTHETIC_UNIT_COST_GRADIENT")
    opt.step()
    M.need(all(torch.equal(a, b) for a, b in zip(actual.parameters(), literal.parameters())),
           "SYNTHETIC_UNIT_COST_ADAMW_STEP")
    M.need(finite["all_finite"] and training_function()[0].__globals__["STEPS"] == 2000,
           "PRODUCTION_2000_STEPS_RETAINED")
    return dict(status="UNIT_COST96_SYNTHETIC_LOSS_GRADIENT_ADAMW_STEP_PASS", function_change=change,
                synthetic_updates_per_implementation=1, natural_training_updates=0,
                loss_value=float(total.detach()), checks=dict(loss=True, gradients=True, optimizer_step=True,
                all_three_cost_sites=True, production_steps_unchanged=True))


def preflight():
    result = synthetic_preflight()
    seal = M.read(TRAIN / "input_seal.json")
    M.need(seal["status"] == "TRAIN_INPUTS_SEALED_BEFORE_EVAL_CONES" and
           seal["counts"]["PAIR"] == 64 and seal["counts"]["FULL"] == 32, "SEALED_MIXED96")
    checked(seal["sources"]["original_train_pack"])
    M.write(OUT / "preflight.json", dict(status="UNIT_COST96_OBJECTIVE_PREFLIGHT_PASS",
            authority=M.bind(AUTH), input_seal=M.bind(TRAIN / "input_seal.json"),
            function_change=result["function_change"], synthetic=result, original_head_training_updates=0,
            natural_training_updates=0, cpu_threads=8, interop_threads=1))
    print("UNIT_COST96_OBJECTIVE_PREFLIGHT_PASS", flush=True)


def compute():
    seal = M.read(TRAIN / "input_seal.json")
    M.need(seal["status"] == "TRAIN_INPUTS_SEALED_BEFORE_EVAL_CONES", "TRAIN_INPUT_SEAL")
    pack = torch.load(checked(seal["sources"]["original_train_pack"]), weights_only=True, map_location="cpu")
    original = M.read(checked(seal["sources"]["original_train_input_seal"]))
    pair, full = pack["pair"]["records"], pack["full"]
    M.need(len(pair) == 64 and len(full) == 32 and len({r["query_id"] for r in pair + full}) == 96,
           "EXACT_ORIGINAL_MIXED96")
    M.need([r["execution_ordinal"] for r in pair] == [r[2] for r in original["original_pair_order"]] and
           [r["execution_ordinal"] for r in full] == original["original_training_order"], "ORIGINAL_TRAIN_ORDER")
    fn, change = training_function()
    M.need(change == M.read(OUT / "preflight.json")["function_change"], "PREFLIGHT_FUNCTION_REPLAY")
    head, losses, finite = fn(pack["pair"], full, "NATIVE7", "C_PAIRED")
    theta = torch.cat([head.weight.detach().flatten(), head.bias.detach()])
    M.need(finite["all_finite"] and len(theta) == 7, "FINITE_UNIT_COST_HEAD")
    return dict(status="UNIT_COST96_MATCHED_TRAINING_COMPLETE", theta_binary64=[float(v).hex() for v in theta],
                final_preupdate_losses=dict(zip(("total", "PAIR", "FULL"), losses)),
                finite_training=finite, function_change=change, authority=M.bind(AUTH),
                input_seal=M.bind(TRAIN / "input_seal.json"),
                original_train_pack=seal["sources"]["original_train_pack"],
                original_head_training_updates=0, new_head_training_updates=STEPS, heldout_label_reads=0,
                train_count=dict(PAIR=64, FULL=32, total=96), cpu_threads=8, interop_threads=1,
                optimizer=dict(name="AdamW", lr=.03, weight_decay=.001, steps=STEPS,
                               seed=17, initialization="zero", dtype="float64"))


def fit():
    M.need(not (OUT / "fit.json").exists(), "IMMUTABLE_UNIT_COST_FIT")
    value = compute()
    M.write(OUT / "fit.json", value)
    nonce = uuid.uuid4().hex
    subprocess.run([sys.executable, __file__, "replay", "--nonce", nonce],
                   env=dict(os.environ, RC_UNIT_COST_REPLAY_NONCE=nonce), check=True)
    print("UNIT_COST96_FRESH_PARAMETER_REPLAY_PASS", flush=True)


def replay(nonce):
    M.need(bool(nonce) and nonce == os.environ.get("RC_UNIT_COST_REPLAY_NONCE"), "FRESH_UNIT_COST_REPLAY")
    saved = M.read(OUT / "fit.json")
    M.need(saved["authority"] == M.bind(AUTH), "FIT_AUTHORITY")
    value = compute()
    M.need(value == saved, "UNIT_COST_PARAMETER_AND_SOURCE_BIT_REPLAY")
    M.write(OUT / "fit_validation.json", dict(status="UNIT_COST96_FRESH_PARAMETER_REPLAY_PASS",
            fit=M.bind(OUT / "fit.json"), authority=M.bind(AUTH), input_seal=M.bind(TRAIN / "input_seal.json"),
            fresh_nonce=nonce, original_head_training_updates=0, new_head_training_updates=STEPS,
            heldout_label_reads=0))


def predict():
    import run_rc_new_hyp593_oof5_v1 as P
    validation = M.read(OUT / "fit_validation.json")
    M.need(validation["status"] == "UNIT_COST96_FRESH_PARAMETER_REPLAY_PASS" and
           validation["fit"] == M.bind(OUT / "fit.json") and validation["authority"] == M.bind(AUTH),
           "VALIDATED_UNIT_COST_FIT")
    fit_value = M.read(OUT / "fit.json")
    theta = torch.tensor([float.fromhex(x) for x in fit_value["theta_binary64"]], dtype=torch.float64)
    previous = M.read(PREV / "fits/receipt.json")
    pv = M.read(PREV / "fits/validation.json")
    M.need(pv["payload"] == previous["payload"] and pv["receipt"] == M.bind(PREV / "fits/receipt.json") and
           pv["status"] == "FIXED_PANEL_RETRAIN_AND_PREDICTION_REPLAY_PASS", "PREVIOUS_PREDICTION_SEAL")
    old = torch.load(checked(previous["payload"]), weights_only=True, map_location="cpu")
    oldpred = {row["query_id"]: row for row in old["predictions"]}
    rows, sources = P.features()
    test = [row for row in rows if row["query_id"] in oldpred]
    M.need(len(oldpred) == len(test) == 160 and len({row["query_id"] for row in test}) == 160,
           "EXACT_OLD160_PREDICTION_POPULATION")
    pred = P.predict({MODEL: dict(theta=theta)}, test)
    for row in pred:
        prior = oldpred[row["query_id"]]
        M.need(all(row[k] == prior[k] for k in ("candidate_physical_rows", "raw_ranked_physical_rows",
               "winner", "challenger_positions")), "UNCHANGED_ACTION_AXES")
        row["models"]["ORIGINAL7"] = prior["models"]["ORIGINAL7"]
    common = dict(unit_cost_fit=M.bind(OUT / "fit.json"),
                  unit_cost_fit_validation=M.bind(OUT / "fit_validation.json"), authority=M.bind(AUTH),
                  heldout_label_reads=0, original_head_training_updates=0, new_head_training_updates=STEPS)
    value = dict(parameters={MODEL: dict(theta=theta), "ORIGINAL7": old["parameters"]["ORIGINAL7"]},
                 predictions=pred, feature_sources=sources, previous_payload=previous["payload"], **common)
    path = OUT / "predictions.pt"
    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    M.write(OUT / "prediction_seal.json", dict(status="UNIT_COST96_TRAIN_ONLY_ALL160_PREDICTIONS_SEALED",
            payload=M.bind(path), **common))
    print("UNIT_COST96_TRAIN_ONLY_ALL160_PREDICTIONS_SEALED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "fit", "replay", "predict"))
    parser.add_argument("--nonce")
    args = parser.parse_args()
    configure_threads()
    sourcecheck(args.stage)
    if args.stage == "preflight":
        preflight()
    elif args.stage == "fit":
        fit()
    elif args.stage == "replay":
        replay(args.nonce)
    else:
        predict()
