#!/usr/bin/env python3
"""Fit native token competition V2 on the phase-matched frozen F128 B_CAL.

This is a separate experiment from v1. Inner selection uses the saved inner-only
B_CAL head/normalization; outer refit uses its saved all-TRAIN counterpart. No
RoMa/ColNomic forward, F-input renormalization, new threshold fit, or held label
read occurs here. Epoch zero is a genuine, auditable baseline fallback.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC / "src"))
from rc_aslo_xf.rebut_qr_qrr_v1 import fixed_anchor_decision, shared_hit_miss_loss
from run_rebut_qr_qrr_v1 import (FrozenInputs, atomic_json, atomic_torch, checked,
                               digest, new_model as baseline_new_model, predict as full_predict, read, restore_rng, rng_state, setup_seed)
from token_competition_data_v2 import TokenInputs, verify_panel, verify_sources, verify_evidence_gate
from rc_aslo_xf.token_competition_v2 import new_model as token_new_model


def new_model(arm, seed, config):
    setup_seed(seed)
    return baseline_new_model(arm, seed, config) if arm == "B_CAL" else token_new_model(arm, seed, config)


@torch.no_grad()
def predict(model, data, ids, scale, labels=False):
    # A zero output layer makes the reader contribution identically zero.
    # Avoid a full native-token panel forward just to replay epoch-zero B_CAL.
    if torch.count_nonzero(model.reader.output.weight).item() != 0:
        return full_predict(model, data, ids, scale, labels)
    rows = predict_frozen_head_only(data, ids, scale, {"head.weight": model.head.weight})
    losses = []
    if labels:
        for row in rows:
            qid = row["query_id"]
            if qid not in data.fold["train_query_ids"]:
                raise ValueError("epoch-zero loss requested outside TRAIN")
            target = data.roles[qid]["identity"]
            truth = torch.tensor([[data.identity[int(p)] == target for p in row["axis"]]])
            losses.append(float(shared_hit_miss_loss(torch.tensor([row["logits"]], dtype=torch.float64),
                                                     truth, torch.tensor([row["winner"]]))))
    return rows, sum(losses) / len(losses) if labels else None


def model_head(state):
    head = {key: value for key, value in state.items() if key.startswith("head.")}
    if set(head) != {"head.weight"} or head["head.weight"].shape != (1, 18):
        raise ValueError("frozen baseline must be the same bias-inclusive 18-parameter head")
    return {key: value.detach().clone().to(torch.float64) for key, value in head.items()}


def assert_head(model, expected):
    actual = model.state_dict()
    if any(not torch.equal(actual[key], value) for key, value in expected.items()):
        raise AssertionError("frozen baseline head changed")
    if any(p.requires_grad or p.grad is not None for p in model.head.parameters()):
        raise AssertionError("frozen baseline head participates in gradient updates")


def initialize(arm, seed, config, head):
    model, _ = new_model(arm, seed, config)
    with torch.no_grad():
        model.head.weight.copy_(head["head.weight"])
    for p in model.head.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.reader.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    trainable = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if trainable != {id(p) for p in model.reader.parameters()}:
        raise AssertionError("optimizer must contain exactly the reader parameters")
    if trainable & {id(p) for p in model.head.parameters()}:
        raise AssertionError("baseline parameters found in residual optimizer")
    assert_head(model, head)
    return model, optimizer


def load_baselines(protocol, fold, fold_id, seed):
    bindings = protocol["baseline_bindings"][str(fold_id)][str(seed)]
    if set(bindings) != {"source_result", "source_model", "source_checkpoint"}:
        raise ValueError("baseline bindings must specify result, model and checkpoint")
    paths = {key: checked(value) for key, value in bindings.items()}
    result = read(paths["source_result"])
    outer = torch.load(paths["source_model"], map_location="cpu", weights_only=False)
    checkpoint = torch.load(paths["source_checkpoint"], map_location="cpu", weights_only=False)
    source_binding = result["binding"]
    if source_binding["arm"] != "B_CAL" or source_binding["fold"] != fold_id or source_binding["seed"] != seed:
        raise ValueError("wrong source fold, seed or baseline arm")
    if (result["status"] != "TRAINING_AND_LABEL_FREE_PREDICTIONS_COMPLETE" or
            checkpoint["stage"] != "done" or outer["binding"] != source_binding or
            checkpoint["binding"] != source_binding):
        raise ValueError("baseline artifacts not consistently completed")
    if digest(paths["source_model"]) != result["model_sha256"]:
        raise ValueError("baseline result/model binding mismatch")
    for key in ("inner_fit_query_ids", "inner_val_query_ids", "train_query_ids", "heldout_query_ids"):
        if result[key] != fold[key]:
            raise ValueError(f"baseline split changed: {key}")
    if result["best_epoch"] != checkpoint["best_epoch"]:
        raise ValueError("baseline best epoch mismatch")
    inner_head = model_head(checkpoint["best_model"])
    outer_head = model_head(outer["model"])
    if any(not torch.equal(checkpoint["model"][k], v) for k, v in outer_head.items()):
        raise ValueError("outer baseline differs between checkpoint and model")
    inner_scale = checkpoint["inner_normalization"].detach().clone().to(torch.float64)
    outer_scale = outer["normalization"].detach().clone().to(torch.float64)
    for scale in (inner_scale, outer_scale):
        if scale.shape != (18,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("invalid frozen baseline normalization")
    if not torch.equal(outer_scale, checkpoint["normalization"]):
        raise ValueError("outer baseline normalization mismatch")
    inner_predictions_path = paths["source_result"].parent / "inner_validation.json"
    inner_predictions = read(inner_predictions_path)
    if inner_predictions["binding"] != source_binding:
        raise ValueError("baseline inner predictions source mismatch")
    if not torch.equal(torch.tensor(inner_predictions["normalization_rms"], dtype=torch.float64), inner_scale):
        raise ValueError("baseline inner RMS mismatch")
    return {"bindings": bindings, "result": result, "inner_head": inner_head,
            "outer_head": outer_head, "inner_scale": inner_scale, "outer_scale": outer_scale,
            "inner_predictions": inner_predictions["records"],
            "inner_predictions_binding": {"path": str(inner_predictions_path),
                                           "sha256": digest(inner_predictions_path)},
            "inherited_tau": float(result["calibration"]["threshold"])}


def verify_same_predictions(actual, expected, context):
    a, e = {r["query_id"]: r for r in actual}, {r["query_id"]: r for r in expected}
    if a.keys() != e.keys():
        raise AssertionError(f"{context}: query set mismatch")
    max_error = 0.0
    exact = True
    for qid, row in a.items():
        old = e[qid]
        if row["axis"] != old["axis"] or row["winner"] != old["winner"]:
            raise AssertionError(f"{context}: candidate axis mismatch")
        z = torch.tensor(row["logits"], dtype=torch.float64)
        zo = torch.tensor(old["logits"], dtype=torch.float64)
        max_error = max(max_error, float((z - zo).abs().max()))
        exact &= torch.equal(z, zo)
        if not torch.equal(z, zo):
            raise AssertionError(f"{context}: baseline scores not bit-exact; max error {max_error}")
    return {"bit_exact": exact, "queries": len(a), "maximum_error": max_error}


def training_metrics(records, data, baseline_records, tau):
    original = {r["query_id"]: r for r in baseline_records}
    decisions = []
    correct = rescues = breaks = switches = baseline_correct = inherited_correct = 0
    for row in records:
        qid, axis = row["query_id"], row["axis"]
        if qid not in data.fold["inner_val_query_ids"]:
            raise ValueError("selection metrics requested outside inner validation")
        anchor = torch.tensor([row["winner"]])
        z = torch.tensor([row["logits"]], dtype=torch.float64)
        zb = torch.tensor([original[qid]["logits"]], dtype=torch.float64)
        chosen = int(fixed_anchor_decision(z, anchor, 0.)[0])
        incumbent = int(fixed_anchor_decision(zb, anchor, 0.)[0])
        inherited = int(fixed_anchor_decision(z, anchor, tau)[0])
        target = data.roles[qid]["identity"]
        is_correct = data.identity[axis[chosen]] == target
        was_correct = data.identity[axis[incumbent]] == target
        correct += int(is_correct)
        baseline_correct += int(was_correct)
        inherited_correct += int(data.identity[axis[inherited]] == target)
        rescues += int(is_correct and not was_correct)
        breaks += int(was_correct and not is_correct)
        switches += int(chosen != row["winner"])
        decisions.append({"query_id": qid, "selected_position": chosen,
                          "selected_physical_row": axis[chosen], "correct": is_correct,
                          "baseline_selected_position": incumbent,
                          "baseline_selected_physical_row": axis[incumbent],
                          "baseline_correct": was_correct,
                          "inherited_tau_selected_position": inherited,
                          "inherited_tau_selected_physical_row": axis[inherited]})
    return {"fixed_zero_correct": correct, "baseline_fixed_zero_correct": baseline_correct,
            "rescues_vs_frozen_baseline": rescues, "breaks_vs_frozen_baseline": breaks,
            "net_vs_frozen_baseline": rescues - breaks, "switches_vs_RAW": switches,
            "inherited_tau_correct": inherited_correct, "queries": len(records),
            "decisions": decisions}


def attach_decisions(records, tau):
    output = []
    for row in records:
        logits = torch.tensor([row["logits"]], dtype=torch.float64)
        anchor = torch.tensor([row["winner"]])
        main = int(fixed_anchor_decision(logits, anchor, 0.)[0])
        secondary = int(fixed_anchor_decision(logits, anchor, tau)[0])
        output.append({**row, "logits_hex": [float(x).hex() for x in row["logits"]],
                       "selected_position": main, "selected_physical_row": row["axis"][main],
                       "selected_fixed_zero_position": main,
                       "selected_fixed_zero_physical_row": row["axis"][main],
                       "inherited_tau_selected_position": secondary,
                       "inherited_tau_selected_physical_row": row["axis"][secondary],
                       "action": "HOLD" if main == row["winner"] else "SWITCH"})
    return output


@torch.no_grad()
def predict_frozen_head_only(data, ids, scale, head):
    """Epoch-zero fallback also skips reader compute, not just reader updates."""
    records = []
    for qid in ids:
        row = data.common[qid]
        x = torch.as_tensor(row["X0"], dtype=torch.float64)[None] / scale
        anchor = torch.tensor([int(row["winner"])])
        logits = torch.nn.functional.linear(x, head["head.weight"]).squeeze(-1)
        anchor_mask = torch.arange(logits.shape[1])[None] == anchor[:, None]
        logits = torch.where(anchor_mask, 0.0, logits)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("fallback prediction nonfinite")
        records.append({"query_id": qid, "axis": list(data.common[qid]["axis"]),
                        "winner": int(anchor[0]), "logits": logits[0].tolist()})
    return records


def gradient_measure(model):
    total = 0.
    for param in model.reader.parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                raise FloatingPointError("nonfinite reader gradient")
            total += float(param.grad.square().sum())
    return {"reader_norm": math.sqrt(total),
            "output_norm": float(model.reader.output.weight.grad.norm()),
            "stem_norm": float(model.reader.encoder.query_stem[0].weight.grad.norm())}


def record_gradient(summary, stage, values):
    bucket = summary.setdefault(stage, {"updates": 0, "nonzero_reader_updates": 0,
                                       "nonzero_stem_updates": 0, "max_reader_norm": 0.,
                                       "max_stem_norm": 0., "first_updates": []})
    bucket["updates"] += 1
    bucket["nonzero_reader_updates"] += int(values["reader_norm"] > 0.)
    bucket["nonzero_stem_updates"] += int(values["stem_norm"] > 0.)
    bucket["max_reader_norm"] = max(bucket["max_reader_norm"], values["reader_norm"])
    bucket["max_stem_norm"] = max(bucket["max_stem_norm"], values["stem_norm"])
    if len(bucket["first_updates"]) < 5:
        bucket["first_updates"].append(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--fold", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--arm", required=True, choices=("TOKEN_QR", "TOKEN_QRR_ANCHOR", "TOKEN_QRR_MULTI"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--wall-seconds", type=float, default=480.)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-output", type=Path)
    args = ap.parse_args()
    if args.wall_seconds <= 0 or args.threads < 1:
        raise ValueError("positive thread count and wall budget required")
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = read(args.protocol)
    verify_panel(protocol)
    verify_sources(protocol)
    if not args.pilot:
        verify_evidence_gate(protocol)
    if args.pilot:
        protocol = {**protocol, "evidence_manifest": str(args.protocol.resolve().parent / "pilot_manifest.json")}
        pilot_manifest = read(protocol["evidence_manifest"])
        if pilot_manifest.get("status") != "TOKEN_COMPETITION_EVIDENCE_PILOT_PASS":
            raise ValueError("validated TRAIN pilot evidence manifest required")
        pilot_receipt = read(args.protocol.resolve().parent / "pilot_evidence_validation.json")
        verifier_path = RC / "programs/verify_token_competition_evidence_v2.py"
        if (pilot_receipt.get("status") != "TOKEN_COMPETITION_EVIDENCE_PILOT_PASS" or
                pilot_receipt.get("manifest") != {"path": protocol["evidence_manifest"],
                                                   "sha256": digest(protocol["evidence_manifest"])} or
                pilot_receipt.get("protocol") != {"path": str(args.protocol.resolve()), "sha256": digest(args.protocol)} or
                pilot_receipt.get("verifier") != {"path": str(verifier_path), "sha256": digest(verifier_path)} or
                pilot_receipt.get("queries") != 1 or pilot_receipt.get("candidates") != 128 or
                pilot_receipt.get("labels_read") != 0):
            raise ValueError("independent TRAIN pilot evidence receipt required")
    source_validation = read(checked(protocol["source_validation"]))
    if (source_validation.get("status") != "F128_BASELINE15_INDEPENDENT_PASS" or
            source_validation.get("protocol") != protocol["source_protocol"]):
        raise ValueError("same-F128 independent baseline seal required")
    if args.arm not in protocol["arms"] or args.seed not in protocol["seeds"]:
        raise ValueError("unregistered arm or seed")
    fold = protocol["folds"][str(args.fold)]
    config = protocol["optimizer"]
    output = Path(protocol["output_root"]) / f"fold{args.fold}" / f"seed{args.seed}" / args.arm
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    baseline = load_baselines(protocol, fold, args.fold, args.seed)
    manifest = checked(protocol["evidence_manifest"])
    binding = {"protocol": {"path": str(args.protocol.resolve()), "sha256": digest(args.protocol)},
               "runner_sha256": digest(__file__),
               "v1_runner_sha256": digest(RC / "programs/run_rebut_qr_qrr_v1.py"),
               "module_sha256": digest(RC / "src/rc_aslo_xf/token_competition_v2.py"),
               "data_adapter_sha256": digest(RC / "programs/token_competition_data_v2.py"),
               "source_runner_sha256": digest(RC / "programs/run_rebut_qr_qrr_frozenbase_v2.py"),
               "evidence_manifest": {"path": str(manifest), "sha256": digest(manifest)},
               "baseline_bindings": baseline["bindings"],
               "baseline_inner_predictions": baseline["inner_predictions_binding"],
               "fold": args.fold, "seed": args.seed, "arm": args.arm}
    result_path = output / "result.json"
    if result_path.exists() and not args.pilot:
        prior = read(result_path)
        if prior["binding"] != binding or prior["status"] != "FROZEN_BASE_RESIDUAL_LABEL_FREE_COMPLETE":
            raise ValueError("completed output binding/status changed")
        for key in ("model", "main_inner_model", "inner_best_ce_model", "inner_validation"):
            checked(prior["artifacts"][key])
        # An interruption after publishing result.json but before sealing the
        # terminal checkpoint must not masquerade as a failed/incomplete fit.
        terminal_path = output / "checkpoint.pt"
        terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
        if terminal["binding"] != binding or terminal["stage"] not in ("predict", "done"):
            raise ValueError("completed result has an inconsistent terminal checkpoint")
        if terminal["stage"] == "predict":
            final_model = torch.load(checked(prior["artifacts"]["model"]),
                                     map_location="cpu", weights_only=False)
            if (terminal["selection"] != prior["selection"] or
                    not torch.equal(terminal["normalization"], final_model["normalization"]) or
                    set(terminal["model"]) != set(final_model["model"]) or
                    any(not torch.equal(tensor, final_model["model"][key])
                        for key, tensor in terminal["model"].items())):
                raise ValueError("terminal repair would cross a model/selection mismatch")
            terminal["stage"] = "done"
            atomic_torch(terminal_path, terminal)
            atomic_json(output / "progress.json", {"status": "COMPLETE_TERMINAL_SEAL_RECOVERED",
                "binding": binding, "stage": "done", "epoch": terminal["epoch"],
                "cursor": terminal["cursor"], "updates": terminal["updates"],
                "main_epoch": terminal["main_epoch"], "best_ce_epoch": terminal["best_ce_epoch"]})
        print(json.dumps({"status": "ALREADY_COMPLETE", "result": str(result_path)}), flush=True)
        return
    pilot_qid = "H593-0091cc1b5a6f73a4ad8d88ae" if args.pilot else None
    if args.pilot and pilot_qid not in fold["inner_fit_query_ids"]:
        raise ValueError("fixed pilot query is not inner TRAIN for this fold")
    data = TokenInputs(protocol, fold, args.arm, pilot_query_id=pilot_qid)
    model, optimizer = initialize(args.arm, args.seed, config, baseline["inner_head"])
    if args.pilot:
        values, norms = [], []
        qid = pilot_qid
        x, ev, anchor, truth = data.sample(qid, baseline["inner_scale"])
        bmodel, _ = new_model("B_CAL", args.seed, config)
        bmodel.load_state_dict(baseline["inner_head"])
        with torch.no_grad():
            initial_exact = torch.equal(model(x, ev, anchor), bmodel(x, None, anchor))
        if not initial_exact:
            raise AssertionError("pilot initial logits do not preserve fitted B_CAL")
        for step in range(3):
            tick = time.perf_counter()
            optimizer.zero_grad()
            loss = shared_hit_miss_loss(model(x, ev, anchor), truth, anchor)
            if not torch.isfinite(loss):
                raise FloatingPointError("pilot loss nonfinite")
            loss.backward()
            norm = gradient_measure(model)
            optimizer.step()
            assert_head(model, baseline["inner_head"])
            norms.append(norm)
            values.append({"step": step + 1, "loss": float(loss.detach()),
                           "seconds": time.perf_counter() - tick})
        if max(g["stem_norm"] for g in norms[1:]) <= 0:
            raise AssertionError("pilot residual gradient path did not open")
        atomic_json(args.pilot_output or output / "pilot.json", {
            "status": "FROZEN_BASE_REAL_QUERY_ENGINEERING_PILOT_PASS", "scientific_result": False,
            "binding": binding, "query_id": qid, "updates": values, "gradients": norms,
            "initial_scores_bit_exact_baseline": initial_exact, "head_exact_unchanged": True,
            "frozen_head_parameters": sum(p.numel() for p in model.head.parameters()),
            "trainable_reader_parameters": sum(p.numel() for p in model.reader.parameters()),
            "input_F_standardized": False, "threads": args.threads})
        print(json.dumps({"status": "PILOT_PASS", "updates": values, "gradients": norms}), flush=True)
        return

    checkpoint_path = output / "checkpoint.pt"
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        if state["binding"] != binding:
            raise ValueError("resume input, source or baseline binding changed")
        head = baseline["inner_head"] if state["stage"] in ("inner", "validate_inner", "select") else baseline["outer_head"]
        model, optimizer = initialize(args.arm, args.seed, config, head)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        assert_head(model, head)
    else:
        # These are the ONLY additional initial forward passes; they validate
        # preservation of the already trained baseline, not another training run.
        inner_rows, inner_loss = predict(model, data, fold["inner_val_query_ids"],
                                        baseline["inner_scale"], labels=True)
        inner_exact = verify_same_predictions(inner_rows, baseline["inner_predictions"], "inner epoch0")
        outer_model, _ = initialize(args.arm, args.seed, config, baseline["outer_head"])
        outer_rows, _ = predict(outer_model, data, fold["heldout_query_ids"],
                                baseline["outer_scale"], labels=False)
        outer_exact = verify_same_predictions(outer_rows, baseline["result"]["predictions"], "outer epoch0")
        # Reset the reader RNG after the independent baseline preservation check.
        model, optimizer = initialize(args.arm, args.seed, config, baseline["inner_head"])
        base_metrics = training_metrics(inner_rows, data, inner_rows, baseline["inherited_tau"])
        event0 = {"stage": "inner", "epoch": 0, "validation_loss": inner_loss,
                  **base_metrics, "selected_best_ce": True, "selected_main": False,
                  "eligible_main": False, "train_loss": None}
        (output / "inner").mkdir(exist_ok=True)
        atomic_json(output / "inner/epoch000.json", {"binding": binding, "metrics": event0,
                                                      "epoch": 0, "validation_loss": inner_loss,
                                                      "fixed0_correct": base_metrics["fixed_zero_correct"],
                                                      "inherited_tau": baseline["inherited_tau"],
                                                      "records": inner_rows})
        state = {"binding": binding, "stage": "inner", "epoch": 0, "cursor": 0, "order": [],
                 "best_ce_epoch": 0, "best_ce_loss": inner_loss,
                 "best_ce_model": copy.deepcopy(model.state_dict()),
                 "best_ce_records": inner_rows, "main_epoch": 0, "main_loss": inner_loss,
                 "main_inner_model": copy.deepcopy(model.state_dict()), "main_records": inner_rows,
                 "baseline_records": inner_rows, "baseline_fixed0_correct": base_metrics["fixed_zero_correct"],
                 "baseline_validation_loss": inner_loss, "patience_used": 0,
                 "history": [event0], "gradient_summary": {}, "updates": 0,
                 "elapsed_compute_seconds": 0., "inner_normalization": baseline["inner_scale"],
                 "outer_normalization": baseline["outer_scale"],
                 "normalization": baseline["inner_scale"],
                 "baseline_replay": {"inner": inner_exact, "outer": outer_exact}}

    def save(status, emit=True):
        expected = baseline["inner_head"] if state["stage"] in ("inner", "validate_inner", "select") else baseline["outer_head"]
        assert_head(model, expected)
        state["model"] = model.state_dict()
        state["optimizer"] = optimizer.state_dict()
        state["rng"] = rng_state()
        atomic_torch(checkpoint_path, state)
        public = {"status": status, "binding": binding, "stage": state["stage"],
                  "epoch": state["epoch"], "cursor": state["cursor"], "updates": state["updates"],
                  "best_ce_epoch": state["best_ce_epoch"], "main_epoch": state["main_epoch"],
                  "patience_used": state["patience_used"],
                  "elapsed_compute_seconds": state["elapsed_compute_seconds"],
                  "prediction_cursor": len(state.get("pending_prediction", {}).get("records", [])),
                  "prediction_total": len(state.get("pending_prediction", {}).get("query_ids", []))}
        atomic_json(output / "progress.json", public)
        if emit:
            print(json.dumps(public), flush=True)

    def predict_resumable(ids, labels, key):
        pending = state.setdefault("pending_prediction", {
            "key": key, "query_ids": list(ids), "labels": labels, "records": [], "losses": []})
        if (pending["key"] != key or pending["query_ids"] != list(ids) or pending["labels"] != labels or
                [r["query_id"] for r in pending["records"]] != list(ids[:len(pending["records"])])):
            raise ValueError("resumed prediction cursor differs from its frozen stage")
        for qid in ids[len(pending["records"]):]:
            if time.monotonic() - started >= args.wall_seconds:
                save("RESUMABLE_PREDICTION_WALL_BUDGET")
                sys.exit(75)
            rows, loss = predict(model, data, [qid], state["normalization"], labels=labels)
            pending["records"].extend(rows)
            if labels:
                pending["losses"].append(loss)
            # A timed-out next forward cannot discard this completed query or
            # roll back training updates from the completed inner epoch.
            save("INNER_VALIDATION_PROGRESS" if labels else "LABEL_FREE_PREDICTION_PROGRESS", emit=False)
        rows, losses = pending["records"], pending["losses"]
        del state["pending_prediction"]
        return rows, sum(losses) / len(losses) if labels else None

    while state["stage"] != "done":
        if time.monotonic() - started >= args.wall_seconds:
            save("RESUMABLE_WALL_BUDGET")
            sys.exit(75)
        stage = state["stage"]
        if stage in ("inner", "refit"):
            ids = fold["inner_fit_query_ids"] if stage == "inner" else fold["train_query_ids"]
            limit = config["max_epochs"] if stage == "inner" else state["main_epoch"]
            if state["epoch"] >= limit:
                state["stage"] = "select" if stage == "inner" else "predict"
                save("STAGE_COMPLETE")
                continue
            if not state["order"]:
                state["order"] = list(ids)
                random.shuffle(state["order"])
                state["cursor"] = 0
                state["epoch_loss_sum"] = 0.
            model.train()
            while state["cursor"] < len(state["order"]):
                tick = time.perf_counter()
                qid = state["order"][state["cursor"]]
                x, ev, anchor, truth = data.sample(qid, state["normalization"])
                optimizer.zero_grad()
                loss = shared_hit_miss_loss(model(x, ev, anchor), truth, anchor)
                if not torch.isfinite(loss):
                    raise FloatingPointError("training loss nonfinite")
                loss.backward()
                gradients = gradient_measure(model)
                optimizer.step()
                expected = baseline["inner_head"] if stage == "inner" else baseline["outer_head"]
                assert_head(model, expected)
                record_gradient(state["gradient_summary"], stage, gradients)
                state["epoch_loss_sum"] += float(loss.detach())
                state["cursor"] += 1
                state["updates"] += 1
                state["elapsed_compute_seconds"] += time.perf_counter() - tick
                if time.monotonic() - started >= args.wall_seconds:
                    save("RESUMABLE_WALL_BUDGET")
                    sys.exit(75)
            state["epoch"] += 1
            event = {"stage": stage, "epoch": state["epoch"],
                     "train_loss": state["epoch_loss_sum"] / len(state["order"])}
            if stage == "inner":
                state["pending_train_event"] = event
                state["stage"] = "validate_inner"
                save("TRAIN_EPOCH_COMPLETE_VALIDATION_PENDING")
                continue
            state["history"].append(event)
            state["order"], state["cursor"] = [], 0
            save("EPOCH_COMPLETE")
        elif stage == "validate_inner":
            event = state["pending_train_event"]
            rows, vl = predict_resumable(fold["inner_val_query_ids"], True,
                                        f"inner_validation_epoch{state['epoch']}")
            metrics = training_metrics(rows, data, state["baseline_records"], baseline["inherited_tau"])
            better_ce = vl < state["best_ce_loss"]
            eligible = metrics["fixed_zero_correct"] > state["baseline_fixed0_correct"]
            better_main = eligible and (state["main_epoch"] == 0 or vl < state["main_loss"])
            event.update({"validation_loss": vl, **metrics, "eligible_main": eligible,
                          "selected_best_ce": better_ce, "selected_main": better_main})
            if better_ce:
                state["best_ce_epoch"], state["best_ce_loss"] = state["epoch"], vl
                state["best_ce_model"] = copy.deepcopy(model.state_dict())
                state["best_ce_records"] = rows
                state["patience_used"] = 0
            else:
                state["patience_used"] += 1
            if better_main:
                state["main_epoch"], state["main_loss"] = state["epoch"], vl
                state["main_inner_model"] = copy.deepcopy(model.state_dict())
                state["main_records"] = rows
            atomic_json(output / f"inner/epoch{state['epoch']:03d}.json",
                        {"binding": binding, "metrics": event, "epoch": state["epoch"],
                         "validation_loss": vl, "fixed0_correct": metrics["fixed_zero_correct"],
                         "inherited_tau": baseline["inherited_tau"], "records": rows})
            state["stage"] = "select" if state["patience_used"] >= config["patience"] else "inner"
            state["history"].append(event)
            del state["pending_train_event"]
            state["order"], state["cursor"] = [], 0
            save("EPOCH_COMPLETE")
        elif stage == "select":
            selection = {"baseline_fixed0_correct": state["baseline_fixed0_correct"],
                         "baseline_validation_loss": state["baseline_validation_loss"],
                         "main_epoch": state["main_epoch"], "main_validation_loss": state["main_loss"],
                         "best_ce_epoch": state["best_ce_epoch"], "best_ce_validation_loss": state["best_ce_loss"],
                         "epoch0_fallback": state["main_epoch"] == 0,
                         "policy": "Eligible epochs strictly improve inner fixed0 correct over frozen B_CAL; "
                                   "minimum validation CE among them, ties earliest; none => epoch0. "
                                   "Early stopping follows CE including epoch0; no threshold fitting."}
            state["selection"] = selection
            atomic_torch(output / "main_inner_model.pt", {
                "model": state["main_inner_model"], "normalization": state["inner_normalization"],
                "epoch": state["main_epoch"], "binding": binding, "head_frozen": True})
            atomic_torch(output / "inner_best_ce_model.pt", {
                "model": state["best_ce_model"], "normalization": state["inner_normalization"],
                "epoch": state["best_ce_epoch"], "binding": binding, "diagnostic_only": True})
            atomic_json(output / "inner_validation.json", {
                "binding": binding, "selection": selection,
                "normalization_rms": state["inner_normalization"].tolist(),
                "baseline_records": state["baseline_records"], "records": state["main_records"],
                "best_ce_records": state["best_ce_records"], "threshold": 0.0,
                "inherited_tau_secondary": baseline["inherited_tau"]})
            model, optimizer = initialize(args.arm, args.seed, config, baseline["outer_head"])
            state["normalization"] = baseline["outer_scale"]
            state["stage"], state["epoch"], state["cursor"], state["order"] = "refit", 0, 0, []
            save("REFIT_READER_FROM_SAME_SEED_FROZEN_OUTER_BASE")
        elif stage == "predict":
            if state["main_epoch"] == 0:
                rows = predict_frozen_head_only(data, fold["heldout_query_ids"],
                                               state["normalization"], baseline["outer_head"])
                state["fallback_replay"] = verify_same_predictions(
                    rows, baseline["result"]["predictions"], "epoch0 final fallback")
                if state["gradient_summary"].get("refit", {}).get("updates", 0) != 0:
                    raise AssertionError("fallback unexpectedly refitted the reader")
            else:
                rows, _ = predict_resumable(fold["heldout_query_ids"], False, "outer_held")
            assert_head(model, baseline["outer_head"])
            rows = attach_decisions(rows, baseline["inherited_tau"])
            atomic_torch(output / "model.pt", {"model": model.state_dict(),
                                               "normalization": state["normalization"],
                                               "binding": binding, "head_frozen": True,
                                               "reader_enabled": state["main_epoch"] > 0})
            artifact_paths = {"model": output / "model.pt", "main_inner_model": output / "main_inner_model.pt",
                              "inner_best_ce_model": output / "inner_best_ce_model.pt",
                              "inner_validation": output / "inner_validation.json"}
            payload = {"status": "FROZEN_BASE_RESIDUAL_LABEL_FREE_COMPLETE", "binding": binding,
                       "baseline_bindings": baseline["bindings"],
                       "panel": "F128 developmental original-fold intersection; native token competition V2 seed0",
                       "population_queries": 128, "development_analysis": True,
                       "token_axis": "native_query_tokens_unpooled",
                       "train_query_ids": fold["train_query_ids"],
                       "inner_fit_query_ids": fold["inner_fit_query_ids"],
                       "inner_val_query_ids": fold["inner_val_query_ids"],
                       "heldout_query_ids": fold["heldout_query_ids"],
                       "selection": state["selection"], "best_epoch": state["main_epoch"],
                       "history": state["history"], "gradient_summary": state["gradient_summary"],
                       "calibration": {"threshold": 0.0, "threshold_hex": 0.0.hex(),
                                       "inherited_tau_secondary": baseline["inherited_tau"],
                                       "new_threshold_fitting": False},
                       "normalization_rms": state["normalization"].tolist(),
                       "normalization_protocol": "Exactly frozen B_CAL RMS per inner/outer stage; F unchanged",
                       "input_F_standardized": False,
                       "reader_enabled": state["main_epoch"] > 0,
                       "baseline_replay": state["baseline_replay"],
                       "fallback_replay": state.get("fallback_replay"),
                       "head_exact_unchanged": True,
                       "frozen_head_parameters": sum(p.numel() for p in model.head.parameters()),
                       "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                       "elapsed_compute_seconds": state["elapsed_compute_seconds"],
                       "timing_scope": "training updates only; excludes IO/validation/baseline replay",
                       "model_sha256": digest(output / "model.pt"),
                       "artifacts": {k: {"path": str(p), "sha256": digest(p)} for k, p in artifact_paths.items()},
                       "evidence_bindings": data.bindings, "predictions": rows,
                       "outer_labels_opened": False,
                       "best_ce_model_held_evaluated": False}
            atomic_json(result_path, payload)
            state["stage"] = "done"
            save("COMPLETE")
        else:
            raise ValueError(f"unrecognized stage: {stage}")


if __name__ == "__main__":
    main()
