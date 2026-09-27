#!/usr/bin/env python3
"""Resume-aware CPU fitting of one frozen fold/seed/arm from shared evidence.

Only training-role labels are opened here. Outer held predictions are label-free;
an independent collector joins their labels after every predeclared fit completes.
Inner normalization, epoch and threshold selection never use held queries. This
runner is for the explicitly frozen F71 prototype, not a silent H593 replacement.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC / "src"))
from rc_aslo_xf.rebut_qr_qrr_v1 import (EvidenceBatch, RebutScorer,
    fixed_anchor_decision, parameter_report, shared_hit_miss_loss)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(binding):
    path = Path(binding["path"] if isinstance(binding, dict) else binding)
    if isinstance(binding, dict) and binding.get("sha256") != digest(path):
        raise ValueError(f"source hash mismatch: {path}")
    return path


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_torch(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


class FrozenInputs:
    def __init__(self, protocol, fold, arm, pilot_query_id=None):
        self.protocol, self.fold, self.arm = protocol, fold, arm
        common = torch.load(checked(protocol["common_features"]), map_location="cpu", weights_only=False)
        self.common = {r["query_id"]: r for r in common["records"]}
        gallery = read(checked(protocol["gallery"]))
        self.identity = {int(r["physical_row"]): r["identity"] for r in gallery["records"]}
        roles_payload = read(checked(fold["train_roles"]))
        self.roles = {r["query_id"]: r for r in roles_payload["records"]}
        train = set(fold["train_query_ids"])
        held = set(fold["heldout_query_ids"])
        inner_fit, inner_val = set(fold["inner_fit_query_ids"]), set(fold["inner_val_query_ids"])
        if train & held or inner_fit & inner_val or inner_fit | inner_val != train:
            raise ValueError("nested query split is not disjoint/exhaustive")
        if not train <= set(self.roles) or held & set(self.roles):
            raise ValueError("training label file contains held queries or lacks training labels")
        if not train or not held or not inner_fit or not inner_val:
            raise ValueError("empty nested split")
        if {self.roles[q]["component"] for q in inner_fit} & {self.roles[q]["component"] for q in inner_val}:
            raise ValueError("inner component leakage")
        if {self.roles[q]["identity"] for q in inner_fit} & {self.roles[q]["identity"] for q in inner_val}:
            raise ValueError("inner query identity leakage")
        self.query_ids = sorted(train | held)
        self.evidence = {}
        self.bindings = {}
        if arm != "B_CAL":
            manifest_path = checked(protocol["evidence_manifest"])
            manifest = read(manifest_path)
            if pilot_query_id is None and manifest.get("status") != "POOLED_F71_COMPLETE":
                raise ValueError("formal fit requires the complete shared F71 evidence cache")
            records = manifest.get("records", manifest.get("queries"))
            if not isinstance(records, list):
                raise ValueError("evidence manifest needs records list")
            by_qid = {r["query_id"]: r for r in records}
            if len(by_qid) != len(records):
                raise ValueError("duplicate query in shared evidence manifest")
            if pilot_query_id is None and (len(records) != 71 or
                    set(by_qid) != set(protocol["panel_query_ids"])):
                raise ValueError("complete F71 manifest does not match the frozen panel")
            for qid in ([pilot_query_id] if pilot_query_id is not None else self.query_ids):
                entry = by_qid[qid]
                binding = entry.get("evidence", entry.get("payload", entry))
                ep = checked(binding)
                data = torch.load(ep, map_location="cpu", weights_only=False)
                common_row = self.common[qid]
                if (data["query_id"] != qid or list(data["axis"]) != list(common_row["axis"]) or
                        int(data["winner"]) != int(common_row["winner"])):
                    raise ValueError("evidence/common candidate axis mismatch")
                ev = EvidenceBatch(data["query"].to(torch.float64).unsqueeze(0),
                                   data["reference"].to(torch.float64).unsqueeze(0),
                                   data["query_valid"].to(torch.bool).unsqueeze(0),
                                   data["reference_valid"].to(torch.bool).unsqueeze(0))
                ev.validate()
                if ev.query.shape != (1, 128, 64, 72) or ev.reference.shape != (1, 128, 64, 72):
                    raise ValueError("F71 shape differs from predeclared [128,64,72]")
                self.evidence[qid] = ev
                self.bindings[qid] = {"path": str(ep), "sha256": digest(ep)}
        for qid in self.query_ids:
            row = self.common[qid]
            x = torch.as_tensor(row["X0"], dtype=torch.float64)
            if x.shape != (128, 18) or not torch.isfinite(x).all():
                raise ValueError("X0 must be finite [128,18]")
            if len(set(row["axis"])) != 128 or not 0 <= int(row["winner"]) < 128:
                raise ValueError("invalid natural physical candidate axis")

    def normalization(self, query_ids):
        # X0 contains its own constant bias. Do not center; RMS rescales usable
        # continuous dimensions, preserves exact-zero HOLD and constant column.
        if not set(query_ids) <= set(self.fold["train_query_ids"]):
            raise ValueError("normalization attempted outside outer TRAIN")
        chunks = []
        for qid in query_ids:
            row = self.common[qid]
            mask = torch.arange(128) != int(row["winner"])
            chunks.append(torch.as_tensor(row["X0"], dtype=torch.float64)[mask])
        rms = torch.cat(chunks).square().mean(0).sqrt()
        return torch.where(rms > 1e-12, rms, torch.ones_like(rms))

    def sample(self, qid, scale, labels=True):
        row = self.common[qid]
        x = torch.as_tensor(row["X0"], dtype=torch.float64).unsqueeze(0) / scale
        anchor = torch.tensor([int(row["winner"])])
        truth = None
        if labels:
            if qid not in self.fold["train_query_ids"]:
                raise ValueError("outer held label requested by trainer")
            ident = self.roles[qid]["identity"]
            truth = torch.tensor([[self.identity[int(p)] == ident for p in row["axis"]]])
        return x, self.evidence.get(qid), anchor, truth


def new_model(arm, seed, optimizer_config):
    setup_seed(seed)
    model = RebutScorer(arm, 18, 72, 72, head_bias=False).to(torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config["lr"],
                                  weight_decay=optimizer_config["weight_decay"])
    return model, optimizer


@torch.no_grad()
def predict(model, data, ids, scale, labels=False):
    model.eval()
    records, losses = [], []
    for qid in ids:
        x, ev, anchor, truth = data.sample(qid, scale, labels)
        logits = model(x, ev, anchor)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite prediction")
        row = data.common[qid]
        records.append({"query_id": qid, "axis": list(row["axis"]),
                        "winner": int(anchor[0]), "logits": logits[0].tolist()})
        if labels:
            losses.append(float(shared_hit_miss_loss(logits, truth, anchor)))
    return records, None if not labels else sum(losses) / len(losses)


def select_threshold(records, data):
    """Inner validation only; maximize correct minus RAW, ties choose larger tau."""
    tops = []
    for row in records:
        if row["query_id"] not in data.fold["inner_val_query_ids"]:
            raise ValueError("threshold selection outside inner validation")
        z = list(row["logits"])
        z.pop(row["winner"])
        tops.append(max(z))
    finite = sorted(set(tops))
    candidates = [math.nextafter(finite[0], -math.inf)] + finite
    best = None
    for tau in candidates:
        rescues = breaks = wrong_to_wrong = switches = correct = raw_correct = 0
        for row in records:
            qid, axis, anchor = row["query_id"], row["axis"], row["winner"]
            z = torch.tensor([row["logits"]], dtype=torch.float64)
            pos = int(fixed_anchor_decision(z, torch.tensor([anchor]), tau)[0])
            target = data.roles[qid]["identity"]
            old = data.identity[axis[anchor]] == target
            new = data.identity[axis[pos]] == target
            rescues += int(new and not old)
            breaks += int(old and not new)
            switches += int(pos != anchor)
            wrong_to_wrong += int(not old and not new and pos != anchor)
            correct += int(new)
            raw_correct += int(old)
        key = (rescues - breaks, tau)
        if best is None or key > best[0]:
            best = (key, {"threshold": tau, "threshold_hex": tau.hex(),
                          "criterion": "inner_val_net_gain_then_largest_threshold",
                          "rescues": rescues, "breaks": breaks,
                          "correct": correct, "raw_correct": raw_correct,
                          "switches": switches, "wrong_to_wrong": wrong_to_wrong,
                          "queries": len(records)})
    return best[1]


def attach_decisions(records, threshold):
    out = []
    for row in records:
        z = torch.tensor([row["logits"]], dtype=torch.float64)
        anchor = torch.tensor([row["winner"]])
        selected = int(fixed_anchor_decision(z, anchor, threshold)[0])
        fixed = int(fixed_anchor_decision(z, anchor, 0.)[0])
        out.append({**row, "logits_hex": [float(v).hex() for v in row["logits"]],
                    "selected_position": selected, "selected_physical_row": row["axis"][selected],
                    "selected_fixed_zero_position": fixed,
                    "selected_fixed_zero_physical_row": row["axis"][fixed],
                    "action": "HOLD" if selected == row["winner"] else "SWITCH"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", choices=("B_CAL", "QR", "QR_VEC", "QRR"), required=True)
    ap.add_argument("--wall-seconds", type=float, default=480.)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pilot", action="store_true", help="three real TRAIN updates only; no claimed fit")
    args = ap.parse_args()
    started = time.monotonic()
    if args.wall_seconds <= 0:
        raise ValueError("wall-seconds must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = read(args.protocol)
    if args.seed not in protocol["seeds"] or args.arm not in protocol["arms"]:
        raise ValueError("unregistered arm or seed")
    fold = protocol["folds"][str(args.fold)]
    config = protocol["optimizer"]
    root = Path(protocol["output_root"])
    output = root / f"fold{args.fold}" / f"seed{args.seed}" / args.arm
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = {"protocol": {"path": str(args.protocol.resolve()), "sha256": digest(args.protocol)},
               "runner_sha256": digest(__file__),
               "module_sha256": digest(RC / "src/rc_aslo_xf/rebut_qr_qrr_v1.py"),
               "fold": args.fold, "seed": args.seed, "arm": args.arm}
    if args.arm != "B_CAL":
        manifest_path = checked(protocol["evidence_manifest"])
        binding["evidence_manifest"] = {"path": str(manifest_path), "sha256": digest(manifest_path)}
    result_path = output / "result.json"
    if result_path.exists() and not args.pilot:
        prior = read(result_path)
        if prior["binding"] != binding:
            raise ValueError("completed result source binding changed")
        print(json.dumps({"status": "ALREADY_COMPLETE", "result": str(result_path)}), flush=True)
        return
    pilot_query_id = fold["inner_fit_query_ids"][0] if args.pilot else None
    data = FrozenInputs(protocol, fold, args.arm, pilot_query_id)
    model, optimizer = new_model(args.arm, args.seed, config)
    if args.pilot:
        scale = data.normalization(fold["inner_fit_query_ids"])
        qid = fold["inner_fit_query_ids"][0]
        values, gradnorm = [], []
        for step in range(3):
            tick = time.perf_counter()
            x, ev, anchor, truth = data.sample(qid, scale)
            optimizer.zero_grad()
            loss = shared_hit_miss_loss(model(x, ev, anchor), truth, anchor)
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("pilot gradient nonfinite")
            if model.reader is not None:
                gradnorm.append(float(model.reader.encoder.query_stem[0].weight.grad.norm()))
            optimizer.step()
            values.append({"step": step + 1, "loss": float(loss.detach()),
                           "seconds": time.perf_counter() - tick})
        if model.reader is not None and max(gradnorm[1:]) <= 0:
            raise ValueError("reader gradient path did not open")
        payload = {"status": "REAL_QUERY_ENGINEERING_PILOT_PASS", "scientific_result": False,
                   "query_id": qid, "binding": binding, "updates": values,
                   "reader_stem_gradient_norm": gradnorm,
                   "trainable_parameters": sum(p.numel() for p in model.parameters()),
                   "threads": args.threads, "evidence_shape": None if ev is None else list(ev.query.shape)}
        atomic_json(output / "pilot.json", payload)
        print(json.dumps(payload), flush=True)
        return

    checkpoint = output / "checkpoint.pt"
    state = {"binding": binding, "stage": "inner", "epoch": 0, "cursor": 0,
             "order": [], "best_epoch": 0, "best_validation_loss": math.inf,
             "patience_used": 0, "history": [], "updates": 0,
             "elapsed_compute_seconds": 0., "normalization": data.normalization(fold["inner_fit_query_ids"])}
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["binding"] != binding:
            raise ValueError("resume source binding changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])

    def save(status):
        state["model"] = model.state_dict()
        state["optimizer"] = optimizer.state_dict()
        state["rng"] = rng_state()
        atomic_torch(checkpoint, state)
        public = {"status": status, "stage": state["stage"], "epoch": state["epoch"],
                  "cursor": state["cursor"], "updates": state["updates"],
                  "best_epoch": state["best_epoch"], "patience_used": state["patience_used"],
                  "elapsed_compute_seconds": state["elapsed_compute_seconds"], "binding": binding}
        atomic_json(output / "progress.json", public)
        print(json.dumps(public), flush=True)

    while state["stage"] != "done":
        # Stop only at completed query updates or epoch/evaluation boundaries.
        if time.monotonic() - started >= args.wall_seconds:
            save("RESUMABLE_WALL_BUDGET")
            sys.exit(75)
        stage = state["stage"]
        if stage in ("inner", "refit"):
            ids = (fold["inner_fit_query_ids"] if stage == "inner" else fold["train_query_ids"])
            limit = config["max_epochs"] if stage == "inner" else state["best_epoch"]
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
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("nonfinite training gradient")
                optimizer.step()
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
                _, vl = predict(model, data, fold["inner_val_query_ids"], state["normalization"], labels=True)
                event["validation_loss"] = vl
                if vl < state["best_validation_loss"]:
                    state["best_validation_loss"] = vl
                    state["best_epoch"] = state["epoch"]
                    state["best_model"] = copy.deepcopy(model.state_dict())
                    state["patience_used"] = 0
                else:
                    state["patience_used"] += 1
                if state["patience_used"] >= config["patience"]:
                    state["stage"] = "select"
            state["history"].append(event)
            state["order"] = []
            state["cursor"] = 0
            save("EPOCH_COMPLETE")
        elif stage == "select":
            model.load_state_dict(state["best_model"])
            rows, _ = predict(model, data, fold["inner_val_query_ids"], state["normalization"], labels=True)
            state["calibration"] = select_threshold(rows, data)
            state["inner_normalization"] = state["normalization"].clone()
            atomic_json(output / "inner_validation.json", {
                "binding": binding, "best_epoch": state["best_epoch"],
                "normalization_rms": state["inner_normalization"].tolist(),
                "calibration": state["calibration"], "records": rows})
            model, optimizer = new_model(args.arm, args.seed, config)
            state["normalization"] = data.normalization(fold["train_query_ids"])
            state["stage"], state["epoch"], state["cursor"], state["order"] = "refit", 0, 0, []
            save("REFIT_FROM_SAME_SEED")
        elif stage == "predict":
            rows, _ = predict(model, data, fold["heldout_query_ids"], state["normalization"], labels=False)
            rows = attach_decisions(rows, state["calibration"]["threshold"])
            atomic_torch(output / "model.pt", {"model": model.state_dict(),
                                               "normalization": state["normalization"], "binding": binding})
            payload = {"status": "TRAINING_AND_LABEL_FREE_PREDICTIONS_COMPLETE", "binding": binding,
                       "panel": "F71 developmental original-fold intersection",
                       "train_query_ids": fold["train_query_ids"],
                       "inner_fit_query_ids": fold["inner_fit_query_ids"],
                       "inner_val_query_ids": fold["inner_val_query_ids"],
                       "heldout_query_ids": fold["heldout_query_ids"],
                       "best_epoch": state["best_epoch"], "history": state["history"],
                       "calibration": state["calibration"],
                       "normalization_rms": state["normalization"].tolist(),
                       "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                       "elapsed_compute_seconds": state["elapsed_compute_seconds"],
                       "timing_scope": "training update compute only; excludes cache IO and validation",
                       "normalization_protocol": "X0 RMS, no centering; inner fit only for selection, "
                           "all outer TRAIN for same-seed refit; cached F evidence unstandardized",
                       "model_sha256": digest(output / "model.pt"),
                       "evidence_bindings": data.bindings, "predictions": rows,
                       "outer_labels_opened": False}
            atomic_json(result_path, payload)
            state["stage"] = "done"
            save("COMPLETE")
        else:
            raise ValueError(f"unknown checkpoint stage {stage}")


if __name__ == "__main__":
    main()
