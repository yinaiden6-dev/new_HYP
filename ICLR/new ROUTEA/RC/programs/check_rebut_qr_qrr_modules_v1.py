#!/usr/bin/env python3
"""Meaningful CPU engineering checks; synthetic inputs, NOT a research result."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import torch

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC / "src"))
from rc_aslo_xf.rebut_qr_qrr_v1 import (EvidenceBatch, RebutScorer,
    fixed_anchor_decision, parameter_report, shared_hit_miss_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=RC / "results/rc_rebut_qr_qrr_v1/engineering_checks.json")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(91)
    dtype = torch.float64
    b, c, nq, nr, d = 4, 5, 7, 6, 72
    q, r = torch.randn(b, c, nq, d, dtype=dtype), torch.randn(b, c, nr, d, dtype=dtype)
    qm, rm = torch.rand(b, c, nq) > .2, torch.rand(b, c, nr) > .2
    qm[0, 1] = False
    rm[0, 1] = False
    evidence = EvidenceBatch(q, r, qm, rm)
    evidence.validate()
    existing = torch.randn(b, c, 18, dtype=dtype)
    anchor = torch.tensor([0, 2, 4, 1])
    truth = torch.zeros(b, c, dtype=torch.bool)
    truth[0, 3] = True
    truth[1, 2] = True
    truth[2, [1, 3]] = True  # two physical refs, one correct identity
    # Query 3 absent: no fake correct anchor.
    initial = torch.linspace(-.3, .3, 18, dtype=dtype)
    models = {arm: RebutScorer(arm, 18, d, d, initial_weight=initial,
                               head_bias=False).to(dtype=dtype)
              for arm in ("B_CAL", "QR", "QR_VEC", "QRR")}
    checks, detail = {}, {}
    baseline = models["B_CAL"](existing, None, anchor)
    for arm, model in models.items():
        z = model(existing, evidence, anchor)
        assert torch.equal(z, baseline), arm
    checks["zero_residual_exact_B_CAL"] = True
    report = parameter_report(models)
    assert report["all_under_250k"] and report["reader_arms_within_10_percent_of_mean"]
    checks["active_parameter_budget_and_10_percent"] = True

    loss_each = shared_hit_miss_loss(baseline, truth, anchor, reduction="none")
    expected_hit = -(baseline[2].softmax(-1)[truth[2]].sum()).log()
    expected_miss = torch.nn.functional.softplus(baseline[3, torch.arange(c) != anchor[3]]).mean()
    torch.testing.assert_close(loss_each[2], expected_hit)
    torch.testing.assert_close(loss_each[3], expected_miss)
    checks["identity_probability_sum_and_absent_softplus"] = True
    absent_logits = torch.tensor([[0., 2., -1.]], dtype=dtype, requires_grad=True)
    loss = shared_hit_miss_loss(absent_logits, torch.zeros(1, 3, dtype=torch.bool), torch.tensor([0]))
    loss.backward()
    assert absent_logits.grad[0, 0] == 0 and (absent_logits.grad[0, 1:] > 0).all()
    checks["absent_gradient_rejects_challengers_without_HOLD_label"] = True

    # Run actual optimization: upstream gradients open after zero-output first step.
    for arm in ("QR", "QR_VEC", "QRR"):
        model = models[arm]
        optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=0)
        losses, reader_grad, stem_grad = [], [], []
        for step in range(35):
            optimizer.zero_grad()
            loss = shared_hit_miss_loss(model(existing, evidence, anchor), truth, anchor)
            losses.append(float(loss.detach()))
            loss.backward()
            reader_grad.append(float(model.reader.output.weight.grad.abs().sum()))
            stem_grad.append(float(model.reader.encoder.query_stem[0].weight.grad.abs().sum()))
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
        assert reader_grad[0] > 0 and any(v > 0 for v in stem_grad[1:])
        assert losses[-1] < .5 * losses[0], (arm, losses[0], losses[-1])
        detail[arm] = {"first_loss": losses[0], "last_loss": losses[-1],
                       "first_output_gradient": reader_grad[0],
                       "max_stem_gradient_after_first_update": max(stem_grad[1:])}
        enc = model.reader.encode(evidence)
        left = torch.arange(c)[None].expand(b, -1)
        right = left.roll(1, 1)
        ab = model.reader.pair_difference(enc, left, right)
        ba = model.reader.pair_difference(enc, right, left)
        assert torch.equal(ab, -ba), arm
        assert torch.count_nonzero(model.reader.pair_difference(enc, left, left)) == 0
        checks[arm + "_antisymmetric_exact_and_self_zero"] = True
        # Candidate permutation preserves final physical IDs and score axis.
        perm = torch.tensor([3, 0, 4, 1, 2])
        inv = torch.argsort(perm)
        ep = EvidenceBatch(q[:, perm], r[:, perm], qm[:, perm], rm[:, perm])
        original = model(existing, evidence, anchor, 2)
        changed = model(existing[:, perm], ep, inv[anchor], 2)
        torch.testing.assert_close(changed, original[:, perm], rtol=1e-12, atol=1e-12)
        chosen = fixed_anchor_decision(original, anchor)
        chosen_p = perm[fixed_anchor_decision(changed, inv[anchor])]
        assert torch.equal(chosen, chosen_p)
        checks[arm + "_candidate_permutation"] = True
        torch.testing.assert_close(model(existing, evidence, anchor, 1),
                                   model(existing, evidence, anchor, c), rtol=1e-12, atol=1e-12)
        checks[arm + "_chunk_invariant"] = True
        # Invalid padding cannot change scores, even if storage contains NaNs.
        qpad, rpad = q.clone(), r.clone()
        qpad[~qm], rpad[~rm] = torch.nan, torch.nan
        epad = EvidenceBatch(qpad, rpad, qm, rm)
        epad.validate()
        torch.testing.assert_close(model(existing, epad, anchor), original,
                                   rtol=1e-12, atol=1e-12)
        checks[arm + "_masked_padding_invariant"] = True
        # Native reference grids may be permuted independently: only pooling.
        rp = torch.tensor([4, 2, 0, 5, 1, 3])
        erp = EvidenceBatch(q, r[:, :, rp], qm, rm[:, :, rp])
        torch.testing.assert_close(model(existing, erp, anchor), original,
                                   rtol=1e-12, atol=1e-12)
        checks[arm + "_reference_native_grid_not_index_compared"] = True
        if arm == "QR":
            score = model.reader.independent_score(enc)
            qalt, ralt = q.clone(), r.clone()
            qalt[:, 1:] *= 100
            ralt[:, 1:] *= -100
            alt = model.reader.independent_score(model.reader.encode(EvidenceBatch(qalt, ralt, qm, rm)))
            assert torch.equal(score[:, 0], alt[:, 0])
            checks["QR_independent_of_rival"] = True
    checks["finite_gradients_reader_learns_loss_decreases"] = True

    tie_logits = torch.tensor([[0., 1., 1.], [0., .2, -.2], [0., -1., -2.]], dtype=dtype)
    chosen = fixed_anchor_decision(tie_logits, torch.zeros(3, dtype=torch.long), threshold=.2)
    assert torch.equal(chosen, torch.zeros(3, dtype=torch.long))
    checks["ties_and_threshold_equal_hold"] = True

    # Frozen backbone stand-in is not part of any optimizer and remains identical.
    assert not q.requires_grad and not r.requires_grad
    assert torch.equal(q, evidence.query) and torch.equal(r, evidence.reference)
    checks["frozen_input_tensors_unchanged"] = True

    # The actual runner's serialized optimizer and RNG must reproduce updates.
    # Capture after one complete query update; shuffled subsequent examples and
    # AdamW momentum are both significant, so weights alone cannot pass this.
    from run_rebut_qr_qrr_v1 import (atomic_torch, new_model, restore_rng,
                                    rng_state, select_threshold)
    config = {"lr": .0003, "weight_decay": .0001}
    uninterrupted, opt = new_model("QRR", 13, config)
    def update_one(net, optimizer, index):
        ev = EvidenceBatch(q[index:index+1], r[index:index+1],
                           qm[index:index+1], rm[index:index+1])
        optimizer.zero_grad()
        loss = shared_hit_miss_loss(net(existing[index:index+1], ev, anchor[index:index+1]),
                                    truth[index:index+1], anchor[index:index+1])
        loss.backward()
        optimizer.step()
    update_one(uninterrupted, opt, 0)
    with tempfile.TemporaryDirectory(prefix="rebut-resume-check-") as temporary:
        cp = Path(temporary) / "checkpoint.pt"
        atomic_torch(cp, {"model": uninterrupted.state_dict(), "optimizer": opt.state_dict(),
                          "rng": rng_state()})
        order = list(range(b))
        random.shuffle(order)
        for i in order:
            update_one(uninterrupted, opt, i)
        resumed, opt2 = new_model("QRR", 13, config)
        saved = torch.load(cp, weights_only=False)
        resumed.load_state_dict(saved["model"])
        opt2.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        order2 = list(range(b))
        random.shuffle(order2)
        assert order == order2
        for i in order2:
            update_one(resumed, opt2, i)
        assert all(torch.equal(uninterrupted.state_dict()[k], v)
                   for k, v in resumed.state_dict().items())
        torch.testing.assert_close(uninterrupted(existing, evidence, anchor),
                                   resumed(existing, evidence, anchor), rtol=0, atol=0)
    checks["atomic_optimizer_rng_resume_bit_exact"] = True

    class ThresholdFixture:
        fold = {"inner_val_query_ids": ["a", "b", "c"]}
        roles = {"a": {"identity": "T"}, "b": {"identity": "W"}, "c": {"identity": "missing"}}
        identity = {0: "W", 1: "T", 2: "X"}
    threshold_rows = [
        {"query_id": "a", "axis": [0, 1, 2], "winner": 0, "logits": [0., .8, .1]},
        {"query_id": "b", "axis": [0, 1, 2], "winner": 0, "logits": [0., .2, .1]},
        {"query_id": "c", "axis": [0, 1, 2], "winner": 0, "logits": [0., .3, .1]},
    ]
    selected_threshold = select_threshold(threshold_rows, ThresholdFixture())
    assert selected_threshold["threshold"] == .3
    assert selected_threshold["rescues"] == 1 and selected_threshold["breaks"] == 0
    checks["inner_threshold_net_gain_conservative_tie"] = True

    # Real target shape and dimensionality, all C128 in one natural-anchor update.
    torch.manual_seed(103)
    qfull = torch.randn(1, 128, 64, 72, dtype=dtype)
    rfull = torch.randn_like(qfull)
    masks = torch.ones(1, 128, 64, dtype=torch.bool)
    full = EvidenceBatch(qfull, rfull, masks, masks)
    xfull = torch.randn(1, 128, 18, dtype=dtype)
    fulltruth = torch.zeros(1, 128, dtype=torch.bool)
    fulltruth[0, 5] = True
    timing = {}
    for arm, model in models.items():
        start = time.perf_counter()
        model.zero_grad()
        loss = shared_hit_miss_loss(model(xfull, full, torch.tensor([0])), fulltruth, torch.tensor([0]))
        loss.backward()
        assert torch.isfinite(loss)
        timing[arm] = time.perf_counter() - start
    checks["real_shape_C128_64tokens72features_finite_forward_backward"] = True
    payload = {"status": "REBUT_QR_QRR_ENGINEERING_PASS", "scientific_result": False,
               "checks": checks, "synthetic_fit": detail, "parameter_report": report,
               "shape_timing_seconds_2CPU": timing,
               "module_sha256": hashlib.sha256((RC / "src/rc_aslo_xf/rebut_qr_qrr_v1.py").read_bytes()).hexdigest(),
               "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "torch_version": torch.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
