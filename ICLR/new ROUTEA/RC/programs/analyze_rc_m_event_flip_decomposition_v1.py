#!/usr/bin/env python3
"""Per-event decomposition of the two AMP_PERMUTE POST flips.

Offline only; no jobs, no GPU, no training. Exactly reproduces the frozen POST
decision from the archived post_row + head, then decomposes each flip.

The three components the open gap asked for were: absolute level, candidate
response difference, and MaxSim match-location change. The exact reproduction
resolves the first one structurally: the POST head consumes

    features = [ (raw[j]-raw[w]) / std(raw),                # RAW, world-invariant
                 (L[j]-L[w]) / (|L[j]|+|L[w]|),             # symmetric content delta
                 1 ]                                        # bias

so POST never reads M directly -- M enters only through the content score L,
and the content feature is a SYMMETRIC RATIO, not an absolute level. The whole
flip therefore lands on one coefficient, head[1], times one feature change.

Layers reported:
  L1  exact reproduction of stored POST logits (expect 0.0 error)
  L2  total target-logit change == head[1] * delta(symmetric L feature)
  L3  that feature change split into target-side / RAW-winner-side / interaction
  L4  the target's own L change split by whether each query token's MaxSim
      argmax moved (descriptive; see the frozen-argmax caveat printed inline)

Usage: python3 analyze_rc_m_event_flip_decomposition_v1.py [--json OUT]
"""
import argparse, json, numpy as np

RC = "/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC"
BRIDGE_INPUTS = f"{RC}/results/rc_m_causal128_attribution_chain_v1/bridge/inputs.json"
EXT = f"{RC}/results/rc_m_fine_c128_extension_v1/phase"
CLOSURE = f"{RC}/results/rc_m_upstream_event_closure_v1/result.json"
WORLD_A, WORLD_B = "NATIVE", "AMP_PERMUTE"


def decision(row, L, head):
    """Frozen POST head, copied from analyze_rc_postllm_signal_decomposition_v2."""
    L = np.asarray(L, np.float64)
    raw = np.asarray(row["raw_scores"], np.float64)
    w = int(row["winner_index"])
    idx = np.asarray(row["challenger_positions"], np.int64)
    assert list(idx) == [k for k in range(128) if k != w]
    feats = np.stack(((raw[idx] - raw[w]) / max(float(raw.std()), 1e-12),
                      (L[idx] - L[w]) / (np.abs(L[idx]) + abs(L[w]) + 1e-12),
                      np.ones(127)), axis=1)
    logits = feats @ np.asarray(head, np.float64)
    k = int(np.argmax(logits))
    return (int(idx[k]) if logits[k] > 0 else w), logits, idx, feats


def sym(Lt, Lw):
    return (Lt - Lw) / (abs(Lt) + abs(Lw) + 1e-12)


def find_events():
    """Locate rows where POST is correct at NATIVE and wrong at AMP_PERMUTE."""
    D = json.load(open(CLOSURE))
    out = []
    for r in D["rows"]:
        if r["family"] != "phase" or not r["target_present"]:
            continue
        c = r["correct"]["POST"]
        if c.get(WORLD_A) and c.get(WORLD_B) is False:
            effs = [v for v in r["effects"].values() if "wrong_positions" in v][0]
            wrong = set(effs["wrong_positions"])
            n = len(effs["candidate_logM_effect"])
            out.append((r["query_id"], r["index"],
                        [i for i in range(n) if i not in wrong][0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    byq = {r["query_id"]: r for r in json.load(open(BRIDGE_INPUTS))["rows"]}
    events = find_events()
    print(f"events found (POST correct at {WORLD_A}, wrong at {WORLD_B}): {len(events)}")
    print("POST feature order: [standardized RAW delta, symmetric L delta, bias]")
    print("-> POST does NOT read M directly; M acts only through the content score L,")
    print("   and the content feature is a symmetric RATIO, not an absolute level.\n")
    results = []
    for q, qi, tgt in events:
        d = json.load(open(f"{EXT}/query{qi:03d}/result.json"))
        pr = byq[q]["post_row"]
        head = d["head"]
        w = int(pr["winner_index"])
        h1 = head[1]
        print("=" * 96)
        print(f"{q}\n  index={qi} target={tgt} RAW_winner={w} head={np.round(head,4).tolist()}")
        rec = dict(query_id=q, index=qi, target=tgt, raw_winner=w, head=head)

        # L1 exact reproduction
        errs, picks, logit_t = {}, {}, {}
        for wd in (WORLD_A, WORLD_B):
            L = np.array(d["worlds"][wd]["L"])
            picked, lg, idx, f = decision(pr, L, head)
            z = np.zeros(128)
            z[pr["challenger_positions"]] = lg
            stored = np.array(d["worlds"][wd]["POST"]["logits128"])
            errs[wd] = float(np.abs(z - stored).max())
            picks[wd] = (picked, int(d["worlds"][wd]["POST"]["prediction_position"]))
            logit_t[wd] = float(lg[list(idx).index(tgt)])
        print(f"  L1 exact replay: max|logit err| {WORLD_A}={errs[WORLD_A]:.3e} "
              f"{WORLD_B}={errs[WORLD_B]:.3e}   picked/stored "
              f"{picks[WORLD_A]} -> {picks[WORLD_B]}")
        rec["replay_max_abs_err"] = errs
        rec["picked"] = {k: v[0] for k, v in picks.items()}

        # L2 whole change is one coefficient times one feature change
        Ln = np.array(d["worlds"][WORLD_A]["L"])
        Lp = np.array(d["worlds"][WORLD_B]["L"])
        tot = logit_t[WORLD_B] - logit_t[WORLD_A]
        dsym = sym(Lp[tgt], Lp[w]) - sym(Ln[tgt], Ln[w])
        print(f"  L2 target logit {logit_t[WORLD_A]:+.6f} -> {logit_t[WORLD_B]:+.6f}"
              f"  = {tot:+.6f}")
        print(f"     RAW feature identical across worlds: "
              f"{np.allclose(decision(pr,Ln,head)[3][:,0], decision(pr,Lp,head)[3][:,0])}")
        print(f"     head[1] * d(symL) = {h1*dsym:+.6f}  exact: {np.isclose(h1*dsym, tot)}")
        rec.update(total_logit_change=tot, d_symL=float(dsym),
                   head1_times_dsymL=float(h1 * dsym))

        # L3 which side of the content contrast moved
        only_t = h1 * (sym(Lp[tgt], Ln[w]) - sym(Ln[tgt], Ln[w]))
        only_w = h1 * (sym(Ln[tgt], Lp[w]) - sym(Ln[tgt], Ln[w]))
        inter = tot - only_t - only_w
        print(f"  L3 target-side L {Ln[tgt]:.6f}->{Lp[tgt]:.6f}: {only_t:+.6f} "
              f"({100*only_t/tot:5.1f}%)")
        print(f"     RAWwin-side L {Ln[w]:.6f}->{Lp[w]:.6f}: {only_w:+.6f} "
              f"({100*only_w/tot:5.1f}%)")
        print(f"     interaction: {inter:+.6f} ({100*inter/tot:5.1f}%)")
        rec.update(target_side=float(only_t), raw_winner_side=float(only_w),
                   interaction=float(inter))

        # L4 token-level split of the target's own L change
        z = np.load(f"{EXT}/query{qi:03d}/patch/{tgt:03d}.npz")
        wl = list(map(str, z["worlds"]))
        i_n, i_p = wl.index(WORLD_A), wl.index(WORLD_B)
        msn, msp = z["maxsim"][i_n], z["maxsim"][i_p]
        moved = z["argmax"][i_n] != z["argmax"][i_p]
        n_tok = msn.size
        dL = msp.mean() - msn.mean()
        tot_sum = msp.sum() - msn.sum()
        mv = (msp[moved] - msn[moved]).sum()
        st_ = (msp[~moved] - msn[~moved]).sum()
        assert np.isclose(msn.mean(), z["L"][i_n]) and np.isclose(msp.mean(), z["L"][i_p])
        print(f"  L4 target L = mean MaxSim over {n_tok} query tokens; dL={dL:+.8f}")
        print(f"     argmax MOVED  {int(moved.sum()):3d}/{n_tok} tokens -> "
              f"{mv/n_tok:+.8f} ({100*mv/tot_sum:5.1f}% of dL)")
        print(f"     argmax STAYED {int((~moved).sum()):3d}/{n_tok} tokens -> "
              f"{st_/n_tok:+.8f} ({100*st_/tot_sum:5.1f}% of dL)")
        print("     CAVEAT: the npz stores only each token's max and its index, not the")
        print("     full 720 x Nref similarity matrix, so a frozen-argmax counterfactual")
        print("     is NOT computable offline. This split is descriptive, not causal.")
        rec.update(dL=float(dL), tokens=int(n_tok), moved_tokens=int(moved.sum()),
                   moved_share_of_dL=float(mv / tot_sum),
                   stayed_share_of_dL=float(st_ / tot_sum))
        results.append(rec)
        print()
    if a.json:
        json.dump(dict(events=results, world_a=WORLD_A, world_b=WORLD_B),
                  open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
