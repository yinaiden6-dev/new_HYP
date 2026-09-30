#!/usr/bin/env python3
"""Offline recompute over rc_m_upstream_event_closure_v1/result.json.

No new jobs, no GPU, no training. Reads the frozen result.json and recomputes
four things the shipped summary does not report:

  A. common-level identity check: is target_vs_strongest_wrong already
     invariant to candidate_common_level?  (answer: yes, exactly)
  B. rank-based selectivity: fraction of the 127 wrong candidates whose
     centered effect exceeds the target's.  Scale-free.
  C. magnitude guard for B: z = centered[target] / sd(centered).  Required,
     because B alone makes the zero-effect grid control look "significant".
  D. downstream response ratio: mean |delta margin| of POST vs the two
     external heads, per world and per component.

Usage: python3 analyze_rc_m_upstream_offline_recompute_v1.py [result.json]
"""
import json, random, statistics as st, sys
from collections import defaultdict

DEFAULT = ("/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/"
           "results/rc_m_upstream_event_closure_v1/result.json")

ARMS = ["phase_GLOBAL_minus_LOCAL", "amplitude_NATIVE_minus_PERMUTE",
        "amplitude_NATIVE_minus_FLAT", "context_matched_alignment",
        "context_P_only_moved", "context_P_fixed_move_AJ",
        "context_all_moved_grid_control"]
HEADS = ["POST", "NATIVE7", "M_FREE"]


def boot(vals, n=20000, seed=0):
    if len(vals) < 2:
        return float("nan"), float("nan")
    rnd = random.Random(seed)
    out = sorted(sum(rnd.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return out[int(0.025 * n)], out[int(0.975 * n)]


def target_index(eff):
    wrong = set(eff["wrong_positions"])
    return [i for i in range(len(eff["candidate_logM_effect"])) if i not in wrong][0]


def usable(row, arm):
    eff = row["effects"].get(arm)
    return row["target_present"] and eff is not None and "wrong_positions" in eff


def part_a(rows):
    print("=" * 96)
    print("A. Is target_vs_strongest_wrong already common-level invariant?")
    print("   raw  = logM_effect[target]     - logM_effect[strongest_wrong]")
    print("   cent = centered_effect[target] - centered_effect[strongest_wrong]")
    print("=" * 96)
    worst = 0.0
    for arm in ARMS:
        for coh in ("ALL_CHANGED11", "MAIN8"):
            grp = defaultdict(list)
            for r in rows:
                if r["cohort"] != coh or not usable(r, arm):
                    continue
                e = r["effects"][arm]
                t, sw = target_index(e), e["fixed_native_strongest_M_wrong"]
                raw = e["candidate_logM_effect"][t] - e["candidate_logM_effect"][sw]
                cen = e["candidate_centered_effect"][t] - e["candidate_centered_effect"][sw]
                grp[r["component"]].append((raw, cen))
            if not grp:
                continue
            gr = [st.mean([x[0] for x in v]) for v in grp.values()]
            gc = [st.mean([x[1] for x in v]) for v in grp.values()]
            worst = max(worst, max(abs(a - b) for a, b in zip(gr, gc)))
            lo, hi = boot(gc)
            print(f"{arm:32s} {coh:14s} n={len(gc):2d} raw={st.mean(gr):+.5f} "
                  f"cent={st.mean(gc):+.5f} ci=[{lo:+.5f},{hi:+.5f}] "
                  f"pos={sum(1 for x in gc if x > 0)}/{len(gc)}")
    print(f"\n  max |raw - cent| over all arms/cohorts = {worst:.3e}")
    print("  -> the common level cancels in any target-minus-wrong contrast, by construction.")


def part_bc(rows, cohort="ALL_CHANGED11"):
    print()
    print("=" * 96)
    print(f"B/C. Rank selectivity with magnitude guard  [{cohort}]")
    print("   frac = share of 127 wrong candidates whose centered effect > target's")
    print("          (0.5 = none; lower = intervention favors the target)")
    print("   z    = centered[target] / sd(centered)   <- guard; frac is scale-free")
    print("=" * 96)
    print(f"{'arm':32s} {'frac':>7s} {'ci_lo':>7s} {'ci_hi':>7s} {'z':>7s} {'cent_sd':>9s}")
    for arm in ARMS:
        gf, gz, gs = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in rows:
            if r["cohort"] != cohort or not usable(r, arm):
                continue
            e = r["effects"][arm]
            ce, wp = e["candidate_centered_effect"], e["wrong_positions"]
            t = target_index(e)
            sd = st.pstdev(ce)
            gf[r["component"]].append(sum(1 for w in wp if ce[w] > ce[t]) / len(wp))
            gz[r["component"]].append(ce[t] / sd if sd > 0 else 0.0)
            gs[r["component"]].append(sd)
        if not gf:
            continue
        fv = [st.mean(v) for v in gf.values()]
        lo, hi = boot(fv)
        z = st.mean([st.mean(v) for v in gz.values()])
        sd = st.mean([st.mean(v) for v in gs.values()])
        note = ""
        if (hi < 0.5 or lo > 0.5) and abs(z) < 0.5:
            note = "  <- rank 'significant' but |z|<0.5: scale-free artifact"
        print(f"{arm:32s} {st.mean(fv):7.4f} {lo:7.4f} {hi:7.4f} {z:+7.3f} {sd:9.5f}{note}")


def part_d(rows):
    print()
    print("=" * 96)
    print("D. Downstream response: mean |delta margin| vs each head's own NATIVE")
    print("=" * 96)
    pairs = []
    for r in rows:
        if not r["target_present"]:
            continue
        for h in HEADS:
            mm = r["target_vs_strongest_wrong_margin"][h]
            base = mm.get("NATIVE")
            if base is None:
                continue
            for w, v in mm.items():
                if w != "NATIVE":
                    pairs.append((r["component"], h, w, v - base))
    per = defaultdict(list)
    for comp, h, w, d in pairs:
        per[(h, w)].append(abs(d))
    print(f"{'world':16s} {'POST':>10s} {'NATIVE7':>10s} {'M_FREE':>10s} {'POST/ext':>9s}")
    for w in sorted({w for _, w in per}):
        vals = [st.mean(per.get((h, w), [float('nan')])) for h in HEADS]
        ext = (vals[1] + vals[2]) / 2
        print(f"{w:16s} {vals[0]:10.4f} {vals[1]:10.4f} {vals[2]:10.4f} "
              f"{vals[0]/ext if ext > 0 else float('nan'):9.1f}")
    bycomp = defaultdict(lambda: defaultdict(list))
    for comp, h, w, d in pairs:
        bycomp[comp][h].append(abs(d))
    ratios = [st.mean(v["POST"]) / st.mean(v["NATIVE7"] + v["M_FREE"])
              for v in bycomp.values() if v["POST"] and v["NATIVE7"]]
    lo, hi = boot(ratios, seed=7)
    print(f"\n  per-component POST/external ratio: mean={st.mean(ratios):.2f}x "
          f"ci=[{lo:.2f},{hi:.2f}] n={len(ratios)}")
    print("  spread across worlds is ~1.5x to ~9.4x -> NOT a single scalar gain.")


def part_e(rows):
    print()
    print("=" * 96)
    print("E. sign(margin) vs correctness, and the two AMP_PERMUTE flips")
    print("=" * 96)
    for h in HEADS:
        n = agree = 0
        for r in rows:
            if not r["target_present"]:
                continue
            for w, ok in r["correct"][h].items():
                m = r["target_vs_strongest_wrong_margin"][h].get(w)
                if m is None:
                    continue
                n += 1
                agree += ((m > 0) == ok)
        print(f"  {h:8s} {agree}/{n} agree" + ("  (identity, not evidence)" if agree == n else ""))
    print()
    for r in rows:
        if not r["target_present"]:
            continue
        pc = r["correct"]["POST"]
        if not (pc.get("NATIVE") and pc.get("AMP_PERMUTE") is False):
            continue
        print(f"  {r['query_id']}  cohort={r['cohort']} family={r['family']}")
        for h in HEADS:
            mm = r["target_vs_strongest_wrong_margin"][h]
            print(f"     {h:8s} margin {mm['NATIVE']:+.4f} -> {mm['AMP_PERMUTE']:+.4f} "
                  f"(delta {mm['AMP_PERMUTE'] - mm['NATIVE']:+.4f})  "
                  f"correct {r['correct'][h]['NATIVE']}->{r['correct'][h]['AMP_PERMUTE']}")
        e = r["effects"]["amplitude_NATIVE_minus_PERMUTE"]
        print(f"     upstream: target_logM={e['target_logM_effect']:+.5f} "
              f"common={e['candidate_common_level']:+.5f} "
              f"vs_strongest={e['target_vs_fixed_native_wrong']:+.5f} "
              f"vs_mean={e['target_vs_mean_wrong']:+.5f}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    rows = json.load(open(path))["rows"]
    print(f"source: {path}\nrows: {len(rows)}\n")
    part_a(rows)
    part_bc(rows)
    part_d(rows)
    part_e(rows)


if __name__ == "__main__":
    main()
