#!/usr/bin/env python3
"""Item 1: which upstream property separates the STRONGEST confusable identity.

Offline only. Reads the frozen per-query sources of rc_m_upstream_event_closure_v1
(verified by sha256) and works on ABSOLUTE per-candidate quantities -- M (with-P
support), L (content / local weighting), and the ratio M/L -- instead of the
intervention effect differences that the shipped summary reports.

Why this reframing: an effect difference (world A minus world B) is only
indirectly related to "who ranks first". Ranking is an absolute-quantity
question, and M/L per candidate are on disk, so it can be answered offline.

Sections
  S1  target rank of the correct identity in each coordinate (M, L, M/L)
  S2  is the strongest confuser a STABLE candidate across worlds, or a moving
      target?  (decides whether "exclude the strongest" is even well posed)
  S3  paired target-vs-strongest-confuser contrast in all three coordinates
  S4  the decisive test: does the target beat its strongest confuser on M
      (absolute support) or on M/L (correspondence gain relative to content)?
  S5  per-world breakdown, and the same contrast restricted to the original
      rescue events.

Usage: python3 analyze_rc_m_strongest_confuser_offline_v1.py [--json OUT]
"""
import argparse, hashlib, json, math, random, statistics as st, sys
from collections import defaultdict

RC = "/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC"
CLOSURE = f"{RC}/results/rc_m_upstream_event_closure_v1/result.json"
EPS = 1e-12


def boot(vals, n=20000, seed=0):
    if len(vals) < 2:
        return float("nan"), float("nan")
    rnd = random.Random(seed)
    out = sorted(sum(rnd.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return out[int(0.025 * n)], out[int(0.975 * n)]


def load(verify=True):
    """Load closure rows joined to their frozen per-query source payloads."""
    D = json.load(open(CLOSURE))
    by_fam = {"phase": {}, "context": {}}
    n_ok = 0
    for s in D["sources"]:
        if verify:
            h = hashlib.sha256(open(s["path"], "rb").read()).hexdigest()
            if h != s["sha256"]:
                sys.exit(f"sha256 mismatch: {s['path']}")
            n_ok += 1
        fam = "phase" if "/phase/" in s["path"] else "context"
        qdir = s["path"].rstrip("/").split("/")[-2]          # query000
        by_fam[fam][int(qdir.replace("query", ""))] = json.load(open(s["path"]))
    recs = []
    for r in D["rows"]:
        src = by_fam[r["family"]].get(r["index"])
        if src is None:
            continue
        effs = [v for v in r["effects"].values() if "wrong_positions" in v]
        if not r["target_present"] or not effs:
            continue                                          # no-target control
        wrong = set(effs[0]["wrong_positions"])
        n = len(effs[0]["candidate_logM_effect"])
        tgt = [i for i in range(n) if i not in wrong][0]
        recs.append(dict(row=r, src=src, target=tgt,
                         native_strongest=effs[0]["fixed_native_strongest_M_wrong"]))
    return D, recs, n_ok


def coords(world):
    """Per-candidate coordinates: M, L, and log(M/L)."""
    M, L = world["M"], world["L"]
    return {
        "M":       [math.log(max(m, EPS)) for m in M],
        "L":       [math.log(max(l, EPS)) for l in L],
        "M_over_L": [math.log(max(m, EPS)) - math.log(max(l, EPS)) for m, L_, l in
                     zip(M, L, L)],
    }


def s1_ranks(recs):
    print("=" * 100)
    print("S1. Rank of the CORRECT identity among 128 candidates (1 = best), NATIVE world")
    print("    M = with-P support | L = content/local weighting | M/L = correspondence gain")
    print("=" * 100)
    per = defaultdict(lambda: defaultdict(list))
    for rec in recs:
        w = rec["src"]["worlds"].get("NATIVE")
        if w is None:
            continue
        t = rec["target"]
        for name, v in coords(w).items():
            rank = 1 + sum(1 for i, x in enumerate(v) if i != t and x > v[t])
            per[name][rec["row"]["component"]].append(rank)
    print(f"{'coord':10s} {'mean rank':>10s} {'median':>7s} {'rank==1':>8s} {'top5':>6s} {'top13(10%)':>11s}")
    out = {}
    for name in ("M", "L", "M_over_L"):
        flat = [x for v in per[name].values() for x in v]
        gm = [st.mean(v) for v in per[name].values()]
        lo, hi = boot(gm, seed=1)
        out[name] = dict(mean_rank=st.mean(flat), median=st.median(flat),
                         frac_rank1=sum(1 for x in flat if x == 1) / len(flat),
                         frac_top5=sum(1 for x in flat if x <= 5) / len(flat),
                         frac_top13=sum(1 for x in flat if x <= 13) / len(flat),
                         group_ci=[lo, hi], n=len(flat))
        o = out[name]
        print(f"{name:10s} {o['mean_rank']:10.2f} {o['median']:7.1f} "
              f"{o['frac_rank1']:8.3f} {o['frac_top5']:6.3f} {o['frac_top13']:11.3f}")
    return out


def s2_stability(recs):
    print()
    print("=" * 100)
    print("S2. Is the strongest confuser STABLE across worlds?")
    print("    (if it moves, 'exclude the strongest' is a moving target and must")
    print("     become 'exclude a top-k set')")
    print("=" * 100)
    rows, n_stable = [], 0
    for rec in recs:
        t = rec["target"]
        ids = {}
        for wn, w in rec["src"]["worlds"].items():
            lm = coords(w)["M"]
            best = max((x for i, x in enumerate(lm) if i != t))
            ids[wn] = [i for i, x in enumerate(lm) if i != t and x == best][0]
        uniq = set(ids.values())
        native = ids.get("NATIVE")
        agree = sum(1 for v in ids.values() if v == native) / len(ids)
        n_stable += (len(uniq) == 1)
        rows.append(dict(query=rec["row"]["query_id"], family=rec["row"]["family"],
                         n_distinct=len(uniq), frac_equal_native=agree,
                         native_strongest_M=native,
                         closure_fixed=rec["native_strongest"],
                         matches_closure=(native == rec["native_strongest"])))
    print(f"  identical strongest confuser in EVERY world: {n_stable}/{len(rows)} query-families")
    nd = [r["n_distinct"] for r in rows]
    print(f"  distinct confusers per query-family: mean={st.mean(nd):.2f} max={max(nd)}")
    fa = [r["frac_equal_native"] for r in rows]
    print(f"  share of worlds whose strongest == NATIVE's strongest: mean={st.mean(fa):.3f} min={min(fa):.3f}")
    mc = sum(1 for r in rows if r["matches_closure"])
    print(f"  agrees with closure's fixed_native_strongest_M_wrong: {mc}/{len(rows)}")
    for r in rows:
        if r["n_distinct"] > 1:
            print(f"    MOVES: {r['query'][:26]} {r['family']:8s} "
                  f"n_distinct={r['n_distinct']} frac_native={r['frac_equal_native']:.2f}")
    return rows


def s3_s4_contrast(recs):
    print()
    print("=" * 100)
    print("S3/S4. Target vs its STRONGEST CONFUSER, NATIVE world, per coordinate")
    print("    margin > 0 means the correct identity is ahead of the hardest wrong one.")
    print("    S4 question: is the target ahead on absolute support M, or on M/L?")
    print("=" * 100)
    per = defaultdict(lambda: defaultdict(list))
    for rec in recs:
        w = rec["src"]["worlds"].get("NATIVE")
        if w is None:
            continue
        t, sw = rec["target"], rec["native_strongest"]
        c = coords(w)
        for name, v in c.items():
            per[name][rec["row"]["component"]].append(v[t] - v[sw])
    print(f"{'coord':10s} {'margin':>9s} {'ci_lo':>9s} {'ci_hi':>9s} {'pos_grp':>9s} {'excl_0':>7s}")
    out = {}
    for name in ("M", "L", "M_over_L"):
        gm = [st.mean(v) for v in per[name].values()]
        lo, hi = boot(gm, seed=2)
        excl = "YES" if (lo > 0 or hi < 0) else "no"
        out[name] = dict(mean=st.mean(gm), ci=[lo, hi],
                         positive_groups=sum(1 for x in gm if x > 0), groups=len(gm),
                         excludes_zero=(lo > 0 or hi < 0))
        print(f"{name:10s} {st.mean(gm):+9.5f} {lo:+9.5f} {hi:+9.5f} "
              f"{out[name]['positive_groups']:4d}/{len(gm):<4d} {excl:>7s}")
    print()
    print("  Reading: if M/L excludes 0 while M does not, the property that separates the")
    print("  strongest confuser is correspondence gain RELATIVE TO CONTENT, not raw support.")
    return out


def s5_worlds_events(recs):
    print()
    print("=" * 100)
    print("S5. Same contrast per world, and restricted to original rescue events")
    print("=" * 100)
    byw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in recs:
        t, sw = rec["target"], rec["native_strongest"]
        comp = rec["row"]["component"]
        for wn, w in rec["src"]["worlds"].items():
            c = coords(w)
            for name in ("M", "L", "M_over_L"):
                byw[wn][name][comp].append(c[name][t] - c[name][sw])
    print(f"{'world':16s} {'M':>10s} {'L':>10s} {'M_over_L':>10s}")
    for wn in sorted(byw):
        vals = []
        for name in ("M", "L", "M_over_L"):
            gm = [st.mean(v) for v in byw[wn][name].values()]
            vals.append(st.mean(gm))
        print(f"{wn:16s} {vals[0]:+10.5f} {vals[1]:+10.5f} {vals[2]:+10.5f}")
    print()
    ev = [r for r in recs if r["row"]["correct"]["POST"].get("NATIVE")
          and not r["row"]["raw_correct"]]
    print(f"  original POST rescue events (RAW wrong -> POST correct): n={len(ev)}")
    if ev:
        for name in ("M", "L", "M_over_L"):
            g = defaultdict(list)
            for rec in ev:
                w = rec["src"]["worlds"]["NATIVE"]
                c = coords(w)
                g[rec["row"]["component"]].append(
                    c[name][rec["target"]] - c[name][rec["native_strongest"]])
            gm = [st.mean(v) for v in g.values()]
            lo, hi = boot(gm, seed=3)
            print(f"    {name:10s} margin={st.mean(gm):+.5f} ci=[{lo:+.5f},{hi:+.5f}] "
                  f"pos={sum(1 for x in gm if x>0)}/{len(gm)}")
    return {w: {n: st.mean([st.mean(v) for v in byw[w][n].values()])
                for n in ("M", "L", "M_over_L")} for w in byw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    D, recs, n_ok = load(verify=not a.no_verify)
    print(f"closure: {CLOSURE}")
    print(f"sha256-verified frozen sources: {n_ok}/{len(D['sources'])}")
    print(f"target-present query-families: {len(recs)}\n")
    res = dict(source=CLOSURE, sources_verified=n_ok,
               n_query_families=len(recs),
               s1_ranks=s1_ranks(recs),
               s2_stability=s2_stability(recs),
               s3_s4_contrast=s3_s4_contrast(recs),
               s5_worlds=s5_worlds_events(recs))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
