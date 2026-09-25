# Route A — Three-Way Head-to-Head: does "just combine ColNomic + SAM-3" work?

**Date:** 2026-07-26 (report written; run completed 2026-07-24)
**Job:** 4916211, `COMPLETED` exit 0, 15:57 wall-clock
**Scripts:** `colnomic_migration/scripts/eval_threeway_72gold.py` (run),
`colnomic_migration/scripts/bootstrap_threeway.py` (paired bootstrap)
**Results:** `colnomic_migration/results/threeway_72gold/{threeway_72gold,bootstrap_threeway}.json`

> **Why this report exists:** the run finished 7-24 and produced a decisive answer,
> but no report was written. The 7-26 log/program cross-check
> (`EXPERIMENT_AUDIT_FULL_20260726.md` §11.1) flagged it as a covered-nowhere result.
> Nothing was re-run; every number below is read off the frozen JSON.

---

## The reviewer question this answers

> *"Retrieval struggles because the target is a small part of a cluttered photo.
> SAM-3 segments packages very well. Why not just run SAM-3, crop each package,
> and let ColNomic retrieve on the crops? Why do you need your own mechanism?"*

This is the single most obvious alternative to Route A, and it uses **both
off-the-shelf tools at their best**. It had never been measured head-to-head.

---

## Protocol (all three arms identical except the arm itself)

72 gold difficult images · the **same** 5413-entry cached ColNomic gallery ·
hit judged by `origin_name` from the authoritative CSV.
Alignment chain verified: 72 gold `image_path` → `difficult_origin_name.csv`
`origin_name` → present in all 5413 gallery setids (**72/72**).

| arm | what it is |
|---|---|
| **D — bare ColNomic** | whole-image query vs gallery, no proposals. Reference. |
| **B — pure part-to-whole** | our trained part/whole head (`e7r_head_seed17.pt`), head-projected query vs head-projected gallery. **No ownership, no crops** — the retrieval mechanism alone. |
| **C — ColNomic + SAM-3** | SAM-3 proposes package crops (its geometry, its strength); **bare ColNomic scores EACH crop**; take the best (crop, gallery) pair. The strongest "combine the two off-the-shelf tools" baseline. |

Arm C is deliberately generous: it takes an **element-wise max over all SAM-3
crops** of the per-gallery score vector, i.e. it is allowed to pick the best crop
*per gallery entry*. If no proposals exist for an image it falls back to the whole
image (= arm D), so C can never be penalised for missing proposals.

---

## Result (72 queries, gallery 5413)

| arm | recall@1 | recall@5 | recall@10 |
|---|---:|---:|---:|
| D — bare ColNomic | 0.6111 | 0.8889 | 0.8889 |
| **B — pure part-to-whole** | **0.6528** | 0.8750 | **0.9167** |
| **C — ColNomic + SAM-3 bestcrop** | **0.5139** | 0.8472 | 0.8889 |

Paired bootstrap, 10 000 resamples (`bootstrap_threeway.json`):

| contrast | mean Δ | 95% CI | excludes 0 | P(Δ>0) |
|---|---:|---|---|---:|
| **D − C** (ColNomic vs ColNomic+SAM-3) | **+0.0975** | [+0.0139, +0.1944] | **YES** | 0.9815 |
| **B − C** (ours vs ColNomic+SAM-3) | **+0.1389** | [+0.0417, +0.2361] | **YES** | 0.9972 |
| B − D (ours vs bare ColNomic) | +0.0415 | [−0.0139, +0.0972] | **no** | 0.8758 |

---

## What this establishes

**1. Bolting SAM-3 onto ColNomic makes retrieval WORSE, significantly.**
D − C = **+0.0975, CI [+0.0139, +0.1944], excludes 0**. Feeding ColNomic SAM-3's
own crops costs ~10 recall@1 points versus just handing it the whole photo. The
obvious combination is not merely unhelpful — it is actively harmful, and this is
established with SAM-3's own segmentation quality, not a weakened stand-in.

**Mechanism of the harm — corrected 2026-07-26 after checking the crop ablation.**

An earlier draft attributed this to "cropping discards context". **That explanation
is wrong.** The EXIF-fixed ablation (`results/oracle_crop/oracle_crop.json`, same
72, same gallery, official SUM MaxSim) shows a **correct** crop *helps*:

| condition | recall@1 |
|---|---:|
| D whole image | 0.6389 |
| **O_bbox — gold target crop** | **0.6667** |
| **Random_bbox — same size, off-target** | **0.4444** |

So cropping is not the problem; **cropping to the wrong thing** is. The real
mechanism for arm C is **selection over many candidates**:

- SAM-3 emits **~24 crops per image on average** (median 11.5, max 81, 4 prompts;
  2 images get none → whole-image fallback) — `sam3_difficult72_export.json`.
- Arm C takes an **element-wise max over all of them** against every gallery entry.
- Only ~1 of those ~24 is the target. The other ~23 are `Random_bbox`-like
  off-target crops, each scoring **0.4444**-quality noise against 5413 entries —
  and the **max** of ~24 noisy score vectors reliably exceeds the one clean signal.

**This is a multiple-comparisons failure, not a context-destruction failure.** The
"generosity" of arm C is precisely what sinks it: more proposals = more chances for
a wrong crop to win. A SAM-3 combination that could *select the right crop* would
not have this problem — but selecting it is exactly the identity-dependent step
SAM-3 cannot do, and which Route A's ownership addresses.

⚠️ **Do not cite this as "cropping destroys ColNomic's retrieval signal."** The
gold-crop ablation directly refutes that reading.

**2. Our part-to-whole head beats the SAM-3 combination decisively.**
B − C = **+0.1389, CI [+0.0417, +0.2361], excludes 0, P=0.997**. This is a real
head-to-head win over the strongest off-the-shelf combination — obtained with **no
ownership, no crops, no SAM-3**, just the trained part/whole projection.

**3. Route A does not need SAM-3 as a component — and the red line is intact.**
This report never claims a *geometry* win over SAM-3 (see limits). It shows the
narrower and more useful thing: **for the retrieval task, adding SAM-3 proposals
without an identity-aware way to choose among them hurts.**

**4. The corrected mechanism makes this a POSITIVE argument for ownership, not just
a negative one about SAM-3.** Since a gold crop *helps* (+0.028) and a random crop
*hurts* (−0.194), the entire question is **which crop you pick**. SAM-3 supplies
~24 candidates and no way to choose; taking the max over them is worse than not
cropping at all. **The missing ingredient is exactly identity-conditioned
selection** — which is what Route A's ownership provides. This reframes the result
from "SAM-3 doesn't help" to "**segmentation without identity is not enough, and
the gap is quantified**."

---

## What this does NOT establish

- ❌ **B > D is NOT significant.** +0.0415 with CI **[−0.0139, +0.0972]**,
  P(Δ>0)=0.876 — **crosses 0**. Our head is *directionally* above bare ColNomic on
  this set but **the win bar (own mechanism > ColNomic) is NOT met here.**
  Do not cite 0.6528 > 0.6111 as a victory over ColNomic.
- ❌ **Not comparable to the official 0.647.** These are **72 gold** images; the
  official ColNomic difficult reference (0.647) is over **224**. On this 72-subset
  bare ColNomic is **0.6111**. Same denominator caution as
  `REPORT_ROUTEA_LOCALIZATION_BOUNDED_20260726.md` (224 / 72 / 44 / 0.611).
  The `colnomic_ref_difficult_full224: 0.647` field in the JSON is a stored
  reference constant, **not** a measurement on these 72.
- ❌ **No geometry claim about SAM-3.** SAM-3's masks remain far better than Route
  A's (E6-R5: 0.982 vs 0.599 on the 74×74 grid). This experiment says nothing
  about segmentation quality — only that *routing retrieval through those masks*
  loses information.
- ❌ **Single head seed** (`e7r_head_seed17.pt`). Arm B is one trained head, not a
  multi-seed result.
- ❌ **Not an ownership result.** Arm B uses no ownership at all. This does not
  speak to whether ownership helps retrieval (that remains gap ① / ②).

---

## Consequence for the ledger (a correction)

`ROUTE_A_FULL_LEDGER_20260725.md` lists "**纯 part-to-whole 超 ColNomic ❌
(+0.009 不显著)**" under NO-GO. Two separate problems with that line:

1. The **+0.009** came from `full_difficult_BvD`, whose query was the **whole
   difficult image, not a part** — already flagged as a mis-measurement
   (audit §0, §7).
2. The correct measurement of *this* contrast on the 72-gold set is **+0.0415,
   CI [−0.0139, +0.0972]** — still **not significant**, so the NO-GO **verdict
   stands**, but the number and the reason both change.

**Recommended ledger wording:** "pure part-to-whole vs bare ColNomic: +0.042 on
72 gold, CI crosses 0 (not significant) — NO-GO on the win bar; but B beats the
ColNomic+SAM-3 combination by +0.139, CI excludes 0."

---

## Reproduce

```bash
sbatch "ICLR/new ROUTEA/colnomic_migration/programs/run_threeway_72gold.sbatch"
python "ICLR/new ROUTEA/colnomic_migration/scripts/bootstrap_threeway.py"
```

Frozen outputs: `results/threeway_72gold/threeway_72gold.json` (point estimates +
per-query D/B/C hit flags), `results/threeway_72gold/bootstrap_threeway.json` (CIs).
