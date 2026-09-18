# RoMa-v2 + ColNomic full-negative external confirmation contract V1

Date: 2026-08-31

## 1. Purpose and evidence boundary

The frozen development result improved target-free full-C128 recognition from
`24/32` to `27/32`, with `3` rescues and `0` breaks.  An independent reducer
recomputed every action and passed all seven integrity checks.

This contract freezes a **one-shot sealed directional confirmation** on the
previously registered `new_difficult` test endpoint (8 identities, 31 primary
queries).  The endpoint is too small for a final paper-strength effect-size
claim.  Final confirmation still requires a newly collected untouched
identity-disjoint external dataset.

Freezing this document does not authorize reading, listing, embedding or
scoring sealed query bytes.  A separate input-binding preflight must first
bind an existing immutable sealed manifest without exposing labels to the
scoring process.

## 2. Frozen mechanism

### 2.1 Candidate generation and base decision

- Base system: frozen raw ColNomic full-gallery retrieval.
- Gallery: the complete registered 5,413 physical reference rows.
- Local axis: the natural ColNomic top-128 exact-label candidates; no target
  insertion, promotion, special slot or oracle completion is allowed.
- Exact-label duplicate aggregation, tie breaking and preprocessing must be
  inherited byte-for-byte from the registered raw ColNomic evaluator.
- If the true label is absent from natural C128, the local route cannot rescue
  it and must not alter the candidate set.

### 2.2 Candidate-conditioned visibility-XF score

For every natural candidate `g`, frozen RoMa v2 produces query and reference
visibility weights `wq` and `wr`.  Frozen ColNomic supplies exact-content token
similarity inside that visibility:

```text
local(g) = sqrt(mean(wq) * mean(wr))
           * sum_i wq_i * max_j[wr_j * cos(q_i, r_gj)]
           / (sum_i wq_i + 1e-12)
```

The query and reference spatial controls use the fixed half-axis cyclic
derangements registered by the balanced32 V2 experiment.  There is no model
update, crop search, top-k search, pooling search, threshold scan or visual
selection.

Pinned model artifacts:

- RoMa v2 source tag `v2.0.1`, commit
  `95c9968145c8906b7b59383258e9f73b02853d89`;
- DINOv3 source commit
  `adc254450203739c8149213a7a69d8d905b4fcfa`;
- RoMa checkpoint SHA256
  `1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7`;
- RoMa update count: `0`.

### 2.3 Frozen HOLD/SWITCH gate

Let `w` be the raw ColNomic winner and `c` a challenger.  The six target-free
features are fixed, in order, as:

1. standardized raw ColNomic gap `(raw[c]-raw[w])/std(raw)`;
2. symmetric local-score difference;
3. symmetric visibility-mass difference;
4. symmetric visibility-normalized local-similarity difference;
5. symmetric query-spatial-robustness difference;
6. symmetric reference-spatial-robustness difference.

For any scalar pair `(a,b)`, the symmetric difference is
`(a-b)/(|a|+|b|+1e-12)`.

The frozen linear logit is:

```text
bias = -1.5548565799571785
weight = [
   0.9812819097198987,
  -3.8126280679963880,
   5.8263201312192550,
   5.8609700457116930,
  -0.14878670951682949,
  -0.24427968856864830
]
```

The highest-logit challenger is selected using the frozen physical-row tie
rule.  A logit strictly greater than zero permits exactly one SWITCH;
otherwise the action is byte-identical HOLD.  No identity-specific parameter,
target label, rank slot, manual mask, box, point, segmentation or OCR is an
input to this gate.

## 3. Process boundary

For each sealed query, the runner must:

1. compute and persist the full-gallery raw ColNomic scores;
2. derive natural C128 and compute all candidate-local REAL/control features;
3. seal the candidate axis, features, logits, HOLD/SWITCH action and hashes;
4. only in a different reducer process join the exact target label;
5. independently recompute all metrics and action decisions.

The target join must occur after all 31 target-free prejoin records are
complete.  Missing, duplicate or malformed primary queries fail closed; no
partial denominator is allowed.

## 4. One-shot metrics and layered decisions

Primary retrieval metrics are strict label R@1 and MRR over all 31 primary
queries.  The report must also include target-in-C128 count, rescue, break,
wrong-to-different-wrong, SWITCH count, target rank and target-versus-final-
rival margin.

### 4.1 Directional retrieval decision

`SEALED_DIRECTIONAL_RETRIEVAL_GO` requires all of:

- final strict R@1 is greater than frozen raw ColNomic R@1;
- final MRR is not lower than frozen raw ColNomic MRR;
- rescues are greater than breaks;
- breaks are at most one.

Otherwise the status is `SEALED_DIRECTIONAL_RETRIEVAL_NO_GO`.  No repair or
threshold adjustment is allowed on this endpoint.

### 4.2 Candidate-binding and spatial claim levels

Retrieval success and causal interpretation are separate decisions.  The
report must include pre-registered candidate-binding, query-coordinate and
reference-coordinate destruction paths, but a failure of one control must
not rewrite a real retrieval gain as an engineering failure.

- Candidate-binding support requires the REAL action increment to exceed the
  fixed-point-free C_BIND increment and REAL rescue retention under C_BIND to
  be below one.
- Spatial support additionally requires REAL to exceed both half-axis
  coordinate-destruction increments, with rescue retention below one for each.
- Retrieval GO without these controls is reported only as retrieval-only or
  candidate-bound nonspatial evidence; it is not called spatial ownership.

## 5. Immutable sources

- internal result SHA256:
  `b296e946ff84ef574605ca5a6f6c6d6892b06dc8f38002b33b913a5afc6f059f`;
- independent validation SHA256:
  `2f112c8daa7b0c199e7e7ca2933116bdaf261b547277140950c809cc3bd264de`;
- full-negative trainer SHA256:
  `3fc3491c05562f59bbca3e983522f3b75d7a19e4f2ee121d89bbb3bc8a7b0310`;
- visibility-XF scorer SHA256:
  `fb73bdd6cc2b585405a9fcb1e411535f487e83d29d6021f8c1f632af78ecfd3c`.

## 6. Explicit prohibitions

Before and after the one-shot endpoint, do not:

- tune weights, bias, SWITCH threshold, C128 width, local score, derangement,
  preprocessing or gallery aggregation;
- inspect sealed labels/images to choose cases or parameters;
- insert a missing target or use filename/stem correctness at inference;
- reinterpret the 31-query endpoint as final paper-strength evidence;
- reuse this sealed endpoint for another adaptive repair after NO-GO.

The only authorized next action after this contract is frozen is an immutable
sealed-input binding preflight.  Actual sealed access remains disabled until
that preflight independently validates the manifest, runner closure and
prejoin/postjoin boundary.
