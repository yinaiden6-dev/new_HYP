# R5-V6 frozen-RoMa full-gallery source E0a: STOP

Date: 2026-09-05

## Decision

`R5V6_ROMAV2_FULL_GALLERY_SOURCE_E0A_STOP_INDEPENDENTLY_VALIDATED`

R5-V6 is a source-only internal STOP.  Frozen RoMaV2 overlap is distinct from
the RAW/D1 neighbourhood, but its full-gallery candidate lists contain
query-independent hubs.  It is not qualified as an upstream candidate source.
No E0b, fixed-235 join, candidate fusion, or training is authorized.

## Execution and validation

- parallel worker array `5132321`: eight of eight query shards completed;
- every query scored all 5,413 physical rows and used restart counts `[0,1]`;
- reducer plus independent validator `5132322`: `COMPLETED 0:0`;
- 5,413 rows reduced to 5,412 corrected identities by maximum score and
  lowest-physical-row tie break;
- all eight independent RoMa pair replays passed, maximum absolute error
  `1.7763588308628009e-09`;
- metric replay error was `0.0`, candidate reorder and dense/streaming replay
  were exact, and all full-gallery path/content hashes validated.

## What passed

The source is genuinely different from RAW/D1 on this target-free panel:

- mean C128 overlap: RAW `4.25`, D1 `4.125`;
- p90 C128 overlap: RAW `6.9`, D1 `6.6`;
- RBO@512: RAW `0.03738`, D1 `0.03528`;
- signed union-top512 correlation: RAW `-0.60653`, D1 `-0.61737`;
- fraction of RoMa C128 whose baseline rank is greater than512:
  RAW `0.91113`, D1 `0.921875`;
- minimum distinct identity-score bins: `5,317`;
- maximum C128-cutoff tie multiplicity: `1`.

These establish novelty, not correct-reference recovery.

## Why E0a stopped

Two frozen hubness checks failed:

- maximum Top-1 identity share: `2/8 = 0.25`, above `0.20`;
- maximum C128 identity inclusion share: `8/8 = 1.00`, above `0.75`.

Four identities occurred in all eight RoMa C128 lists, and another occurred in
seven.  The full score vectors also had mean cross-query Pearson correlation
about `0.401`.  The direct interpretation is that symmetric mean overlap
contains a strong gallery-reference prior: generic flat/reference layouts can
receive high overlap against many unrelated queries.

The failed hubness gates may not be relaxed on these eight queries.  Mean,
maximum, thresholded visibility, precision/cycle weighting, crop variants, or
other RoMa-score scans are not authorized in this lineage.

## Access and claim boundary

- target label/insertion reads: `0`;
- fixed R5-235 postjoin reads: `0`;
- `new_difficult` sealed reads: `0`;
- OCR reads: `0`;
- ownership/superregion/action reads: `0`;
- model updates: `0`.

The correct reference was never evaluated.  This result therefore does not
say whether a correct candidate entered C128, and it does not test downstream
D1/RoMa candidate processing.  It only rejects RoMa mean-overlap as the
full-gallery candidate generator.

## Authority

- producer result SHA256:
  `3ef82e3b0d359994445c9ccb47c031c036bf30c7c79152ddeb75f14a24de3892`;
- independent validation SHA256:
  `0427df66c9af4af4ff6979dcdb60c5e359423a63016837f487ea76b74cacc9af`;
- execution contract SHA256:
  `ba5d4daea235c3e01fe2d5fa95cef71a9672292dfb603312bf51986df8a0debd`.

