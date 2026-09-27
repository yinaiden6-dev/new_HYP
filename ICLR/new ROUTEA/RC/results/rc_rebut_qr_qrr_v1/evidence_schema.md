# Shared pooled F71 evidence schema

Version: `POOLED_F71_COARSE17_NATIVE_CENTER_WARP_V1`.

The fixed panel is existing coordinate-cache execution indices 0 through 70, each with all 128 natural ColNomic candidates. No query or token is selected by an identity label, correctness, retrieval rank, or support threshold. The source feature bank covers all H593, but the complete reusable warp intersection covers 71 queries. This is not an F593 cache.

## Entry points

- `evidence_manifest.json`: lives in this directory, alongside this schema.
- `evidence/queryNNN.pt`: per-query tensor payload.
- `evidence/queryNNN.json`: per-query receipt and provenance.
- `evidence/projected_images/<key>.pt`: reused per-image pooled-descriptor projection.
- `evidence_engineering_checks.json`: partial engineering verification, not final completion.
- `evidence_validation.json`: final independent validation, only when all 71 outputs are complete.

The early `evidence/evidence_manifest.json` is an obsolete initial snapshot. Consumers must use the manifest in the parent experiment directory.

## Tensor fields

| Key | Shape / type | Meaning |
|---|---|---|
| `query` | `[128,64,72]`, float32 | Pair-conditioned evidence on the same query 8×8 geometric axis for every reference |
| `reference` | `[128,64,72]`, float32 | Reverse-direction evidence on each reference's own 8×8 axis |
| `query_valid` | `[128,64]`, bool | At least one source-geometry-valid query token in this bin |
| `reference_valid` | `[128,64]`, bool | At least one source-geometry-valid reference token in this bin |
| `query_mapping_valid` | `[128,64]`, bool | At least one valid AB center mapping in this bin |
| `reference_mapping_valid` | `[128,64]`, bool | At least one valid BA center mapping in this bin |
| `M0` | `[128]`, float64 | Original terminal `sqrt(mean(u)*mean(v))`, bit-preserved |
| `native_X` | `[127,6]`, float64 | Original head evidence, in `challenger_positions` order |
| `raw_candidate_scores` | `[128]`, float64 | Original candidate RAW scores |
| `axis` | 128 integer entries | Physical gallery rows in original candidate order |
| `winner` | integer | Original RAW winner's position in `axis` |
| `challenger_positions` | 127 integers | Original challenger positions, excluding `winner` |
| `query_id`, `execution_ordinal` | linkage only | Used to join fixed folds; must not be model inputs |

No identity labels are stored in this evidence payload. The training process owns the separately authorized fold-label join.

## The 72 channels

| Slice | Count | Content |
|---|---:|---|
| `0:32` |32| Source-side `coarse_17` pooled descriptors, projected to32 and averaged over source-valid token cells |
| `32:64` |32| Other-side projected descriptors sampled at each original token's saved native warp, then averaged over mapping-valid cells |
| `64` |1| Source-side support, averaged over source-valid cells; includes zero/low values |
| `65` |1| Other-side support sampled at the saved native warp, averaged over mapping-valid cells |
| `66:68` |2| Mean source-cell coordinates, normalized to[-1,1] |
| `68:70` |2| Mean mapped target coordinates, normalized to[-1,1], over valid mappings |
| `70` |1| Fraction of geometrically valid source tokens among all source token cells assigned to this bin |
| `71` |1| Fraction of valid mappings among geometrically valid source tokens in this bin |

Source content and source support survive a missing/invalid mapping. Only unavailable other-side values are zeroed. Empty source bins have all72 channels zero. The separate masks and fractions distinguish missing mappings from visible zero support.

## Alignment and pooling

1. Use frozen RoMa `coarse_17` descriptors already area-pooled onto each original image's ColNomic token cells. Pooling was performed in FP64; saved descriptors are FP32.
2. Apply the same label-free Rademacher projection to both sides: 1024→32, seed20260927, entries±1/√32. No supervised normalization or learned projection is fit here.
3. For each original source token, read the actual terminal RoMa warp sampled at that cell's center. This coordinate is in the other image's EXIF-oriented RoMa frame, normalized to[-1,1].
4. Locate the other image cell by its explicit oriented `cell_boxes_xyxy`, using nearest center with containment verification. Never equate reference token-array indices across different images.
5. A mapping is valid only if the source cell is geometrically valid, warp coordinates are finite/in bounds, and the located other cell is geometrically valid. Certainty does not decide this mask; geometric validity is not a ground-truth visibility or correct-match label.
6. Only after this sampling, bin source cell centers into a fixed8×8 normalized image grid and take arithmetic means. Averaging warp first and then sampling is not equivalent and is not used.
7. Reverse the procedure using BA warp for the reference-side tensor.

This interface combines same-layer coarse descriptors on both sides with terminal support and coordinates. It is not a claim that all fields come from one RoMa stage, nor that the raw high-resolution descriptor maps were archived. The original terminalM0 is retained separately.

## Fair model consumption

All QR, QR-vec, and QRR arms receive these same tensors/masks and the same original evidence at the common final decision head. QR independently compresses each candidate pair; QR-vec preserves its independent vector; QRR can compare two candidates on the common query bin axis before pooling. Reference-side bins belong to different images and must be independently summarized, not subtracted by matching array indices.

Masking out `query_mapping_valid=false` as if the entire source bin were absent would discard available source content. Use `query_valid`/`reference_valid` as source pooling masks and expose mapping flags/fractions as evidence.

## Provenance and completion

Per-query receipts bind the original feature/coordinate records, large-source sealedSHA values, newly consumed descriptor/geometry/warp array digests, and full new payloadSHA. Large source tensors are memory-mapped; no new encoder or RoMa forward is run. Final completion requires all71 sealed payloads plus independent validation, not a scheduler state or partial manifest.

Builder: `../../programs/build_rebut_qr_qrr_evidence_v1.py`.
Independent validator: `../../programs/verify_rebut_qr_qrr_evidence_v1.py`.
