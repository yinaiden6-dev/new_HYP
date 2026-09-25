# RoMa v2 correspondence-source qualification V1

Date: 2026-08-31

## Purpose

Current ColNomic-RGH produced zero legal directions in 1,222/1,222 atom
directions.  Reference-visibility filtering retained enough cells in only
27/1,222 directions and recovered zero legal directions.  DINO-RCDE geometric
exclusivity and Qwen conditional representation are already formal NO-GO.

RoMa v2 is introduced only as a new, pretrained dense correspondence and
overlap-certainty source.  It does not replace ColNomic retrieval, does not
receive target labels, and does not directly add a retrieval bonus.

Pinned inputs:

- RoMa v2 source tag `v2.0.1`, commit
  `95c9968145c8906b7b59383258e9f73b02853d89`;
- DINOv3 source commit
  `adc254450203739c8149213a7a69d8d905b4fcfa`;
- RoMa checkpoint SHA256
  `1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7`.

All code, weights, caches and temporary compilation files reside in the
workspace or Slurm scratch.  Home configuration/files must not be modified.

## E0

One already-opened pair (`DIFFICULT-0043`, target physical row 1130) verifies:

- offline model load from pinned workspace artifacts;
- complete finite bidirectional warp, overlap and precision outputs;
- overlap is bounded in `[0,1]`, nonconstant and nonempty;
- both directions expose at least one cell above the model-semantic `0.5`
  overlap probability;
- candidate/source hashes and EXIF-oriented image dimensions are recorded;
- no training, retrieval action, P0, opened-new, or sealed access.

E0 pass only authorizes the fixed six-case target/competitor correspondence
qualification.  It is not scientific evidence.

