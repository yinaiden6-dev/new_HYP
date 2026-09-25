# Additional primary-source checks: spatial support and frozen correlation readers

2026-09-25. Literature/implementation audit; no new inference or training.

## DELF

Source: https://arxiv.org/html/1612.06321v2#S4.SS2 ; §4.2.1–4.2.3 and §4.4.
DELF uses image-level labels to learn attention for selecting useful local retrieval descriptors, then uses local matching/geometric verification. Its descriptor attention is computed per image, unlike our pair-conditioned support from a separate frozen matcher. Accordingly neither weakly supervised local selection nor task training without boxes/masks is a stand-alone novelty claim here. A difference in conditioning source does not establish overall methodological originality by itself.

## CVNet

Source: https://arxiv.org/html/2204.01458v1#S4 ; §§4.1–4.3, equations4–6.
CVNet-Rerank consumes local feature maps from a trained, frozen backbone, constructs cross-scale4D correlations, then pools a4D CNN output to classify image pairs under identity-derived supervision. Thus frozen features, learning from matching identities, and compressing spatial relations into a similarity scalar all have direct precedents. Our distinct interface keeps a separate content reader and routes a separate matcher-derived scalar through either a score head or the query representation; CVNet is not reported as performing our fixed-model response decomposition.

## RoMa v2

Source: https://arxiv.org/html/2511.15706v1 ; §1, §3.2, AppendixC covisibility.
RoMa v2 explicitly produces directional warp confidence reflecting covisibility. Multi-view context, correspondence distributions, and the frozen matching model are inherited capabilities. We should attribute them to RoMa rather than claim to invent reference-dependent spatial support. The downstream identity use, its paired controls, and comparisons of external calibration with internal modulation are the subject of our study.

## Actual internal adapter, not a newly invented attention operator

Local source: ../../../programs/rc_prellm_m_adapter_v1.py, QualityResidualAdapter.forward (around line216); ../../../programs/rc_prellm_m_scaled_adapter_v4.py, ScaledQualityResidualAdapter.

Ignoring BF16 rounding, its learned residual can be written as

`A(h_i,z) = h_i + rho W_up GELU(W_h LN(h_i) + w_z z + b_down) + rho b_up`.

`z` is the TRAIN-standardized log of candidate-specific M, with pinned input gain. The same z is repeated at all image-token positions. A common response is therefore an available and natural mechanism of this architecture; the algebra is not itself a novel theorem. Our fixed-model deletion/recomposition measures whether that route actually carries the trained corrections, rather than inferring it solely from the architecture. The final normalization means a shared additive displacement can change different token cosines differently; it is not the same as a uniform positive scaling. This is a mathematical interpretation of our implementation, not a claim of strict equivalence to an external scalar head.
