# Matching priors: primary-source audit (2026-09-25)

Scope: literature verification only; no experiments, no manuscript edits. Comparison target is the current `RC_NEW_HYP_PAPER_CLOSURE_DRAFT_20260924.md` and `RC_NEW_HYP_UNIFIED_SUPPORT_CLOSURE_20260925.md`, including the later scalar-M/post-LLM interpretation. Local numerical claims were read, not independently recounted here. “Not found” below means absent from the inspected methods/analyses, not a universal priority claim. Benchmark authorship is not counted as new HYP's contribution.

## 1. ELViS — paper facts

Primary: [ELViS v1, 30 March 2026](https://arxiv.org/html/2603.28603v1), Sections 3.1–3.4, Appendix B.

- **Inputs/freeze:** pretrained local descriptors; trainable dimensionality reduction. Training uses pre-extracted descriptors. Backbone freezing does not mean every downstream representation is frozen.
- **Support:** cosine matrix S; entropy-regularized OT with dustbins, equations (1)–(3). Dustbin gains h(q_i), h(x_j) are descriptor-dependent; the resulting OT matrix depends on both images. Its symbols u,v denote dustbin gains, not new HYP's support maps.
- **Content/readout:** row/column maxima of refined S′, equation (4); learned scalar vote transformation f, then summation, equation (5).
- **Supervision:** positive/negative instance pairs; learned g before BCE. No task correspondence labels are required by this objective. Appendix B includes hard-pair mining and separately trained local-feature detectors.
- **Inference:** discard g; rerank top 400 by local score. Appendix B reports ELViS without global-score fusion. No base-winner HOLD/SWITCH objective is specified.
- **Spatial/scalar analysis:** Figure 4 visualizes correspondences and dustbins. Correspondence strength becomes scalar image similarity. Appendix E retrains OT variants and warns that train/test hyperparameter mismatch degrades performance. No scalar-support-to-token feedback or common/residual-response experiment is described.

### Official-code cross-check

[elvis.py](https://github.com/pavelsuma/ELViS/blob/main/elvis/models/elvis.py), `Elvis.forward`, lines 71–108; [matcher.py](https://github.com/pavelsuma/ELViS/blob/main/elvis/models/matcher.py), lines 21–31; inspected main branch, not commit-pinned.

The implementation remaps each descriptor set, normalizes, predicts each side's dustbin from that side alone, constructs cross-image correlations, applies OT, pools maxima, transforms votes, and sums. `g` runs only in training. No coordinate argument, scalar-M input, or feedback from the final score to descriptors appears in this path. This supports a narrow architectural distinction: new HYP's candidate-conditioned post-LLM adapter is not this ELViS scorer. It does **not** support claiming that ELViS never modifies representations: its learned projection does so. The scorer's lack of coordinate inputs also does not prove that backbone descriptors contain no spatial information.

## 2. To Match or Not to Match — paper facts

Primary: [v2, 22 April 2025](https://arxiv.org/html/2504.06116v2), Sections 4.1–4.3 and Figures 2, 3, 6.

- **Inputs:** MegaLoc's top-100 image candidates and existing image matchers, including RoMa, MASt3R, SuperGlue and LoFTR, with default matching settings.
- **Ranking:** descending geometrically verified inlier count for every candidate, Section 4.1.
- **Confidence:** uncertainty u_q = −i_q^(1), using query–original-top1 inliers, Section 4.2. This differs from a learned relative challenger score.
- **Supervision/calibration:** a logistic regressor is trained on MASt3R inliers from MSLS validation to predict retrieval correctness; Figure 6 transfers its outputs to other datasets. Figure caption/body use opposite wrong/correct conventions; the transferable point is binary correctness calibration, not that wording.
- **Conservative use:** explicitly motivates leaving confident original predictions untouched and reranking uncertain queries. This idea therefore predates new HYP.
- **Evidence boundary:** analyses associate uncertainty with reranking benefit; the method description does not give a trained full-shortlist, cost-sensitive switching policy. Matchers/retriever are evaluated as existing models rather than jointly retrained here.
- **Representation:** no support-conditioned descriptor adapter or common-response decomposition is described. Inlier counts already exemplify spatial matching evidence used through a scalar downstream interface.

### Official-code cross-check

[reranking.py](https://github.com/FarInHeight/To-Match-or-Not-to-Match/blob/main/reranking.py), lines 41–49, sorts every query's candidates by `num_inliers`; no confidence gate is present in that script. [eval.py](https://github.com/FarInHeight/To-Match-or-Not-to-Match/blob/main/vpr_uncertainty/eval.py), lines 46–55, uses original-top1 correctness and original-top1 inliers to compute a precision–recall curve/AUC after monotone normalization. It is an uncertainty evaluation, not a learned action rule. [match_queries_preds.py](https://github.com/FarInHeight/To-Match-or-Not-to-Match/blob/main/match_queries_preds.py), lines 40, 58–64, calls an existing matcher on image pairs. These are inspected main-branch files, not a complete historical repository audit. The paper's logistic-regression figure is not implemented in the inspected evaluation script; do not infer that it was never implemented elsewhere.

## 3. Two additional, operationally close priors

“Closer” here means closer to a particular interface, not globally closer in task or a claim that both appeared after ELViS. Neither paper establishes probability calibration merely by calling a score confidence.

### VGGT-MPR — directly relevant to scalar support

Primary: [v1, 23 February 2026](https://arxiv.org/html/2602.19735v1), Section III-C, equations (2)–(5), Section IV-C.3.

Frozen VGGT processes query/candidate images and mask-guided keypoints. Its tracking-confidence map is aggregated through median confidence, above-threshold fraction, and transformed inverse standard deviation; their weighted sum reranks candidates. The reranking module requires no additional parameter optimization, while global multimodal retrieval is trained separately. This is a particularly close precedent for taking confidence from a frozen geometric foundation model, compressing it to scalars, and correcting retrieval. Its reported mechanism is external confidence scoring, without new HYP's scalar-conditioned post-LLM adapter. It does not present the deletion/retraining or common-response controls in the current new HYP draft. Do not represent the entire multimodal pipeline as training-free.

### FoL++ / Region Matters — reliability weighting and fusion

Primary: [v1, 24 April 2026](https://arxiv.org/html/2604.22390v1), Sections 3.2–3.6, equations (2), (8), (11)–(17).

A learned per-image reliability map combines with aggregation saliency; pseudo-correspondences supervise local descriptors without manually supplied correspondence labels. Local mutual-nearest-neighbor matches receive bilateral weight sqrt(R_q(u)R_c(v)); summation yields S_l, followed by S_final = gamma S_g + S_l, with gamma chosen on validation data. Candidate depth is selected adaptively from global scores. Reliability is described as unconditional per-image matchability, unlike new HYP's pair-conditioned support. Thus reliability maps, bilateral weighting, weak correspondence supervision and global/local fusion are already occupied claims. Its specified interface does not inject a pairwise support scalar into post-LLM query tokens or analyze a common response retaining corrections. This is an end-to-end trained reliability framework, not the same frozen-matcher setting.

## 4. What remains a defensible comparative finding

The following are **claims about the current local experiments**, not architectural firsts established by this search. They require the original result artifacts and remain restricted to their recorded models/panels.

| Current new HYP finding | Comparative assessment after this audit |
|---|---|
| Reference-conditioned support improves identity decisions using only task retrieval labels | Useful task result; neither pair-only learning nor matching-supported reranking is new by itself. |
| Deleting local weighting hurts a fixed head, but retraining the simpler readout restores the overall gain | A specific controlled finding not located in these inspected methods. Distinguish distribution/parameter mismatch from operation necessity. ELViS's own mismatch warning prevents presenting that general lesson as unprecedented. |
| RAW+M explains most external gain; an explicit M×L term adds no stable benefit | An informative diagnosis of this system. Scalar confidence reranking already exists; the contribution is testing the stronger alternatives under the same constraints, not introducing scalar confidence. |
| Same support source works externally and through a post-LLM adapter whose terminal head does not read M | A concrete interface comparison absent from the two principal papers and the two additions as specified. General conditional representation modulation still needs separate prior-art coverage. |
| Common token response retains most internal corrections | The strongest specific mechanistic result in this set: an empirical decomposition of the trained adapter, not a new theorem about attention, spatial evidence or sufficiency. |
| Support creation can involve spatial relations although its scalar downstream use does not preserve positions | Correct distinction, but existing geometric matching-to-scalar pipelines already instantiate it. Novelty must come from the actual upstream/downstream controls, not the logical distinction alone. |

Recommended comparative statement: **The new evidence concerns which uses of pretrained matching support are necessary for this retrieval correction system, and how the same candidate-conditioned support can change decisions through external scoring or internal representation modulation.** Restrict the stronger statement to the controlled simplification, binding, head/content exchange and response-decomposition results. Do not claim the benchmark, small heads, retrieval-only supervision, match confidence, spatial-to-scalar reduction, or conservative reranking as independent firsts.

Remaining search boundary: this note is not an exhaustive audit of FiLM, adapters, conditional metric learning, selective prediction or all VPR literature. It identifies direct overlap and a narrower empirical distinction; it does not certify global novelty.
