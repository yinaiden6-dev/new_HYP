# M attribution: candidate discrimination, property restoration and two downstream uses

This experiment asks **which information makes the compressed RoMa support score M distinguish the correct reference from a confusing reference, and how that distinction reaches external calibration and internal content modulation**. It is an attribution study of existing mechanisms, not another QR/QRR model search or an attempt to select the highest accuracy from new heads.

**The 128-image main chain, new nine-group property replication, same-model downstream propagation and bounded property-compensation diagnostic are complete. All nine required experiment receipts pass.** The finite four-step analysis is complete within the tested interventions and model families; unique causality and unrestricted irreplaceability are not established.

Start with the [final analysis and conclusions](<../../ICLR/new ROUTEA/RC/reports/REPORT_M_CAUSAL128_FINAL_ANALYSIS_20260927.md>) and [independent interpretation review](<../../ICLR/new ROUTEA/RC/reports/REPORT_M_CAUSAL128_INTERPRETATION_REVIEW_20260927.md>). The [captured completion status](EXPERIMENT_STATUS.md), [files.json](files.json) and [validation.json](validation.json) describe the precise export snapshot, hashes and omitted artifacts. Initial preparation occurred while the jobs were running; final publication refreshes that snapshot after independent acceptance. Export-byte validation is distinct from scientific acceptance and from successful GitHub push.

## Main findings, including the counterexample

On the nine groups outside the earlier F71 mechanism-development groups, restoring consistent relative phase structure while amplitude assignments remain permuted improves the target-minus-fixed-wrong logM gap and both existing downstream mechanisms. The very same frozen external head and POST_REAL model receive the changed M; no intervention-specific head is fitted for this comparison.

| Same nine-group intervention | Change in logM gap | Change in NATIVE7 action gap | Change in POST content gap |
|---|---:|---:|---:|
| Permuted amplitude: GLOBAL minus LOCAL phase | +0.024178 | +0.017087 | +0.002179 |
| Native amplitude: GLOBAL minus LOCAL phase | +0.017600 | +0.014301 | **−0.000429** |

For the first row, all three endpoints are positive in 8/9 groups; exploratory 95% group-bootstrap intervals are respectively [0.007618, 0.040450], [0.003293, 0.032702], and [0.000817, 0.003567]. These are continuous margin effects on fixed pairs, not additional correctly retrieved images.

**The second row is an equally important result:** the logM gap improves but internal content discrimination becomes worse (POST interval [−0.000863, −0.000053], positive in only 2/9 groups). Better M separation therefore does not guarantee that every internal or external endpoint improves. POST reads each candidate's absolute standardized logM together with content; the effect is not determined by the scalar gap alone. The shared interface supports conditional use by both mechanisms, not mathematical equivalence or universally monotonic propagation.

On the 128-image input-path comparison, adding P with J present increases the group-mean logM gap by 0.613814; adding P without the direct J input decreases it by 0.076750. P's usefulness depends on context. P is itself constructed from joint computation, so this does not separate J and P into independent causal sources.

The compensation test also limits the claim. After certified optimization on the old seven groups, native-amplitude LOCAL retains 0.020685 more test ranking loss than the GLOBAL control, with exploratory interval [0.002876, 0.045092]: this bounded readout did not compensate for that disturbance. Yet amplitude-permuted cells retain useful ranking and lower mean losses in this test. All four refitted cells have positive mean margins in 7/9 groups. This supports conditional mechanism sets and some uncompensated effects in the tested family; **it does not prove that a property is intrinsically irreplaceable**, that one decomposition is unique, or that the refitted cells improve full-C128 retrieval.

## Protocol and panels

| Component | Data and model | What it tests |
|---|---|---|
| Main input-path analysis | 128 images from the frozen H593 execution manifest; each image retains its own natural ColNomic C128 and original five-fold group assignment | Conditional effects of two RoMa coarse-predictor inputs on candidate-relative M and downstream margins |
| Frozen downstream propagation | The same 128 images and candidates, original per-fold external NATIVE7/M_FREE and internal POST_REAL endpoints | Whether the same altered M affects external decision margins and internally modulated content/decision margins |
| Deleted-input compensation | Four input worlds × original five folds = 20 PRODUCT5 COST1 fits; fixed training recipe and zero decision threshold | Whether the tested alternative inputs can compensate after retraining the original model family |
| Historical property-restoration propagation | Seven target-present groups from the opened eight-query panel; seven target-versus-wrong comparisons, 14 query/reference candidate pairs × 18 arms | Whether controlled amplitude/phase restoration changes M gaps and the two frozen downstream paths |
| New-group property replication | Nine groups outside the earlier F71 explanation-development groups; nine target-versus-wrong comparisons, 18 query/reference candidate pairs × 18 arms | Whether the predeclared amplitude/phase contrasts repeat beyond the groups used to develop the explanation |
| New-group downstream property bridge | The same nine comparisons, 18 query/reference candidate pairs and 18 arms; frozen original external and POST_REAL endpoints | Whether the property-to-M contrasts also reach the same external/internal paths on these new groups |
| Specific-property compensation | Train on the historical seven groups, test on the nine prespecified groups; four amplitude/phase cells plus RAW/L_ONLY | Whether a matched, bounded scalar readout can compensate after a property is damaged |

The main 128 images are a subset of H593, **not automatically the older EVAL128 panel**. The nine replication groups were not used to select this property explanation, but remain part of historical H593; this is not untouched external-dataset confirmation. Sparse target/wrong interventions cannot establish complete C128 accuracy. Missing-target and eligible-pair counts are recorded in the individual results.

The original new nine-group experiment repeats the upstream property-to-M comparison. A separately frozen cache-only extension propagates all 18 arms through the existing external and internal endpoints on these same groups. It requires its own `NEWGROUP_PROPERTY_BRIDGE_INDEPENDENT_PASS` receipt; the earlier upstream receipt alone cannot establish downstream replication. The final export requires this additional receipt and the specific-property compensation receipt as well as all seven original receipts.

## What A, J, P and M mean here

- **A:** appearance input to the frozen coarse predictor.
- **J:** its direct jointly computed query/reference context input.
- **P:** its correspondence-distribution embedding. This embedding was itself generated from the original joint computation. Removing the direct J input does **not** remove all joint information already carried by P.
- **M:** the scalar support derived from both sides' average weights, `sqrt(mean(u) * mean(v))`.

The four worlds are `A1J1P1`, `A1J1P0`, `A1J0P1` and `A1J0P0`. Conditional contrasts compare P with/without J, J with/without P, and their interaction. The original HR1 deployment result is a separate anchor; it must not be confused with the native coarse intervention baseline.

Propagation changes M and its required descendants while holding the original local-content shape fixed. Consequently these are **effects through the M path**, not the total effect of changing every downstream use of the spatial weights. The internal path uses the already trained POST_REAL adapter and its actual content scores; it does not fit a new model for each intervention.

The label-free single-image-product check asks whether M could reduce to `a(query) * b(reference)`, using crossed log-score differences on shared references. Rejecting that restricted explanation establishes nonseparability, **not a newly discovered category beyond pairwise information**, and not by itself identity usefulness. Overlapping crossed comparisons are not independent observations.

## How to read the evidence

The intended chain is:

`controlled input/property change → target versus fixed-wrong M gap → external decision gap / internal content and decision gap`.

The four scientific requirements are kept separate: damage a property, restore it, permit compensation after deletion, and repeat on groups not used to choose the explanation. Each now has an accepted experiment in a stated finite scope. The completed whole-J/P-input retraining comparison does **not** substitute for the separately completed, fixed-pair phase/amplitude compensation test. Reports state which mechanism level each comparison supports; they do not assert a unique causal decomposition or that M is sufficient for identity.

GLOBAL applies the same phase rotation to every P patch, retaining their relative inner products. LOCAL applies patch-dependent signs while preserving each patch's spectral power. AMP_PERMUTE reassigns patch norms while retaining the norm distribution and each vector's direction. Thus GLOBAL−LOCAL tests more than a simple amplitude change, but GLOBAL is **not the original untransformed NATIVE P**. Both transformations may also alter compatibility with the fixed A/J inputs. These computational interventions do not uniquely isolate physical geometry or certify recovered object regions.

Accuracy is secondary to the continuous, fixed-opponent comparisons in this study. Previously published H593 scores and the original 481/492 heads are not replaced by numbers from this 128-image subset. A change in M's mean is also distinct from a change in the correct-versus-wrong gap.

## Additional bounded compensation and propagation diagnostics

The two cache-only extensions completed as jobs `5167749` (property compensation), `5167750` (new-group downstream bridge) and `5167751` (bridge join), with separate independent acceptance receipts. Their final results complement the original main-chain summary rather than silently rewriting its earlier scope.

The property-compensation design trains on the historical seven groups and tests once on the nine prespecified groups. Four cells cross native/permuted amplitude assignment with global/local phase transformations; each group averages four predeclared direction/sign repeats. A four-dimensional antisymmetric linear readout uses RAW difference, M×free-content contrast, M contrast and free-content contrast. Four matched cell fits and a RAW/L_ONLY fit use deterministic, strictly convex regularized ranking loss, with numerical optimality certificates. Parameters and scales are sealed before reading the new-group intervention values. This is **not the original COST1 objective** and not retraining RoMa. It tests bounded readout compensation, not unrestricted information necessity, complete C128 accuracy or target-free deployment.

The new-group bridge makes no new RoMa, vision-encoder or LLM forward passes and performs no fitting. It sends each saved intervention M through the original held-fold NATIVE7/M_FREE and POST_REAL endpoints, saving target and fixed-wrong effects separately as well as their differences. Only the two preselected candidates are intervened on. If the RAW winner is a third candidate, its original HR1 M/L anchor stays fixed. This is an M-path propagation test, not a full-candidate intervention or complete mediation proof.

Both extensions use already generated caches. Their results belong alongside, rather than being substituted for, the full128 input-path analysis and the earlier seven-group restoration.

| Completed property analysis | Report and tables | Accepted scope |
|---|---|---|
| Upstream restoration replication | [Phase/amplitude report](<../../ICLR/new ROUTEA/RC/results/rc_m_causal128_attribution_chain_v1/phase_replication/report.md>) | Nine groups, 18 fixed candidates × 18 arms; effects on M |
| Same-model downstream propagation | [New-group bridge report](<../../ICLR/new ROUTEA/RC/results/rc_m_causal128_attribution_chain_v1/newgroup_property_bridge_v1/report.md>) · [endpoint CSV](<../../ICLR/new ROUTEA/RC/results/rc_m_causal128_attribution_chain_v1/final_analysis_v1/endpoint_effects.csv>) | Those same property interventions reach frozen external/internal endpoints; the sign-reversal counterexample is retained |
| Matched property compensation | [Compensation report](<../../ICLR/new ROUTEA/RC/results/rc_m_causal128_attribution_chain_v1/property_compensation_fixed_pairs_v1/report.md>) · [compensation CSV](<../../ICLR/new ROUTEA/RC/results/rc_m_causal128_attribution_chain_v1/final_analysis_v1/property_compensation.csv>) | Old seven-group training, new nine-group test, five certified low-dimensional fits; bounded compensation only |

An [independent second review](<../../ICLR/new ROUTEA/RC/results/rc_m_causal128_attribution_chain_v1/independent_property_review_v1/validation.json>) verifies 478 source bindings, disjoint training/test components, features, fitting certificates and reported effects. A separate [actual POST-input and patch replay audit](<../../ICLR/new ROUTEA/RC/results/rc_m_causal128_attribution_chain_v1/independent_property_review_v1/post_absolute_input_validation.json>) checks all 324 candidate/arm cases: the supplied condition is the recorded absolute M transformation, hidden content remains fixed across arms, and patch outputs and MaxSim argmax replay exactly. These are implementation and arithmetic checks, not additional independent data.

## Results and provenance entry points

All source-relative paths remain under `ICLR/new ROUTEA/RC/`:

- [Frozen attribution plan](<../../ICLR/new ROUTEA/RC/plan/RC_M_CAUSAL128_ATTRIBUTION_CHAIN_V1_20260927.md>)
- [Original submission and chain description](<../../ICLR/new ROUTEA/RC/reports/REPORT_M_CAUSAL128_ATTRIBUTION_CHAIN_SUBMITTED_20260927.md>)
- [Pilot-save repair and completed-branch check](<../../ICLR/new ROUTEA/RC/reports/REPORT_M_CAUSAL128_PROGRESS_AND_PILOT_SAVE_REPAIR_20260927.md>)
- [Result root: protocols, source bindings, submission records and validation receipts](<../../ICLR/new ROUTEA/RC/results/rc_m_causal128_attribution_chain_v1>)
- [Path extraction and deletion/retraining](<../../ICLR/new ROUTEA/RC/results/rc_m_causal128_attribution_chain_v1/paths/report.md>)
- [Frozen external/internal propagation](<../../ICLR/new ROUTEA/RC/results/rc_m_causal128_attribution_chain_v1/bridge/report.md>)
- [Restricted single-image-product check](<../../ICLR/new ROUTEA/RC/results/rc_m_causal128_attribution_chain_v1/pair_structure/report.md>)
- [Specific-property compensation design](<../../ICLR/new ROUTEA/RC/reports/REPORT_M_CAUSAL128_PROPERTY_COMPENSATION_DESIGN_20260927.md>)
- [Final machine-readable result](<../../ICLR/new ROUTEA/RC/results/rc_m_causal128_attribution_chain_v1/final_analysis_v1/result.json>) · [all-nine-receipt analysis validation](<../../ICLR/new ROUTEA/RC/results/rc_m_causal128_attribution_chain_v1/final_analysis_v1/validation.json>)
- The original sealed `final_summary/` predates the two add-ons and preserves its historical scope. Use the final analysis above for the complete conclusion; do not treat that older summary's remaining-work statements as the current status.

The original failed pilot save and cancelled dependents are preserved as execution history. Successor jobs reuse the completed scientific branches and the saved pilot forward outputs. The pilot repair does not change the scientific transformations or create a new experimental arm.

## Reproduction and archive limits

Completion refers to this frozen batch, not to exhausting every causal explanation. Two remaining identification questions—separating the common log-M level from the target/wrong gap in POST, and separating P's internal relative structure from its binding to A/J—are recorded in the [unexecuted follow-up design](<../../ICLR/new ROUTEA/RC/plan/RC_M_CAUSAL128_REMAINING_IDENTIFICATIONS_20260927.md>). They are not included among this batch's completed experiments.

The execution entry point is [submit_rc_m_causal128_attribution_chain_v1.py](<../../ICLR/new ROUTEA/RC/programs/submit_rc_m_causal128_attribution_chain_v1.py>) (`prepare`, `submit`, `run`); the scoped successor is [submit_rc_m_causal128_phase_save_repair_v1.py](<../../ICLR/new ROUTEA/RC/programs/submit_rc_m_causal128_phase_save_repair_v1.py>). These retain absolute workspace paths, frozen source hashes, original model/cache provenance, Slurm settings, per-stage checkpoints and independent validators. Do not blindly submit them from a fresh clone: restore their recorded dependencies and adapt execution paths under a new protocol first.

The add-on entry points are `programs/run_rc_m_causal128_property_compensation_v1.py` (`prepare`, `fit`, `evaluate`, or `run`) and `programs/run_rc_m_causal128_newgroup_property_bridge_v1.py` (`prepare`, `worker`, `join`), with launcher `slurm/rc_m_causal128_final_addons_v1.sbatch`. Their frozen protocols distinguish old-group training from new-group evaluation and bind the original candidate axes and endpoints.

The final analysis is rebuilt by `programs/summarize_rc_m_causal128_final_analysis_v1.py`, which checks the nine experiment receipts and bound artifacts before writing the result, endpoint/compensation tables and final report. It does not rerun the scientific experiments or select new models.

The 128-image analysis reuses existing intervention caches. The nine-group restoration branch requires 19 previously missing single-image descriptors; it encodes those once with the frozen CPU encoder, then captures the required matching intermediates. Scheduler requests for one GPU are resource-placement choices: this chain's computations remain on CPU.

Included: source, plans, reports, original group/candidate manifests, scalar head JSON, full available per-candidate scalar results, per-pair receipts, training summaries, validations, source hashes, submission and repair metadata. Excluded: model/adapter/optimizer `.pt` checkpoints, raw images, token/feature/confidence tensors (`.pt`, `.npz`, etc.), locks, process IDs and runtime logs. The exclusion list identifies the exact omitted paths and sizes. Scalar receipts are not substitutes for those tensors; full patch-level replay requires the original workspace or regeneration of the recorded inputs.

## Refreshing this export

From the export checkout, run:

```bash
python3 tools/export_m_causal128_attribution_20260927.py
```

This synchronizes only this chain and its text/source dependencies, verifies SHA256 and scans for credential-like content, then refreshes the manifest and captured status. It does not train models, change predictions, stage files, commit, or push. Files are individually stable snapshots rather than a global transaction across running jobs; repeat after all final receipts exist before treating the export as the completed experiment.
