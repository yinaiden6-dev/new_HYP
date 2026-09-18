# new HYP ISIC: fixed product-head exploratory transfer

User authorized an ISIC experiment on 2026-09-14. This independent exploratory
branch tests the current frozen product-trained mechanism on exact-lesion
retrieval. It does not replace, modify, or tune against GroZi/RPC results.
It is a first zero-update transfer test, distinct from the older proposed
patient-disjoint domain-training / A-B-C architecture experiment.

## Data fixed before image inspection and model evaluation

Use the existing 989 local IMA++ JPEGs and public img_metadata.csv. The existing
collection was assembled for historical experiments, including mask/class
availability filters: it is an opened, selected cohort, not a new untouched
medical endpoint. This run reads only image ID, lesion ID, patient ID and RGB
bytes. No diagnosis, mask, age, sex, site, or previous model prediction is used
for selection, scoring, or fitting. Patient ID is curator-only for grouping.

Retain lesions with at least two distinct local images, complete patient IDs
and a consistent single patient. Metadata screening gives 927 images, 390
lesions and 346 patients; exclude the 62 images of 28 lesions missing patient
information before any model results. Check source byte and EXIF-RGB hashes;
if duplicates or conflicting identities are found, stop before inference and
record the problem rather than silently replacing examples.

One reference per lesion is the smallest SHA256 of
`ISIC_TRANSFER_V1_20260914|reference|<image_id>`; all other images are queries.
Physical lesion ordering uses an independent fixed hash; query ordering uses
another hash. Expected 390 references and 537 queries, with anonymous worker
IDs. The curator mapping is excluded from prediction workers. Reference
selection is not based on visible quality or any model score.

## Frozen computation

Reuse the externally sealed full-H593 head bundle unchanged: COST1 primary,
CE secondary, COST4, GROUP_COST4, RAW2_CE controls and untrained RAW. Frozen
ColNomic/RoMa, original full-reference soft visibility/MaxSim, original six
statistics/seven parameters, natural C128 from the full 390-reference gallery,
all127 challengers, threshold0 HOLD/SWITCH. No target insertion. Preserve RAW
FP32 reduction followed by FP64 storage, C4/head FP64, stable physical tie order.
No ISIC training, calibration, threshold choice or checkpoint selection.
CBIND rotates the full candidate evidence by64 on the physical C128 axis,
holding RAW values fixed. No reuse of historical ISIC-trained heads.

Reference and query processor-frame conventions are inherited unchanged from
the qualified RPC pipeline and reported as such. Adapt only dataset paths,
gallery/query counts, partial last shard, and patient-group statistics. Pin
all operators and heads; retain CPU C4 replay and independent NumPy actions.

## Complete-cohort reporting and exploratory decision

Seal all537 predictions before joining identities. Report accuracy, MRR,
recallC128 (all missing targets remain failures in the denominator), rescue,
break, wrong-to-wrong, HOLD, per-patient and per-lesion results.

Use346 patient clusters; within a patient all query outcomes remain together.
Primary interval: equal-patient mean paired accuracy difference, percentile
bootstrap100000 draws, seed20260914. Report query-micro difference separately;
it is not the estimand of that interval. Predeclared positive exploratory
signal requires COST1 to beat RAW, COST4, GROUP_COST4, RAW2_CE and its own CBIND
in query net, equal-patient mean, and each paired95% lower bound. CE remains
secondary regardless of results. Failure is reported as failure of frozen
product-head transfer on this cohort, not impossibility of domain-trained
reference-conditioned retrieval. No claim of clinical diagnosis, spatial
ownership, no-text causality, untouched external confirmation, or universal
generality. Unknown backbone pretraining exposure remains unexcluded.

## Execution

GPU reference encoding15min. Query RAW10min and RoMa15min per8-query shard;
68 shards (last1query), pilot shard0 before remaining67, concurrency at most46
per stage, then patient-group join10min. Verify natural pilot execution before
leaving jobs to run. CPU image integrity precedes reference encoding.
All outputs are new under results/rc_new_hyp_isic_transfer_v1. Previously sealed
RPC/GroZi/product heads and prediction results remain immutable.
