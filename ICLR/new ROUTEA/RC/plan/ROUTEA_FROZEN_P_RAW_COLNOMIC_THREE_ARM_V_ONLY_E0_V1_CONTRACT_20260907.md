# Route A frozen-P RAW-ColNomic three-arm V-only E0 V1

Date: 2026-09-07

This is the non-overlapping downstream of the separate full-C128 HYP/P-only
mainline.  It never creates, trains, ranks, calibrates, selects, expands, or
moves a Proposal.  It accepts only a hash-bound frozen connected P view.

For the same opaque candidate pair it emits three fixed RAW-ColNomic arms:

1. `ALL_PATCH_RAW_COLNOMIC`: all query patches against the complete reference;
2. `FROZEN_QUERY_REGION_FULL_REFERENCE_RAW_COLNOMIC`: only each candidate's
   frozen query support, against its complete reference, normalized on the
   shared pair union;
3. `FROZEN_PAIRED_REGION_RAW_COLNOMIC`: the same shared query union, but each
   supported query patch may read only the reference patches paired by P.

Every patch score is cosine MaxSim for the full-reference arms.  The paired arm
uses the arithmetic mean cosine over the frozen paired reference atoms for a
query patch.  Unsupported members of the shared union contribute exact zero;
both-H0 yields exact zero regional arms.  The all-patch arm is a matched
baseline and may not rescue, select, or alter an H0 regional path.

The V core has zero parameters and reads no target, identity, filename, D1
score/rank/gap, P score, selector state, H0 threshold, opened/sealed endpoint,
or action.  Candidate swap must exchange scores and negate every margin.
Changing only frozen reference indices must leave the query-full arm exact and
may change only the paired arm.  Inputs and P hashes must be byte-identical
before and after V.

E0 uses synthetic connected H1/H0 fixtures only.  It authorizes no natural
score, HYP claim, retrieval action, HOLD/SWITCH, ownership or scientific
GO/NO-GO.  Natural execution must wait for the other HYP line's independently
validated full-C128 P seals and a separate authority binding their exact
schema, population and hashes.
