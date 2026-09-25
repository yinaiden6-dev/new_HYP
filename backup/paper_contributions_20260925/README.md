# Four-part paper contribution update — 2026-09-25

Current writing combines **existing-retriever benchmarking, a reusable spatial-support interface, a unified framework, and controlled mechanism findings**. The central method and knowledge claim is the spatial-support interface together with evidence explaining its external and internal uses.

## Read the updated documents

- [Paper draft: four contributions, Chinese contribution paragraph, English abstract and contribution paragraph, and paper outline](../../ICLR/new%20ROUTEA/RC/reports/RC_NEW_HYP_PAPER_CLOSURE_DRAFT_20260924.md)
- [Unified spatial-support framework and evidence](../../ICLR/new%20ROUTEA/RC/reports/RC_NEW_HYP_UNIFIED_SUPPORT_CLOSURE_20260925.md)
- [Novelty audit and contribution positioning](../../ICLR/new%20ROUTEA/RC/reports/REPORT_NEW_HYP_NOVELTY_AUDIT_20260925.md)
- [Writing scope and completed historical tasks](../../ICLR/new%20ROUTEA/RC/plan/RC_NEW_HYP_THREE_REMAINING_CLAIMS_20260924.md)

The benchmark is a comparison of existing retrievers, coordinated by the senior student, not a new dataset. Its uncollected model results remain pending. The external interface has cross-retriever experiments; the internal adaptation is scoped to ColNomic. ColQwen-base is kept distinct from retrieval-trained ColQwen.

The text now distinguishes framework definitions, algebraic properties, and empirical mechanism findings. External weighting changes patch contributions to scores. Internal modulation reads pooled M after the LLM and before the retrieval projection. Current experiments and their original lineage are unchanged.

## Validation and snapshot scope

This is a documentation-only update. The validation checked 11 documents, 116 local links, 17 unchanged scientific-source hashes, and three unchanged evidence/table hashes. A second read-only review checked claim scope and consistency. No training or new experiment was run. No photos, model weights, token caches, or execution logs are added.

[files.json](files.json) records source-byte hashes and prior hashes. [validation.json](validation.json) records this update's checks. Source documents retain their original paths; workspace-absolute links were checked against the corresponding exported files. Older archive manifests remain snapshots of their own commits. The preceding full GroZi experimental archive is commit `46d2a56bd36caff609c98cc51c83d20335a8f322`; the preceding paper/novelty archive is `82cb2889214ea5db34c89e4a91a3634968ef5139`.

Run from the repository root at this documentation snapshot:

```bash
python - <<'PYCODE'
import hashlib, json
from pathlib import Path
m = json.loads(Path('backup/paper_contributions_20260925/files.json').read_text())
for r in m['records'] + m['repository_entrypoints']:
    assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() == r['sha256'], r['path']
for r in m['scientific_sources']:
    p = Path('ICLR/new ROUTEA/RC') / r['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == r['sha256'], str(p)
print('PASS: contribution documents and unchanged scientific sources')
PYCODE
```
