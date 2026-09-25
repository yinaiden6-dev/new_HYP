# Spatial support paper wording and novelty audit

This increment updates the current manuscript and explanatory materials in their original repository directories. It adds the primary-literature audit and the evidence needed by the updated documents. No new experiment was run for this upload.

## Current documents

- [Paper draft](../../ICLR/new%20ROUTEA/RC/reports/RC_NEW_HYP_PAPER_CLOSURE_DRAFT_20260924.md)
- [Unified spatial-support explanation](../../ICLR/new%20ROUTEA/RC/reports/RC_NEW_HYP_UNIFIED_SUPPORT_CLOSURE_20260925.md)
- [Novelty and closest-method audit](../../ICLR/new%20ROUTEA/RC/reports/REPORT_NEW_HYP_NOVELTY_AUDIT_20260925.md)
- [Primary-source audit notes](../../ICLR/new%20ROUTEA/RC/reports/data/new_hyp_novelty_audit_20260925/)
- [Unified evidence and tables](../../ICLR/new%20ROUTEA/RC/reports/data/new_hyp_unified_support_closure_20260925/)
- [Updated writing scope](../../ICLR/new%20ROUTEA/RC/plan/RC_NEW_HYP_THREE_REMAINING_CLAIMS_20260924.md)
- [GroZi internal frozen-transfer result](../../ICLR/new%20ROUTEA/RC/results/rc_postllm_grozi_external_v1/result.json) and [validation](../../ICLR/new%20ROUTEA/RC/results/rc_postllm_grozi_external_v1/validation.json)

The account separates generation and use of reference-conditioned spatial support: external weighting changes patch contributions to matching, while the internal adapter uses pooled M after the LLM and before the retrieval projection. Common-response replay explains much of the tested internal behavior; it does not assert a change to LLM attention or transfer of the full spatial maps.

The new benchmark is assigned to the senior student and is not reported as completed. Literature comparisons do not stand for official same-protocol benchmark reproductions.

## Snapshot semantics

[files.json](files.json) records source-byte hashes, changed files and previous hashes. Historical backup manifests are preserved as records of their own commits. Their verification scripts should be run at those archival commits, not assumed to validate documents revised afterward. The preceding post-LLM attribution archive is commit `aeeeeb42dcee3635aeebea82866267113387c44d`.

In particular, running `tools/verify_restore_mechanism_20260925.py` directly at the new HEAD will report the revised plan's old hash as a mismatch. Run that historical verifier in an isolated checkout of `aeeeeb42dcee3635aeebea82866267113387c44d`; use the command below for this increment. Earlier document snapshots are preserved at:

| Snapshot | Commit |
|---|---|
| Three claims / internal-M archive | `d465b9f41d332cd72f146edea5e652918c5dc75c` |
| Paper and internal-M V3 final | `0f1eead88613d5d875f81bd45ef0dea0f173ed3b` |
| Paper and V4 diagnostics | `5a55100216208478839fb9ff30db83d94e6edb1b` |
| Full post-LLM attribution | `aeeeeb42dcee3635aeebea82866267113387c44d` |

Source documents and scientific JSON are copied byte-for-byte, including original workspace paths. The links above provide repository-relative entrypoints. Weights, image/token caches, original photos and execution logs are not part of this increment.

Verify this increment from the repository root:

```bash
python - <<'PYCODE'
import hashlib, json
from pathlib import Path
m = json.loads(Path('backup/paper_support_novelty_20260925/files.json').read_text())
for r in m['records']:
    assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() == r['sha256'], r['path']
for r in m['repository_entrypoints']:
    assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() == r['sha256'], r['path']
print('PASS:', len(m['records']), 'referenced files')
PYCODE
```
