#!/usr/bin/env python3
"""Publish completed scalar attribution, endpoint vectors and replay code only."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime,timezone

RC=Path(__file__).resolve().parents[1];WS=RC.parents[2]
DEST=WS/'github_exports/new_HYP_20260919';BACKUP=DEST/'backup/m_upstream_completed_20260930'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    names=['rc_m_fine_c128_extension_v1','rc_m_upstream_event_closure_v1','rc_m_fine_c128_attribution_v1','rc_m_predictive_and_fine_c128_chain_v1']
    statuses={}
    for n in names:
        p=RC/'results'/n/'validation.json';v=read(p);assert 'PASS' in v['status'];statuses[n]=v['status']
        if 'result' in v:assert sha(Path(v['result']['path']))==v['result']['sha256']
    review=RC/'results/rc_m_upstream_event_closure_v1/completion_review_20260930/result.json'
    assert read(review)['status']=='UPSTREAM_COMPLETION_SECOND_ACCOUNTING_PASS'
    files={Path(__file__),RC/'programs/analyze_rc_m_upstream_event_results_20260930.py'}
    for n in names:
        root=RC/'results'/n
        for pat in ['*.json','*.md','*.sbatch']:files.update(root.glob(pat))
        for sub in ['phase','context_bridge','completion_review_20260930']:
            for pat in ['*.json','*.md']:files.update((root/sub).rglob(pat))
        # Small completed collection receipts; retain tensor bindings, not tensors.
        files.update(root.glob('query*/gpu_validation.json'))
    for p in [RC/'results/rc_m_fine_c128_extension_v1/authority.json',RC/'results/rc_m_upstream_event_closure_v1/protocol.json']:
        value=read(p)
        for b in value.get('sources',value.get('code_sources',[])):
            src=Path(b['path']);assert sha(src)==b['sha256'];files.add(src)
    for name in ['RC_M_FINE_C128_EXTENSION_V1_20260929.md','RC_M_UPSTREAM_IDENTITY_CLOSURE_20260929.md']:files.add(RC/'plan'/name)
    files.add(RC/'reports/REPORT_M_UPSTREAM_COMPLETED_20260930.md')
    records=[];changed=0
    for src in sorted(files):
        assert src.is_file() and src.suffix in ['.json','.md','.py','.sbatch','.sh','.csv']
        relative=src.relative_to(WS);dst=DEST/relative;before=sha(src);dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists() or sha(dst)!=before:shutil.copyfile(src,dst);changed+=1
        assert sha(src)==sha(dst)==before
        records.append(dict(source=str(relative),export=str(relative),sha256=before,bytes=src.stat().st_size))
    BACKUP.mkdir(parents=True,exist_ok=True)
    metadata=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),base_commit=subprocess.check_output(['git','-C',str(DEST),'rev-parse','HEAD'],text=True).strip(),
        files=records,statuses=statuses,changed=changed,bytes=sum(x['bytes'] for x in records),
        excluded=['images','model weights','tensor/token/J/P/confidence/patch caches','err/out logs'])
    (BACKUP/'files.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (BACKUP/'validation.json').write_text(json.dumps(dict(status='COMPLETED_RESULTS_EXPORT_HASH_PASS',files=len(records),bytes=metadata['bytes']),indent=2)+'\n')
    (BACKUP/'README.md').write_text('''# Completed upstream attribution:19 queries, natural C128

This archive supersedes the pending execution status of the2026-09-29 snapshot. All original11-query work and the8-query extension, including collection, frozen external/POST replay, matched-registration controls and final accounting, passed validation. The snapshot preserves source-relative paths and SHA256 bindings. A second read-only review recomputed1539 decisions and candidate-level/group contrasts; maximum arithmetic difference was1.11e-15.

The19-query cohort includes the complete union of original rescue/break events of three frozen models within the opened first128 H593 queries, plus the unchanged8 controls. This is not a new19-query benchmark or a population accuracy estimate.

Findings:
- The tested phase and registration perturbations retained all original rescues.
- Amplitude flattening or permutation retained10/10 rescues for each external head, but6/8 for POST; the lost cases were OUTCOME-0476 and DIFFICULT-0013.
- Removing the whole P input retained7/10,6/10 and2/8 rescues for NATIVE7, M_FREE and POST respectively. These are rescue-retention counts, not total correct counts.
- A target advantage over the average wrong candidate does not establish an advantage over the strongest native-M wrong candidate. The latter was not stably improved by the main property contrasts.
- OUTCOME-0476 lost its POST correction even while its target/RAW-wrong M ratio increased. Absolute values and the downstream response cannot be omitted from the explanation.

The POST coarse-native anchor itself introduces a break on OUTCOME-0334 that the original HR1 anchor did not have. Preserve that distinction when comparing interventions. Cohorts, group sizes, competing-candidate definitions, per-world breaks and selected-event limitations are recorded in the review. There is no unique-root-cause or new accuracy claim.

Read `ICLR/new ROUTEA/RC/reports/REPORT_M_UPSTREAM_COMPLETED_20260930.md` and `results/rc_m_upstream_event_closure_v1/completion_review_20260930/`. Full scalar M/content/action vectors are retained under the original result roots. Images, model and intermediate tensors and runtime logs remain local; their provenance bindings are included where needed. No new training or experimental jobs were submitted during this completion review.
''')
    print(json.dumps(dict(files=len(records),changed=changed,MB=metadata['bytes']/1e6)))

if __name__=='__main__':main()
