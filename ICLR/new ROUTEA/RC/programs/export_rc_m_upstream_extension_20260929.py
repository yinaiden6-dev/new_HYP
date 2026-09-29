#!/usr/bin/env python3
"""Archive source, scalar results and queued attribution contracts; no tensors/images/logs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime,timezone

RC=Path(__file__).resolve().parents[1];WS=RC.parents[2]
DEST=WS/'github_exports/new_HYP_20260919'
BACKUP=DEST/'backup/m_upstream_event_extension_20260929'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    files={Path(__file__)}
    for name in ['run_rc_m_fine_c128_extension_v1.py','submit_rc_m_fine_c128_extension_v1.py',
                 'run_rc_m_upstream_event_closure_v1.py','submit_rc_m_upstream_event_closure_v1.py',
                 'analyze_rc_post_m_predictive_response_v1.py','analyze_rc_m_fine_c128_phase_scope_v1.py']:
        files.add(RC/'programs'/name)
    for name in ['rc_m_fine_c128_extension_v1.sbatch','rc_m_upstream_event_closure_v1.sbatch']:files.add(RC/'slurm'/name)
    for name in ['RC_M_FINE_C128_EXTENSION_V1_20260929.md','RC_M_UPSTREAM_IDENTITY_CLOSURE_20260929.md']:files.add(RC/'plan'/name)
    for name in ['REPORT_M_UPSTREAM_EXTENSION_SUBMITTED_20260929.md','REPORT_M_CONTEXT_BINDING_SCIENTIFIC_READING_20260927.md','REPORT_M_CONTEXT_BINDING_INDEPENDENT_INTERPRETATION_20260927.md']:
        files.add(RC/'reports'/name)
    for p in [RC/'results/rc_m_fine_c128_extension_v1/authority.json',RC/'results/rc_m_fine_c128_attribution_v1/protocol.json']:
        x=read(p)
        for b in x.get('sources',x.get('code_sources',[])):
            f=Path(b['path']);assert sha(f)==b['sha256'];files.add(f)
    for name in ['rc_m_fine_c128_extension_v1','rc_m_upstream_event_closure_v1','rc_m_predictive_and_fine_c128_chain_v1']:
        root=RC/'results'/name
        for pat in ['*.json','*.md','*.sbatch']:files.update(root.glob(pat))
    root=RC/'results/rc_post_m_predictive_response_v1'
    for pat in ['*.json','*.md']:files.update(root.rglob(pat))
    root=RC/'results/rc_m_fine_c128_attribution_v1'
    for pat in ['*.json','*.md']:files.update(root.glob(pat));files.update((root/'phase').rglob(pat))
    for name in ['rc_m_coarse_path_correction_audit_v1','rc_m_context_binding_v1/independent_interpretation']:
        for pat in ['*.json','*.md']:files.update((RC/'results'/name).glob(pat))
    records=[];changed=0
    for src in sorted(files):
        assert src.is_file() and src.suffix in ['.json','.md','.py','.sbatch','.sh','.csv']
        rel=src.relative_to(WS);dst=DEST/rel;before=sha(src)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists() or sha(dst)!=before:shutil.copyfile(src,dst);changed+=1
        assert sha(dst)==before and sha(src)==before
        records.append(dict(source=str(rel),export=str(rel),sha256=before,bytes=src.stat().st_size))
    BACKUP.mkdir(parents=True,exist_ok=True)
    metadata=dict(schema='M_UPSTREAM_EVENT_EXTENSION_ARCHIVE_V1',snapshot_utc=datetime.now(timezone.utc).isoformat(),
        previous_commit=subprocess.check_output(['git','-C',str(DEST),'rev-parse','HEAD'],text=True).strip(),
        files=records,files_copied_or_updated=changed,bytes=sum(r['bytes'] for r in records),
        excludes=['original images','model/checkpoint tensors','token/descriptor/J/P/confidence tensor caches','err/out logs'],
        state='New19-query extension submitted; no new scientific completion claim. Prior predictive and phase results included.')
    (BACKUP/'files.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (BACKUP/'validation.json').write_text(json.dumps(dict(status='EXPORT_SOURCE_HASHES_PASS',files=len(records),bytes=metadata['bytes']),indent=2)+'\n')
    text='''# Expanded upstream attribution: source and submission archive

This update addresses the limitation that the earlier fine-property analysis covered only three historical rescue cases. The new retrospective event census contains every original rescue/break within the opened first128 H593 queries, plus the unchanged8 controls:19 queries, each with its complete natural ColNomic C128.

- NATIVE7 and M_FREE:10 original rescues and1 original break each.
- POST:8 original rescues and0 original breaks.
- Reuse the completed11-query phase results; add8 missing query axes only.
- Collect missing coarse descriptors/J/P once. Reuse them for phase/amplitude and CPU registration interventions. No retraining or new fine-refiner/ColNomic encoding.

Submitted chains:5170218 pilot ->5170219 collection ->5170220 frozen replay ->5170221 phase accounting. The same capture feeds5170229 preparation ->5170230 CPU registration ->5170231 mass validation ->5170232 frozen replay ->5170233 combined identity-selectivity analysis. The last task also requires the historical11-query final validation5168069. Each allocation is10 minutes with bounded same-ID continuation.

The added analysis separates a common increase in candidate support from preferential support for the correct identity, retains all127 competing references, and connects each world to original rescue loss/retention and breaks. Candidates sharing one query are not independent samples. Original-event selection is explicit; this is not a new population accuracy benchmark.

These are submitted protocols, not completed expanded scientific results. Previous nine-group registration evidence increased both target and wrong absolute support; it did not establish stable preferential logM separation. Completing this workload is not automatically proof of a unique upstream cause.

Source and scalar artifacts retain their workspace-relative paths under `ICLR/new ROUTEA/RC/`. Follow the two `RC_M_*20260929.md` plans and the submitted launchers for replay in the original workspace. `files.json` verifies the exported bytes. Absolute artifact bindings preserve provenance; excluded local tensors and model/data dependencies are required for new forward execution and are not embedded in this Git archive.
'''
    (BACKUP/'README.md').write_text(text)
    print(json.dumps(dict(files=len(records),changed=changed,MB=metadata['bytes']/1e6,archive=str(BACKUP))))

if __name__=='__main__':main()
