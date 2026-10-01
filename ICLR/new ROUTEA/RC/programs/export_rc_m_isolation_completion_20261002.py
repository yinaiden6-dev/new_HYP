#!/usr/bin/env python3
"""Archive this experiment only; omit images, tensor caches, weights and logs."""
import hashlib
import json
from pathlib import Path
import re

RC = Path(__file__).resolve().parents[1]
WS = RC.parents[2]
DEST = WS / 'github_exports/new_HYP_20260919'
BACKUP = DEST / 'backup/m_structure_binding_isolation_complete_20261002'
RESULTS = ['rc_m_conditional_prediction_v1', 'rc_m_structure_binding_isolation_v1',
           'rc_m_structure_binding_isolation_v2']


def main():
    files = set()
    for name in RESULTS:
        root = RC / 'results' / name
        # Snapshot of completed atomic scalar/text outputs only.
        files.update(p for p in root.rglob('*') if p.is_file() and p.suffix in {'.json','.md','.sbatch'})
        protocol = json.loads((root/'protocol.json').read_text())
        for source in protocol.get('code_sources', []) + ([protocol['code']] if 'code' in protocol else []):
            p = Path(source['path'])
            assert p.is_relative_to(WS)
            assert hashlib.sha256(p.read_bytes()).hexdigest() == source['sha256']
            files.add(p)
    files.update(RC/'programs'/n for n in [
        'audit_rc_m_conditional_prediction_v1.py',
        'audit_rc_m_structure_binding_isolation_v2.py',
        'summarize_rc_m_isolation_contrasts_20261002.py',
        'run_rc_m_isolation_dev_cpu_resume_20261001.py',
        'export_rc_m_identification_20261001.py',
        'submit_rc_m_identification_20260930.py',
        'submit_rc_m_isolation_precision_repair_20261001.py',
        Path(__file__).name,
    ])
    files.update(RC/'slurm'/n for n in [
        'rc_m_identification_20260930.sbatch', 'rc_m_isolation_precision_repair_20261001.sbatch',
        'rc_m_isolation_dev_cpu_resume_20261001.sbatch'])
    files.update(RC/'reports'/n for n in [
        'REPORT_M_CONDITIONAL_PREDICTION_AND_ISOLATION_REPAIR_20261001.md',
        'M_COMPLETED_ARTIFACT_STATUS_20261001.json',
        'REPORT_M_STRUCTURE_BINDING_ISOLATION_COMPLETE_20261002.md'])
    files.add(RC/'plan/RC_M_CONDITIONAL_PREDICTION_AND_UPSTREAM_ISOLATION_20260930.md')
    entries=[]
    for p in sorted(files):
        raw=p.read_bytes()
        assert raw==p.read_bytes(), ('CHANGED_DURING_EXPORT',str(p))
        assert len(raw)<90*1024**2
        assert not re.search(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)',raw), ('SECRET_PATTERN',str(p))
        if p.suffix=='.json': json.loads(raw)
        relative=p.relative_to(WS);target=DEST/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() or target.read_bytes()!=raw: target.write_bytes(raw)
        entries.append(dict(path=str(relative),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    BACKUP.mkdir(parents=True,exist_ok=True)
    manifest=dict(scope='Completed upstream structure-binding isolation, independent audit, numeric repair and CPU migration provenance; includes H593 conditional prediction archive',
        complete='Upstream25queries77pairs1001worlds accepted; independent lineage and endpoint audit passed. Held-out30models on570queries already accepted.',
        pending=None,
        excluded=['original images','model weights','tensor/feature caches','dense confidence tensors','patch NPZ arrays','err/out logs','locks'],
        files=entries,total_bytes=sum(x['bytes'] for x in entries))
    (BACKUP/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (BACKUP/'README.md').write_text("""# Completed P structure and content-binding isolation

Jobs 5174143 (CPU continuation) and 5173348 (join) completed with exit 0:0. The final result has 25 queries in 23 identity components, 77 query/reference pairs and 13 intervention arms (1001 candidate worlds). This is a fixed-opponent mechanism panel, not a complete-C128 accuracy experiment.

The independent audit checks source hashes, reused versus recomputed provenance, all world labels, M pooling, POST patch means, candidate differences, contrast arithmetic and group means. Maximum M recount error is 1.11e-16; POST and contrast recount errors are zero. The original invariant gates remain unchanged. Sixty-three FP32-source pairs were reused; fourteen BF16-source pairs were rerun with a uniform FP32 intervention convention and original-dtype native controls.

What this establishes:

- Preserving the selected relational Gram matrices and local inner products does not preserve the frozen predictor's candidate discrimination. A common orthogonal channel rotation still changes M substantially; the trained decoder's channel basis must be treated as a confound.
- On the historical event cohort, this rotation increases target logM by about 1.04572 and wrong logM by 2.40941, reducing the target-versus-fixed-content-strongest-wrong gap by 1.36369.
- Restoring the named local inner products does not establish a unique structure/binding split. The additional-group intervals and opponent-dependent effects remain visible; no unique causal percentages are claimed.
- The completed H593 grouped held-out statistical comparison separately supports predictive information beyond the scalar content score: C+M minus C-only NLL is -0.558735, exploratory 95% interval [-0.782504, -0.359249]. It is not a direct conditional-MI estimate or a new deployed-head result.

Read the Chinese interpretation in `ICLR/new ROUTEA/RC/reports/REPORT_M_STRUCTURE_BINDING_ISOLATION_COMPLETE_20261002.md`. Full scalar results, source SHA records, invariant audits, post-hoc contrasts, programs, launchers and migration records are included. Original images, model weights, tensor/feature caches, patch NPZ files and scheduler err/out logs remain excluded.

No new experiment was submitted during this completion audit.
""")
    paths=[x['path'] for x in entries]+[str((BACKUP/n).relative_to(DEST)) for n in ['manifest.json','README.md']]
    Path('/tmp/new_hyp_isolation_complete_export_paths.txt').write_text('\n'.join(paths)+'\n')
    print(json.dumps(dict(files=len(entries),MiB=manifest['total_bytes']/1024**2,manifest=str(BACKUP/'manifest.json'))))


if __name__=='__main__':
    main()
