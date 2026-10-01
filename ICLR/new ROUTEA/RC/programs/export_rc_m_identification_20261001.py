#!/usr/bin/env python3
"""Archive this experiment only; omit images, tensor caches, weights and logs."""
import hashlib
import json
from pathlib import Path
import re

RC = Path(__file__).resolve().parents[1]
WS = RC.parents[2]
DEST = WS / 'github_exports/new_HYP_20260919'
BACKUP = DEST / 'backup/m_conditional_prediction_and_isolation_20261001'
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
        'submit_rc_m_identification_20260930.py',
        'submit_rc_m_isolation_precision_repair_20261001.py',
        Path(__file__).name,
    ])
    files.update(RC/'slurm'/n for n in [
        'rc_m_identification_20260930.sbatch', 'rc_m_isolation_precision_repair_20261001.sbatch'])
    files.update(RC/'reports'/n for n in [
        'REPORT_M_CONDITIONAL_PREDICTION_AND_ISOLATION_REPAIR_20261001.md',
        'M_COMPLETED_ARTIFACT_STATUS_20261001.json'])
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
    manifest=dict(scope='H593 held-out conditional prediction, upstream isolation v1 records and precision repair v2 snapshot',
        complete='Held-out: 30 models / five original folds / 570 target-present queries; independent model replay passed.',
        pending='Repair chain 5173346 -> 5173347 -> 5173348 submitted; no claim of acceptance from this snapshot.',
        excluded=['original images','model weights','tensor/feature caches','dense confidence tensors','patch NPZ arrays','err/out logs','locks'],
        files=entries,total_bytes=sum(x['bytes'] for x in entries))
    (BACKUP/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (BACKUP/'README.md').write_text('''# M conditional prediction and upstream isolation

This archive contains the completed grouped held-out predictive comparison, its independent coefficient replay, and the source-precision repair of the upstream isolation experiment. It preserves the failed v1 records rather than silently replacing them.

- H593: original five component folds and natural ColNomic C128. 570 target-present queries; 23 candidate misses reported separately.
- C+M versus C-only: group-equal held NLL difference -0.558735, exploratory group-bootstrap 95% interval [-0.782504, -0.359249]. All five fold differences are negative. Parameter-matched C, estimated M residual, and RAW+C controls are included.
- These are statistical probes on an already opened panel, not a new deployed-model accuracy claim, a direct conditional-mutual-information estimate, or a unique causal explanation.
- Upstream v1 accepted 63 complete pairs. Fourteen additional-group pairs encountered BF16 requantization that violated the intended rotation invariants. v2 reuses verified FP32 results and reruns all 13 arms for these 14 pairs with a common FP32 convention and explicit original-dtype native controls. The invariant gates remain unchanged.
- Repair chain: 5173346 (pilot), 5173347 (14 workers), 5173348 (join). It was submitted and spool-verified; this archive does not certify its final outcome. The obsolete failed-dependency join 5171647 was cancelled.

The Chinese report is `ICLR/new ROUTEA/RC/reports/REPORT_M_CONDITIONAL_PREDICTION_AND_ISOLATION_REPAIR_20261001.md`. Programs, launchers, protocols, scalar parameters, all held-out candidate scores, and source SHA records are included. Raw images, large weights, intermediate tensor/NPZ caches and scheduler logs remain in the workspace and are not uploaded. The manifest records this exact snapshot.
''')
    paths=[x['path'] for x in entries]+[str((BACKUP/n).relative_to(DEST)) for n in ['manifest.json','README.md']]
    Path('/tmp/new_hyp_identification_export_paths.txt').write_text('\n'.join(paths)+'\n')
    print(json.dumps(dict(files=len(entries),MiB=manifest['total_bytes']/1024**2,manifest=str(BACKUP/'manifest.json'))))


if __name__=='__main__':
    main()
