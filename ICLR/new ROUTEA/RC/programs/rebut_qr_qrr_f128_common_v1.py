#!/usr/bin/env python3
"""F128 data adapter. Original F71 implementations and artifacts stay immutable."""
from pathlib import Path
import torch
from run_rebut_qr_qrr_v1 import FrozenInputs, EvidenceBatch, checked, digest, read

RC = Path(__file__).resolve().parents[1]


def verify_sources(protocol):
    for field in ('sources', 'code_sources', 'dependency_bindings'):
        for value in protocol.get(field, {}).values():
            if isinstance(value, dict) and 'path' in value and 'sha256' in value:
                checked(value)


def verify_panel(protocol):
    panel = protocol['panel_query_ids']
    if len(panel) != 128 or len(set(panel)) != 128 or protocol['seeds'] != [0, 1, 2]:
        raise ValueError('F128 requires exactly 128 unique queries and three registered seeds')
    seen = []
    for f in range(5):
        split = protocol['folds'][str(f)]
        train, held = set(split['train_query_ids']), set(split['heldout_query_ids'])
        if train & held or train | held != set(panel):
            raise ValueError('F128 outer fold is not an exact panel partition')
        seen.extend(split['heldout_query_ids'])
    if len(seen) != 128 or set(seen) != set(panel):
        raise ValueError('F128 held folds must cover the panel exactly once')
    return set(panel)


def verify_evidence_gate(protocol):
    manifest = checked(protocol['evidence_manifest'])
    receipt_path = manifest.parent / 'evidence_validation.json'
    receipt = read(receipt_path)
    binding = {'path': str(manifest.resolve()), 'sha256': digest(manifest)}
    if (receipt.get('status') != 'POOLED_F128_EVIDENCE_INDEPENDENT_PASS' or
            receipt.get('queries') != 128 or receipt.get('candidates') != 128*128 or
            {k: receipt.get('manifest', {}).get(k) for k in ('path', 'sha256')} != binding or receipt.get('labels_read') != 0):
        raise ValueError('F128 independent evidence verification required before training')
    checked(receipt['manifest'])
    checked(receipt['verifier'])
    return {'path': str(receipt_path.resolve()), 'sha256': digest(receipt_path)}


class F128Inputs(FrozenInputs):
    """Use v1 TRAIN-only common-feature logic, then load explicit F128 evidence.

    Pilot may load its one legal TRAIN query from a partial manifest. Formal
    fitting, including B_CAL, requires all 128 evidence records and SHA checks.
    Mapping validity is preserved in cached channels; no confidence filtering.
    """
    def __init__(self, protocol, fold, arm, pilot_query_id=None):
        verify_sources(protocol)
        panel = verify_panel(protocol)
        if pilot_query_id is None:
            verify_evidence_gate(protocol)
        super().__init__(protocol, fold, 'B_CAL', pilot_query_id)
        self.arm = arm
        ordinals = [int(self.common[q]['execution_ordinal']) for q in panel]
        if sorted(ordinals) != list(range(128)):
            raise ValueError('F128 population must be original execution ordinals 0..127')
        manifest_path = checked(protocol['evidence_manifest'])
        manifest = read(manifest_path)
        records = manifest.get('records')
        if not isinstance(records, list):
            raise ValueError('F128 evidence needs records list')
        entries = {r['query_id']: r for r in records}
        if len(entries) != len(records):
            raise ValueError('duplicate F128 evidence query')
        if pilot_query_id is None:
            if manifest.get('status') != 'POOLED_F128_COMPLETE':
                raise ValueError('formal F128 fit requires POOLED_F128_COMPLETE')
            if (len(records) != 128 or set(entries) != panel or manifest.get('expected_queries') != 128 or
                    not isinstance(manifest.get('expected_records'), list) or
                    len(manifest['expected_records']) != 128 or
                    {r['query_id'] for r in manifest['expected_records']} != panel or
                    sorted(int(r['execution_ordinal']) for r in manifest['expected_records']) != list(range(128))):
                raise ValueError('F128 evidence manifest population mismatch')
        elif pilot_query_id not in set(fold['inner_fit_query_ids']):
            raise ValueError('pilot is limited to registered inner TRAIN')
        selected = [pilot_query_id] if pilot_query_id is not None else self.query_ids
        for qid in selected:
            if qid not in entries:
                raise ValueError('F128 evidence missing: ' + qid)
            entry = entries[qid]
            ep = checked(entry['payload'])
            if int(entry['execution_ordinal']) != int(self.common[qid]['execution_ordinal']):
                raise ValueError('F128 evidence ordinal mismatch')
            if arm == 'B_CAL':
                # Baseline never uses F; its complete manifest is still checked.
                continue
            payload = torch.load(ep, map_location='cpu', weights_only=False)
            row = self.common[qid]
            if (payload['query_id'] != qid or payload['axis'] != row['axis'] or
                    int(payload['winner']) != int(row['winner']) or
                    int(payload['execution_ordinal']) != int(row['execution_ordinal'])):
                raise ValueError('F128 evidence/common candidate or query mismatch')
            ev = EvidenceBatch(payload['query'].double()[None], payload['reference'].double()[None],
                               payload['query_valid'].bool()[None], payload['reference_valid'].bool()[None])
            ev.validate()
            if ev.query.shape != ev.reference.shape or ev.query.shape != (1, 128, 64, 72):
                raise ValueError('F128 evidence must retain the declared [128,64,72] interface')
            self.evidence[qid] = ev
            self.bindings[qid] = {'path': str(ep), 'sha256': digest(ep)}
