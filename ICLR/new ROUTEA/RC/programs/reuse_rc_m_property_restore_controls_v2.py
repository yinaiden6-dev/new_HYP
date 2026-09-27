#!/usr/bin/env python3
"""Carry completed v1 control tensors into v2 with explicit byte-level lineage."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'programs'))
import probe_rc_roma_property_restore_cpu_v3 as H

OLD = ROOT / 'results/rc_m_property_restoration_factorial_v1'
NEW = ROOT / 'results/rc_m_property_restoration_factorial_v2'


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.reshape(-1).contiguous().view(torch.uint8), b.reshape(-1).contiguous().view(torch.uint8))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for aa, bb in zip(a, b):
            same(aa, bb)
    else:
        assert a == b


def main():
    torch.set_num_threads(1)
    old_protocol, new_protocol = H.binding(OLD / 'protocol.json'), H.binding(NEW / 'protocol.json')
    op, np = H.read(old_protocol['path']), H.read(new_protocol['path'])
    assert op['arms'] == np['arms'] and op['workers'] == np['workers']
    assert op['sources']['source_head'] == np['sources']['source_head']
    records = []
    for work in np['workers']:
        folder = f"query{work['index']:03d}"
        if not (OLD / folder / 'backend.json').exists():
            continue
        H.write_once(NEW / folder / 'backend.json', H.read(OLD / folder / 'backend.json'))
        backend = H.binding(NEW / folder / 'backend.json')
        for pos in work['positions']:
            for arm in np['arms']:
                if '__' in arm:
                    continue  # Joint arms are the new repaired path, never borrowed.
                relative = Path(folder) / f'pair{pos:03d}' / f'{arm}.pt'
                source, target = OLD / relative, NEW / relative
                if not source.exists():
                    continue
                old = torch.load(source, map_location='cpu', weights_only=True)
                assert old['protocol'] == old_protocol and old['arm'] == arm and old['position'] == pos
                row = {**old, 'protocol': new_protocol, 'backend_binding': backend,
                       'reused_from': H.binding(source),
                       'reuse_reason': 'Unchanged legacy control branch; values preserved, only successor protocol metadata rebound.'}
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    H.save_once(target, row, torch)
                fresh = torch.load(target, map_location='cpu', weights_only=True)
                for key in old:
                    if key not in ('protocol', 'backend_binding'):
                        same(old[key], fresh[key])
                assert fresh['protocol'] == new_protocol and fresh['reused_from'] == H.binding(source)
                records.append({'source': H.binding(source), 'destination': H.binding(target),
                                'index': work['index'], 'position': pos, 'arm': arm,
                                'M': old['M'], 'all_preserved_values_bit_exact': True})
    H.write_once(NEW / 'reused_controls_validation.json', {
        'status': 'V1_CONTROL_CACHE_REUSE_BIT_EXACT_PASS',
        'source_protocol': old_protocol, 'protocol': new_protocol,
        'reused_arms': len(records), 'records': records,
        'joint_arms_reused': 0})
    print(json.dumps({'status': 'V1_CONTROL_CACHE_REUSE_BIT_EXACT_PASS', 'reused_arms': len(records)}))


if __name__ == '__main__':
    main()
