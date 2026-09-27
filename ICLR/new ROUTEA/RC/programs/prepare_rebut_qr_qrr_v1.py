#!/usr/bin/env python3
"""Prepare label-free old evidence and fixed development splits; no new fitting."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from validate_rc_h593_content_complement_v1 import basis, content_check

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_rebut_qr_qrr_v1'


def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(b):
    assert bind(b['path']) == b
    return read(b['path'])
def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Do not overwrite a frozen experiment'
    simple = ROOT/'results/rc_h593_simple_explanations_v1'
    cache = read(simple/'cache.json')
    auth = read(ROOT/'registry/rc_h593_box_ce_polish_authority_v1_20260921.json')
    cv = read(ROOT/'results/rc_h593_m_scalar_headroom_audit_20260927_v1/result.json')
    assert bind(simple/'cache.json') == cv['sources']['cache']
    content = {}
    for b in auth['content_features']:
        v = checked(b['validation']); d = checked(b['payload'])
        assert v['status'] == 'CONTENT_ALL_PAIRS_NUMPY_PASS' and v['payload'] == b['payload']
        for r in d['records']:
            content_check(r)
            assert r['query_id'] not in content
            content[r['query_id']] = r
    assert len(content) == len(cache['rows']) == 593
    records = []; byid = {}
    for r in cache['rows']:
        c = content[r['query_id']]
        assert c['candidate_physical_rows'] == r['axis'] and c['challenger_positions'] == r['challengers']
        assert c['winner'] == r['winner']
        old = np.asarray(r['native_X']); more = np.asarray(c['X'])
        assert np.array_equal(old[:, 0], more[:, 0])
        x = np.concatenate([old, more[:, 1:]], axis=-1)
        ph = basis(x[None], 'JOINT_CONTENT_CE18')[0]
        full = np.zeros((128, 18)); full[r['challengers'], :17] = ph
        full[r['challengers'], 17] = 1.
        row = dict(query_id=r['query_id'], execution_ordinal=r['execution_ordinal'],
                   axis=r['axis'], winner=r['winner'], X0=torch.from_numpy(full),
                   M0=torch.tensor(r['mass'], dtype=torch.float64))
        records.append(row); byid[row['query_id']] = row
    # Replay the strongest historical light-head comparator before using its basis.
    maximum = 0.; oldpredictions = {}
    for fold in range(5):
        folder = ROOT/f'results/rc_h593_box_ce_polish_v1/fold{fold}'
        v = read(folder/'validation.json'); d = checked(v['payload'])
        assert v['status'] == 'BOX_CE_POLISH_EXACT_REPLAY_PASS'
        name = 'CONTENT_BOX_CE_POLISH18'
        theta = np.array([float.fromhex(t) for t in d['parameters'][name]])
        for p in d['predictions']:
            r = byid[p['query_id']]; z = r['X0'].numpy() @ theta
            cs = p['challenger_positions']
            want = np.array([float.fromhex(t) for t in p['models'][name]['logits_hex']])
            err = float(np.max(abs(z[cs] - want))); maximum = max(maximum, err)
            assert err < 2e-10
            best = cs[int(np.argmax(z[cs]))]
            selected = r['axis'][best if z[best] > 0 else r['winner']]
            assert selected == p['models'][name]['selected']
            oldpredictions[r['query_id']] = dict(scores=z.tolist(), selected=selected, fold=fold)
    torch.save(dict(records=records, labels_included=False, feature_order='historical CONTENT18 including bias; physical C128 order; HOLD row zero'), OUT/'common_features.pt')
    write(OUT/'historical496_predictions.json', dict(records=oldpredictions, maximum_replay_error=maximum,
          label_reads=0, new_training_updates=0))
    panel = [r for r in records if r['execution_ordinal'] <= 70]
    assert len(panel) == 71 and {r['execution_ordinal'] for r in panel} == set(range(71))
    panelids = {r['query_id'] for r in panel}
    split = checked(auth['public_sources']['split']); folds = {}
    for f in range(5):
        s = split['folds'][f]; rb = auth['fold_sources'][str(f)]['train_roles']
        roles = {r['query_id']: r for r in checked(rb)['records']}
        train = sorted(set(s['train_query_ids']) & panelids)
        held = sorted(set(s['heldout_query_ids']) & panelids)
        assert set(train) <= set(roles) and not set(held) & set(roles)
        groups = sorted({roles[q]['component'] for q in train},
                        key=lambda c: hashlib.sha256(('REBUT_QR_QRR_V1|'+c).encode()).hexdigest())
        valgroups = set(groups[:max(1, math.ceil(.2*len(groups)))])
        val = [q for q in train if roles[q]['component'] in valgroups]
        fit = [q for q in train if roles[q]['component'] not in valgroups]
        assert fit and val and held
        assert not {roles[q]['identity'] for q in fit} & {roles[q]['identity'] for q in val}
        folds[str(f)] = dict(train_roles=rb, train_query_ids=train, heldout_query_ids=held,
                             inner_fit_query_ids=fit, inner_val_query_ids=val)
    protocol = dict(version=1, status='F71_PROTOCOL_FROZEN_BEFORE_NEW_FIT',
        user_authorization='2026-09-27 keep full593 M/481/492 audit; evaluate and execute uploaded QR/QRR plan; reuse completed analysis',
        output_root=str(OUT), common_features=bind(OUT/'common_features.pt'),
        evidence_manifest=str(OUT/'evidence_manifest.json'), gallery=auth['public_sources']['gallery'],
        folds=folds, arms=['B_CAL','QR','QR_VEC','QRR'], seeds=[0,1,2],
        optimizer=dict(name='AdamW', lr=3e-4, weight_decay=1e-4, max_epochs=100, patience=10, batch_queries=1),
        loss='identity-set FULL-C128 CE when present; mean challenger softplus when absent; query equal weighting',
        checkpoint='inner grouped validation loss chooses epoch count; refit from same seed on all outer TRAIN',
        threshold='inner-validation rescue-minus-break; ties choose more conservative tau; keep zero-threshold companion',
        head='common18 historical CONTENT_BOX_CE_POLISH18 basis; zero initialization; no held-trained checkpoint in inner fit',
        normalization='X0 per-column RMS without centering on inner-fit only; refit RMS on all outer TRAIN; constant bias column unchanged; no held statistics; local descriptors use frozen shared projection only',
        panel_query_ids=sorted(panelids), panel_rule='all pre-existing complete coordinate caches000-070, no correctness/M screening',
        candidate_source='unchanged original natural ColNomic C128', backbone_updates=0, new_backbone_forwards=0,
        evidence_level='opened F71 development pilot; no extrapolation to full593',
        original593_analysis=bind(ROOT/'results/rc_h593_m_481_492_comparison_20260927_v1/validation.json'),
        historical_comparators=['RAW426','NATIVE_COST1_481','GAP_BIAS2_492','CONTENT_BOX_CE_POLISH18_496'],
        historical_comparator_boundary='compare on same71; old heads trained on broader original outer TRAIN, report training-population difference',
        primary_structural_comparison='QRR versus QR_VEC', secondary='QR versus B_CAL',
        final_main_population=593, F593_status='requires522 missing corresponding warp sets; not scheduled by this prototype',
        uniqueness_claim=False, formal_test_status=False,
        sources=dict(simple_cache=bind(simple/'cache.json'), content_features=auth['content_features'],
                     grouped_split=auth['public_sources']['split'], preparation_program=bind(__file__)))
    write(OUT/'protocol.json', protocol)
    print(json.dumps(dict(status=protocol['status'], maximum496_replay_error=maximum,
        common_features=593, prototype_queries=71, folds={f:{k:len(v[k]) for k in ['inner_fit_query_ids','inner_val_query_ids','heldout_query_ids']} for f,v in folds.items()})))


if __name__ == '__main__': main()
