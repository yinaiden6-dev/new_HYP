#!/usr/bin/env python3
"""Saved 16x16 RoMa warp diagnostics. No fit, encoder or matcher execution.

These are diagnostics of predicted coordinates, not ground-truth geometry.
The panel and formulas are frozen before joining target labels. A low affine
residual alone is degenerate for collapsed maps; endpoint occupancy is retained.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_pair_relation_localization_v1/geometry'
VIS = ROOT / 'results/rc_h593_m_visual_origin_v1'
PANEL = ROOT / 'results/rc_h593_cached_visual_trend_v1/result.json'
LABELS = ROOT / 'results/rc_h593_visual55_mechanism_v1/result.json'
STAGES = ('COARSE', 'HR1')


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def bind(p):
    p = Path(p).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for a in iter(lambda: f.read(4 << 20), b''):
            h.update(a)
    return dict(path=str(p), sha256=h.hexdigest())


def checked(b):
    assert bind(b['path']) == b, b['path']
    return Path(b['path'])


def geometry(ab, ba, u, v):
    n, h, w, _ = ab.shape
    y, x = torch.meshgrid(torch.linspace(-1+1/h, 1-1/h, h, dtype=ab.dtype),
                          torch.linspace(-1+1/w, 1-1/w, w, dtype=ab.dtype), indexing='ij')
    xy = torch.stack((x, y), -1)
    design = torch.cat((xy.reshape(-1, 2), torch.ones(h*w, 1, dtype=ab.dtype)), 1)
    pinv = torch.linalg.pinv(design)
    values = []
    for warp, reverse, conf in ((ab, ba, u), (ba, ab, v)):
        cycle = F.grid_sample(reverse.permute(0, 3, 1, 2), warp, mode='bilinear',
                              padding_mode='border', align_corners=False).permute(0, 2, 3, 1)
        cycle_error = (cycle - xy).norm(dim=-1) / np.sqrt(8.)
        # Only endpoints within the actual 16x16 sample-centre support avoid
        # extrapolating reduced-resolution reverse coordinates.
        valid = (warp[..., 0].abs() <= 1-1/w) & (warp[..., 1].abs() <= 1-1/h)
        vf = valid.double()
        denom = vf.sum((1, 2)).clamp_min(1.)
        weights = vf * conf
        weighted = weights.sum((1, 2)).clamp_min(1e-30)
        targets = warp.reshape(n, -1, 2)
        coef = pinv @ targets
        residual = (targets - design @ coef).square().sum(-1) / 8.
        wd = conf.reshape(n, -1)
        # Weighted least squares: each candidate retains its own weight map.
        sw = wd.sqrt()
        dx = design[None] * sw[..., None]
        wy = targets * sw[..., None]
        wcoef = torch.linalg.lstsq(dx, wy, driver='gelsd').solution
        wres = (targets - design @ wcoef).square().sum(-1) / 8.
        endpoint_bins = ((warp + 1.) * torch.tensor([w/2, h/2], dtype=warp.dtype)).floor().long()
        endpoint_bins[..., 0].clamp_(0, w-1)
        endpoint_bins[..., 1].clamp_(0, h-1)
        occupancy = []
        for bins in endpoint_bins:
            occupancy.append(len(torch.unique(bins[..., 1]*w+bins[..., 0])) / (h*w))
        values.append(dict(cycle_negative=-(cycle_error*vf).sum((1,2))/denom,
            cycle_conf_negative=-(cycle_error*weights).sum((1,2))/weighted,
            valid_cycle_fraction=vf.mean((1,2)),
            affine_negative=-residual.mean(1).sqrt(),
            affine_conf_negative=-((wres*wd).sum(1)/wd.sum(1).clamp_min(1e-30)).sqrt(),
            endpoint_occupancy=torch.tensor(occupancy, dtype=ab.dtype)))
    return {k:(values[0][k]+values[1][k])/2 for k in values[0]}, values


def numpy_cycle(reverse, warp):
    h, w = reverse.shape[:2]
    at = (warp + 1.) * np.array([w/2, h/2]) - .5
    at[..., 0] = np.clip(at[..., 0], 0, w-1)
    at[..., 1] = np.clip(at[..., 1], 0, h-1)
    a = np.floor(at).astype(int)
    b = np.minimum(a + 1, [w-1,h-1])
    t = at-a
    return (reverse[a[...,1],a[...,0]]*(1-t[...,0,None])*(1-t[...,1,None])+
            reverse[a[...,1],b[...,0]]*t[...,0,None]*(1-t[...,1,None])+
            reverse[b[...,1],a[...,0]]*(1-t[...,0,None])*t[...,1,None]+
            reverse[b[...,1],b[...,0]]*t[...,0,None]*t[...,1,None])


def verify_math():
    y, x = torch.meshgrid(torch.linspace(-15/16,15/16,16,dtype=torch.float64),
                         torch.linspace(-15/16,15/16,16,dtype=torch.float64), indexing='ij')
    identity = torch.stack((x,y),-1)[None]
    ones = torch.ones(1,16,16,dtype=torch.float64)
    got, _ = geometry(identity, identity, ones, ones)
    assert abs(float(got['cycle_negative'][0])) < 1e-14
    assert abs(float(got['affine_negative'][0])) < 1e-14
    assert float(got['endpoint_occupancy'][0]) == 1.
    collapsed, _ = geometry(identity*0, identity*0, ones, ones)
    assert abs(float(collapsed['affine_negative'][0])) < 1e-14
    assert float(collapsed['cycle_negative'][0]) < -.1
    assert float(collapsed['endpoint_occupancy'][0]) == 1/256
    return dict(identity_pass=True, collapsed_map_degeneracy_accounted=True)


def extract(index, protocol):
    dest = OUT / 'queries' / f'query{index:03d}.json'
    if dest.exists():
        assert read(dest)['protocol'] == protocol
        return
    vp = VIS/f'query{index:03d}/validation.json'
    val = read(vp)
    assert val['status'] == 'M_VISUAL_ORIGIN_QUERY_PASS'
    payload = read(checked(val['payload']))
    arrays = {s:{k:[] for k in ['ab','ba','u','v','M']} for s in STAGES}
    pos, physical, free, sources = [], [], [], [bind(vp),val['payload']]
    for src in payload['parts']:
        part = torch.load(checked(src), map_location='cpu', weights_only=True, mmap=True)
        sources.append(src)
        for pair in part['pairs']:
            pos.append(pair['candidate_position']); physical.append(pair['physical_row'])
            free.append(pair['free_content'])
            for stage in STAGES:
                data = pair['arms']['NATIVE']['stages'][stage]
                for key,side,field in [('ab','AB','warp_grid16'),('ba','BA','warp_grid16'),
                                       ('u','AB','overlap_grid16'),('v','BA','overlap_grid16')]:
                    arrays[stage][key].append(data[side][field].double().clone())
                arrays[stage]['M'].append(data['M'])
    assert pos == list(range(128))
    assert physical == payload['candidate_physical_rows']
    metrics, checks = {}, {}
    for stage, record in arrays.items():
        ab, ba, u, v = [torch.stack(record[k]) for k in ['ab','ba','u','v']]
        g, _ = geometry(ab, ba, u, v)
        for k,t in g.items():
            assert bool(torch.isfinite(t).all())
            metrics[f'{stage}/{k}'] = t.tolist()
        metrics[f'{stage}/M'] = record['M']
        tc = F.grid_sample(ba[:1].permute(0,3,1,2),ab[:1],padding_mode='border',align_corners=False)
        nc = numpy_cycle(ba[0].numpy(), ab[0].numpy())
        err = float(np.max(np.abs(nc-tc[0].permute(1,2,0).numpy())))
        assert err < 1e-12
        checks[stage] = dict(independent_numpy_cycle_error=err)
    metrics['FREE_CONTENT'] = free
    write(dest, dict(protocol=protocol,index=index,query_id=payload['query_id'],
        physical_rows=physical,raw_winner_position=payload['raw_winner'],
        metrics=metrics,sources=sources,checks=checks,labels_used=False))
    print(json.dumps(dict(stage='extracted',index=index,pairs=128)), flush=True)


def corr(a,b):
    a,b=rankdata(a),rankdata(b)
    if np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def joined(protocol, panel):
    # Join previously opened identities only after all formula-fixed scores exist.
    target_rows = {r['index']:r for r in read(LABELS)['rows']}
    meta = {r['index']:r for r in panel['rows']}
    rows = []
    keys = None
    for index in protocol['indices']:
        p = OUT/'queries'/f'query{index:03d}.json'
        data = read(p)
        assert data['protocol'] == protocol
        r = dict(index=index,query_id=data['query_id'],component=meta[index]['component'],
            original_query_id=meta[index]['original_query_id'],raw_correct=meta[index]['raw_correct'],
            native_correct=meta[index]['native_correct'],target_present=meta[index]['target_present'],
            metrics={},source=bind(p))
        assert data['query_id'] == meta[index]['query_id']
        keys = list(data['metrics'])
        assert len(data['metrics']['HR1/M']) == 128
        if r['target_present']:
            t = data['physical_rows'].index(target_rows[index]['target_physical_row'])
            w = data['raw_winner_position']
            assert 0<=w<128
            for key in keys:
                vals = np.asarray(data['metrics'][key]); wrong=np.delete(vals,t)
                r['metrics'][key]=dict(target_rank_min=int(1+sum(wrong>vals[t])),
                    target_strict_first=bool(vals[t]>wrong.max()),
                    target_pairwise_win_fraction=float((sum(vals[t]>wrong)+.5*sum(vals[t]==wrong))/127),
                    target_minus_raw=float(vals[t]-vals[w]),
                    target_beats_raw=bool(vals[t]>vals[w]),
                    rho_with_final_M=corr(vals,data['metrics']['HR1/M']),
                    target_value=float(vals[t]),strongest_wrong_value=float(wrong.max()))
        rows.append(r)
    summary={}
    selected=[r for r in rows if r['target_present']]
    groups=sorted({r['component'] for r in selected})
    rng=np.random.default_rng(20260924)
    draws=rng.integers(0,len(groups),size=(2000,len(groups)))
    for key in keys:
        # valid coverage is reported but not interpreted as a geometry ranker.
        z=[r['metrics'][key] for r in selected]
        gmean=np.asarray([np.mean([r['metrics'][key]['target_pairwise_win_fraction'] for r in selected if r['component']==g]) for g in groups])
        base=np.asarray([np.mean([r['metrics']['HR1/M']['target_pairwise_win_fraction'] for r in selected if r['component']==g]) for g in groups])
        delta=gmean-base
        raw_wrong=[r for r in selected if not r['raw_correct']]
        rescues=[r for r in raw_wrong if r['native_correct']]
        rho=[r['rho_with_final_M'] for r in z if r['rho_with_final_M'] is not None]
        summary[key]=dict(target_strict_first=sum(r['target_strict_first'] for r in z),denominator=len(z),
            group_equal_pairwise_win_fraction=float(gmean.mean()),
            descriptive_group_bootstrap95_delta_vs_M=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
            median_query_rho_with_final_M=float(np.median(rho)) if rho else None,
            raw_wrong_target_beats_raw=sum(r['metrics'][key]['target_beats_raw'] for r in raw_wrong),raw_wrong_n=len(raw_wrong),
            original_rescues_target_beats_raw=sum(r['metrics'][key]['target_beats_raw'] for r in rescues),original_rescues_n=len(rescues))
    write(OUT/'result.json',dict(status='SAVED_WARP_DIAGNOSTICS_COMPLETE',protocol=protocol,
        sources=[bind(PANEL),bind(LABELS)],queries=len(rows),target_present=len(selected),groups=len(groups),
        summary=summary,rows=rows,training=False,new_gpu_forwards=0,
        limitations=['Predicted and downsampled 16x16 coordinates, not ground-truth correspondences.',
          'Native association diagnostics cannot establish origin or causal necessity of information.',
          'Confidence-weighted residuals reuse the same confidence source as M, not an independent representation.',
          'No head training or accuracy-selected diagnostic. These ranks do not replace original COST1 results.',
          'Complete previously opened 55 panel retained; three absent targets excluded only from within-C128 ranking.',
          'Bootstrap intervals are descriptive; no multiple-comparison claim or independent confirmation.']))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    # Extract fixed existing-panel membership, without selecting outcomes.
    panel=read(PANEL)
    indices=sorted(r['index'] for r in panel['rows'])
    assert len(indices)==55 and len(set(indices))==55
    protocol=dict(program=bind(__file__),panel=bind(PANEL),indices=indices,
        stages=list(STAGES),arms=['NATIVE'],candidates=128,grid=16,
        cycle='Bidirectional bilinear reverse-warp composition on centre-supported endpoints; divide by image diagonal.',
        affine='Full 16x16 unweighted and overlap-weighted least-squares affine residual divided by image diagonal.',
        occupancy='Fraction of 16x16 endpoint bins reached, both directions equally weighted.',
        score_sign='Negative residual is higher better; occupancy higher better.',
        selection='All existing55 queries, no choice using outcomes or new metrics.',
        role='Opened-panel diagnostics only; no classifier or new accuracy claim')
    pp=OUT/'protocol.json'
    if pp.exists():assert read(pp)==protocol
    else:write(pp,protocol)
    write(OUT/'preflight.json',verify_math())
    for i in indices:extract(i,protocol)
    joined(protocol,panel)


if __name__=='__main__':main()
