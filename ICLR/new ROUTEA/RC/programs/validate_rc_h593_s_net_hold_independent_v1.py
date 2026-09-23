#!/usr/bin/env python3
"""Independent exhaustive HOLD-gain validation; no trainer/solver imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_h593_s_net_hold_optimization_v1'
AUTH=ROOT/'registry/rc_h593_s_net_hold_optimization_authority_v1_20260920.json'

def read(p):return json.loads(Path(p).read_text())
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def checked(b):
    assert bind(b['path'])==b,b['path']
    return Path(b['path'])
def frombits(b):return struct.unpack('>d',struct.pack('>Q',b))[0]
def tobits(x):return struct.unpack('>Q',struct.pack('>d',x))[0]
def lifted(m,h,a):return m if a==0. or m>0 else float(m+float(a*h))

def transition(m,h):
    if m>0 or h==0:return None
    if lifted(m,h,sys.float_info.max)<=0:return None
    # Start from the real boundary, then independently verify adjacent doubles.
    a=-m/h
    if math.isfinite(a):
        for _ in range(16):
            if lifted(m,h,a)>0:
                prev=math.nextafter(a,0.)
                if lifted(m,h,prev)<=0:return a
                a=prev
            else:a=math.nextafter(a,math.inf)
    # Rare underflow/overflow fallback searches the positive IEEE-754 lattice.
    lo,hi=0,0x7fefffffffffffff
    while hi-lo>1:
        mid=(lo+hi)//2
        if lifted(m,h,frombits(mid))>0:hi=mid
        else:lo=mid
    return frombits(hi)

def validate(fold):
    # Imports precede the access barrier; only authorized result files are read.
    import numpy as np
    import torch
    torch.set_num_threads(1)
    a=read(AUTH);p_path=OUT/f'fold{fold}/payload.json';p=read(p_path)
    assert p['authority']==bind(AUTH) and p['fold']==fold
    assert a['code_sources']['independent_validator']==bind(__file__)
    for b in a['code_sources'].values():checked(b)
    sources=a['fold_sources'][str(fold)]
    allowed={Path(b['path']).resolve() for b in a['public_sources'].values()}
    allowed.update(Path(b['path']).resolve() for b in sources.values())
    for d in a['features']:allowed.update(Path(b['path']).resolve() for b in d.values())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        if ROOT/'results' in path.parents:
            assert path in allowed or OUT/f'fold{fold}' in path.parents,str(path)
        assert 'curator_roles' not in str(path)
    sys.addaudithook(audit)
    split=read(checked(a['public_sources']['split']))['folds']
    train=set(split[fold]['train_query_ids']);held=set(split[fold]['heldout_query_ids'])
    assert train.isdisjoint(held) and set(p['train_query_ids'])==train
    roles={r['query_id']:r for r in read(checked(sources['train_roles']))['records']}
    assert set(roles)==train
    gallery=read(checked(a['public_sources']['gallery']))
    labels={r['physical_row']:r['identity'] for r in gallery['records']}
    rows=[]
    for d in a['features']:rows.extend(torch.load(checked(d['payload']),map_location='cpu',weights_only=True)['records'])
    byid={r['query_id']:r for r in rows};assert len(byid)==593
    calibration={r['query_id']:r for r in p['calibration']}
    assert len(calibration)==len(p['calibration'])==len(train) and set(calibration)==train
    seen=set();maxerr=0.;inner_checks=0
    for inner in p['inner_heads']:
        ih=set(inner['heldout_query_ids']);it=set(inner['train_query_ids'])
        assert inner['inner_fold']!=fold and ih==set(split[inner['inner_fold']]['heldout_query_ids'])
        assert it==train-ih and it.isdisjoint(held) and not seen&ih
        seen.update(ih)
        for key in ('identity','component'):
            assert {roles[q][key] for q in it}.isdisjoint({roles[q][key] for q in ih})
        assert {byid[q]['source_image_sha256'] for q in it}.isdisjoint({byid[q]['source_image_sha256'] for q in ih|held})
        expected=[r['query_id'] for r in sorted(rows,key=lambda r:r['execution_ordinal']) if r['query_id'] in it and
                  any(labels[x]==roles[r['query_id']]['identity'] for x in r['candidate_physical_rows'])]
        assert inner['effective_train_query_ids']==expected
        theta=np.array([float.fromhex(v) for v in inner['theta_hex']])
        for q in ih:
            r=byid[q];c=calibration[q];x=r['modes']['REAL']['X'].numpy()
            z=np.sum(x*theta[:-1],axis=1)+theta[-1]
            ref=np.array([float.fromhex(v) for v in c['logits_hex']])
            err=float(abs(z-ref).max());assert err<2e-10
            maxerr=max(maxerr,err);inner_checks+=127
            j=int(np.argmax(ref));assert j==c['original_top']==int(np.argmax(z))
            assert float(ref[j]).hex()==c['m'].hex()
            assert max(float(x[j,1]),0.).hex()==c['h_s'].hex()
            raw=r['candidate_physical_rows'][r['winner']]
            top=r['candidate_physical_rows'][r['challenger_positions'][j]]
            delta=int(labels[top]==roles[q]['identity'])-int(labels[raw]==roles[q]['identity'])
            assert delta==c['delta']
    assert seen==train
    fits={}
    for name in ('GAP_NET1','S_NET1','S_SAFE1'):
        records=[(r['m'],1. if name=='GAP_NET1' else r['h_s'],r['delta']) for r in p['calibration']]
        transitions=[transition(m,h) for m,h,_ in records]
        candidates=sorted({0.}|{v for v in transitions if v is not None})
        best=(0,0,0.);bestalpha=0.;bestcounts=(0,0,0)
        for alpha in candidates:
            deltas=[d for m,h,d in records if m<=0 and lifted(m,h,alpha)>0]
            rescue=deltas.count(1);loss=deltas.count(-1)
            if name=='S_SAFE1' and loss:continue
            key=(rescue-loss,-len(deltas),-alpha)
            if key>best:
                best=key;bestalpha=alpha;bestcounts=(rescue,loss,len(deltas))
        fit=p['parameters'][name]
        assert fit['alpha_hex']==bestalpha.hex()
        assert (fit['training_rescues'],fit['training_breaks'],fit['training_changed_holds'])==bestcounts
        assert fit['training_net_gain']==best[0]
        assert len(fit['certificates'])==len(records)
        for record,cross,certificate in zip(records,transitions,fit['certificates']):
            expected=None if cross is None else cross.hex()
            assert certificate['alpha_hex']==expected
            if cross is not None:
                m,h,_=record;assert lifted(m,h,cross)>0 and lifted(m,h,math.nextafter(cross,0.))<=0
        fits[name]=dict(alpha_hex=bestalpha.hex(),net_gain=best[0],rescues=bestcounts[0],breaks=bestcounts[1],
                        changed=bestcounts[2],distinct_candidates=len(candidates))
    assert {r['query_id'] for r in p['predictions']}==held
    old=read(checked(sources['full_payload']));oldpred={r['query_id']:r for r in old['predictions']}
    for pred in p['predictions']:
        q=pred['query_id'];r=byid[q]
        for key in ('candidate_physical_rows','winner','challenger_positions'):assert pred[key]==r[key]
        assert pred['models']['COST1_FULL']==oldpred[q]['models']['COST1']
        z=[float.fromhex(v) for v in pred['models']['COST1_FULL']['logits_hex']]
        j=max(range(127),key=z.__getitem__);assert j==pred['original_top']
        hs=max(float(r['modes']['REAL']['X'][j,1]),0.);assert hs.hex()==pred['h_s'].hex()
        for name in fits:
            out=list(z);alpha=float.fromhex(fits[name]['alpha_hex'])
            out[j]=lifted(z[j],1. if name=='GAP_NET1' else hs,alpha)
            assert [v.hex() for v in out]==pred['models'][name]['logits_hex']
            k=max(range(127),key=out.__getitem__);assert k==j
            pos=r['challenger_positions'][k] if out[k]>0 else r['winner']
            assert r['candidate_physical_rows'][pos]==pred['models'][name]['selected']
    output=dict(status='S_NET_INDEPENDENT_NESTED_EXHAUSTIVE_OPTIMALITY_PASS',passed=True,
                authority=bind(AUTH),payload=bind(p_path),fold=fold,calibration_queries=len(train),heldout_queries=len(held),
                inner_logit_checks=inner_checks,max_abs_inner_error=maxerr,fits=fits,
                original_switches_and_rank_locked=True,heldout_label_reads=0)
    path=OUT/f'fold{fold}/independent_validation.json'
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,required=True,choices=range(5))
    validate(parser.parse_args().fold)
