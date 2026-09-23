#!/usr/bin/env python3
"""Independent candidate generation, conditional threshold scan and action audit."""
import argparse
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(array):
    return hashlib.sha256(np.asarray(array,dtype='<f8').tobytes()).hexdigest()


def independent_search(records, beta0):
    rows=[r for r in records if r['m']<=0]
    roots=set()
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            denominator=float(rows[i]['d'])-float(rows[j]['d'])
            if denominator==0:continue
            root=(float(rows[j]['m'])-float(rows[i]['m']))/denominator
            if root>0 and math.isfinite(root):roots.add(root)
    rr=sorted(roots);candidates={0.,float(beta0)};last=0.
    for root in rr:
        candidates.update((root,math.nextafter(root,-math.inf),math.nextafter(root,math.inf),
                           last+(root-last)/2.))
        last=root
    candidates.add(last+max(1.,abs(last)))
    bs=np.array(sorted(x for x in candidates if math.isfinite(x) and x>=0),dtype=np.float64)
    n=len(rows);table=np.zeros((len(bs),6),dtype=np.float64)
    if not n:return bs,table
    m=np.array([r['m'] for r in rows]);d=np.array([r['d'] for r in rows]);y=np.array([r['delta'] for r in rows])
    number=np.arange(1,n+1)[None,:]
    # Sort actual FP64 transition biases rather than score offsets.
    for start in range(0,len(bs),191):
        beta=bs[start:start+191]
        u=np.add(np.add(m,0.)[None,:],np.multiply(beta[:,None],d[None,:]))
        assert np.isfinite(u).all()
        transition=np.nextafter(-u,np.inf)
        order=np.argsort(transition,axis=1,kind='stable')
        transitions=np.take_along_axis(transition,order,axis=1)
        yy=y[order];net=np.cumsum(yy,axis=1)
        ends=np.ones_like(net,dtype=bool)
        ends[:,:-1]=transitions[:,:-1]!=transitions[:,1:]
        # Max gain first, then the shortest prefix with that gain.
        best_net=np.max(np.where(ends,net,-n-1),axis=1)
        possible=ends & (net==best_net[:,None])
        k=np.argmax(possible,axis=1);ix=np.arange(len(beta))
        lower=transitions[ix,k];upper=np.full(len(beta),np.finfo(float).max)
        has=k+1<n
        upper[has]=np.nextafter(transitions[ix[has],k[has]+1],-np.inf)
        bias=np.maximum(lower,np.minimum(0.,upper));bias[bias==0]=0.
        switch=np.add(u,bias[:,None])>0
        rescues=np.sum(switch & (y==1),axis=1)
        breaks=np.sum(switch & (y==-1),axis=1)
        neutral=np.sum(switch & (y==0),axis=1)
        positive=best_net>0
        assert np.array_equal((rescues-breaks)[positive],best_net[positive])
        assert np.array_equal((rescues+breaks+neutral)[positive],(k+1)[positive])
        values=np.column_stack((rescues-breaks,rescues+breaks+neutral,rescues,breaks,neutral,bias)).astype(float)
        values[~positive]=0.
        table[start:start+len(beta)]=values
    return bs,table


def literal_counts(records, head):
    b=float.fromhex(head['bias_hex']);beta=float.fromhex(head['beta_hex'])
    outcomes=[]
    for r in records:
        if r['m']<=0 and not head['disabled']:
            g=float(float(float(r['m'])+0.)+float(beta*r['d']))+b
            if g>0:outcomes.append(r['delta'])
    return dict(net=sum(outcomes),changes=len(outcomes),rescues=outcomes.count(1),
                breaks=outcomes.count(-1),neutral=outcomes.count(0))


def validate(fold):
    sys.path.insert(0,str(ROOT/'programs'))
    import run_rc_h593_gap_net2_v1 as R
    a=R.guard('verify',fold);read,checked,bind=R.read,R.checked,R.bind
    folder=R.OUT/f'fold{fold}';p=read(folder/'payload.json')
    assert p['authority']==bind(R.AUTH) and p['fold']==fold and p['status']=='GAP_NET2_OUTER_SEALED'
    assert p['heldout_label_reads']==0
    source=a['fold_sources'][str(fold)]
    old=read(checked(source['parent_payload']))
    pv=read(checked(source['parent_validation']));iv=read(checked(source['parent_independent_validation']))
    assert pv['payload']==iv['payload']==source['parent_payload']==p['parent_payload']
    assert pv['authority']==iv['authority']==old['authority']==a['parent']
    assert pv['status']=='S_BIAS_COMPETITION_FRESH_REPLAY_PASS'
    assert iv['passed'] and iv['status']=='S_BIAS_COMPETITION_INDEPENDENT_PASS'
    assert p['calibration']==old['calibration'] and p['train_query_ids']==old['train_query_ids']
    split=read(checked(a['public_sources']['split']))['folds']
    train=set(split[fold]['train_query_ids']);held=set(split[fold]['heldout_query_ids'])
    assert train.isdisjoint(held) and train==set(p['train_query_ids'])
    assert {r['query_id'] for r in p['calibration']}==train
    roles={r['query_id']:r for r in read(checked(source['train_roles']))['records']}
    origin=read(checked(source['inner_origin_payload']))
    assert old['parent_payload']==source['inner_origin_payload'] and set(roles)==train
    seen=set()
    for h in origin['inner_heads']:
        ih=set(h['heldout_query_ids']);it=set(h['train_query_ids'])
        assert ih==set(split[h['inner_fold']]['heldout_query_ids']) and it==train-ih
        assert it.isdisjoint(ih) and ih.isdisjoint(seen)
        for key in ('identity','component'):
            assert {roles[q][key] for q in it}.isdisjoint({roles[q][key] for q in ih})
        seen.update(ih)
    assert seen==train
    for r in p['calibration']:
        z=[float.fromhex(v) for v in r['logits_hex']];j=max(range(127),key=z.__getitem__)
        d=z[j]-max(z[k] for k in range(127) if k!=j)
        assert j==r['original_top'] and z[j].hex()==float(r['m']).hex() and d.hex()==float(r['d']).hex()
        assert r['delta']==int(r['top_correct'])-int(r['raw_correct'])
    incumbent=old['parameters']['GAP_BIAS2'];head=p['parameters']['GAP_NET2']
    beta0=float.fromhex(incumbent['beta_hex']);prior=literal_counts(p['calibration'],incumbent)
    bs,table=independent_search(p['calibration'],beta0)
    search=head['search']
    assert search['candidate_count']==len(bs) and search['slope_sha256']==sha(bs) and search['table_sha256']==sha(table)
    assert search['incumbent_counts']==prior
    best=int(np.max(table[:,0],initial=0.));assert search['best_candidate_net']==best
    if best<=prior['net']:
        assert search['incumbent_retained'] and search['selected_index'] is None
        for key in ('alpha_hex','beta_hex','bias_hex','disabled'):assert head[key]==incumbent[key]
    else:
        assert not search['incumbent_retained']
        candidates=np.flatnonzero(table[:,0]==best)
        index=min(map(int,candidates),key=lambda i:(int(table[i,1]),abs(math.log1p(float(bs[i]))-math.log1p(beta0)),
                                                   abs(float(table[i,5])),float(bs[i]),float(table[i,5])))
        assert search['selected_index']==index
        assert head['alpha_hex']==0.0.hex() and head['beta_hex']==float(bs[index]).hex()
        assert head['bias_hex']==float(table[index,5]).hex() and not head['disabled']
    counts=literal_counts(p['calibration'],head)
    assert counts==search['selected_counts'] and counts['net']>=prior['net']
    for k,field in [('net','training_net_gain'),('changes','training_changed_holds'),('rescues','training_rescues'),
                    ('breaks','training_breaks'),('neutral','training_neutral')]:assert head[field]==counts[k]
    originals={r['query_id']:r for r in old['predictions']}
    assert len(p['predictions'])==len(held) and {r['query_id'] for r in p['predictions']}==held
    beta=float.fromhex(head['beta_hex']);bias=float.fromhex(head['bias_hex']);score_checks=0
    for r in p['predictions']:
        parent=originals[r['query_id']]
        for k,value in parent.items():
            if k!='models':assert r[k]==value
        for name,value in parent['models'].items():assert r['models'][name]==value
        z=[float.fromhex(v) for v in parent['models']['COST1_FULL']['logits_hex']]
        j=max(range(127),key=z.__getitem__);out=list(z)
        if z[j]<=0 and not head['disabled']:
            g=float(float(z[j]+0.)+float(beta*r['d']))+bias
            if g>0:out[j]=g
        assert r['models']['GAP_NET2']['logits_hex']==[v.hex() for v in out]
        assert max(range(127),key=out.__getitem__)==j
        pos=r['challenger_positions'][j] if out[j]>0 else r['winner']
        assert r['models']['GAP_NET2']['selected']==r['candidate_physical_rows'][pos]
        score_checks+=127
    R.write(folder/'independent_validation.json',dict(status='GAP_NET2_INDEPENDENT_PASS',passed=True,
            authority=bind(R.AUTH),payload=bind(folder/'payload.json'),fold=fold,
            candidate_count=len(bs),independent_best_net=best,incumbent_counts=prior,selected_counts=counts,
            heldout_queries=len(held),score_checks=score_checks,heldout_label_reads=0,
            full_binary64_plane_optimum_claimed=False))
    print('GAP_NET2_INDEPENDENT_PASS',fold,'slopes',len(bs),'net',prior['net'],'->',counts['net'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=range(5))
    validate(p.parse_args().fold)
