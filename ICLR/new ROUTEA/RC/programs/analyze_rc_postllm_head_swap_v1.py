#!/usr/bin/env python3
"""Fixed-score/fixed-head 2x2 diagnostic; no fit or threshold selection."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rc_postllm_m_v1'
PRE = ROOT/'results/rc_internal_m_v4_probe_v1'


def read(p):
    return json.loads(Path(p).read_text())


def checked(b):
    p = Path(b['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == b['sha256']
    return p


def bind(p):
    p = Path(p)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    receipt = read(OUT/'validation.json')
    result = read(checked(receipt['result']))
    a = read(checked(receipt['authority']))
    m = read(checked(a['manifest']))
    postseal = read(OUT/'prelabel_validation.json')
    for b in postseal['decisions'] + postseal['refits']:
        checked(b)
    labels = {r['query_id']: r for r in result['rows'] if r['split']=='probe'}
    preseals = [read(PRE/lane/'validation.json') for lane in ('REAL','CONTROLS')]
    prebindings = {b['path']: b for s in preseals for b in s['predictions']}
    postbindings = {b['path']:b for b in read(OUT/'POST_REAL/probe_validation.json')['predictions']}
    sources = [bind(OUT/'validation.json'), receipt['result'], receipt['authority'], a['manifest'], bind(__file__)]
    rows=[];summaries={}
    for r in m['probe_rows']:
        q=r['query_id'];identity=labels[q]['identity'];raw=np.asarray(r['raw_scores'],dtype=np.float64)
        idx=r['challenger_positions'];w=r['winner_index'];assert idx==[i for i in range(128) if i!=w]
        records={}
        for name,p,bindings in [('PRE',PRE/'predictions/REAL'/(q+'.json'),prebindings),
                                ('POST',OUT/'POST_REAL/probe/native'/(q+'.json'),postbindings)]:
            b=bindings[str(p)];d=read(checked(b));sources.append(b)
            assert d['candidate_ids']==r['candidate_ids'] and d['raw_scores']==r['raw_scores']
            records[name]=d
        predictions={}
        for lname,d in records.items():
            L=np.asarray(d['L'],dtype=np.float64)
            X=np.stack([(raw[idx]-raw[w])/raw.std(),(L[idx]-L[w])/(np.abs(L[idx])+abs(L[w])+1e-12),np.ones(127)],axis=1)
            for hname,h in records.items():
                theta=np.asarray(h['decision']['theta'],dtype=np.float64);z=X@theta
                pos=idx[int(z.argmax())] if z.max()>0 else w
                name=lname+'_L_'+hname+'_HEAD';pred=r['candidate_identities'][pos]
                info=dict(prediction=pred,correct=pred==identity,logits=z.tolist(),theta=theta.tolist(),switched=pos!=w)
                targets=[j for j,c in enumerate(r['candidate_identities']) if c==identity]
                if targets:
                    assert len(targets)==1;t=targets[0]
                    terms=np.zeros(3) if t==w else X[idx.index(t)]*theta
                    info.update(target_score=float(terms.sum()),target_terms=terms.tolist(),
                        target_L_minus_winner=float(L[t]-L[w]))
                if lname==hname:
                    assert np.max(np.abs(z-np.asarray(d['decision']['logits'])))<1e-10
                    assert pred==d['decision']['prediction_identity']
                predictions[name]=info
        rows.append(dict(query_id=q,identity=identity,raw_correct=labels[q]['correct']['RAW'],predictions=predictions))
    for name in rows[0]['predictions']:
        summaries[name]=dict(correct=sum(r['predictions'][name]['correct'] for r in rows),n=8,
            rescues=[r['query_id'] for r in rows if r['predictions'][name]['correct'] and not r['raw_correct']],
            breaks=[r['query_id'] for r in rows if not r['predictions'][name]['correct'] and r['raw_correct']])
    output=dict(status='FROZEN_PRE_POST_SCORE_HEAD_SWAP_REPLAY_PASS',summary=summaries,rows=rows,sources=sources,
        fitting_updates=0,new_model_forwards=0,threshold_changes=0,
        scope='Post-hoc diagnosis on already opened PROBE8; not a new model selection or independent confirmation')
    p=OUT/'head_swap_diagnostic.json';p.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':
    main()
