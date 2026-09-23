#!/usr/bin/env python3
"""Frozen content-only head readout and loss diagnostics; no retraining."""
import json
from pathlib import Path
import numpy as np
from audit_rc_h593_ablation_independent_v1 import checked, read, target, choose, objective, bindings

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/rc_h593_learned_colnomic_only_v1'
OUT = ROOT/'results/rc_h593_ablation_independent_audit_v1'
COLS = {'COST1_CONTENT7':list(range(6)), 'CE_CONTENT7':list(range(6)),
        'COST1_MAXSIM3':[0,1], 'CE_MAXSIM3':[0,1], 'RAW2_CE':[0]}

def main():
    validation = read(SOURCE/'validation.json')
    result = read(checked(validation['result']))
    authority = read(checked(validation['authority']))
    gallery = read(checked(authority['gallery']))
    labels = {r['physical_row']:r['identity'] for r in gallery['records']}
    roles = {r['query_id']:r for r in read(ROOT/'results/rc_new_hyp593_oof5_v1/metadata/curator_roles.json')['records']}
    rows = []
    for shard in range(75):
        v = read(SOURCE/f'features/shard{shard:02d}/validation.json')
        assert v['status']=='CONTENT_ALL_PAIRS_NUMPY_PASS'
        rows.extend(read(checked(v['payload']))['records'])
    data = {r['query_id']:r for r in rows}
    assert len(data)==len(rows)==593
    all_losses = []
    checks = 0
    max_error = 0.
    for fold in range(5):
        v = read(SOURCE/f'fold{fold}/validation.json')
        p = read(checked(v['payload']))
        for pred in p['predictions']:
            r=data[pred['query_id']];x=np.array(r['X'])
            for model,keep in COLS.items():
                theta = np.array([float.fromhex(t) for t in p['parameters'][model]])
                z=np.sum(x[:,keep]*theta[:-1],axis=1)+theta[-1]
                ref=np.array([float.fromhex(t) for t in pred['models'][model]['logits_hex']])
                max_error=max(max_error,float(np.max(np.abs(z-ref))))
                assert np.max(np.abs(z-ref))<2e-10
                assert choose(z,r)==pred['models'][model]['selected']
                checks+=127
        for stage,ids in [('TRAIN',p['train_query_ids']),('OOF',[r['query_id'] for r in p['predictions']])]:
            use=[q for q in ids if target(data[q],roles[q],labels)>=-1]
            y=np.array([target(data[q],roles[q],labels) for q in use])
            x=np.array([data[q]['X'] for q in use])
            for model,keep in COLS.items():
                t=np.array([float.fromhex(v) for v in p['parameters'][model]])
                z=np.sum(x[:,:,keep]*t[:-1],axis=2)+t[-1]
                a=np.argmax(np.c_[np.zeros(len(z)),z],axis=1)
                all_losses.append(dict(fold=fold,stage=stage,model=model,n=len(y),raw_correct=int((y==-1).sum()),
                                       correct=int((a==y+1).sum()),switches=int((a!=0).sum()),
                                       loss=float(objective(z,y,'COST1' if model.startswith('COST1') else 'CE').mean())))
    aggregate=[]
    for stage in ['TRAIN','OOF']:
        for model in COLS:
            rs=[r for r in all_losses if r['stage']==stage and r['model']==model]
            n=sum(r['n'] for r in rs)
            row=dict(stage=stage,model=model,n=n,raw_correct=sum(r['raw_correct'] for r in rs),
                     correct=sum(r['correct'] for r in rs),loss=sum(r['loss']*r['n'] for r in rs)/n)
            if stage=='OOF':assert row['correct']==result['summary'][model]['correct']
            aggregate.append(row)
    output=dict(status='CONTENT_CACHED_FEATURE_FROZEN_LOGITS_AND_LOSS_PASS',logit_checks=checks,
                max_abs_error=max_error,aggregate=aggregate,folds=all_losses,source_bindings=bindings,
                limits=['TRAIN aggregate repeats each eligible query in four folds',
                        'Loss excludes 23 recall-absent queries following the original training contract',
                        'No retraining, global convergence claim, or parameter selection',
                        'Cached statistics readout only; original token matrices were not regenerated'])
    OUT.mkdir(exist_ok=True)
    (OUT/'content_surrogate_diagnostics.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['folds','source_bindings']},indent=2))

if __name__=='__main__':
    main()
