"""Independent target/selection/refit/metric replay for reviewed-box positions.

Feature heads use cached vectors only. OCR vertical order and its explicitly
coordinate-fed control remain geometric proxies, not reading-row identity.
"""
from collections import Counter
import json
import numpy as np
import torch
from scripts.colnomic_mechanism.common import ROOT, data, read, rows, write, sha
from scripts.colnomic_mechanism.completion_audit import model_at, regression_fit, regression_predict, regression_metrics

BASE=ROOT/'results/colnomic_full_audit/20261013_resume'


def centers(spec):
    l=spec['layout']; k=np.arange(l['count'])
    x=(k%l['width']+.5)/l['width']; y=(k//l['width']+.5)/l['height']
    o=spec['exif_orientation']
    if o==2:x=1-x
    elif o==3:x,y=1-x,1-y
    elif o==4:y=1-y
    elif o==5:x,y=y,x
    elif o==6:x,y=1-y,x
    elif o==7:x,y=1-y,1-x
    elif o==8:x,y=y,1-x
    else:assert o==1
    return np.column_stack([x,y]).astype(np.float32)


def reconstruct(d,kind):
    expected={}; boxes=read(BASE/'manual_box_review/subject_boxes_after_user_review.json')['records']
    for qi,qid in enumerate(d['qb']['ids']):
        spec=d['metadata']['photos'][qi]; assert spec['id']==qid
        l=spec['layout']; k=np.arange(l['count']); tokens=k+l['start']; xy=centers(spec)
        b=boxes[qid]; box=np.array(b['bbox'])/np.tile([b['width'],b['height']],2)
        keep=np.ones(len(k),bool)
        target=np.column_stack([(k%l['width']+.5)/l['width'],(k//l['width']+.5)/l['height']])
        if kind!='absolute':
            keep=np.all((xy>=box[:2])&(xy<box[2:]),axis=1); target=(xy-box[:2])/(box[2:]-box[:2])
        if kind=='ocr_vertical_order':
            record=d['qo'][qid]; dims=np.array([record['width'],record['height']])
            eligible=[line for line in record['lines'] if np.all((np.mean(line['polygon'],axis=0)/dims>=box[:2])&(np.mean(line['polygon'],axis=0)/dims<box[2:]))]
            if len(eligible)<4:continue
            eligible=sorted(eligible,key=lambda line:np.mean(line['polygon'],axis=0)[1])
            bounds=np.array([line['bbox'] for line in eligible])/np.tile(dims,2)
            half=np.array([.5/l['width'],.5/l['height']])
            if spec['exif_orientation']>=5:half=half[::-1]
            overlap=np.maximum(np.minimum(xy[:,None]+half,bounds[None,:,2:])-np.maximum(xy[:,None]-half,bounds[None,:,:2]),0).prod(2)/(4*half.prod())
            assigned=overlap>=.05; keep&=assigned.sum(1)==1
            target=((assigned.argmax(1)+.5)/len(eligible))[:,None]
        tokens=tokens[keep]; target=target[keep]
        if not len(tokens):continue
        chosen=np.sort(np.random.default_rng(20261013+qi).choice(len(tokens),min(128,len(tokens)),replace=False))
        expected[qid]={int(token):label for token,label in zip(tokens[chosen],target[chosen])}
    return expected,boxes


def main():
    torch.set_num_threads(2);d=data();folder=BASE/'positions_reviewed';metrics=rows(folder/'position_metrics.csv')
    protocol=read(ROOT/'results/token_mechanism_experiments/20261013_v2/preregistration.json')
    photo_counts=Counter(p['package_id'] for p in d['photos'].values()); qindex={q:i for i,q in enumerate(d['qb']['ids'])}
    results=[]; checks=[]
    for kind in ['absolute','subject_relative','ocr_vertical_order']:
        expected,boxes=reconstruct(d,kind);index=rows(folder/(kind+'_sample_index.csv'))
        assert len(index)==sum(map(len,expected.values()))
        seen=set();features=[];target_errors=[]
        for row in index:
            qid=row['photo_id']; token=int(row['token_index']);key=(qid,token); assert key not in seen;seen.add(key)
            wanted=expected[qid][token];actual=np.array(json.loads(row['target']))
            target_errors.append(float(np.max(np.abs(actual-wanted))))
            assert np.array_equal(actual,wanted)
            assert row['package_id']==d['photos'][qid]['package_id'] and row['split']==d['splits'][qid]
            features.append(d['qb']['embeddings'][qindex[qid]][token].float().numpy())
        assert seen=={(qid,t) for qid,ts in expected.items() for t in ts}
        x=np.stack(features);y=np.array([json.loads(r['target']) for r in index]);packages=np.array([r['package_id'] for r in index]);splits=np.array([r['split'] for r in index])
        counts=Counter(r['photo_id'] for r in index);weight=np.array([1/(counts[r['photo_id']]*photo_counts[r['package_id']]) for r in index])
        train=splits=='train';val=splits=='val';test=splits=='test';model,stored=model_at(folder/(kind+'_head.pt'));alpha=float(stored['alpha'])
        candidates=[]
        for penalty in protocol['probes']['ridge_grid']:
            candidate=regression_fit(x[train],y[train],penalty,weight[train]);m=regression_metrics(y[val],regression_predict(candidate,x[val]),packages[val])
            candidates.append((m['package_macro_mae'],penalty,candidate))
        selected=min(candidates,key=lambda r:(r[0],-r[1]));assert selected[1]==alpha==read(folder/(kind+'_selection.json'))['alpha']
        assert np.array_equal(selected[2]['mean'],model['mean']) and np.array_equal(selected[2]['scale'],model['scale'])
        coefficient_error=float(np.max(np.abs(selected[2]['coefficient']-model['coefficient'])));assert coefficient_error<1e-8
        pred=regression_predict(model,x);old=np.array([json.loads(r['predicted']) for r in index]);prediction_error=float(np.max(np.abs(pred-old)));assert prediction_error<1e-10
        train_mean=np.average(y[train],axis=0,weights=weight[train]);majority=[int(np.bincount(np.minimum((y[train,j]*4).astype(int),3),minlength=4).argmax()) for j in range(y.shape[1])]
        for split,mask in [('val',val),('test',test)]:
            for control,guess in [('feature_only',pred[mask]),('train_mean',np.broadcast_to(train_mean,y[mask].shape))]:
                m=regression_metrics(y[mask],guess,packages[mask]);prior=next(r for r in metrics if r['probe']==kind and r['control']==control and r['split']==split)
                assert max(abs(float(prior[k])-v) for k,v in m.items())<1e-10
                if control=='train_mean':
                    wanted=[float(np.mean(np.minimum((y[mask,j]*4).astype(int),3)==majority[j])) for j in range(y.shape[1])]
                    assert wanted==json.loads(prior['majority_four_bin_accuracy'])
        for seed in protocol['probes']['shuffle_seeds']:
            labels=y[train].copy();rng=np.random.default_rng(seed)
            for pkg in sorted(set(packages[train])):
                loc=np.flatnonzero(packages[train]==pkg);labels[loc]=labels[rng.permutation(loc)]
            null,record=model_at(folder/f'{kind}_shuffle_{seed}_head.pt');fit=regression_fit(x[train],labels,alpha,weight[train])
            assert np.max(np.abs(fit['coefficient']-null['coefficient']))<1e-8
            prediction=regression_predict(null,x);saved=rows(folder/f'{kind}_shuffle_{seed}_predictions.csv')
            assert [(r['photo_id'],r['token_index'],r['split']) for r in saved]==[(r['photo_id'],r['token_index'],r['split']) for r in index]
            assert np.max(np.abs(prediction-np.array([json.loads(r['predicted']) for r in saved])))<1e-10
            m=regression_metrics(y[test],prediction[test],packages[test]);prior=next(r for r in metrics if r['probe']==kind and r['control']=='label_shuffle' and int(r['seed'])==seed)
            assert max(abs(float(prior[k])-v) for k,v in m.items())<1e-10
        if kind=='ocr_vertical_order':
            spec_by_id={s['id']:s for s in d['metadata']['photos']};vertical=[]
            for row in index:
                qid=row['photo_id'];spec=spec_by_id[qid];b=boxes[qid];box=np.array(b['bbox'])/np.tile([b['width'],b['height']],2)
                vertical.append((centers(spec)[int(row['token_index'])-spec['layout']['start'],1]-box[1])/(box[3]-box[1]))
            vertical=np.array(vertical);choices=[]
            for degree in [1,2,3]:
                z=np.column_stack([vertical**k for k in range(1,degree+1)])
                for penalty in protocol['probes']['ridge_grid']:
                    candidate=regression_fit(z[train],y[train],penalty,weight[train]);m=regression_metrics(y[val],regression_predict(candidate,z[val]),packages[val])
                    choices.append((m['package_macro_mae'],degree,penalty,candidate))
            chosen=min(choices,key=lambda r:(r[0],r[1],-r[2]));baseline,record=model_at(folder/'ocr_vertical_y_only_head.pt')
            assert chosen[1]==record['degree'] and chosen[2]==record['alpha']
            assert np.max(np.abs(chosen[3]['coefficient']-baseline['coefficient']))<1e-8
            z=np.column_stack([vertical**k for k in range(1,chosen[1]+1)])
            for split,mask in [('val',val),('test',test)]:
                prediction=regression_predict(baseline,z[mask]);saved=rows(folder/f'{kind}_y_only_{split}_predictions.csv')
                assert np.max(np.abs(prediction[:,0]-np.array([float(r['predicted']) for r in saved])))<1e-10
                m=regression_metrics(y[mask],prediction,packages[mask]);prior=next(r for r in metrics if r['probe']==kind and r['control']=='y_coordinate_only' and r['split']==split)
                assert max(abs(float(prior[k])-v) for k,v in m.items())<1e-10
        results.append(dict(probe=kind,samples=len(index),target_max_error=max(target_errors),head_refit_coefficient_error=coefficient_error,prediction_max_error=prediction_error,
            identity_counts={s:len(set(packages[splits==s])) for s in ['train','val','test']}))
        checks.append(kind);print('Independent reviewed position verification',kind,len(index),'passed',flush=True)
    assert len(checks)==3
    files=[folder/'verification.json',folder/'position_metrics.csv',BASE/'manual_box_review/subject_boxes_after_user_review.json']
    files+=sorted(folder.glob('*head.pt'))+sorted(folder.glob('*sample_index.csv'))+sorted(folder.glob('*predictions.csv'))
    write(folder/'independent_validation.json',dict(status='passed',position_tasks=results,null_heads_verified=9,coordinate_control_verified=True,
        split_verification=d['split_verification'],source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in files},verifier_SHA256=sha(__file__),
        scope='reviewed box targets, all cached feature inputs, validation selections, main/null heads/predictions and all metrics',
        interpretation='OCR vertical order is not true reading row; y-only control deliberately receives coordinates; absolute position is not identity-layout proof'))


if __name__=='__main__':main()
