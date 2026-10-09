#!/usr/bin/env python3
"""Cached controls, original certified property readout, and final scope ledger."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys
import numpy as np
import run_rc_m_upstream_all593_v1 as U
RC,OUT=U.RC,U.OUT
read,write,bind,checked=U.read,U.write,U.bind,U.checked


def cached():
    U.guard();v=read(U.REPLICATION/'validation.json');rep=read(checked(v['result']))
    assert rep['queries']==593
    labels=[read(checked(b))['row'] for b in rep['query_receipts']]
    rows=[];coordinate=[]
    protocol=read(U.REPLICATION/'protocol.json');cache=read(checked(protocol['cache']))
    for i,lab in enumerate(labels):
        im=read(RC/f'results/rc_h593_m_inside_v1/query{i:03d}/inside_manifest.json')
        vi=read(RC/f'results/rc_h593_m_visual_origin_v1/query{i:03d}/payload.json')
        cvpath=RC/f'results/rc_h593_roma_coordinate_precision_v2/query{i:03d}/validation.json'
        cv=read(cvpath);assert cv['status']=='ROMA_COORDINATE_QUERY_PASS'
        co=read(checked(cv['payload']));assert co['query_id']==lab['query_id'] and co['candidate_physical_rows']==lab['axis']
        assert im['query_id']==vi['query_id']==lab['query_id']
        target,wrong=lab['target_position'],lab['fixed_wrong_position'];values={}
        if target is not None:
            for arm,stages in im['records'][target]['branches'].items():
                for stage,mt in stages.items():
                    mw=im['records'][wrong]['branches'][arm][stage]
                    values['inside/'+arm+'/'+stage]=math.log(mt)-math.log(mw)
            for arm,stages in vi['masses'].items():
                for stage,m in stages.items():values['visual/'+arm+'/'+stage]=math.log(m[target])-math.log(m[wrong])
            for arm,data in co['modes'].items():
                mm=[x['visibility_mass'] for x in data['scores']]
                values['coordinate/'+arm+'/FINAL']=math.log(mm[target])-math.log(mm[wrong])
        theta=np.array([float.fromhex(x) for x in cache['native_parameters'][lab['fold']]['COST1']])
        coord_scores={}
        for arm,data in co['modes'].items():
            x=np.array(data['X']);z=x@theta[:-1]+theta[-1]
            independent=np.array([math.fsum(float(a)*float(b) for a,b in zip(xx,theta[:-1]))+theta[-1] for xx in x])
            assert np.max(abs(z-independent))<2e-10
            pick=co['challenger_positions'][int(z.argmax())] if z.max()>0 else co['winner']
            coord_scores[arm]=dict(logits127=z.tolist(),prediction_position=pick,correct=pick==target)
        coordinate.append(dict(index=i,query_id=lab['query_id'],component=lab['component'],validation=bind(cvpath),scores=coord_scores))
        rows.append(dict(index=i,component=lab['component'],target_present=target is not None,gaps=values,
                    sources=[bind(RC/f'results/rc_h593_m_inside_v1/query{i:03d}/inside_manifest.json'),
                             bind(RC/f'results/rc_h593_m_visual_origin_v1/query{i:03d}/payload.json')]))
    stats={}
    for name,rs in [('FRONT128',rows[:128]),('TAIL465',rows[128:]),('ALL593',rows)]:
        rr=[r for r in rs if r['target_present']];summary={}
        for key in rr[0]['gaps']:
            family,arm,stage=key.split('/');baseline=f'{family}/'+('A1J1P1' if family=='inside' else 'NATIVE')+'/'+stage
            summary[key]=U.bootstrap([(r['component'],r['gaps'][key]-r['gaps'][baseline]) for r in rr])
        stats[name]=summary
    root=OUT/'cached_controls';write(root/'result.json',dict(status='ALL593_ALL_CACHED_STAGES_ANALYZED',queries=593,rows=rows,panels=stats,
        source_replication=v['result'],coordinate_rows=coordinate,coordinate_frozen_head='Original fold NATIVE7 COST1',
        new_forwards=False,new_POST=False,unique_visual_semantics=False))
    write(root/'validation.json',dict(status='ALL593_CACHED_STAGES_PASS',queries=593,result=bind(root/'result.json')))
    return 0


def propfit(fold):
    U.guard();p,pb=U.protocol();v=read(OUT/'validation.json');r=read(checked(v['result']))
    import run_rc_m_causal128_property_compensation_v1 as C
    oldp=read(C.OUT/'protocol.json');rp=read(checked(p['replication']));auth=read(checked(rp['simple_authority']))
    split=read(checked(auth['public']['split']))['folds'][fold]
    trids=set(split['train_query_ids']);heids=set(split['heldout_query_ids']);assert not trids&heids
    rows=[];metadata=[]
    for b in r['query_receipts']:
        q=read(checked(b));source=read(checked(p['query_inputs'][q['index']]))
        if q['target'] is None:continue
        target=q['target'];wrong=q['fixed_opponents']['C_STRONGEST'];row=source['original_row'];raw=np.zeros(128)
        raw[row['challengers']]=np.asarray(row['native_X'])[:,0];free=np.asarray(row['free_content'])
        rec=dict(raw_scores_pair=[raw[target],raw[wrong]],raw_C128_population_std=1.,
                 free_content_pair=[free[target],free[wrong]])
        values={}
        for cell,arms in oldp['cells'].items():
            values[cell]=[]
            for arm in arms:
                m=q['M']['phase/'+arm];mm=[m[target],m[wrong]]
                x=C.feature(rec,mm);assert np.max(abs(x-C.scalar_feature(rec,mm)))<1e-13
                values[cell].append(x)
        rows.append(values);metadata.append(dict(index=q['index'],query_id=q['query_id'],component=q['component']))
    train=[i for i,d in enumerate(metadata) if d['query_id'] in trids];held=[i for i,d in enumerate(metadata) if d['query_id'] in heids]
    assert train and held and len(train)+len(held)==570
    assert not {metadata[i]['component'] for i in train}&{metadata[i]['component'] for i in held}
    arrays={cell:np.array([row[cell] for row in rows]) for cell in oldp['cells']}
    scales=np.sqrt(np.mean(arrays[oldp['reference_cell']][train]**2,axis=(0,1)));scales=np.where(scales==0,1.,scales)
    models={}
    for name in oldp['fit_models']:
        cell=oldp['reference_cell'] if name=='RAW_L_ONLY' else name
        idx=oldp['baseline_features'] if name=='RAW_L_ONLY' else list(range(4))
        x=(arrays[cell]/scales)[:,:,idx];sol=C.solve(x[train].reshape(-1,len(idx)),oldp)
        theta=np.array(sol['theta']);heldz=x[held]@theta
        check=np.array([[math.fsum(float(a)*float(b) for a,b in zip(xx,theta)) for xx in case] for case in x[held]])
        assert np.max(abs(check-heldz))<2e-10
        models[name]=dict(**sol,scales=scales.tolist(),feature_indices=idx,
                          held_margins=heldz.tolist(),train_loss=float(np.logaddexp(0.,-(x[train]@theta)).mean()),
                          held_nll=np.logaddexp(0.,-heldz).mean(1).tolist())
    root=OUT/'property_compensation';write(root/f'fold{fold}.json',dict(status='PROPERTY593_FOLD_CERTIFIED',protocol=pb,
          original_solver_protocol=bind(C.OUT/'protocol.json'),fold=fold,train=[metadata[i] for i in train],held=[metadata[i] for i in held],
          train_scale_only=True,models=models,readout_family='Original 4D FP64 ridge pair ranker; 2D RAW_L control',
          interpretation='Original opened H593 folds; no held tuning, no impossibility claim.'))
    return 0


def propjoin():
    U.guard();p,pb=U.protocol();root=OUT/'property_compensation';seen=set();values=defaultdict(list);fits=[]
    for fold in range(5):
        d=read(root/f'fold{fold}.json');assert d['protocol']==pb and d['status']=='PROPERTY593_FOLD_CERTIFIED'
        ids={r['index'] for r in d['held']};assert not seen&ids;seen|=ids
        for model,v in d['models'].items():
            assert v['certificate']['gradient_inf']<=1.01e-10
            for q,nll in zip(d['held'],v['held_nll']):values[model].append((q['component'],nll))
        fits.append(bind(root/f'fold{fold}.json'))
    assert len(seen)==570
    summary={k:U.bootstrap(v) for k,v in values.items()};diff={}
    base=values['AMP_NATIVE_GLOBAL']
    for model,v in values.items():
        assert [g for g,_ in base]==[g for g,_ in v]
        diff[model]=U.bootstrap([(g,x-y) for (g,x),(_,y) in zip(v,base)])
    write(root/'result.json',dict(status='PROPERTY593_ALL5_CERTIFIED',target_present=570,folds=fits,models=25,
          held_nll=summary,difference_from_native_global=diff,unique_necessity=False,opened_groups=True))
    write(root/'validation.json',dict(status='PROPERTY593_ALL5_PASS',result=bind(root/'result.json')));return 0


def final():
    U.guard();required=[OUT/'validation.json',OUT/'property_compensation/validation.json',
          OUT/'path_compensation/validation.json',OUT/'pair_structure/validation.json',OUT/'cached_controls/validation.json']
    receipts=[]
    for path in required:
        v=read(path);assert 'PASS' in v['status'];checked(v['result']);receipts.append(bind(path))
    write(OUT/'final_validation.json',dict(status='ALL593_UPSTREAM_ALL_FAMILIES_CLOSED',queries=593,validations=receipts,
          engineering_complete=True,unique_cause_claim=False,new_POST=False))
    (OUT/'final_report.md').write_text('# 全部上游归因 H593 扩展验收\n\n'
        '所有规定分支产物完整、独立核算通过。科学结论应读取各分支的效应、区间、拟合证书与原生差异；验收通过不表示所有效应成立或唯一因果已识别。\n\n'
        '- 新细性质：result.json / report.md，60条件、完整自然C128。\n'
        '- 入口删除后补偿：path_compensation/report.md，原PRODUCT5、COST1、原五折。\n'
        '- 性质补偿：property_compensation/result.json，原四维读出与解析优化证书。\n'
        '- 单图可分性：pair_structure/report.md。\n'
        '- 全阶段已有干预：cached_controls/result.json。\n'
        '- 原25张、原128及历史输出保持封存；不存在新增POST或外部未见数据结论。\n')
    return 0


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['cached','property-fit','property-join','final']);ap.add_argument('--fold',type=int,default=0);a=ap.parse_args()
    sys.exit({'cached':cached,'property-fit':lambda:propfit(a.fold),'property-join':propjoin,'final':final}[a.mode]())
