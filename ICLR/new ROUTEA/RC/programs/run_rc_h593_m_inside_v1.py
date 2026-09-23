#!/usr/bin/env python3
"""Native-image upstream predictor-path experiment, using the validated runner."""
import argparse
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_h593_m_visual_origin_v1 as U
import rc_roma_visual_origin_v1 as N
import rc_roma_m_inside_v1 as I

RUNNER=Path(U.__file__)
U.__file__=__file__
U.OUT=ROOT/'results/rc_h593_m_inside_v1'
U.AUTH=ROOT/'registry/rc_h593_m_inside_authority_v1_20260922.json'
U.PLAN=ROOT/'plan/RC_H593_M_INSIDE_V1_20260922.md'
U.GPU=ROOT/'slurm/rc_h593_m_inside_v1.sbatch'
U.CPU=ROOT/'slurm/rc_h593_m_inside_control_v1.sbatch'


class Sink:
    def __init__(self,index):
        self.index=index;self.folder=U.OUT/f'query{index:03d}'/'inside'
        self.capture_full=index==0 and not any(self.folder.glob('*.pt'))
    def __call__(self,identity,inside):
        self.folder.mkdir(parents=True,exist_ok=True)
        path=self.folder/(identity[1]+'.pt')
        if not path.exists():
            U.save(path,dict(authority=U.bind(U.AUTH),index=self.index,query_sha=identity[0],reference_sha=identity[1],
                branches=list(I.BRANCHES),sides=inside,geometry='Native warp and native refiner evidence held fixed',
                full_first_pair_components=self.capture_full))
        self.capture_full=False


def configure(index):
    U.V=SimpleNamespace(__file__=I.__file__,ARMS=('NATIVE',),STAGES=N.STAGES,
        self_test=I.self_test,
        Observer=lambda model,core,input_sink:I.Observer(model,core,input_sink,Sink(index)))


def prepare():
    assert not U.AUTH.exists();test=I.self_test();old=U.read(U.C.AUTH)
    ea=U.read(ROOT/'registry/rc_h593_quality_operator_eval_authority_v1_20260922.json')
    sources=[Path(__file__),Path(I.__file__),RUNNER,Path(N.__file__),Path(U.C.__file__),Path(U.C.M.__file__),
        ROOT/'programs/rc_roma_exact_feature_cache_v2.py',U.PLAN,U.GPU,U.CPU]
    U.write(U.AUTH,dict(status='M_VISUAL_ORIGIN_AUTHORIZED',experiment='M_INSIDE_PREDICTOR_PATHS',
        user_authorization='User asks which upstream components make M useful; continue authorized mechanism localization',
        sources=[U.bind(p) for p in sources],profile=old['profile'],workers=old['workers'],
        operator_authority=ea['operator_authority'],gallery=ea['public_sources']['gallery'],
        old_result=U.bind(ROOT/'results/rc_h593_quality_operator_eval_v1/result.json'),
        old_validation=U.bind(ROOT/'results/rc_h593_quality_operator_eval_v1/validation.json'),
        arms=['NATIVE'],stages=list(N.STAGES),inside_branches=list(I.BRANCHES),queries=593,candidates=128,
        chunk_seconds=600,part_pairs=8,max_chunks=32,max_parallel=46,
        scope='Opened H593 full-C128 controlled internal mediator experiment; no training, native geometry fixed',external_GO=False))
    test['authority']=U.bind(U.AUTH);U.write(U.OUT/'preflight.json',test)
    print(dict(status='PREPARED',authority=U.bind(U.AUTH)),flush=True)


def inside_verify(a,w,index):
    import numpy as np
    folder=U.OUT/f'query{index:03d}';p=U.read(folder/'payload.json')
    records=[];sources=[];error=0.
    for source in p['parts']:
        part=torch.load(U.checked(source),map_location='cpu',weights_only=True)
        for pair in part['pairs']:
            path=folder/'inside'/(pair['reference_image']['sha256']+'.pt')
            data=torch.load(path,map_location='cpu',weights_only=True)
            assert data['authority']==U.bind(U.AUTH) and data['index']==index and data['query_sha']==p['query_image']['sha256']
            assert data['reference_sha']==pair['reference_image']['sha256'] and data['branches']==list(I.BRANCHES)
            assert set(data['sides'])=={0,1}
            record=dict(candidate_position=pair['candidate_position'],physical_row=pair['physical_row'],branches={})
            for branch in I.BRANCHES:
                checkpoints={}
                for stage in ('COARSE',*N.STAGES[1:]):
                    sides=[data['sides'][side]['coarse'][branch] if stage=='COARSE' else data['sides'][side]['stages'][stage]['branches'][branch] for side in (0,1)]
                    assert all(math.isfinite(item['mean_weight']) and 0<=item['mean_weight']<=1 for item in sides)
                    if stage in ('COARSE','HR1'):
                        for side,item in enumerate(sides):
                            err=abs(float(np.mean(item['weights'].numpy()))-item['mean_weight']);error=max(error,err);assert err<2e-12
                            if branch=='A1J1P1':
                                expected=pair['arms']['NATIVE']['stages'][stage]['AB' if side==0 else 'BA']['weights']
                                assert U.C.M.bit_equal(expected,item['weights']), ('NATIVE_TOKEN_PARITY',stage)
                    checkpoints[stage]=math.sqrt(sides[0]['mean_weight']*sides[1]['mean_weight'])
                record['branches'][branch]=checkpoints
            records.append(record);sources.append(U.bind(path))
    assert [r['candidate_position'] for r in records]==list(range(128))
    U.write(folder/'inside_manifest.json',dict(status='M_INSIDE_PATHS_SEALED',authority=U.bind(U.AUTH),
        query_id=p['query_id'],records=records,sources=sources,max_numpy_error=error))
    U.write(folder/'inside_validation.json',dict(status='M_INSIDE_PATHS_PASS',authority=U.bind(U.AUTH),
        payload=U.bind(folder/'inside_manifest.json'),candidates=128,branches=len(I.BRANCHES)))
    U.verify(a,w,index)


def join(a):
    observed=[]
    for index in range(593):
        folder=U.OUT/f'query{index:03d}'
        native=U.read(folder/'validation.json');iv=U.read(folder/'inside_validation.json')
        assert native['status']=='M_VISUAL_ORIGIN_QUERY_PASS' and iv['status']=='M_INSIDE_PATHS_PASS'
        assert native['authority']==iv['authority']==U.bind(U.AUTH)
        p=U.read(U.checked(native['payload']));extra=U.read(U.checked(iv['payload']))
        for source in extra['sources']:U.checked(source)
        observed.append((p,extra))
    oldval=U.read(U.checked(a['old_validation']));assert oldval['result']==a['old_result']
    old={r['query_id']:r for r in U.read(U.checked(a['old_result']))['rows']}
    labels={r['physical_row']:r['identity'] for r in U.read(U.checked(a['gallery']))['records']}
    rows=[]
    for p,inside in observed:
        r=old[p['query_id']];axis=p['candidate_physical_rows'];targets=[i for i,x in enumerate(axis) if labels[x]==r['identity']]
        assert len(targets)<=1 and bool(targets)==r['target_in_C128']
        row=dict(query_id=p['query_id'],fold=r['fold'],component=r['component'],raw_correct=r['correct']['RAW'],
            simple_correct=r['correct']['COST1_REFIT_M1Q0R0'],recall=bool(targets),branches={})
        for branch in I.BRANCHES:
            row['branches'][branch]={}
            for stage in N.STAGES:
                ms=[v['branches'][branch][stage] for v in inside['records']]
                best=max(range(128),key=lambda i:(ms[i],-axis[i]));item=dict(direct_correct=best in targets,
                    selected_physical_row=axis[best],candidate_mean=sum(ms)/128)
                if targets:
                    t=targets[0];worst=max(ms[i] for i in range(128) if i!=t);win=p['raw_winner']
                    item.update(target_M=ms[t],raw_M=ms[win],strongest_wrong_M=worst,
                        target_rank=1+sum(ms[i]>ms[t] for i in range(128) if i!=t),
                        target_vs_raw=(ms[t]-ms[win])/(abs(ms[t])+abs(ms[win])+1e-12),
                        target_vs_strongest_wrong=(ms[t]-worst)/(abs(ms[t])+abs(worst)+1e-12))
                row['branches'][branch][stage]=item
        rows.append(row)
    assert len(rows)==593 and sum(r['recall'] for r in rows)==570
    assert sum(r['branches']['A1J1P1']['HR1']['direct_correct'] for r in rows)==379
    groups={'ALL593':rows,'RAW_CORRECT':[r for r in rows if r['raw_correct']],
        'RAW_WRONG_RECALL_PRESENT':[r for r in rows if not r['raw_correct'] and r['recall']],
        'SIMPLE_RESCUES':[r for r in rows if not r['raw_correct'] and r['simple_correct']]}
    summary={}
    for name,rs in groups.items():
        summary[name]={}
        for branch in I.BRANCHES:
            summary[name][branch]={}
            for stage in N.STAGES:
                items=[r['branches'][branch][stage] for r in rs]
                valid=[x for x in items if 'target_M' in x]
                item=dict(n=len(rs),recall_present=len(valid),direct_correct=sum(x['direct_correct'] for x in items),
                    target_gt_raw=sum(x['target_vs_raw']>0 for x in valid),
                    target_gt_strongest_wrong=sum(x['target_vs_strongest_wrong']>0 for x in valid),
                    mean_candidate_M=sum(x['candidate_mean'] for x in items)/len(items))
                for key in ('target_M','strongest_wrong_M','target_vs_raw','target_vs_strongest_wrong'):
                    item['mean_'+key]=sum(x[key] for x in valid)/len(valid) if valid else None
                item['mean_normalized_gap_change_vs_native']=sum(
                    r['branches'][branch][stage]['target_vs_strongest_wrong']-
                    r['branches']['A1J1P1'][stage]['target_vs_strongest_wrong'] for r in rs if r['recall'])/len(valid) if valid else None
                summary[name][branch][stage]=item
    U.write(U.OUT/'result.json',dict(status='M_INSIDE_ALL593_COMPLETE',authority=U.bind(U.AUTH),rows=rows,summary=summary,
        scope=a['scope'],external_GO=False,interpretation='Conditional predictor/confidence paths, not total geometry-changing effects'))
    U.write(U.OUT/'validation.json',dict(status='M_ORIGIN_JOIN_COUNTS_PASS',authority=U.bind(U.AUTH),result=U.bind(U.OUT/'result.json')))
    print(dict(status='M_INSIDE_ALL593_COMPLETE'),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','worker','verify','control','join'));ap.add_argument('--index',type=int)
    args=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1);configure(args.index)
    if args.stage=='prepare':prepare()
    else:
        a,ws=U.guard(args.stage,args.index)
        if args.stage=='worker':U.worker(a,ws[args.index],args.index)
        elif args.stage=='verify':inside_verify(a,ws[args.index],args.index)
        elif args.stage=='control':U.control(a,ws)
        else:join(a)
