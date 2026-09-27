#!/usr/bin/env python3
"""Cache-only propagation of sealed new-group phase/amplitude interventions.

No RoMa/vision/LLM forward and no fitting. The same frozen external heads and
POST_REAL model as the original bridge receive the newly measured M values.
Only the two preselected candidates are intervened on. A third RAW winner keeps
its original HR1 M/L anchor; no complete-C128 predictions or accuracy are emitted.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import torch

RC=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RC/'programs'),str(RC/'src')]
import run_rc_m_causal128_frozen_bridge_v1 as B
BASE=RC/'results/rc_m_causal128_attribution_chain_v1'
DEFAULT=BASE/'newgroup_property_bridge_v1'
SELECTED=[78,82,93,96,100,101,104,108,118]
CONTRASTS=('phase_restoration_after_amplitude_permutation',
           'amplitude_restoration_under_local_phase','factorial_interaction','phase_restoration_native_amplitude')
GAP_METRICS=('M_gap','logM_gap','POST_content_gap','POST_action_gap','NATIVE7_action_gap','M_FREE_action_gap')
SIDE_METRICS=('M','logM','POST_content','POST_action','NATIVE7_action','M_FREE_action')
read,bind,checked,write=B.read,B.bind,B.checked,B.write


def prepare(root,source,bridge):
    """Freeze selected metadata and existing model interfaces; no new M is read."""
    pp=read(source/'protocol.json');bp=read(bridge/'protocol.json')
    bv=read(bridge/'validation.json')
    assert bv['status']=='CAUSAL128_FROZEN_BRIDGE_INDEPENDENT_ARITHMETIC_PASS'
    assert bv['protocol']==bind(bridge/'protocol.json')
    assert pp['indices']==SELECTED and len(pp['workers'])==9 and pp['new_groups']==9
    items={r['index']:r for r in read(checked(bp['inputs']))['rows']}
    workers=[]
    for w in pp['workers']:
        item=items[w['index']]
        assert item['query_id']==w['query_id'] and item['component']==w['group'] and item['fold']==w['fold']
        assert item['post_row']['candidate_ids']==w['axis'] and w['index'] in bp['group_replication_indices']
        positions=sorted([w['target_position'],w['fixed_wrong_position']]) if w['target_present'] else []
        assert positions==[p['position'] for p in sorted(w['pairs'],key=lambda p:p['position'])]
        workers.append(dict(index=w['index'],query_id=w['query_id'],component=w['group'],fold=w['fold'],
            axis=w['axis'],target_present=w['target_present'],target_position=w['target_position'],
            fixed_wrong_position=w['fixed_wrong_position'],positions=positions,
            query_validation_path=str(source/f"query{w['index']:03d}/validation.json")))
    assert len({w['component'] for w in workers})==9
    assert all(w['target_present'] and len(w['positions'])==2 for w in workers)
    assert len(pp['arms'])==18 and len(pp['phase_arms'])==8
    root.mkdir(parents=True,exist_ok=True)
    protocol=dict(status='NEWGROUP_PROPERTY_BRIDGE_PROTOCOL_FROZEN',version=1,
        source_root=str(source.resolve()),bridge_root=str(bridge.resolve()),
        phase_protocol=bind(source/'protocol.json'),parent_bridge_protocol=bind(bridge/'protocol.json'),
        parent_bridge_validation=bind(bridge/'validation.json'),inputs=bp['inputs'],
        authority=bp['authority'],manifest=bp['manifest'],endpoints=bp['endpoints'],
        inherited_code_sources=bp['code_sources'],code_sources=[bind(Path(__file__)),bind(Path(B.__file__))],
        workers=workers,indices=SELECTED,groups=9,pairs=18,arms=pp['arms'],phase_arms=pp['phase_arms'],
        contrasts=list(CONTRASTS),threads=4,maximum_numpy_error=2e-10,
        source_join_status_required='PHASE_NEWGROUP_REPLICATION_JOIN_PASS',
        primary='phase_restoration_after_amplitude_permutation on target-minus-fixed-wrong gaps; also report target and wrong separately',
        anchor_policy='Only the two preselected target/fixed-wrong M values change. Both posterior L values are recomputed under the SAME frozen POST_REAL. If RAW winner is neither pair, its original HR1 M/L stay fixed; if in the pair, its own intervened values are used. HOLD logit is exactly zero.',
        external_path='Recompute every algebraic M descendant of NATIVE7 and M_FREE while original local weight shape, ColNomic evidence, natural axis, and theta stay fixed.',
        post_path='Same original held-fold POST_REAL adapter and INTERNAL3 head. Real M enters the adapter; final head has no direct M input.',
        candidate_axis='Original natural ColNomic C128 is verified but only two fixed candidates are measured; no full-C128 ranking or accuracy output.',
        observation_boundary='The locked nine groups were absent from prior F71 mechanism work but were already used in historical H593 evaluation. This is group-isolated mechanism replication, not untouched external confirmation.',
        prepare_new_M_reads=0,training=False,new_parameters=0,new_gpu_forwards=0,
        new_RoMa_visual_or_LLM_forwards=0,CPU_post_adapter_and_projection_only=True,
        unique_causality_claim=False,identity_sufficiency_claim=False)
    write(root/'protocol.json',protocol,immutable=True)
    write(root/'preparation_validation.json',dict(status='NEWGROUP_PROPERTY_BRIDGE_PREPARATION_PASS',
        protocol=bind(root/'protocol.json'),groups=9,pairs=18,arms=18,new_M_reads=0),immutable=True)
    print(json.dumps(dict(status=protocol['status'],groups=9,pairs=18,arms=18,new_M_reads=0)),flush=True)
    return 0


def guard(root,require_join=True):
    p=read(root/'protocol.json');pb=bind(root/'protocol.json')
    assert p['status']=='NEWGROUP_PROPERTY_BRIDGE_PROTOCOL_FROZEN'
    for b in [*p['code_sources'],*p['inherited_code_sources'],p['phase_protocol'],p['parent_bridge_protocol'],
              p['parent_bridge_validation'],p['inputs'],p['authority'],p['manifest']]:checked(b)
    for endpoint in p['endpoints'].values():checked(endpoint['snapshot'])
    bp=read(p['parent_bridge_protocol']['path'])
    assert bp['endpoints']==p['endpoints'] and bp['authority']==p['authority']
    assert p['threads']==4 and p['indices']==SELECTED
    source=Path(p['source_root']);joined=None
    if require_join:
        path=source/'validation.json'
        if not path.exists():return p,pb,bp,None
        joined=read(path)
        assert joined['status']==p['source_join_status_required'] and joined['protocol']==p['phase_protocol']
        checked(joined['result']);checked(joined['independent_cells'])
        assert joined['candidate_arms']==p['pairs']*len(p['arms'])==324
        assert joined['groups']==9 and joined['all_science_cells_CPU']
    return p,pb,bp,joined


def source_pairs(p,w):
    q=read(w['query_validation_path'])
    assert q['status']=='PHASE_NEWGROUP_QUERY_PASS' and q['protocol']==p['phase_protocol']
    assert q['index']==w['index'] and q['query_id']==w['query_id'] and q['group']==w['component']
    assert len(q['pairs'])==2 and q['target_present']
    pairs={}
    for binding in q['pairs']:
        pair=read(checked(binding));pos=pair['position']
        assert pair['status']=='PHASE_NEWGROUP_PAIR_PASS' and pair['protocol']==p['phase_protocol']
        assert pair['index']==w['index'] and pos in w['positions']
        assert pair['role']==('target' if pos==w['target_position'] else 'fixed_wrong')
        assert [a['arm'] for a in pair['arms']]==p['arms']
        pairs[pos]=(binding,pair)
    assert sorted(pairs)==w['positions']
    return pairs


def score_metrics(item,masses,content,head,target,wrong):
    post=B.decision(item['post_row'],content,head);z=np.zeros(128)
    z[item['post_row']['challenger_positions']]=post['logits']
    ext=B.external(item['external'],masses)
    sides={}
    for role,pos in [('target',target),('fixed_wrong',wrong)]:
        sides[role]=dict(M=float(masses[pos]),logM=float(np.log(masses[pos])),POST_content=float(content[pos]),
            POST_action=float(z[pos]),NATIVE7_action=ext['NATIVE7']['logits128'][pos],
            M_FREE_action=ext['M_FREE']['logits128'][pos])
    gap=dict(M_gap=sides['target']['M']-sides['fixed_wrong']['M'],
        logM_gap=sides['target']['logM']-sides['fixed_wrong']['logM'],
        POST_content_gap=sides['target']['POST_content']-sides['fixed_wrong']['POST_content'],
        POST_action_gap=sides['target']['POST_action']-sides['fixed_wrong']['POST_action'],
        NATIVE7_action_gap=sides['target']['NATIVE7_action']-sides['fixed_wrong']['NATIVE7_action'],
        M_FREE_action_gap=sides['target']['M_FREE_action']-sides['fixed_wrong']['M_FREE_action'])
    # Predictions are intentionally not returned; their scores contain an
    # explicitly sparse intervention, not a fully rerun candidate universe.
    return gap,sides


def audit_row(row,item,arms):
    maximum=B.audit_joint_row(row,item,arms)
    target,wrong=row['target_position'],row['fixed_wrong_position'];winner=item['post_row']['winner_index']
    original=read(checked(row['original_POST']))['arms']['NATIVE']['L'];data={}
    for pos,b in zip(sorted([target,wrong]),row['patch_outputs']):
        with np.load(checked(b)) as z:
            data[pos]=dict(M=z['M'].copy(),L=z['maxsim'].mean(1))
    for a,arm in enumerate(arms):
        ms=np.asarray(item['masses']['ORIGINAL_HR1']).copy();ls=np.asarray(original).copy()
        for pos in data:ms[pos]=data[pos]['M'][a];ls[pos]=data[pos]['L'][a]
        for role,pos in [('target',target),('fixed_wrong',wrong)]:
            # The separate NumPy formula explicitly maps RAW to HOLD=0.
            difference=B.independent_gap(item,ms,ls,row['head'],pos,winner)
            independent=dict(M=float(ms[pos]),logM=float(np.log(ms[pos])),POST_content=float(ls[pos]),
                **{key:difference[key+'_gap'] for key in ('POST_action','NATIVE7_action','M_FREE_action')})
            error=max(abs(independent[k]-row['individual_endpoints'][arm][role][k]) for k in SIDE_METRICS)
            assert error<2e-10,('INDEPENDENT_SINGLE_CANDIDATE_ENDPOINT',row['index'],arm,role,error)
            maximum=max(maximum,error)
    return maximum


def worker(root,budget,index=None):
    p,pb,bp,joined=guard(root)
    if joined is None:
        write(root/'waiting.json',dict(status='NEWGROUP_PROPERTY_BRIDGE_WAITING_FOR_PHASE_JOIN',protocol=pb));return 75
    items={r['index']:r for r in read(p['inputs']['path'])['rows']}
    context=B.Context(bp);started=time.monotonic();deadline=started+budget
    selected=p['workers'] if index is None else [w for w in p['workers'] if w['index']==index]
    assert selected and (index is None or len(selected)==1)
    for w in selected:
        folder=root/f"query{w['index']:03d}";folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (folder/'validation.json').exists():
                v=read(folder/'validation.json');assert v['protocol']==pb;checked(v['result']);continue
            if time.monotonic()>deadline-20:return 75
            item=items[w['index']];pairs=source_pairs(p,w);adapter,head,cache,refs=context.load(item)
            bypos={};input_bindings=[];patch_bindings=[];pair_bindings=[]
            for pos in w['positions']:
                binding,pair=pairs[pos];pair_bindings.append(binding);masses=[]
                for arm in p['arms']:
                    source=next(r for r in pair['arms'] if r['arm']==arm)
                    tensor=torch.load(checked(source['payload']),map_location='cpu',weights_only=True,mmap=True)
                    assert tensor['protocol']==p['phase_protocol'] and tensor['capture']==pair['capture']
                    assert tensor['index']==w['index'] and tensor['position']==pos and tensor['arm']==arm
                    assert tensor['physical_row']==w['axis'][pos]
                    mass=math.sqrt(math.fsum(tensor['sides'][0]['weights'].tolist())/tensor['sides'][0]['weights'].numel()
                        *math.fsum(tensor['sides'][1]['weights'].tolist())/tensor['sides'][1]['weights'].numel())
                    assert abs(mass-source['M'])<2e-12 and abs(mass-tensor['M'])<2e-12
                    # Preserve the sealed actual M rounding, audit independently above.
                    masses.append(float(tensor['M']));input_bindings.append(source['payload'])
                patch=folder/f'pair{pos:03d}.npz';receipt=folder/f'pair{pos:03d}.json'
                if receipt.exists():
                    rec=read(receipt);assert rec['protocol']==pb and rec['source_pair']==binding and rec['position']==pos
                    with np.load(checked(rec['patch'])) as z:
                        assert list(z['worlds'])==p['arms'] and np.array_equal(z['M'],masses)
                        values=z['maxsim'].copy();locations=z['argmax'].copy()
                        assert np.max(np.abs(values.mean(1)-z['L']))<1e-14
                else:
                    if time.monotonic()>deadline-20:return 75
                    values,locations=context.candidate(adapter,cache,refs[pos],masses)
                    if patch.exists():
                        with np.load(patch) as z:
                            assert np.array_equal(z['M'],masses) and np.array_equal(z['maxsim'],values) and np.array_equal(z['argmax'],locations)
                    else:B.npz_once(patch,worlds=np.asarray(p['arms']),M=np.asarray(masses),maxsim=values,argmax=locations,L=values.mean(1))
                    write(receipt,dict(status='NEWGROUP_PROPERTY_BRIDGE_PAIR_COMPLETE',protocol=pb,index=w['index'],position=pos,
                        source_pair=binding,patch=bind(patch),snapshot=p['endpoints'][str(w['fold'])]['snapshot']),immutable=True)
                bypos[pos]=dict(M=np.asarray(masses),L=values.mean(1));patch_bindings.append(bind(patch))
            original_path=RC/'results/rc_postllm_h593_decomposition_v1/queries'/w['query_id']/'result.json'
            original=read(original_path)['arms']['NATIVE']['L'];metrics={};sides={}
            for a,arm in enumerate(p['arms']):
                ms=np.asarray(item['masses']['ORIGINAL_HR1']).copy();ls=np.asarray(original).copy()
                for pos in w['positions']:ms[pos]=bypos[pos]['M'][a];ls[pos]=bypos[pos]['L'][a]
                metrics[arm],sides[arm]=score_metrics(item,ms,ls,head,w['target_position'],w['fixed_wrong_position'])
            row=dict(status='NEWGROUP_PROPERTY_BRIDGE_QUERY_COMPLETE',protocol=pb,index=w['index'],query_id=w['query_id'],
                component=w['component'],fold=w['fold'],candidate_physical_rows=w['axis'],
                target_position=w['target_position'],fixed_wrong_position=w['fixed_wrong_position'],
                RAW_winner=item['post_row']['winner_index'],head=head,snapshot=p['endpoints'][str(w['fold'])]['snapshot'],
                original_POST=bind(original_path),saved_arm_inputs=input_bindings,patch_outputs=patch_bindings,
                source_pair_validations=pair_bindings,source_query_validation=bind(w['query_validation_path']),
                source_phase_join=bind(Path(p['source_root'])/'validation.json'),arms=metrics,individual_endpoints=sides,
                full_C128_predictions_computed=False,anchor_policy=p['anchor_policy'])
            maximum=audit_row(row,item,p['arms'])
            write(folder/'result.json',row,immutable=True)
            write(folder/'validation.json',dict(status='NEWGROUP_PROPERTY_BRIDGE_QUERY_NUMPY_PASS',protocol=pb,
                result=bind(folder/'result.json'),index=w['index'],pairs=2,arms=18,
                independent_numpy_max_error=maximum,RAW_HOLD_zero_explicit=True),immutable=True)
            print(json.dumps(dict(status='NEWGROUP_PROPERTY_BRIDGE_QUERY_NUMPY_PASS',index=w['index'],max_error=maximum)),flush=True)
    return 0


def contrasts(values):
    def average(prefix,scope):
        return float(np.mean([values[prefix+f'{scope}_{axis}_{sign}'] for axis in ('X','Y') for sign in ('PLUS','MINUS')]))
    a0g,a0l,a1g,a1l=average('','GLOBAL'),average('','LOCAL'),average('AMP_PERMUTE__','GLOBAL'),average('AMP_PERMUTE__','LOCAL')
    return dict(phase_restoration_after_amplitude_permutation=a1g-a1l,
        amplitude_restoration_under_local_phase=a0l-a1l,factorial_interaction=(a0g-a0l)-(a1g-a1l),
        phase_restoration_native_amplitude=a0g-a0l)


def join(root):
    p,pb,bp,joined=guard(root)
    if joined is None:return 75
    items={r['index']:r for r in read(p['inputs']['path'])['rows']};rows=[];seals=[];maximum=0.;heads={}
    for w in p['workers']:
        path=root/f"query{w['index']:03d}/validation.json"
        if not path.exists():return 75
        v=read(path);assert v['status']=='NEWGROUP_PROPERTY_BRIDGE_QUERY_NUMPY_PASS' and v['protocol']==pb
        row=read(checked(v['result']));assert row['protocol']==pb and row['index']==w['index']
        assert row['query_id']==w['query_id'] and row['component']==w['component'] and row['fold']==w['fold']
        assert row['candidate_physical_rows']==w['axis'] and row['full_C128_predictions_computed'] is False
        assert row['target_position']==w['target_position'] and row['fixed_wrong_position']==w['fixed_wrong_position']
        assert row['source_phase_join']==bind(Path(p['source_root'])/'validation.json')
        for b in [*row['source_pair_validations'],row['source_query_validation']]:checked(b)
        if w['fold'] not in heads:
            cp=torch.load(checked(p['endpoints'][str(w['fold'])]['snapshot']),map_location='cpu',weights_only=True)
            assert cp['arm']=='POST_REAL' and cp['authority']==p['authority'];heads[w['fold']]=cp['head'].double().tolist()
        assert row['head']==heads[w['fold']]
        maximum=max(maximum,audit_row(row,items[w['index']],p['arms']))
        rows.append(row);seals.append(bind(path))
    assert len(rows)==9 and len({r['component'] for r in rows})==9
    summary={};individual={}
    for metric in GAP_METRICS:
        effects=[dict(component=r['component'],**contrasts({a:r['arms'][a][metric] for a in p['arms']})) for r in rows]
        summary[metric]={key:B.group_summary(effects,key) for key in CONTRASTS}
    for role in ('target','fixed_wrong'):
        individual[role]={}
        for metric in SIDE_METRICS:
            effects=[dict(component=r['component'],**contrasts({a:r['individual_endpoints'][a][role][metric] for a in p['arms']})) for r in rows]
            individual[role][metric]={key:B.group_summary(effects,key) for key in CONTRASTS}
    result=dict(status='NEWGROUP_PROPERTY_BRIDGE_COMPLETE',protocol=pb,groups=9,queries=9,pairs=18,arms=18,
        gap_summary=summary,individual_summary=individual,rows=rows,primary=p['primary'],
        anchor_policy=p['anchor_policy'],observation_boundary=p['observation_boundary'],
        source_phase_join=bind(Path(p['source_root'])/'validation.json'),parent_bridge_validation=p['parent_bridge_validation'],
        full_C128_predictions_computed=False,accuracy_claim=False,training=False,new_parameters=0,new_gpu_forwards=0,
        new_RoMa_visual_or_LLM_forwards=0,unique_causality_claim=False,
        scope='Same frozen external NATIVE7 / M_FREE and POST_REAL conditional property propagation, fixed two candidates only.')
    write(root/'result.json',result,immutable=True)
    lines=['# New-group property restoration: frozen external and internal propagation','',p['observation_boundary'],'',
        p['anchor_policy'],'','No training, no RoMa/vision/LLM forward, no new parameters. Only cached POST hidden states, frozen adapter/projector and reference tokens are used.',
        '', '| Contrast | log M gap | NATIVE7 action gap | M_FREE action gap | POST content gap | POST action gap |',
        '|---|---|---|---|---|---|']
    def cell(v):return f"{v['mean']:+.8f} {v['exploratory_bootstrap95']} ({v['positive_groups']}/{v['groups']} positive groups)"
    for contrast in CONTRASTS:
        lines.append('| '+contrast+' | '+' | '.join(cell(summary[m][contrast]) for m in ('logM_gap','NATIVE7_action_gap','M_FREE_action_gap','POST_content_gap','POST_action_gap'))+' |')
    lines+=['','Target and fixed wrong are also reported separately to distinguish an increased target score from suppression of the wrong candidate:',
        '', '| Endpoint | Contrast | Target effect | Fixed wrong effect |','|---|---|---|---|']
    for metric in ('logM','NATIVE7_action','M_FREE_action','POST_content','POST_action'):
        for contrast in CONTRASTS:lines.append('| '+metric+' | '+contrast+' | '+cell(individual['target'][metric][contrast])+' | '+cell(individual['fixed_wrong'][metric][contrast])+' |')
    lines+=['','Intervals are exploratory group bootstraps. Fixed target/wrong pair selection and the third-RAW-winner anchor do not license a full-C128 accuracy claim.',
        'This closes same-model property-effect propagation on the new mechanism groups; it does not perform phase/amplitude-specific deletion and retraining, establish unique causality, or demonstrate external generalization.','']
    text='\n'.join(lines);report=root/'report.md'
    if report.exists():assert report.read_text()==text,'IMMUTABLE_REPORT_DRIFT'
    else:report.write_text(text)
    write(root/'validation.json',dict(status='NEWGROUP_PROPERTY_BRIDGE_INDEPENDENT_PASS',protocol=pb,
        result=bind(root/'result.json'),report=bind(report),query_validations=seals,groups=9,pairs=18,candidate_arms=324,
        independent_numpy_M_patch_L_action_and_individual_max_error=maximum,
        original_fold_models_and_heads_preserved=True,full_C128_predictions_computed=False,new_gpu_forwards=0),immutable=True)
    print(json.dumps(dict(status='NEWGROUP_PROPERTY_BRIDGE_INDEPENDENT_PASS',groups=9,candidate_arms=324,max_error=maximum)),flush=True)
    return 0


def self_test():
    # Use only already-opened historical bridge inputs. New replication M is not read.
    rows=read(B.PHASE/'inputs.json')['rows'];head=[.17,.73,-.09];maxerr=0.;tests=0
    for item in rows:
        w=item['post_row']['winner_index'];mass=np.asarray(item['masses']['NATIVE']);content=np.asarray(item['external']['free_content'])
        for target,wrong in [(w,(w+1)%128),((w+1)%128,w),((w+2)%128,(w+3)%128)]:
            gap,sides=score_metrics(item,mass,content,head,target,wrong)
            expected=B.independent_gap(item,mass,content,head,target,wrong)
            maxerr=max(maxerr,max(abs(gap[k]-expected[k]) for k in GAP_METRICS));tests+=1
            for role,pos in [('target',target),('fixed_wrong',wrong)]:
                independent=B.independent_gap(item,mass,content,head,pos,w)
                for metric in ('POST_action','NATIVE7_action','M_FREE_action'):
                    assert abs(sides[role][metric]-independent[metric+'_gap'])<2e-10
                if pos==w:assert sides[role]['POST_action']==sides[role]['NATIVE7_action']==sides[role]['M_FREE_action']==0.
            assert 'prediction_position' not in gap and 'prediction_position' not in sides
    assert maxerr<2e-10
    values={}
    for prefix,g,l in [('',11.,5.),('AMP_PERMUTE__',3.,1.)]:
        for scope,value in [('GLOBAL',g),('LOCAL',l)]:
            for axis in ('X','Y'):
                for sign in ('PLUS','MINUS'):values[prefix+scope+'_'+axis+'_'+sign]=value
    assert contrasts(values)==dict(phase_restoration_after_amplitude_permutation=2.,amplitude_restoration_under_local_phase=4.,factorial_interaction=4.,phase_restoration_native_amplitude=6.)
    print(json.dumps(dict(status='NEWGROUP_PROPERTY_BRIDGE_SELF_TEST_PASS',old_query_count=len(rows),old_pair_cases=tests,
        max_independent_error=maxerr,RAW_HOLD_zero_explicit=True,four_factorial_contrasts=True,new_replication_M_reads=0,
        new_model_forwards=0)),flush=True)
    return 0


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','worker','join','self-test'))
    ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--source-root',type=Path,default=BASE/'phase_replication')
    ap.add_argument('--bridge-root',type=Path,default=BASE/'bridge');ap.add_argument('--budget',type=float,default=430);ap.add_argument('--index',type=int)
    args=ap.parse_args()
    if args.stage=='self-test':return self_test()
    if args.stage=='prepare':return prepare(args.root,args.source_root,args.bridge_root)
    B.configure()
    try:return worker(args.root,args.budget,args.index) if args.stage=='worker' else join(args.root)
    except (AssertionError,KeyError,ValueError,FileNotFoundError) as exc:
        write(args.root/'blocked.json',dict(status='NEWGROUP_PROPERTY_BRIDGE_BLOCKED',error_type=type(exc).__name__,error=str(exc)))
        print(json.dumps(dict(status='NEWGROUP_PROPERTY_BRIDGE_BLOCKED',error=str(exc))),flush=True);return 2

if __name__=='__main__':sys.exit(main())
