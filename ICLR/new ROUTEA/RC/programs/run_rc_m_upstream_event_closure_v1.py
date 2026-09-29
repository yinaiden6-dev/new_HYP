#!/usr/bin/env python3
"""Extend matched registration to the full original-event cohort and audit identity selectivity."""
import argparse
from pathlib import Path
import sys
import math
import numpy as np

RC=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RC/'programs'),str(RC/'src')]
import run_rc_m_fine_c128_extension_v1 as E
F=E.F
read,write,bind,checked=E.read,E.write,E.bind,E.checked
OUT=RC/'results/rc_m_upstream_event_closure_v1'


def prepare(torch):
    ep,eb=E.guard()
    import run_rc_roma_position_factor_v1 as G
    pairs=[];queries=[];sources=[]
    for ordinal,index in enumerate(E.NEW):
        item=next(x for x in ep['items'] if x['index']==index)
        worker=next(x for x in read(E.OUT/'manifest.json')['workers'] if x['execution_ordinal']==index)
        gvpath=E.OUT/f'query{index:03d}/gpu_validation.json';gv=read(gvpath)
        assert gv['status']=='POSITION_FACTOR_ALL128_PASS' and gv['authority']==ep['authority']
        rp=torch.load(checked(worker['roma']['payload']),map_location='cpu',weights_only=True,mmap=True)
        r=rp['records'][worker['source_index']];axis=item['post_row']['candidate_ids']
        assert axis==gv['candidate_physical_rows']==r['candidate_physical_rows']
        for pos,b in enumerate(gv['pairs']):
            pairs.append(dict(pair_index=len(pairs),index=index,ordinal=ordinal,position=pos,physical_row=axis[pos],old_pair=b,
                geometry_metadata=[r['query_geometry'],r['candidates'][pos]['reference_geometry']],
                image_shas=[r['query_source_sha256'],r['candidates'][pos]['reference_image_sha256']]))
        queries.append(dict(index=index,ordinal=ordinal,query_id=item['query_id'],cohort='ALL_CHANGED11',item=item,
            original_bridge=ep['original_anchors'][str(index)],phase_reuse=None))
        sources.append(bind(gvpath))
    write(OUT/'inputs.json',dict(queries=queries,pairs=pairs))
    fp=read(F.OUT/'protocol.json');pilot=read(E.OUT/'pilot_validation.json')
    write(OUT/'protocol.json',dict(status='UPSTREAM_EVENT_CONTEXT_FROZEN',extension_protocol=eb,inputs=bind(OUT/'inputs.json'),
        sources=sources,head_state=pilot['head_state'],bridge_protocol=ep['bridge_protocol'],
        inherited_fine_protocol=bind(F.OUT/'protocol.json'),
        code_sources=[bind(Path(__file__)),bind(RC/'slurm/rc_m_upstream_event_closure_v1.sbatch'),bind(Path(E.__file__)),*fp['code_sources']],
        shards=8,queries=8,pairs=1024,context_arms=F.CONTEXT,threads=4,torch_version=str(torch.__version__),
        scope='New8 complete C128 context worlds on the same CPU backend as the historical11. Merge19 only after all source validations pass; selected original events remain separate from8 fixed controls.',
        mediator=fp['mediator'],new_GPU_encoder_matcher_LLM_forwards=0,new_training=0,
        joint_fine_cause_claim=False,identity_selectivity='Report target-versus-every-wrong logM differences, mean candidate-level support changes, fixed-native strongest-M wrong, and retained/lost original corrections. Keep decoder common-shift sensitivity explicit.'))
    print('UPSTREAM_EVENT_CONTEXT_PREPARED',flush=True)


def guard(root,torch):
    assert root==OUT
    p=read(OUT/'protocol.json');assert p['threads']==torch.get_num_threads()==4 and p['torch_version']==str(torch.__version__)
    for b in p['code_sources']+p['sources']:checked(b)
    for k in ['extension_protocol','inputs','head_state','bridge_protocol','inherited_fine_protocol']:checked(p[k])
    return p,bind(OUT/'protocol.json'),read(p['inputs']['path'])


def masses(torch):
    p,pb,inputs=guard(OUT,torch);rows={q['index']:dict(index=q['index'],M={a:[] for a in F.CONTEXT}) for q in inputs['queries']}
    sources=[];error=0.
    for w in inputs['pairs']:
        path=OUT/'context_pairs'/f"{w['pair_index']:05d}"/'validation.json';v=read(path)
        assert v['protocol']==pb and v['status']=='FINE_C128_CONTEXT_PAIR_PASS'
        assert [x['arm'] for x in v['arms']]==F.CONTEXT
        for a in v['arms']:
            x=torch.load(checked(a['payload']),map_location='cpu',weights_only=True,mmap=True)
            assert (x['index'],x['position'])==(w['index'],w['position']) and x['protocol']==pb
            m=F.scalar_mass(x['sides']);other=float(np.sqrt(np.mean(x['sides'][0]['weights'].numpy())*np.mean(x['sides'][1]['weights'].numpy())))
            error=max(error,abs(m-other));assert abs(m-other)<2e-12 and abs(m-x['M'])<2e-12 and m==a['M']
            rows[w['index']]['M'][a['arm']].append(m)
        sources.append(bind(path))
    assert len(sources)==1024 and all(len(m)==128 for r in rows.values() for m in r['M'].values())
    write(OUT/'context_masses.json',dict(protocol=pb,rows=list(rows.values()),sources=sources))
    write(OUT/'context_validation.json',dict(status='FINE_C128_CONTEXT_MASSES_PASS',protocol=pb,result=bind(OUT/'context_masses.json'),pairs=1024,arms=13,maximum_mass_error=error,backend='CPU'))
    return 0


def characterize(d,item,label,family):
    targets=[k for k,x in enumerate(item['post_row']['candidate_identities']) if x==label['identity']];assert len(targets)<=1
    wrong=[k for k in range(128) if k not in targets]
    native=np.log(np.asarray(d['worlds']['NATIVE']['M']));fixed=max(wrong,key=lambda k:native[k])
    coeff=F.continuous_contrasts(family=='phase');effects={}
    for name,weights in coeff.items():
        # Every contrast stays within its own CPU or historical-GPU family.
        lm=sum(c*np.log(np.asarray(d['worlds'][a]['M'])) for a,c in weights.items())
        common=float(lm.mean());res=lm-common
        value=dict(candidate_logM_effect=lm.tolist(),candidate_common_level=common,candidate_centered_effect=res.tolist(),
            centered_mean=float(res.mean()),fixed_native_strongest_M_wrong=fixed)
        assert abs(float(res.mean()))<1e-12
        if targets:
            t=targets[0];gaps=lm[t]-lm[wrong]
            value.update(target_logM_effect=float(lm[t]),mean_wrong_logM_effect=float(lm[wrong].mean()),
                target_vs_mean_wrong=float(gaps.mean()),target_vs_fixed_native_wrong=float(lm[t]-lm[fixed]),
                target_vs_each_wrong=gaps.tolist(),wrong_positions=wrong,positive_comparisons=int((gaps>0).sum()),
                comparison_count=len(wrong),note='127 comparisons share one query; they are not127 independent observations.')
        effects[name]=value
    correct={};pred={};margins={}
    for model in F.MODELS:
        correct[model]={};pred[model]={};margins[model]={}
        for a,v in d['worlds'].items():
            x=v['POST'] if model=='POST' else v['external'][model];pos=x['prediction_position'];z=x['logits128']
            correct[model][a]=pos in targets;pred[model][a]=pos;margins[model][a]=None if not targets else z[targets[0]]-max(z[k] for k in wrong)
    return dict(index=d['index'],query_id=d['query_id'],component=item['component'],family=family,
        cohort='MAIN8' if d['index'] in E.MAIN else 'ALL_CHANGED11',target_present=bool(targets),
        raw_correct=d['winner'] in targets,correct=correct,prediction=pred,target_vs_strongest_wrong_margin=margins,effects=effects)


def analyze(torch):
    p,pb,inputs=guard(OUT,torch);ep,eb=E.guard()
    ev=read(E.OUT/'validation.json');fv=read(F.OUT/'validation.json')
    assert ev['status']=='FINE_C128_EXTENSION_ACCOUNTING_PASS' and fv['status']=='FINE_C128_THREE_PROPERTY_ACCOUNTING_PASS'
    checked(ev['result']);checked(fv['result'])
    bp=read(checked(p['bridge_protocol']));items={r['index']:r for r in read(checked(bp['inputs']))['rows']}
    labels={r['query_id']:r for r in read(checked(ep['label_source']))['rows']};rows=[];sources=[];maximum=0.
    for family in ['phase','context']:
        for index in ep['all_indices']:
            if family=='phase':
                source=E.OUT if index in E.NEW else F.OUT;folder=source/'phase'/f'query{index:03d}'
            else:
                source=OUT if index in E.NEW else F.OUT;folder=source/'context_bridge'/f'query{index:03d}'
            v=read(folder/'validation.json');b=v['result'];d=read(checked(b));item=items[index]
            expected=eb if source==E.OUT else (pb if source==OUT else bind(F.OUT/'protocol.json'))
            assert v['protocol']==expected and d['axis']==item['post_row']['candidate_ids']
            maximum=max(maximum,F.independent_scores(item,d['worlds'],d['head']))
            for pos,bp0 in enumerate(d['patches']):
                with np.load(checked(bp0)) as z:
                    arms=F.PHASE_ARMS if family=='phase' else F.CONTEXT
                    assert z['worlds'].tolist()==arms
                    assert np.array_equal(z['maxsim'].mean(1),[d['worlds'][a]['L'][pos] for a in arms])
                    assert np.array_equal(z['M'],[d['worlds'][a]['M'][pos] for a in arms])
            rows.append(characterize(d,item,labels[d['query_id']],family));sources.append(b)
    summary={};stats={}
    for family in ['phase','context']:
        rr=[r for r in rows if r['family']==family];arms=(F.PHASE_ARMS if family=='phase' else F.CONTEXT)+['ORIGINAL_HR1']
        summary[family]={c:{m:{a:F.aggregate(rr,m,a,c) for a in arms} for m in F.MODELS} for c in ['MAIN8','ALL_CHANGED11']}
        stats[family]={}
        for c in ['MAIN8','ALL_CHANGED11']:
            cc=[r for r in rr if r['cohort']==c and r['target_present']];stats[family][c]={}
            for name in F.continuous_contrasts(family=='phase'):
                stats[family][c][name]={metric:F.group_stats([(r['component'],r['effects'][name][metric]) for r in cc]) for metric in
                    ['candidate_common_level','target_logM_effect','mean_wrong_logM_effect','target_vs_mean_wrong','target_vs_fixed_native_wrong']}
        for m,want in ep['original_cohorts'].items():
            o=summary[family]['ALL_CHANGED11'][m]['ORIGINAL_HR1'];assert o['rescues']==want['rescues'] and o['breaks']==want['breaks']
    write(OUT/'result.json',dict(protocol=pb,rows=rows,summary=summary,group_stats=stats,sources=sources,
        status='UPSTREAM_EVENT_COVERAGE_AND_SELECTIVITY_COMPLETE',scope=p['scope'],
        unresolved=['Conditional interventions do not identify a unique semantic cause.',
                    'Coherence and registration are separate experiments, not a crossed factorial decomposition.',
                    'A/J/P decoder sees their sum; source-specific unique allocation is not generally identifiable.',
                    'The opened selected-event cohort is not untouched generalization evidence.']))
    write(OUT/'validation.json',dict(status='UPSTREAM_EVENT_COVERAGE_AND_SELECTIVITY_PASS',protocol=pb,result=bind(OUT/'result.json'),
        queries=19,families=2,candidates=128,maximum_numpy_error=maximum,unique_root_cause_claim=False))
    lines=['# Upstream discrimination and original-event coverage','',p['scope'],'',
        'Phase and registration have separate native backends. Absolute support, target-vs-wrong relative advantage and downstream original corrections are reported separately.',
        '', '| Family | Cohort | Contrast | Target vs mean wrong logM | Exploratory group95% interval |','|---|---|---|---:|---|']
    for family,cs in stats.items():
        for c,names in cs.items():
            for name,ms in names.items():
                s=ms['target_vs_mean_wrong'];lines.append(f"| {family} | {c} | {name} | {s['mean']} | {s['exploratory_bootstrap95']} |")
    lines+=['','Original rescue-retention sets and breaks are preserved for every world and endpoint in result.json. Positive common-level changes alone do not establish identity-selective support. Exploratory intervals are not equivalence tests or unique-cause certificates.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');return 0


if __name__=='__main__':
    contract=read(OUT/'launch_contract.json')
    for source in contract['sources']:checked(source)
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','context','masses','bridge','analyze']);ap.add_argument('--index',type=int,default=0);x=ap.parse_args();t=F.configure()
    if x.stage=='prepare':prepare(t);code=0
    else:
        F.guard=guard
        if x.stage=='context':code=F.context_worker(OUT,x.index,8,430,0,t)
        elif x.stage=='masses':code=masses(t)
        elif x.stage=='bridge':code=F.score_worker(OUT,x.index,430,t,False)
        else:code=analyze(t)
    raise SystemExit(code)
