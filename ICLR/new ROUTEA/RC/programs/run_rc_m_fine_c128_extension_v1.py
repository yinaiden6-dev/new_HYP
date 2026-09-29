#!/usr/bin/env python3
"""Complete the original changed-decision cohort; leave historical runs sealed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

RC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RC/'programs'), str(RC/'src')]
import run_rc_m_fine_c128_attribution_v1 as F

OUT = RC/'results/rc_m_fine_c128_extension_v1'
OLD = RC/'results/rc_roma_position_factor_v1'
AUTH = OUT/'authority.json'
MAIN = [0,1,2,4,5,6,7,8]
CHANGED = [9,21,32,41,44,55,82,84,110,116,118]
NEW = [41,44,55,82,84,110,116,118]
read, write, bind, checked = F.read, F.write, F.bind, F.checked


def prepare():
    import run_rc_roma_position_factor_v1 as G
    audit = RC/'results/rc_m_coarse_path_correction_audit_v1/result.json'
    ar = read(audit); assert ar['queries'] == 128
    rows = ar['rows']; selected = set(); cohorts = {}
    for model in F.MODELS:
        rescue = [r['index'] for r in rows if not r['raw_correct'] and r['models'][model]['ORIGINAL_HR1']['correct']]
        breaks = [r['index'] for r in rows if r['raw_correct'] and not r['models'][model]['ORIGINAL_HR1']['correct']]
        selected.update(rescue+breaks); cohorts[model] = dict(rescues=rescue, breaks=breaks)
    assert sorted(selected) == CHANGED
    bp = read(F.BRIDGE/'protocol.json'); items = {r['index']:r for r in read(checked(bp['inputs']))['rows']}
    old = read(OLD/'manifest.json'); workers = read(G.C.WORKERS)['records']
    manifest = dict(indices=NEW, workers=[w for w in workers if w['execution_ordinal'] in NEW],
        radius=old['radius'], old_manifest=bind(OLD/'manifest.json'))
    assert len(manifest['workers']) == 8
    write(OUT/'manifest.json', manifest)
    bindings = []
    for i in NEW:
        path = G.INSIDE/f'query{i:03d}/inside_manifest.json'
        m = read(path); assert len(m['sources']) == 128 and m['query_id'] == items[i]['query_id']
        bindings.append(bind(path))
    inherited = read(RC/'registry/rc_roma_position_factor_authority_v1_20260924.json')
    code = [Path(__file__), RC/'programs/submit_rc_m_fine_c128_extension_v1.py',
        RC/'slurm/rc_m_fine_c128_extension_v1.sbatch', RC/'plan/RC_M_FINE_C128_EXTENSION_V1_20260929.md',
        Path(G.__file__), Path(F.__file__)]
    code += [Path(b['path']) for b in inherited['sources'] + bp['code_sources']]
    a = dict(inherited, sources=[bind(x) for x in dict.fromkeys(code)],
        manifest=bind(OUT/'manifest.json'), query_count=8, budget_seconds=430,
        user_authorization='Expand attribution beyond three historical rescues, reuse existing data, submit and archive then exit.',
        scope='New8 complete natural C128 axes; same12 phase/amplitude worlds. No retraining; no fine refiner or ColNomic encoder.',
        note='Isolated reuse of validated collector; original authority and files remain unchanged.')
    write(AUTH,a); write(OUT/'preflight.json',dict(**G.P.self_test(),authority=bind(AUTH)))
    availability=read(OUT/'descriptor_availability.json')
    write(OUT/'protocol.json',dict(status='FINE_C128_EXTENSION_FROZEN', authority=bind(AUTH),
        original_audit=bind(audit), original_cohorts=cohorts, main=MAIN, changed=CHANGED, new=NEW,
        all_indices=sorted(MAIN+CHANGED), query_count=19, original_query_universe=128, candidate_count=128,
        phase_arms=F.PHASE_ARMS, bridge_protocol=bind(F.BRIDGE/'protocol.json'), label_source=bp['original_label_source'],
        items=[items[i] for i in NEW], inside_manifests=bindings,
        old_phase_validation=bind(F.OUT/'phase_validation.json'), old_phase_accounting=bind(F.OUT/'phase_accounting.json'),
        original_anchors={str(i):bind(F.BRIDGE/'queries'/f'{i:03d}'/'result.json') for i in NEW},
        existing_phase_results={str(i):bind(F.OUT/'phase'/f'query{i:03d}'/'result.json') for i in old['indices']},
        old_descriptor_receipts={k:str(Path(v).resolve()) if v else None for k,v in availability['images'].items()},
        scope='All original rescue/break events of three frozen models within the opened first128 H593 queries, plus the unchanged8 controls. Not all593, not independent confirmation, not an unbiased estimate of population gains.',
        intervention='Change only phase/amplitude of P at the coarse endpoint; mediate via M with all dependent scalar features updated and local content held fixed. Preserve HR1 and coarse native anchors separately.',
        retention='Save image descriptors once, complete J/P capture, confidence/weights, M, all128 logits, POST patch maxsim/argmax and source hashes.',
        existing_context_chain_untouched=True, new_training=0))
    print(json.dumps(dict(status='PREPARED',queries=19,new_queries=8,cohorts=cohorts)),flush=True)


def guard():
    p = read(OUT/'protocol.json'); checked(p['authority']); a=read(AUTH)
    for b in a['sources']+[a['manifest'],a['profile'],a['heads']]+p['inside_manifests']: checked(b)
    for k in ['original_audit','bridge_protocol','label_source','old_phase_validation','old_phase_accounting']: checked(p[k])
    return p,bind(OUT/'protocol.json')


def collect(ordinal,pilot):
    p,pb=guard()
    import run_rc_roma_position_factor_v1 as G
    G.OUT=OUT; G.AUTH=AUTH
    original_descriptor=G.descriptor
    def reuse_descriptor(model,core,item,stats):
        receipt=p['old_descriptor_receipts'].get(item['sha256'])
        if receipt:
            d=read(receipt)
            data=G.torch.load(checked(d['payload']),weights_only=True,mmap=True,map_location='cpu')
            assert data['profile']==bind(G.C.PROFILE)
            if 'image' in data: assert data['image']['sha256']==item['sha256']
            stats['historical_descriptor_reads']+=1
            return [t.to('cuda').clone() for t in data['features']],d['payload']
        return original_descriptor(model,core,item,stats)
    G.descriptor=reuse_descriptor
    return G.gpu(NEW[ordinal],pilot=pilot)


def replay(ordinal,budget):
    torch=F.configure(); p,pb=guard(); index=NEW[ordinal]
    item=next(r for r in p['items'] if r['index']==index)
    folder=OUT/'phase'/f'query{index:03d}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'validation.json').exists():
            v=read(folder/'validation.json'); assert v['protocol']==pb;checked(v['result']);return 0
        gv=read(OUT/f'query{index:03d}/gpu_validation.json')
        assert gv['status']=='POSITION_FACTOR_ALL128_PASS' and gv['authority']==p['authority']
        axis=item['post_row']['candidate_ids'];assert axis==gv['candidate_physical_rows']
        masses={a:[] for a in F.PHASE_ARMS};maximum=0.
        for pos,b in enumerate(gv['pairs']):
            pair=torch.load(checked(b),weights_only=True,mmap=True,map_location='cpu')
            assert pair['authority']==p['authority'] and (pair['index'],pair['position'],pair['physical_row'])==(index,pos,axis[pos])
            for arm in F.PHASE_ARMS:
                sides=pair['branches'][arm];m=F.scalar_mass(sides)
                other=float(np.sqrt(np.mean(sides[0]['weights'].numpy())*np.mean(sides[1]['weights'].numpy())))
                maximum=max(maximum,abs(m-other));assert abs(m-other)<2e-12
                masses[arm].append(m)
        import run_rc_m_causal128_frozen_bridge_v1 as B
        context=B.Context(read(checked(p['bridge_protocol'])));adapter,head,cache,refs=context.load(item)
        origin=read(checked(p['original_anchors'][str(index)]));assert head==origin['head']
        deadline=time.monotonic()+budget;values=[];patches=[]
        for pos in range(128):
            path=folder/'patch'/f'{pos:03d}.npz';ms=[masses[a][pos] for a in F.PHASE_ARMS]
            if path.exists():
                with np.load(path) as z:
                    assert z['worlds'].tolist()==F.PHASE_ARMS and np.array_equal(z['M'],ms)
                    v=z['maxsim'].copy();assert np.array_equal(v.mean(1),z['L'])
            else:
                if deadline-time.monotonic()<30:return 75
                v,loc=context.candidate(adapter,cache,refs[pos],ms)
                B.npz_once(path,worlds=np.asarray(F.PHASE_ARMS),M=np.asarray(ms),maxsim=v,argmax=loc,L=v.mean(1))
            values.append(v.mean(1));patches.append(bind(path))
        matrix=np.asarray(values).T;content={a:matrix[k].tolist() for k,a in enumerate(F.PHASE_ARMS)}
        worlds=F.all_scores(item,masses,content,head);worlds['ORIGINAL_HR1']=origin['worlds']['ORIGINAL_HR1']
        error=F.independent_scores(item,worlds,head)
        write(folder/'result.json',dict(protocol=pb,index=index,query_id=item['query_id'],cohort='ALL_CHANGED11',
            fold=item['fold'],component=item['component'],axis=axis,winner=item['post_row']['winner_index'],
            worlds=worlds,head=head,patches=patches,original_anchor=p['original_anchors'][str(index)],full_C128_predictions_computed=True))
        write(folder/'validation.json',dict(status='EXTENSION_FROZEN_SCORES_PASS',protocol=pb,result=bind(folder/'result.json'),
            maximum_mass_error=maximum,maximum_numpy_error=error,candidates=128,worlds=len(worlds)))
    return 0


def join():
    torch=F.configure();p,pb=guard();labels={r['query_id']:r for r in read(checked(p['label_source']))['rows']}
    bp=read(checked(p['bridge_protocol']));items={r['index']:r for r in read(checked(bp['inputs']))['rows']}
    rows=[];sources=[]
    for index in p['all_indices']:
        if index in NEW:
            v=read(OUT/'phase'/f'query{index:03d}'/'validation.json');assert v['protocol']==pb
            b=v['result']
        else:b=p['existing_phase_results'][str(index)]
        d=read(checked(b));sources.append(b);item=items[index];lab=labels[d['query_id']]
        F.independent_scores(item,d['worlds'],d['head'])
        for pos,patch in enumerate(d['patches']):
            with np.load(checked(patch)) as z:
                assert z['worlds'].tolist()==F.PHASE_ARMS
                assert np.array_equal(z['M'],[d['worlds'][a]['M'][pos] for a in F.PHASE_ARMS])
                assert np.array_equal(z['maxsim'].mean(1),[d['worlds'][a]['L'][pos] for a in F.PHASE_ARMS])
                assert np.all(z['argmax']>=0)
        targets=[k for k,x in enumerate(item['post_row']['candidate_identities']) if x==lab['identity']];assert len(targets)<=1
        row=dict(index=index,query_id=d['query_id'],original_query_id=lab['original_query_id'],component=item['component'],
            cohort='MAIN8' if index in MAIN else 'ALL_CHANGED11',target_present=bool(targets),raw_correct=d['winner'] in targets,
            correct={},prediction={},target_vs_strongest_wrong_margin={})
        for model in F.MODELS:
            row['correct'][model]={};row['prediction'][model]={};row['target_vs_strongest_wrong_margin'][model]={}
            for arm,world in d['worlds'].items():
                x=world['POST'] if model=='POST' else world['external'][model]
                row['correct'][model][arm]=x['prediction_position'] in targets
                row['prediction'][model][arm]=x['prediction_position']
                z=x['logits128'];row['target_vs_strongest_wrong_margin'][model][arm]=None if not targets else z[targets[0]]-max(z[k] for k in range(128) if k not in targets)
        rows.append(row)
    worlds=F.PHASE_ARMS+['ORIGINAL_HR1']
    summary={c:{m:{a:F.aggregate(rows,m,a,c) for a in worlds} for m in F.MODELS} for c in ['MAIN8','ALL_CHANGED11']}
    for m,expected in p['original_cohorts'].items():
        original=summary['ALL_CHANGED11'][m]['ORIGINAL_HR1']
        assert original['rescues']==expected['rescues'] and original['breaks']==expected['breaks']
    write(OUT/'result.json',dict(protocol=pb,status='ALL_ORIGINAL_EVENTS_ACCOUNTED',rows=rows,summary=summary,sources=sources,scope=p['scope'],
        limitation='Selected original events estimate their retention/loss, not population net gain. Fine interventions are not a unique decomposition; J/P removal and phase changes differ.'))
    write(OUT/'validation.json',dict(status='FINE_C128_EXTENSION_ACCOUNTING_PASS',protocol=pb,result=bind(OUT/'result.json'),queries=19,candidates=128,
        original_rescue_coverage={m:len(x['rescues']) for m,x in p['original_cohorts'].items()},complete_original_events=True))
    lines=['# Complete original rescue/break cohort: phase and amplitude attribution','',p['scope'],'',
        '| Model | Original rescues | Retained under P zero | Original breaks |','|---|---:|---:|---:|']
    for m in F.MODELS:
        s=summary['ALL_CHANGED11'][m];lines.append(f"| {m} | {len(s['ORIGINAL_HR1']['rescues'])} | {len(s['ZERO']['original_rescues_retained'])} | {len(s['ORIGINAL_HR1']['breaks'])} |")
    lines+=['','Full per-world decisions, rescue loss/recovery, breaks and margins are in result.json. Eight historical controls are reported separately. No new accuracy or unique-cause claim.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='FINE_C128_EXTENSION_ACCOUNTING_PASS',queries=19)),flush=True);return 0


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','pilot','collect','replay','join']);a.add_argument('--ordinal',type=int,default=0);a.add_argument('--budget',type=int,default=430);x=a.parse_args()
    if x.stage=='prepare':prepare();code=0
    elif x.stage in ['pilot','collect']:code=collect(x.ordinal,x.stage=='pilot')
    elif x.stage=='replay':code=replay(x.ordinal,x.budget)
    else:code=join()
    raise SystemExit(code)
