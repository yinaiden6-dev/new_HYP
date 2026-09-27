#!/usr/bin/env python3
"""Post-hoc, cache-only common-level versus relative-gap intervention.

No training, RoMa, vision, language-model or GPU forward. Replays frozen POST
adapter/projector on qualified hidden states. Log coordinates are primary;
log-odds coordinates are a separately named domain-complete sensitivity check.
"""
from __future__ import annotations
import argparse
import fcntl
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

RC=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RC/'programs'),str(RC/'src')]
import run_rc_m_causal128_frozen_bridge_v1 as B
import run_rc_m_causal128_newgroup_property_bridge_v1 as N
BASE=RC/'results/rc_m_causal128_attribution_chain_v1'
DEFAULT=RC/'results/rc_post_m_level_gap_v1'
CELLS=('LL','LG','GL','GG')
COORDINATES=('log_M','log_odds_M')
EDGES=(('P_with_J','A1J1P0','A1J1P1'),('P_without_J','A1J0P0','A1J0P1'),
       ('J_with_P','A1J0P1','A1J1P1'),('J_without_P','A1J0P0','A1J1P0'))
EFFECTS=('delta_at_mu_L','delta_at_mu_G','mu_at_delta_L','mu_at_delta_G',
         'interaction','total','symmetric_delta','symmetric_mu')
read,bind,checked,write=B.read,B.bind,B.checked,B.write
TOL=2e-10


def transform(values,coordinate):
    x=np.asarray(values,dtype=np.float64)
    assert np.all(np.isfinite(x)) and np.all((x>0)&(x<1)),('SOURCE_M_DOMAIN',x.tolist())
    return np.log(x) if coordinate=='log_M' else np.log(x)-np.log1p(-x)


def invert(values,coordinate):
    x=np.asarray(values,dtype=np.float64)
    if coordinate=='log_M':return np.exp(x)
    # Stable logistic, without clipping or changing valid values.
    return np.exp(-np.logaddexp(0.,-x))


def cross_values(low,high,coordinate):
    logs=[transform(low,coordinate),transform(high,coordinate)]
    mu=[float(z.mean()) for z in logs];delta=[float(z[0]-z[1]) for z in logs];cells={}
    for cell in CELLS:
        a=int(cell[0]=='G');b=int(cell[1]=='G')
        ms=invert([mu[a]+delta[b]/2,mu[a]-delta[b]/2],coordinate)
        if cell=='LL':assert np.max(np.abs(ms-low))<2e-14;ms=np.asarray(low,dtype=np.float64)
        if cell=='GG':assert np.max(np.abs(ms-high))<2e-14;ms=np.asarray(high,dtype=np.float64)
        cells[cell]=dict(mu=mu[a],delta=delta[b],M=ms.tolist(),valid=bool(np.isfinite(ms).all() and ((ms>0)&(ms<=1)).all()))
    return cells


def effects(values):
    ll,lg,gl,gg=[float(values[k]) for k in CELLS]
    d0,d1,m0,m1=lg-ll,gg-gl,gl-ll,gg-lg
    out=dict(delta_at_mu_L=d0,delta_at_mu_G=d1,mu_at_delta_L=m0,mu_at_delta_G=m1,
        interaction=gg-gl-lg+ll,total=gg-ll,symmetric_delta=(d0+d1)/2,symmetric_mu=(m0+m1)/2)
    assert max(abs((d0+m1)-out['total']),abs((m0+d1)-out['total']),
        abs(out['symmetric_delta']+out['symmetric_mu']-out['total']),
        abs(d1-d0-out['interaction']),abs(m1-m0-out['interaction']))<1e-10
    return out


def paths_for_item(item):
    row=item['post_row'];vp=Path(row['hidden_validation']['path']) if row.get('hidden_validation') else Path(row['hidden_cache_dir'])/'validation.json'
    v=read(vp);assert 'PASS' in v['status'];payload=v['payload'];assert Path(payload['path']).is_file()
    return dict(validation=bind(vp),payload=payload)


def preparation(root):
    bridge=BASE/'bridge';phase=BASE/'newgroup_property_bridge_v1';bp=read(bridge/'protocol.json')
    bv=read(bridge/'validation.json');nv=read(phase/'validation.json')
    assert bv['status']=='CAUSAL128_FROZEN_BRIDGE_INDEPENDENT_ARITHMETIC_PASS'
    assert nv['status']=='NEWGROUP_PROPERTY_BRIDGE_INDEPENDENT_PASS'
    rows=read(checked(bp['inputs']))['rows'];joined=read(checked(bv['result']));phase_rows={r['index']:r for r in read(checked(nv['result']))['rows']}
    workers=[];domain=[];domain_all=[];timings=[]
    for selection in joined['rows']:
        if not selection['target_present']:continue
        idx=selection['index'];item=rows[idx];target=selection['target_position'];wrong=selection['fixed_wrong_position']
        assert item['index']==idx and selection['query_id']==item['query_id'] and selection['fold']==item['fold']
        assert target!=wrong and item['post_row']['candidate_identities'][target]!=item['post_row']['candidate_identities'][wrong]
        query=read(bridge/'queries'/f'{idx:03d}'/'result.json');assert query['query_id']==item['query_id']
        sources={};bindings=[]
        for pos in (target,wrong):
            receipt=read(bridge/'queries'/f'{idx:03d}'/'candidates'/f'{pos:03d}.json')
            assert receipt['position']==pos and receipt['candidate_id']==item['post_row']['candidate_ids'][pos]
            bindings.append(receipt['patch_evidence']);timings.append(receipt['seconds'])
            with np.load(checked(receipt['patch_evidence'])) as z:
                for j,world in enumerate(z['worlds']):
                    sources.setdefault(str(world),{})[str(pos)]=dict(M=float(z['M'][j]),patch=receipt['patch_evidence'],row=j,L=float(z['L'][j]))
                    assert abs(z['M'][j]-item['masses'][str(world)][pos])<1e-14
        phase_record=None
        if idx in phase_rows:
            pr=phase_rows[idx];assert pr['target_position']==target and pr['fixed_wrong_position']==wrong
            assert pr['candidate_physical_rows']==item['post_row']['candidate_ids']
            phase_record=bind(phase/f'query{idx:03d}/result.json')
            for pos,b in zip(sorted([target,wrong]),pr['patch_outputs']):
                bindings.append(b)
                with np.load(checked(b)) as z:
                    for j,world in enumerate(z['worlds']):
                        sources.setdefault('PHASE_'+str(world),{})[str(pos)]=dict(M=float(z['M'][j]),patch=b,row=j,L=float(z['L'][j]))
        comparisons=[]
        definitions=[('jp128',name,None,lo,hi) for name,lo,hi in EDGES]
        if phase_record:
            for pfx,bg in (('','native_amplitude'),('AMP_PERMUTE__','permuted_amplitude')):
                for axis in ('X','Y'):
                    for sign in ('PLUS','MINUS'):
                        definitions.append(('phase9',bg,axis+'_'+sign,'PHASE_'+pfx+'LOCAL_'+axis+'_'+sign,'PHASE_'+pfx+'GLOBAL_'+axis+'_'+sign))
        for panel,name,rep,lo,hi in definitions:
            low=[sources[lo][str(pos)]['M'] for pos in (target,wrong)];high=[sources[hi][str(pos)]['M'] for pos in (target,wrong)]
            for coordinate in COORDINATES:
                cells=cross_values(low,high,coordinate)
                comp=dict(panel=panel,contrast=name,replicate=rep,coordinate=coordinate,low=lo,high=hi,cells=cells,
                    complete_legal_grid=all(c['valid'] for c in cells.values()))
                comparisons.append(comp)
                for c,value in cells.items():
                    rec=dict(index=idx,query_id=item['query_id'],panel=panel,contrast=name,replicate=rep,coordinate=coordinate,cell=c,**value)
                    domain_all.append(rec)
                    if not value['valid']:domain.append(rec)
        hidden=paths_for_item(item)
        refbindings=[]
        for pos in (target,wrong):
            ref=item['post_row']['reference_tokens'][pos];ref=ref.get('token_file',ref);assert Path(ref['path']).is_file();refbindings.append(ref)
        workers.append(dict(index=idx,query_id=item['query_id'],component=item['component'],fold=item['fold'],
            target_position=target,fixed_wrong_position=wrong,RAW_winner=item['post_row']['winner_index'],axis=item['post_row']['candidate_ids'],
            hidden=hidden,references=refbindings,original_POST=query['original_anchor']['source'],bridge_row=bind(bridge/'queries'/f'{idx:03d}/result.json'),
            phase_row=phase_record,sources=sources,comparisons=comparisons))
    assert len(workers)==120 and len({w['component'] for w in workers})==45
    assert len(phase_rows)==9 and sum(w['phase_row'] is not None for w in workers)==9
    assert len(domain)==21 and all(d['panel']=='jp128' and d['coordinate']=='log_M' and d['contrast']=='J_with_P' and d['cell']=='LG' for d in domain)
    root.mkdir(parents=True,exist_ok=True)
    write(root/'workers.json',dict(workers=workers),immutable=True)
    write(root/'domain_audit.json',dict(status='LEVEL_GAP_DOMAIN_COMPLETE',invalid_cells=domain,total_candidate_values=len(domain_all)*2,
        minimum=float(min(min(r['M']) for r in domain_all)),maximum=float(max(max(r['M']) for r in domain_all)),
        cells=domain_all,clipped=0,source_endpoints_strictly_between_zero_and_one=True),immutable=True)
    protocol=dict(status='POST_M_LEVEL_GAP_PROTOCOL_FROZEN',version=1,workers=bind(root/'workers.json'),domain=bind(root/'domain_audit.json'),
        parent_bridge_protocol=bind(bridge/'protocol.json'),parent_bridge_validation=bind(bridge/'validation.json'),
        phase_bridge_validation=bind(phase/'validation.json'),inputs=bp['inputs'],authority=bp['authority'],manifest=bp['manifest'],endpoints=bp['endpoints'],
        source_code=bp['code_sources']+[bind(Path(B.__file__)),bind(Path(N.__file__)),bind(Path(__file__))],
        coordinates=list(COORDINATES),cells=list(CELLS),effects=list(EFFECTS),edges=[list(e) for e in EDGES],
        data_scope='All 120 target-present of the existing 128-query panel, 45 components; 9 phase queries/9 groups also reported separately. Not old EVAL128, not untouched external data.',
        motivation='Post-hoc mechanism diagnosis after observing opposite logM-gap and POST-content-gap effects in native-amplitude phase restoration.',
        domain_policy='No clipping. Primary log_M J_with_P: retain all 120 endpoints and explicitly mark 21 incomplete grids; full decomposition only in a domain-limited 99-query appendix. Other three log_M edges all 120. Pre-frozen log_odds_M sensitivity: all four edges/all120 and phase9; always report both coordinates, never select by outcomes.',
        anchor_policy='Only target/fixed-wrong M and POST content change. If RAW winner is a third candidate, hold original HR1 M/L fixed; if in pair, update its values. HOLD=0. Thus full128 LL/GG candidate M and content replay old sources, but action gaps are recomputed under this new fixed-third-anchor protocol and do not purport to replay previous all-world actions.',
        endpoint_replay='LL/GG and all existing source endpoints are actually forwarded through the original-fold frozen POST and must reproduce old maxsim, argmax and mean content; inherited mass floats are used exactly at endpoints.',
        attribution_boundary='Exact finite computational decomposition at the M interface, conditional on frozen content/head/query and chosen coordinates. Symmetric contributions are not a unique causal allocation or necessarily realizable upstream image interventions.',
        supervision='None; fixed diagnostic target/wrong selection inherited unchanged. No prediction or full-C128 accuracy reported.',
        threads=4,budget_seconds=430,recommended_shards=8,query_count=120,phase_query_count=9,new_training=0,new_gpu_forwards=0,
        new_RoMa_vision_LLM_forwards=0,CPU_POST_adapter_projection_only=True,maximum_error=TOL,
        expected_unique_candidate_mass_forwards_upper_bound=6336,cached_five_mass_candidate_median_seconds=float(np.median(timings)),
        expected_total_CPU_worker_wall_seconds_range=[300,1500],runtime_estimate_not_measurement=True)
    write(root/'protocol.json',protocol,immutable=True)
    write(root/'preparation_validation.json',dict(status='POST_M_LEVEL_GAP_PREPARATION_PASS',protocol=bind(root/'protocol.json'),
        queries=120,groups=45,phase_queries=9,invalid_log_crosses=21,invalid_log_odds_crosses=0,new_forwards=0),immutable=True)
    print(__import__('json').dumps(dict(status=protocol['status'],protocol=bind(root/'protocol.json'),queries=120,invalid_log_crosses=21)),flush=True)
    return 0


def guard(root):
    p=read(root/'protocol.json');assert p['status']=='POST_M_LEVEL_GAP_PROTOCOL_FROZEN'
    for b in p['source_code']+[p['workers'],p['domain'],p['parent_bridge_protocol'],p['parent_bridge_validation'],
                p['phase_bridge_validation'],p['inputs'],p['authority'],p['manifest']]:checked(b)
    assert p['coordinates']==list(COORDINATES) and p['edges']==[list(e) for e in EDGES]
    for e in p['endpoints'].values():checked(e['snapshot'])
    return p,bind(root/'protocol.json'),read(p['workers']['path'])['workers']


def get_scores(item,head,original,target,wrong,masses,content):
    ms=np.asarray(item['masses']['ORIGINAL_HR1'],dtype=np.float64).copy();ls=np.asarray(original,dtype=np.float64).copy()
    for j,pos in enumerate((target,wrong)):ms[pos]=masses[j];ls[pos]=content[j]
    gaps,sides=N.score_metrics(item,ms,ls,head,target,wrong)
    independent=B.independent_gap(item,ms,ls,head,target,wrong)
    maximum=max(abs(gaps[k]-independent[k]) for k in N.GAP_METRICS)
    winner=item['post_row']['winner_index']
    for role,pos in [('target',target),('fixed_wrong',wrong)]:
        independent=B.independent_gap(item,ms,ls,head,pos,winner)
        for key in ('POST_action','NATIVE7_action','M_FREE_action'):
            maximum=max(maximum,abs(sides[role][key]-independent[key+'_gap']))
        if pos==winner:assert all(sides[role][key]==0 for key in ('POST_action','NATIVE7_action','M_FREE_action'))
    assert maximum<TOL
    return dict(gaps=gaps,sides=sides),maximum


def read_patch(binding):
    with np.load(checked(binding)) as z:return {k:z[k].copy() for k in z.files}


def run_worker(root,shard,shards,budget,index=None):
    p,pb,workers=guard(root);items={r['index']:r for r in read(p['inputs']['path'])['rows']}
    selection=[w for w in workers if w['index']==index] if index is not None else workers[shard::shards]
    assert selection and 0<=shard<shards
    started=time.monotonic();deadline=started+budget;context=None
    for w in selection:
        folder=root/'queries'/f"{w['index']:03d}";folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (folder/'validation.json').exists():
                v=read(folder/'validation.json');assert v['protocol']==pb;checked(v['result']);continue
            if time.monotonic()>deadline-20:return 75
            if context is None:context=B.Context(read(p['parent_bridge_protocol']['path']))
            item=items[w['index']];assert item['post_row']['candidate_ids']==w['axis'] and item['query_id']==w['query_id']
            for b in [w['bridge_row'],w['hidden']['validation'],w['hidden']['payload'],*w['references'],w['original_POST']]:checked(b)
            if w['phase_row']:checked(w['phase_row'])
            adapter,head,cache,refs=context.load(item);positions=(w['target_position'],w['fixed_wrong_position']);patches=[];bypos={};replay_error=0.
            for j,pos in enumerate(positions):
                # Deduplicate solely exact mass bit patterns, never by scores.
                masses={};source_checks={}
                for comp in w['comparisons']:
                    for cell,values in comp['cells'].items():
                        if values['valid']:masses[float(values['M'][j]).hex()]=float(values['M'][j])
                for arm,source in w['sources'].items():
                    src=source[str(pos)];key=float(src['M']).hex();masses[key]=src['M'];source_checks.setdefault(key,[]).append(dict(arm=arm,**src))
                keys=list(masses);values=[];locations=[];receipts=[]
                for k,key in enumerate(keys):
                    path=folder/f'p{pos:03d}'/f'{k:03d}.npz';receipt=path.with_suffix('.json')
                    if receipt.exists():
                        rec=read(receipt);assert rec['protocol']==pb and rec['mass_hex']==key and rec['position']==pos
                        patch=read_patch(rec['patch']);assert patch['M'][0]==masses[key]
                        best,where=patch['maxsim'][0],patch['argmax'][0]
                    else:
                        if time.monotonic()>deadline-15:return 75
                        tick=time.monotonic();vv,ii=context.candidate(adapter,cache,refs[pos],[masses[key]])
                        best,where=vv[0],ii[0]
                        if path.exists():
                            old=read_patch(bind(path));assert old['M'][0]==masses[key] and np.array_equal(old['maxsim'],vv) and np.array_equal(old['argmax'],ii)
                        else:B.npz_once(path,M=np.asarray([masses[key]]),maxsim=vv,argmax=ii,L=vv.mean(1),physical_row=np.asarray([w['axis'][pos]]))
                        rec=dict(status='LEVEL_GAP_CANDIDATE_VALUE_COMPLETE',protocol=pb,index=w['index'],position=pos,
                            physical_row=w['axis'][pos],mass_hex=key,patch=bind(path),seconds=time.monotonic()-tick)
                        write(receipt,rec,immutable=True)
                    for src in source_checks.get(key,[]):
                        old=read_patch(src['patch']);jj=src['row'];assert old['M'][jj]==masses[key]
                        error=float(np.max(np.abs(best-old['maxsim'][jj])));replay_error=max(replay_error,error)
                        assert error<TOL and np.array_equal(where,old['argmax'][jj]),('ENDPOINT_PATCH_REPLAY',w['index'],pos,src['arm'],error)
                    values.append(best);locations.append(where);receipts.append(bind(receipt))
                aggregate=folder/f'pair{pos:03d}.npz';vv=np.stack(values);ii=np.stack(locations)
                if aggregate.exists():
                    old=read_patch(bind(aggregate));assert np.array_equal(old['M'],list(masses.values())) and np.array_equal(old['maxsim'],vv) and np.array_equal(old['argmax'],ii)
                else:B.npz_once(aggregate,mass_hex=np.asarray(keys),M=np.asarray(list(masses.values())),maxsim=vv,argmax=ii,L=vv.mean(1),physical_row=np.asarray([w['axis'][pos]]))
                patches.append(dict(position=pos,patch=bind(aggregate),receipts=receipts));bypos[pos]={key:float(v.mean()) for key,v in zip(keys,values)}
            original=read(w['original_POST']['path'])['arms']['NATIVE']['L'];comparisons=[];maximum=0.
            for comp in w['comparisons']:
                cs={}
                for cell,value in comp['cells'].items():
                    if not value['valid']:cs[cell]=dict(**value,status='OUTSIDE_M_DOMAIN_NOT_FORWARDED');continue
                    content=[bypos[pos][float(value['M'][j]).hex()] for j,pos in enumerate(positions)]
                    metrics,err=get_scores(item,head,original,*positions,value['M'],content);maximum=max(maximum,err)
                    cs[cell]=dict(**value,**metrics,status='VALID')
                result=dict(**{k:v for k,v in comp.items() if k!='cells'},cells=cs,effects={})
                if comp['complete_legal_grid']:
                    for key in N.GAP_METRICS:result['effects'][key]=effects({c:cs[c]['gaps'][key] for c in CELLS})
                    result['side_effects']={role:{key:effects({c:cs[c]['sides'][role][key] for c in CELLS}) for key in N.SIDE_METRICS} for role in ('target','fixed_wrong')}
                comparisons.append(result)
            row=dict(status='POST_M_LEVEL_GAP_QUERY_COMPLETE',protocol=pb,index=w['index'],query_id=w['query_id'],component=w['component'],fold=w['fold'],
                target_position=positions[0],fixed_wrong_position=positions[1],RAW_winner=w['RAW_winner'],candidate_axis=w['axis'],head=head,
                snapshot=p['endpoints'][str(w['fold'])]['snapshot'],patch_outputs=patches,comparisons=comparisons,
                endpoint_patch_replay_max_error=replay_error,independent_numpy_action_max_error=maximum,full_C128_predictions_computed=False)
            write(folder/'result.json',row,immutable=True)
            validate_query(root,w,p,pb,items,row)
            print(__import__('json').dumps(dict(status='POST_M_LEVEL_GAP_QUERY_PASS',index=w['index'],comparisons=len(comparisons),replay_error=replay_error)),flush=True)
    return 0


def validate_query(root,w,p,pb,items,row=None):
    folder=root/'queries'/f"{w['index']:03d}";row=read(folder/'result.json') if row is None else row
    assert row['protocol']==pb and row['index']==w['index'] and row['candidate_axis']==w['axis'] and row['snapshot']==p['endpoints'][str(w['fold'])]['snapshot']
    assert row['target_position']==w['target_position'] and row['fixed_wrong_position']==w['fixed_wrong_position'] and row['component']==w['component']
    item=items[w['index']];original=read(checked(w['original_POST']))['arms']['NATIVE']['L'];data={};replay=0.;error=0.
    assert len(row['patch_outputs'])==2
    for entry,pos in zip(row['patch_outputs'],(w['target_position'],w['fixed_wrong_position'])):
        assert entry['position']==pos
        z=read_patch(entry['patch']);assert z['physical_row'][0]==w['axis'][pos]
        assert np.max(np.abs(z['L']-z['maxsim'].mean(1)))<1e-14
        data[pos]={str(key):dict(M=float(m),L=float(v.mean()),maxsim=v,argmax=i) for key,m,v,i in zip(z['mass_hex'],z['M'],z['maxsim'],z['argmax'])}
        for source in w['sources'].values():
            src=source[str(pos)];old=read_patch(src['patch']);k=src['row'];new=data[pos][float(src['M']).hex()]
            err=float(np.max(np.abs(new['maxsim']-old['maxsim'][k])));replay=max(replay,err)
            assert err<TOL and np.array_equal(new['argmax'],old['argmax'][k])
    assert len(row['comparisons'])==len(w['comparisons'])
    for out,design in zip(row['comparisons'],w['comparisons']):
        for k in ('panel','contrast','coordinate','replicate','low','high','complete_legal_grid'):assert out[k]==design[k]
        positions=(w['target_position'],w['fixed_wrong_position']);low=[w['sources'][design['low']][str(pos)]['M'] for pos in positions];high=[w['sources'][design['high']][str(pos)]['M'] for pos in positions]
        for cell,expected in cross_values(low,high,design['coordinate']).items():
            actual=out['cells'][cell];assert actual['M']==expected['M'] and actual['valid']==expected['valid']
            if not expected['valid']:assert actual['status']=='OUTSIDE_M_DOMAIN_NOT_FORWARDED';continue
            # Scalar math independently reconstructs chosen coordinates, without numpy transforms.
            a=int(cell[0]=='G');b=int(cell[1]=='G');zz=[]
            for vals in (low,high):zz.append([math.log(m) if design['coordinate']=='log_M' else math.log(m/(1-m)) for m in vals])
            mu=sum(zz[a])/2;delta=zz[b][0]-zz[b][1]
            assert abs(mu-actual['mu'])<1e-12 and abs(delta-actual['delta'])<1e-12
            ls=[data[pos][float(expected['M'][j]).hex()]['L'] for j,pos in enumerate(positions)]
            metrics,err=get_scores(item,row['head'],original,*positions,expected['M'],ls);error=max(error,err)
            for key in N.GAP_METRICS:error=max(error,abs(metrics['gaps'][key]-actual['gaps'][key]))
            for role in ('target','fixed_wrong'):
                for key in N.SIDE_METRICS:error=max(error,abs(metrics['sides'][role][key]-actual['sides'][role][key]))
        if design['complete_legal_grid']:
            for key in N.GAP_METRICS:
                v={c:out['cells'][c]['gaps'][key] for c in CELLS};expected=effects(v)
                for k in EFFECTS:error=max(error,abs(out['effects'][key][k]-expected[k]))
                # Separate matrix multiplication verifies eight finite contrasts.
                x=np.asarray([v[c] for c in CELLS]);mat=np.asarray([[-1,1,0,0],[0,0,-1,1],[-1,0,1,0],[0,-1,0,1],[1,-1,-1,1],[-1,0,0,1],[-.5,.5,-.5,.5],[-.5,-.5,.5,.5]])
                error=max(error,float(np.max(np.abs(mat@x-np.asarray([out['effects'][key][k] for k in EFFECTS])))))
            for role in ('target','fixed_wrong'):
                for key in N.SIDE_METRICS:
                    ex=effects({c:out['cells'][c]['sides'][role][key] for c in CELLS})
                    error=max(error,max(abs(out['side_effects'][role][key][k]-ex[k]) for k in EFFECTS))
        else:assert not out['effects']
    assert error<TOL and replay<TOL
    cp=torch.load(checked(row['snapshot']),map_location='cpu',weights_only=True)
    assert cp['arm']=='POST_REAL' and cp['head'].double().tolist()==row['head']
    val=dict(status='POST_M_LEVEL_GAP_QUERY_INDEPENDENT_PASS',protocol=pb,result=bind(folder/'result.json'),
        index=w['index'],independent_numpy_max_error=error,endpoint_patch_replay_max_error=replay,
        invalid_crosses_explicit=True,raw_HOLD_zero=True,new_gpu_forwards=0)
    write(folder/'validation.json',val,immutable=True)
    return val


def aggregate_summary(rows):
    summary={};coverage={}
    for row in rows:
        for comp in row['comparisons']:
            scope='/'.join([comp['panel'],comp['coordinate'],comp['contrast']])
            coverage.setdefault(scope,dict(queries=set(),valid_queries=set(),invalid_queries=set(),components=set()))
            cov=coverage[scope];cov['queries'].add(row['index']);cov['components'].add(row['component'])
            cov['valid_queries' if comp['complete_legal_grid'] else 'invalid_queries'].add(row['index'])
            if not comp['complete_legal_grid']:continue
            for key,vals in comp['effects'].items():
                summary.setdefault(scope,{}).setdefault(key,[]).append(dict(component=row['component'],**vals))
            for role,metrics in comp['side_effects'].items():
                for key,vals in metrics.items():summary.setdefault(scope,{}).setdefault(role+'/'+key,[]).append(dict(component=row['component'],**vals))
    for scope,metrics in summary.items():
        for key,entries in metrics.items():summary[scope][key]={effect:B.group_summary(entries,effect) for effect in EFFECTS}
    coverage={scope:{k:sorted(v) for k,v in cov.items()} for scope,cov in coverage.items()}
    return summary,coverage


def join(root):
    p,pb,workers=guard(root);items={r['index']:r for r in read(p['inputs']['path'])['rows']};rows=[];vals=[]
    missing=[w['index'] for w in workers if not (root/'queries'/f"{w['index']:03d}"/'validation.json').exists()]
    if missing:print(__import__('json').dumps(dict(status='WAITING',remaining=missing)),flush=True);return 75
    for w in workers:
        v=validate_query(root,w,p,pb,items);rows.append(read(v['result']['path']));vals.append(bind(root/'queries'/f"{w['index']:03d}"/'validation.json'))
    summary,coverage=aggregate_summary(rows)
    assert len(rows)==120 and len(summary)==12
    for scope,c in coverage.items():
        if scope.startswith('phase9/'):assert len(c['queries'])==len(c['valid_queries'])==9
        else:
            assert len(c['queries'])==120
            if scope=='jp128/log_M/J_with_P':assert len(c['valid_queries'])==99 and len(c['invalid_queries'])==21
            else:assert len(c['valid_queries'])==120 and not c['invalid_queries']
    result=dict(status='POST_M_LEVEL_GAP_COMPLETE',protocol=pb,summary=summary,coverage=coverage,rows=rows,
        inference='Post-hoc coordinate-dependent computational attribution; no unique causal decomposition, prediction accuracy, model improvement or external replication claim.',
        new_gpu_forwards=0,new_RoMa_vision_LLM_forwards=0,training=False)
    write(root/'result.json',result,immutable=True)
    lines=['# Frozen POST: common M level versus relative candidate gap','',p['motivation'],'',p['data_scope'],'',p['domain_policy'],'',p['anchor_policy'],'',p['attribution_boundary'],'',
        'The log-odds sensitivity was fixed after checking only numerical domains and before observing any new cross-cell POST result. Both coordinate systems are reported.','',
        '| Scope | Endpoint | Total GG−LL | Symmetric common-level contribution | Symmetric relative-gap contribution | Interaction |',
        '|---|---|---|---|---|---|']
    def fmt(v):return f"{v['mean']:+.8f}; CI={v['exploratory_bootstrap95']}; {v['positive_groups']}/{v['groups']} positive groups"
    for scope in sorted(summary):
        for metric in ('logM_gap','POST_content_gap','POST_action_gap','NATIVE7_action_gap','M_FREE_action_gap'):
            s=summary[scope][metric];label=scope+(' [99-query legal-domain appendix only]' if scope=='jp128/log_M/J_with_P' else '')
            lines.append('| '+label+' | '+metric+' | '+' | '.join(fmt(s[k]) for k in ('total','symmetric_mu','symmetric_delta','interaction'))+' |')
    lines+=['','All four conditional effects, both candidate-specific effects, per-axis records, exact inputs, patch MaxSim/argmax and all finite contrasts are retained in result.json and query NPZ files.',
        'Group bootstrap intervals are exploratory. Four axis/sign repetitions are averaged within query/group and are not four independent samples. The phase9 and jp128 panels overlap and are never pooled.',
        'Out-of-domain log-M cells are recorded but never forwarded, clipped, replaced or silently dropped. J_with_P endpoints for all120 remain available. Its99-query decomposition is domain conditional, not an all120 result.','']
    text='\n'.join(lines);rp=root/'report.md'
    if rp.exists():assert rp.read_text()==text
    else:rp.write_text(text)
    write(root/'validation.json',dict(status='POST_M_LEVEL_GAP_INDEPENDENT_PASS',protocol=pb,result=bind(root/'result.json'),report=bind(rp),
        query_validations=vals,queries=120,groups=45,phase_queries=9,invalid_log_cells=21,all_log_odds_cells_legal=True,
        endpoint_patch_replay_max_error=max(r['endpoint_patch_replay_max_error'] for r in rows),
        independent_numpy_max_error=max(read(v['path'])['independent_numpy_max_error'] for v in vals),new_gpu_forwards=0),immutable=True)
    print(__import__('json').dumps(dict(status='POST_M_LEVEL_GAP_INDEPENDENT_PASS',queries=120,groups=45)),flush=True)
    return 0


def self_test():
    for coordinate in COORDINATES:
        lo=[.12,.21];hi=[.07,.05];cells=cross_values(lo,hi,coordinate)
        assert cells['LL']['M']==lo and cells['GG']['M']==hi
        for cell,record in cells.items():
            z=transform(record['M'],coordinate)
            assert abs(z.mean()-record['mu'])<1e-13 and abs(z[0]-z[1]-record['delta'])<1e-13
        values={c:record['mu']**2+record['delta']+record['mu']*record['delta'] for c,record in cells.items()}
        e=effects(values);assert abs(e['symmetric_mu']+e['symmetric_delta']-e['total'])<1e-12
    assert not cross_values([.8,.8],[.9,.01],'log_M')['LG']['valid']
    assert all(c['valid'] for c in cross_values([.8,.8],[.9,.01],'log_odds_M').values())
    e=effects(dict(LL=0,LG=2,GL=3,GG=11));assert e['interaction']==6 and e['symmetric_delta']==5 and e['symmetric_mu']==6
    print('{"status":"POST_M_LEVEL_GAP_SYNTHETIC_SELF_TEST_PASS","new_forwards":0}',flush=True)
    return 0


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','worker','join','self-test']);ap.add_argument('--root',type=Path,default=DEFAULT)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=8);ap.add_argument('--index',type=int);ap.add_argument('--budget',type=float,default=430)
    args=ap.parse_args();B.configure()
    if args.stage=='prepare':return preparation(args.root)
    if args.stage=='worker':return run_worker(args.root,args.shard,args.shards,args.budget,args.index)
    if args.stage=='join':return join(args.root)
    return self_test()

if __name__=='__main__':raise SystemExit(main())
