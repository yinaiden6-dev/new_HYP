#!/usr/bin/env python3
"""Independent sealed-output audit; no new model execution or parameter fitting."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

RC=Path(__file__).resolve().parents[1]
ROOT=RC/'results/rc_m_structure_binding_isolation_v2'


def read(p):return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve()
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return dict(path=str(p),sha256=h.hexdigest())
def checked(b):
    assert bind(b['path'])=={k:b[k] for k in ['path','sha256']}
    return Path(b['path'])


def main():
    torch.set_num_threads(4)
    p=read(ROOT/'protocol.json');pb=bind(ROOT/'protocol.json');d=read(checked(p['inputs']))
    output=read(ROOT/'result.json');validation=read(ROOT/'validation.json')
    checked(validation['result']);assert output['protocol']==pb
    for source in p['code_sources']+[p['parent_protocol']]:checked(source)
    maxima={k:0. for k in ['M','POST','gap','contrast','summary_mean','norm','mean','fixed_Gram','fixed_local_inner_product','fixed_sum_variance','conditional_invariant']}
    values={};audits={};native_controls=[];count=0
    for w in d['workers']:
        folder=ROOT/'pairs'/f"{w['ordinal']:03d}";v=read(folder/'validation.json')
        assert v['protocol']==pb and v['worker']==w and v['status']=='ISOLATION_PAIR_COMPLETE'
        assert [a['arm'] for a in v['arms']]==p['arms']
        with np.load(checked(v['post_patch'])) as z:
            assert z['worlds'].tolist()==p['arms']
            means=np.sum(z['maxsim'],axis=1)/z['maxsim'].shape[1]
            assert np.allclose(means,z['L'],rtol=0,atol=1e-14)
            maxima['POST']=max(maxima['POST'],float(np.max(abs(means-np.asarray(v['POST_L'])))))
        record={};audit_record={}
        for i,r in enumerate(v['arms']):
            payload=torch.load(checked(r['payload']),weights_only=True,map_location='cpu',mmap=True)
            assert payload['index']==w['index'] and payload['position']==w['position'] and payload['arm']==r['arm']
            assert payload['capture']==w['capture']
            if 'reused_from' in r:
                prior=read(checked(r['reused_from']));assert prior['payload']==r['payload'] and payload['protocol']==prior['protocol']
            else:assert payload['protocol']==pb
            sides=payload['sides']
            mass=math.sqrt(float(np.mean(sides[0]['weights'].numpy()))*float(np.mean(sides[1]['weights'].numpy())))
            assert 0<mass<=1
            maxima['M']=max(maxima['M'],abs(mass-r['M']),abs(mass-payload['M']))
            record[r['arm']]=dict(logM=math.log(mass),POST_content=float(means[i]))
            audit_record[r['arm']]=[s['input_audit'] for s in sides]
            for a in audit_record[r['arm']]:
                maxima['norm']=max(maxima['norm'],a['norm_relative_max'])
                maxima['mean']=max(maxima['mean'],a['mean_scaled_max'])
                arm=r['arm']
                if arm=='NATIVE' or any(arm.startswith(k) for k in ['Q_P_','Q_C_','Q_ALL_']):
                    maxima['fixed_Gram']=max(maxima['fixed_Gram'],a['selected64_P_Gram_relative_max'])
                if arm=='NATIVE' or any(arm.startswith(k) for k in ['S_P_','Q_ALL_']):
                    maxima['fixed_local_inner_product']=max(maxima['fixed_local_inner_product'],max(a['local_content_inner_product_scaled_max']))
                    maxima['fixed_sum_variance']=max(maxima['fixed_sum_variance'],a['sum_LN_variance_relative_max'])
                if a['conditional_joint_invariant_error'] is not None:
                    maxima['conditional_invariant']=max(maxima['conditional_invariant'],a['conditional_joint_invariant_error'])
            if r['arm']=='NATIVE' and sides[0].get('precision_native'):
                old=math.sqrt(float(np.mean(sides[0]['precision_native']['original_weights'].numpy()))*
                              float(np.mean(sides[1]['precision_native']['original_weights'].numpy())))
                native_controls.append(dict(ordinal=w['ordinal'],index=w['index'],position=w['position'],source_M=old,FP32_M=mass,
                    logM_difference=math.log(mass/old)))
            count+=1
        values[w['index'],w['position']]=record;audits[w['ordinal']]=audit_record
    assert len(values)==77 and count==1001 and len(d['cases'])==25
    by_case={q['index']:q for q in d['cases']}
    for row in output['rows']:
        q=by_case[row['index']]
        assert row['target']==q['target'] and row['wrong']==q['opponents'][row['opponent']]
        assert row['cohort']==q['cohort'] and row['component']==q['component']
        target=values[q['index'],q['target']];wrong=values[q['index'],row['wrong']]
        for endpoint,m in row['metrics'].items():
            g={a:target[a][endpoint]-wrong[a][endpoint] for a in p['arms']}
            for a in p['arms']:maxima['gap']=max(maxima['gap'],abs(g[a]-m['world_gaps'][a]))
            cc={name:[] for name in m['contrasts']}
            for sign in ['PLUS','MINUS']:
                n=g['NATIVE'];qp=g['Q_P_'+sign];qc=g['Q_C_'+sign];qa=g['Q_ALL_'+sign];s=g['S_P_'+sign]
                qs=g['Q_AFTER_S_'+sign];sq=g['S_AFTER_Q_'+sign]
                expected=[(n+qa-qp-qc)/2,s-n,qa-n,qs-s-qp+n,sq-qp-s+n,sq-qs]
                for name,x in zip(['binding_interaction','local_stats_preserved_effect','common_basis_effect',
                                   'B_after_S_interaction','S_after_B_interaction','order_effect'],expected):cc[name].append(x)
            for name,x in cc.items():maxima['contrast']=max(maxima['contrast'],abs(float(np.mean(x))-m['contrasts'][name]))
    for cohort,opponents in output['summaries'].items():
        for opponent,endpoints in opponents.items():
            rows=[r for r in output['rows'] if r['cohort']==cohort and r['opponent']==opponent]
            for endpoint,stats in endpoints.items():
                for name,s in stats.items():
                    groups={}
                    for r in rows:groups.setdefault(r['component'],[]).append(r['metrics'][endpoint]['contrasts'][name])
                    avg=float(np.mean([np.mean(x) for x in groups.values()]))
                    maxima['summary_mean']=max(maxima['summary_mean'],abs(avg-s['group_equal_mean']))
                    assert len(rows)==s['queries'] and len(groups)==s['groups']
    assert all(maxima[k]<2e-12 for k in ['M','POST','gap','contrast','summary_mean'])
    assert all(maxima[k]<3e-6 for k in ['norm','mean','fixed_Gram','fixed_local_inner_product','conditional_invariant'])
    assert maxima['fixed_sum_variance']<5e-6 and len(native_controls)==14
    precision_lookup={x['ordinal']:x for x in output['native_precision_controls']}
    for c in native_controls:assert abs(c['logM_difference']-precision_lookup[c['ordinal']]['logM_difference'])<1e-12
    result=dict(status='INDEPENDENT_77PAIR_1001WORLD_LINEAGE_AND_ENDPOINT_AUDIT_PASS',pairs=77,worlds=1001,queries=25,
        reused_pairs=len(p['reused_pair_ordinals']),recomputed_pairs=len(p['recompute_pair_ordinals']),max_errors=maxima,
        native_precision_controls=native_controls,result=bind(ROOT/'result.json'),auditor=bind(Path(__file__)),
        boundary='Recounts saved endpoints and checks stored invariant receipts and lineage; does not rerun the encoder, DPT, or POST and does not identify unique causal shares.')
    (ROOT/'independent_audit_20261002.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (ROOT/'invariant_audit_table_20261002.json').write_text(json.dumps(audits,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='native_precision_controls'}),flush=True)


if __name__=='__main__':main()
