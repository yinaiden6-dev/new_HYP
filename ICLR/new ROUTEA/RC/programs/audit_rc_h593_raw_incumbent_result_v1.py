#!/usr/bin/env python3
"""Post-join independent recount and frozen-coordinate diagnostics; no fits."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'programs'))
from audit_rc_h593_gap_net2_result_v1 import read, binding, checked, comparison, changes

OUT = ROOT/'results/rc_h593_raw_incumbent_gate_v1'
AUTH = ROOT/'registry/rc_h593_raw_incumbent_gate_authority_v1_20260921.json'
ARMS = ('GAP_RAWPAIR3', 'GAP_RAWNEAR3')
BASE, OLD = 'COST1_FULL', 'GAP_BIAS2'


def score(row, head, name):
    raw = row['r_pair'] if name == ARMS[0] else row['r_near']
    return ((row['m']+head['gamma']*raw)+head['beta']*row['d'])+head['bias']


def main():
    a, result, val = read(AUTH), read(OUT/'result.json'), read(OUT/'validation.json')
    assert result['status'] == 'H593_RAW_INCUMBENT_COMPLETE'
    assert val['status'] == 'RAW_INCUMBENT_ALL_COUNTS_PASS'
    assert val['authority'] == result['authority'] == binding(AUTH)
    assert val['result'] == binding(OUT/'result.json')
    for section in ('code_sources','public_sources','join_sources'):
        for source in a[section].values():checked(source)
    checked(a['parent'])
    seal = read(OUT/'all_predictions_prelabel_seal.json')
    assert seal['authority'] == binding(AUTH) and len(seal['validations']) == 10
    for source in seal['validations']:checked(source)
    gallery = {r['physical_row']:r['identity'] for r in read(a['public_sources']['gallery']['path'])['records']}
    roles = {r['query_id']:r for r in read(a['join_sources']['curator']['path'])['records']}
    joined = {r['query_id']:r for r in result['rows']}
    parent_rows = {r['query_id']:r for r in read(a['join_sources']['parent_result']['path'])['rows']}
    rows, folds, traces, dominance_inputs = [], [], [], []
    preserved = absent = 0
    for f in range(5):
        for source in a['fold_sources'][str(f)].values():checked(source)
        folder = OUT/f'fold{f}'
        p, fv, iv = (read(folder/n) for n in ('payload.json','validation.json','independent_validation.json'))
        assert fv['status'] == 'RAW_INCUMBENT_FRESH_REPLAY_PASS'
        assert iv['status'] == 'RAW_INCUMBENT_INDEPENDENT_PASS' and iv['passed']
        assert fv['payload'] == iv['payload'] == binding(folder/'payload.json')
        assert fv['authority'] == iv['authority'] == p['authority'] == binding(AUTH)
        assert p['heldout_label_reads'] == p['encoder_forwards'] == p['base_training_updates'] == 0
        parent = read(checked(p['parent_payload']))
        assert p['train_query_ids'] == parent['train_query_ids']
        old = parent['parameters'][OLD]
        assert p['matched_control']['optimization']['theta_hex'] == old['optimization']['theta_hex']
        prior = {r['query_id']:r for r in parent['predictions']}
        inner = {}
        for name, head in p['parameters'].items():
            records = []
            for r, original in zip(p['calibration'], parent['calibration']):
                assert all(r[k] == v for k,v in original.items())
                role = roles[r['query_id']]
                before = ((r['m']+0.)+old['beta']*r['d'])+old['bias']
                after = score(r,head,name)
                os = r['m']<=0 and not old['disabled'] and before>0
                ns = r['m']<=0 and not head['disabled'] and after>0
                records.append(dict(query_id=r['query_id'],original_query_id=role['original_query_id'],
                                    component=role['component'],identity=role['identity'],
                                    old_switch=os,new_switch=ns,old_gate=before,new_gate=after,
                                    accuracy_delta=(int(ns)-int(os))*r['delta']))
            inner[name] = changes(records)
            assert inner[name]['net'] == head['training_net_gain']-old['training_net_gain']
        fold_rows = []
        for pred in p['predictions']:
            q = pred['query_id']; role = roles[q]
            assert role['outer_fold'] == f and q not in p['train_query_ids']
            assert all(pred[k] == v for k,v in prior[q].items() if k!='models')
            assert {k:v for k,v in pred['models'].items() if k not in ARMS} == prior[q]['models']
            z = list(map(float.fromhex,pred['models'][BASE]['logits_hex']))
            top = max(range(127),key=z.__getitem__);m=z[top]
            d=m-max(v for j,v in enumerate(z) if j!=top)
            assert top == pred['original_top'] and d.hex()==float(pred['d']).hex()
            raw = pred['candidate_physical_rows'][pred['winner']]
            alt = pred['candidate_physical_rows'][pred['challenger_positions'][top]]
            decisions = dict(RAW=raw)
            for name,model in pred['models'].items():
                zz=list(map(float.fromhex,model['logits_hex']));j=max(range(127),key=zz.__getitem__)
                pos=pred['challenger_positions'][j] if zz[j]>0 else pred['winner']
                decisions[name]=pred['candidate_physical_rows'][pos]
                assert decisions[name]==model['selected']
            present = any(gallery[c]==role['identity'] for c in pred['candidate_physical_rows'])
            absent += int(not present)
            assert present == joined[q]['target_in_C128']
            correct = {name:gallery[c]==role['identity'] for name,c in decisions.items()}
            assert correct == joined[q]['correct'] and decisions == joined[q]['selected']
            assert {k:v for k,v in correct.items() if k not in ARMS} == parent_rows[q]['correct']
            if m>0:preserved+=1
            else:
                dominance_inputs.append(dict(query_id=q,original_query_id=role['original_query_id'],fold=f,
                                             m=m,d=d,r_pair=pred['r_pair'],r_near=pred['r_near'],
                                             delta=int(gallery[alt]==role['identity'])-int(correct['RAW'])))
            before=((m+0.)+old['beta']*d)+old['bias']
            for name,head in p['parameters'].items():
                raw_value=pred['r_pair'] if name==ARMS[0] else pred['r_near']
                after=score(dict(pred,m=m),head,name)
                expected=list(pred['models'][BASE]['logits_hex'])
                if m<=0 and not head['disabled'] and after>0:expected[top]=after.hex()
                assert expected==pred['models'][name]['logits_hex']
                if m>0:assert pred['models'][name]==pred['models'][BASE]
                specifications={
                    'DROP_RAW_KEEP_NEW_BETA_BIAS':(0.,head['beta'],head['bias']),
                    'OLD_GATE_PLUS_RAW':(head['gamma'],old['beta'],old['bias']),
                    'NEW_SLOPES_OLD_BIAS':(head['gamma'],head['beta'],old['bias']),
                }
                for suffix,(gamma,beta,bias) in specifications.items():
                    g=((m+gamma*raw_value)+beta*d)+bias
                    choice=alt if m>0 or (not head['disabled'] and g>0) else raw
                    key=name+'__'+suffix
                    decisions[key]=choice;correct[key]=gallery[choice]==role['identity']
                if decisions[name]!=decisions[OLD] or role['original_query_id'] in ('OUTCOME-0721','OUTCOME-0809','OUTCOME-0529'):
                    traces.append(dict(arm=name,query_id=q,original_query_id=role['original_query_id'],fold=f,
                                       component=role['component'],identity=role['identity'],
                                       raw_identity=gallery[raw],challenger_identity=gallery[alt],
                                       m=m,d=d,r_pair=pred['r_pair'],r_near=pred['r_near'],
                                       old_gate=before,new_gate=after,raw_term=head['gamma']*raw_value,
                                       beta_change_term=(head['beta']-old['beta'])*d,
                                       bias_change_term=head['bias']-old['bias'],
                                       raw_correct=correct['RAW'],old_correct=correct[OLD],new_correct=correct[name],
                                       accuracy_delta=int(correct[name])-int(correct[OLD]),
                                       old_switch=decisions[OLD]!=raw,new_switch=decisions[name]!=raw))
            row=dict(query_id=q,original_query_id=role['original_query_id'],component=role['component'],
                     fold=f,correct=correct,selected=decisions)
            rows.append(row);fold_rows.append(row)
        folds.append(dict(fold=f,queries=len(fold_rows),inner_changes=inner,
                          old_inner_net=old['training_net_gain'],
                          new_inner_net={n:h['training_net_gain'] for n,h in p['parameters'].items()},
                          old_loss=old['optimization']['fitted_surrogate_loss'],
                          new_loss={n:h['optimization']['fitted_surrogate_loss'] for n,h in p['parameters'].items()},
                          parameter_summary={n:{k:h[k] for k in ('gamma','beta','bias')} for n,h in p['parameters'].items()},
                          fold_correct={n:sum(r['correct'][n] for r in fold_rows) for n in fold_rows[0]['correct']},
                          runtime=read(folder/'runtime.json')))
    assert len(rows)==len({r['query_id'] for r in rows})==593
    assert len({r['component'] for r in rows})==64 and preserved==78 and absent==23
    counts={n:sum(r['correct'][n] for r in rows) for n in rows[0]['correct']}
    for name,expected in result['summary'].items():
        assert counts[name]==expected['correct']
        assert sum(r['selected'][name]!=r['selected']['RAW'] for r in rows)==expected['switches']
        for base,key in [('RAW','against_RAW'),(BASE,'against_COST1')]:
            assert comparison(rows,base,name)==expected[key]
        assert {str(f):sum(r['correct'][name] for r in rows if r['fold']==f) for f in range(5)}==expected['fold_correct']
    for key,expected in result['comparisons'].items():
        base,new=key.split('__to__');assert comparison(rows,base,new)==expected
    types={}
    for name in ARMS:
        categories={}
        for r in rows:
            if r['correct'][name]==r['correct'][OLD]:continue
            key=('rescue' if r['correct'][name] else 'loss')+('_raw_correct' if r['correct']['RAW'] else '_raw_wrong')
            categories.setdefault(key,[]).append(r['original_query_id'])
        types[name]=categories
    # Exact componentwise inequalities, no optimization or selected parameters.
    # With nonnegative slopes an N row at least as large in every input always
    # scores at least as high as P, so P=SWITCH and N=HOLD cannot both hold.
    certificates={}
    for name,keys in [('GAP_BIAS2',('m','d')),('GAP_RAWPAIR3',('m','d','r_pair')),
                      ('GAP_RAWNEAR3',('m','d','r_near'))]:
        pairs=[]
        for positive in dominance_inputs:
            if positive['delta']!=1:continue
            for negative in dominance_inputs:
                if negative['delta']!=-1 or negative['fold']!=positive['fold']:continue
                if all(negative[k]>=positive[k] for k in keys):
                    pairs.append(dict(should_switch=positive,should_hold=negative,
                                      nonnegative_input_differences={k:negative[k]-positive[k] for k in keys}))
        certificates[name]=dict(pair_count=len(pairs),
                                distinct_positive_queries=len({v['should_switch']['query_id'] for v in pairs}),
                                pairs=pairs)
    assert sum(r['delta']==1 for r in dominance_inputs)==46
    output=dict(status='RAW_INCUMBENT_POST_JOIN_RECOUNT_PASS',code=binding(__file__),
                shared_helpers=binding(ROOT/'programs/audit_rc_h593_gap_net2_result_v1.py'),
                authority=binding(AUTH),result=binding(OUT/'result.json'),counts=counts,
                queries=593,components=64,target_missing=23,original_switches_preserved=78,
                parameter_updates=0,comparisons=result['comparisons'],folds=folds,change_types=types,
                changed_traces=traces,posthoc_fixed_coordinate_interventions=True,
                monotone_gate_conflicts=certificates,
                conflict_scope='Same-fold original HOLD, fixed m coefficient=1, beta/gamma>=0; no claim for arbitrary nonlinear classifiers or net-gain impossibility.',
                intervention_comparisons={n+'__drop_raw_to_full':comparison(rows,n+'__DROP_RAW_KEEP_NEW_BETA_BIAS',n) for n in ARMS},
                limits=['Frozen-coordinate diagnostics added after observing results; not independently trained arms.',
                        'No parameter fitting or selection in this audit.',
                        'Independent recount covers actions, accuracy, candidate inclusion and group statistics; MRR uses existing join replay.',
                        'Opened H593 development; historical model selection is not corrected by bootstrap intervals.'])
    (OUT/'post_join_analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (OUT/'changed_cases.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(traces[0]));writer.writeheader();writer.writerows(traces)
    print(json.dumps(dict(status=output['status'],counts=counts,change_types=types,
                         intervention_comparisons=output['intervention_comparisons']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
