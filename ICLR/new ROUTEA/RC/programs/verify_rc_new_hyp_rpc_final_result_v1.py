#!/usr/bin/env python3
"""Independent result audit from sealed logits; no producer imports or model calls."""
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_new_hyp_rpc_transfer_v1'
AUTH = ROOT / 'registry/rc_new_hyp_rpc_inference_authority_v1_20260913.json'

def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p = Path(p)
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(b):
    assert bind(b['path']) == b, 'SHA_MISMATCH:' + b['path']
    return read(b['path'])
def main():
    a = read(AUTH); ab = bind(AUTH)
    rv = read(OUT / 'result_validation.json')
    assert rv['authority'] == ab and rv['status'] == 'RPC600_ALL_SEALS_COUNTS_MRR_ACTIONS_PASS'
    result = checked(rv['result'])
    assert result['authority'] == ab and result['primary'] == 'COST1'
    for b in a['sources'].values():
        if Path(b['path']).stat().st_size < 16 * (1 << 20): assert bind(b['path']) == b
    for sources in a['heads'].values():
        for b in sources.values(): assert bind(b['path']) == b
    seal = read(OUT / 'all_predictions_prejoin_seal.json')
    assert seal['authority'] == ab and seal['queries'] == 600 and len(seal['validations']) == 75
    assert rv['all_prediction_validations'] == seal['validations']
    workers = checked(a['sources']['worker'])['records']
    gallery = checked(a['sources']['gallery'])['records']
    target_rows = {r['identity']: r['physical_row'] for r in gallery}
    assert len(target_rows) == 200 and set(target_rows.values()) == set(range(200))
    roles = checked(a['curator_after_all_seals'])['records']
    roles = {r['query_id']: r for r in roles}
    rows, payload_bindings, action_checks = [], [], 0
    original = {r['query_id']:r for r in result['rows']}
    max_error = 0.0
    for shard, vb in enumerate(seal['validations']):
        v = checked(vb)
        assert v['authority'] == ab and v['status'] == 'RPC_FROZEN_HEAD_NUMPY_ACTION_PASS'
        max_error = max(max_error, v['max_abs_error'])
        payload = checked(v['payload']); payload_bindings.append(v['payload'])
        assert payload['authority'] == ab and payload['target_reads'] == payload['training_updates'] == 0
        assert len(payload['records']) == 8
        for kind in ('raw','roma'):
            stage = OUT/kind/f'shard{shard:02d}'
            receipt, validation = read(stage/'receipt.json'), read(stage/'validation.json')
            assert receipt['authority'] == validation['authority'] == ab
            assert receipt['payload'] == validation['payload'] == payload[kind]
            assert validation['status'] == 'RPC_'+kind.upper()+'_CPU_PASS'
            if kind == 'roma': assert validation['C4_checks'] == 4096
        for p in payload['records']:
            order = p['raw_ranked_physical_rows']
            assert len(order) == len(set(order)) == 200
            assert p['raw_selected'] == order[0]
            ordinal = len(rows); role = roles[p['query_id']]
            assert p['query_id'] == workers[ordinal]['query_id'] and p['execution_ordinal'] == ordinal
            target = target_rows[role['identity']]
            assert target in order
            candidate_axis = sorted(order[:128]); winner = order[0]
            challengers = [x for x in candidate_axis if x != winner]
            selected = {'RAW':winner}; top = {}; logit = {}
            for model, prediction in p['models'].items():
                z = [float.fromhex(x) for x in prediction['logits_hex']]
                assert len(z) == 127 and all(math.isfinite(x) for x in z)
                k = max(range(127),key=lambda i:z[i])
                chosen = challengers[k] if z[k] > 0 else winner
                assert chosen == prediction['selected']
                selected[model] = chosen; top[model] = challengers[k]; logit[model] = z[k]
                action_checks += 1
            correct = {m: s == target for m,s in selected.items()}
            ranks = {}
            for m,s in selected.items():
                reranked = [s] + [x for x in order if x != s]
                ranks[m] = reranked.index(target)+1
            raw_rank = order.index(target)+1
            stored = original[p['query_id']]
            assert stored['selected'] == selected and stored['correct'] == correct and stored['ranks'] == ranks
            assert stored['target_in_C128'] == (raw_rank <=128)
            assert stored['identity'] == role['identity'] and stored['component'] == role['component']
            assert stored['camera']==role['camera'] and stored['stratum']==role['stratum']
            rows.append(dict(query_id=p['query_id'],identity=role['identity'],component=role['component'],stratum=role['stratum'],camera=role['camera'],target=target,raw_rank=raw_rank,correct=correct,selected=selected,ranks=ranks,top=top,max_logit=logit))
    assert len(rows) == 600 and len({r['identity'] for r in rows}) == 200
    assert all(n == 3 for n in collections.Counter(r['identity'] for r in rows).values())
    models = sorted(rows[0]['correct'])
    counts = {m:sum(r['correct'][m] for r in rows) for m in models}
    assert counts == result['counts']
    for m in models:
        mrr = math.fsum(1/r['ranks'][m] for r in rows)/600
        assert abs(mrr-result['MRR'][m]) < 1e-12
    identities=sorted({r['identity'] for r in rows});assert len(identities)==200
    strata=sorted({r['stratum'] for r in rows});assert len(strata)==17
    sku_rows={identity:[r for r in rows if r['identity']==identity] for identity in identities}
    assert all(len(v)==3 and {r['camera'] for r in v}=={1,2,3} and len({r['stratum'] for r in v})==1 for v in sku_rows.values())
    sku_accuracy=np.array([[sum(r['correct'][m] for r in sku_rows[i])/3 for m in models] for i in identities])
    stratum_indices={s:np.array([j for j,i in enumerate(identities) if sku_rows[i][0]['stratum']==s]) for s in strata}
    rng=np.random.default_rng(20260913)
    sampled=[]
    for s in strata:
        ids=stratum_indices[s]
        sampled.append(ids[rng.integers(len(ids),size=(100000,len(ids)))])
    sampled=np.concatenate(sampled,axis=1);assert sampled.shape==(100000,200)
    group_draws=np.random.default_rng(20260913).integers(17,size=(100000,17))
    comparisons={}
    for key,c in result['comparisons'].items():
        base,model=key.split('__to__')
        rescue=sum(not r['correct'][base] and r['correct'][model] for r in rows)
        loss=sum(r['correct'][base] and not r['correct'][model] for r in rows)
        delta=sku_accuracy[:,models.index(model)]-sku_accuracy[:,models.index(base)]
        ci=np.quantile(delta[sampled].mean(axis=1),[.025,.975]).tolist()
        group_means=np.array([delta[stratum_indices[s]].mean() for s in strata])
        group_ci=np.quantile(group_means[group_draws].mean(axis=1),[.025,.975]).tolist()
        assert (rescue,loss,rescue-loss)==(c['rescue'],c['loss'],c['net'])
        assert np.allclose(ci,c['sku_stratified_bootstrap95'],rtol=0,atol=1e-12)
        assert np.allclose(group_ci,c['supercategory_bootstrap95'],rtol=0,atol=1e-12)
        assert abs(float(delta.mean())-c['equal_sku_difference'])<1e-12
        assert abs(float(group_means.mean())-c['supercategory_equal_mean_difference'])<1e-12
        reliable=bool(rescue>loss and delta.mean()>0 and ci[0]>0)
        assert reliable==c['reliable_positive']
        comparisons[key]=dict(rescue=rescue,loss=loss,net=rescue-loss,equal_sku_difference=float(delta.mean()),sku_stratified_bootstrap95=ci,supercategory_equal_mean_difference=float(group_means.mean()),supercategory_bootstrap95=group_ci,reliable_positive=reliable,positive_SKU=int((delta>1e-15).sum()),negative_SKU=int((delta < -1e-15).sum()))
    for field,column in [('camera','by_camera'),('stratum','by_supercategory'),('identity','by_sku')]:
        for value in sorted({r[field] for r in rows}):
            rr=[r for r in rows if r[field]==value];stored=result[column][str(value)]
            assert stored['queries']==len(rr)
            assert stored['correct']=={m:sum(r['correct'][m] for r in rr) for m in models}
            assert stored['recall_C128']==sum(r['raw_rank']<=128 for r in rr)
            for m in models:assert abs(stored['MRR'][m]-math.fsum(1/r['ranks'][m] for r in rr)/len(rr))<1e-12
    gates=[comparisons[b+'__to__COST1']['reliable_positive'] for b in ('RAW','COST4','GROUP_COST4','RAW2_CE','COST1_CBIND')]
    assert all(gates)==result['scoped_external_new_HYP_GO']
    assert sum(r['raw_rank']<=128 for r in rows)==result['target_recall_C128']==575
    breakdown={}
    for m in models:
        if m=='RAW':continue
        wrong=[r for r in rows if not r['correct'][m]]
        switches=[r for r in rows if r['selected'][m]!=r['selected']['RAW']]
        breakdown[m]=dict(holds=600-len(switches),switches=len(switches),rescues_vs_RAW=sum(not r['correct']['RAW'] and r['correct'][m] for r in rows),breaks_vs_RAW=sum(r['correct']['RAW'] and not r['correct'][m] for r in rows),wrong_to_other_wrong=sum(not r['correct']['RAW'] and not r['correct'][m] for r in switches),recall_absent_errors=sum(r['raw_rank']>128 for r in wrong),candidate_present_errors=sum(r['raw_rank']<=128 for r in wrong),correct_challenger_held=sum(r['raw_rank']<=128 and r['top'][m]==r['target'] and r['max_logit'][m]<=0 for r in wrong))
    for m,b in result['action_breakdown'].items():
        assert all(breakdown[m][key]==value for key,value in b.items())
    assert action_checks==6000
    output=dict(status='RPC600_INDEPENDENT_FINAL_AUDIT_PASS',checked_utc=datetime.now(timezone.utc).isoformat(),program=bind(__file__),result=rv['result'],authority=ab,prediction_payloads=payload_bindings,counts=counts,comparisons=comparisons,action_breakdown=breakdown,head_action_reconstructions=action_checks,head_numpy_max_abs_error=max_error,qualified_C4_scalar_checks=75*4096,query_count=600,identities=200,supercategory_count=17,target_recall_C128=575,scoped_external_new_HYP_GO=all(gates),scope=result['GO_scope'],model_forwards=0,training_updates=0)
    path=OUT/'independent_final_audit_v1.json'
    with path.open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('prediction_payloads','action_breakdown')},ensure_ascii=False))
    print(json.dumps({'action_breakdown':{k:breakdown[k] for k in ('COST1','CE','COST1_CBIND')}}))
if __name__=='__main__':main()
