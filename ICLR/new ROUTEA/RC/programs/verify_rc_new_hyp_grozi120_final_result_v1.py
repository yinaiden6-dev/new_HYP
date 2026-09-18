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
OUT = ROOT / 'results/rc_new_hyp_grozi120_external_v1'
AUTH = ROOT / 'registry/rc_new_hyp_grozi120_inference_authority_v1_20260913.json'

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
    assert rv['authority'] == ab and rv['status'] == 'GROZI480_ALL_SEALS_COUNTS_GROUPS_PASS'
    result = checked(rv['result'])
    assert result['authority'] == ab and result['primary'] == 'COST1'
    for b in a['sources'].values():
        if Path(b['path']).stat().st_size < 16 * (1 << 20): assert bind(b['path']) == b
    for sources in a['heads'].values():
        for b in sources.values(): assert bind(b['path']) == b
    seal = read(OUT / 'all_predictions_prejoin_seal.json')
    assert seal['authority'] == ab and seal['queries'] == 480 and len(seal['validations']) == 60
    assert rv['all_prediction_validations'] == seal['validations']
    workers = checked(a['sources']['worker'])['records']
    gallery = checked(a['sources']['gallery_append'])['records']
    target_rows = {r['identity']: r['physical_row'] for r in gallery}
    assert len(target_rows) == 120 and set(target_rows.values()) == set(range(5413,5533))
    roles = checked(a['curator_after_all_seals'])['records']
    roles = {r['query_id']: r for r in roles}
    rows, payload_bindings, action_checks = [], [], 0
    original = {r['query_id']:r for r in result['rows']}
    max_error = 0.0
    for shard, vb in enumerate(seal['validations']):
        v = checked(vb)
        assert v['authority'] == ab and v['status'] == 'GROZI_FROZEN_HEAD_NUMPY_ACTION_PASS'
        max_error = max(max_error, v['max_abs_error'])
        payload = checked(v['payload']); payload_bindings.append(v['payload'])
        assert payload['authority'] == ab and payload['target_reads'] == payload['training_updates'] == 0
        assert len(payload['records']) == 8
        for kind in ('raw','roma'):
            stage = OUT/kind/f'shard{shard:02d}'
            receipt, validation = read(stage/'receipt.json'), read(stage/'validation.json')
            assert receipt['authority'] == validation['authority'] == ab
            assert receipt['payload'] == validation['payload'] == payload[kind]
            assert validation['status'] == 'GROZI_'+kind.upper()+'_CPU_PASS'
            if kind == 'roma': assert validation['C4_checks'] == 4096
        for p in payload['records']:
            order = p['raw_ranked_physical_rows']
            assert len(order) == len(set(order)) == 5532
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
            rows.append(dict(query_id=p['query_id'],identity=role['identity'],component=role['component'],target=target,raw_rank=raw_rank,correct=correct,selected=selected,ranks=ranks,top=top,max_logit=logit))
    assert len(rows) == 480 and len({r['identity'] for r in rows}) == 120
    assert all(n == 4 for n in collections.Counter(r['identity'] for r in rows).values())
    models = sorted(rows[0]['correct'])
    counts = {m:sum(r['correct'][m] for r in rows) for m in models}
    assert counts == result['counts']
    for m in models:
        mrr = math.fsum(1/r['ranks'][m] for r in rows)/480
        assert abs(mrr-result['MRR'][m]) < 1e-12
    groups = sorted({r['component'] for r in rows}); assert len(groups) == 27
    group_accuracy = {g:{m:sum(r['correct'][m] for r in rows if r['component']==g)/sum(r['component']==g for r in rows) for m in models} for g in groups}
    draws = np.random.default_rng(20260913).integers(27,size=(100000,27))
    comparisons={}
    for key,c in result['comparisons'].items():
        base,model = key.split('__to__')
        rescue=sum(not r['correct'][base] and r['correct'][model] for r in rows)
        loss=sum(r['correct'][base] and not r['correct'][model] for r in rows)
        delta=np.array([group_accuracy[g][model]-group_accuracy[g][base] for g in groups])
        samples=np.sum(delta[draws],axis=1)/27
        ci=np.quantile(samples,[.025,.975]).tolist()
        assert (rescue,loss,rescue-loss)==(c['rescue'],c['loss'],c['net'])
        assert np.allclose(ci,c['video_bootstrap95'],rtol=0,atol=1e-12)
        assert abs(float(delta.mean())-c['equal_video_difference']) < 1e-12
        reliable=bool(rescue>loss and delta.mean()>0 and ci[0]>0)
        assert reliable==c['reliable_positive']
        comparisons[key]=dict(rescue=rescue,loss=loss,net=rescue-loss,equal_video_difference=float(delta.mean()),video_bootstrap95=ci,reliable_positive=reliable,positive_video_groups=int((delta>1e-15).sum()),negative_video_groups=int((delta < -1e-15).sum()))
    gates=[comparisons[b+'__to__COST1']['reliable_positive'] for b in ('RAW','COST4','GROUP_COST4','RAW2_CE','COST1_CBIND')]
    assert all(gates)==result['scoped_external_new_HYP_GO']
    assert sum(r['raw_rank']<=128 for r in rows)==result['target_recall_C128']==410
    breakdown={}
    for m in models:
        if m=='RAW':continue
        wrong=[r for r in rows if not r['correct'][m]]
        switches=[r for r in rows if r['selected'][m]!=r['selected']['RAW']]
        breakdown[m]=dict(holds=480-len(switches),switches=len(switches),rescues_vs_RAW=sum(not r['correct']['RAW'] and r['correct'][m] for r in rows),breaks_vs_RAW=sum(r['correct']['RAW'] and not r['correct'][m] for r in rows),wrong_to_other_wrong=sum(not r['correct']['RAW'] and not r['correct'][m] for r in switches),recall_absent_errors=sum(r['raw_rank']>128 for r in wrong),candidate_present_errors=sum(r['raw_rank']<=128 for r in wrong),correct_challenger_held=sum(r['raw_rank']<=128 and r['top'][m]==r['target'] and r['max_logit'][m]<=0 for r in wrong))
    output=dict(status='GROZI480_INDEPENDENT_FINAL_AUDIT_PASS',checked_utc=datetime.now(timezone.utc).isoformat(),program=bind(__file__),result=rv['result'],authority=ab,prediction_payloads=payload_bindings,counts=counts,comparisons=comparisons,action_breakdown=breakdown,head_action_reconstructions=action_checks,head_numpy_max_abs_error=max_error,qualified_C4_scalar_checks=60*4096,query_count=480,identities=120,source_video_groups=27,target_recall_C128=410,scoped_external_new_HYP_GO=all(gates),scope=result['GO_scope'],model_forwards=0,training_updates=0)
    path=OUT/'independent_final_audit_v1.json'
    with path.open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('prediction_payloads','action_breakdown')},ensure_ascii=False))
    print(json.dumps({'action_breakdown':{k:breakdown[k] for k in ('COST1','CE','COST1_CBIND')}}))
if __name__=='__main__':main()
