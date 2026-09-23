#!/usr/bin/env python3
"""Per-query heldout loss accounting only; never used as training input."""
from collections import defaultdict
import numpy as np
import run_rc_h593_content_complement_v1 as J


def main():
    a=J.read(J.AUTH);v=J.read(J.OUT/'validation.json')
    assert v['status']=='CONTENT_COMPLEMENT_ALL_COUNTS_PASS' and v['authority']==J.bind(J.AUTH)
    result=J.read(J.checked(v['result']));joined={r['query_id']:r for r in result['rows']}
    roles={r['query_id']:r for r in J.read(J.checked(a['join_sources']['curator']))['records']}
    labels={r['physical_row']:r['identity'] for r in J.read(J.checked(a['public_sources']['gallery']))['records']}
    models=('CE_FULL','DIAG_CE13','JOINT_CONTENT_CE18');rows=[]
    for f in range(5):
        seal=J.read(J.checked(v['fold_validations'][f]));p=J.read(J.checked(seal['payload']))
        for pred in p['predictions']:
            q=pred['query_id'];axis=pred['candidate_physical_rows']
            targets=[i for i,physical in enumerate(axis) if labels[physical]==roles[q]['identity']]
            assert len(targets)<=1 and bool(targets)==joined[q]['target_in_C128']
            if not targets:continue
            k=0 if targets[0]==pred['winner'] else pred['challenger_positions'].index(targets[0])+1
            item=dict(query_id=q,fold=f,original_query_id=joined[q]['original_query_id'],models={})
            for model in models:
                z=np.r_[0.,[float.fromhex(s) for s in pred['models'][model]['logits_hex']]]
                rival=np.delete(z,k).max();margin=float(z[k]-rival);peak=z.max()
                correct=pred['models'][model]['selected']==axis[targets[0]]
                assert correct==joined[q]['correct'][model]
                item['models'][model]=dict(CE=float(peak+np.log(np.exp(z-peak).sum())-z[k]),
                    full_top1_softplus=float(np.logaddexp(0.,-margin)),bounded_margin=float(np.exp(-np.logaddexp(0.,margin))),
                    target_margin=margin,correct=correct)
            rows.append(item)
    assert len(rows)==570
    groups=defaultdict(list)
    for row in rows:
        old=row['models']['DIAG_CE13'];new=row['models']['JOINT_CONTENT_CE18']
        group=('gain' if new['correct'] else 'loss') if old['correct']!=new['correct'] else ('both_correct' if new['correct'] else 'both_wrong')
        groups[group].append(new['CE']-old['CE'])
    strata={g:dict(n=len(values),CE_delta_sum=float(sum(values)),CE_delta_mean=float(np.mean(values))) for g,values in groups.items()}
    delta=sum(x['CE_delta_sum'] for x in strata.values())
    aggregate={m:{k:float(np.mean([row['models'][m][k] for row in rows])) for k in
        ('CE','full_top1_softplus','bounded_margin','target_margin')} for m in models}
    output=dict(status='CONTENT_LOSS_READOUT_PASS',source=J.bind(__file__),result=v['result'],present=570,absent=23,
        aggregate=aggregate,CE_delta_strata=strata,
        unchanged_share_of_CE_decrease=(strata['both_correct']['CE_delta_sum']+strata['both_wrong']['CE_delta_sum'])/delta,
        rows=rows,scope='Opened OOF post-hoc loss accounting; no fitting, no new accuracy, no convergence or universal cause claim.')
    J.write(J.OUT/'loss_readout.json',output)
    print({k:output[k] for k in ('status','aggregate','CE_delta_strata','unchanged_share_of_CE_decrease')})


if __name__=='__main__': main()
