#!/usr/bin/env python3
"""Independent JSON/NumPy recount of the fixed pooled-feature readout results."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_pair_relation_localization_v1/feature_readout'
def read(p):return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def checked(b):
    assert bind(b['path'])==b;return b['path']

if __name__=='__main__':
    result=read(OUT/'result.json');spec=read(checked(result['spec']));seal=read(checked(result['score_seal']))
    checked(spec['program']);join=read(OUT/'join_spec.json');checked(join['program'])
    assert join['source_score_seal']==result['score_seal']
    gallery={r['physical_row']:r['identity'] for r in read(checked(spec['gallery']))['records']}
    roles={r['query_id']:r for r in read(checked(spec['identities']))['records']}
    hist={r['query_id']:r for r in read(checked(spec['historical_result']))['rows']}
    scores=[read(checked(b)) for b in seal['predictions']]
    assert len(scores)==71 and [p['execution_ordinal'] for p in scores]==list(range(71))
    recount={k:dict(correct=0,rescue=0,breaks=0) for k in result['summary']}
    rowmap={r['query_id']:r for r in result['rows']};count=0;maxerr=0.
    groups={k:defaultdict(list) for k in recount}
    for p in scores:
        qid=p['query_id'];r=rowmap[qid];role=roles[qid];c=p['candidate_physical_rows']
        assert len(c)==len(set(c))==128 and p['spec']==result['spec']
        target=[i for i,x in enumerate(c) if gallery[x]==role['identity']]
        assert r['target_position']==(target[0] if target else None)
        raw=bool(hist[qid]['correct']['RAW'])
        for method,acc in recount.items():
            v=np.asarray(p['scores'][method],dtype=np.float64);assert v.shape==(128,) and np.isfinite(v).all()
            selection=c[int(np.argmax(v))];correct=gallery[selection]==role['identity'];count+=128
            assert selection==r['scores'][method]['selected_physical_row'] and correct==r['scores'][method]['top1_correct']
            acc['correct']+=int(correct);acc['rescue']+=int(correct and not raw);acc['breaks']+=int(raw and not correct)
            groups[method][role['component']].append(int(correct)-int(raw))
            if target:
                t=target[0];others=np.delete(v,t);m=float(v[t]-others.max())
                maxerr=max(maxerr,abs(m-r['scores'][method]['target_minus_max_wrong']))
                assert int((v>v[t]).sum()+1)==r['scores'][method]['target_rank_min']
    assert maxerr<1e-12 and seal['numpy_stat_checks']==6 and seal['max_numpy_error']<2e-6
    for method,r in recount.items():
        s=result['summary'][method]
        assert r['correct']==s['correct'] and r['rescue']==s['rescue_vs_RAW'] and r['breaks']==s['breaks_vs_RAW']
        assert r['correct']==result['RAW']+r['rescue']-r['breaks']
        g=float(np.mean([np.mean(x) for x in groups[method].values()]))
        assert abs(g-s['group_equal_top1_delta_vs_RAW'])<1e-12
    output=dict(status='READOUT71_SOURCE_SHA_NUMPY_IDENTITY_RECOUNT_PASS',program=bind(__file__),
                result=bind(OUT/'result.json'),score_seal=bind(OUT/'score_seal.json'),queries=71,candidates=9088,
                complete_readouts=len(recount),candidate_scores_checked=count,max_margin_error=maxerr,
                independent_float64_matrix_stat_checks=seal['numpy_stat_checks'],
                max_independent_float64_matrix_stat_error=seal['max_numpy_error'],
                geometry_pooling_not_independently_recomputed=True,
                evidence='Engineering validation of pooled-score artifacts, not native-matcher causal verification')
    (OUT/'validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))
