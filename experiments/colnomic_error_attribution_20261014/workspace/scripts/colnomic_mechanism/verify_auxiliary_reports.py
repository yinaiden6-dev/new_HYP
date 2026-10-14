"""Independent matrix/relationship and multilabel35 result verification."""
from collections import defaultdict
import numpy as np
import torch
from scripts.colnomic_mechanism.common import ROOT,data,read,write,rows,sha
from scripts.colnomic_mechanism.verify_shared_phrase_probe import relation
BASE=ROOT/'results/colnomic_full_audit/20261013_resume'


def main():
    torch.set_num_threads(2);d=data();qi={q:i for i,q in enumerate(d['qb']['ids'])};ri={r:i for i,r in enumerate(d['rb']['ids'])}
    folder=BASE/'correspondence_controls';index=read(folder/'matrix_index.json');maximum=0.;feature_error=0.;matrix_lookup={}
    from scripts.colnomic_mechanism.cross_token_probe import NAMES
    for entry in index:
        path=folder/entry['path'];assert sha(path)==entry['SHA256'];array=np.load(path)
        q=d['qb']['embeddings'][qi[entry['photo_id']]].float().numpy();r=d['rb']['embeddings'][ri[entry['reference_id']]].float().numpy()
        qix=np.flatnonzero(np.linalg.norm(q,axis=1)>1e-8);rix=np.flatnonzero(np.linalg.norm(r,axis=1)>1e-8)
        assert np.array_equal(qix,array['query_indices']) and np.array_equal(rix,array['reference_indices'])
        expected=q[qix]@r[rix].T;error=float(np.max(np.abs(expected-array['similarity'])));maximum=max(maximum,error);assert error<3e-6
        features=relation(array['similarity']);stored=np.array([entry['metrics'][name] for name in NAMES]);err=float(np.max(np.abs(features-stored)));feature_error=max(feature_error,err);assert err<2e-6
        matrix_lookup[entry['photo_id'],entry['reference_id']]=entry
    gaps=rows(folder/'fixed_candidate_same_face_controls.csv');gap_error=0.;controls={r['photo_id']:r for r in read(BASE/'visual_consolidated/same_face_success_controls.json')}
    for item in gaps:
        q=item['failure_photo'];s=item['success_photo'];assert s in controls[q][item['configuration']]
        positive=item['positive_reference'];negative=item['negative_reference'];expected={}
        for qid,label in [(q,'failure'),(s,'success')]:
            p=matrix_lookup[qid,positive];n=matrix_lookup[qid,negative]
            expected[label]=p['metrics']['forward_top1_mean']-n['metrics']['forward_top1_mean']
            assert abs(expected[label]-float(item[label+'_image_gap']))<1e-12
        gap_error=max(gap_error,abs(expected['success']-expected['failure']-float(item['gap_success_minus_failure'])))
    assert gap_error<1e-12
    for path,digest in read(folder/'validation.json')['source_SHA256'].items():assert sha(ROOT/path)==digest,'source changed: '+path
    write(folder/'independent_validation.json',dict(status='passed',complete_matrices_replayed=len(index),controls_checked=len(gaps),
        pointwise_max_error=maximum,relation_statistic_max_error=feature_error,gap_max_error=gap_error,
        scope='all exported fixed-candidate matrices and confirmed-control ties; numerical replay does not establish semantic causality'))
    folder=BASE/'difficulty_config';visual={r['photo_id']:r for r in read(BASE/'visual_consolidated/records.json')}
    pred=defaultdict(dict)
    for row in rows(BASE/'scene35_reviewed/prediction_matrix.csv'):
        pred[row['photo_id']][row['config_id']]=None if row['prediction_available']!='True' else row['correct']=='True'
    summary=read(folder/'validation.json');configs=sorted(next(iter(pred.values())));checked=0
    for row in rows(folder/'difficulty_configuration_rates.csv'):
        subset=[q for q,r in visual.items() if row['difficulty']=='__all__' or row['difficulty'] in r['difficulty_labels']]
        c=row['configuration'];available=[q for q in subset if pred[q][c] is not None];failed=sum(not pred[q][c] for q in available)
        assert int(row['photos'])==len(subset) and int(row['available_predictions'])==len(available) and int(row['failed'])==failed
        if available:assert abs(float(row['within_difficulty_failure_rate'])-failed/len(available))<1e-12
        checked+=1
    trans=rows(folder/'difficulty_method_transitions.csv')
    for row in trans:
        subset=[q for q,r in visual.items() if row['difficulty']=='__all__' or row['difficulty'] in r['difficulty_labels']]
        c=row['configuration'];values=[(pred[q]['B03'],pred[q][c]) for q in subset if pred[q]['B03'] is not None and pred[q][c] is not None]
        facts={'repair_B03_errors':sum(not a and b for a,b in values),'harm_B03_successes':sum(a and not b for a,b in values),
            'both_fail':sum(not a and not b for a,b in values),'both_correct':sum(a and b for a,b in values)}
        assert all(int(row[k])==v for k,v in facts.items()) and int(row['net'])==facts['repair_B03_errors']-facts['harm_B03_successes']
    allfail=[q for q,p in pred.items() if len(p)==35 and all(v is False for v in p.values())]
    assert len(allfail)==summary['comparison_scopes']['all35_fail']['queries']
    for path,digest in summary['source_SHA256'].items():assert sha(ROOT/path)==digest,'source changed: '+path
    write(folder/'independent_validation.json',dict(status='passed',queries=len(pred),configurations=len(configs),rate_rows_checked=checked,transition_rows_checked=len(trans),
        all35_failed_query_ids=allfail,review_scope='derived from human-authored normalized multilabel records; raw images not newly reviewed by this numerical validator',
        leakage_or_causality_inferred=False))
    print('Independent auxiliary matrix and cross-configuration verification passed',len(index),checked,len(trans),flush=True)


if __name__=='__main__':main()
