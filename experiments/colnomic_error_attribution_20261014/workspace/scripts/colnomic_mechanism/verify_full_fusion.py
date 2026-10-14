"""Read-only, separately implemented ranking and fusion artifact checks."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch

from scripts.colnomic_mechanism.common import ROOT, data, read, rows, write, sha


def independent_ranks(matrix, hashes, positives, ids):
    result = []
    for i, qid in enumerate(ids):
        seen = set()
        rank = None
        top = None
        for j in np.lexsort((np.arange(matrix.shape[1]), -matrix[i])):
            h = hashes[j]
            if h in seen:
                continue
            seen.add(h)
            if top is None:
                top = int(j)
            if rank is None and h in positives[qid]:
                rank = len(seen)
        assert rank is not None
        result.append((rank, top))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    d = data()
    folder = args.folder.resolve()
    axes = read(folder/'score_axes.json')
    checks = {'query_axes_match': axes['query_ids']==d['qb']['ids'], 'reference_axes_match': axes['reference_ids']==d['rb']['ids'],
              'reference_SHA_match': axes['reference_sha256']==d['rb']['sha256']}
    ledger = read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')
    positives = {r['photo_id']:set(r['positive_reference_sha256']) for r in ledger}
    arrays = np.load(folder/'full_gallery_scores.npz')
    pred_rows = rows(folder/'per_query_rank_changes.csv')
    reported = {(r['method'],r['photo_id']):r for r in pred_rows}
    all_rankings = {}
    mismatches = []
    for method in arrays.files:
        matrix = arrays[method]
        assert matrix.shape==d['image']['scores'].shape and np.isfinite(matrix).all()
        rankings = independent_ranks(matrix,d['rb']['sha256'],positives,d['qb']['ids'])
        all_rankings[method] = rankings
        for qid,(rank,top) in zip(d['qb']['ids'],rankings):
            row=reported.get((method,qid))
            if row is None or int(row['rank'])!=rank or row['top1']!=d['rb']['ids'][top]:
                mismatches.append(dict(method=method,photo_id=qid,rank=rank,top=top))
    checks['all_saved_ranks_and_top1_match_independent_full_sort'] = not mismatches
    checks['no_duplicate_or_missing_prediction_rows'] = len(reported)==len(pred_rows)==len(arrays.files)*len(d['qb']['ids'])
    metric_errors=[]
    qids=d['qb']['ids']
    for row in rows(folder/'metrics.csv'):
        keep=[i for i,q in enumerate(qids) if row['split']=='all' or d['splits'][q]==row['split']]
        ranks=np.array([all_rankings[row['method']][i][0] for i in keep])
        before=np.array([all_rankings[row['baseline']][i][0] for i in keep])
        for key,val in [('r1',np.mean(ranks==1)),('r5',np.mean(ranks<=5)),('r10',np.mean(ranks<=10)),('mrr',np.mean(1/ranks))]:
            metric_errors.append(abs(float(row[key])-val))
        rescued=int(((ranks==1)&(before!=1)).sum())
        harmed=int(((ranks!=1)&(before==1)).sum())
        assert (rescued,harmed,rescued-harmed)==(int(row['rescued']),int(row['harmed']),int(row['net']))
        original_rescues=[i for i in keep if all_rankings['image'][i][0]!=1 and all_rankings['original_fusion'][i][0]==1]
        lost=sum(all_rankings[row['method']][i][0]!=1 for i in original_rescues)
        assert lost==int(row['original_fusion_rescues_lost'])
        for key, fn in [('r1',lambda r:r==1),('r5',lambda r:r<=5),('r10',lambda r:r<=10),('mrr',lambda r:1/r)]:
            values=defaultdict(list)
            for i in keep:
                values[d['photos'][qids[i]]['package_id']].append(fn(all_rankings[row['method']][i][0]))
            metric_errors.append(abs(float(row['package_macro_'+key])-np.mean([np.mean(v) for v in values.values()])))
    checks['metrics_match_independent_ranks'] = max(metric_errors)<1e-12
    terms=defaultdict(lambda:np.zeros(2))
    for r in rows(folder/'OCR_term_contributions.csv'):
        terms[(r['photo_id'],r['positive'],r['negative'])]+=np.array([float(r['BM25_positive']),float(r['BM25_negative'])])
    decomposition_errors=[]
    term_errors=[]
    for r in rows(folder/'all_query_decomposition.csv'):
        decomposition_errors.append(abs(float(r['normalized_image_contribution'])+float(r['normalized_text_contribution'])-float(r['fusion_gap'])))
        termsum=terms[(r['photo_id'],r['positive'],r['negative'])]
        term_errors.extend(np.abs(termsum-np.array([float(r['BM25_positive']),float(r['BM25_negative'])])).tolist())
    checks['normalized_components_conserve'] = max(decomposition_errors)<1e-5
    # The term table is FP64 analytic arithmetic; baseline accumulates terms FP32.
    checks['analytic_OCR_terms_replay_FP32_scores_with_documented_roundoff'] = max(term_errors)<.01
    grid=rows(folder/'validation_grid.csv')
    selection=read(folder/'selection.json')
    candidates=[(float(r['package_macro_r1']),float(r['package_macro_mrr']),-abs(float(r['alpha'])-.5),-float(r['confidence_threshold']),float(r['alpha']),float(r['confidence_threshold'])) for r in grid]
    best=max(candidates)
    constant=max(r for r in candidates if r[-1]==0.)
    checks['selection_replays_validation_grid']=all(selection['selected'][name]['alpha']==r[-2] and selection['selected'][name]['confidence_threshold']==r[-1] for name,r in [('constant',constant),('query_confidence_gate',best)])
    checks['validation_membership_matches']=set(selection['validation_query_ids'])=={q for q in qids if d['splits'][q]=='val'}
    prior=read(folder/'verification.json')
    checks['input_caches_SHA_unchanged']=all(sha(ROOT/path)==digest for path,digest in prior['input_SHA256'].items())
    joint=Counter()
    for (a,_),(b,_) in zip(all_rankings['image'],all_rankings['original_fusion']):
        joint['both_correct' if a==1 and b==1 else 'fusion_rescue' if b==1 else 'fusion_harm' if a==1 else 'both_wrong']+=1
    checks['joint_counts_recomputed_match']=dict(joint)==prior['actual_joint_outcomes']
    checks={name:bool(value) for name,value in checks.items()}
    result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,rank_mismatches=mismatches,
        query_count=len(qids),reference_rows=len(d['rb']['ids']),methods=arrays.files,max_metric_error=float(max(metric_errors)),
        max_component_conservation_error=max(decomposition_errors),max_analytic_BM25_term_roundoff=max(term_errors),
        actual_joint_outcomes=dict(joint),source=str(folder),independent_implementation='numpy.lexsort then full SHA dedup scan')
    write(args.output,result)
    assert all(checks.values()), result
    print(result,flush=True)


if __name__=='__main__':
    main()
