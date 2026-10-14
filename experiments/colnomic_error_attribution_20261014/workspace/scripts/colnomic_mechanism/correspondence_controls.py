"""Descriptive all-query matching statistics and verified same-face controls.

Fixed positive AND negative references are held identical across failed/success
photos. Complete pair matrices are saved. Token cardinality is retained and
concentration is normalized; this remains association with view/scale confounds.
"""
from collections import Counter,defaultdict
import numpy as np
import torch
from scripts.colnomic_mechanism.common import ROOT,data,read,write,rows,csvout,sha
from scripts.colnomic_mechanism.cross_token_probe import cpu_relation,NAMES

BASE=ROOT/'results/colnomic_full_audit/20261013_resume'


def dot_pair(q,r):
    qi=torch.nonzero(q.float().norm(dim=1)>1e-8).flatten();ri=torch.nonzero(r.float().norm(dim=1)>1e-8).flatten()
    matrix=(q[qi].float()@r[ri].float().T).numpy()
    return matrix,qi.numpy(),ri.numpy()


def measure(matrix):
    vector=cpu_relation(matrix);return dict(zip(NAMES,map(float,vector)))


def main():
    torch.set_num_threads(2);d=data();folder=BASE/'correspondence_controls';folder.mkdir(exist_ok=True);(folder/'matrices').mkdir(exist_ok=True)
    qi={q:i for i,q in enumerate(d['qb']['ids'])};ri={r:i for i,r in enumerate(d['rb']['ids'])}
    dossiers=read(BASE/'full_tokens/dossiers_all.json');byq={r['photo_id']:r for r in dossiers}
    visual={r['photo_id']:r for r in read(BASE/'visual_consolidated/records.json')}
    controls=read(BASE/'visual_consolidated/same_face_success_controls.json');ledger={r['photo_id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    comparisons={q:read(BASE/f'full_tokens/queries/{q}/comparisons.json') for q in qi}
    reference_counts=np.array([int((z.float().norm(dim=1)>1e-8).sum()) for z in d['rb']['embeddings']]);cuts=np.unique(np.quantile(reference_counts,[.25,.5,.75]))
    overview=[]
    for qid,items in comparisons.items():
        for item in items:
            for side in ['positive','negative']:
                m=item[side+'_correspondence'];nq=m['query_visual_tokens'];nr=m['candidate_valid_tokens'];scene=visual[qid].get('scene_dimensions',{})
                overview.append(dict(photo_id=qid,package_id=byq[qid]['package_id'],group_id=byq[qid]['group_id'],split=byq[qid]['split'],outcome=byq[qid]['outcome'],
                    positive_reference_id=item['positive'],negative_reference_id=item['negative'],side=side,reference_id=item[side],
                    candidate_valid_tokens=nr,query_visual_tokens=nq,candidate_count_bin=int(np.searchsorted(cuts,nr,side='right')),
                    visual_target_coverage=m['target_visual_coverage'],normalized_concentration=m['target_concentration_hhi']*nr,
                    maximum_target_load=m['max_queries_per_target']/max(nq,1),near_tie_fraction=m['near_ties_within_1e_4']/max(nq,1),
                    expected_line_mismatch_fraction=m.get('differing_expected_ocr_line_tokens',0)/m['expected_ocr_line_evaluable_tokens'] if m.get('expected_ocr_line_evaluable_tokens') else None,
                    expected_line_evaluable_tokens=m.get('expected_ocr_line_evaluable_tokens'),geometric_evaluable_tokens=m.get('geometric_evaluable_tokens'),
                    geometry_distance_median_patches=m.get('geometry_distance_median_patches'),geometry_over_3_patch_fraction=m.get('geometry_over_3_patch_fraction'),
                    geometry_scope='historical SIFT transform, full query incl background; not revised OCR spatial scoring',
                    scene_dimensions=scene,observable_difficulties=visual[qid].get('observable_difficulties',[]),
                    packaging_face_relation='unknown; correspondence does not establish candidate face matching'))
    csvout(folder/'all_query_correspondence.csv',overview)
    bins=[]
    for outcome in sorted({r['outcome'] for r in overview}):
        for side in ['positive','negative']:
            for countbin in sorted({r['candidate_count_bin'] for r in overview}):
                subset=[r for r in overview if r['outcome']==outcome and r['side']==side and r['candidate_count_bin']==countbin]
                if not subset:continue
                for feature in ['visual_target_coverage','normalized_concentration','maximum_target_load','near_tie_fraction','expected_line_mismatch_fraction','geometry_distance_median_patches','geometry_over_3_patch_fraction']:
                    usable=[r for r in subset if r[feature] is not None]
                    if usable:bins.append(dict(outcome=outcome,side=side,candidate_count_bin=countbin,statistic=feature,
                        contrasts=len(subset),observed_contrasts=len(usable),queries=len({r['photo_id'] for r in usable}),identities=len({r['package_id'] for r in usable}),
                        median=float(np.median([r[feature] for r in usable])),mean=float(np.mean([r[feature] for r in usable])),
                        limitation='descriptive repetition; multiple positives/negatives not independent photos; count-bin control not causal'))
    csvout(folder/'candidate_count_strata.csv',bins)
    results=[];matrices={};maxerror=0;accepted_controls=set()
    for row in controls:
        qid=row['photo_id']
        for config,role in [('B03','image_strongest_wrong'),('F03','fusion_strongest_wrong')]:
            for sid in row.get(config,[]):
                negative=ledger[qid][role]
                assert byq[qid]['package_id']==byq[sid]['package_id']
                rankkey='image_positive_rank' if config=='B03' else 'fusion_positive_rank'
                assert byq[qid][rankkey]>1 and byq[sid][rankkey]==1
                accepted_controls.add((qid,sid))
                for positive in ledger[qid]['positive_reference_ids']:
                    record=dict(failure_photo=qid,success_photo=sid,configuration=config,package_id=byq[qid]['package_id'],group_id=byq[qid]['group_id'],
                        split=byq[qid]['split'],positive_reference=positive,negative_reference=negative,negative_source_role=role)
                    values={}
                    for source_id in [qid,sid]:
                        for label,rid in [('positive',positive),('negative',negative)]:
                            key=(source_id,rid)
                            if key not in matrices:
                                matrix,qix,rix=dot_pair(d['qb']['embeddings'][qi[source_id]],d['rb']['embeddings'][ri[rid]])
                                path=folder/'matrices'/f'{source_id}__{rid}.npz';np.savez_compressed(path,similarity=matrix,query_indices=qix,reference_indices=rix)
                                metrics=measure(matrix);error=abs(metrics['forward_top1_mean']-float(d['image']['scores'][qi[source_id],ri[rid]]))
                                maxerror=max(error,maxerror);assert error<2e-6
                                matrices[key]=dict(path=str(path.relative_to(folder)),SHA256=sha(path),query_valid_tokens=len(qix),reference_valid_tokens=len(rix),metrics=metrics)
                            values[(source_id,label)]=matrices[key]
                    assert values[(qid,'positive')]['reference_valid_tokens']==values[(sid,'positive')]['reference_valid_tokens']
                    assert values[(qid,'negative')]['reference_valid_tokens']==values[(sid,'negative')]['reference_valid_tokens']
                    for source_id,tag in [(qid,'failure'),(sid,'success')]:
                        pm=values[(source_id,'positive')];nm=values[(source_id,'negative')]
                        record[tag+'_query_valid_tokens']=pm['query_valid_tokens'];record['positive_reference_valid_tokens']=pm['reference_valid_tokens'];record['negative_reference_valid_tokens']=nm['reference_valid_tokens']
                        record[tag+'_image_gap']=pm['metrics']['forward_top1_mean']-nm['metrics']['forward_top1_mean']
                        for label,value in [('positive',pm),('negative',nm)]:
                            for metric in ['forward_assignment_entropy','forward_maximum_target_load','forward_target_coverage','forward_effective_target_fraction','forward_mutual_nearest_fraction','forward_top1_minus_top2_mean']:
                                record[tag+'_'+label+'_'+metric]=value['metrics'][metric]
                    record['gap_success_minus_failure']=record['success_image_gap']-record['failure_image_gap']
                    record['residual_confounding']='same verified identity/face and fixed reference token counts; camera angle, clarity, scale, visible content and query cardinality not matched'
                    results.append(record)
    csvout(folder/'fixed_candidate_same_face_controls.csv',results)
    matrix_index=[dict(photo_id=qid,reference_id=rid,**value) for (qid,rid),value in matrices.items()];write(folder/'matrix_index.json',matrix_index)
    intervals=[]
    for config in ['B03','F03']:
        selected=[r for r in results if r['configuration']==config]
        for key in ['gap_success_minus_failure','positive_forward_assignment_entropy','positive_forward_target_coverage','positive_forward_maximum_target_load']:
            values=defaultdict(list)
            for r in selected:
                value=r[key] if key=='gap_success_minus_failure' else r['success_'+key]-r['failure_'+key]
                values[r['group_id']].append(value)
            groupmeans=np.array([np.mean(v) for v in values.values()])
            if not len(groupmeans):continue
            draw=np.random.default_rng(20261014).integers(len(groupmeans),size=(5000,len(groupmeans)));boot=groupmeans[draw].mean(1)
            intervals.append(dict(configuration=config,contrast=key,source_groups=len(groupmeans),identities=len({r['package_id'] for r in selected}),
                failed_queries=len({r['failure_photo'] for r in selected}),pairs=len(selected),group_macro_mean=float(groupmeans.mean()),
                ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),seed=20261014,iterations=5000,
                scope='selected confirmed controls, not random population; group-block CI does not remove selection/quality confounds'))
    csvout(folder/'control_group_bootstrap.csv',intervals)
    assert len(overview)==2*sum(map(len,comparisons.values())) and len(comparisons)==1117
    write(folder/'validation.json',dict(status='passed',queries=len(comparisons),contrast_rows=len(overview),confirmed_control_photo_pairs=len(accepted_controls),
        configuration_contrasts=len(results),selected_matrices=len(matrices),maximum_cached_score_error=maxerror,reference_count_cuts=cuts.tolist(),
        fixed_reference_counts_checked=True,method_success_verified=True,source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in [
            BASE/'visual_consolidated/same_face_success_controls.json',BASE/'visual_consolidated/records.json',BASE/'full_tokens/dossiers_all.json']},
        interpretation='line crossings/concentration/residuals may occur in correct retrieval; no automatic mechanism conclusion from these descriptive statistics'))
    write(folder/'protocol.json',dict(scope='all1117 exported correspondence and all actually confirmed same-face successful photos per configuration',
        selection='use every independently reviewed same-face success; original failed query positives and actual strongest-negative reference fixed for both photographs',
        query_region='complete original valid tokens including templates for new fixed-pair matrices; historical descriptive fields may restrict query visual tokens',
        normalization='target mass shared for ties, entropy/log cardinality, coverage/candidate cardinality; retain exact candidate/query counts',
        no_anomaly_threshold_or_mechanism_inference=True))
    print('Correspondence controls',len(comparisons),len(results),len(matrices),'passed',flush=True)


if __name__=='__main__':main()
