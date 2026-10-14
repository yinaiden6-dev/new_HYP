"""Link observed multilabel difficulties to all35 repeated predictions.

No labels or mechanisms are inferred from method success. Shared failure is
literal all35 top1 failure; oracle and 'B03-only' comparison scopes are explicit.
"""
from collections import Counter,defaultdict
import numpy as np
from scripts.colnomic_mechanism.common import ROOT,read,write,rows,csvout,sha
BASE=ROOT/'results/colnomic_full_audit/20261013_resume'


def main():
    folder=BASE/'difficulty_config';folder.mkdir(exist_ok=True)
    records=read(BASE/'visual_consolidated/records.json');v={r['photo_id']:r for r in records};assert len(v)==1117
    original=rows(BASE/'scene35_reviewed/prediction_matrix.csv');pred=defaultdict(dict)
    for r in original:
        assert r['photo_id'] in v and r['config_id'] not in pred[r['photo_id']]
        pred[r['photo_id']][r['config_id']]=None if r['prediction_available']!='True' else r['correct']=='True'
    configs=sorted({r['config_id'] for r in original});assert len(configs)==35 and all(set(a)==set(configs) for a in pred.values())
    specs=read(BASE/'scene35_reviewed/configuration_semantics.json');image=[r['config_id'] for r in specs if r['mode']=='image' and r['query_input']=='RGB full original']
    alternatives=[c for c in image if c!='B03'];assert alternatives and 'B03' in image
    cohort=[]
    for qid,p in pred.items():
        available=[c for c in configs if p[c] is not None];success=[c for c in available if p[c]]
        cohort.append(dict(photo_id=qid,package_id=v[qid]['package_id'],group_id=v[qid]['group_id'],split=v[qid]['split'],
            available_configurations=len(available),successful_configurations=success,
            all35_fail=len(available)==35 and not success,any_method_oracle_correct=bool(success),
            B03_fail=p['B03'] is False,F03_fail=p['F03'] is False,fusion_introduced=p['B03'] is True and p['F03'] is False,
            B03_fail_others_all_original_image_correct=p['B03'] is False and all(p[c] is True for c in alternatives),
            B03_fail_at_least_one_original_image_correct=p['B03'] is False and any(p[c] is True for c in alternatives),
            observable_difficulties=v[qid]['difficulty_labels'],raw_observable_tags=v[qid].get('observable_difficulties',[]),scene_dimensions=v[qid].get('scene_dimensions',{}),
            evidence_status=v[qid]['review_status'],fine_detail_fully_verified=v[qid].get('fine_detail_verified',False),
            token_evidence=f'../full_tokens/queries/{qid}/index.html'))
    csvout(folder/'per_photo_intersections.csv',cohort)
    tags=sorted({t for r in records for t in r['difficulty_labels']})
    metrics=[];cross=[];totalfail={c:sum(p[c] is False for p in pred.values()) for c in configs};bootstrap=[]
    for tag in ['__all__']+tags:
        qids=sorted(q for q in v if tag=='__all__' or tag in v[q]['difficulty_labels']);den=len(qids)
        for config in configs:
            available=[q for q in qids if pred[q][config] is not None];failed=[q for q in available if not pred[q][config]]
            metrics.append(dict(difficulty=tag,configuration=config,photos=den,available_predictions=len(available),
                failed=len(failed),identities=len({v[q]['package_id'] for q in qids}),failed_identities=len({v[q]['package_id'] for q in failed}),
                within_difficulty_failure_rate=len(failed)/len(available) if available else None,
                fraction_of_all_configuration_errors=len(failed)/totalfail[config] if totalfail[config] else None,
                observed_difficulty_occurrence_rate=den/1117,denominator_independent_queries=1117,
                labels_multilabel=True,review_scope='actual overview; fine words not all transcribed; difference labels are not causes'))
            if config=='B03':continue
            paired=[q for q in qids if pred[q]['B03'] is not None and pred[q][config] is not None]
            repaired=sum(not pred[q]['B03'] and pred[q][config] for q in paired);harmed=sum(pred[q]['B03'] and not pred[q][config] for q in paired)
            cross.append(dict(difficulty=tag,configuration=config,photos=len(paired),repair_B03_errors=repaired,harm_B03_successes=harmed,net=repaired-harmed,
                both_fail=sum(not pred[q]['B03'] and not pred[q][config] for q in paired),both_correct=sum(pred[q]['B03'] and pred[q][config] for q in paired),
                identities=len({v[q]['package_id'] for q in paired}),scope='repeated evaluation, not independent method observations or mechanism proof'))
            groups=defaultdict(list)
            for q in paired:groups[v[q]['group_id']].append(int(pred[q][config])-int(pred[q]['B03']))
            if not groups:continue
            # Cluster resampling retains unequal numbers of photographs/group.
            means=np.array([sum(g) for g in groups.values()]);counts=np.array([len(g) for g in groups.values()]);rng=np.random.default_rng(20261014)
            chosen=rng.integers(len(groups),size=(1000,len(groups)));boot=means[chosen].sum(1)/counts[chosen].sum(1)
            bootstrap.append(dict(difficulty=tag,configuration=config,source_groups=len(groups),queries=len(paired),net_accuracy_difference=(repaired-harmed)/len(paired),
                ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),seed=20261014,rounds=1000,
                limitation='paired cluster association with observed multilabel difficulty; no adjustment for identity/face/quality confounding; many exploratory comparisons'))
    csvout(folder/'difficulty_configuration_rates.csv',metrics);csvout(folder/'difficulty_method_transitions.csv',cross);csvout(folder/'source_group_paired_bootstrap.csv',bootstrap)
    intersections=[]
    for name in ['all35_fail','fusion_introduced','B03_fail_others_all_original_image_correct','B03_fail_at_least_one_original_image_correct']:
        subset=[r for r in cohort if r[name]]
        for tag in tags:
            included=[r for r in subset if tag in r['observable_difficulties']]
            intersections.append(dict(comparison_scope=name,difficulty=tag,observed=len(included),subset_photos=len(subset),
                fraction_within_subset=len(included)/len(subset) if subset else None,identities=len({r['package_id'] for r in included}),
                selection='literal outcome intersection; all existing overview labels included independent of hypothesis'))
    csvout(folder/'shared_specific_failure_difficulties.csv',intersections)
    summary=dict(status='passed',unique_queries=len(v),configurations=len(configs),repeated_predictions=len(original),
        missing_predictions=sum(p is None for a in pred.values() for p in a.values()),original_image_configs=image,
        comparison_scopes={name:dict(queries=sum(r[name] for r in cohort),identities=len({r['package_id'] for r in cohort if r[name]})) for name in [
            'all35_fail','any_method_oracle_correct','fusion_introduced','B03_fail_others_all_original_image_correct','B03_fail_at_least_one_original_image_correct']},
        difficulty_labels=tags,source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in [BASE/'visual_consolidated/records.json',BASE/'visual_consolidated/difficulty_taxonomy.json',BASE/'scene35_reviewed/prediction_matrix.csv',BASE/'scene35_reviewed/configuration_semantics.json']},
        interpretation='oracle is not deployable; comparative B03 failure not a causal encoder explanation; overlapping observed labels cannot sum to photo totals')
    assert len(cohort)==1117 and all(r['both_fail']+r['both_correct']+r['repair_B03_errors']+r['harm_B03_successes']==r['photos'] for r in cross)
    write(folder/'validation.json',summary)
    lines=['# 可观察困难与35配置交集','',f'1117张独立图，{len(original)}次重复评测；原图image对照按已核实的输入合同选择：'+', '.join(image)+'。',
        '',json_block(summary['comparison_scopes']),'',
        'all35_fail 表示35种设置均Top1错误；有一个方法正确的oracle需要真值，不能当作系统成绩。B03特有的两个字段严格限定为相对于列出的原图image设置，并不证明ColNomic某层缺陷。',
        '', '逐配置/困难表分别保存类内失败率、该配置错误构成、困难发生率；多标签允许重叠。source-group配对bootstrap对查询聚类，但未消除身份、可见面、清晰度和尺度混杂。',
        '', '同一场景中的不同差额来源、跨场景重复的融合错误及匹配/汇总响应，沿每图Token链接和总主档案追溯；无法识别机制的照片保留未知。']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Difficulty/configuration analysis',len(cohort),len(metrics),len(cross),'passed',flush=True)


def json_block(value):
    import json
    return '```json\n'+json.dumps(value,ensure_ascii=False,indent=2)+'\n```'


if __name__=='__main__':main()
