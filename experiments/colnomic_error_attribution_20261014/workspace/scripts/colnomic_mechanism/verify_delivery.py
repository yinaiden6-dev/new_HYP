"""Final independently computed acceptance for the assembled analysis.

Archive integrity is a later external SHA/CRC/extracted-workspace receipt. This
validator cannot certify causal explanations or a human transcription of every
word; numerical, overview, details and semantic ROI coverage remain separate.
"""
from collections import Counter
from pathlib import Path
import json,re
from urllib.parse import urlsplit,unquote
import numpy as np
from scripts.colnomic_mechanism.common import ROOT,data,read,write,rows,sha,workspace_path
from scripts.colnomic_mechanism.package_full_audit import Links
BASE=ROOT/'results/colnomic_full_audit/20261013_resume'


def assembled_links(folder):
    """Check assembled links against actual files, including report Markdown.

    Archive verification separately checks targets against archive membership.
    This avoids certifying a target simply because a source string lists it.
    """
    checked=[];broken=[]
    paths=[p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.html','.md']]
    paths += list((folder/'cases').glob('*/dossier.json'))
    paths += [p for p in [BASE/'REPORT.md',BASE/'index.html'] if p.is_file()]
    for path in paths:
        if path.suffix=='.html':
            parser=Links();parser.feed(path.read_text());targets=parser.links
        elif path.suffix=='.md':targets=re.findall(r'\]\(([^)]+)\)',path.read_text())
        else:targets=list(read(path).get('links',{}).values())
        for target in targets:
            target=str(target).strip('<>');url=urlsplit(target)
            if url.scheme or url.netloc or not url.path or '${' in target:continue
            resolved=(path.parent/unquote(url.path)).resolve()
            row=dict(source=str(path.relative_to(ROOT)),target=target)
            if not resolved.is_relative_to(ROOT) or not resolved.exists():broken.append(row)
            else:checked.append(row)
    return dict(status='passed' if not broken else 'failed',checked_relative_links=len(checked),
        broken_relative_links=broken,scope='actual assembled HTML/Markdown/dossier targets within workspace; archive membership independently verified after sealing')


def main():
    d=data();folder=BASE/'delivery';items=read(folder/'dossiers_all1117.json');byid={r['photo_id']:r for r in items}
    assert len(items)==len(byid)==1117 and list(byid)==d['qb']['ids']
    original=read(BASE/'full_tokens/dossiers_all.json');outcomes=Counter();visual=read(BASE/'visual_consolidated/records.json');vr={r['photo_id']:r for r in visual}
    assert len(vr)==1117 and all(r['candidate_overview_reviewed'] for r in visual)
    prediction=rows(BASE/'scene35_reviewed/prediction_matrix.csv');keys={(r['photo_id'],r['config_id']) for r in prediction}
    assert len(keys)==len(prediction)==1117*35 and {q for q,c in keys}==set(byid)
    methods={c for q,c in keys};assert len(methods)==35 and all((q,c) in keys for q in byid for c in methods)
    expected_predictions={q:[] for q in byid}
    for row in prediction:expected_predictions[row['photo_id']].append(row)
    sourcechecks=0
    for i,item in enumerate(items):
        qid=item['photo_id'];old=original[i];assert qid==old['photo_id'] and item['package_id']==d['photos'][qid]['package_id']
        assert item['split']==d['splits'][qid] and item['all35_predictions']==expected_predictions[qid]
        assert item['visual_review']==vr[qid] and item['observed_difficulties']==vr[qid]['difficulty_labels']
        assert sha(workspace_path(item['original_path']))==d['qb']['sha256'][i];sourcechecks+=1
        assert (folder/'cases'/qid/'index.html').is_file() and read(folder/'cases'/qid/'dossier.json')==item
        outcomes[item['outcome']]+=1
    # Check every gallery image against its original inventory; cached tensors
    # and metadata include original hashes, without treating filenames as SHA.
    references=rows(ROOT/'dataset/lists/references.csv');assert len(references)==5405
    for row in references:
        assert sha(ROOT/'dataset'/row['path'])==row['sha256'];sourcechecks+=1
    summary=read(folder/'summary.json');assert summary['objective_outcomes']==dict(outcomes)
    assert not summary['pending_cross_token_relation'] and not summary['pending_current_semantic_regions']
    count=Counter(tag for item in items for tag in set(item['observed_difficulties']));assert summary['observable_difficulty_counts']==dict(count)
    assert sum(outcomes.values())==1117 and sum(v is None for row in prediction for v in [row['correct_rank']])==0
    for path,digest in read(folder/'source_manifest.json')['sources'].items():assert sha(ROOT/path)==digest
    receipts={
        'original_tokens_all_matrices':'delivery_verification/full_tokens/independent_validation.json',
        'original_BM25_fusion':'checkpoint_2108/fusion_independent_validation.json',
        'original_geometry_ablation':'split_geometry/validation.json',
        'reviewed_geometry':'reviewed_boxes/validation.json',
        'weighted_current_boxes':'checkpoint_2108/weighted_reviewed_independent_validation.json',
        'pure_spatial_matching_aggregation':'weighted_spatial_only_reviewed/independent_verification.json',
        'pooled_set_full_gallery':'checkpoint_2108/set_probes_independent_validation.json',
        'signed_layout_all_qualified':'completion_audit/signed_layout_independent_validation.json',
        'current_position_heads_nulls':'positions_reviewed/independent_validation.json',
        'OCR_content_position_correlation':'position_content_control/validation.json',
        'natural_shared_phrase_features':'natural_shared_phrase_layout/independent/validation.json',
        'complete_cross_token_heads':'cross_token_relation/independent_validation.json',
        'cross_token_training_input_join':'cross_token_relation/restricted_join_independent_validation.json',
        'controlled_details':'detail_interventions/analysis/independent_validation.json',
        'crossview_layout_pilot':'controlled_crossview_layouts/analysis/independent_validation.json',
        'reviewed_crops':'reviewed_crops/analysis/independent_validation.json',
        'current_regions':'reviewed_regions/validation.json',
        'current_query_region_summaries':'reviewed_regions/query_summary_independent_validation.json',
        'literal35_predictions_scenarios':'scene35_reviewed/validation.json',
        'confirmed_face_token_controls':'correspondence_controls/independent_validation.json',
        'difficulty35_intersections':'difficulty_config/independent_validation.json',
        'foreground_harm_details':'foreground_harms_review/validation.json',
        'original_scene_and_overview_sources':'visual_consolidated/validation.json'}
    receipts['delivered_UI_exact_filters']='delivery/ui_validation.json'
    receipts['full_one_click_frozen_entry']='reproduction_support/final_full_verification.json'
    receipts['fresh_reference_intervention_candidates']='fresh_reference_interventions/independent/validation.json'
    receipts['fresh_reference_identity_bootstrap']='fresh_reference_interventions/interpretation_independent/validation.json'
    receipts['fresh_reference_feature_changes']='fresh_reference_interventions/feature_bridge/validation.json'
    receipts['final_combined_entry_quick']='reproduction_support/final_dispatcher_quick_validation.json'
    evidence={}
    for name,relative in receipts.items():
        path=BASE/relative;record=read(path)
        if name=='literal35_predictions_scenarios':
            inventory=record['inventory'];assert inventory['matrix_shape']==[1117,35]
            assert inventory['evaluation_occurrences']==39095 and inventory['missing_prediction_rows']==0
            assert inventory['original_reviewed_photos']==118 and inventory['reviewed_count_matches_historical']
            assert len(record['score_replays'])==35
            assert all(r['status']=='passed' and r['shape']==[1117,5405] and r['verified_predictions']==1117
                for r in record['score_replays'].values())
            assert len(record['historical_table_comparisons'])==8
            assert all(r['matching'] and not r['differences'] for r in record['historical_table_comparisons'].values())
            assert all(r['matching'] for r in record['historical_output_hashes'].values())
            assert record['matrix_missing_mask_consistent']
        elif name=='controlled_details':
            assert record['passed'] is True and record['matrix_files']==145
            assert record['matrix_SHA_and_axes_checked']==record['matrix_files']
            assert record['max_matrix_score_error']<2e-6 and record['max_pair_gap_error']<2e-6
            assert record['max_region_gap_error']<2e-6 and record['max_switch_partition_error']<2e-6
        elif name=='original_scene_and_overview_sources':
            def all_asserted(value):
                if isinstance(value,dict):return bool(value) and all(all_asserted(v) for v in value.values())
                return value is True
            assert all_asserted(record),record
        else:
            assert record['status']=='passed',(name,record.get('status'))
            if name=='full_one_click_frozen_entry':
                assert record['scope']=='full_frozen_numerical_verification'
                assert len(record['module_verification'])==10 and all(r['status']=='passed' for r in record['module_verification'])
                assert record['entry_sha256']==sha(ROOT/'scripts/colnomic_mechanism/reproduce_full_audit.py')
            if name=='fresh_reference_intervention_candidates':
                assert record['query_parity_rows']==record['query_base_bitwise_exact']==12
                assert record['reference_parity_rows']==28 and record['new_encoding_records']==40
                assert record['full_pointwise_matrices']==290 and record['fresh_matrices']==record['frozen_matrices']==145
                assert record['pair_arm_rows']==170 and record['region_rows']==850 and record['fresh_frozen_comparisons']==85
                assert record['max_matrix_error']==0 and record['max_scalar_conservation_error']<2e-6
            if name=='fresh_reference_feature_changes':
                assert record['query_arms']==60 and record['identities']==12 and record['replayed_feature_token_rows']==63575
                assert record['max_L2_error']<2e-6 and record['max_cosine_error']<2e-6
        evidence[name]=dict(path=relative,SHA256=sha(path),numerical_or_semantic_scope=record.get('scope',record.get('interpretation','see individual verifier receipt')))
    token=read(BASE/receipts['original_tokens_all_matrices']);assert token['queries_checked']==1117 and token['checks']['pointwise_matrices_replayed']==token['checks']['full_fp32_matrices_checked']
    assert token['maximum_errors']['token_gap_conservation_error']<2e-7 and token['maximum_errors']['region_gap_conservation_error']<2e-7
    region=read(BASE/'reviewed_regions/localization_coverage.json');coverage=read(BASE/'visual_consolidated/coverage.json')
    assert coverage['candidate_overview_reviewed']==1117 and coverage['original_images_sha_verified']==1117
    foreground=read(BASE/'foreground_harms_review/manual_reviews.json')['records'];assert len(foreground)==8
    assert all(r['actual_original_crop_and_GT_negative_detail_view_completed'] and not r['causal_mechanism_claimed'] for r in foreground)
    changed=read(BASE/'manual_box_review/box_revisions_final.json')['changed_records'];actual_changed=[r for r in changed if r['before'].get('bbox')!=r['after'].get('bbox')];assert len(actual_changed)==5
    intervention_ids={}
    for branch in ['symmetric_layouts','detail_interventions']:
        source=read(BASE/branch/'sources.json');intervention_ids[branch]=sorted({r['package_id'] for r in source})
    intervention_ids['controlled_crossview_layouts']=sorted({r['package_id'] for r in read(BASE/'controlled_crossview_layouts/encoding_manifest.json')})
    scene_checks=read(BASE/'scene35_reviewed/final_scene_statistics/validation.json')
    assert len(scene_checks)==7 and all(value is True for value in scene_checks.values()),scene_checks
    evidence['final_audited_scene_statistics']=dict(path='scene35_reviewed/final_scene_statistics/validation.json',
        SHA256=sha(BASE/'scene35_reviewed/final_scene_statistics/validation.json'),actual_checks=scene_checks)
    # The report links its own receipt. Until all links pass, this file records
    # a pending validation, and never claims acceptance prematurely.
    write(folder/'acceptance.json',dict(status='verification_in_progress',queries_checked=1117))
    links=assembled_links(folder);write(folder/'link_validation.json',links)
    assert not links['broken_relative_links'],links['broken_relative_links'][:5]
    testlog=(BASE/'final_tests.log').read_text();matched=re.search(r'(\d+) passed in',testlog)
    assert matched and int(matched.group(1))>=56 and not re.search(r'\d+ (failed|errors?)',testlog)
    protocol=read(ROOT/'results/token_mechanism_experiments/20261013_v2/preregistration.json')
    limitations=[
        '1117原图和关键候选overview已实际审核；细字、可见面边界或身份信息无法识别的范围保留未知，overview不等于全字转录',
        '原测试集已观察；包装/实际设计/来源group/hash隔离已验，不额外声称未定义的设计族或形态家族隔离',
        f'{region["manual_semantic_queries"]}张重点图完成语义ROI定位；全量区域分区守恒但绝大多数身份关键数字/布局/图案ROI仍未定位',
        '同面成功控制仅为确认子集，剩余未确认不等于不存在；角度/清晰度/主体大小与控制复用残留混杂',
        '自然共享四短语不等于全文同文字/布局单因素，ROI有外部坐标选择和整图上下文影响；测试8身份、区间宽',
        '跨视角词位置pilot仅3身份，2train/1val/0test，不能形成held-test布局Probe结论',
        '旧图库冻结缓存可忠实评分；缺少旧processor/中间层/上游文件，唯一重新编码漂移原因尚未查明',
        '手工修订框、裁切和遮挡改变多个因素；新Query对旧图库条件排名不当作新编码器全图库准确率',
        '28新参考与12重新编码base及60兼容编辑臂作完整预选候选对照；worker源码SHA不同原因未知，新旧相同ordinal轴不证明物理格点相同',
        'E0-E7与人工困难规范化是未校准分类，不与已经验证的机制混同；OCR纵向序号非真实阅读行',
        '完成全量审计与可执行实验不表示为全部错误确定唯一原因；最终Token不能归因具体编码层',
        '归档整包SHA/CRC及实际解包的验证由封存后的外部delivery_support receipt给出，避免把自引用SHA写入自身归档']
    result=dict(status='analysis_accepted',queries=1117,references=5405,distinct_SHA_designs=len(set(d['rb']['sha256'])),
        independent_prediction_matrix_shape=[1117,35],repeated_predictions=39095,original_query_and_reference_image_SHA_checks=sourcechecks,
        actual_box_coordinate_revisions=len(actual_changed),numeric_receipts=evidence,split_checks=d['split_verification'],
        coverage=dict(data_queries=1117,visual_overview=coverage['candidate_overview_reviewed'],scene_by_dimension=coverage['scene_audited_by_dimension'],
            scene_known_by_dimension=coverage['scene_known_by_dimension'],explicit_limited_detail_verified=coverage['explicit_fine_detail_verified'],
            selected_detail_observation_records=coverage['selected_detail_observation_records'],manual_semantic_ROI_queries=region['manual_semantic_queries'],
            token_queries=1117,token_contrasts=token['checks']['comparison_conservation_checks'],complete_key_token_matrices=token['checks']['pointwise_matrices_replayed'],
            intervention_identities_by_experiment={k:len(v) for k,v in intervention_ids.items()},
            unique_intervention_identities=len(set().union(*map(set,intervention_ids.values()))),
            deterministic_score_evidence_queries=1117,unique_causal_failure_mechanism_coverage='not identifiable from this audit; do not replace with1117'),
        tests=dict(passed=int(matched.group(1)),log_SHA256=sha(BASE/'final_tests.log')),relative_links_verified=links['checked_relative_links'],limitations=limitations,
        task_specification_SHA256=sha(BASE/'checkpoint_2108/TASK_SPEC_latest.txt'),verifier_SHA256=sha(__file__),
        originals_and_history='original images/cache/predictions unchanged; only user-authorized annotation revisions with before/after snapshots',
        archive_integrity_scope='separate final external archive-validation receipt; analysis acceptance does not assert a not-yet-built archive is valid')
    write(folder/'acceptance.json',result)
    print('Final analysis acceptance passed',sourcechecks,'image SHA checks',len(evidence),'receipts',flush=True)


if __name__=='__main__':main()
