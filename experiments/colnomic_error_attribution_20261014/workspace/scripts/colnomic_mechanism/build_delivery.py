"""Join independently produced evidence into a browsable Chinese audit delivery.

Visual labels, deterministic score facts, and experimental responses remain
separate. Historical artifacts are read-only; incomplete work stays explicit.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path

from scripts.colnomic_mechanism.common import ROOT, read, write, rows, csvout, sha, workspace_path

OUTCOME_ZH = {'both_correct':'图像/融合都正确','fusion_rescued':'图像错、融合对',
              'fusion_harmed':'图像对、融合错','both_wrong':'图像/融合都错误'}
CSS = 'body{font:16px system-ui;color:#172033;margin:24px;max-width:1600px}table{border-collapse:collapse;width:100%}td,th{padding:7px;border:1px solid #ccd;font-size:13px;vertical-align:top}th{background:#edf2ff}img{max-width:100%}.notice{background:#fff3d7;padding:14px}a{color:#1646ad}select,input{margin:6px;padding:7px}details{margin:14px 0}pre{white-space:pre-wrap}'
FILTER_SCRIPT = """
const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
function render(){
 const get=id=>document.querySelector('#'+id).value;
 const search=get('search').toLowerCase(),outcome=get('outcome'),split=get('split'),config=get('config'),correct=get('correct');
 const shown=data.filter(r=>(!outcome||r.outcome===outcome)&&(!split||r.split===split)&&
   ['background','placement','occlusion'].every(k=>!get(k)||r.scene[k]===get(k))&&
   (!config||!correct||(r.configs[config].available&&r.configs[config].correct===(correct==='correct')))&&
   JSON.stringify(r).toLowerCase().includes(search));
 document.querySelector('#count').textContent=shown.length+' / '+data.length+' 张';
 document.querySelector('#body').innerHTML=shown.map(r=>'<tr><td><a href="cases/'+esc(r.photo_id)+'/index.html">'+esc(r.photo_id)+
  '</a></td><td>'+esc(r.package_id)+' / '+esc(r.split)+'</td><td>'+r.image_rank+' / '+r.fusion_rank+
  (config?'；'+esc(config)+' rank='+esc(r.configs[config].rank):'')+'</td><td>'+esc(r.outcome)+'</td><td>'+esc(JSON.stringify(r.scene))+
  '</td><td>'+esc(r.labels.join('，'))+'</td><td>'+esc(r.review)+'</td></tr>').join('');
}
for(const el of document.querySelectorAll('input,select'))el.addEventListener('input',render);render();</script>
"""


def opt(path, default=None):
    return read(path) if path.is_file() else default


def optrows(path):
    return rows(path) if path.is_file() else []


def h(value):
    return html.escape(str(value))


def table_html(items, fields):
    return '<table><thead><tr>'+''.join('<th>'+h(f)+'</th>' for f in fields)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+h(r.get(f,''))+'</td>' for f in fields)+'</tr>' for r in items)+'</tbody></table>'


def mdtable(items, fields):
    def cell(v): return str(v).replace('|','/').replace('\n',' ')
    return '|'+ '|'.join(fields)+'|\n|'+ '|'.join('---' for _ in fields)+'|\n'+''.join(
        '|'+ '|'.join(cell(r.get(f,'')) for f in fields)+'|\n' for r in items)


def method_rank_index(path):
    result=defaultdict(list)
    for r in optrows(path): result[r['photo_id']].append(r)
    return result


def build(source, folder):
    folder.mkdir(parents=True,exist_ok=True)
    dossiers=read(source/'full_tokens/dossiers_all.json'); ids=[r['photo_id'] for r in dossiers]
    assert len(ids)==len(set(ids))==1117
    visual_rows=opt(source/'visual_consolidated/records.json',[])
    visuals={r['photo_id']:r for r in visual_rows}; assert len(visuals)==len(visual_rows)
    controls={r['photo_id']:r for r in opt(source/'visual_consolidated/same_face_success_controls.json',[])}
    scene_folder=source/'scene35_reviewed' if (source/'scene35_reviewed/validation.json').is_file() else source/'scene35'
    predictions=rows(scene_folder/'prediction_matrix.csv'); byphoto=defaultdict(list)
    for r in predictions: byphoto[r['photo_id']].append(r)
    assert set(byphoto)==set(ids) and all(len(v)==35 for v in byphoto.values())
    fusion=defaultdict(list)
    for r in rows(source/'full_fusion/all_query_decomposition.csv'):fusion[r['photo_id']].append(r)
    geometry=method_rank_index(source/'reviewed_boxes/per_photo_actual_scores.csv')
    factorial=method_rank_index(source/'weighted_spatial_only_reviewed/per_photo_factorial_transitions.csv')
    revised={r['photo_id']:r for r in opt(source/'reviewed_crops/analysis/changed_query_results.json',[])}
    changed_boxes={r['photo_id']:r for r in read(source/'manual_box_review/box_revisions_final.json')['changed_records']
                   if r['before'].get('bbox')!=r['after'].get('bbox')}
    intervention=defaultdict(list)
    for branch in ['symmetric_layouts','detail_interventions']:
        for r in opt(source/branch/'sources.json',[]):
            qid=r.get('photo_id',r.get('source_id',r.get('id')))
            if qid:intervention[qid].append(dict(experiment=branch,source=r))
    for r in opt(source/'controlled_crossview_layouts/encoding_manifest.json',[]):
        if r['arm']=='base':intervention[r['source_id']].append(dict(experiment='controlled_crossview_layouts',source=r,scope='3-identity pilot; no held-test group'))
    matrices=read(source/'full_tokens/matrix_index_all.json')
    fresh_reference=method_rank_index(source/'fresh_reference_interventions/analysis/pair_effects.csv')
    token_validation=opt(source/'delivery_verification/full_tokens/independent_validation.json') or read(source/'full_tokens/independent_validation.json')
    foreground={r['photo_id']:r for r in opt(source/'foreground_harms_review/manual_reviews.json',{}).get('records',[])}
    masters=[]; evidence_counts=Counter(); candidate_rows=[]; scene_config_rows=[]
    for original in dossiers:
        qid=original['photo_id']; v=visuals.get(qid)
        item=dict(original)
        item['frozen_token_export_visual_fields_are_historical']=True
        item['visual_review']=v
        item['visual_review_status']=v.get('review_status') if v else 'missing_actual_visual_review'
        item['direct_observations']=v.get('direct_observations',[]) if v else []
        item['observed_difficulties']=v.get('difficulty_labels',[]) if v else []
        item['raw_observable_tags']=v.get('observable_difficulties',[]) if v else []
        item['visible_face']=v.get('visible_face') if v else None
        item['identity_information_visible']=v.get('identity_information_visible') if v else None
        item['same_face_success_control']=controls.get(qid)
        item['scene_labels']=v.get('scene_labels',{}) if v else {}
        item['scene_dimensions']=v.get('scene_dimensions',{}) if v else {}
        item['fusion_decomposition']=fusion[qid]
        item['scoring_intervention_responses']=geometry[qid]
        item['matching_aggregation_factorial_responses']=factorial[qid]
        item['controlled_image_experiments']=intervention[qid]
        item['fresh_reference_intervention_responses']=fresh_reference[qid]
        item['manual_box_revision']=changed_boxes.get(qid)
        item['D01_crop_sensitivity']=revised.get(qid)
        item['foreground_identity_region_review']=foreground.get(qid)
        item['historical_subject_regions_and_crop_scope']='pre-user-review box snapshot; use reviewed_regions for current partition' if qid in changed_boxes else 'unchanged box coordinates'
        item['all35_predictions']=byphoto[qid]
        item['data_and_reference_validity']={'metadata_mapping':'verified',
            'physical_identity_label':v.get('identity_ambiguity','not_visually_adjudicated') if v else 'not_visually_adjudicated',
            'original_ground_truth_preserved_for_metrics':True}
        mechanism=[{'factor':'匹配和贡献','level':'确定性分解',
            'evidence':f'full_tokens/queries/{qid}/comparisons.json',
            'claim':'Token Δ均值复现原始正负差额；仅定位差额来源，不证明编码层或语义机制。'}]
        if original['outcome']=='fusion_harmed':
            mechanism.append({'factor':'融合','level':'确定性评分阶段定位','evidence':'full_fusion/all_query_decomposition.csv',
                'claim':'图像候选原本正确；按历史公式加入文字后错误。正负差额分解可检查文字项是否压过图像优势；OCR为何偏向错误仍需视觉证据。'})
        if revised.get(qid):
            r=revised[qid]
            mechanism.append({'factor':'主体定位/输入','level':'受控裁切评分响应','evidence':'reviewed_crops/analysis/changed_query_results.json',
                'claim':f"同一冻结图库下D01 rank {r['before_D01_rank']}→{r['after_D01_rank']}；人工定位、背景、尺度、保留内容同时变化。"})
        if intervention[qid]:
            mechanism.append({'factor':'受控图像编辑','level':'像素干预响应，限定实验范围',
                'evidence':[x['experiment'] for x in intervention[qid]],
                'claim':'合成词位置或数字区域遮挡的响应；不自动等于自然包装身份机制。'})
        if fresh_reference[qid]:
            mechanism.append({'factor':'编码运行时/受控候选','level':'新参考与兼容编辑臂的受限配对响应',
                'evidence':'fresh_reference_interventions/analysis/pair_effects.csv',
                'claim':'12原图逐位复现桥接兼容冻结编辑臂；全部预选参考重新编码，分别保留新旧参考的edit-minus-sham及贡献。不是完整图库新运行时精度；不由Token长度相同推物理格点相同。'})
        item['mechanism_evidence']=mechanism
        item['alternative_explanations']=list(dict.fromkeys(original.get('alternative_explanations',[])+[
            '可见包装面/分辨率/身份标签仍可能影响对比；跨方法成功不直接定位编码器原因。',
            '最终Token统计不能区分视觉骨干与投影层；历史图库重编码来源缺失。']))
        if revised.get(qid):
            item['alternative_explanations'].append('主体框属于额外目标定位监督；若原图同见多个药盒，需区分目标指代未明确与目标特征不可分，不能仅由裁切修复认定编码器缺陷。')
        item['counterevidence']=['所有原本成功Query均参加完整图库评分干预与损害检查。',
            '空间打乱保持对齐可用性；它保留的收益不能归为真实空间对应。']
        item['unresolved']=list(v.get('uncertainty',[])) if v else ['缺少真实视觉审核']
        item['unresolved']+=['不能为每张图由当前证据确定唯一失败原因。']
        item['link_base']=f'cases/{qid}/'
        item['links']={'token_card':f'../../../full_tokens/queries/{qid}/index.html',
            'comparisons':f'../../../full_tokens/queries/{qid}/comparisons.json',
            'original':os.path.relpath(workspace_path(original['original_path']),folder/'cases'/qid),
            'current_regions':f'../../../reviewed_regions/queries/{qid}/summary.json',
            'visual': '../../../visual_consolidated/index.html'}
        masters.append(item)
        for e in mechanism:evidence_counts[e['level']]+=1
        for c in original['candidates']:
            candidate_rows.append(dict(photo_id=qid,package_id=original['package_id'],split=original['split'],**c))
        for p in byphoto[qid]:
            scene_config_rows.append(dict(**p,current_scene_labels=item['scene_labels'],
                visual_review_status=item['visual_review_status'],observable_difficulties=item['observed_difficulties'],
                mechanism_evidence_levels=[e['level'] for e in mechanism],
                main_dossier=f'cases/{qid}/dossier.json',token_evidence=f'../full_tokens/queries/{qid}/index.html'))
        target=folder/'cases'/qid;target.mkdir(parents=True,exist_ok=True);write(target/'dossier.json',item)
        body=["<!doctype html><meta charset='utf-8'>",f'<title>{qid}</title><style>{CSS}</style>',
            '<a href="../../index.html">总筛选索引</a>',f'<h1>{qid}</h1>',
            f"<p>{h(original['identity'])}；{h(original['split'])}；{h(OUTCOME_ZH[original['outcome']])}。图像rank={original['image_positive_rank']}，融合rank={original['fusion_positive_rank']}。</p>",
            '<p class="notice">审核、分数分解、改评分响应和像素干预分层保存；不存在“1117原因全部查明”的结论。细则未辨清与标签歧义保留。</p>',
            '<p><a href="dossier.json">机器可读主档案</a> · '+
            f'<a href="../../../full_tokens/queries/{qid}/index.html">全部匹配连线/Token表/原始矩阵</a> · '+
            f'<a href="{h(item["links"]["current_regions"])}">修订框与人工语义区域贡献</a> · '+
            f'<a href="{h(item["links"]["original"])}">原图</a></p>',
            f'<img loading="lazy" src="../../../full_tokens/queries/{qid}/evidence.jpg">',
            '<h2>直接观察</h2><ul>'+''.join('<li>'+h(s)+'</li>' for s in item['direct_observations'])+'</ul>',
            '<h2>困难与不确定性</h2><p>'+h(', '.join(item['observed_difficulties']))+'</p><ul>'+''.join('<li>'+h(s)+'</li>' for s in item['unresolved'])+'</ul>',
            '<h2>机制证据</h2>'+table_html(mechanism,['factor','level','claim','evidence']),
            '<h2>图像/文字/融合正负差额</h2>'+table_html(fusion[qid],['positive','negative','image_raw_gap','BM25_raw_gap','normalized_image_contribution','normalized_text_contribution','fusion_gap']),
            '<h2>35配置逐图结果</h2>'+table_html(byphoto[qid],['config_id','prediction_available','correct_rank','top1_reference_id','positive_negative_gap','automatic_main_label']),
            '<details><summary>完整审核与成功对照记录</summary><pre>'+h(json.dumps(v,ensure_ascii=False,indent=2))+'\n'+h(json.dumps(controls.get(qid),ensure_ascii=False,indent=2))+'</pre></details>']
        if qid in changed_boxes:
            body.insert(8,'<p class="notice">此图上方历史对照使用审核前框。新框与新旧裁切见下图，原始图像/Token缓存不变。</p>'+
                f'<img src="../../../reviewed_crops/previews/{qid}.jpg">')
        if fresh_reference[qid]:
            body.append('<h2>新旧参考的干预配对响应</h2><p>仅预选候选，非完整图库准确率；原worker源码快照缺失，数值兼容由版本/核心与12原图逐位复现桥接。</p>'+
                table_html(fresh_reference[qid],['bank','positive','negative','arm','gap','edit_minus_sham_gap','switched_tokens','interaction_gap'])+
                '<p><a href="../../../fresh_reference_interventions/analysis/REPORT.md">补充实验报告与范围</a></p>')
        (target/'index.html').write_text('\n'.join(body))
    write(folder/'dossiers_all1117.json',masters);csvout(folder/'candidate_table.csv',candidate_rows)
    csvout(folder/'config_scene_photo_token_evidence.csv',scene_config_rows)
    observed=Counter(t for r in masters for t in set(r['observed_difficulties']))
    outcomes=Counter(r['outcome'] for r in masters)
    coverage=opt(source/'visual_consolidated/coverage.json',{})
    summary=dict(generated_utc=datetime.now(timezone.utc).isoformat(),queries=len(ids),
        identities=len({r['package_id'] for r in masters}),configurations=len({r['config_id'] for r in predictions}),
        prediction_occurrences=len(predictions),independent_photos=len(ids),
        data_coverage=len(dossiers)/len(ids),token_coverage=len(dossiers)/len(ids),
        visual_coverage=coverage,token_validation=token_validation,
        objective_outcomes=dict(outcomes),observable_difficulty_counts=dict(observed),
        observable_labels_are_multilabel_not_mechanisms=True,scene_source=str(scene_folder.relative_to(source)),
        evidence_records_by_grade=dict(evidence_counts),unique_failure_mechanisms_identified_for_all_queries=False,
        pending_cross_token_relation=opt(source/'cross_token_relation/independent_validation.json',{}).get('status')!='passed',
        pending_current_semantic_regions=not (source/'reviewed_regions/validation.json').exists())
    write(folder/'summary.json',summary)
    report_lines=report(source,scene_folder,summary)
    (folder/'REPORT.md').write_text('\n'.join(report_lines)+'\n')
    compact=[dict(photo_id=r['photo_id'],package_id=r['package_id'],split=r['split'],outcome=r['outcome'],
        scene=r['scene_labels'],labels=r['observed_difficulties'],image_rank=r['image_positive_rank'],
        fusion_rank=r['fusion_positive_rank'],review=r['visual_review_status'],
        configs={p['config_id']:dict(available=p['prediction_available']=='True',correct=p['correct']=='True',rank=p['correct_rank']) for p in r['all35_predictions']}) for r in masters]
    index=['<!doctype html><meta charset="utf-8">',f'<title>ColNomic 1117全量机制审计</title><style>{CSS}</style>',
        '<h1>ColNomic 1117查询：配置—场景—照片—Token证据</h1>',
        '<p><a href="REPORT.md">中文总报告</a> · <a href="summary.json">覆盖/验证摘要</a> · <a href="dossiers_all1117.json">全部主档案</a> · <a href="config_scene_photo_token_evidence.csv">35配置关联表</a></p>',
        f'<p class="notice">{len(ids)}张独立图，{len(predictions)}次重复评测。视觉与Token覆盖分开验收；可观察外观差异不能自动等于机制。历史测试集已被观察。请从照片卡查看具体证据和限制。</p>',
        '<label>搜索照片/身份/困难 <input id="search"></label><label>客观结果 <select id="outcome"><option value="">全部</option>'+''.join(f'<option value="{k}">{v}</option>' for k,v in OUTCOME_ZH.items())+'</select></label>',
        '<label>划分 <select id="split"><option value="">全部</option><option>train</option><option>val</option><option>test</option></select></label><p id="count"></p>',
        '<table><thead><tr><th>照片</th><th>身份/划分</th><th>图像/融合rank</th><th>客观结果</th><th>场景</th><th>观察困难（多标签）</th><th>审核</th></tr></thead><tbody id="body"></tbody></table>',
        '<h2>报告与完整实验</h2><ul>'+''.join(f'<li><a href="../{path}">{h(name)}</a></li>' for name,path in [
            ('统一视觉审核','visual_consolidated/index.html'),('35配置场景和互补分析',str(scene_folder.relative_to(source))+'/REPORT.md'),
            ('冻结Token完整矩阵验收','full_tokens/independent_validation.json'),('修订框后的完整几何评分','reviewed_boxes/validation.json'),
            ('修订裁切D01逐图结果','reviewed_crops/analysis/changed_query_results.json'),('融合权重/门控','full_fusion/selection.json'),
            ('受控左右词位置独立验收','completion_audit/signed_layout_independent_validation.json'),
            ('目标规格遮挡/非关键区域对照','detail_interventions/analysis/REPORT.md'),
            ('新参考对照及按身份统计','fresh_reference_interventions/interpretation/REPORT.md'),
            ('历史接续缺项记录（已由当前总验收取代）','completion_audit/REPORT.md')])+'</ul>',
        '<script>const data='+json.dumps(compact,ensure_ascii=False).replace('</',r'<\/')+';',
        "const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('\"','&quot;'); function render(){let s=document.querySelector('#search').value.toLowerCase(),o=document.querySelector('#outcome').value,t=document.querySelector('#split').value;let ds=data.filter(r=>(!o||r.outcome===o)&&(!t||r.split===t)&&JSON.stringify(r).toLowerCase().includes(s));document.querySelector('#count').textContent=ds.length+' / '+data.length+' 张';document.querySelector('#body').innerHTML=ds.map(r=>'<tr><td><a href=\"cases/'+esc(r.photo_id)+'/index.html\">'+esc(r.photo_id)+'</a></td><td>'+esc(r.package_id)+' / '+esc(r.split)+'</td><td>'+r.image_rank+' / '+r.fusion_rank+'</td><td>'+esc(r.outcome)+'</td><td>'+esc(JSON.stringify(r.scene))+'</td><td>'+esc(r.labels.join('，'))+'</td><td>'+esc(r.review)+'</td></tr>').join('')}for(const el of document.querySelectorAll('input,select'))el.addEventListener('input',render);render();</script>"]
    configuration_ids=sorted({r['config_id'] for r in predictions})
    filters='<label>配置 <select id="config"><option value="">全部</option>'+''.join(f'<option>{c}</option>' for c in configuration_ids)+'</select></label>'
    filters+='<label>所选配置结果 <select id="correct"><option value="">全部</option><option value="correct">正确</option><option value="wrong">错误</option></select></label>'
    for key,label,choices in [('background','物体数量',['single_object','multi_object','unclear']),('placement','拍摄方式',['handheld','tabletop']),('occlusion','前景遮挡',['foreground','none','unclear'])]:
        filters+=f'<label>{label} <select id="{key}"><option value="">全部</option>'+''.join(f'<option>{c}</option>' for c in choices)+'</select></label>'
    index.insert(next(i for i,line in enumerate(index) if line.startswith('<table><thead><tr><th>照片')),filters)
    index[-1]=FILTER_SCRIPT
    (folder/'index.html').write_text('\n'.join(index))
    source_files=[source/'full_tokens/dossiers_all.json',source/'full_tokens/independent_validation.json',
        scene_folder/'prediction_matrix.csv',source/'full_fusion/all_query_decomposition.csv']
    for path in [source/'visual_consolidated/records.json',source/'visual_consolidated/coverage.json',
        source/'visual_consolidated/same_face_success_controls.json',source/'reviewed_regions/validation.json',
        source/'cross_token_relation/independent_validation.json',source/'natural_shared_phrase_layout/independent/validation.json',
        source/'positions_reviewed/independent_validation.json',source/'position_content_control/validation.json',
        source/'foreground_harms_review/manual_reviews.json',source/'foreground_harms_review/join_records.json',
        source/'scene35_reviewed/final_scene_statistics/validation.json',source/'scene35_reviewed/scene_label_revisions.json',
        source/'reviewed_boxes/metrics.csv',source/'full_fusion/metrics.csv',
        source/'set_probes/natural/full_gallery/metrics.json',source/'set_probes_symmetric/layout/layout_metrics.csv',
        source/'natural_shared_phrase_layout/metrics.csv',source/'cross_token_relation/full_gallery/metrics.csv',
        source/'positions_reviewed/position_metrics.csv',source/'position_content_control/metrics.csv',
        source/'weighted_spatial_only_reviewed/factorial_2x2_and_reward_controls.csv',
        source/'weighted_spatial_only_reviewed/per_photo_factorial_transitions.csv',
        source/'weighted_spatial_only_reviewed/independent_verification.json',
        source/'weighted_spatial_only_reviewed/selection.json',source/'correspondence_controls/independent_validation.json',
        source/'difficulty_config/independent_validation.json',source/'reviewed_crops/analysis/independent_validation.json',
        source/'controlled_crossview_layouts/analysis/independent_validation.json',
        source/'difficulty_config/validation.json',source/'natural_shared_phrase_layout/eligibility.json',
        source/'natural_shared_phrase_layout/independent/package_bootstrap.csv',
        source/'visual_consolidated/same_face_control_coverage.csv',source/'reviewed_regions/localization_coverage.json',
        source/'reviewed_crops/analysis/changed_query_results.json',source/'cross_token_relation/REPORT.md',
        source/'fresh_reference_interventions/analysis/validation.json',source/'fresh_reference_interventions/analysis/pair_effects.csv',
        source/'fresh_reference_interventions/analysis/fresh_vs_frozen.csv',source/'fresh_reference_interventions/interpretation_corrections.json',
        source/'fresh_reference_interventions/independent/validation.json',source/'fresh_reference_interventions/interpretation_independent/validation.json',
        source/'fresh_reference_interventions/feature_bridge/validation.json',source/'reproduction_support/final_full_verification.json',
        source/'reproduction_support/final_dispatcher_quick_validation.json']:
        if path.exists():source_files.append(path)
    write(folder/'source_manifest.json',{'generated_by':str(Path(__file__).relative_to(ROOT)),
        'generator_SHA256':sha(Path(__file__)),'sources':{str(p.relative_to(ROOT)):sha(p) for p in source_files}})
    print('Joined',len(masters),'dossiers and',len(scene_config_rows),'configuration records',flush=True)


def report(source,scene_folder,s):
    n=s['queries']; out=s['objective_outcomes']; coverage=s['visual_coverage']
    geo=[r for r in rows(source/'reviewed_boxes/metrics.csv') if r['split']=='all' and not r.get('stratum')]
    fusion=[r for r in rows(source/'full_fusion/metrics.csv') if r['split']=='all' and r['baseline']=='image']
    methods=[dict(configuration=r['configuration'],method=r['method'],R1=round(float(r['r1']),6),
        R5=round(float(r['r5']),6),R10=round(float(r['r10']),6),MRR=round(float(r['mrr']),6),
        repairs=r.get('repaired',r.get('rescued')),harms=r['harmed'],net=r['net']) for r in geo]
    crop=opt(source/'reviewed_crops/analysis/changed_query_results.json',[])
    full_probe=read(source/'set_probes/natural/full_gallery/metrics.json')
    probe_table=[dict(head=k,**next(r for r in v if r['split']=='test')) for k,v in full_probe.items()]
    signed=[r for r in rows(source/'set_probes_symmetric/layout/layout_metrics.csv') if r['split']=='test' and r['threshold_strategy']=='validation_calibrated' and not r.get('control')]
    natural=[r for r in optrows(source/'natural_shared_phrase_layout/metrics.csv') if r['split']=='test' and r['threshold_strategy']=='validation_calibrated' and not r.get('control')]
    natural_ci=optrows(source/'natural_shared_phrase_layout/independent/package_bootstrap.csv')
    relation=[r for r in optrows(source/'cross_token_relation/full_gallery/metrics.csv') if r['split']=='test' and r['head'] in ['relation','count_only']]
    eligibility=opt(source/'natural_shared_phrase_layout/eligibility.json',{})
    shared=opt(source/'difficulty_config/validation.json',{}).get('comparison_scopes',{})
    controls=optrows(source/'visual_consolidated/same_face_control_coverage.csv')
    region=opt(source/'reviewed_regions/localization_coverage.json',{})
    positions=[r for r in optrows(source/'positions_reviewed/position_metrics.csv') if r['split']=='test' and r['control'] in ['feature_only','train_mean','y_coordinate_only']]
    content_positions=[r for r in optrows(source/'position_content_control/metrics.csv') if r['split']=='test']
    factorial=[r for r in optrows(source/'weighted_spatial_only_reviewed/factorial_2x2_and_reward_controls.csv') if r['split']=='all']
    scenes=[r for r in rows(scene_folder/'B03_F03_scene_transitions.csv') if r['label_set']=='verified_snapshot'
        and r['split']=='all' and r['dimension'] in ['background','placement','occlusion']]
    token_checks=s['token_validation'].get('checks',{})
    observed=[dict(difficulty=k,photos=v,denominator=n,fraction=round(v/n,6)) for k,v in sorted(s['observable_difficulty_counts'].items(),key=lambda r:-r[1])]
    return [
        '# ColNomic：1117张查询的错误、Token与竞争机制审计',
        '',f'本报告讨论 {n} 张独立查询、5405 条图库记录、5330 个SHA设计和35个历史配置。39095是重复评测次数。原标签用于复现历史成绩；视觉发现的标签歧义单独保留。',
        '', '## 首页五个问题', '',
        f'**哪些情况下失败？** B03错误 {out["fusion_rescued"]+out["both_wrong"]}/{n}，F03错误 {out["fusion_harmed"]+out["both_wrong"]}/{n}。两者都对 {out["both_correct"]}，图像错融合对 {out["fusion_rescued"]}，图像对融合错 {out["fusion_harmed"]}，两者都错 {out["both_wrong"]}。场景、面、数字规格、排版、主色、图案、清晰度等是可观察多标签，频数不能相加，也不能直接叫失败机制。',
        '', '**哪些原因有证据？** 冻结特征与原始评分的确定性复放、Token贡献守恒、融合新增错误发生在评分公式加入文字之后，有直接证据。输入可辨识性、身份标签歧义、空间约束、特征Probe和像素编辑分别有范围限定的证据；不支持为全部失败指定唯一编码器原因。',
        '', '**匹配、汇总和融合分别改变什么？** 下表来自全部1117×5405实际评分。OCR对齐包含文字支持，γ/λ拆分后不能把对齐可用性奖励的收益都算为布局匹配。Query自权重另保存完整候选界和确切top10/rank，未计算的低界scalar明确NaN。融合选α与门控仅用验证集，同时保存修复、损害和原143救回样本的损失。',
        '',mdtable(methods,['configuration','method','R1','R5','R10','MRR','repairs','harms','net']),
        '', '匹配规则×汇总规则的2×2对照独立列出。空间两行均为γ=0、λ=0.1；Query权重η=0.2仅从验证身份选择。最后两行额外保留γ=0.2的文字对齐可用性奖励作为对照，不能合称纯空间匹配。修复/损害列均相对于original_uniform；各自均匀汇总的损益和来源组bootstrap另见逐图过渡与交互表。',
        '', mdtable(factorial,['cell','photos','eta','gamma','spatial_lambda','r1','r5','r10','mrr','package_macro_r1','repaired_vs_original_uniform','harmed_vs_original_uniform']),
        '', '[完整候选界、精确top10/正确候选排名、参数选择和逐图2×2交互](../weighted_spatial_only_reviewed/REPORT.md)已保存。未精算的候选保留NaN及有效上界，不能把排名认证说成保存了全部修正后的标量分数。空间约束与加权没有形成通用修复；原本成功样本的损害一并保留。',
        '', '**多因素怎样共同作用？** 每张卡分别列直接观察、分数差额、改评分响应、像素干预、竞争解释和未知。文字内容、布局、尺度、包装面和标签可同时不同；无法隔离时保留混杂。例如Mucus DM重点例不仅布局不同，正/负剂量是60/1200mg与30/600mg；Capstar负设计为猫+狗且计数不同，Query下部计数被鼠标遮挡。',
        '', '**哪些可写进论文？** 可以报告忠实基线成绩、融合修复/损害、限定身份分组的Probe/完整图库差距、OCR可用性与空间项的对照、经过像素守恒验收的合成响应、人工修订定位的输入有效性反例。不能声称绝对位置预测证明自然实例布局可分，不能把Probe失败写成信息不存在，不能将共享文字相关或少数编辑响应升级成普遍因果机制。历史测试已被观察，不能称新盲测。',
        '', '## 覆盖与分母', '',
        f'全部主档案 {n}/{n}；全部Token分析 {n}/{n}；去重正负对 {token_checks.get("comparison_conservation_checks")}；完整关键矩阵{token_checks.get("pointwise_matrices_replayed")}。详细验收：[本轮Token独立复放](../delivery_verification/full_tokens/independent_validation.json)。',
        '', '视觉覆盖必须按overview、场景各维度、细节放大和机制证据分别解释：',
        '', '```json\n'+json.dumps(coverage,ensure_ascii=False,indent=2)+'\n```',
        '', '全部现有可观察困难的发生率如下，分母是1117，包括成功控制；多标签同时出现。细则未核验不填成“无差异”。错误内部比例及身份分布见统一视觉和场景报告。',
        '',mdtable(observed,['difficulty','photos','denominator','fraction']),
        '', '以上沿用人工观察的既有18类统一映射，原始细标签与审核历史一并保留。实际同面成功控制覆盖如下；missing或unconfirmed不表示该照片不存在：',
        '',mdtable([r for r in controls if r['category'] in ['all','layout_or_linewrap','numbers_specs_or_quantity','shared_text_or_brand']],['configuration','category','failure_queries','queries_with_actual_same_face_success','control_missing_or_unconfirmed','identities_with_control']),
        '', '[同面固定正负候选对照](../correspondence_controls/fixed_candidate_same_face_controls.csv)同时保持两种参考和候选Token数相同，保存完整矩阵及来源组区间；Query尺度、清晰度、角度仍未匹配。全量匹配集中度按候选Token数分层，跨行和多对一现象不能直接判为错误。',
        '', f'语义ROI：{region.get("manual_semantic_queries")}张重点图、{region.get("manual_ROIs")}个人工区域；全量独立分区仍守恒，但其余关键规格/布局/图案未定位时保留未知。[审后区域范围](../reviewed_regions/localization_coverage.json)。视觉patch可跨行，并非单词。',
        '', '## 35配置、三个场景维度与互补性', '',
        f'权威明细：[35配置报告](../{scene_folder.relative_to(source)}/REPORT.md)、[逐图配置—场景—Token关联表](config_scene_photo_token_evidence.csv)。物体数量、手持/桌面、前景遮挡独立统计；前三者不能相加。原标签、118已审核子集和审后统计并行保留，原审核子集有选择偏差。',
        '', 'E0–E7保留原规则与优先级，未校准。E2只是正确设计排名>10，E3只是2–10；E4/E5是通道正确性关系。E2主标签数不等于完整Recall@10未命中数。C0–C7沿用项目定义，未人工核验的不自行推断。',
        '', 'B03是原图ColNomic、B05是BM25、B08是OCR→图像、B12组合B03/B08、F03组合B03/B05；这些融合输入不同。B16 image_crisp改变forward/reverse Token评分，并未裁切或提高清晰度。D01人工框裁切具有额外标注监督；D03是SAM3遮背景并裁切且有fallback。裁切同时改变背景、缩放和内容完整性。',
        '', '原标签中多物体177→181与单体693→791、有前景遮挡32→24均逐图复核，详见原场景过渡表及8张融合新增错的文字项分解。这些是原标签下统计，审后数字另列；不能由32张图像全对推普遍遮挡鲁棒性，不能将融合错误当成遮挡因果作用。',
        '', '全量实际审核后的三个独立维度如下。每行分母为该类可用照片数；未知值单列。区间为身份相关的来源组配对bootstrap，尤其小类不宜作因果解释。',
        '',mdtable(scenes,['dimension','label','available_photos','identities','baseline_correct','variant_correct','both_correct','rescues','harms','both_wrong','net','source_group_bootstrap_low','source_group_bootstrap_high']),
        '', '任一35方法成功的oracle只是需真值的比较上界；共同失败/互补逐图保存。总体成绩接近不保证错误相同，跨模型成功只支持竞争比较。',
        '', '实际交集如下，B03比较对象按真实“原图image”输入合同预先选择，不把文字/裁切方法混作同一编码器对照：',
        '',mdtable([dict(scope=k,**v) for k,v in shared.items()],['scope','queries','identities']),
        '', '[困难×配置的类内失败率、错误构成、逐图交集与损益](../difficulty_config/REPORT.md)保留独立分母和多标签重叠；多个方法共有的错误、相对于指定方法的B03错误和融合新增错分别可追溯。它们不是机制已查明数量。',
        '', '## 冻结特征Probe与受控布局', '',
        '身份隔离为91/21/18包装和82/18/18来源组，783/173/161查询，所有实际交集已检查。标准化仅训练数据，参数/阈值仅验证；feature-only不输入坐标/ID。坐标与ROI辅助头单独标额外输入。阈值准确率、同Query配对排序、差额、AUC、身份宏平均并行报告。',
        '', '原受限候选对的高配对成绩无法代表检索：', '',
        mdtable(probe_table,['head','queries','identities','recall_at_1','recall_at_5','recall_at_10','mrr','repaired','harmed']),
        '', '上表测试集161张/18身份，全图库5405。池化只检验均值/标准差组合，kernel mean是有限集合分布摘要，二者失败都不能证明完整Token没有布局信息。完整Query×Reference点积关系在池化前读取的38维头，以及Token数量单独评分对照的完整图库成绩如下：',
        '',mdtable(relation,['head','queries','identities','recall_at_1','recall_at_5','recall_at_10','mrr','identity_macro_recall_at_1','repaired','harmed','net_change']),
        '', '[关系头全部配对、困难对、完整图库及独立验收](../cross_token_relation/REPORT.md)均保留。完整图库候选的Token数量范围与训练配对明显不同；小Token参考大量成为赢家，这是支持范围差异的证据，不能独自解释全部下降。此头仍为有限关系摘要，受限配对高分不能当作检索修复。',
        '', 'OCR行序标签是主体OCR行的纵向排序位置，非阅读行号。仅y坐标基线显著解释它，绝对位置预测不直接检验同文字身份布局。旧targets的9个打乱头已恢复，当前审核框也重新拟合并独立重建标签/验证全部主头、9null与坐标对照：',
        '',mdtable(positions,['probe','control','mae','package_macro_mae','r2','dim0_four_bin_accuracy']),
        '', '[当前定位Probe验收](../positions_reviewed/independent_validation.json)只验证冻结特征可预测这些几何标签；坐标对照故意输入y，与feature-only头分开。',
        '', '文字内容单独预测同一OCR纵向序号也有可测关联：训练词表的256维词存在向量，不输入坐标/Token、索引、图像ID或OCR行数；词表/标准化仅训练，ridge仅验证选择，主头与3null闭式独立重拟合。它检查文字/位置相关性，不直接证明feature-only头利用某种捷径。',
        '',mdtable(content_positions,['control','seed','samples','packages','mae','package_macro_mae','r2','dim0_four_bin_accuracy']),
        '', '[文字内容与位置关联控制](../position_content_control/REPORT.md)的样本关联仍依靠OCR多边形/主体框外部监督。',
        '', '从130身份按固定像素质量规则选出全部41合格词位置源，27/9/5身份划分，246编码，41原图与缓存逐位一致。等幅左右移动原字形，原图/sham/editor-only对照均保存；只修改单个孤立词，不能称自然换行或身份布局实验。测试5身份，解释限于合成编辑：',
        '',mdtable(signed,['probe','accuracy','auc','pair_accuracy','package_macro_pair_accuracy','packages','gap_mean']),
        '', '历史唯一测试身份50%是pilot，阈值和配对必须拆开。旧38身份实验比较位移幅度，同样不是身份布局证据。全图库精确OCR词bag相同且异SHA组0是代理匹配缺失，不能证明自然同文字包装不存在。',
        '', f'自然共享短语局部身份配对：固定质量规则在1117查询/130身份上筛得{eligibility.get("pairs")}组、{eligibility.get("queries")}张图、{eligibility.get("identities")}身份，训练/验证/测试身份{eligibility.get("identities_by_split")}。每组正负均含相同四条OCR短语，但其它内容、颜色、字体与整图上下文不同；只检验这些局部特征与身份判断的关系，不是布局单因素实验。65身份的首例四短语实际放大查验，后续同身份案例依旧含OCR核验范围限制。',
        '',mdtable(natural,['probe','samples','packages','accuracy','auc','pair_accuracy','package_macro_pair_accuracy','gap_mean']),
        '',mdtable(natural_ci,['probe','test_identities','test_pairs','identity_macro_pair_accuracy','ci95_low','ci95_high']),
        '', '测试8身份/24配对，小样本区间较宽；ROI选Token含OCR位置辅助监督，patch可越过选定短语。所有5头、15null、标准化/阈值/参数均独立重放，结果不能证明纯布局可分或特征完全无信息。[局部Probe验收](../natural_shared_phrase_layout/independent/validation.json)。',
        '', '跨两张不同原图的同词左右等幅编辑进一步排除了同一图片自身匹配捷径。全130身份固定规则实际筛查、视觉核实后仅3身份合格，2训练/1验证/0测试，6原图/36编码；因此只报告预定义同向/反向关系和sham、editor-only差额，不训练/宣称held-test Probe或自然身份精度。[跨视角pilot](../controlled_crossview_layouts/analysis/REPORT.md)。',
        '', '## 编码审计、图像干预与人工框修订', '',
        '近期Query drift已定位PIL后端；TV+FP32 LoRA原图可逐位复现。旧图库5405均从已缺失的历史canonical/gallery缓存复用，缺少原processor张量与中间层日志；现多recipe测试仍无法逐位复现。不能唯一归因GPU型号、CPU卸载或某个编码层。冻结分析不受此影响。',
        '', '12独立身份、4客观结果各3张进行了数字区域/等面积非关键区/组合遮挡与sham，60编码、145矩阵保存Token变化、匹配切换/未切换贡献与交互项。数字OCR行有的并非身份关键，有的非关键区在背景；编辑影响语义、对比度和纹理，报告保留这些限定。新Query对旧冻结图库的条件排名不等于新运行时全图库准确率。',
        '', '另补同数值运行时参考对照：预先冻结的全部28参考及12原Query重新编码，12 Query base全部逐位等于已有编辑臂基准；包版本、模型/处理后端、数值encoder源码一致，因此复用60个兼容编辑臂。历史worker完整源码SHA不同且快照缺失，无法唯一说明其差异。新旧参考各145个完整矩阵、17对×5arms×2参考bank的170行配对和区域/交互分解均保留。28个新参考均未逐位复现旧缓存；17个base正负差额的符号均保持不变，不能由特征漂移直接断言它造成原错误。',
        '', '[新参考配对及独立重放](../fresh_reference_interventions/analysis/REPORT.md)明确区分新参考条件响应和原始全图库统计。同Token长度或ordinal索引轴不证明新旧物理格点相同；跨bank argmax编号变化仅为索引诊断，同bank edit-sham切换另保存。',
        '', '[按身份重采样的编辑响应和新旧参考差异](../fresh_reference_interventions/interpretation/REPORT.md)给出全部预定义单因素/组合效应及区间，保留改善、无变化和反向响应；它们不是编辑后身份准确率。',
        '', '用户复核后5张框坐标变化，旧框/新框版本分别保存，旧统计不覆写。5张旧D01裁切重新编码全部逐位一致；完整图库比较为：',
        '',mdtable(crop,['photo_id','split','B03_rank','F03_rank','before_D01_rank','fresh_before_D01_rank','after_D01_rank']),
        '', '这是历史结果被查看后的人工作业，不能作为无标注定位算法或独立盲测成绩。B03/F03原图特征与基线分数未改变。',
        '', '[8张原前景遮挡融合新增错的实际关键区域审核](../foreground_harms_review/REPORT.md)区分“有遮挡”和“规格/身份信息被挡”；图像项为正而文字项反向更大是确定性评分事实，是否由遮挡改变OCR还需要反事实对照，不能仅凭这8例推因果。',
        '', '## 复现、验收和限制', '',
        'Token/区域守恒、评分轴、SHA排序、参数选择、损益、矩阵逐点重放均由独立程序检查。范围边界：[当前逐项验收与限制](acceptance.json)。completion_audit中的旧inventory仅记录接续时情况，当前总验收以此处为准。冻结数学验证与语义人工审核不是同一保证。',
        '', '只验证和全部可复现分析入口、代码/环境/协议/日志/恢复点及归档清单位于交付根。原缓存保持不变；subject_boxes是用户授权修改，前后快照及SHA另存。',
        '', '未知/未执行/不适用的范围继续显式留在验收矩阵；1117张有数据与审计，不等于1117种失败原因全部查明。'
    ]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.source.resolve(),args.output.resolve())


if __name__=='__main__':main()
