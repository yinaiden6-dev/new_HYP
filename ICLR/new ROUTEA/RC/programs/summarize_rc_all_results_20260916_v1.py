#!/usr/bin/env python3
"""Read sealed Route A results and build a lineage-preserving results dossier."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/new_hyp_complete_results_20260916_v1'
SOURCES, TABLES, CHECKS = {}, {}, []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def src(p):
    p = Path(p)
    if not p.is_absolute(): p = ROOT / p
    key = str(p.relative_to(ROOT))
    assert not any(x in key.lower() for x in ['d1_mi', 'formal392'])
    if key not in SOURCES: SOURCES[key] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    return p


def read(p): return json.loads(src(p).read_text())
def result(name): return read('results/' + name + '/result.json')


def csvout(name, rows):
    assert rows
    TABLES[name] = rows
    with (OUT / (name + '.csv')).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def mtable(headers, rows):
    def cell(v): return str(v).replace('|', '\\|').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
        ['| ' + ' | '.join(map(cell, r)) + ' |' for r in rows])


def link(p, label='原结果'):
    return f'[{label}](<{os.path.relpath(ROOT / p, OUT)}>)'


def recount(d, panel, source, level):
    rows = d['rows']; n = len(rows)
    assert len({x['query_id'] for x in rows}) == n
    models = list(rows[0]['correct'])
    output = []
    for model in models:
        count = sum(bool(x['correct'][model]) for x in rows)
        if model in d.get('counts', {}): assert count == d['counts'][model]
        if model in d.get('scores', {}): assert count == d['scores'][model]['correct']
        if 'RAW' in models:
            baseline = sum(bool(x['correct']['RAW']) for x in rows)
            rescue = sum(not x['correct']['RAW'] and x['correct'][model] for x in rows)
            loss = sum(x['correct']['RAW'] and not x['correct'][model] for x in rows)
            assert count - baseline == rescue - loss
        else: baseline = rescue = loss = ''
        mrr = ''
        if 'ranks' in rows[0] and model in rows[0]['ranks']:
            mrr = sum(1 / x['ranks'][model] for x in rows) / n
            expected = d.get('MRR', {}).get(model, d.get('scores', {}).get(model, {}).get('MRR'))
            if expected is not None: assert abs(mrr - expected) < 1e-12
        output.append(dict(panel=panel, model=model, queries=n, correct=count,
            accuracy_percent=count / n * 100, RAW_correct=baseline, rescue_vs_RAW=rescue,
            loss_vs_RAW=loss, net_vs_RAW=rescue-loss if rescue != '' else '', MRR=mrr,
            candidate_source='RAW natural C128', action='127 challengers; SWITCH iff max logit > 0 else HOLD',
            evidence_level=level, source=source, verification='recounted from per-query correct/rank records'))
    # Check sealed pair counts and independently reconstruct grouped point estimates.
    for pair, c in d.get('comparisons', {}).items():
        if '__to__' not in pair: continue
        a, b = pair.split('__to__')
        if a not in models or b not in models: continue
        rescue = sum(not x['correct'][a] and x['correct'][b] for x in rows)
        loss = sum(x['correct'][a] and not x['correct'][b] for x in rows)
        if 'rescue' in c: assert rescue == c['rescue']
        if 'loss' in c: assert loss == c['loss']
        group_key = 'component' if 'component' in rows[0] else 'group'
        if group_key in rows[0]:
            groups = {}
            for x in rows: groups.setdefault(x[group_key], []).append(int(x['correct'][b])-int(x['correct'][a]))
            mean = sum(sum(v)/len(v) for v in groups.values())/len(groups)
            for key in ['equal_component_difference','equal_video_difference','equal_sku_difference','equal_patient_difference']:
                if key in c: assert abs(mean-c[key]) < 1e-12, (panel, pair, key)
    CHECKS.append(dict(source=source, panel=panel, queries=n, models=len(models), passed=True))
    return output


def make_xlsx():
    def col(n):
        out=''
        while n: n,k=divmod(n-1,26);out=chr(65+k)+out
        return out
    sheets=list(TABLES)
    with zipfile.ZipFile(OUT/'all_results.xlsx','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(sheets)+1))+'</Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml','<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+''.join(f'<sheet name="{name[:31]}" sheetId="{i}" r:id="rId{i}"/>' for i,name in enumerate(sheets,1))+'</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(sheets)+1))+'</Relationships>')
        for i,name in enumerate(sheets,1):
            rows=TABLES[name];keys=list(rows[0]);data=[keys]+[[r[k] for k in keys] for r in rows];xml=[]
            for j,row in enumerate(data,1):
                cells=[]
                for k,v in enumerate(row,1):
                    addr=f'{col(k)}{j}'
                    if isinstance(v,(int,float)) and not isinstance(v,bool): cells.append(f'<c r="{addr}"><v>{v}</v></c>')
                    else: cells.append(f'<c r="{addr}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>')
                xml.append(f'<row r="{j}">'+''.join(cells)+'</row>')
            z.writestr(f'xl/worksheets/sheet{i}.xml','<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="25" width="24" customWidth="1"/></cols><sheetData>'+''.join(xml)+f'</sheetData><autoFilter ref="A1:{col(len(keys))}{len(data)}"/></worksheet>')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    legacy=[]
    for name in ['rc_absolute_evidence_scale_calibration_v1','rc_native7_training_objective_factorial_v1',
                 'rc_pair_control_scale_harmonization_v1','rc_full_reference_content_bridge_v1',
                 'rc_retrained_evidence_sufficiency_v1','rc_product_response_factorial_v1',
                 'rc_reference_support_maxmin_readout_v1','rc_shared_projection_direction9_v1',
                 'rc_same_support_specificity_v1','rc_original7_train128_readout_v1']:
        d=result(name)
        for panel,models in d['metrics'].items():
            for model,arms in models.items():
                for arm,metric in arms.items():
                    if not isinstance(metric,dict) or 'final_top1' not in metric:continue
                    actions=d['actions'][panel][model][arm]
                    n=len(actions);count=sum(a['final_correct'] for a in actions)
                    rescue=sum(not a['base_correct'] and a['final_correct'] for a in actions)
                    loss=sum(a['base_correct'] and not a['final_correct'] for a in actions)
                    baseline=sum(a['base_correct'] for a in actions)
                    assert count==metric['final_top1'] and n==metric['query_count']
                    assert rescue==metric['rescue'] and loss==metric['break']
                    legacy.append(dict(experiment=name,panel=panel,model=model,arm=arm,queries=n,correct=count,
                        RAW_correct=baseline,rescue_vs_RAW=rescue,loss_vs_RAW=loss,
                        MRR=metric.get('final_MRR',metric.get('final_full_gallery_MRR','')),
                        evidence='opened internal; TRAIN entries descriptive only',source='results/'+name+'/result.json'))
        CHECKS.append(dict(source='results/'+name+'/result.json',panels=list(d['metrics']),actions_recounted=True,passed=True))
    csvout('early_and_expanded_arms',legacy)
    old=result('rc_original7_eval128_full_evidence_v1');interventions=[]
    for arm,metric in old['metrics_all128'].items():
        actions=old['actions'][arm];assert len(actions)==128
        assert sum(a['final_correct'] for a in actions)==metric['final_top1']
        assert sum(not a['base_correct'] and a['final_correct'] for a in actions)==metric['rescue']
        assert sum(a['base_correct'] and not a['final_correct'] for a in actions)==metric['break']
        interventions.append(dict(panel='EVAL128',model='frozen ORIGINAL7',arm=arm,queries=128,
            correct=metric['final_top1'],rescue_vs_RAW=metric['rescue'],loss_vs_RAW=metric['break'],
            MRR=metric['final_full_gallery_MRR'],source='results/rc_original7_eval128_full_evidence_v1/result.json'))
    csvout('ORIGINAL7_interventions',interventions)
    standard = [
      ('rc_new_hyp593_oof5_v1','H593 OOF5'),
      ('rc_h593_group_risk_strong_base_v1','H593 OOF5'),
      ('rc_h593_increment_attribution_v1','H593 OOF5'),
      ('rc_h593_endpoint_competition_v1','H593 OOF5; endpoint comparison'),
      ('rc_six_cause_isolation_v1/loss_binding','H593 OOF5'),
      ('rc_six_cause_isolation_v1/coverage','H593 OOF5; training coverage sensitivity'),
      ('rc_six_cause_isolation_v1/bridge','TRAIN128 OOF4; representation probes'),
      ('rc_train128_disagreement_oof4_v1','TRAIN128 OOF4'),
      ('rc_train128_hold_lift_exact_oof4_v1','TRAIN128 OOF4'),
      ('rc_train128_protected_projection_oof4_v1','TRAIN128 OOF4'),
      ('rc_train128_group_held_selection_v1','TRAIN128 OOF4'),
      ('rc_query_content_routing_oof4_v1','TRAIN128 OOF4'),
      ('rc_paired_local_evidence_oof4_v1','TRAIN128 OOF4')]
    internal=[]
    for name,panel in standard:
        d=result(name)
        if 'rows' in d and d['rows'] and isinstance(d['rows'][0].get('correct'),dict):
            internal += recount(d,panel,'results/'+name+'/result.json','opened development; fold-specific heads')
        else: raise ValueError((name,'unsupported per-query schema'))
    fixed=[]; fixed_matrix={}
    for name in ['rc_fixed_panels_train269_group_risk_v1','rc_original_mixed96_group_risk_v1',
                 'rc_opened_convex_cause_readout_v1','rc_full_candidate_identity_loss_v1','rc_global7_train128_fixed_panels_v1']:
        d=result(name)
        for panel,v in d['panels'].items():
            records=recount(v,panel,'results/'+name+'/result.json','historically opened fixed panel')
            fixed+=records
            for row in records:
                if not row['model'].endswith('CBIND'): fixed_matrix.setdefault(row['model'],{})[panel]=row['correct']
    d=result('rc_original7_train128_readout_v1')
    for panel,models in d['metrics'].items():
        for model,arms in models.items():
            if panel.startswith('EVAL'): fixed_matrix.setdefault(model,{})[panel]=arms['REAL']['final_top1']
    gap=result('rc_original7_gap_hold_followup_v1')
    for panel,v in gap['panels'].items():fixed_matrix.setdefault('GAP_HOLD1',{})[panel]=v['metrics']['GAP_HOLD1']['final_top1']
    csvout('internal_all_models',internal);csvout('fixed_panel_models',fixed)
    fixedrows=[dict(model=k,EVAL32=v.get('EVAL32',''),EVAL128=v.get('EVAL128','')) for k,v in fixed_matrix.items()]
    csvout('fixed_panel_matrix',fixedrows)

    ext_specs=[('GroZi480','rc_new_hyp_grozi120_external_v1','scoped external confirmation','video_bootstrap95','equal_video_difference'),
               ('RPC600','rc_new_hyp_rpc_transfer_v1','scoped external confirmation','sku_stratified_bootstrap95','equal_sku_difference'),
               ('ISIC537','rc_new_hyp_isic_transfer_v1','opened cohort; exploratory transfer','patient_bootstrap95','equal_patient_difference')]
    external=[]; comparisons=[]; ext={}
    for panel,name,level,ci,estimand in ext_specs:
        d=result(name);ext[panel]=d
        for vf in ['result_validation.json','independent_final_audit_v1.json']:
            assert 'PASS' in read('results/'+name+'/'+vf)['status']
        external+=recount(d,panel,'results/'+name+'/result.json',level)
        assert d['target_recall_C128']==sum(x['target_in_C128'] for x in d['rows'])
        for key,v in d['comparisons'].items():
            a,b=key.split('__to__')
            comparisons.append(dict(panel=panel,baseline=a,model=b,rescue=v['rescue'],loss=v['loss'],
                net=v['rescue']-v['loss'],query_gain_pp=(v['rescue']-v['loss'])/len(d['rows'])*100,
                grouped_estimand=estimand,group_mean_pp=v[estimand]*100,CI95_low_pp=v[ci][0]*100,
                CI95_high_pp=v[ci][1]*100,reliable_positive=v['reliable_positive'],
                CI_source='reused sealed bootstrap; grouped point estimate recounted',evidence_level=level))
    csvout('external_all_models',external);csvout('external_comparisons',comparisons)
    six=read('results/rc_six_cause_isolation_v1/analysis.json')
    read('results/rc_six_cause_isolation_v1/analysis_validation.json')
    h593=result('rc_six_cause_isolation_v1/loss_binding')
    assert six['counts']==h593['counts']
    hcomp=[]
    for k,v in six['comparisons'].items():
        a,b=k.split('__to__');hcomp.append(dict(baseline=a,model=b,rescue=v['rescue'],loss=v['loss'],net=v['net'],
            group_mean_pp=v['equal_component_difference']*100,CI95_low_pp=v['component_bootstrap95'][0]*100,
            CI95_high_pp=v['component_bootstrap95'][1]*100,evidence='H593 development OOF5; 64 components'))
    csvout('H593_comparisons',hcomp)
    split=read('results/rc_new_hyp593_oof5_v1/metadata/split_manifest.json')
    assert sum(split['fold_image_counts'])==593 and split['identities']==68 and split['components']==64
    foldrows=[]
    for f in split['folds']:
        n=len(f['heldout_query_ids']);matched=[x for x in h593['rows'] if x['fold']==f['fold']]
        assert {x['query_id'] for x in matched}==set(f['heldout_query_ids'])
        row=dict(fold=f['fold'],heldout=n,nominal_train=len(f['train_query_ids']))
        row.update({m:sum(x['correct'][m] for x in matched) for m in ['RAW','ALL_COST4','GROUP_BASE','COST1','ALL_CE']})
        foldrows.append(row)
    csvout('H593_by_fold',foldrows)
    gates=[]
    for name in ['rc_lth_p_only_gisc_optimization_v5_compat','rc_competitive_witness_development_v6',
                 'rc_grouped_residual_development_v7','rc_coherent_residual_development_v8']:
        d=result(name);g=d.get('gate_reduction',d.get('matched_gate_reduction'))
        gates.append(dict(experiment=name,panel='opened TRAIN32; post-hoc P-only development',
            target_connected=g['selected_target_connected_h1_coverage_count'],correct=g['real_success_count'],
            vs_query_only_net=g['query_only']['paired_net'],C_BIND_positive_drop=g['c_bind_positive_drop_count'],
            P_COORD_positive_drop=g['p_coord_positive_drop_count'],gate_go=g['go'],
            producer_scientific_status=d.get('scientific_GO_or_NO_GO'),
            failures=';'.join(d['all_simultaneous_failures']),source='results/'+name+'/result.json'))
    csvout('spatial_P_history',gates)

    # Index all completed-result-era mainline reports; snapshots retain their original dates/status.
    index=[]
    for p in sorted((ROOT/'reports').glob('*.md')):
        date=re.search(r'202609(\d{2})',p.name)
        if not date or not 8<=int(date[1])<=16:continue
        if any(s in p.name for s in ['DINO','D1_MI','CW1_SR0','CW0_RGH','PREEXECUTION']):continue
        if not any(s in p.name for s in ['NEW_HYP','H593','ORIGINAL7','RC_HYP','RC_V6','RC_V9','RC_REFERENCE_HYP',
            'CONVEX','PAIRED_CE','PAIRED_LOCAL','QUERY_CONTENT','FULL_CANDIDATE','GLOBAL7','MIXED96',
            'TRAIN269','TRAIN128','SIX_CAUSE','PRODUCT_RESPONSE','RETRAINED_EVIDENCE','ROMA_RGH',
            'LTH_P_ONLY_LEGAL','RC_ORIGINAL_ROMA','RC_RETRIEVAL_ONLY','JOINT_EVAL','RC_OPENED_EVAL']):continue
        if 'WORK_STATE' in p.name or 'HANDOFF' in p.name:continue
        content=src(p).read_text(); title=next((x.lstrip('# ') for x in content.splitlines() if x.startswith('# ')),p.stem)
        index.append(dict(date='2026-09-'+date[1],title=title,report=str(p.relative_to(ROOT)),
            sha256=SOURCES[str(p.relative_to(ROOT))]['sha256'],note='historical report; current conclusions follow this consolidated summary'))
    csvout('report_index',index)
    src('plan/RC_NEW_HYP_PAPER_SCOPE_FREEZE_V1_20260915.md')
    src('results/rc_new_hyp_external_head_freeze_v1/bundle.json')
    src('reports/REPORT_NEW_HYP_GROZI120_EXTERNAL_CONFIRMATION_V1_20260913.md')
    src('reports/REPORT_NEW_HYP_RPC_EXTERNAL_CONFIRMATION_V1_20260914.md')
    src('reports/REPORT_NEW_HYP_ISIC_FROZEN_TRANSFER_V1_20260914.md')
    text=src('programs/rc_all_results_20260916_report_template.md').read_text()
    tokens={
      'FIXED_TABLE':mtable(['同一套模型','EVAL32 /32','EVAL128 /128'],[[r['model'],r['EVAL32'],r['EVAL128']] for r in fixedrows]),
      'H593_TABLE':mtable(['模型','正确/593','准确率','对 RAW 救/损','对 RAW 净增'],[
          [m,str(six['counts'][m])+'/593',f"{six['counts'][m]/593*100:.2f}%",
           f"{sum(not x['correct']['RAW'] and x['correct'][m] for x in h593['rows'])}/{sum(x['correct']['RAW'] and not x['correct'][m] for x in h593['rows'])}",six['counts'][m]-426]
          for m in ['RAW','ALL_COST4','GROUP_BASE','COST1','ALL_CE','RAW2_CE','COST1_CBIND','ALL_CE_CBIND']]),
      'H593_COMPARE':mtable(['基线→新模型','救/损','净增','等组差 pp','95%组区间 pp'],[
          [r['baseline']+' → '+r['model'],f"{r['rescue']}/{r['loss']}",r['net'],f"{r['group_mean_pp']:.2f}",f"[{r['CI95_low_pp']:.2f}, {r['CI95_high_pp']:.2f}]"] for r in hcomp]),
      'FOLDS':mtable(['折','测试图','训练候选图','RAW','COST4','GROUP','COST1','CE'],[
          [r[k] for k in ['fold','heldout','nominal_train','RAW','ALL_COST4','GROUP_BASE','COST1','ALL_CE']] for r in foldrows]),
      'SPATIAL':mtable(['实验','target有连通H1 /32','REAL胜 strongest wrong /32','相对 query-only 净增','门'],[
          [r['experiment'].replace('rc_',''),r['target_connected'],r['correct'],r['vs_query_only_net'],'NO-GO'] for r in gates]),
      'EXTERNAL_MATRIX':mtable(['模型 / 对照','GroZi /480','RPC /600','ISIC /537'],[
          [m]+[ext[p]['counts'][m] for p in ['GroZi480','RPC600','ISIC537']]
          for m in ['RAW','COST4','GROUP_COST4','COST1','CE','RAW2_CE','COST1_CBIND','CE_CBIND','COST4_CBIND','GROUP_COST4_CBIND','RAW2_CE_CBIND']]),
      'EXTERNAL_FULL':'\n\n'.join('### '+panel+'\n\n'+mtable(['模型','正确','准确率','MRR','对RAW救/损'],[
          [r['model'],f"{r['correct']}/{r['queries']}",f"{r['accuracy_percent']:.2f}%",f"{r['MRR']:.5f}",f"{r['rescue_vs_RAW']}/{r['loss_vs_RAW']}"]
          for r in external if r['panel']==panel]) for panel in ext),
      'EXTERNAL_COMPARE':mtable(['面板','基线→模型','救/损','净增','等组差 pp','95%组区间 pp'],[
          [r['panel'],r['baseline']+' → '+r['model'],f"{r['rescue']}/{r['loss']}",r['net'],f"{r['group_mean_pp']:.2f}",f"[{r['CI95_low_pp']:.2f}, {r['CI95_high_pp']:.2f}]"]
          for r in comparisons if r['model']=='COST1' or (r['baseline']=='COST1' and r['model']=='CE')]),
      'INTERNAL_TABLE':mtable(['实验来源','面板','模型','正确','对RAW救/损'],[
          [link(r['source'],r['source'].split('/')[1]),r['panel'],r['model'],f"{r['correct']}/{r['queries']}",f"{r['rescue_vs_RAW']}/{r['loss_vs_RAW']}"] for r in internal]),
      'EARLY_TABLE':mtable(['实验','模型','REAL EVAL正确 /32','对RAW救/损'],[
          [link(r['source'],r['experiment']),r['model'],r['correct'],f"{r['rescue_vs_RAW']}/{r['loss_vs_RAW']}"]
          for r in legacy if r['panel']=='EVAL' and r['arm']=='REAL']),
      'OLD_INTERVENTIONS':mtable(['ORIGINAL7干预','正确 /128','对RAW救/损','MRR'],[
          [r['arm'],r['correct'],f"{r['rescue_vs_RAW']}/{r['loss_vs_RAW']}",f"{r['MRR']:.5f}"] for r in interventions]),
      'REPORT_INDEX':mtable(['日期','归档报告'],[[r['date'],link(r['report'],r['title'])] for r in index]),
      'COUNTS':f"本次逐图复算覆盖 {len(CHECKS)} 个结果/面板检查条目及原128干预，导出 {len(internal)+len(fixed)+len(external)+len(legacy)+len(interventions)} 行模型汇总（包含不同账本的重复基线，不是独立实验数）；索引 {len(index)} 份历史报告。"}
    for key,value in tokens.items():text=text.replace('{{'+key+'}}',value)
    assert not re.search(r'\{\{[A-Z_]+\}\}',text)
    (OUT/'complete_results_zh.md').write_text(text)
    make_xlsx()
    make_plot(six,ext)
    (OUT/'style.html').write_text('<style>body{font:17px/1.65 system-ui,sans-serif;color:#183047;max-width:1480px;margin:auto;padding:30px;background:#fafcfe}h1,h2,h3{color:#087f8c}a{color:#087f8c}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:14px;margin:22px 0}td,th{border:1px solid #d9e3eb;padding:8px 11px}th{background:#edf4f7}tr:nth-child(even){background:#f0f5f8}img{max-width:100%}code{white-space:pre-wrap}</style>')
    subprocess.run(['pandoc','complete_results_zh.md','--standalone','--toc','--toc-depth=2','--metadata','title=药盒与外部数据集实验总账','--include-in-header=style.html','-o','complete_results_zh.html'],cwd=OUT,check=True)
    subprocess.run(['pandoc','complete_results_zh.md','--standalone','--toc','--toc-depth=2','-o','complete_results_zh.docx'],cwd=OUT,check=True)
    src(__file__)
    (OUT/'source_manifest.json').write_text(json.dumps(SOURCES,ensure_ascii=False,indent=2)+'\n')
    (OUT/'validation.json').write_text(json.dumps(dict(status='RESULTS_DOSSIER_COUNTS_AND_LINKED_EVIDENCE_VERIFIED',checks=CHECKS,
        source_count=len(SOURCES),source_scope='current RAW/RoMa/new HYP medicine development and completed external panels',
        bootstrap_intervals='sealed results reused, no resampling',training_updates=0,new_inference_calls=0,
        historical_diagnostic_proofs='cited archived reports, no new solver calls'),ensure_ascii=False,indent=2)+'\n')
    import xml.etree.ElementTree as ET
    for name in ['all_results.xlsx','complete_results_zh.docx']:
        with zipfile.ZipFile(OUT/name) as z:
            assert z.testzip() is None
            for item in z.namelist():
                if item.endswith('.xml'):ET.fromstring(z.read(item))
    archive=OUT/'new_HYP_complete_results_20260916.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p!=archive:z.write(p,p.name)
    print(json.dumps(dict(output=str(OUT),checked_panels=len(CHECKS),internal_rows=len(internal),fixed_rows=len(fixed),
        external_rows=len(external),early_arms=len(legacy),historical_reports=len(index),sources=len(SOURCES),archive_sha256=sha(archive)),ensure_ascii=False))


def make_plot(six,ext):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/new_hyp_all_results_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf')
    plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'path'})
    fig,axs=plt.subplots(2,2,figsize=(15,10),facecolor='#FAFCFE')
    panels=[('药盒/商品 H593：开发 OOF5',593,[426,440,481,486])]+[
        (p+('：探索' if p.startswith('ISIC') else '：外部确认'),len(d['rows']),[d['counts'][m] for m in ['RAW','COST4','COST1','CE']]) for p,d in ext.items()]
    for ax,(title,n,counts) in zip(axs.flat,panels):
        ax.set_facecolor('#FAFCFE');bars=ax.bar(['RAW','COST4','COST1 主模型','CE 次模型'],[c/n*100 for c in counts],color=['#A8B5C5','#507DAD','#087F8C','#C18B32'],width=.62)
        ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('准确率 (%)');ax.set_title(title,pad=18,fontsize=15)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
        for bar,c in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,f'{c}/{n}\n{c/n*100:.2f}%',ha='center',fontsize=11)
    fig.suptitle('相同证据接口和小型头：成本训练改动的内部与跨数据集结果',fontsize=21,y=.99)
    fig.text(.05,.015,'H593 为五套折内头；外部三集使用同一批全 H593 固定头。各面板图库、分组与证据级别独立，不能合并分母。',fontsize=11,color='#586B7D')
    fig.tight_layout(rect=[.02,.045,.99,.955],h_pad=3,w_pad=3)
    for extn in ['png','pdf','svg']:fig.savefig(OUT/('main_results.'+extn),dpi=180,facecolor='#FAFCFE')
    plt.close(fig)


if __name__=='__main__':main()
