#!/usr/bin/env python3
"""Exploratory readout of already sealed common queries; no GPU and no fitting."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_h593_existing_subset_analysis_v1'
def read(p):return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(b):assert bind(b['path'])==b;return Path(b['path'])
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)

def snapshot():
    p=OUT/'snapshot.json'
    if p.exists():return read(p)
    records=read(ROOT/'results/rc_crisp_manual_baseline_v1/H593_workers.json')['records']
    unique={x['roma']['payload']['path']:x['roma'] for x in records};total=0;scalars=0;statuses=collections.Counter()
    for bundle in unique.values():
        a=read(checked(bundle['receipt']));v=read(checked(bundle['validation']))
        assert a['payload']==v['payload']==bundle['payload'] and Path(bundle['payload']['path']).is_file()
        total+=a['query_count'];count=v.get('independently_recomputed_C4_scalars',v.get('independent_C4_scalars'))
        assert count==a['query_count']*128*4
        if 'candidate_occurrence_count' in a:assert a['candidate_occurrence_count']==a['query_count']*128
        scalars+=count;statuses[v['status']]+=1
    assert total==len(records)==593
    cat=read(ROOT/'cache/rc_h593_gpu_data_catalog_v1/catalog.json')
    common=[x for x in cat['rows'] if set(x['sources'])=={'inside','visual','coordinate'}]
    assert len(common)>=32
    value=dict(status='EXPLORATORY_COMPLETED_SUBSET_FIXED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        selection='All currently validated queries common to three families, using cache coverage only; no selection on outcomes',
        queries=len(common),indices=[x['index'] for x in common],sources=common,training=False,formal_all593_result=False,
        original_eval32_or_eval128=False,opened_historical_results=True,one_historical_row_schema_previewed=True,
        scientific_scope='Exploratory direct rankings from cached M and fixed free-content score; no HOLD/SWITCH, no new head',
        native_provenance=dict(queries=total,query_candidate_pairs=total*128,unique_shards=len(unique),
          C4_scalar_replays_recorded=scalars,statuses=dict(statuses),
          verification='Receipt and validation SHA checked, payload existence and bound SHA checked; large tensor payloads not rehashed',
          sources=dict(collections.Counter(x['roma']['payload']['path'].split('/results/')[1].split('/')[0] for x in records))))
    write(p,value);return value

def main():
    snap=snapshot();authority=read(ROOT/'registry/rc_h593_m_inside_authority_v1_20260922.json')
    old=read(checked(authority['old_result']));oldrows={x['query_id']:x for x in old['rows']}
    labels={x['physical_row']:x['identity'] for x in read(checked(authority['gallery']))['records']}
    workers=read(ROOT/'results/rc_crisp_manual_baseline_v1/H593_workers.json')['records']
    rows=[]
    def score(values,axis,target):
        best=max(range(128),key=lambda i:(values[i],-axis[i]));result=dict(correct=target is not None and best==target,selected=axis[best])
        if target is not None:
            wrong=max(values[i] for i in range(128) if i!=target)
            result['normalized_target_gap']=(values[target]-wrong)/(abs(values[target])+abs(wrong)+1e-12)
        return result
    for selected in snap['sources']:
        sources=selected['sources'];inside=read(checked(sources['inside']['paths_payload']))
        p=read(checked(sources['visual']['payload']));n=read(checked(sources['inside']['payload']))
        assert p['candidate_physical_rows']==n['candidate_physical_rows'] and p['query_id']==inside['query_id']==n['query_id']
        for st in authority['stages']:assert p['masses']['NATIVE'][st]==n['masses']['NATIVE'][st]
        oldrow=oldrows[p['query_id']];axis=p['candidate_physical_rows'];targets=[i for i,x in enumerate(axis) if labels[x]==oldrow['identity']]
        assert len(targets)<=1 and bool(targets)==oldrow['target_in_C128'];target=targets[0] if targets else None
        op=read(checked(p['operator_source']));assert op['candidate_physical_rows']==axis
        content=[x['real_score'] for x in op['modes']['M0Q0R0']['scores']]
        row=dict(index=selected['index'],query_id=p['query_id'],component=oldrow['component'],fold=oldrow['fold'],
            source=workers[selected['index']]['roma']['payload']['path'].split('/results/')[1].split('/')[0],
            target_in_C128=bool(targets),historical_correct={k:oldrow['correct'][k] for k in ('RAW','COST1_FROZEN_NATIVE','CE_FROZEN_NATIVE')},
            free_content=score(content,axis,target),comparisons={})
        for family,arms in [('visual',p['masses']),('inside',{b:{st:[x['branches'][b][st] for x in inside['records']] for st in authority['stages']} for b in authority['inside_branches']})]:
            for arm,stages in arms.items():
                for stage,masses in stages.items():
                    for readout,values in [('M',masses),('M_times_free_content',[m*f for m,f in zip(masses,content)])]:
                        row['comparisons']['/'.join((family,arm,stage,readout))]=score(values,axis,target)
        rows.append(row)
    summary={}
    for key in rows[0]['comparisons']:
        family,arm,stage,mode=key.split('/');base='/'.join((family,'NATIVE' if family=='visual' else 'A1J1P1',stage,mode))
        values=[r['comparisons'][key] for r in rows];valid=[x for x in values if 'normalized_target_gap' in x]
        rescued=sum(r['comparisons'][key]['correct'] and not r['comparisons'][base]['correct'] for r in rows)
        broken=sum(not r['comparisons'][key]['correct'] and r['comparisons'][base]['correct'] for r in rows)
        summary[key]=dict(n=len(rows),correct=sum(x['correct'] for x in values),rescues_vs_same_stage_native=rescued,breaks_vs_same_stage_native=broken,
             mean_normalized_target_gap=statistics.mean(x['normalized_target_gap'] for x in valid) if valid else None)
    result=dict(status='EXISTING_SUBSET_EXPLORATORY_READOUT_COMPLETE',snapshot=bind(OUT/'snapshot.json'),code=bind(__file__),
       queries=len(rows),groups=len({r['component'] for r in rows}),fold_counts=dict(collections.Counter(r['fold'] for r in rows)),
       source_counts=dict(collections.Counter(r['source'] for r in rows)),recall_present=sum(r['target_in_C128'] for r in rows),
       historical_correct={k:sum(r['historical_correct'][k] for r in rows) for k in rows[0]['historical_correct']},
       free_content_direct_correct=sum(r['free_content']['correct'] for r in rows),summary=summary,rows=rows,
       GPU_forwards=0,models_trained=0,formal_result_replaced=False,
       boundary='Current-completion sample, not a random new test. Direct M or M*content rankings do not evaluate the COST1/CE action. Zeroing internal components is a controlled predictor intervention at native geometry, not retraining or global necessity proof.')
    write(OUT/'result.json',result)
    lines=['# 已完成共同子集：无需新增 GPU 的机制探索','',f"共 {len(rows)} 张、{result['groups']} 个组；三类数据完整后纳入。不是旧 EVAL32/EVAL128，也不是全 H593 结论。",
      '',f"原生 H593 缓存：593 张，75 个分片，75904 个 query–candidate 配对；全部原生分片 receipt/validation 已核对。",
      '', '下表只比较候选直接排序，不经过 COST1/CE 小头和 HOLD/SWITCH。M×内容中的内容固定为完整 reference 的自由 MaxSim。没有训练或新增 GPU 前向。','',
      '|干预（HR1）|M 排第一正确数|M×固定自由内容正确数|相对原生 M×内容：救回/损失|','|---|---:|---:|---:|']
    for family in ('visual','inside'):
        arms=list(dict.fromkeys(k.split('/')[1] for k in summary if k.startswith(family+'/')))
        for arm in arms:
            m=summary[f'{family}/{arm}/HR1/M'];v=summary[f'{family}/{arm}/HR1/M_times_free_content']
            lines.append(f"|{family}/{arm}|{m['correct']}/{len(rows)}|{v['correct']}/{len(rows)}|{v['rescues_vs_same_stage_native']}/{v['breaks_vs_same_stage_native']}|")
    lines+=['','全部七阶段逐 query 数值见 `result.json`。此轮未运行坐标干预下的小头评测，不能把坐标数据已存在说成最终性能评测已完成。',
      '',f"当前子集历史 RAW/COST1/CE 决策正确数（仅做样本背景）：{result['historical_correct']}。这些数与表内直接排序是不同读出机制。",
      '', '样本量减少可加速机制定位；应按身份/组分布检查代表性。任何从本次探索选择的方法，需在其余未用于选择的数据上固定验证；不能据此缩改既有593评测分母。']
    report=ROOT/'reports/REPORT_H593_EXISTING_SUBSET_EXPLORATION_20260923.md';report.write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('status','queries','groups','fold_counts','source_counts','recall_present','historical_correct','free_content_direct_correct','GPU_forwards')},ensure_ascii=False))
    print('\n'.join(lines[8:]))

if __name__=='__main__':main()
