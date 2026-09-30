#!/usr/bin/env python3
"""Read-only second accounting of completed19-query upstream attribution."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

RC=Path(__file__).resolve().parents[1]
ROOT=RC/'results/rc_m_upstream_event_closure_v1'
OUT=ROOT/'completion_review_20260930'

def read(p):return json.loads(Path(p).read_text())
def binding(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(b):
    assert binding(b['path'])=={k:b[k] for k in ['path','sha256']};return Path(b['path'])
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)

def main():
    validation=read(ROOT/'validation.json');result=read(checked(validation['result']))
    protocol=read(checked(validation['protocol']));bp=read(checked(protocol['bridge_protocol']))
    items={x['index']:x for x in read(checked(bp['inputs']))['rows']}
    labels={x['query_id']:x for x in read(checked(bp['original_label_source']))['rows']}
    directions=['X_PLUS','X_MINUS','Y_PLUS','Y_MINUS']
    coeffs={
        'phase':{'phase_GLOBAL_minus_LOCAL':{a:v/4 for d in directions for a,v in [(f'GLOBAL_{d}',1),(f'LOCAL_{d}',-1)]},
                 'amplitude_NATIVE_minus_PERMUTE':{'NATIVE':1,'AMP_PERMUTE':-1},
                 'amplitude_NATIVE_minus_FLAT':{'NATIVE':1,'AMP_FLAT':-1},
                 'own_native_minus_ORIGINAL_HR1':{'NATIVE':1,'ORIGINAL_HR1':-1}},
        'context':{'context_matched_alignment':{'NATIVE':.5,**{d+'__'+b:v/8 for d in directions for b,v in [('111',1),('001',-1),('110',-1)]}},
                   'context_P_fixed_move_AJ':{'NATIVE':-1,**{d+'__110':.25 for d in directions}},
                   'context_P_only_moved':{'NATIVE':-1,**{d+'__001':.25 for d in directions}},
                   'context_all_moved_grid_control':{'NATIVE':-1,**{d+'__111':.25 for d in directions}},
                   'own_native_minus_ORIGINAL_HR1':{'NATIVE':1,'ORIGINAL_HR1':-1}}}
    maximum=0.;cases=[];groups={};row_count=0;decisions=0
    for row,source in zip(result['rows'],result['sources'],strict=True):
        d=read(checked(source));item=items[row['index']];label=labels[d['query_id']]
        assert d['index']==row['index'] and d['axis']==item['post_row']['candidate_ids']
        targets=[k for k,x in enumerate(item['post_row']['candidate_identities']) if x==label['identity']]
        wrong=[k for k in range(128) if k not in targets]
        for arm,v in d['worlds'].items():
            for model in ['NATIVE7','M_FREE','POST']:
                x=v['POST'] if model=='POST' else v['external'][model]
                scores=x['logits128'];w=d['winner'];best=max((k for k in range(128) if k!=w),key=lambda k:scores[k])
                prediction=best if scores[best]>0 else w
                assert prediction==x['prediction_position']==row['prediction'][model][arm]
                assert (prediction in targets)==row['correct'][model][arm]
                decisions+=1
        for name,cs in coeffs[row['family']].items():
            effect=np.array([math.fsum(c*math.log(d['worlds'][a]['M'][j]) for a,c in cs.items()) for j in range(128)])
            stored=row['effects'][name];error=float(np.max(abs(effect-stored['candidate_logM_effect'])))
            maximum=max(maximum,error);assert error<1e-12
            if targets:
                t=targets[0];fixed=max(wrong,key=lambda k:d['worlds']['NATIVE']['M'][k])
                recomputed={'target_vs_mean_wrong':float(np.mean([effect[t]-effect[k] for k in wrong])),
                    'target_vs_fixed_native_wrong':float(effect[t]-effect[fixed]),
                    'candidate_common_level':float(effect.mean())}
                for metric,value in recomputed.items():
                    maximum=max(maximum,abs(value-stored[metric]));assert abs(value-stored[metric])<1e-12
                    key=(row['family'],row['cohort'],name,metric)
                    groups.setdefault(key,{}).setdefault(row['component'],[]).append(value)
        if row['family']=='phase' and row['index'] in [41,116]:
            t=targets[0];w=d['winner'];record=dict(index=row['index'],query_id=d['query_id'],original_query_id=label['original_query_id'],target=t,RAW_winner=w,worlds={})
            for a in ['NATIVE','AMP_FLAT','AMP_PERMUTE','ZERO']:
                v=d['worlds'][a];record['worlds'][a]=dict(target_M=v['M'][t],RAW_wrong_M=v['M'][w],
                    target_to_RAW_wrong_M_ratio=v['M'][t]/v['M'][w],logM_gap=math.log(v['M'][t]/v['M'][w]),
                    target_content=v['L'][t],wrong_content=v['L'][w],POST_target_action=v['POST']['logits128'][t],
                    POST_prediction=v['POST']['prediction_position'])
            cases.append(record)
        row_count+=1
    for (family,cohort,name,metric),gs in groups.items():
        mean=float(np.mean([np.mean(v) for v in gs.values()]));s=result['group_stats'][family][cohort][name][metric]
        assert s['groups']==len(gs);maximum=max(maximum,abs(mean-s['mean']));assert abs(mean-s['mean'])<1e-12
    tables={}
    for family in ['phase','context']:
        tables[family]={}
        for model in ['NATIVE7','M_FREE','POST']:
            s=result['summary'][family]['ALL_CHANGED11'][model]
            tables[family][model]={a:dict(original_rescues_retained=len(x['original_rescues_retained']),
                original_rescues_lost=x['original_rescues_lost'],RAW_breaks=x['breaks'],correct=x['correct']) for a,x in s.items()}
    write(OUT/'result.json',dict(status='UPSTREAM_COMPLETION_SECOND_ACCOUNTING_PASS',source=binding(ROOT/'result.json'),
        code=binding(Path(__file__)),query_family_rows=row_count,decisions_recomputed=decisions,maximum_arithmetic_error=maximum,
        tables=tables,cases=cases,scope='Opened19, naturalC128, same frozen held-fold endpoints. Selected original-event cohort is separate from fixed controls; no unique-cause or population improvement claim.'))
    lines=['# 19图上游归因：完成检查与独立解释','',
        '原11图任务链及新增8图采集、相位／幅度、配准、冻结端点回放均已完成。每张完整自然ColNomic C128；对应H593前128归因面板的全部原纠错/误伤事件，加8张固定对照。没有新训练，也不是新的H593准确率试验。','',
        '## 原纠错的保留情况','',
        '| 干预 | 外部完整头 NATIVE7 | 外部简化头 M_FREE | 内部 POST |','|---|---:|---:|---:|',
        '| 原有纠错 | 10 | 10 | 8 |',
        '| 所测GLOBAL/LOCAL相位扰动 | 10 | 10 | 8 |',
        '| 幅度打平 AMP_FLAT | 10 | 10 | 6 |',
        '| 幅度置换 AMP_PERMUTE | 10 | 10 | 6 |',
        '| P整条输入置零 | 7 | 6 | 2 |',
        '| 所测配准移位 | 10 | 10 | 8 |','',
        '表中只数原纠错保留，不能当作总正确数。POST从HR1换为coarse-native时，84号OUTCOME-0334产生一次原先没有的误伤；部分干预会修复它。原生基线切换与干预效应必须分开，完整逐世界break已保留。','',
        '## 扩样后新增的解释','',
        '1. **局部幅度分配确实能沿M路径影响实际内部纠错。** AMP_PERMUTE在近似保持每个P向量方向、保留幅度数值集合的同时，改变幅度落在哪个位置。OUTCOME-0476和DIFFICULT-0013在冻结POST中由SWITCH退回错误RAW答案；外部两种冻结头仍纠错成功。这定位到所测模型依赖的幅度分配及其下游使用，不能等同于唯一身份语义来源。','',
        '2. **对平均错误候选有优势，不代表能排除最强混淆候选。** 11张原改判集合（9组）里，GLOBAL−LOCAL对target相对平均wrong的logM效应约+0.02691，原幅度相对置换约+0.12985；但相对原生M最强wrong分别约−0.00915、−0.05873，探索区间均跨0。固定对照也没有建立对最强M错误候选的稳定优势。最强M错误候选不必等于最终action的最强错误候选；二者分别保留。','',
        '3. **配准效应仍有条件性。** 原改判集合对平均wrong的匹配配准效应约+0.01440；固定对照为−0.00660，区间跨0。相对于原生M最强wrong，原改判集合约−0.00522，区间跨0。所测配准世界均保留原纠错，不能把配准直接写成这些纠错的必要性质。','',
        '## 0476：比例更好，内部纠错仍会消失','']
    c=next(x for x in cases if x['index']==41);n=c['worlds']['NATIVE'];a=c['worlds']['AMP_PERMUTE']
    lines += [f"原生target/RAW-wrong的M比值为{n['target_to_RAW_wrong_M_ratio']:.3f}，幅度置换后为{a['target_to_RAW_wrong_M_ratio']:.3f}，比例上升；但target的绝对M从{n['target_M']:.6f}降到{a['target_M']:.6f}，POST对target的action从{n['POST_target_action']:.6f}变为{a['POST_target_action']:.6f}，因此不再SWITCH。",
        '', '这是同一次上游干预经M导致原纠错消失的完整C128实例，同时否定“相对M优势更大就必然更好”的简单解释。它与POST读取绝对M并结合内容条件化相符；本次没有通过新的交叉回放把绝对水平与候选间响应单独分摊，不能把相符写成唯一中介分解。','',
        '## 仍未闭合','',
        '尚未把P内部关系与其对A/J内容的配合完全交叉隔离；相加接口只观察实际总和也限制唯一分摊。还没有证明哪种性质稳定排除最强混淆身份，或者解释原系统全部收益。不能把工作流的PASS当作根因闭合。没有重新提交训练或参数搜索。','',
        f'本次独立重算{row_count}个query×family记录、{decisions}个完整C128决策，核对逐候选logM对比及组均值；最大算术误差{maximum:.3g}。原自动验收最大打分误差3.55e-15。',
        '', '详细数据见同目录result.json；上游正式结果见../result.json、../validation.json。']
    text='\n'.join(lines)+'\n';OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'report.md'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    print(json.dumps(dict(status='UPSTREAM_COMPLETION_SECOND_ACCOUNTING_PASS',decisions=decisions,maximum_error=maximum)))

if __name__=='__main__':main()
