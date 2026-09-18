#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs')]
import materialize_rc_new_hyp593_inputs_v1 as M
OUT=ROOT/'results/rc_original_mixed96_group_risk_v1'
v=M.read(OUT/'result_validation.json');M.need(v['status']=='ORIGINAL_MIXED96_GROUP_INDEPENDENT_RESULT_PASS' and v['result']==M.bind(OUT/'result.json'),'VALIDATED_RESULT_ONLY');r=M.read(OUT/'result.json')
lines=['# 原PAIR64＋FULL32：单因素组权重结果','', '原ec7及160图预测直接复用已验证封存，原头训练更新为0。唯一新头保留原96张训练图、原PAIR+FULL目标及1:1混合、七参数、FP64、2000步，仅将两个训练池各自的图像平均改为组平均。', '', '| 面板 | RAW | 原ec7 | GROUP_MIXED96 | 对原头救回／损失／净增 |','| --- | ---: | ---: | ---: | --- |']
for name in ('EVAL32','EVAL128'):
 p=r['panels'][name];s=p['scores'];c=p['comparisons']['ORIGINAL7__to__GROUP_MIXED96'];n=p['population'];lines.append(f"| {name} | {s['RAW']['correct']}/{n} | {s['ORIGINAL7']['correct']}/{n} | {s['GROUP_MIXED96']['correct']}/{n} | {c['rescue']}／{c['loss']}／{c['rescue']-c['loss']:+d} |")
lines+=['', '每套面板分别按观察净增判定，不要求零损失，也不根据评价重新选参数。以下区间描述组间不确定性，不改写观察净增计数：','']
for name,p in r['panels'].items():
 c=p['comparisons']['ORIGINAL7__to__GROUP_MIXED96'];lines.append(f"- {name}：等组差{c['equal_component_difference']:.6f}；组bootstrap95% {c['component_bootstrap95']}；自然C128召回{p['recall_C128']}/{p['population']}。")
lines+=['', '本次不重跑593五折。原32/128与593是不同评价协议，不能把单头160图的结果当作593新成绩。全部都是历史已查看开发数据，不是外部确认；没有自动替换原模型，也没有宣称统一理论已证。','', '[结果](../results/rc_original_mixed96_group_risk_v1/result.json)，[独立验证](../results/rc_original_mixed96_group_risk_v1/result_validation.json)，[原训练输入封存](../results/rc_original_mixed96_group_risk_v1/train_input_seal.json)。']
p=ROOT/'reports/REPORT_ORIGINAL_MIXED96_GROUP_RISK_V1_20260911.md'
with p.open('x') as f:f.write('\n'.join(lines)+'\n')
print(str(p),flush=True)
