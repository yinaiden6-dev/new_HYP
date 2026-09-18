#!/usr/bin/env python3
"""Render the requested RAW -> original -> optimized ledger only after validation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs')]
import materialize_rc_new_hyp593_inputs_v1 as M
OUT=ROOT/'results/rc_fixed_panels_train269_group_risk_v1'
def main():
 v=M.read(OUT/'result_validation.json');M.need(v['status']=='FIXED_PANELS_INDEPENDENT_ACTION_RANK_COUNTS_PASS','VALIDATED_NEW_RESULTS_ONLY');M.need(v['result']==M.bind(OUT/'result.json'),'VALIDATED_RESULT_BINDING');r=M.read(OUT/'result.json')
 h=ROOT/'results/rc_h593_group_risk_strong_base_v1';hv=M.read(h/'result_validation.json');M.need(hv['status']=='GROUP_RISK_INDEPENDENT_ACTION_RANK_AND_GROUP_COUNTS_PASS' and hv['result']==M.bind(h/'result.json'),'VALIDATED593');old=M.read(h/'result.json')
 lines=['# 固定评价集：RAW、原组合模型、分组训练优化','', '所有主路径均为RAW自然C128、RoMa软可见性与ColNomic内容证据、七参数头及127 challenger HOLD/SWITCH。RoMa/ColNomic冻结，仅用query-reference检索身份标签训练小头。','', '| 评价协议 | RAW | 原组合模型 | 本轮GROUP模型 |','| --- | ---: | ---: | ---: |']
 for name,n in [('EVAL32',32),('EVAL128',128)]:
  s=r['panels'][name]['scores'];lines.append(f"| 原{name}固定面板 | {s['RAW']['correct']}/{n} | ec7 {s['ORIGINAL7']['correct']}/{n} | GROUP269 {s['GROUP269']['correct']}/{n} |")
 s=old['scores'];lines.append(f"| 新593图五折 | {s['RAW']['correct']}/593 | 原结构SMALL {s['SMALL_BASE']['correct']}/593；ALL {s['ALL_BASE']['correct']}/593 | GROUP_BASE {s['GROUP_BASE']['correct']}/593 |")
 lines+=['','593图的447来自五折分别重训，不是原ec7单头成绩，不能当作旧32/128改进。原结构较强SMALL基线为444；相同ALL训练图下是440→447（8救1损）。包括新增补偿的更强SMALL_CONST对照为446，447对它为4救3损、净增1。','', '本轮旧面板使用原训练身份及组范围内全部269张唯一图（32身份、32来源组），与两套评价面板在图像、身份、组及component上均不重叠。IMAGE269与GROUP269共用相同训练数据、七参数结构、FULL-C128 SIGN目标及2000步预算，唯一区别是图像等权与组等权。原ec7是原PAIR+FULL训练，其与新头之差不能全部归因于组权重。','']
 for name in ('EVAL32','EVAL128'):
  p=r['panels'][name];s=p['scores'];lines.append(f"## {name}完整读数\n\n| 模型 | 正确 | MRR（完整gallery） |\n| --- | ---: | ---: |")
  for k in ('RAW','ORIGINAL7','IMAGE269','GROUP269'):lines.append(f"| {k} | {s[k]['correct']}/{p['population']} | {s[k]['MRR']:.6f} |")
  lines.append('')
  for key,c in p['comparisons'].items():lines.append(f"- {key}：{c['rescue']}救回、{c['loss']}损失、净增{c['rescue']-c['loss']:+d}；组均差{c['equal_component_difference']:.6f}，组bootstrap95% {c['component_bootstrap95']}。")
  c=p['comparisons']['ORIGINAL7__to__GROUP269'];net=c['rescue']-c['loss'];lines.append(f"\n按当前允许损失、要求净增的开发标准，本面板{'达到观察净增' if net>0 else '尚未提高原模型正确数'}。自然C128召回{p['recall_C128']}/{p['population']}，缺席target的图未删除。\n")
 lines+=['这些面板与593数据历史上均被查看，属于开发复测，不是未触碰外部确认。模型从零训练且评价身份不进入本轮拟合，仍不能消除方法选择对历史评价的间接利用。没有自动替换原模型；正净增也不自动证明普遍理论或所有组受益。','', '[新结果](../results/rc_fixed_panels_train269_group_risk_v1/result.json)，[独立验证](../results/rc_fixed_panels_train269_group_risk_v1/result_validation.json)，[旧候选输入一致性](../results/rc_fixed_panels_train269_group_risk_v1/historic_input_parity.json)，[593结果](../results/rc_h593_group_risk_strong_base_v1/result.json)。']
 p=ROOT/'reports/REPORT_FIXED_PANELS_TRAIN269_GROUP_RISK_V1_20260911.md'
 with p.open('x') as f:f.write('\n'.join(lines)+'\n')
 print(str(p),flush=True)
if __name__=='__main__':main()
