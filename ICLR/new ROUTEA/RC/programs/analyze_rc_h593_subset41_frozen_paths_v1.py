#!/usr/bin/env python3
"""Read sealed CPU replay artifacts; independently recount decisions and outcomes."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_subset41_frozen_paths_v1'

def read(p):
    return json.loads(Path(p).read_text())

def binding(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def checked(b):
    assert binding(b['path']) == b, b['path']
    return read(b['path'])

v = read(OUT / 'validation.json')
assert v['status'] == 'SUBSET41_FIXED_COST1_PATH_JOIN_PASS'
x = checked(v['result'])
a = checked(x['authority'])
for b in a['sources']:
    assert binding(b['path']) == b
seal = checked(x['seal'])
heads = checked(a['heads'])['heads']
old = {r['query_id']: r for r in checked(a['old_result'])['rows']}
labels = {r['physical_row']: r['identity'] for r in checked(a['gallery'])['records']}
records = {r['index']: r for r in x['rows']}
assert len(records) == 41 and len({r['component'] for r in records.values()}) == 30
counts = {k: Counter() for k in x['summary']}
logits_checked = 0
max_margin_error = 0.0
query_validations = []
for vb in seal['validations']:
    qv = checked(vb)
    assert qv['status'] == 'SUBSET41_COST1_FIXED_PATH_CPU_PASS'
    assert qv['native_all127_logits_bit_exact'] and qv['authority'] == x['authority']
    p = checked(qv['payload'])
    assert binding(p['features']['path']) == p['features']
    r = records[p['index']]
    row = p['row']
    qid = row['query_id']
    assert qid == r['query_id']
    assert p['head'] == heads[qid] and p['fold'] == old[qid]['fold']
    assert p['models']['FULL_UV/native/HR1'] == p['models']['M_ONLY/native/HR1'] == p['head']['native']
    axis = row['candidate_physical_rows']
    challengers = row['challenger_positions']
    assert len(axis) == 128 and len(set(axis)) == 128 and len(challengers) == 127
    assert challengers == [i for i in range(128) if i != row['winner']]
    assert set(p['models']) == set(counts) == set(r['models'])
    raw_correct = labels[axis[row['winner']]] == old[qid]['identity']
    native_correct = labels[p['head']['native']['selected']] == old[qid]['identity']
    assert (raw_correct, native_correct) == (r['raw_correct'], r['native_correct'])
    for name, prediction in p['models'].items():
        z = [float.fromhex(t) for t in prediction['logits_hex']]
        assert len(z) == 127 and all(math.isfinite(t) for t in z)
        k = max(range(127), key=z.__getitem__)
        selected = axis[challengers[k]] if z[k] > 0 else axis[row['winner']]
        d = r['models'][name]
        correct = labels[selected] == old[qid]['identity']
        assert selected == prediction['selected'] == d['selected'] and correct == d['correct']
        assert d['switched'] == (selected != axis[row['winner']])
        if r['target_in_C128']:
            scores = {axis[row['winner']]: 0.0, **{axis[i]: value for i, value in zip(challengers, z)}}
            targets = [i for i in axis if labels[i] == old[qid]['identity']]
            assert len(targets) == 1
            margin = scores[targets[0]] - max(value for i, value in scores.items() if i not in targets)
            assert abs(margin - d['target_margin']) < 1e-12
            err = abs(sum(d['target_margin_contributions'].values()) - margin)
            max_margin_error = max(max_margin_error, err)
            assert err < 2e-10
        c = counts[name]
        c['correct'] += correct
        c['retained_original_rescues'] += correct and native_correct and not raw_correct
        c['rescue_vs_native'] += correct and not native_correct
        c['break_vs_native'] += not correct and native_correct
        c['rescue_vs_raw'] += correct and not raw_correct
        c['break_vs_raw'] += not correct and raw_correct
        c['changed_vs_native'] += selected != p['head']['native']['selected']
        logits_checked += len(z)
    query_validations.append(qv)
assert {k: dict(c) for k, c in counts.items()} == x['summary']
assert sum(r['raw_correct'] for r in records.values()) == 32
assert sum(r['native_correct'] for r in records.values()) == 35
report = dict(status='SEALED_DECISIONS_LABELS_COUNTS_AND_MARGIN_RECOUNT_PASS', result=v['result'], script=binding(__file__),
              queries=41, groups=30, model_paths=len(counts), logits_recounted=logits_checked,
              target_absent=sum(not r['target_in_C128'] for r in records.values()),
              original_rescues=x['original_rescue_ids'], maximum_margin_recount_error=max_margin_error,
              native_all127_bit_exact_queries=len(query_validations),
              original_numpy_logit_error_max=max(q['max_numpy_logit_error'] for q in query_validations),
              original_numpy_scalar_error_max=max(q['max_numpy_scalar_error'] for q in query_validations),
              original_scalar_checks=sum(q['scalars_checked'] for q in query_validations),
              no_new_fit=True, no_new_gpu_forward=True, full593_unchanged=True)
(OUT / 'interpretation_validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

lines = ['# 冻结COST1头：41张共同子集的通路定位', '',
'已完成5158914首图、5158917其余40图、5158918汇总；5158915为自动推进任务。首图57秒，后40图55–66秒，均正常退出。', '',
'使用原H593五折中每张query对应的held-out COST1冻结参数、自然RAW C128和HOLD=0规则。RAW为32/41，原头35/41，救3损0。41张来自三类缓存共同完成的固定子集，覆盖30组，3张目标不在C128。这不是新的独立测试，也不替换全593或旧32/128的成绩。', '',
'38个缓存臂×2种作用范围，加9个既有算子对照，共85条读出。全部41张原生127分数逐位复现；本报告另从封存分数重算442595个候选分数对应的决策、标签和所有汇总，验证文件为 interpretation_validation.json。中间四评分、六列输入、参数及全部候选分数均保留。', '',
'## 哪些改动影响原3次纠错', '',
'M_ONLY仅改变整体质量及其对评分的缩放，保留原生局部加权/内容比较；FULL_UV替换两侧完整权重并重新汇聚。二者均沿用原头，不训练、不选阈值。', '',
'|通路改动|M_ONLY 正确 / 保留原纠错|FULL_UV 正确 / 保留原纠错|', '|---|---:|---:|']
selected = [
 ('原生最终阶段', 'native/HR1'), ('只读粗匹配阶段', 'native/COARSE'),
 ('质量预测头入口置零单图外观A', 'inside/A0J1P1/HR1'),
 ('质量预测头入口置零双图交互J', 'inside/A1J0P1/HR1'),
 ('质量预测头入口置零匹配位置编码P', 'inside/A1J1P0/HR1'),
 ('去掉粗阶段置信度初值', 'inside/NO_COARSE_LOGIT/HR1'),
 ('去掉全部后续置信度增量', 'inside/NO_REFINER_DELTA/HR1'),
 ('query灰度', 'visual/Q_GRAY/HR1'), ('reference灰度', 'visual/R_GRAY/HR1'),
 ('query低通', 'visual/Q_LOWPASS/HR1'), ('reference低通', 'visual/R_LOWPASS/HR1'),
 ('query分块打乱', 'visual/Q_SHUFFLE/HR1'), ('reference分块打乱', 'visual/R_SHUFFLE/HR1')]
for title, arm in selected:
    m, f = counts['M_ONLY/' + arm], counts['FULL_UV/' + arm]
    lines.append(f"|{title}|{m['correct']}/41；{m['retained_original_rescues']}/3|{f['correct']}/41；{f['retained_original_rescues']}/3|")
lines += ['',
'这次将上游干预接回了实际纠错决策：粗阶段置信度和质量预测头接收的双图交互通路，支撑了这个子集原有的3次纠错；后续置信度细化增量并非保住这3次纠错所必需。单图外观直接输入置零仍保留纠错，不能推出外观信息完全无用，因为J、P本来仍含外观信息。', '',
'低通只施加于RoMa输入；ColNomic tokens与内容匹配保持原始值。仅让低通产生的质量取代M，就足以使3次纠错全部丢失。因此影响可以经过整体质量通路传到最终决策，不需要先损坏ColNomic内容或改变局部汇聚。低通为16×16均值池化后插值回原尺寸，是强干预；不能单独归因于文字、logo、某种纹理，也不能将其和灰度效果当作等强度因果比较。', '',
'## 原3次纠错怎样消失', '',
'下表为正确候选相对HOLD=0的分数，比较对象固定，避免更换最强错误候选导致逐项贡献含义变化。', '',
'|query|原生头|粗阶段完整权重|仅用query低通的M|仅用J置零的M|', '|---|---:|---:|---:|---:|']
for row in x['rows']:
    if row['original_query_id'] not in x['original_rescue_ids']:
        continue
    keys = ('OPERATOR/NATIVE','FULL_UV/native/COARSE','M_ONLY/visual/Q_LOWPASS/HR1','M_ONLY/inside/A1J0P1/HR1')
    values = '|'.join(f"{row['models'][k]['target_logit']:+.4f}" for k in keys)
    lines.append(f"|{row['original_query_id']}|{values}|")
lines += ['',
'query低通和J置零这两项M_ONLY干预中，三张都回到HOLD，保留原RAW错误；不是正确候选退出C128。质量通路改变后，原头不再给正确挑战者足以越过零阈值的分数。这能解释已观测纠错为何消失，但不能将整个作用归给单独一个线性系数，因为M改变会重算所有依赖它的输入。', '',
'## 整体质量与局部权重怎样配合', '',
'沿用原头的既有算子对照：去掉M的四种组合均32/41、原3次救回均丢失；只保留M而用自由内容为34/41、保留2/3；保留M和reference权重、去掉query局部加权为35/41、保留3/3；固定自由内容命中位置、仅保留reference权重数值也为35/41、保留3/3。后一项不证明完整593可无损删去reference对命中位置的影响。', '',
'## 可执行结论与边界', '',
'本子集支持优先验证“粗阶段双图交互产生的质量＋原冻结纠错头”是否能够在更大面板维持收益，并真正提前结束后续计算。当前读出使用已完成的完整前向缓存，尚未实测提前退出的加速或确认可删除整个refiner。坐标GRID64为34/41、GRID16反而35/41，不能概括成精度越高就越好。', '',
'J与P在模型里有依赖关系：P由原生交互特征匹配而来；J置零仅发生在质量预测头入口，P和原生几何仍保留。置信度增量干预也保留原生细化证据。故这些是固定参数下的条件干预，输入分布变化可能参与下降，不构成所有架构都必须含该通路的必要性证明。', '',
'没有臂超过原35/41。这轮得到的是纠错证据来源和可简化环节的定位，不是新的全593准确率提升。当前粗/细特征直接融合训练仍是另一个问题；本结果不证明其不能替代匹配器。', '',
'完整逐臂表：reports/REPORT_H593_SUBSET41_FROZEN_PATHS_20260923.md；封存结果：results/rc_h593_subset41_frozen_paths_v1/result.json。']
path = ROOT / 'reports/REPORT_H593_SUBSET41_FROZEN_PATH_INTERPRETATION_20260923.md'
path.write_text('\n'.join(lines) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
print(path)
