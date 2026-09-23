#!/usr/bin/env python3
"""Population action-space audit of frozen GAP_BIAS2=492; no model fitting."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/rc_h593_gap_curve_v1'
OUT = ROOT/'results/rc_h593_population_opportunities_v1'
CATEGORIES = {
    'already_correct': '当前492模型正确',
    'target_absent': 'target不在自然C128',
    'locked_wrong_SWITCH': '原错误SWITCH锁定',
    'wrong_RAW_and_challenger': 'RAW和最高challenger均错误',
    'correct_challenger_still_HOLD': '最高challenger正确但未切换',
    'new_wrong_SWITCH': 'RAW正确但新增SWITCH误切',
}


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(b):
    assert bind(b['path']) == b
    return Path(b['path'])


def main():
    validation = read(SOURCE/'validation.json')
    assert validation['status'] == 'GAP_CURVE_ALL_COUNTS_PASS'
    result = read(checked(validation['result']))
    authority = read(checked(validation['authority']))
    gallery = read(checked(authority['public_sources']['gallery']))
    identity = {r['physical_row']:r['identity'] for r in gallery['records']}
    predictions = {}
    seals = []
    for fold in range(5):
        v = read(SOURCE/f'fold{fold}/validation.json')
        iv = read(SOURCE/f'fold{fold}/independent_validation.json')
        assert v['payload'] == iv['payload'] and iv['passed']
        p = read(checked(v['payload']))
        predictions.update({r['query_id']:r for r in p['predictions']})
        seals.append(v['payload'])
    rows = []
    for r in result['rows']:
        p = predictions[r['query_id']]
        z = [float.fromhex(x) for x in p['models']['COST1_FULL']['logits_hex']]
        top = max(range(127), key=z.__getitem__)
        axis = p['candidate_physical_rows']
        raw = identity[axis[p['winner']]] == r['identity']
        top_correct = identity[axis[p['challenger_positions'][top]]] == r['identity']
        present = any(identity[x] == r['identity'] for x in axis)
        assert present == r['target_in_C128']
        if r['correct']['GAP_BIAS2']:
            category = 'already_correct'
        elif not present:
            category = 'target_absent'
        elif z[top] > 0.0:
            category = 'locked_wrong_SWITCH'
        elif raw:
            category = 'new_wrong_SWITCH'
        elif top_correct:
            category = 'correct_challenger_still_HOLD'
        else:
            category = 'wrong_RAW_and_challenger'
        scores = [0.0]*128
        for j, position in enumerate(p['challenger_positions']):
            scores[position] = z[j]
        order = sorted(range(128), key=lambda j: (-scores[j], j))
        rank = next((i+1 for i,j in enumerate(order) if identity[axis[j]] == r['identity']), None)
        rows.append(dict(query_id=r['query_id'], original_query_id=r['original_query_id'],
                         fold=r['fold'], component=r['component'], category=category,
                         target_in_C128=present, RAW_correct=raw, original_top_correct=top_correct,
                         original_positive_SWITCH=z[top]>0.0, original_unified128_diagnostic_rank=rank))
    counts = Counter(r['category'] for r in rows)
    assert dict(counts) == dict(already_correct=492,target_absent=23,locked_wrong_SWITCH=15,
                               wrong_RAW_and_challenger=28,correct_challenger_still_HOLD=30,new_wrong_SWITCH=5)
    assert len(rows) == len({r['query_id'] for r in rows}) == 593
    blocked = [r for r in rows if r['category'] in ('locked_wrong_SWITCH','wrong_RAW_and_challenger')]
    accessible = counts['correct_challenger_still_HOLD']+counts['new_wrong_SWITCH']
    report = dict(status='POPULATION_OPPORTUNITIES_RECOUNT_PASS', baseline='GAP_BIAS2',
                  source=validation['result'], authority=validation['authority'], payloads=seals,
                  program=bind(__file__), counts=counts, total_queries=593, error_with_target_in_C128=78,
                  existing_gate_oracle_repair_capacity=accessible,
                  existing_gate_oracle_accuracy_ceiling=492+accessible,
                  locked_or_ranking_errors=43,
                  diagnostic_target_top5=sum(r['original_unified128_diagnostic_rank'] <= 5 for r in blocked),
                  diagnostic_target_top10=sum(r['original_unified128_diagnostic_rank'] <= 10 for r in blocked),
                  per_fold={str(f):dict(Counter(r['category'] for r in rows if r['fold']==f)) for f in range(5)},
                  limits=['Oracle action-space capacity is not an achievable accuracy prediction.',
                          'Unified128 diagnostic rank is not pipeline MRR and does not define a new top-K filter.',
                          'Opened H593 development: future fitting must use TRAIN identity labels only, with full593 evaluation.',
                          'Removing ranking/SWITCH locks is necessary for43 cases, not sufficient to fix them.'],
                  model_fits=0, external_GO=False)
    OUT.mkdir(exist_ok=True)
    (OUT/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'all593_categories.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# H593：按全体剩余错误重新界定优化范围','',
           '基线固定为GAP_BIAS2=492/593。不是围绕新490的个别错例选规则；统一检查原492模型的全部101例错误。', '',
           '| 类别 | 数量 | 当前固定最高challenger、锁定旧SWITCH的门能否修复 |','|---|---:|---|']
    for key in ('target_absent','correct_challenger_still_HOLD','new_wrong_SWITCH','wrong_RAW_and_challenger','locked_wrong_SWITCH'):
        lines.append(f"| {CATEGORIES[key]} | {counts[key]} | {'动作空间允许' if key in ('correct_challenger_still_HOLD','new_wrong_SWITCH') else '不能'} |")
    lines += ['', '扣除23例候选缺失，剩78例有target在候选内。其中35例在当前二选一动作空间内；另外43例需要重新选择候选或撤回旧错误SWITCH，继续单改HOLD门触及不到。527仅是当前动作空间的oracle上限，不是预计可达到的结果。', '',
              '43例中，以原COST1的RAW=0加127候选分数作统一诊断排序，27例target在前5，33例在前10。这里不是模型新成绩，不改现有MRR，也不据此设top-K截断。完整C128仍保留。', '',
              '## 下一步范围修正','',
              '1. 以全部593例净正确数及组件差为主指标，同时单列492原正确的损失、78例的救回、23例候选缺失。不得只优化或报告错误子集。',
              '2. 新模型的动作空间应覆盖完整C128候选竞争及撤回原错误SWITCH；停止仅由单个冲突样本推动后置门加项。原COST1与492继续作为冻结对照。',
              '3. 先对照历史完整候选训练、CE及内容头，明确哪项新信息或新学习目标此前没有隔离；解除限制本身不算创新，也不保证提升。',
              '4. 仍仅使用TRAIN检索身份监督、内外层分组隔离；无手工照片类型、框、mask或按query ID规则。外层剩余78例只用于总体诊断，不作为专门训练集。',
              '5. 暂不改变C128召回，23例另列召回瓶颈。H593已打开，后续结果仍属开发证据。', '',
              '历史full head、CE、特征消融和纯内容头不能被概括为“从未试过候选评分”；本轮只是明确当前这组HOLD后置实验覆盖不到43例。', '',
              '计数已由独立只读核算复核。本报告没有训练新模型或提交新任务。', '',
              '证据：`results/rc_h593_population_opportunities_v1/result.json`及`all593_categories.csv`。', '']
    (ROOT/'reports/REPORT_H593_POPULATION_OPPORTUNITIES_V1_20260921.md').write_text('\n'.join(lines))
    print(json.dumps({k:report[k] for k in ('status','counts','error_with_target_in_C128','existing_gate_oracle_repair_capacity','locked_or_ranking_errors','diagnostic_target_top5','diagnostic_target_top10')},ensure_ascii=False))


if __name__ == '__main__':
    main()
