#!/usr/bin/env python3
"""Describe completed phase accounting and its original-rescue denominator.

Read saved decisions only. This does not relabel the 11-query panel as a
population estimate or infer absent tensors from compact summaries.
"""
import hashlib
import json
from pathlib import Path

RC = Path(__file__).resolve().parents[1]
ROOT = RC / 'results/rc_m_fine_c128_attribution_v1'


def bound(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    receipt = json.loads((ROOT / 'phase_validation.json').read_text())
    assert receipt['status'] == 'FINE_C128_PHASE_ACCOUNTING_PASS'
    source = Path(receipt['result']['path'])
    assert bound(source) == receipt['result']
    phase = json.loads(source.read_text())
    oldpath = RC / 'results/rc_m_coarse_path_correction_audit_v1/result.json'
    old = json.loads(oldpath.read_text())
    panel = {r['index'] for r in phase['rows']}
    models = ['NATIVE7', 'M_FREE', 'POST']
    coverage = {}
    comparisons = {}
    for model in models:
        rescue = [r for r in old['rows'] if not r['raw_correct'] and r['models'][model]['ORIGINAL_HR1']['correct']]
        breaks = [r for r in old['rows'] if r['raw_correct'] and not r['models'][model]['ORIGINAL_HR1']['correct']]
        coverage[model] = dict(
            original128_rescues=[r['index'] for r in rescue],
            original128_breaks=[r['index'] for r in breaks],
            covered_rescues=[r['index'] for r in rescue if r['index'] in panel],
            not_covered_rescues=[r['index'] for r in rescue if r['index'] not in panel],
            covered_breaks=[r['index'] for r in breaks if r['index'] in panel],
            original_rescue_names={str(r['index']): r['original_query_id'] for r in rescue})
        comparisons[model] = {}
        for cohort in ['MAIN8', 'SELECTED_CASES3']:
            rows = [r for r in phase['rows'] if r['cohort'] == cohort]
            originals = [r['index'] for r in rows if not r['raw_correct'] and r['correct'][model]['ORIGINAL_HR1']]
            worlds = {}
            for arm in rows[0]['prediction'][model]:
                changed = [r['index'] for r in rows if r['prediction'][model][arm] != r['prediction'][model]['NATIVE']]
                retained = [r['index'] for r in rows if r['index'] in originals and r['correct'][model][arm]]
                worlds[arm] = dict(correct=sum(r['correct'][model][arm] for r in rows),
                    prediction_changed_from_native=changed, original_rescues_retained=retained,
                    retention_fraction=len(retained)/len(originals) if originals else None)
            comparisons[model][cohort] = dict(total=len(rows), original_rescues=originals, worlds=worlds)
    result = dict(status='FINE_C128_PHASE_SCOPE_ANALYSIS_COMPLETE', source=receipt['result'],
        source_receipt=bound(ROOT / 'phase_validation.json'), original128=bound(oldpath),
        program=bound(Path(__file__)), coverage=coverage, comparisons=comparisons,
        all128_property_attribution_complete=False,
        boundary='Phase comparisons use complete natural C128 within each of eleven opened queries. '
        'MAIN8 has no original rescues. CASES3 was selected for historical correction outcomes. '
        'Missing coverage means not included in this frozen fine-property panel; it is not an exhaustive filesystem assertion that no other cache exists.')
    (ROOT/'phase_scope_analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# 相位／幅度完整候选核算：已完成部分与覆盖缺口', '',
        '固定原折模型、原自然 C128；只通过 M 及其派生项传播干预，原内容与局部权重形状保留。'
        '以下是完成的相位／幅度分支；配准分支另行核算。', '',
        '| 模型 | 原128纠错数 | 本轮覆盖的原纠错 | 尚未覆盖的原纠错索引 |',
        '|---|---:|---|---|']
    for model,c in coverage.items():
        lines.append(f"| {model} | {len(c['original128_rescues'])} | {c['covered_rescues']} | {c['not_covered_rescues']} |")
    lines += ['', '8张按历史固定规则选出的主面板，三模型原本均为6/8，没有新增纠错。'
              '因此该面板的原纠错保留率未定义，不能用0/0或3张案例替代分母。', '',
              '| 3张历史案例 | 原生 | 整体P编码置零 | 幅度平坦／置换 | 各方向GLOBAL／LOCAL相位扰动 |',
              '|---|---:|---:|---|---|']
    for model in models:
        w=comparisons[model]['SELECTED_CASES3']['worlds']
        amp=[w[a]['correct'] for a in ('AMP_FLAT','AMP_PERMUTE')]
        ph=[v['correct'] for a,v in w.items() if a.startswith(('GLOBAL_','LOCAL_'))]
        lines.append(f"| {model} | {w['NATIVE']['correct']}/3 | {w['ZERO']['correct']}/3 | {amp} | {ph} |")
    lines += ['', '完整P通路置零会损失部分纠错，而当前幅度与相位扰动没有损失这些纠错。'
              '这支持“整条通路的作用”与“所测细性质的必要性”分开陈述。'
              '分差改变而最终答案不变，可能处于决策边界的同一侧；不能把连续变化直接换算成纠错贡献比例。', '',
              '当前结果不能把原128的10次外部纠错或8次内部纠错，分成三个唯一、可加的因果百分比。'
              '尚未覆盖的原纠错应作为完整预定义集合核查缓存与后续设计；不能仅挑当前3例作为全量结论。', '',
              '本文不证明没有其他历史缓存，也不启动新采集；完整逐世界决策与连续分差见 phase_accounting.json。']
    (ROOT/'phase_scope_analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'], coverage=coverage)))


if __name__ == '__main__':
    main()
