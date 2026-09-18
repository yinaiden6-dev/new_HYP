#!/usr/bin/env python3
"""Display completed TRAIN128 OOF4 results; never fit, tune, or read EVAL.

Execution requires explicit result/validation SHA256 pins and a completed fresh
join validation. All 32 groups are shown in lexical order, including zero and
negative effects. This is a presentation consumer, not a new scientific test.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = Path(__file__).resolve()
SOURCE = ROOT / 'results/rc_train128_disagreement_oof4_v1'
OUT = ROOT / 'results/rc_train128_disagreement_oof4_display_v2_layout'
REPORT = ROOT / 'reports/REPORT_NEW_HYP_TRAIN128_DISAGREEMENT_OOF4_RESULT_V2_LAYOUT_20260910.md'
FIT_PROGRAM = ROOT / 'programs/run_rc_train128_disagreement_oof4_v1.py'
FIT_PROGRAM_SHA = 'd7302eca66c7c6469a7f5ef99a77afeae8af4b25c3a253a1bdcbb2da7af3a715'
MODELS = ('BASE7', 'CONSTANT1', 'CONDITIONAL4')
CONTRASTS = (('CONDITIONAL4', 'BASE7'), ('CONDITIONAL4', 'CONSTANT1'), ('CONSTANT1', 'BASE7'))
INPUT_ALLOWED = set()
BLOCKED = []


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def audit(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    s = str(p).lower()
    forbidden = any(token in s for token in ('curator', '/grozi/', 'd1_mi', 'd1-mi',
                                            '/target_join/', '/role_shards/', 'oracle'))
    if ROOT / 'results' in p.parents:
        forbidden |= p not in INPUT_ALLOWED and OUT not in p.parents
    if forbidden:
        BLOCKED.append(str(p))
        raise PermissionError('POSTRESULT_DISPLAY_READ_BOUNDARY:' + str(p))


sys.addaudithook(audit)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def check_binding(binding, expected_path):
    expected_path = Path(expected_path).resolve()
    need(Path(binding['path']).resolve() == expected_path, 'FIXED_INPUT_PATH')
    INPUT_ALLOWED.add(expected_path)
    need(sha(expected_path) == binding['sha256'], 'BOUND_ARTIFACT_DRIFT:' + str(expected_path))
    return expected_path


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == data, 'APPEND_ONLY_DISPLAY_OUTPUT:' + str(path))
        return
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def qualified_result(result_sha, validation_sha):
    result_path, validation_path = SOURCE / 'result.json', SOURCE / 'validation.json'
    need(result_path.is_file() and validation_path.is_file(), 'WAIT_FOR_BOTH_COMPLETED_RESULT_AND_VALIDATION')
    INPUT_ALLOWED.update((result_path.resolve(), validation_path.resolve()))
    need(sha(result_path) == result_sha and sha(validation_path) == validation_sha, 'EXPLICIT_RESULT_VALIDATION_PINS')
    validation = read(validation_path)
    need(validation['status'] == 'TRAIN128_DISAGREEMENT_OOF4_ALL_FOLDS_AND_JOIN_REPLAY_PASS', 'FRESH_JOIN_REPLAY_PASS_REQUIRED')
    need(validation['result'] == bind(result_path), 'VALIDATION_RESULT_BINDING')
    need(validation['query_count'] == 128 and validation['group_count'] == 32 and
         validation['all_heldout_logit_checks'] == 128*127*3, 'ALL128_OOF_LOGITS_QUALIFIED')
    need(validation['EVAL_reads'] == validation['forbidden_read_attempts'] == 0, 'QUALIFIED_TRAIN_ONLY_BOUNDARY')
    need(validation['program'] == {'path': str(FIT_PROGRAM), 'sha256': FIT_PROGRAM_SHA}, 'FROZEN_OOF_PROGRAM_BINDING')
    need(sha(FIT_PROGRAM) == FIT_PROGRAM_SHA, 'FROZEN_OOF_PROGRAM_BYTES')
    authority_path = ROOT / 'registry/rc_train128_disagreement_oof4_execution_authority_v1_20260910.json'
    check_binding(validation['authority'], authority_path)
    result = read(result_path)
    need(result['status'] == 'TRAIN128_DISAGREEMENT_OOF4_JOINED', 'OOF_RESULT_STATUS')
    need(result['authority'] == validation['authority'] and result['program'] == validation['program'], 'RESULT_LINEAGE')
    need(result['OOF_gate_pass'] == validation['OOF_gate_pass'], 'VALIDATED_GATE')
    need(result['EVAL_reads'] == result['forbidden_read_attempts'] == 0 and
         result['final_ec7_fit_performed'] is False, 'NO_EVAL_OR_FINAL_EC7_FIT')
    need(result['query_count'] == 128 and result['identity_count'] == result['group_count'] == 32, 'POPULATION')
    parameters, bindings = {}, {}
    for fold in range(4):
        directory = SOURCE / ('fold%02d' % fold)
        vp = check_binding(result['fold_validations'][str(fold)], directory / 'validation.json')
        val = read(vp)
        need(val['status'] == 'TRAIN_OOF4_FOLD_FRESH_REFIT_PREDICTION_REPLAY_PASS' and val['fold'] == fold,
             'QUALIFIED_FOLD')
        need(val['authority'] == result['authority'] and val['program'] == result['program'], 'FOLD_LINEAGE')
        sp = check_binding(val['seal'], directory / 'seal.json')
        seal = read(sp)
        need(seal['fold'] == fold and seal['authority'] == result['authority'], 'SEALED_FOLD_LINEAGE')
        pp = check_binding(seal['parameters'], directory / 'parameters.json')
        parameters[str(fold)] = read(pp)
        bindings[str(fold)] = {'validation': bind(vp), 'seal': bind(sp), 'parameters': bind(pp)}
    return result, parameters, bindings


def count_review(result):
    rows = result['rows']
    need(len(rows) == 128 and [r['execution_ordinal'] for r in rows] == list(range(128)), 'EXACT128_ORDER')
    need(len({r['query_id'] for r in rows}) == 128, 'DISTINCT_QUERY_IDS')
    groups = sorted({r['group'] for r in rows})
    need(len(groups) == len({r['identity'] for r in rows}) == 32, 'ALL32_GROUPS_IDENTITIES')
    need(groups == result['paired_group_statistics']['ordered_groups'], 'FROZEN_ALL_GROUP_ORDER')
    need(sum(r['RAW_correct'] for r in rows) == result['RAW_correct'], 'RAW_CORRECT_COUNT')
    need(sum(not r['target_in_natural_C128'] for r in rows) == result['target_absent_count'], 'C128_MISSES_RETAINED')
    group_rows = []
    for group in groups:
        members = [r for r in rows if r['group'] == group]
        folds = {r['fold'] for r in members}
        need(len(folds) == len({r['identity'] for r in members}) == 1, 'GROUP_FOLD_IDENTITY_LOCK')
        entry = {'group': group, 'fold': next(iter(folds)), 'image_count': len(members),
                 'RAW_correct': sum(r['RAW_correct'] for r in members)}
        for model in MODELS:
            entry[model + '_correct'] = sum(r['correct'][model] for r in members)
            entry[model + '_RAW_rescue'] = sum(r['correct'][model] and not r['RAW_correct'] for r in members)
            entry[model + '_RAW_break'] = sum(not r['correct'][model] and r['RAW_correct'] for r in members)
            fraction = Fraction(entry[model+'_correct'], len(members))
            need(result['scores'][model]['group_metrics'][group] ==
                 {'correct': entry[model+'_correct'], 'count': len(members), 'accuracy_fraction': str(fraction)},
                 'SOURCE_GROUP_METRIC')
        for model, base in CONTRASTS:
            key = model + '_minus_' + base
            entry[key + '_gain'] = sum(r['correct'][model] and not r['correct'][base] for r in members)
            entry[key + '_loss'] = sum(not r['correct'][model] and r['correct'][base] for r in members)
        group_rows.append(entry)
    for model in MODELS:
        score = result['scores'][model]
        need(sum(g[model+'_correct'] for g in group_rows) == score['correct'], 'MODEL_CORRECT_TOTAL')
        need(sum(g[model+'_RAW_rescue'] for g in group_rows) == score['rescue_vs_RAW'] and
             sum(g[model+'_RAW_break'] for g in group_rows) == score['break_vs_RAW'], 'RAW_RESCUE_BREAK_TOTAL')
        average = sum((Fraction(g[model+'_correct'], g['image_count']) for g in group_rows), Fraction(0))/32
        need(str(average) == score['equal_group_accuracy_fraction'] and float(average) == score['equal_group_accuracy'],
             'EQUAL_GROUP_ACCURACY')
    for model, base in CONTRASTS:
        key = model+'_minus_'+base
        pair = result['paired_group_statistics']['contrasts'][key]
        gains = [r['original_query_id'] for r in rows if r['correct'][model] and not r['correct'][base]]
        losses = [r['original_query_id'] for r in rows if not r['correct'][model] and r['correct'][base]]
        need(pair['rescue_query_ids'] == gains and pair['break_query_ids'] == losses and
             pair['rescue'] == len(gains) and pair['break'] == len(losses) and
             pair['net_correct_images'] == len(gains)-len(losses), 'PAIRED_GAIN_LOSS')
        diffs = [Fraction(g[model+'_correct']-g[base+'_correct'], g['image_count']) for g in group_rows]
        need(str(sum(diffs, Fraction(0))/32) == pair['equal_group_mean_difference_fraction'], 'GROUP_EFFECT')
        need(pair['positive_group_count'] == sum(d > 0 for d in diffs) and
             pair['negative_group_count'] == sum(d < 0 for d in diffs) and
             pair['zero_group_count'] == sum(d == 0 for d in diffs), 'ALL_EFFECT_DIRECTIONS_COUNTED')
    for fold in range(4):
        fr = [r for r in rows if r['fold'] == fold]
        need(len({r['group'] for r in fr}) == 8, 'EIGHT_GROUPS_PER_FOLD')
        need(result['fold_metrics'][str(fold)] == {'count': len(fr), 'RAW_correct': sum(r['RAW_correct'] for r in fr),
             'correct': {model: sum(r['correct'][model] for r in fr) for model in MODELS}}, 'FOLD_COUNTS')
    expected_gates = {f'count_strictly_above_{m}': result['scores']['CONDITIONAL4']['correct'] > result['scores'][m]['correct']
                      for m in ('BASE7', 'CONSTANT1')}
    expected_gates.update({f'equal_group_strictly_above_{m}': Fraction(result['scores']['CONDITIONAL4']['equal_group_accuracy_fraction']) >
        Fraction(result['scores'][m]['equal_group_accuracy_fraction']) for m in ('BASE7', 'CONSTANT1')})
    expected_gates['RAW_break_not_above_BASE7'] = result['scores']['CONDITIONAL4']['break_vs_RAW'] <= result['scores']['BASE7']['break_vs_RAW']
    need(expected_gates == result['gates'] and all(expected_gates.values()) == result['OOF_gate_pass'], 'PRESPECIFIED_GATE_DISPLAY')
    return group_rows


def plot(groups, result):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/rc-train128-oof4-display-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.hashsalt': 'rc-train128-oof4-v1'})
    y = np.arange(32)
    fig, axes = plt.subplots(1, 2, figsize=(17.4, 19.6), gridspec_kw={'width_ratios': [1, 1.1]}, sharey=True)
    colors = {'RAW': '#8b939b', 'BASE7': '#2878b5', 'CONSTANT1': '#df972c', 'CONDITIONAL4': '#7650a3'}
    offsets = {'RAW': -.24, 'BASE7': -.08, 'CONSTANT1': .08, 'CONDITIONAL4': .24}
    for model in ('RAW',)+MODELS:
        values = [g[model+'_correct']/g['image_count'] for g in groups]
        axes[0].scatter(values, y+offsets[model], color=colors[model], s=25 if model=='RAW' else 34,
                        marker='x' if model=='RAW' else 'o', label=model, zorder=3)
    axes[0].set_xlim(-.04, 1.04)
    axes[0].set_xticks(np.linspace(0, 1, 6))
    axes[0].set_xlabel('Accuracy within source group')
    axes[0].set_title('All 32 groups: accuracy', pad=14, fontsize=15)
    labels = [f"{g['group']}   [n={g['image_count']}, fold={g['fold']}]" for g in groups]
    axes[0].set_yticks(y, labels)
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.04), ncol=2, frameon=False)
    upper = 1
    for j, baseline in enumerate(('BASE7', 'CONSTANT1')):
        prefix = 'CONDITIONAL4_minus_'+baseline
        gains = np.array([g[prefix+'_gain'] for g in groups])
        losses = np.array([g[prefix+'_loss'] for g in groups])
        upper = max(upper, int(gains.max()), int(losses.max()))
        ys = y+(-.19 if j==0 else .19)
        hatch = '' if j==0 else '///'
        axes[1].barh(ys, gains, height=.31, color='#37936b', edgecolor='white', hatch=hatch,
                     label='Gain vs '+baseline)
        axes[1].barh(ys, -losses, height=.31, color='#bd5b54', edgecolor='white', hatch=hatch,
                     label='Loss vs '+baseline)
        for yy, gain, loss in zip(ys, gains, losses):
            if gain:
                axes[1].text(float(gain)+.07, yy, str(gain), ha='left', va='center', fontsize=9)
            if loss:
                axes[1].text(-float(loss)-.07, yy, str(loss), ha='right', va='center', fontsize=9)
    axes[1].set_xlim(-upper-.8, upper+.8)
    axes[1].axvline(0, color='#777777', lw=.8)
    axes[1].set_xlabel('CONDITIONAL4 paired image loss (-) / gain (+)')
    axes[1].set_title('Paired outcomes; every zero group retained', pad=14, fontsize=15)
    axes[1].legend(loc='lower left', bbox_to_anchor=(0, 1.04), ncol=2, frameon=False)
    for ax in axes:
        ax.set_ylim(31.65, -.65)
        ax.grid(axis='x', color='#dddddd', linewidth=.6)
        for yy in y:
            ax.axhline(yy+.5, color='#eeeeee', lw=.6, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('TRAIN128 grouped out-of-fold screen: 128 images / 32 identities / 32 groups', fontsize=19, y=.982)
    counts = ' | '.join(f"{m}: {result['scores'][m]['correct']}/128" for m in MODELS)
    fig.text(.5, .955, counts+' | Prespecified OOF gate: '+('PASS' if result['OOF_gate_pass'] else 'NO-GO'), ha='center', fontsize=13)
    fig.text(.5, .023, 'Each fold refits its BASE7 without heldout groups. This is TRAIN OOF development, not frozen ec7 EVAL.',
             ha='center', fontsize=12)
    fig.subplots_adjust(left=.245, right=.965, top=.878, bottom=.057, wspace=.13)
    outputs = {}
    for extension in ('png', 'svg', 'pdf'):
        buffer = io.BytesIO()
        metadata = {'Software': 'Matplotlib'} if extension=='png' else ({'Date': None} if extension=='svg' else {'CreationDate': None, 'ModDate': None})
        fig.savefig(buffer, format=extension, dpi=170, metadata=metadata, bbox_inches='tight', pad_inches=.15)
        path = OUT / ('all32_group_results.'+extension)
        atomic(path, buffer.getvalue())
        outputs[extension] = bind(path)
    plt.close(fig)
    return outputs


def fmt_hex(value):
    number = float.fromhex(value)
    need(math.isfinite(number), 'FINITE_DISPLAY_PARAMETER')
    return format(number, '.9g')


def report(result, groups, parameters):
    passed = result['OOF_gate_pass']
    lines = ['# new HYP：TRAIN128 按来源组 OOF4 条件化检验', '',
        '**预设 OOF 门：'+('通过。可进入另行授权的最终拟合；本轮尚未产生 EVAL 结果。' if passed else
          '未通过。本次条件化机制按计划停止，不进入 EVAL。')+'**', '',
        '128 张图、32 个身份、32 个来源组，4 折各留出 8 组；所有候选均来自自然 RAW C128。'
        '每折 BASE7 仅用排除留出组后的原 FULL32／PAIR64 检索监督重新训练，再冻结基头训练补偿。'
        '这不是原 ec7 头；本轮 OOF 数字不能与原 ec7 的 EVAL128 99/128 当作同一模型、同一数据比较。', '',
        f"RAW 正确 {result['RAW_correct']}/128；自然 C128 缺失 target {result['target_absent_count']} 张，全部保留统计。", '',
        '| 模型 | 正确 | 对 RAW 救回／损失 | 等权来源组准确率 |',
        '|---|---:|---:|---:|']
    for model in MODELS:
        s = result['scores'][model]
        lines.append(f"| {model} | {s['correct']}/128 | {s['rescue_vs_RAW']} / {s['break_vs_RAW']} | {100*s['equal_group_accuracy']:.3f}% |")
    lines += ['', 'BASE7 不加补偿；CONSTANT1 学习一个全局非负补偿强度；CONDITIONAL4 根据 RAW 差、匹配位置分歧差和 reference 平均权重差学习条件化强度。所有超参数和末步取值规则预先固定。', '',
        '| 成对比较 | 新增正确／损失正确 | 净增图数 | 等权组差（百分点） | 组 bootstrap 95% 区间 | 组 sign-flip 双侧 p |',
        '|---|---:|---:|---:|---:|---:|']
    for model, base in CONTRASTS:
        d = result['paired_group_statistics']['contrasts'][model+'_minus_'+base]
        lo, hi = d['bootstrap_95_percentile_interval']
        lines.append(f"| {model} − {base} | {d['rescue']} / {d['break']} | {d['net_correct_images']:+d} | "
                     f"{100*d['equal_group_mean_difference']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] | {d['exact_two_sided_group_signflip']['two_sided_p']:.6g} |")
    lines += ['', '统计单位为全部 32 个来源组，等权计组；bootstrap 10,000 次，固定 seed 20260910。双侧 sign-flip p 为精确组级计算，零差组保留。p 值不是本轮晋级门。', '',
        '预设门要求 CONDITIONAL4 的正确图数和等权组准确率均严格超过 BASE7 与 CONSTANT1，并且对 RAW 的 break 不多于 BASE7。', '']
    gate_labels = {'count_strictly_above_BASE7': '正确图数超过 BASE7', 'count_strictly_above_CONSTANT1': '正确图数超过 CONSTANT1',
                   'equal_group_strictly_above_BASE7': '等权组准确率超过 BASE7', 'equal_group_strictly_above_CONSTANT1': '等权组准确率超过 CONSTANT1',
                   'RAW_break_not_above_BASE7': 'RAW break 不多于 BASE7'}
    lines.extend(f"- {gate_labels[k]}：{'通过' if value else '未通过'}。" for k, value in result['gates'].items())
    lines += ['', '各折参数如下（显示值取 9 位有效数字，完整 FP64 十六进制值保存在参数产物）。'
              'BASE7 顺序为 RAW、S、M、L、Q、R、bias；CONDITIONAL4 顺序为截距、−xRAW、D_w−D_c、sym(mean_wr_c, mean_wr_w)。', '',
              '| 折 | 留出图数 | BASE7／CONSTANT1／CONDITIONAL4 正确数 |', '|---|---:|---:|']
    for fold in range(4):
        metric = result['fold_metrics'][str(fold)]
        counts = ' / '.join(str(metric['correct'][m]) for m in MODELS)
        lines.append(f"| {fold} | {metric['count']} | {counts} |")
    for fold in range(4):
        p = parameters[str(fold)]
        base = p['BASE7']['weight_binary64']+[p['BASE7']['bias_binary64']]
        lines += ['', f"折 {fold}：", '',
            '- BASE7：`['+', '.join(map(fmt_hex, base))+']`',
            '- CONSTANT1 α：`['+', '.join(map(fmt_hex, p['CONSTANT1']['parameters_binary64']))+']`',
            '- CONDITIONAL4 β：`['+', '.join(map(fmt_hex, p['CONDITIONAL4']['parameters_binary64']))+']`']
    lines += ['', f'[全部32组结果图]({OUT}/all32_group_results.png) · [全组逐项表]({OUT}/all32_group_metrics.csv) · [完整各折参数]({OUT}/fold_parameters.json)', '',
        f'[来源结果]({SOURCE}/result.json) · [独立复核]({SOURCE}/validation.json)', '',
        '此报告只展示已封存结果，没有重新训练、读取 EVAL、挑选参数或筛掉负向／零增益组。'
        'OOF 门通过也只是为后续独立检验提供资格，不等于已证明普遍增益或零 break。', '']
    atomic(REPORT, '\n'.join(lines).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-sha', required=True)
    parser.add_argument('--validation-sha', required=True)
    args = parser.parse_args()
    result, parameters, fold_sources = qualified_result(args.result_sha, args.validation_sha)
    group_rows = count_review(result)
    # Archive exact supplied parameter values without changing or selecting them.
    write_json(OUT / 'fold_parameters.json', {'parameters': parameters, 'fold_source_bindings': fold_sources})
    table = io.StringIO(newline='')
    writer = csv.DictWriter(table, fieldnames=list(group_rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(group_rows)
    atomic(OUT / 'all32_group_metrics.csv', table.getvalue().encode())
    figures = plot(group_rows, result)
    report(result, group_rows, parameters)
    need(not BLOCKED, 'NO_FORBIDDEN_READ_ATTEMPTS')
    write_json(OUT / 'manifest.json', {'status': 'TRAIN128_OOF4_ALL32_GROUP_DISPLAY_COMPLETE',
        'program': bind(PROGRAM), 'layout_revision': 'full-length source-group labels included by tight bounding box; no data or statistic changes', 'source_result': bind(SOURCE / 'result.json'),
        'source_validation': bind(SOURCE / 'validation.json'), 'fold_source_bindings': fold_sources,
        'report': bind(REPORT), 'figures': figures, 'group_table': bind(OUT / 'all32_group_metrics.csv'),
        'fold_parameters': bind(OUT / 'fold_parameters.json'), 'group_count': 32, 'query_count': 128,
        'display_group_order': 'all source groups in lexical order', 'all_counts_and_group_means_rechecked': True,
        'source_bootstrap_and_signflip_reused_after_validation': True,
        'OOF_gate_pass': result['OOF_gate_pass'], 'EVAL_reads': 0, 'training_updates': 0, 'parameter_selection': False})
    print(json.dumps({'status': 'TRAIN128_OOF4_ALL32_GROUP_DISPLAY_COMPLETE',
                      'manifest': bind(OUT / 'manifest.json'), 'report': bind(REPORT)}), flush=True)


if __name__ == '__main__':
    main()
