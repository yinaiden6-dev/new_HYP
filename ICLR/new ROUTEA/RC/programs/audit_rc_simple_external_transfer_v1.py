#!/usr/bin/env python3
"""Independent post-result scalar audit of frozen simplified external transfer.

Never fits a model or imports the producer. It refuses to start before the
producer's complete result validator passes. Heads and predictions are read
only; Python scalar arithmetic reconstructs every candidate logit and action.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_simple_external_transfer_v1'
AUTH = ROOT / 'registry/rc_simple_external_transfer_authority_v1_20260924.json'
REPORT = ROOT / 'reports/REPORT_SIMPLE_EXTERNAL_TRANSFER_V1_20260924.md'
MODELS = ('RAW', 'NATIVE7', 'MASS5', 'ADDITIVE4')
PAIRS = (('RAW', 'NATIVE7'), ('RAW', 'MASS5'), ('RAW', 'ADDITIVE4'),
         ('NATIVE7', 'MASS5'), ('NATIVE7', 'ADDITIVE4'), ('MASS5', 'ADDITIVE4'))
DATA = {
    'grozi': dict(root=ROOT / 'results/rc_new_hyp_grozi120_external_v1',
                  auth=ROOT / 'registry/rc_new_hyp_grozi120_inference_authority_v1_20260913.json',
                  queries=480, tasks=60, groups=27, group_unit='source_video',
                  legacy_validation='GROZI480_ALL_SEALS_COUNTS_GROUPS_PASS', label='GroZi480'),
    'isic': dict(root=ROOT / 'results/rc_new_hyp_isic_transfer_v1',
                 auth=ROOT / 'registry/rc_new_hyp_isic_inference_authority_v1_20260914.json',
                 queries=537, tasks=68, groups=346, group_unit='known_patient',
                 legacy_validation='ISIC537_ALL_SEALS_COUNTS_MRR_ACTIONS_PASS', label='ISIC537（探索）'),
}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(item):
    need(bind(item['path']) == item, 'SHA_DRIFT:' + item['path'])
    return Path(item['path'])


def write_immutable(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_text() == text, 'IMMUTABLE_AUDIT_OUTPUT:' + str(path))
        return
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sym(a, b):
    return (a - b) / (abs(a) + abs(b) + 1e-12)


def scalar_action(x, theta, query):
    need(all(len(row) + 1 == len(theta) for row in x), 'HEAD_DIMENSION')
    z = [math.fsum(float(a) * float(b) for a, b in zip(row, theta[:-1])) + theta[-1]
         for row in x]
    need(len(z) == 127 and all(math.isfinite(v) for v in z), 'FINITE127_LOGITS')
    # max returns the first occurrence; HOLD wins a zero-margin tie.
    best = max(range(127), key=z.__getitem__)
    pos = query['challenger_positions'][best] if z[best] > 0. else query['winner']
    return z, query['axis'][pos]


def verify_simplified_features(query):
    raw, pairs = query['raw_scores'], query['pairs']
    need(len(raw) == len(pairs) == 128 and all(math.isfinite(v) for v in raw), 'RAW128_PAIRS128')
    mean = math.fsum(raw) / 128
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in raw) / 128)
    winner = query['winner']
    w = pairs[winner]
    maximum = 0.
    for c, supplied in zip(query['challenger_positions'], query['simplified_X']):
        p = pairs[c]
        expected = [(raw[c] - raw[winner]) / max(std, 1e-12),
                    sym(p['score'], w['score']), sym(p['mass'], w['mass']),
                    sym(p['score'] / max(p['mass'], 1e-12), w['score'] / max(w['mass'], 1e-12))]
        need(len(supplied) == 4, 'SIMPLIFIED4_COLUMNS')
        error = max(abs(a - b) for a, b in zip(expected, supplied))
        maximum = max(maximum, error)
        need(error < 2e-10, 'SCALAR_SIMPLIFIED_FEATURES')
    for p in pairs:
        need(0. <= p['mass'] <= 1. and math.isfinite(p['free_content']), 'MASS_CONTENT_FINITE_DOMAIN')
        need(abs(p['score'] - p['mass'] * p['free_content']) < 2e-10, 'M_TIMES_FREE_CONTENT')
        need(abs(p['feature_content'] - p['score'] / max(p['mass'], 1e-12)) < 2e-10, 'CONTENT_NORMALIZER')
    return maximum


def group_comparison(rows, baseline, model, published):
    """Counts/deltas use Python; independently resample the declared groups."""
    import numpy as np
    values = defaultdict(list)
    for row in rows:
        values[row['component']].append(int(row['correct'][model]) - int(row['correct'][baseline]))
    deltas = [sum(v) / len(v) for _, v in sorted(values.items())]
    rescue = sum(row['correct'][model] and not row['correct'][baseline] for row in rows)
    loss = sum(row['correct'][baseline] and not row['correct'][model] for row in rows)
    changed = sum(row['selected'][model] != row['selected'][baseline] for row in rows)
    mean = math.fsum(deltas) / len(deltas)
    need((rescue, loss, rescue-loss, changed, len(deltas)) ==
         (published['rescue'], published['loss'], published['net'],
          published['changed_choices'], published['group_count']), 'COMPARISON_INTEGER_REPLAY')
    need(abs(mean - published['equal_group_difference']) < 2e-12 and
         abs((rescue-loss) / len(rows) - published['query_difference']) < 2e-12, 'GROUP_AND_QUERY_ESTIMANDS')
    rng = np.random.default_rng(20260924)
    delta = np.asarray(deltas, dtype=np.float64)
    draws = [delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(axis=1) for _ in range(50)]
    ci = np.quantile(np.concatenate(draws), [.025, .975]).tolist()
    need(max(abs(a-b) for a, b in zip(ci, published['group_bootstrap95'])) < 2e-12, 'GROUP_BOOTSTRAP_REPLAY')
    need(bool(ci[0] > 0) == published['positive_group_interval'], 'GROUP_INTERVAL_SIGN')
    baseline_count = sum(row['correct'][baseline] for row in rows)
    return dict(rescue=rescue, breaks=loss, net=rescue-loss, changed_choices=changed,
                baseline_correct=baseline_count, baseline_correct_loss_rate=loss / max(1, baseline_count),
                query_difference=(rescue-loss) / len(rows), groups=len(deltas),
                equal_group_difference=mean, group_bootstrap95=ci,
                bootstrap_seed=20260924, bootstrap_draws=100000,
                positive_group_interval=bool(ci[0] > 0))


def load_predictions(seal, authority, bundle):
    predictions = defaultdict(list)
    legacy_predictions = defaultdict(dict)
    seen = set()
    need(len(seal['validations']) == 128, 'ALL128_SHARD_SEALS')
    for task, b in enumerate(seal['validations']):
        name, shard = ('grozi', task) if task < 60 else ('isic', task - 60)
        v = read(checked(b))
        need(v['status'] == 'EXTERNAL_SIMPLE_SHARD_PASS' and v['authority'] == authority and
             v['bundle'] == bundle and v['dataset'] == name and v['shard'] == shard and
             v['task'] == task and v['target_reads'] == 0, 'SHARD_BOUNDARY')
        old = read(checked(v['legacy_predictions']))
        need(old['target_reads'] == 0 and len(old['records']) == len(v['queries']), 'OLD_PREDICTION_SEAL')
        for ob in old['records']:
            need(ob['query_id'] not in legacy_predictions[name], 'OLD_QUERY_UNIQUE')
            legacy_predictions[name][ob['query_id']] = ob
        for qb in v['queries']:
            qv = read(checked(qb))
            need(qv['status'] == 'EXTERNAL_SIMPLE_QUERY_NUMPY_PASS' and qv['authority'] == authority and
                 qv['bundle'] == bundle and qv['target_reads'] == 0, 'QUERY_VALIDATION')
            q = read(checked(qv['payload']))
            need(q['query_id'] not in seen, 'ALL1017_QUERY_IDS_UNIQUE')
            seen.add(q['query_id'])
            need(q['authority'] == authority and q['bundle'] == bundle and q['dataset'] == name and
                 q['task'] == task and q['sources'] == qv['sources'] == v['sources'], 'QUERY_BINDINGS')
            need(q['target_reads'] == q['training_updates'] == q['new_encoder_forwards'] == q['new_RoMa_forwards'] == 0,
                 'NO_NEW_FIT_OR_GPU')
            predictions[name].append(q)
    need(len(seen) == 1017, 'ALL1017_QUERIES')
    return predictions, legacy_predictions


def dataset_audit(name, predictions, old_predictions, theta, result, authority, seal_binding):
    d = DATA[name]
    need(result['status'] == 'FIXED_SIMPLE_EXTERNAL_DATASET_COMPLETE' and result['authority'] == authority and
         result['dataset'] == name and result['primary'] == 'MASS5' and result['secondary'] == 'ADDITIVE4' and
         result['predictions_seal'] == seal_binding and result['external_training_updates'] == 0 and
         result['untouched_confirmation_claimed'] is False, 'NEW_RESULT_LINEAGE')
    olda = read(d['auth'])
    roles = {r['query_id']: r for r in read(checked(olda['curator_after_all_seals']))['records']}
    workers = read(checked(olda['sources']['worker']))['records']
    need([q['query_id'] for q in predictions] == [r['query_id'] for r in workers], 'EXTERNAL_QUERY_ORDER')
    if name == 'grozi':
        pf = read(OUT / 'preflight.json')
        base = read(checked(pf['legacy_gallery']))['records']
        tail = read(checked(olda['sources']['gallery_append']))['records']
        gallery = base + tail
        need([g['physical_row'] for g in gallery] == list(range(5533)), 'GROZI_FULL_GALLERY')
    else:
        gallery = read(checked(olda['sources']['gallery']))['records']
        need([g['physical_row'] for g in gallery] == list(range(390)), 'ISIC_FULL_GALLERY')
    labels = {r['physical_row']: r['identity'] for r in gallery}
    oldv = read(d['root'] / 'result_validation.json')
    need(oldv['status'] == d['legacy_validation'] and oldv['result'] == result['legacy_result'], 'OLD_RESULT_VALIDATED')
    oldresult = read(checked(oldv['result']))
    oldrows = {r['query_id']: r for r in oldresult['rows']}
    newrows = {r['query_id']: r for r in result['rows']}
    need(len(newrows) == len(predictions) == len(roles) == d['queries'], 'EXTERNAL_DENOMINATOR')
    rows = []
    maximum_logit_error = maximum_feature_error = maximum_legacy_logit_error = 0.
    zero_mass = clamped_mass = 0
    for q in predictions:
        qid = q['query_id']
        role, expected, oldrow, oldpred = roles[qid], newrows[qid], oldrows[qid], old_predictions[qid]
        axis, ch, winner = q['axis'], q['challenger_positions'], q['winner']
        need(len(set(axis)) == 128 and axis == sorted(axis) and ch == [i for i in range(128) if i != winner], 'FULL_C128_AXIS_AND_TIES')
        need(q['raw_selected'] == axis[winner] == q['raw_ranked_physical_rows'][0], 'RAW_HOLD_IDENTITY')
        need(set(axis) == set(q['raw_ranked_physical_rows'][:128]), 'NATURAL_C128_MEMBERSHIP')
        maximum_feature_error = max(maximum_feature_error, verify_simplified_features(q))
        zero_mass += sum(p['mass'] == 0. for p in q['pairs'])
        clamped_mass += sum(p['mass'] < 1e-12 for p in q['pairs'])
        choices = {'RAW': q['raw_selected']}
        for model in MODELS[1:]:
            x = q['native_X'] if model == 'NATIVE7' else q['simplified_X']
            if model == 'ADDITIVE4':
                x = [[r[0], r[2], r[3]] for r in x]
            z, physical = scalar_action(x, theta[model], q)
            saved = [float.fromhex(v) for v in q['models'][model]['logits_hex']]
            need(len(saved) == 127, 'SAVED127_LOGITS')
            error = max(abs(a-b) for a, b in zip(z, saved))
            maximum_logit_error = max(maximum_logit_error, error)
            need(error < 2e-10 and physical == q['models'][model]['selected'], 'SCALAR_LOGITS_AND_ACTION')
            choices[model] = physical
            if model == 'NATIVE7':
                oldz = [float.fromhex(v) for v in oldpred['models']['COST1']['logits_hex']]
                legacy_error = max(abs(a-b) for a, b in zip(z, oldz))
                maximum_legacy_logit_error = max(maximum_legacy_logit_error, legacy_error)
                need(legacy_error < 2e-10 and physical == oldpred['models']['COST1']['selected'], 'SCALAR_OLD_NATIVE_PARITY')
        target = role['identity']
        correct = {m: labels[v] == target for m, v in choices.items()}
        order = q['raw_ranked_physical_rows']
        raw_rank = next(i+1 for i, p in enumerate(order) if labels[p] == target)
        ranks = {m: next(i+1 for i, p in enumerate([v]+[p for p in order if p != v]) if labels[p] == target)
                 for m, v in choices.items()}
        record = dict(query_id=qid, identity=target, component=role['component'], target_in_C128=raw_rank <= 128,
                      selected=choices, correct=correct, ranks=ranks)
        need(record == expected, 'FULL_JOINED_ROW_SCALAR_REPLAY')
        for current, legacy in (('RAW', 'RAW'), ('NATIVE7', 'COST1')):
            need(choices[current] == oldrow['selected'][legacy] and correct[current] == oldrow['correct'][legacy], 'OLD_RESULT_QUERY_PARITY')
        need(record['component'] == oldrow['component'] and record['target_in_C128'] == oldrow['target_in_C128'], 'OLD_COMPONENT_RECALL_PARITY')
        rows.append(record)
    counts = {m: sum(r['correct'][m] for r in rows) for m in MODELS}
    recall = sum(r['target_in_C128'] for r in rows)
    groups = len({r['component'] for r in rows})
    need(counts == result['counts'] and recall == result['target_recall_C128'], 'INDEPENDENT_COUNTS_RECALL')
    need(groups == result['groups'] == d['groups'] and result['group_unit'] == d['group_unit'], 'GROUP_UNITS')
    need(result['bootstrap'] == dict(draws=100000, seed=20260924, estimand='equally weighted group mean paired accuracy difference'), 'BOOTSTRAP_PROTOCOL')
    mrr = {m: math.fsum(1./r['ranks'][m] for r in rows) / len(rows) for m in MODELS}
    need(max(abs(mrr[m] - result['MRR'][m]) for m in MODELS) < 2e-12, 'FULL_RANK_MRR')
    comparisons = {a+'__to__'+b: group_comparison(rows, a, b, result['comparisons'][a+'__to__'+b]) for a, b in PAIRS}
    return dict(status='INDEPENDENT_SCALAR_EXTERNAL_DATASET_PASS', dataset=name, query_count=len(rows),
                group_count=groups, group_unit=d['group_unit'], target_recall_C128=recall,
                counts=counts, MRR=mrr, comparisons=comparisons,
                max_scalar_logit_error=maximum_logit_error, max_scalar_simplified_feature_error=maximum_feature_error,
                max_scalar_legacy_logit_error=maximum_legacy_logit_error,
                candidates_with_exact_zero_mass=zero_mass, candidates_with_clamped_mass=clamped_mass,
                native_all_choices_match_legacy=True, native_baseline_retained=True,
                scalar_actions_checked=len(rows)*3, scalar_logits_checked=len(rows)*3*127,
                fixed_threshold=0., external_training_updates=0, untouched_confirmation_claimed=False,
                source_result=bind(OUT / name / 'result.json'), legacy_result=oldv['result'],
                roles=olda['curator_after_all_seals'], rows=rows)


def report_text(audit, bundle):
    lines = ['# 简化接口的固定头外部迁移：独立复核', '',
             '**本轮检验固定MASS5主臂和ADDITIVE4对照能否迁移；原完整七参数COST1继续保留。** '
             'GroZi与ISIC此前均已查看过结果，因此本轮是已使用外部面板上的固定方案复核，不是全新未接触数据确认。ISIC属于实例匹配的探索性扩展，不涉及临床诊断；RPC不纳入本轮主要实验。', '',
             '## 模型、候选与训练边界', '',
             '- 源域：全部H593；570张目标在自然C128中的query用于拟合，23张候选缺席图不参与拟合。这是一次全源域最终头训练，不是H593五折OOF成绩。',
             '- 固定COST1损失、FP64、AdamW、学习率0.03、weight decay 0.001、零初始化、2000步；外部不训练、不选阈值、不挑折或主臂。',
             '- MASS5：RAW差距、整体质量M、自由内容L及M×L对应的四个差值特征，加一个偏置；ADDITIVE4移除乘积差值。NATIVE7独立重训后与原冻结COST1的参数及源域全部动作、logits精确一致。',
             '- 外部均沿用原ColNomic全图库自然C128，保留全部127个挑战者，RAW原答案的HOLD分数为0，只有最高挑战者严格大于0才切换。',
             '- 自由L由既有tokens重新计算完整reference FP64 MaxSim；不以旧局部加权分数除以M代替。无新增编码器或RoMa前向。', '',
             '## 全量结果', '',
             '| 外部面板 | 模型 | 正确 / 总数 | 对RAW救回 / 误伤 | 对原COST1救回 / 误伤 | 对原COST1净增 | MRR |',
             '|---|---|---:|---:|---:|---:|---:|']
    for name in DATA:
        d = audit['datasets'][name]
        for model in MODELS:
            rc = d['comparisons'].get('RAW__to__'+model, {'rescue':0, 'breaks':0})
            nc = d['comparisons'].get('NATIVE7__to__'+model)
            ncell = f"{nc['rescue']} / {nc['breaks']}" if nc else ('0 / 0' if model == 'NATIVE7' else '—')
            net = str(nc['net']) if nc else ('0' if model == 'NATIVE7' else '—')
            label = model + ('（预定主臂）' if model == 'MASS5' else '')
            lines.append(f"| {DATA[name]['label']} | {label} | {d['counts'][model]} / {d['query_count']} | "
                         f"{rc['rescue']} / {rc['breaks']} | {ncell} | {net} | {d['MRR'][model]:.6f} |")
    lines += ['', 'MRR按固定动作后的全图库排序计算：将最终选中项移至首位，其余维持原RAW顺序；不是完整128项联合重排的MRR。', '',
              '## 群体不确定性与原模型保留情况', '',
              '| 面板 | 候选召回 | 分组单位 / 组数 | 主臂对RAW净增 | 等组加权差值95%区间 | 主臂对原COST1净增 | 等组加权差值95%区间 |',
              '|---|---:|---|---:|---|---:|---|']
    for name in DATA:
        d = audit['datasets'][name]
        rc, nc = d['comparisons']['RAW__to__MASS5'], d['comparisons']['NATIVE7__to__MASS5']
        ci = lambda c: '[' + ', '.join(f'{100*v:.3f} pp' for v in c['group_bootstrap95']) + ']'
        group = '来源视频' if name == 'grozi' else '已知患者'
        lines.append(f"| {DATA[name]['label']} | {d['target_recall_C128']} / {d['query_count']} | {group} / {d['group_count']} | "
                     f"{rc['net']:+d} | {ci(rc)} | {nc['net']:+d} | {ci(nc)} |")
    lines += ['', '区间为预先固定seed=20260924的100000次组bootstrap，估计各组等权的配对准确率差；'
              '它与按query计数的净增是不同加权口径。组成员、差值、救回/误伤和区间均由本sidecar重新核算。'
              '不将多个面板或模型里最好的数值改称预定主结果。', '']
    for name in DATA:
        d = audit['datasets'][name]
        raw = d['comparisons']['RAW__to__MASS5']
        native = d['comparisons']['NATIVE7__to__MASS5']
        sign = '区间下界大于0' if raw['positive_group_interval'] else '区间没有排除0或负向差值'
        lines.append(f"- {DATA[name]['label']}：固定MASS5对RAW为{raw['rescue']}救/{raw['breaks']}损，净增{raw['net']:+d}；"
                     f"等组区间{sign}。对原COST1为{native['rescue']}救/{native['breaks']}损，净增{native['net']:+d}。"
                     '旧NATIVE7对该面板每张图的决策均与原封存COST1完全一致，旧结果未被替换。')
    lines += ['', '## 独立验收及可复用数据', '',
              f"已逐一复核128份分片封存、1017份query预测和{audit['scalar_actions_checked']}次模型动作；"
              f"标准Python点积重建{audit['scalar_logits_checked']}个挑战者logit，最大误差{audit['max_scalar_logit_error']:.3g}。"
              '独立读原curator及gallery核算身份，不直接信任生产脚本给出的正确性；同时对照新结果及旧native逐图选择。', '']
    for name in DATA:
        d = audit['datasets'][name]
        lines.append(f"- {DATA[name]['label']}：M=0候选{d['candidates_with_exact_zero_mass']}个；M<1e-12候选"
                     f"{d['candidates_with_clamped_mass']}个。冻结特征定义的第四列使用S/max(M,1e-12)；正常M时等于自由L至数值舍入，极小M时保留既有钳位行为。")
    lines += ['', '参数在外部预测前冻结，全部外部预测在统一身份join前封存。完整127个分数、HOLD=0、候选轴、M、L、特征和源文件SHA保留于结果目录。'
              '本轮只能判断这两个固定简化头在这些已使用面板上的迁移表现；不证明RoMa必不可替代、精确空间ownership、任意检索器的普遍定理或内部注入一定有效。', '',
              '- [独立核算JSON](../results/rc_simple_external_transfer_v1/independent_audit.json)',
              '- [源域冻结bundle](../results/rc_simple_external_transfer_v1/source/bundle.json)',
              '- [原始汇总结果](../results/rc_simple_external_transfer_v1/result.json)',
              '- [全部预测的prelabel封存](../results/rc_simple_external_transfer_v1/all_predictions_prelabel_seal.json)', '',
              '源域头SHA256：', '']
    for name, entry in sorted(bundle['heads'].items()):
        lines.append(f"- {name}: `{entry['head']['sha256']}`")
    return '\n'.join(lines) + '\n'


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    validator_path = OUT / 'result_validation.json'
    need(validator_path.exists(), 'RESULT_NOT_READY_NO_AUDIT_RUN')
    validator = read(validator_path)
    need(validator['status'] == 'ALL1017_SEALS_COUNTS_ACTIONS_GROUPS_PASS', 'RESULT_VALIDATOR_NOT_PASS')
    authority = bind(AUTH)
    need(validator['authority'] == authority, 'RESULT_AUTHORITY')
    summary = read(checked(validator['result']))
    need(summary['status'] == 'FIXED_SIMPLE_EXTERNAL_TRANSFER_COMPLETE' and summary['authority'] == authority and
         summary['primary'] == 'MASS5' and summary['secondary'] == 'ADDITIVE4' and summary['query_count'] == 1017,
         'SUMMARY_SCOPE')
    seal_binding = summary['predictions_seal']
    seal = read(checked(seal_binding))
    need(seal['status'] == 'ALL1017_PREDICTIONS_SEALED' and seal['authority'] == authority and
         seal['target_reads'] == 0 and seal['primary'] == 'MASS5', 'PRELABEL_SEAL')
    bundle_binding = summary['bundle']
    bundle = read(checked(bundle_binding))
    need(bundle['status'] == 'SIMPLE_SOURCE_HEADS_FROZEN_PASS' and bundle['authority'] == authority and
         bundle['external_queries_read'] == 0 and bundle['native_theta_and_source_logits_exact_legacy_parity']
         and bundle['primary'] == 'MASS5' and bundle_binding == seal['bundle'], 'SOURCE_HEAD_FREEZE')
    theta = {}
    for name in MODELS[1:]:
        entry = bundle['heads'][name]
        v = read(checked(entry['validation']))
        need(v['status'] == 'SOURCE_HEAD_NUMPY_ACTIONS_PASS' and v['head'] == entry['head'], 'SOURCE_HEAD_VALIDATED')
        h = read(checked(entry['head']))
        need(h['authority'] == authority and h['model'] == name and h['external_queries_read'] == 0,
             'SOURCE_HEAD_IDENTITY')
        theta[name] = [float.fromhex(x) for x in h['theta_hex']]
        need(all(math.isfinite(v) for v in theta[name]) and h['inference']['threshold'] == 0., 'FINITE_FIXED_HEAD')
    predictions, old_predictions = load_predictions(seal, authority, bundle_binding)
    datasets = {}
    for name in DATA:
        need(summary['datasets'][name] == validator['datasets'][name], 'DATASET_SEAL')
        result = read(checked(summary['datasets'][name]))
        datasets[name] = dataset_audit(name, predictions[name], old_predictions[name], theta,
                                       result, authority, seal_binding)
    audit = dict(status='INDEPENDENT1017_PYTHON_SCALAR_ACTIONS_IDENTITY_GROUPS_PASS',
                 authority=authority, producer_validation=bind(validator_path), result=validator['result'],
                 program=bind(__file__), bundle=bundle_binding, predictions_seal=seal_binding,
                 primary='MASS5', secondary='ADDITIVE4', query_count=1017, datasets=datasets,
                 scalar_actions_checked=sum(d['scalar_actions_checked'] for d in datasets.values()),
                 scalar_logits_checked=sum(d['scalar_logits_checked'] for d in datasets.values()),
                 max_scalar_logit_error=max(d['max_scalar_logit_error'] for d in datasets.values()),
                 audit_after_complete_result_validation=True, old_native_model_retained=True,
                 external_training_updates=0, new_gpu_forwards=0, untouched_confirmation_claimed=False,
                 clinical_diagnosis_claimed=False, ownership_claimed=False)
    text = report_text(audit, bundle)
    write_immutable(REPORT, text)
    audit['report'] = bind(REPORT)
    write_immutable(OUT / 'independent_audit.json', json.dumps(audit, ensure_ascii=False, indent=2,
                                                            allow_nan=False, sort_keys=True) + '\n')
    print(json.dumps(dict(status=audit['status'], counts={name:d['counts'] for name,d in datasets.items()},
                          report=str(REPORT), audit=str(OUT / 'independent_audit.json'))), flush=True)


if __name__ == '__main__':
    main()
