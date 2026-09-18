#!/usr/bin/env python3
"""Join every sealed prediction, then report easy-query regressions without tuning."""
import collections
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'programs'))
import materialize_rc_new_hyp593_inputs_v1 as M
OUT = ROOT / 'results/rc_new_hyp_processed128_regression_v1'
AUTH = ROOT / 'registry/rc_new_hyp_processed128_authority_v1_20260916.json'


def comparison(rows, model, baseline='RAW', draws=100000):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['component']].append(r)
    counts = []; deltas = []
    for _, rr in sorted(groups.items()):
        counts.append(len(rr))
        deltas.append(sum(int(r['correct'][model]) - int(r['correct'][baseline]) for r in rr))
    counts = np.array(counts); deltas = np.array(deltas)
    rng = np.random.default_rng(20260916)
    boots = []
    for start in range(0, draws, 2000):
        ix = rng.integers(len(groups), size=(min(2000, draws - start), len(groups)))
        boots.append(deltas[ix].sum(axis=1) / counts[ix].sum(axis=1))
    ci = np.quantile(np.concatenate(boots), [.025, .975]).tolist()
    rescue = sum(not r['correct'][baseline] and r['correct'][model] for r in rows)
    loss = sum(r['correct'][baseline] and not r['correct'][model] for r in rows)
    old_correct = sum(r['correct'][baseline] for r in rows)
    assert rescue - loss == int(deltas.sum())
    return dict(baseline=baseline, model=model, queries=len(rows), components=len(groups),
                rescues=rescue, breaks=loss, net=rescue-loss, accuracy_difference=(rescue-loss)/len(rows),
                component_bootstrap95=ci, baseline_correct=old_correct,
                break_rate_among_baseline_correct=loss/old_correct if old_correct else None,
                zero_observed_breaks=loss == 0, reliable_net_positive=ci[0] > 0,
                reliable_net_negative=ci[1] < 0)


def summary(rows, models):
    return dict(queries=len(rows), counts={m:sum(r['correct'][m] for r in rows) for m in models},
                target_recall_C128=sum(r['target_in_C128'] for r in rows),
                rescues={m:sum(not r['correct']['RAW'] and r['correct'][m] for r in rows) for m in models},
                breaks={m:sum(r['correct']['RAW'] and not r['correct'][m] for r in rows) for m in models})


def main():
    M.need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    a = M.read(AUTH); ab = M.bind(AUTH)
    M.checked(a['sources']['join_program'])
    predictions, validations = [], []
    for shard in range(16):
        f = OUT / 'predictions' / f'shard{shard:02d}'
        v = M.read(f / 'validation.json')
        M.need(v['status'] == 'PROCESSED_FROZEN_HEAD_NUMPY_ACTION_PASS' and v['authority'] == ab, 'ALL16_VALIDATED')
        p = M.read(M.checked(v['payload']))
        M.need(p['authority'] == ab and len(p['records']) == 8 and p['target_reads'] == p['training_updates'] == 0, 'SEALED_WORKER')
        for kind in ('raw', 'roma'):
            receipt = M.read(OUT / kind / f'shard{shard:02d}' / 'receipt.json')
            val = M.read(OUT / kind / f'shard{shard:02d}' / 'validation.json')
            M.need(receipt['payload'] == val['payload'] == p[kind] and val['authority'] == ab, 'INPUT_VALIDATION_CHAIN')
        predictions.extend(p['records']); validations.append(M.bind(f / 'validation.json'))
    workers = M.read(M.checked(a['sources']['worker']))['records']
    M.need([r['query_id'] for r in predictions] == [w['query_id'] for w in workers] and len(predictions) == 128, 'COMPLETE_QUERY_ORDER')
    actions = 0
    for p in predictions:
        order = p['raw_ranked_physical_rows']; axis = sorted(order[:128]); winner = order[0]
        assert len(order) == len(set(order)) == 5412 and p['raw_selected'] == winner
        challengers = [v for v in axis if v != winner]
        assert set(p['models']) == set(a['heads']) | {m + '_CBIND' for m in a['heads']}
        for v in p['models'].values():
            z = [float.fromhex(h) for h in v['logits_hex']]
            assert len(z) == 127 and np.isfinite(z).all()
            j = max(range(127), key=lambda k: z[k])
            assert v['selected'] == (challengers[j] if z[j] > 0 else winner)
            actions += 1
    M.write(OUT / 'all_predictions_prejoin_seal.json', dict(authority=ab, validations=validations,
            queries=128, independently_replayed_actions=actions))
    # Labels are read only after the complete action axis is fixed and verified.
    gallery = M.read(M.checked(a['sources']['gallery']))['records']
    labels = [r['identity'] for r in gallery]
    roles = M.read(M.checked(a['curator']))['records']
    assert [r['query_id'] for r in roles] == [r['query_id'] for r in predictions]
    rows = []
    for role, p in zip(roles, predictions):
        chosen = dict(RAW=p['raw_selected'], **{m:v['selected'] for m,v in p['models'].items()})
        order = p['raw_ranked_physical_rows']
        correct = {m: labels[row] == role['identity'] for m, row in chosen.items()}
        assert len(set(labels[i] for i in order)) == 5412
        rows.append(dict(**role, correct=correct, selected=chosen,
                    target_in_C128=role['identity'] in [labels[i] for i in order[:128]],
                    raw_target_rank=next(i+1 for i,j in enumerate(order) if labels[j] == role['identity']),
                    holds={m:j == chosen['RAW'] for m,j in chosen.items()},
                    max_logits={m:max(float.fromhex(h) for h in v['logits_hex']) for m,v in p['models'].items()}))
    models = list(rows[0]['correct']); overall = summary(rows, models)
    comparisons = {m:comparison(rows, m) for m in models if m != 'RAW'}
    independent = {m:sum(labels[r['selected'][m]] == r['identity'] for r in rows) for m in models}
    assert independent == overall['counts']
    for m, c in comparisons.items():
        assert overall['counts'][m] - overall['counts']['RAW'] == c['rescues'] - c['breaks']
    result = dict(status='PROCESSED128_FROZEN_REGRESSION_COMPLETE', authority=ab,
             primary='COST1', secondary='CE', head_lineage='Frozen full-H593 heads from 20260913, no retraining',
             scope='128 hash-selected synthetic processed queries; 128 distinct reference identities; not independent external confirmation',
             candidate_source='Legacy5413 physical references / 5412 corrected identities; naturalC128',
             action='All127 challengers; SWITCH iff maxlogit>0 else HOLD',
             **overall, comparisons=comparisons, rows=rows,
             by_corruption={s:summary([r for r in rows if r['corruption'] == s], models) for s in sorted({r['corruption'] for r in rows})},
             by_training_identity_overlap={str(v):summary([r for r in rows if r['training_identity_overlap'] == v], models) for v in (False, True)},
             training_image_byte_overlap=sum(r['training_image_byte_overlap'] for r in rows),
             formal_GO_claimed=False, training_updates=0, bootstrap_draws=100000, bootstrap_seed=20260916)
    M.write(OUT / 'result.json', result)
    with (OUT / 'per_query.csv').open('x', newline='') as f:
        fields = ['query_id','origin_name','original_path','corruption','training_identity_overlap','target_in_C128','raw_target_rank']
        writer = csv.DictWriter(f, fieldnames=fields + models)
        writer.writeheader()
        writer.writerows({**{k:r[k] for k in fields}, **{m:int(r['correct'][m]) for m in models}} for r in rows)
    lines = ['# processed128：是否伤害原正确', '',
        '冻结 full-H593 头；完整 5413 reference／5412 身份，自然 C128，阈值不变。', '',
        '| 模型 | 正确 / 128 | 救回 | 改错 | 净增 |', '|---|---:|---:|---:|---:|']
    for m in ('RAW','COST1','CE','COST4','GROUP_COST4','RAW2_CE'):
        c = comparisons.get(m, dict(rescues=0, breaks=0, net=0))
        lines.append(f"| {m} | {overall['counts'][m]} | {c['rescues']} | {c['breaks']} | {c['net']:+d} |")
    lines += ['', f"C128 内含 target：{overall['target_recall_C128']}/128。", '',
              '这是合成处理图回归，不是独立外部确认。未观察到损失不等于总体永不损失。']
    for m in ('COST1','CE'):
        c = comparisons[m]
        lines += ['', f"## {m}", '', f"RAW 原正确中的改错：{c['breaks']}/{c['baseline_correct']}；来源分组净增区间：{c['component_bootstrap95']}。", '',
                  '| 变化 | query | 原图 |', '|---|---|---|']
        for r in rows:
            if r['correct'][m] != r['correct']['RAW']:
                action = '救回' if r['correct'][m] else '改错'
                lines.append(f"| {action} | {r['query_id']} | [{r['origin_name']}](<{r['original_path']}>) |")
    with (OUT / 'report_zh.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    M.write(OUT / 'result_validation.json', dict(status='PROCESSED128_SEALS_COUNTS_ACTIONS_PASS',
            authority=ab, result=M.bind(OUT / 'result.json'), query_count=128,
            independently_replayed_actions=actions, predictions=validations))
    print(json.dumps(dict(counts=overall['counts'], COST1=comparisons['COST1'], CE=comparisons['CE'])), flush=True)


if __name__ == '__main__':
    main()
