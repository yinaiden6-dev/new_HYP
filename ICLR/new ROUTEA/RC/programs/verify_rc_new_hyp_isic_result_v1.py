#!/usr/bin/env python3
"""Independent scalar action/ranking and patient-bootstrap reconstruction."""
import collections
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_new_hyp_isic_transfer_v1'


def read(p): return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p)
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def checked(b):
    assert bind(b['path']) == b, 'HASH_DRIFT'
    return Path(b['path'])


def main():
    result = read(OUT / 'result.json')
    authority = read(checked(result['authority']))
    for family in ('sources',):
        for b in authority[family].values():
            if Path(b['path']).stat().st_size < 16 * (1 << 20): checked(b)
    for sources in authority['heads'].values():
        for b in sources.values(): checked(b)
    gallery = read(checked(authority['sources']['gallery']))['records']
    physical = {r['identity']: r['physical_row'] for r in gallery}
    roles = {r['query_id']: r for r in read(checked(authority['curator_after_all_seals']))['records']}
    expected = {r['query_id']: r for r in result['rows']}
    workers = read(checked(authority['sources']['worker']))['records']
    all_rows, query_order, actions = [], [], 0
    for s in range(68):
        folder = OUT / 'predictions' / f'shard{s:02d}'
        v = read(folder / 'validation.json')
        assert v['authority'] == result['authority'] and v['status'] == 'ISIC_FROZEN_HEAD_NUMPY_ACTION_PASS'
        p = read(checked(v['payload']))
        assert p['authority'] == result['authority'] and p['training_updates'] == p['target_reads'] == 0
        assert len(p['records']) == (1 if s == 67 else 8)
        for q in p['records']:
            qid = q['query_id']; role = roles[qid]; target = physical[role['identity']]
            order = q['raw_ranked_physical_rows']
            assert sorted(order) == list(range(390))
            base = order[0]; challengers = sorted(set(order[:128]) - {base})
            selections = {'RAW': base}; held_target = {}
            for model, entry in q['models'].items():
                scores = list(map(float.fromhex, entry['logits_hex']))
                assert len(scores) == 127 and all(map(math.isfinite, scores))
                ranked = sorted(zip(scores, challengers), key=lambda t: (-t[0], t[1]))
                score, challenger = ranked[0]
                selected = challenger if score > 0 else base
                assert selected == entry['selected']; selections[model] = selected; actions += 1
                held_target[model] = bool(challenger == target and score <= 0)
            correct, ranks, holds = {}, {}, {}
            for model, selected in selections.items():
                reranked = order.copy(); reranked.remove(selected); reranked.insert(0, selected)
                ranks[model] = reranked.index(target) + 1; correct[model] = selected == target
                if model != 'RAW': holds[model] = selected == base
            row = dict(query_id=qid, identity=role['identity'], component=role['component'],
                target_in_C128=target in order[:128], correct=correct, ranks=ranks,
                selected=selections, holds=holds, target_top_challenger_held=held_target)
            assert row == expected[qid], 'ROW_RECONSTRUCTION'
            all_rows.append(row); query_order.append(qid)
    assert query_order == [w['query_id'] for w in workers] and len(all_rows) == 537 and actions == 5370
    assert sum(r['target_in_C128'] for r in all_rows) == result['target_recall_C128']
    for model in result['counts']:
        assert sum(r['correct'][model] for r in all_rows) == result['counts'][model]
        assert abs(math.fsum(1/r['ranks'][model] for r in all_rows)/537 - result['MRR'][model]) < 1e-12
    for field, source in [('by_patient','component'), ('by_lesion','identity')]:
        for key, reported in result[field].items():
            rr = [r for r in all_rows if r[source] == key]
            assert len(rr) == reported['queries']
            assert sum(r['target_in_C128'] for r in rr) == reported['recall_C128']
            for model in result['counts']:
                assert sum(r['correct'][model] for r in rr) == reported['correct'][model]
                assert abs(math.fsum(1/r['ranks'][model] for r in rr)/len(rr)-reported['MRR'][model]) < 1e-12
    for model, reported in result['action_breakdown'].items():
        switched = [r for r in all_rows if r['selected'][model] != r['selected']['RAW']]
        assert len(switched) == reported['switches'] and 537-len(switched) == reported['holds']
        assert sum(not r['correct']['RAW'] and r['correct'][model] for r in all_rows) == reported['rescues_vs_RAW']
        assert sum(r['correct']['RAW'] and not r['correct'][model] for r in all_rows) == reported['breaks_vs_RAW']
        assert sum(not r['correct']['RAW'] and not r['correct'][model] for r in switched) == reported['wrong_to_other_wrong']
        assert sum(r['target_top_challenger_held'][model] for r in all_rows) == reported['target_top_challenger_held']
    # Independently construct one global resampling-index matrix, unlike chunked producer.
    groups = sorted({r['component'] for r in all_rows}); assert len(groups) == 346
    sampled = np.random.default_rng(20260914).integers(346, size=(100000, 346))
    checks = {}
    for key, reported in result['comparisons'].items():
        baseline, model = key.split('__to__')
        contributions = collections.defaultdict(list)
        rescue = loss = 0
        for r in all_rows:
            b, m = r['correct'][baseline], r['correct'][model]
            contributions[r['component']].append(int(m)-int(b))
            rescue += (not b and m); loss += (b and not m)
        delta = np.array([sum(contributions[g])/len(contributions[g]) for g in groups])
        ci = np.quantile(delta[sampled].sum(axis=1)/346, [.025, .975])
        assert rescue == reported['rescue'] and loss == reported['loss'] and rescue-loss == reported['net']
        assert abs(delta.mean()-reported['equal_patient_difference']) < 1e-12
        assert abs((rescue-loss)/537-reported['query_difference']) < 1e-12
        assert np.allclose(ci, reported['patient_bootstrap95'], rtol=0, atol=1e-12)
        assert bool(rescue > loss and delta.mean() > 0 and ci[0] > 0) == reported['reliable_positive']
        checks[key] = True
    baselines = ['RAW','COST4','GROUP_COST4','RAW2_CE','COST1_CBIND']
    signal = all(result['comparisons'][b+'__to__COST1']['reliable_positive'] for b in baselines)
    assert signal == result['positive_exploratory_transfer_signal'] and not result['untouched_external_GO_claimed']
    audit = dict(status='ISIC537_INDEPENDENT_ACTION_PATIENT_AUDIT_PASS', result=bind(OUT/'result.json'),
        authority=result['authority'], program=bind(__file__), independent_actions=actions, query_count=537,
        patient_count=346, comparisons_verified=checks, positive_exploratory_transfer_signal=signal,
        untouched_external_GO_claimed=False)
    with (OUT/'independent_final_audit_v1.json').open('x') as f:
        json.dump(audit, f, sort_keys=True, indent=2); f.write('\n')
    print(audit['status'], flush=True)


if __name__ == '__main__': main()
