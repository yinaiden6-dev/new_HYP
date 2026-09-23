#!/usr/bin/env python3
"""Read-only post-join diagnosis. No fitting, threshold selection or deployment."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_rank_action_v1'


def read(p):
    return json.loads(Path(p).read_text())


def checked(b):
    p = Path(b['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == b['sha256'], p
    return p


def main():
    v = read(OUT / 'validation.json')
    assert v['status'] == 'RANK_ACTION_ALL_COUNTS_PASS'
    result = read(checked(v['result']))
    a = read(checked(result['authority']))
    for b in a['code_sources'].values():
        checked(b)
    labels = {r['physical_row']: r['identity'] for r in read(checked(a['public_sources']['gallery']))['records']}
    roles = {r['query_id']: r for r in read(checked(a['join_sources']['curator']))['records']}
    rows = result['rows']
    assert len(rows) == len({r['query_id'] for r in rows}) == 593
    index = {r['query_id']: r for r in rows}
    for m, s in result['summary'].items():
        n = sum(labels[r['selected'][m]] == roles[r['query_id']]['identity'] for r in rows)
        rescue = sum(r['correct'][m] and not r['correct']['RAW'] for r in rows)
        loss = sum(not r['correct'][m] and r['correct']['RAW'] for r in rows)
        assert (n, rescue, loss) == (s['correct'], s['rescue_vs_RAW'], s['break_vs_RAW'])
        assert n == 426 + rescue - loss
    calibration = []
    predictions = {}
    for f in range(5):
        fv = read(checked(v['fold_validations'][f]))
        p = read(checked(fv['payload']))
        assert p['fold'] == f and fv['status'] == 'RANK_ACTION_NESTED_FRESH_NUMPY_PASS'
        predictions.update({r['query_id']: r for r in p['predictions']})
        for arm, model in [('RANK_ONLY6', 'LINEAR_ACTION3'), ('DIAG12', 'DIAG_ACTION3')]:
            h = p['action_parameters'][arm]
            t = np.array([float.fromhex(x) for x in h['theta_hex']])
            s = np.array([float.fromhex(x) for x in h['scale_hex']])
            records = p['calibration'][arm]
            x = np.array([[float.fromhex(x) for x in r['features_hex']] for r in records])
            g = ((x[:, 0] / s[0]) * t[0] + (x[:, 1] / s[1]) * t[1]) + t[2]
            raw = np.array([r['raw_correct'] for r in records])
            top = np.array([r['top_correct'] for r in records])
            delta = top.astype(int) - raw.astype(int)
            assert delta.tolist() == [r['delta'] for r in records]
            chosen = np.where(g > 0, top, raw)
            rescue = int(np.count_nonzero(chosen & ~raw))
            loss = int(np.count_nonzero(~chosen & raw))
            calibration.append(dict(fold=f, arm=arm, N=len(records), raw=int(raw.sum()),
                correct=int(chosen.sum()), rescue=rescue, loss=loss, net=rescue-loss,
                available_rescues=int(np.count_nonzero(top & ~raw)),
                held_correct_challengers=int(np.count_nonzero((g <= 0) & top & ~raw)),
                optimizer_success=h['diagnostics']['optimizer_success'],
                gradient_inf=h['diagnostics']['gradient_inf_norm'], theta=t.tolist(), rms=s.tolist()))
            for r in p['predictions']:
                m = r['models'][model]
                x = np.array([float.fromhex(z) for z in m['features_hex']])
                g = ((x[0] / s[0]) * t[0] + (x[1] / s[1]) * t[1]) + t[2]
                assert g.hex() == m['score_hex']
                pos = r['challenger_positions'][m['top_index']] if g > 0 else r['winner']
                assert r['candidate_physical_rows'][pos] == m['selected'] == index[r['query_id']]['selected'][model]
    breakdown = {}
    for model, arm in [('LINEAR_ACTION3', 'RANK_ONLY6'), ('DIAG_ACTION3', 'DIAG12')]:
        lost, gained, errors = Counter(), Counter(), Counter()
        for r in rows:
            raw = r['correct']['RAW']
            top = labels[r['top_physical'][arm]] == roles[r['query_id']]['identity']
            switch = r['selected'][model] != r['selected']['RAW']
            if r['correct']['GAP_BIAS2'] and not r['correct'][model]:
                reason = 'new_RAW_break' if raw else ('correct_top_HELD' if top and not switch else 'top_wrong')
                lost[reason] += 1
            if not r['correct']['GAP_BIAS2'] and r['correct'][model]:
                gained['old_RAW_break_reverted' if raw else 'new_rescue'] += 1
            if not r['correct'][model]:
                reason = ('target_absent' if not r['target_in_C128'] else
                    'RAW_correct_broken' if raw else 'correct_top_HELD' if top else 'top_wrong')
                errors[reason] += 1
        breakdown[model] = dict(losses_vs_492=dict(lost), gains_vs_492=dict(gained), all_errors=dict(errors))
        assert sum(lost.values()) == result['comparisons'][model]['GAP_BIAS2']['loss']
        assert sum(gained.values()) == result['comparisons'][model]['GAP_BIAS2']['rescue']
    blocked_rescues = []
    for r in rows:
        if not r['original40_ranking_blocked']:
            continue
        if labels[r['top_physical']['DIAG12']] != roles[r['query_id']]['identity']:
            continue
        p = predictions[r['query_id']]['models']['DIAG_ACTION3']
        blocked_rescues.append(dict(query_id=r['query_id'], original_query_id=r['original_query_id'], fold=r['fold'],
            score=float.fromhex(p['score_hex']), features=[float.fromhex(x) for x in p['features_hex']],
            switched=p['switch'], final_correct=r['correct']['DIAG_ACTION3']))
    assert len(blocked_rescues) == 4 and not any(r['switched'] for r in blocked_rescues)
    output = dict(status='RANK_ACTION_POSTJOIN_DIAGNOSIS_PASS', result=v['result'],
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        natural_fits=0, threshold_search=False, calibration=calibration,
        calibration_note='Each row is action-fit performance on inner rank OOF within one outer TRAIN; not independent action validation; queries repeat across outer TRAIN sets.',
        breakdown=breakdown, original40_DIAG_rank_rescues=blocked_rescues,
        interpretation='The fixed two-input gate misses correct tops inside calibration and on outer heldout. This isolates decision suppression, not a proof that missing features alone explain all failures.')
    (OUT / 'post_join_analysis.json').write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
