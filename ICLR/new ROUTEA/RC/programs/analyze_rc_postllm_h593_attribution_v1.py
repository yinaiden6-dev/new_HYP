#!/usr/bin/env python3
"""Post-hoc, no-training attribution from sealed H593 OOF candidate scores.

Swaps only existing contents and INTERNAL3 heads. This is a diagnostic, not a
new fitted model, causal mediation estimate, or untouched confirmation.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/rc_postllm_h593_v1'
OUT = ROOT / 'results/rc_postllm_h593_attribution_v1'


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def replay(raw, content, winner, theta):
    indices = np.array([i for i in range(128) if i != winner])
    content = np.asarray(content, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    x = np.column_stack([
        (raw[indices] - raw[winner]) / max(raw.std(), 1e-12),
        (content[indices] - content[winner]) /
        (np.abs(content[indices]) + abs(content[winner]) + 1e-12),
        np.ones(127),
    ])
    scores = np.zeros(128)
    scores[indices] = x @ np.asarray(theta, dtype=np.float64)
    challenger = int(indices[np.argmax(scores[indices])])
    selected = challenger if scores[challenger] > 0 else winner
    return scores, selected, challenger


def main():
    source = read(SOURCE / 'result.json')
    outputs, bindings, errors = [], [], []
    for r in source['rows']:
        qid, fold = r['query_id'], r['fold']
        base = SOURCE / f'fold{fold}/POST_REAL/held'
        pp, cp = base/'native'/f'{qid}.json', base/'constant'/f'{qid}.json'
        p, c = read(pp), read(cp)
        bindings.extend([binding(pp), binding(cp)])
        assert p['query_id'] == c['query_id'] == qid
        assert p['candidate_ids'] == c['candidate_ids']
        assert p['raw_scores'] == c['raw_scores'] and p['winner_index'] == c['winner_index']
        old = r['decisions']['EXTERNAL_INTERNAL3']['theta']
        final = r['decisions']['POST_REAL']['theta']
        specs = {
            'UNADAPTED_WARM_HEAD': (p['L0'], old, 'EXTERNAL_INTERNAL3'),
            'UNADAPTED_FINAL_HEAD': (p['L0'], final, None),
            'ADAPTED_WARM_HEAD': (p['L'], old, None),
            'ADAPTED_FINAL_HEAD': (p['L'], final, 'POST_REAL'),
            'CONSTANT_FINAL_HEAD': (c['L'], final, 'POST_REAL_CONSTANT'),
        }
        row = {k: r[k] for k in ['query_id', 'fold', 'component', 'target_in_C128']}
        row.update(raw_correct=r['correct']['RAW'], paths={})
        scores_by_path = {}
        for name, (content, theta, original) in specs.items():
            scores, pos, challenger = replay(p['raw_scores'], content, p['winner_index'], theta)
            identity = p['candidate_identities'][pos]
            if original:
                errors.append(float(np.max(np.abs(scores - r['decisions'][original]['scores128']))))
                assert identity == r['selected'][original]
            scores_by_path[name] = scores
            row['paths'][name] = dict(position=pos, identity=identity,
                correct=identity == r['identity'], switched=pos != p['winner_index'],
                top_challenger_correct=p['candidate_identities'][challenger] == r['identity'])
        # Actual decision failures of the constant-M counterfactual on real-M rescues.
        real, constant = row['paths']['ADAPTED_FINAL_HEAD'], row['paths']['CONSTANT_FINAL_HEAD']
        if real['correct'] and not row['raw_correct']:
            row['constant_on_real_rescue'] = ('also_correct' if constant['correct'] else
                'correct_challenger_below_hold' if constant['top_challenger_correct'] else
                'wrong_challenger_ranked_first')
        if r['target_in_C128']:
            target = np.asarray(r['target_positions'], dtype=int)
            sr, sc = scores_by_path['ADAPTED_FINAL_HEAD'], scores_by_path['CONSTANT_FINAL_HEAD']
            row['target_logit_M_change'] = float(sr[target].max() - sc[target].max())
        row['external_correct'] = r['correct']['EXTERNAL_ADDITIVE4']
        outputs.append(row)
    assert len(outputs) == len({r['query_id'] for r in outputs}) == 593
    assert max(errors) < 1e-10
    totals = {}
    for name in outputs[0]['paths']:
        totals[name] = {
            'correct': sum(r['paths'][name]['correct'] for r in outputs),
            'rescues_RAW': sum(r['paths'][name]['correct'] and not r['raw_correct'] for r in outputs),
            'breaks_RAW': sum(not r['paths'][name]['correct'] and r['raw_correct'] for r in outputs),
            'switches': sum(r['paths'][name]['switched'] for r in outputs),
            'fold_correct': [sum(r['paths'][name]['correct'] for r in outputs if r['fold'] == f) for f in range(5)],
        }
    real_rescues = [r for r in outputs if r['paths']['ADAPTED_FINAL_HEAD']['correct'] and not r['raw_correct']]
    ext_rescues = [r for r in outputs if r['external_correct'] and not r['raw_correct']]
    real_ids, ext_ids = {r['query_id'] for r in real_rescues}, {r['query_id'] for r in ext_rescues}
    summary = dict(totals=totals, max_original_logit_error=max(errors),
        original_logits_recomputed=len(errors)*127,
        constant_counterfactual_on_54_real_rescues=dict(Counter(r['constant_on_real_rescue'] for r in real_rescues)),
        internal_external_rescue_overlap=dict(common=len(real_ids & ext_ids), internal_only=len(real_ids-ext_ids), external_only=len(ext_ids-real_ids)),
        real_rescues_target_logit_increased_by_M=sum(r['target_logit_M_change'] > 0 for r in real_rescues))
    result = dict(status='SAVED_OOF_CONTENT_HEAD_SWAP_REPLAY_PASS', source=binding(SOURCE/'result.json'),
        scope='Opened H593 original grouped 5-fold OOF; original natural ColNomic C128; exploratory fixed-parameter replay, not refit or external confirmation',
        caveats=['Head swaps change feature calibration; a worse swap is not a necessary-role proof.',
                 'Ranking/acceptance counts classify observed failures; they are not additive causal percentages.',
                 'No claim about patch-common versus patch-specific response in full H593; that decomposition was only run on the earlier pilot.'],
        summary=summary, input_files=bindings, rows=outputs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
