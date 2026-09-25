#!/usr/bin/env python3
"""Reconstruct an already-opened POST rescue from saved full-C128 scores.

No new model forward, training, candidate selection, or scheduler action.
The chosen case is descriptive post-hoc analysis, not a new held evaluation.
"""
from pathlib import Path
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/rc_postllm_m_signal_decomposition_v2/queries/H593-90acde9b0567a472f4232120/result.json'
OUT = ROOT / 'results/rc_existing_evidence_attribution_v1/internal_decision.json'


def binding(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    source = json.loads(SOURCE.read_text())
    assert source['status'] and len(source['candidate_ids']) == 128
    target, winner = 87, source['winner_index']
    assert target != winner and winner == 104
    rows = {}
    max_error = 0.
    for arm, item in source['arms'].items():
        decision, content = item['decision'], item['L']
        theta = decision['theta']
        challengers = decision['challenger_positions']
        assert challengers == [i for i in range(128) if i != winner]
        raw = source['raw_scores']; std = max(statistics.pstdev(raw), 1e-12)
        reconstructed = [theta[0] * (raw[i]-raw[winner])/std
            + theta[1] * (content[i]-content[winner])/(abs(content[i])+abs(content[winner])+1e-12)
            + theta[2] for i in challengers]
        error = max(abs(a-b) for a,b in zip(reconstructed,decision['logits']))
        max_error = max(max_error,error)
        assert error < 1e-10
        top = max(range(127),key=lambda j:reconstructed[j])
        predicted = challengers[top] if reconstructed[top] > 0 else winner
        assert predicted == decision['prediction_position']
        wrong = max((v,i) for v,i in zip(reconstructed,challengers) if i != target)
        raw_term = theta[0]*(raw[target]-raw[winner])/std
        sym_l = (content[target]-content[winner])/(abs(content[target])+abs(content[winner])+1e-12)
        target_z = reconstructed[challengers.index(target)]
        rows[arm] = dict(target_L=content[target], RAW_winner_L=content[winner],
            target_minus_RAW_L=content[target]-content[winner],
            target_logit=target_z, target_raw_term=raw_term,
            target_content_term=theta[1]*sym_l, bias=theta[2],
            strongest_wrong_challenger_position=wrong[1], strongest_wrong_challenger_logit=wrong[0],
            target_margin_including_HOLD=target_z-max(0.,wrong[0]),
            prediction_position=predicted, prediction_identity=decision['prediction_identity'],
            target_beats_other_challengers=target_z>wrong[0])
    h = source['arms']['NATIVE']['decision']['theta']
    threshold = -(rows['NATIVE']['target_raw_term']+h[2])/h[1]
    constant, native = rows['CONSTANT'],rows['NATIVE']
    result = dict(status='SAVED_C128_DECISION_ATTRIBUTION_RECOUNT_PASS', source=binding(SOURCE),
        program=binding(Path(__file__)), query_id=source['query_id'], target_position=target,
        RAW_winner_position=winner, theta=h, candidate_count=128, arms=rows,
        max_reconstructed_logit_error=max_error,
        target_required_sym_L_to_beat_HOLD=threshold,
        positive_content_ratio_threshold_ignoring_epsilon=(1+threshold)/(1-threshold),
        target_content_increase=native['target_L']-constant['target_L'],
        RAW_winner_content_increase=native['RAW_winner_L']-constant['RAW_winner_L'],
        scope='One previously opened successful probe; saved CPU path; not a population causal percentage',
        new_training=False, new_model_forwards=False,
        explanation='The fixed head and HOLD threshold remain unchanged. Existing-match content changes already make the target beat all wrong challengers and HOLD; rematching adds margin. Both target and RAW wrong-reference similarities rise.')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('arms',)},ensure_ascii=False))


if __name__ == '__main__':
    main()
