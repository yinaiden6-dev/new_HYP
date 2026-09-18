#!/usr/bin/env python3
"""Join all sealed predictions and report exploratory paired patient evidence."""
import collections
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
OUT = ROOT / 'results/rc_new_hyp_isic_transfer_v1'
AUTH = ROOT / 'registry/rc_new_hyp_isic_inference_authority_v1_20260914.json'


def comparison(rows, baseline, model, draws=100000):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['component']].append(int(r['correct'][model]) - int(r['correct'][baseline]))
    delta = np.array([np.mean(v) for _, v in sorted(groups.items())], dtype=np.float64)
    rng = np.random.default_rng(20260914)
    # Chunking bounds memory and leaves the fixed RNG draw sequence unchanged.
    boot = np.concatenate([delta[rng.integers(len(delta), size=(min(2000, draws-i), len(delta)))].mean(axis=1)
                           for i in range(0, draws, 2000)])
    ci = np.quantile(boot, [.025, .975]).tolist()
    rescue = sum(not r['correct'][baseline] and r['correct'][model] for r in rows)
    loss = sum(r['correct'][baseline] and not r['correct'][model] for r in rows)
    return dict(rescue=rescue, loss=loss, net=rescue-loss, query_difference=(rescue-loss)/len(rows),
        equal_patient_difference=float(delta.mean()), patient_bootstrap95=ci, patients=len(delta),
        reliable_positive=bool(rescue > loss and delta.mean() > 0 and ci[0] > 0))


def summarize(rows, models):
    return dict(queries=len(rows), correct={m: sum(r['correct'][m] for r in rows) for m in models},
        MRR={m: math.fsum(1 / r['ranks'][m] for r in rows) / len(rows) for m in models},
        recall_C128=sum(r['target_in_C128'] for r in rows))


def main():
    M.need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    a = M.read(AUTH); ab = M.bind(AUTH)
    M.checked(a['sources']['join_program'])
    predictions, validations = [], []
    for shard in range(68):
        folder = OUT / 'predictions' / f'shard{shard:02d}'
        v = M.read(folder / 'validation.json')
        M.need(v['status'] == 'ISIC_FROZEN_HEAD_NUMPY_ACTION_PASS' and v['authority'] == ab, 'ALL68_QUALIFIED')
        p = M.read(M.checked(v['payload']))
        M.need(p['authority'] == ab and len(p['records']) == (1 if shard == 67 else 8)
               and p['target_reads'] == p['training_updates'] == 0, 'SEALED_WORKER')
        predictions.extend(p['records']); validations.append(M.bind(folder / 'validation.json'))
    workers = M.read(M.checked(a['sources']['worker']))['records']
    M.need([p['query_id'] for p in predictions] == [w['query_id'] for w in workers]
           and len(predictions) == 537, 'COMPLETE537_ORDER')
    M.write(OUT / 'all_predictions_prejoin_seal.json', dict(authority=ab, validations=validations, queries=537))
    gallery = M.read(M.checked(a['sources']['gallery']))['records']
    labels = [r['identity'] for r in gallery]
    M.need([r['physical_row'] for r in gallery] == list(range(390)) and len(set(labels)) == 390, 'GALLERY390')
    roles = {r['query_id']: r for r in M.read(M.checked(a['curator_after_all_seals']))['records']}
    rows, independent, logit_actions = [], collections.Counter(), 0
    for p in predictions:
        role = roles[p['query_id']]; order = p['raw_ranked_physical_rows']
        M.need(len(order) == 390 and set(order) == set(range(390)) and p['raw_selected'] == order[0], 'NATURAL_FULL390')
        target = labels.index(role['identity']); raw_rank = order.index(target) + 1
        chosen = {'RAW': order[0]}; correct = {'RAW': order[0] == target}; ranks = {'RAW': raw_rank}
        holds, top_target_held = {}, {}
        challengers = [x for x in sorted(order[:128]) if x != order[0]]
        for model, v in p['models'].items():
            z = [float.fromhex(h) for h in v['logits_hex']]
            M.need(len(z) == 127 and all(math.isfinite(x) for x in z), 'LOGIT_VECTOR127')
            k = max(range(127), key=lambda j: z[j]); selected = challengers[k] if z[k] > 0 else order[0]
            M.need(selected == v['selected'], 'INDEPENDENT_LOGIT_ACTION'); logit_actions += 1
            chosen[model] = selected; correct[model] = selected == target; holds[model] = selected == order[0]
            ranks[model] = 1 if correct[model] else raw_rank + int(order.index(selected) + 1 > raw_rank)
            top_target_held[model] = bool(challengers[k] == target and z[k] <= 0)
        for model, selected in chosen.items():
            independent[model] += selected == target
            direct = [selected] + [x for x in order if x != selected]
            M.need(direct.index(target) + 1 == ranks[model], 'INDEPENDENT_MRR_REORDER')
        rows.append(dict(query_id=p['query_id'], identity=role['identity'], component=role['component'],
            target_in_C128=raw_rank <= 128, correct=correct, ranks=ranks, selected=chosen, holds=holds,
            target_top_challenger_held=top_target_held))
    M.need(len({r['identity'] for r in rows}) == 390 and len({r['component'] for r in rows}) == 346, 'FIXED_COHORT')
    models = list(rows[0]['correct']); overall = summarize(rows, models)
    M.need(dict(independent) == overall['correct'], 'INDEPENDENT_COUNTS')
    baselines = ('RAW', 'COST4', 'GROUP_COST4', 'RAW2_CE', 'COST1_CBIND')
    pairs = [(b, 'COST1') for b in baselines] + [('COST1', 'CE'), ('RAW', 'CE'), ('CE_CBIND', 'CE')]
    comparisons = {b + '__to__' + m: comparison(rows, b, m) for b, m in pairs}
    actions = {}
    for m in models:
        if m == 'RAW': continue
        switched = [r for r in rows if not r['holds'][m]]
        actions[m] = dict(holds=537-len(switched), switches=len(switched),
            rescues_vs_RAW=sum(not r['correct']['RAW'] and r['correct'][m] for r in rows),
            breaks_vs_RAW=sum(r['correct']['RAW'] and not r['correct'][m] for r in rows),
            wrong_to_other_wrong=sum(not r['correct']['RAW'] and not r['correct'][m] for r in switched),
            target_top_challenger_held=sum(r['target_top_challenger_held'][m] for r in rows))
    signal = all(comparisons[b+'__to__COST1']['reliable_positive'] for b in baselines)
    result = dict(status='ISIC537_OPENED_FROZEN_TRANSFER_COMPLETE', authority=ab, primary='COST1', secondary='CE',
        model_lineage='frozen full-H593 product heads; zero ISIC training/calibration',
        query_panel='537 non-identical dermoscopic query images;390 lesions;346 known patients;opened selected IMA++ cohort',
        candidate_source='independent390-reference gallery;naturalC128', action='all127 challengers;maxlogit>0 SWITCH else HOLD',
        counts=overall['correct'], MRR=overall['MRR'], target_recall_C128=overall['recall_C128'], comparisons=comparisons,
        by_patient={c: summarize([r for r in rows if r['component'] == c], models) for c in sorted({r['component'] for r in rows})},
        by_lesion={c: summarize([r for r in rows if r['identity'] == c], models) for c in sorted({r['identity'] for r in rows})},
        action_breakdown=actions, rows=rows, positive_exploratory_transfer_signal=signal,
        untouched_external_GO_claimed=False, clinical_diagnosis_claimed=False, training_updates=0,
        bootstrap=dict(estimand='equal-patient mean paired accuracy difference', draws=100000, seed=20260914),
        limitations=['opened cohort historically selected for mask/class availability', 'unknown pretraining exposure',
            'distinct image/pixel hashes do not certify independent capture sessions or rule out near duplicates',
            'no clinical diagnosis, ownership, universal generality, or no-text causal claim'],
        old_EVAL32_128_retested=False, product_results_unchanged=True)
    M.write(OUT / 'result.json', result)
    M.write(OUT / 'result_validation.json', dict(status='ISIC537_ALL_SEALS_COUNTS_MRR_ACTIONS_PASS', authority=ab,
        result=M.bind(OUT / 'result.json'), all_prediction_validations=validations, independent_logit_actions=logit_actions,
        query_count=537, patient_count=346))
    print(json.dumps(dict(counts=result['counts'], positive_exploratory_transfer_signal=signal)), flush=True)
    verifier = M.checked(a['sources']['independent_verifier'])
    subprocess.run([sys.executable, str(verifier)], check=True)


if __name__ == '__main__': main()
