#!/usr/bin/env python3
"""Join all frozen external predictions once; resample source-video groups."""
import collections
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
import run_rc_train128_disagreement_oof4_v1 as P

OUT = ROOT / 'results/rc_new_hyp_grozi120_external_v1'
AUTH = ROOT / 'registry/rc_new_hyp_grozi120_inference_authority_v1_20260913.json'


def comparison(rows, baseline, model):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['component']].append(int(row['correct'][model]) - int(row['correct'][baseline]))
    means = np.array([np.mean(xs) for _, xs in sorted(groups.items())])
    rng = np.random.default_rng(20260913)
    boot = np.mean(means[rng.integers(len(means), size=(100000, len(means)))], axis=1)
    low, high = np.quantile(boot, [.025, .975])
    rescue = sum(not r['correct'][baseline] and r['correct'][model] for r in rows)
    loss = sum(r['correct'][baseline] and not r['correct'][model] for r in rows)
    return dict(rescue=rescue, loss=loss, net=rescue-loss, query_difference=(rescue-loss)/len(rows),
                equal_video_difference=float(means.mean()), video_bootstrap95=[float(low), float(high)],
                video_groups=len(means), reliable_positive=bool(rescue>loss and means.mean()>0 and low>0))


def main():
    M.need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    authority = M.read(AUTH)
    M.checked(authority['sources']['join_program'])
    predictions, validations = [], []
    for shard in range(60):
        folder = OUT / 'predictions' / f'shard{shard:02d}'
        v = M.read(folder / 'validation.json')
        M.need(v['status'] == 'GROZI_FROZEN_HEAD_NUMPY_ACTION_PASS' and v['authority'] == M.bind(AUTH), 'ALL60_QUALIFIED')
        payload = M.read(M.checked(v['payload']))
        M.need(payload['authority'] == M.bind(AUTH) and len(payload['records']) == 8 and payload['target_reads'] == 0, 'SEALED_WORKER_BOUNDARY')
        predictions.extend(payload['records'])
        validations.append(M.bind(folder / 'validation.json'))
    workers = M.read(OUT / 'worker_manifest.json')['records']
    M.need([r['query_id'] for r in predictions] == [w['query_id'] for w in workers], 'ALL480_WORKER_ORDER')
    M.write(OUT / 'all_predictions_prejoin_seal.json', dict(authority=M.bind(AUTH), validations=validations, queries=480))
    # External target labels are consumed only after all 60 prediction shards are sealed.
    labels = list(P.gallery_labels()[0]) + [r['identity'] for r in M.read(OUT / 'gallery_append_manifest.json')['records']]
    source = authority['curator_after_all_seals']
    roles = {r['query_id']: r for r in M.read(M.checked(source))['records']}
    rows = []
    for p in predictions:
        role = roles[p['query_id']]
        raw_rank = next(i+1 for i, physical in enumerate(p['raw_ranked_physical_rows']) if labels[physical] == role['identity'])
        correct = {'RAW': labels[p['raw_selected']] == role['identity']}
        ranks = {'RAW': raw_rank}
        selected, holds = {'RAW': p['raw_selected']}, {}
        for model, prediction in p['models'].items():
            chosen = prediction['selected']; selected[model] = chosen
            correct[model] = labels[chosen] == role['identity']
            selected_rank = p['raw_ranked_physical_rows'].index(chosen)+1
            ranks[model] = 1 if correct[model] else raw_rank + int(selected_rank > raw_rank)
            holds[model] = chosen == p['raw_selected']
        rows.append(dict(query_id=p['query_id'], identity=role['identity'], component=role['component'],
                         source_video=role['video'], source_frame=role['frame'], target_in_C128=raw_rank<=128,
                         correct=correct, ranks=ranks, selected=selected, holds=holds))
    M.need(len(rows)==480 and len({r['identity'] for r in rows})==120 and len({r['component'] for r in rows})==27, 'FROZEN_DENOMINATORS')
    counts = {m: sum(r['correct'][m] for r in rows) for m in rows[0]['correct']}
    mrr = {m: float(np.mean([1/r['ranks'][m] for r in rows])) for m in counts}
    primary_baselines = ['RAW', 'COST4', 'GROUP_COST4', 'RAW2_CE', 'COST1_CBIND']
    pairs = [(b, 'COST1') for b in primary_baselines] + [('COST1', 'CE'), ('GROUP_COST4', 'CE'), ('RAW', 'CE'), ('CE_CBIND', 'CE')]
    comparisons = {b+'__to__'+m: comparison(rows, b, m) for b, m in pairs}
    performance = all(comparisons[b+'__to__COST1']['reliable_positive'] for b in ('RAW','COST4','GROUP_COST4'))
    mechanism = all(comparisons[b+'__to__COST1']['reliable_positive'] for b in ('RAW2_CE','COST1_CBIND'))
    per_video = {}
    for video in sorted({r['source_video'] for r in rows}):
        rr = [r for r in rows if r['source_video']==video]
        per_video[str(video)] = dict(count=len(rr), identities=len({r['identity'] for r in rr}), correct={m:sum(r['correct'][m] for r in rr) for m in counts})
    result = dict(status='GROZI120_EXTERNAL_FROZEN480_COMPLETE', authority=M.bind(AUTH),
                  model_lineage='all opened H593 final heads; zero GroZi training or calibration',
                  query_panel='480 dataset-provided product crops;120 identities;27 source videos',
                  candidate_source='legacy5413 physical gallery plus120 new references; naturalC128',
                  action='all127 challengers; SWITCH iff maxlogit>0 else HOLD',
                  counts=counts, MRR=mrr, target_recall_C128=sum(r['target_in_C128'] for r in rows),
                  comparisons=comparisons, per_video=per_video, rows=rows,
                  primary='COST1', secondary='CE', performance_gate=performance, binding_gate=mechanism,
                  scoped_external_new_HYP_GO=performance and mechanism,
                  GO_scope='This fixed GroZi cropped-product identity benchmark only; no full-scene localization, multi-store generalization, ownership or complete Omnimemory claim.',
                  old_EVAL32_128_retested=False, RPC_results_pending=True)
    M.write(OUT / 'result.json', result)
    # Separate count computation and complete partition accounting.
    independently = collections.Counter()
    for p in predictions:
        target = roles[p['query_id']]['identity']
        independently['RAW'] += labels[p['raw_selected']]==target
        for m, v in p['models'].items(): independently[m] += labels[v['selected']]==target
    M.need(dict(independently)==counts and sum(v['count'] for v in per_video.values())==480, 'INDEPENDENT_COUNTS_AND_PARTITION')
    M.write(OUT / 'result_validation.json', dict(status='GROZI480_ALL_SEALS_COUNTS_GROUPS_PASS', authority=M.bind(AUTH), result=M.bind(OUT/'result.json'), all_prediction_validations=validations))
    print(json.dumps(dict(counts=counts, performance_gate=performance, binding_gate=mechanism, scoped_external_new_HYP_GO=performance and mechanism)), flush=True)


if __name__ == '__main__': main()
