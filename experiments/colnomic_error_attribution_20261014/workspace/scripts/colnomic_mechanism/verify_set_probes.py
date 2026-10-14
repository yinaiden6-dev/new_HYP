"""Independent read-only replays of append-only set-probe artifacts (CPU)."""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT, data, read, rows, write, sha
from scripts.colnomic_mechanism.probes import logistic_predict, shuffled_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, default=ROOT / 'results/colnomic_full_audit/20261013_resume/set_probes')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    folder = args.folder
    d = data()
    ledger = {r['photo_id']: r for r in read(ROOT / 'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    axes = read(folder / 'natural/full_gallery/score_axes.json')
    checks = dict(axes_equal_original_banks=axes['query_ids'] == d['qb']['ids'] and axes['reference_ids'] == d['rb']['ids']
                  and axes['reference_sha256'] == d['rb']['sha256'],
                  original_split_checks_passed=d['split_verification']['status'] == 'passed')
    hashes = np.asarray(d['rb']['sha256'])
    summary = []
    for kind in ['raw_maxsim', 'maxsim_only', 'pooled_only', 'pooled_plus_maxsim', 'kernel_set_only']:
        path = folder / 'natural/full_gallery' / kind
        scores = d['image']['scores'].numpy() if kind == 'raw_maxsim' else np.load(path / 'scores.npy', mmap_mode='r')
        saved_order = np.load(path / 'stable_sorted_reference_indices.npy', mmap_mode='r')
        saved_ranks = np.load(path / 'sha_design_ranks.npy', mmap_mode='r')
        cases = rows(path / 'query_rankings.csv')
        passed = scores.shape == saved_order.shape == saved_ranks.shape == (1117, 5405) and len(cases) == 1117
        true_ranks = []
        maximum_score_order_errors = 0
        for qi, qid in enumerate(d['qb']['ids']):
            score = scores[qi]
            order = np.argsort(-score, kind='stable')
            maximum_score_order_errors = max(maximum_score_order_errors, int(np.count_nonzero(order != saved_order[qi])))
            passed &= bool(np.isfinite(score).all()) and np.array_equal(order, saved_order[qi])
            rankmap = {}
            for ri in order:
                if hashes[ri] not in rankmap:
                    rankmap[hashes[ri]] = len(rankmap) + 1
            rank = min(rankmap[h] for h in ledger[qid]['positive_reference_sha256'])
            true_ranks.append(rank)
            reconstructed = np.asarray([rankmap[h] for h in hashes], dtype=np.int32)
            case = cases[qi]
            passed &= (case['photo_id'] == qid and int(case['positive_rank']) == rank and
                       case['top_reference_id'] == d['rb']['ids'][order[0]] and
                       np.array_equal(reconstructed, saved_ranks[qi]))
        checks[kind + '_full_gallery_rank_replay'] = bool(passed)
        summary.append(dict(probe=kind, queries=1117, reference_rows=5405, passed=bool(passed),
                            maximum_sorted_index_mismatches=maximum_score_order_errors,
                            recall_at_1=float((np.asarray(true_ranks) == 1).mean())))
    a = np.load(folder / 'natural/restricted_inputs.npz')
    prior = np.load(ROOT / 'results/token_mechanism_experiments/20261013_v2/stage3_probes/identity_probe_inputs.npz')
    inputs = dict(maxsim_only=prior['maxsim'], pooled_only=prior['pooled'],
                  pooled_plus_maxsim=np.column_stack([prior['pooled'], prior['maxsim']]), kernel_set_only=a['kernel_set_only'])
    y, packages, parts = [a[k] for k in ['labels', 'package_ids', 'splits']]
    nullrows = rows(folder / 'natural/shuffle_predictions.csv')
    replay = []
    train = parts == 'train'
    count = {p: int((packages == p).sum()) for p in set(packages)}
    weights = np.array([1. / count[p] for p in packages])
    for kind, x in inputs.items():
        for seed in [17, 31, 53]:
            stored = torch.load(folder / 'natural' / f'{kind}_shuffle_{seed}_head.pt', map_location='cpu', weights_only=True)
            model = {k: stored[k].numpy() for k in ['mean', 'scale', 'coefficient']}
            selected = [r for r in nullrows if r['probe'] == kind and int(r['seed']) == seed]
            logits = logistic_predict(model, x)
            error = float(np.max(np.abs(logits - np.array([float(r['logit']) for r in selected]))))
            expected = shuffled_labels(y[train], packages[train], seed)
            observed = np.array([int(r['shuffled_training_label']) for r in selected if r['split'] == 'train'])
            mean = np.average(x[train], axis=0, weights=weights[train])
            meanerror = float(np.max(np.abs(mean - model['mean'])))
            passed = len(selected) == len(y) and error < 1e-10 and meanerror < 1e-12 and np.array_equal(expected, observed)
            checks[kind + f'_shuffle_{seed}_replay_and_train_standardization'] = bool(passed)
            replay.append(dict(probe=kind, seed=seed, prediction_maximum_error=error, training_mean_maximum_error=meanerror,
                               labels_equal_recorded_shuffle=bool(np.array_equal(expected, observed)), passed=bool(passed)))
    hard = rows(folder / 'natural/all1117_hard_pair_predictions.csv')
    hardqids = {r['photo_id'] for r in hard}
    checks['natural_hard_pair_1117_coverage'] = hardqids == set(d['qb']['ids'])
    landmarks = rows(folder / 'natural/kernel_landmark_index.csv')
    allowed_q = {qid for qid in d['qb']['ids'] if d['splits'][qid] == 'train'}
    allowed_r = {rid for qid, row in ledger.items() if d['splits'][qid] == 'train' for rid in row['positive_reference_ids']}
    checks['kernel_landmarks_training_only'] = all(r['source_id'] in allowed_q | allowed_r for r in landmarks)
    completion = read(folder / 'completion.json')
    snapshot = list(folder.glob('source_executed_' + completion['source_code_sha256'][:12] + '*.py'))
    checks['executed_source_snapshot_sha'] = len(snapshot) == 1 and sha(snapshot[0]) == completion['source_code_sha256']
    result = dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
                  complete_gallery_replay=summary, shuffle_replay=replay, split_verification=d['split_verification'],
                  scope='scores/rankings/null predictions and standardization; no new visual-audit or causal claim')
    output = args.output or folder / 'independent_verification.json'
    write(output, result)
    print(result['status'], 'checks', len(checks), 'failed', [k for k, v in checks.items() if not v], flush=True)
    assert all(checks.values()), 'Independent replay failed'


if __name__ == '__main__':
    main()
