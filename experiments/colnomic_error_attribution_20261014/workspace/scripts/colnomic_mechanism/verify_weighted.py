"""Independently certify weighted full-gallery ranks and saved bound evidence."""
import argparse
from pathlib import Path
import numpy as np
import torch

from scripts.colnomic_mechanism.common import ROOT, data, read, rows, write, csvout, sha, workspace_path
from scripts.colnomic_mechanism.weighted_maxsim import ETAS, weight_mask, score_bounds
from evaluation.scoring import valid_tokens


def verify(folder, d):
    protocol = read(folder / 'protocol.json')
    qb, rb = d['qb'], d['rb']
    ledger = {r['photo_id']: r for r in read(ROOT / 'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    hashes = np.asarray(rb['sha256'])
    norms = np.array([float(valid_tokens(z).double().norm(dim=1).max()) for z in rb['embeddings']])
    means = d['image']['scores'].numpy()
    alignment = {}
    gamma, lam = protocol['alignment_gamma'], protocol['spatial_lambda']
    if protocol.get('spatial_manifest_path'):
        from scripts.colnomic_mechanism.reviewed_box_sensitivity import load_spatial_inputs
        source = workspace_path(protocol['spatial_manifest_path'])
        assert sha(source) == protocol['spatial_manifest_SHA256']
        means, alignment, _, parameters = load_spatial_inputs(source, d, selection_key=protocol.get('selection_key','B03.combined'))
        assert gamma == parameters['gamma'] and lam == parameters['lambda_value']
    elif gamma or lam:
        cached = np.load(ROOT / 'results/token_mechanism_experiments/20261013_v1/stage2_geometry/full_gallery_scores.npz')
        k = int(np.argmin(abs(cached['lambdas'] - lam)))
        assert abs(cached['lambdas'][k] - lam) < 1e-10
        means = cached['geo_penalty'][k].copy()
        qis = {q: i for i, q in enumerate(qb['ids'])}
        ris = {r: i for i, r in enumerate(rb['ids'])}
        for fit in read(ROOT / 'results/token_mechanism_experiments/20261013_v1/stage2_geometry/alignments.json'):
            qi, ri = qis[fit['photo_id']], ris[fit['reference_id']]
            alignment[qi, ri] = fit
            means[qi, ri] += gamma * fit['subject_tokens'] / len(valid_tokens(qb['embeddings'][qi]))
    allrows = []
    maximum_bound_error = 0.
    slacks = []
    for qi, qid in enumerate(qb['ids']):
        path = folder / 'queries' / qid
        complete = read(path / 'complete.json')
        assert sha(path / 'scores_and_bounds.npz') == complete['scores_SHA256'], qid
        assert complete['query_tensor_SHA256'] == d['metadata']['photos'][qi]['all_tokens_sha256']
        a = np.load(path / 'scores_and_bounds.npz')
        exact, lower, upper, evaluated = [a[k] for k in ['exact', 'lower', 'upper', 'evaluated']]
        assert exact.shape == lower.shape == upper.shape == (len(ETAS), len(hashes))
        assert np.isfinite(lower).all() and np.isfinite(upper).all() and (lower <= upper).all()
        mask = weight_mask(valid_tokens(qb['embeddings'][qi]))
        assert np.array_equal(mask, a['query_weight_mask'])
        flags = np.array([(qi, ri) in alignment for ri in range(len(hashes))], dtype=float)
        expected_low, expected_high = score_bounds(valid_tokens(qb['embeddings'][qi]), mask, norms, means[qi],
            tolerance=protocol['safety_tolerance'], spatial_lambda=lam * flags, alignment_gamma=gamma * flags)
        error = max(float(abs(expected_low - lower).max()), float(abs(expected_high - upper).max()))
        maximum_bound_error = max(maximum_bound_error, error)
        assert error < 1e-12, (qid, error)
        remaining = np.setdiff1d(np.arange(len(hashes)), evaluated)
        assert np.isfinite(exact[:, evaluated]).all() and np.isnan(exact[:, remaining]).all()
        assert (exact[:, evaluated] >= lower[:, evaluated] - 1e-8).all()
        assert (exact[:, evaluated] <= upper[:, evaluated] + 1e-8).all()
        assert abs(exact[0, evaluated] - means[qi, evaluated]).max() < 1e-7
        positive_rows = np.flatnonzero(np.isin(hashes, ledger[qid]['positive_reference_sha256']))
        assert set(positive_rows) <= set(evaluated.tolist())
        minimum = 1.
        for ei, eta in enumerate(ETAS):
            order = sorted(evaluated.tolist(), key=lambda ri: (-float(exact[ei, ri]), ri))
            unique, seen = [], set()
            for ri in order:
                if hashes[ri] not in seen:
                    unique.append(ri)
                    seen.add(hashes[ri])
            rank = next(k + 1 for k, ri in enumerate(unique) if ri in positive_rows)
            stored = complete['rankings'][ei]
            assert stored['eta'] == eta and stored['photo_id'] == qid and stored['rank'] == rank
            assert stored['top1'] == rb['ids'][unique[0]] and stored['top10'] == [rb['ids'][ri] for ri in unique[:10]]
            assert stored['correct'] == (rank == 1)
            threshold = min(exact[ei, unique[9]], exact[ei, positive_rows].max())
            slack = float((threshold - upper[ei, remaining]).min()) if len(remaining) else 1.
            assert slack > 0, (qid, eta, slack)
            minimum = min(minimum, slack)
            allrows.append(stored)
        assert abs(minimum - complete['proof_min_slack']) < 1e-12
        slacks.append(minimum)
        if qi % 100 == 0:
            print(folder.name, qi + 1, '/', len(qb['ids']), flush=True)
    saved = rows(folder / 'all_parameter_rankings.csv')
    assert len(saved) == len(allrows)
    for a, b in zip(saved, allrows):
        assert a['photo_id'] == b['photo_id'] and float(a['eta']) == b['eta'] and int(a['rank']) == b['rank']
    grid = rows(folder / 'validation_grid.csv')
    eta = float(max(grid, key=lambda r: (float(r['package_macro_r1']), float(r['package_macro_mrr']), -float(r['eta'])))['eta'])
    assert eta == read(folder / 'selection.json')['eta']
    baseline = {r['photo_id']: r for r in allrows if r['eta'] == 0}
    for m in rows(folder / 'metrics.csv'):
        chosen = [r for r in allrows if r['eta'] == float(m['eta']) and (m['split'] == 'all' or r['split'] == m['split'])]
        rescued = sum(r['correct'] and not baseline[r['photo_id']]['correct'] for r in chosen)
        harmed = sum(not r['correct'] and baseline[r['photo_id']]['correct'] for r in chosen)
        assert int(m['photos']) == len(chosen)
        assert int(m['rescued']) == rescued and int(m['harmed']) == harmed and int(m['net']) == rescued - harmed
        assert abs(float(m['r1']) - np.mean([r['correct'] for r in chosen])) < 1e-12
    return dict(queries=len(qb['ids']), gallery_rows=len(hashes), parameter_rankings=len(allrows),
        recomputed_bounds_max_error=maximum_bound_error, minimum_uncomputed_rank_slack=min(slacks),
        selected_eta=eta, all_gallery_rows_certified=True, all_gallery_scalar_scores_computed=False,
        certificate_scope='exact top10 and best positive rank from finite exact scores plus conservative bounds; uncomputed scalar values remain NaN')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folders', nargs='+', default=['weighted_original', 'weighted_spatial'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    d = data()
    checks = {name: verify(ROOT / 'results/colnomic_full_audit/20261013_resume' / name, d)
              for name in args.folders}
    write(args.output, dict(status='passed', independently_executed_checks=checks, split_verification=d['split_verification']))
    print('All weighted rank certificates replayed', flush=True)


if __name__ == '__main__':
    main()
