#!/usr/bin/env python3
"""Independently audit complete MILK10k banks, scores and reported ranks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'other_datasets/MILK10K'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    manifest = json.loads((DATA / 'cache/test_manifest.json').read_text())
    model_config = json.loads((DATA / 'baseline_config.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', choices=list(model_config['models']))
    args = parser.parse_args()
    models = args.model or list(model_config['models'])
    with (DATA / 'MILK10k_Test_Metadata.csv').open() as f:
        metadata = {r['isic_id']: r for r in csv.DictReader(f)}
    audit = {'protocol': manifest['protocol'], 'models': {}, 'passed': False}
    assert len(manifest['pairs']) == 479
    for model in models:
        directory = DATA / 'results' / model
        metrics = json.loads((directory / 'metrics.json').read_text())
        config = json.loads((directory / 'config.json').read_text())
        predictions = [json.loads(r) for r in (directory / 'predictions.jsonl').read_text().splitlines()]
        assert config['manifest_signature'] == manifest['signature']
        assert config['model_spec'] == model_config['models'][model]
        banks = {}
        for side in ('clinical', 'dermoscopic'):
            bank = torch.load(DATA / 'cache/image_embeddings' / model / (side + '.pt'),
                              map_location='cpu', weights_only=True)
            assert bank['ids'] == [p[side]['image_id'] for p in manifest['pairs']]
            assert bank['lesion_ids'] == [metadata[i]['lesion_id'] for i in bank['ids']]
            assert len(bank['embeddings']) == 479
            assert bank['contract'] == config['model_spec']
            for relative, expected in zip(bank['image_paths'], bank['image_sha256']):
                assert relative.startswith('MILK10k_Test_Input/')
                assert sha(DATA / relative) == expected
            for vector in bank['embeddings']:
                assert torch.isfinite(vector).all()
                norms = torch.linalg.vector_norm(vector.double(), dim=-1)
                norms = norms[norms > 1e-8]
                assert norms.numel() and torch.all((norms - 1).abs() < .03)
            for relative, expected in bank['source'].items():
                assert sha(ROOT / relative) == expected, f'Source changed: {relative}'
            if 'execution_source' in bank:
                if 'execution_source' in config:
                    assert bank['execution_source'] == config['execution_source']
                for relative, expected in bank['execution_source'].items():
                    assert sha(DATA / 'cache/intermediate/source_snapshot' / relative) == expected
            banks[side] = bank
        q, r = banks['clinical'], banks['dermoscopic']
        path = DATA / 'cache/scores' / (model + '.pt')
        cached = torch.load(path, map_location='cpu', weights_only=True)
        assert sha(path) == config['score_sha256']
        assert cached['query_ids'] == q['ids'] and cached['gallery_ids'] == r['ids']
        assert cached['recipe'] == config['recipe']
        assert cached['recipe']['query_signature'] == q['signature']
        assert cached['recipe']['gallery_signature'] == r['signature']
        scores = cached['scores'].numpy()
        assert scores.shape == (479, 479) and np.isfinite(scores).all()
        ranks = []
        positive = {metadata[i]['lesion_id']: j for j, i in enumerate(r['ids'])}
        for i, query in enumerate(q['ids']):
            lesion = metadata[query]['lesion_id']
            correct = positive[lesion]
            # Count better scores; use gallery order to resolve ties independently of argsort.
            rank = 1 + int(np.sum(scores[i] > scores[i, correct]))
            rank += int(np.sum(scores[i, :correct] == scores[i, correct]))
            ranks.append(rank)
            p = predictions[i]
            assert p['query_id'] == query and p['positive_id'] == r['ids'][correct]
            assert p['lesion_id'] == lesion and p['rank'] == rank
            order = sorted(range(479), key=lambda j: (-float(scores[i, j]), j))[:10]
            assert p['top10_ids'] == [r['ids'][j] for j in order]
            assert p['top10_scores'] == [float(scores[i, j]) for j in order]
            assert p['top1_id'] == r['ids'][order[0]]
        ranks = np.array(ranks)
        assert len(predictions) == metrics['queries'] == metrics['gallery'] == 479
        for k in (1, 5, 10):
            assert abs(float((ranks <= k).mean()) - metrics[f'R@{k}']) < 1e-12
        assert abs(float(np.mean(1. / ranks)) - metrics['MRR']) < 1e-12
        assert int((ranks == 1).sum()) == metrics['correct_at_1']
        assert float(ranks.mean()) == metrics['mean_rank']
        assert float(np.median(ranks)) == metrics['median_rank']
        if isinstance(q['embeddings'], torch.Tensor):
            independent = q['embeddings'].numpy().astype(np.float64) @ r['embeddings'].numpy().astype(np.float64).T
            error = float(np.max(np.abs(independent - scores)))
            checked = 479 * 479
        else:
            errors = []
            for i, j in [(i, j) for i in (0, 73, 239, 478) for j in (i, (i + 1) % 479, (i + 137) % 479)]:
                a = q['embeddings'][i].double()
                b = r['embeddings'][j].double()
                a = a[a.norm(dim=-1) > 1e-8]
                b = b[b.norm(dim=-1) > 1e-8]
                independent = (a @ b.T).max(dim=1).values.mean().item()
                errors.append(abs(independent - float(scores[i, j])))
            error, checked = max(errors), len(errors)
        assert error < 2e-5, (model, error)
        audit['models'][model] = dict(passed=True, query_count=479, gallery_count=479,
            ranks_verified=479, score_pairs_recomputed_fp64=checked, max_score_error=error,
            embedding_dimensions=[list(q['embeddings'][0].shape), list(r['embeddings'][0].shape)],
            cache_score_sha256=config['score_sha256'])
        print(model, 'PASS', 'score error', error, flush=True)
    audit['passed'] = True
    audit['all_models_verified'] = set(models) == set(model_config['models'])
    name = 'verification.json' if audit['all_models_verified'] else 'verification.partial.json'
    (DATA / 'results' / name).write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
