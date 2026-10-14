"""Replay original BM25/fusion and audit every query, including rescue and harm."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import numpy as np
import torch

from scripts.colnomic_mechanism.common import ROOT, data, read, write, csvout, sha
from scripts.colnomic_mechanism.geometry import rank_rows, metrics
from evaluation.scoring import bm25_scores, tokenize, zscore, fuse


def bootstrap(before, after, d, split, seed=20261013):
    old = {r['photo_id']: r for r in before}
    subset = [r for r in after if split == 'all' or r['split'] == split]
    groups = sorted({d['groups'][r['package_id']] for r in subset})
    values = np.array([np.mean([int(r['correct']) - int(old[r['photo_id']]['correct']) for r in subset
                                if d['groups'][r['package_id']] == g]) for g in groups])
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(groups), size=(5000, len(groups)))].mean(1)
    return dict(source_groups=len(groups), source_group_macro_r1_difference=float(values.mean()),
                ci95=np.quantile(draws, [.025, .975]).tolist(), seed=seed, draws=5000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'verification.json').exists():
        raise FileExistsError('Choose a fresh directory; completed outputs are immutable')
    torch.set_num_threads(2)
    d = data()
    ledger = {r['photo_id']: r for r in read(ROOT / 'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    bm_recipe = d['bm25']['recipe']
    config = dict(alpha_grid=[k / 10 for k in range(11)], confidence_threshold_grid=[0., .9, .95, .98],
                  selection='validation identity macro R@1, then macro MRR, then proximity to original alpha=.5, then lower confidence threshold',
                  gate='query-only OCR character-weighted mean confidence; if below threshold use image z-score',
                  original_formula='population z-score over all 5405 rows, .5 image + .5 BM25; original empty-OCR image fallback',
                  historical_test_exposure=True, split_checks=d['split_verification'], bm25_recipe=bm_recipe)
    write(folder / 'protocol.json', config)
    print('Replaying complete-gallery BM25', flush=True)
    bm = bm25_scores([d['qo'][q]['text'] for q in d['qb']['ids']], [d['ro'][r]['text'] for r in d['rb']['ids']],
                     k1=bm_recipe['k1'], b=bm_recipe['b'])
    bm_error = float((bm - d['bm25']['scores']).abs().max())
    assert bm_error < 1e-5, ('BM25 replay', bm_error)
    has_text = [bool(d['qo'][q]['text'].strip()) for q in d['qb']['ids']]
    replay = fuse(d['image']['scores'], bm, has_text, .5)
    fusion_error = float((replay - d['fusion']['scores']).abs().max())
    assert fusion_error < 1e-5, ('Fusion replay', fusion_error)
    image = d['image']['scores'].numpy()
    text = bm.numpy()
    zi = zscore(d['image']['scores']).numpy()
    zt = zscore(bm).numpy()
    confidence = np.array([sum(len(l['text']) * l['confidence'] for l in d['qo'][q]['lines']) /
                           max(1, sum(len(l['text']) for l in d['qo'][q]['lines'])) for q in d['qb']['ids']])
    no_text = ~np.array(has_text)

    def scores(alpha, cutoff):
        weights = np.where((confidence >= cutoff) & ~no_text, alpha, 0.).astype(np.float32)[:, None]
        return (1 - weights) * zi + weights * zt

    choices = []
    grid = []
    for cutoff in config['confidence_threshold_grid']:
        for alpha in config['alpha_grid']:
            m = metrics(rank_rows(scores(alpha, cutoff), d, 'val'))
            grid.append(dict(alpha=alpha, confidence_threshold=cutoff, **m))
            choices.append((m['package_macro_r1'], m['package_macro_mrr'], -abs(alpha - .5), -cutoff, alpha, cutoff))
    constant = max(r for r in choices if r[-1] == 0.)
    gated = max(choices)
    selection = {name: dict(alpha=r[-2], confidence_threshold=r[-1]) for name, r in [('constant', constant), ('query_confidence_gate', gated)]}
    write(folder / 'selection.json', dict(selected=selection, validation_query_ids=[q for q in d['qb']['ids'] if d['splits'][q] == 'val'],
          criterion=config['selection'], used_test_labels=False))
    csvout(folder / 'validation_grid.csv', grid)
    variants = {'image': image, 'BM25': text, 'original_fusion': replay.numpy(),
                **{name: scores(**dict(alpha=cfg['alpha'], cutoff=cfg['confidence_threshold'])) for name, cfg in selection.items()}}
    predictions = {name: rank_rows(value, d) for name, value in variants.items()}
    np.savez_compressed(folder / 'full_gallery_scores.npz', **variants)
    write(folder / 'score_axes.json', dict(query_ids=d['qb']['ids'], reference_ids=d['rb']['ids'], reference_sha256=d['rb']['sha256'],
          dedup='stable descending score order, first reference per SHA; original row order resolves ties'))
    original_image = {r['photo_id']: r for r in predictions['image']}
    original_fusion = {r['photo_id']: r for r in predictions['original_fusion']}
    tables, changes, intervals = [], [], []
    objective = Counter()
    for q in d['qb']['ids']:
        a, b = original_image[q]['correct'], original_fusion[q]['correct']
        objective['both_correct' if a and b else 'fusion_rescue' if b else 'fusion_harm' if a else 'both_wrong'] += 1
        assert original_image[q]['rank'] == ledger[q]['image_positive_rank']
        assert original_fusion[q]['rank'] == ledger[q]['fusion_positive_rank']
    for method, pred in predictions.items():
        for split in ['train', 'val', 'test', 'all']:
            subset = [r for r in pred if split == 'all' or r['split'] == split]
            for baseline, old in [('image', original_image), ('original_fusion', original_fusion)]:
                rescued = sum(r['correct'] and not old[r['photo_id']]['correct'] for r in subset)
                harmed = sum(not r['correct'] and old[r['photo_id']]['correct'] for r in subset)
                originally_rescued = [r for r in subset if original_fusion[r['photo_id']]['correct'] and not original_image[r['photo_id']]['correct']]
                tables.append(dict(method=method, split=split, baseline=baseline, **metrics(subset), rescued=rescued, harmed=harmed, net=rescued-harmed,
                                   original_fusion_rescues=len(originally_rescued), original_fusion_rescues_lost=sum(not r['correct'] for r in originally_rescued)))
                intervals.append(dict(method=method, baseline=baseline, split=split, **bootstrap(predictions[baseline], pred, d, split)))
        for r in pred:
            q = r['photo_id']
            changes.append(dict(method=method, **r, image_rank=original_image[q]['rank'], fusion_rank=original_fusion[q]['rank'],
                                repaired_vs_image=r['correct'] and not original_image[q]['correct'], harmed_vs_image=not r['correct'] and original_image[q]['correct'],
                                repaired_vs_fusion=r['correct'] and not original_fusion[q]['correct'], harmed_vs_fusion=not r['correct'] and original_fusion[q]['correct']))
    csvout(folder / 'metrics.csv', tables)
    csvout(folder / 'per_query_rank_changes.csv', changes)
    write(folder / 'group_bootstrap.json', intervals)

    # Exact scoring components and OCR term evidence use fixed candidates, including originally successful queries.
    documents = [Counter(tokenize(d['ro'][rid]['text'])) for rid in d['rb']['ids']]
    lengths = np.array([sum(x.values()) for x in documents], dtype=float)
    avg = max(float(lengths.mean()), 1e-12)
    df = Counter(w for doc in documents for w in doc)
    rindex = {rid: i for i, rid in enumerate(d['rb']['ids'])}
    decomposition, terms = [], []
    image_std = d['image']['scores'].std(-1, unbiased=False).numpy()
    text_std = bm.std(-1, unbiased=False).numpy()
    for qi, q in enumerate(d['qb']['ids']):
        row = ledger[q]
        positive = rindex[row['positive_reference_ids'][0]]
        for negative_id in dict.fromkeys([row['image_strongest_wrong'], row['fusion_strongest_wrong']]):
            negative = rindex[negative_id]
            image_term = .5 * float(zi[qi, positive] - zi[qi, negative]) if has_text[qi] else float(zi[qi, positive] - zi[qi, negative])
            text_term = .5 * float(zt[qi, positive] - zt[qi, negative]) if has_text[qi] else 0.
            gap = float(replay[qi, positive] - replay[qi, negative])
            error = abs(image_term + text_term - gap)
            assert error < 1e-5
            decomposition.append(dict(photo_id=q, package_id=row['package_id'], group_id=row['group_id'], split=row['split'],
                positive=row['positive_reference_ids'][0], negative=negative_id,
                objective_outcome='both_correct' if original_image[q]['correct'] and original_fusion[q]['correct'] else 'fusion_rescue' if original_fusion[q]['correct'] else 'fusion_harm' if original_image[q]['correct'] else 'both_wrong',
                image_positive=float(image[qi,positive]), image_negative=float(image[qi,negative]), image_raw_gap=float(image[qi,positive]-image[qi,negative]),
                BM25_positive=float(text[qi,positive]), BM25_negative=float(text[qi,negative]), BM25_raw_gap=float(text[qi,positive]-text[qi,negative]),
                image_population_std=float(image_std[qi]), BM25_population_std=float(text_std[qi]),
                normalized_image_contribution=image_term, normalized_text_contribution=text_term, fusion_gap=gap, conservation_error=error,
                query_OCR_weighted_confidence=float(confidence[qi]), OCR_terms_not_visual_review=True))
            for word, qf in Counter(tokenize(d['qo'][q]['text'])).items():
                idf = math.log(1 + (len(documents)-df[word]+.5)/(df[word]+.5))
                contributions = []
                for index in [positive, negative]:
                    tf = documents[index][word]
                    contributions.append(qf*idf*tf*(bm_recipe['k1']+1)/(tf+bm_recipe['k1']*(1-bm_recipe['b']+bm_recipe['b']*lengths[index]/avg)))
                if contributions[0] or contributions[1]:
                    terms.append(dict(photo_id=q, positive=row['positive_reference_ids'][0], negative=negative_id, query_OCR_term=word,
                        query_frequency=qf, gallery_document_frequency=df[word], BM25_positive=contributions[0], BM25_negative=contributions[1],
                        BM25_delta=contributions[0]-contributions[1], contains_digit=any(c.isdigit() for c in word),
                        common_to_both_candidates=bool(documents[positive][word] and documents[negative][word]),
                        evidence_scope='OCR transcription term contribution; not a visual determination of true text or OCR correctness'))
    csvout(folder / 'all_query_decomposition.csv', decomposition)
    csvout(folder / 'OCR_term_contributions.csv', terms)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/'cache/scores/bm25/bm25.pt', ROOT/'cache/scores/bm25_image_fusion/colnomic7b/alpha_0.5.pt', ROOT/'cache/scores/colnomic7b/image.pt']}
    verification = dict(status='passed', query_count=len(original_image), reference_rows=image.shape[1], unique_reference_SHA=len(set(d['rb']['sha256'])),
        bm25_replay_max_error=bm_error, fusion_replay_max_error=fusion_error, all_baseline_ranks_match_ledger=True,
        decomposition_count=len(decomposition), decomposition_query_coverage=len({r['photo_id'] for r in decomposition}),
        max_decomposition_conservation_error=max(r['conservation_error'] for r in decomposition), actual_joint_outcomes=dict(objective),
        split_verification=d['split_verification'], input_SHA256=hashes, selected=selection,
        test_historically_observed=True, explanations_not_inferred_from_rank_outcomes=True)
    write(folder / 'verification.json', verification)
    print('Fusion complete', dict(objective), selection, 'BM25 error', bm_error, 'fusion error', fusion_error, flush=True)


if __name__ == '__main__':
    main()
