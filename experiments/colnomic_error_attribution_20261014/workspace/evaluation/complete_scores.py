"""Fill baseline score caches while keeping published experiment results intact."""
import argparse
from pathlib import Path
import time

import torch

from .cache import Cache, save_tensor
from .data import Dataset, digest, file_hash, read_json, write_json
from .dense import DENSE_DIMENSIONS
from .scoring import (SCORING_VERSION, CRISP_SOURCE, bm25_scores, dot_scores,
                      fuse, multivector_scores, tokenize)


def reusable_rows(old, query, previous, recipe, reference_ids, old_hash):
    """Trust old scores only for byte-identical vectors and the same scorer/gallery."""
    old_recipe = previous.get('recipe', {})
    if (previous.get('signature') != digest(old_recipe) or old_recipe.get('query') != old_hash
            or {k: v for k, v in old_recipe.items() if k != 'query'} !=
               {k: v for k, v in recipe.items() if k != 'query'}
            or previous.get('reference_ids') != reference_ids
            or previous.get('query_ids') != old.get('ids')
            or any(old.get(k) != query.get(k) for k in ('model', 'view', 'contract'))):
        return []
    scores = previous.get('scores')
    if (not isinstance(scores, torch.Tensor) or scores.shape != (len(old['ids']), len(reference_ids))
            or not torch.isfinite(scores).all() or len(set(old['ids'])) != len(old['ids'])):
        return []
    indices = {identifier: i for i, identifier in enumerate(old['ids'])}
    fields = ['sha256', 'source_sha256']
    if query['view'] == 'text':
        fields += ['text_sha256', 'has_text', 'input_token_counts']
    return [(position, indices[identifier]) for position, identifier in enumerate(query['ids'])
            if identifier in indices
            and all(old[field][indices[identifier]] == query[field][position] for field in fields)
            and torch.equal(old['embeddings'][indices[identifier]], query['embeddings'][position])]


def neural(cache, model, view, reference_view, args):
    query = cache.embeddings(model, view, 'photos')
    gallery = cache.embeddings(model, reference_view, 'references')
    recipe = dict(version=SCORING_VERSION, model=model, view=view, query=query['_file_sha256'],
                  reference=gallery['_file_sha256'], crisp_source=CRISP_SOURCE,
                  scorer_sha256=file_hash(Path(__file__).with_name('scoring.py')),
                  valid_tokens='remove padding and zero-norm vectors; preserve attended template tokens',
                  maxsim='mean over valid query tokens', dtype='float32', tf32=False)
    if reference_view == 'text':
        recipe.update(reference_view='text', empty_query='count as retrieval failure',
                      empty_reference='rank after nonempty OCR')
    if model in ('dinov3', 'siglip2'):
        recipe.update(reference_view='image', pooling=cache.config['models'][model]['image_contract']['pooling'])
    mode = 'image' if view == 'image' else ('ocr_to_ocr' if reference_view == 'text' else 'ocr_to_image')
    names = [mode] + ([mode + '_crisp'] if model not in DENSE_DIMENSIONS else [])
    paths = [cache.root / 'cache/scores' / model / (name + '.pt') for name in names]
    if all(path.exists() for path in paths):
        warm = [torch.load(path, weights_only=True, map_location='cpu') for path in paths]
        if all(value.get('signature') == digest(recipe)
               and value.get('query_ids') == query['ids'] and value.get('reference_ids') == gallery['ids']
               and value['scores'].shape == (len(query['ids']), len(gallery['ids']))
               and torch.isfinite(value['scores']).all() for value in warm):
            print(f'Ready scores {model}/{mode}: {len(query["ids"])}', flush=True)
            return dict(zip(names, warm, strict=True)), query
    matrices = [torch.zeros(len(query['ids']), len(gallery['ids'])) for _ in names]
    reused = []
    if args.before:
        folder = 'image_embeddings' if view == 'image' else 'ocr_embeddings'
        old_path = args.before / 'cache' / folder / model / 'photos.pt'
        score_paths = [args.before / 'cache/scores' / model / (name + '.pt') for name in names]
        if old_path.exists() and all(path.exists() for path in score_paths):
            old = torch.load(old_path, weights_only=True, mmap=True, map_location='cpu')
            old_hash = file_hash(old_path)
            scores = [torch.load(path, weights_only=True, map_location='cpu') for path in score_paths]
            candidates = [reusable_rows(old, query, score, recipe, gallery['ids'], old_hash) for score in scores]
            if all(candidate == candidates[0] for candidate in candidates):
                reused = candidates[0]
                if reused:
                    current, prior = zip(*reused)
                    for matrix, score in zip(matrices, scores, strict=True):
                        matrix[list(current)] = score['scores'][list(prior)]
            del old, scores
    reused_positions = {current for current, _ in reused}
    pending = [i for i in range(len(query['ids'])) if i not in reused_positions]
    print(f'Scoring {model}/{mode}: {len(pending)} new, {len(reused)} reused', flush=True)
    started = time.monotonic()
    if pending:
        if model in DENSE_DIMENSIONS:
            pair = [dot_scores(query['embeddings'][pending], gallery['embeddings'])]
            if reference_view == 'text':
                pair[0][:, ~torch.as_tensor(gallery['has_text'], dtype=torch.bool)] = -1.000001
        else:
            pair = multivector_scores([query['embeddings'][i] for i in pending], gallery['embeddings'],
                device=args.device, query_batch=args.query_batch, reference_batch=args.reference_batch,
                token_block=args.token_block,
                progress=lambda done, total, seconds: print(f'{model}/{mode}: {done}/{total} ({seconds:.1f}s)', flush=True))
        for matrix, part in zip(matrices, pair, strict=True):
            matrix[pending] = part
    seconds = time.monotonic() - started
    results = {}
    for name, matrix in zip(names, matrices, strict=True):
        if not torch.isfinite(matrix).all():
            raise ValueError(f'Non-finite scores: {model}/{name}')
        value = dict(signature=digest(recipe), recipe=recipe, scores=matrix,
                     query_ids=query['ids'], reference_ids=gallery['ids'], compute_seconds=seconds,
                     timing_scope='incremental scoring; excludes reused historical rows', cache_hit=not pending,
                     device='cpu' if model in DENSE_DIMENSIONS else args.device,
                     blocks=dict(query=args.query_batch, reference=args.reference_batch, token=args.token_block),
                     incremental=dict(reused_rows=len(reused), generated_rows=len(pending),
                                      before=str(args.before) if args.before else None))
        save_tensor(cache.root / 'cache/scores' / model / (name + '.pt'), value)
        results[name] = value
    return results, query


def run(args):
    torch.set_num_threads(2)
    dataset = Dataset(args.root)
    settings = read_json(dataset.root / 'configs/baselines.json')
    models = list(settings['models']) + ['bm25'] if args.model == 'all' else [args.model]
    if any(model not in settings['models'] and model != 'bm25' for model in models):
        raise ValueError(f'Unknown models: {models}')
    report = {}
    for model in models:
        cache = Cache(dataset, settings, args.device, cache_only=True)
        if model == 'bm25':
            query = cache.ocr('photos'); gallery = cache.ocr('references')
            recipe = dict(version=SCORING_VERSION, k1=1.2, b=.75,
                tokenization='Unicode letters/digits; lowercase; query frequency',
                scorer_sha256=file_hash(Path(__file__).with_name('scoring.py')),
                query_order=digest([r['photo_id'] for r in dataset.photos]),
                reference_order=digest([r['reference_id'] for r in dataset.references]),
                query_ocr=file_hash(dataset.root / 'cache/ocr/photos.json'),
                reference_ocr=file_hash(dataset.root / 'cache/ocr/references.json'))
            started = time.monotonic()
            scores = bm25_scores([query[r['photo_id']]['text'] for r in dataset.photos],
                                 [gallery[r['reference_id']]['text'] for r in dataset.references])
            save_tensor(dataset.root / 'cache/scores/bm25/bm25.pt',
                        dict(signature=digest(recipe), recipe=recipe, scores=scores,
                             query_ids=[r['photo_id'] for r in dataset.photos],
                             reference_ids=[r['reference_id'] for r in dataset.references],
                             compute_seconds=time.monotonic()-started))
            report[model] = {'bm25': dict(generated_rows=len(dataset.photos))}
            continue
        values, _ = neural(cache, model, 'image', 'image', args)
        if 'text_model' in settings['models'][model]:
            text_values, text_query = neural(cache, model, 'text', 'image', args)
            values.update(text_values)
            if model == 'Qwen3-VL-Embedding-2B':
                text_values, _ = neural(cache, model, 'text', 'text', args)
                values.update(text_values)
            for mode, source in [('fusion', 'ocr_to_image')] + (
                    [('ocr_fusion', 'ocr_to_ocr')] if model == 'Qwen3-VL-Embedding-2B' else []):
                recipe = dict(image=values['image']['signature'], text=values[source]['signature'],
                              alpha=.5, normalization='per-query population z-score', empty_text='image fallback')
                started = time.monotonic()
                scores = fuse(values['image']['scores'], values[source]['scores'], text_query['has_text'], .5)
                value = dict(signature=digest(recipe), recipe=recipe, scores=scores,
                             query_ids=text_query['ids'], reference_ids=[r['reference_id'] for r in dataset.references],
                             compute_seconds=time.monotonic()-started, timing_scope='cached base-score fusion',
                             cache_hit=False, device='cpu')
                save_tensor(dataset.root / 'cache/scores' / model / (mode + '_alpha_0.5.pt'), value)
        report[model] = {name: value.get('incremental', {'cache_hit': True}) for name, value in values.items()}
    if args.model == 'all':
        bm25 = torch.load(dataset.root / 'cache/scores/bm25/bm25.pt', weights_only=True, map_location='cpu')
        ocr = read_json(dataset.root / 'cache/ocr/photos.json')
        has_text = [bool(tokenize(ocr[row['photo_id']]['text'])) for row in dataset.photos]
        for model in ('colpali', 'colnomic3b', 'colnomic7b', 'Qwen3-VL-Embedding-2B'):
            path = dataset.root / 'cache/scores' / model / 'image.pt'
            image = torch.load(path, weights_only=True, map_location='cpu')
            recipe = dict(method='bm25_image_late_fusion_v1', image=image['signature'], bm25=bm25['signature'],
                          image_score_sha256=file_hash(path),
                          bm25_score_sha256=file_hash(dataset.root / 'cache/scores/bm25/bm25.pt'),
                          bm25_weight=.5, image_weight=.5, normalization='per-query population z-score',
                          empty_ocr='image fallback', query_has_text_sha256=digest(has_text),
                          scorer_sha256=file_hash(Path(__file__).with_name('scoring.py')))
            save_tensor(dataset.root / 'cache/scores/bm25_image_fusion' / model / 'alpha_0.5.pt',
                        dict(signature=digest(recipe), recipe=recipe,
                             scores=fuse(image['scores'], bm25['scores'], has_text, .5),
                             query_ids=[r['photo_id'] for r in dataset.photos],
                             reference_ids=[r['reference_id'] for r in dataset.references],
                             timing_scope='cached base-score fusion'))
    if args.report:
        write_json(args.report, dict(status='complete', photos=len(dataset.photos),
                                    references=len(dataset.references), models=report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model', default='all')
    parser.add_argument('--before', type=Path, help='Snapshot root containing the old cache/ directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--query-batch', type=int, default=4)
    parser.add_argument('--reference-batch', type=int, default=16)
    parser.add_argument('--token-block', type=int, default=512)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if min(args.query_batch, args.reference_batch, args.token_block) < 1:
        parser.error('Scoring batch sizes must be positive')
    run(args)


if __name__ == '__main__':
    main()
