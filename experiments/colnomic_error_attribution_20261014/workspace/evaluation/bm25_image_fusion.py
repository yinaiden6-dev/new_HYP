"""Fixed equal-weight BM25 + image retrieval, using validated full-cohort scores."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from .cache import save_tensor
from .data import Dataset, digest, file_hash, read_json, write_json
from .metrics import evaluate, paired_report
from .scoring import fuse, tokenize


MODELS = ('colpali', 'colnomic3b', 'colnomic7b', 'Qwen3-VL-Embedding-2B', 'colqwen2.5-v0.2')
LABELS = ('ColPali', 'ColNomic 3B', 'ColNomic 7B', 'Qwen3-VL-Embedding-2B', 'ColQwen2.5 v0.2')
ALPHA = 0.5


def load_source(dataset, model, mode):
    root = dataset.root
    folder = root / 'results' / model / mode
    config = read_json(folder / 'config.json')
    if (not dataset.matches_full_inventory(config['dataset_sha256']) or
            config['annotation_sha256'] != dataset.annotation_sha256):
        raise ValueError(f'Stale source result: {folder}')
    path = root / 'cache/scores' / model / (mode + '.pt')
    value = torch.load(path, map_location='cpu', weights_only=True)
    signature = digest(config['scoring']) if mode == 'bm25' else config['score_signature']
    if value['signature'] != signature:
        raise ValueError(f'Source signature mismatch: {path}')
    shape = (len(dataset.photos), len(dataset.references))
    if tuple(value['scores'].shape) != shape or not torch.isfinite(value['scores']).all():
        raise ValueError(f'Invalid scores: {path}')
    query_ids = [r['photo_id'] for r in dataset.photos]
    reference_ids = [r['reference_id'] for r in dataset.references]
    recipe = config['scoring']
    if recipe['scorer_sha256'] != file_hash(Path(__file__).with_name('scoring.py')):
        raise ValueError('The source scoring implementation has changed')
    if mode == 'bm25':
        expected = dict(query_order=digest(query_ids), reference_order=digest(reference_ids),
                        query_ocr=file_hash(root / 'cache/ocr/photos.json'),
                        reference_ocr=file_hash(root / 'cache/ocr/references.json'))
        if any(recipe[k] != v for k, v in expected.items()):
            raise ValueError('BM25 OCR contents or row order have changed')
    else:
        if value['query_ids'] != query_ids or value['reference_ids'] != reference_ids:
            raise ValueError('Image score order does not match the formal inventories')
        metadata = read_json(root / 'cache/image_embeddings/metadata.json')['models'][model]['files']
        if (value['recipe'] != recipe or digest(recipe) != signature or
                recipe['query'] != metadata['photos']['sha256'] or
                recipe['reference'] != metadata['references']['sha256']):
            raise ValueError('Image score provenance does not match current embedding metadata')
    predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
    if [r['photo_id'] for r in predictions] != query_ids:
        raise ValueError('Source predictions do not match formal queries')
    provenance = dict(score_path=str(path.relative_to(root)), score_sha256=file_hash(path),
                      signature=signature, config_path=str((folder / 'config.json').relative_to(root)),
                      config_sha256=file_hash(folder / 'config.json'),
                      predictions_sha256=file_hash(folder / 'predictions.jsonl'))
    return value, predictions, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    torch.set_num_threads(2)
    dataset = Dataset(args.root)
    if dataset.active_difference is not None:
        raise ValueError('Resolve the active/formal inventory difference before running')
    bm25, bm_predictions, bm_source = load_source(dataset, 'bm25', 'bm25')
    ocr = read_json(dataset.root / 'cache/ocr/photos.json')
    has_text = [bool(tokenize(ocr[r['photo_id']]['text'])) for r in dataset.photos]
    output_root = dataset.root / 'results/bm25_image_fusion'
    rows = []
    for index, model in enumerate(MODELS):
        image, image_predictions, image_source = load_source(dataset, model, 'image')
        recipe = dict(method='bm25_image_late_fusion_v1', image=image['signature'],
                      bm25=bm25['signature'], image_score_sha256=image_source['score_sha256'],
                      bm25_score_sha256=bm_source['score_sha256'], bm25_weight=ALPHA,
                      image_weight=1-ALPHA, normalization='per-query population z-score',
                      empty_ocr='image fallback', query_has_text_sha256=digest(has_text),
                      scorer_sha256=file_hash(Path(__file__).with_name('scoring.py')))
        start = time.perf_counter()
        scores = fuse(image['scores'], bm25['scores'], has_text, ALPHA)
        fusion_seconds = time.perf_counter() - start
        cache_path = dataset.root / 'cache/scores/bm25_image_fusion' / model / 'alpha_0.5.pt'
        save_tensor(cache_path, dict(scores=scores, signature=digest(recipe), recipe=recipe,
                                    query_ids=[r['photo_id'] for r in dataset.photos],
                                    reference_ids=[r['reference_id'] for r in dataset.references],
                                    compute_seconds=fusion_seconds,
                                    timing_scope='fusion of two cached score matrices only'))
        start = time.perf_counter()
        predictions, metrics, by_type = evaluate(scores, dataset)
        evaluation_seconds = time.perf_counter() - start
        comparisons = dict(versus_bm25=paired_report(predictions, bm_predictions),
                           versus_image=paired_report(predictions, image_predictions))
        identifier = f'F{index+1:02d}'
        folder = output_root / model / 'alpha_0.5'
        config = dict(id=identifier, protocol=dataset.full_protocol, model=model, mode='bm25_image_fusion',
                      created_utc=datetime.now(timezone.utc).isoformat(),
                      queries=len(dataset.photos), references=len(dataset.references),
                      dataset_sha256=dataset.hashes, annotation_sha256=dataset.annotation_sha256,
                      active_difference=dataset.active_difference, scoring=recipe,
                      score_signature=digest(recipe), score_path=str(cache_path.relative_to(dataset.root)),
                      sources=dict(image=image_source, bm25=bm_source),
                      labels_used_only_for_evaluation=True, weight_selection='fixed 0.5 before this run; no sweep',
                      empty_ocr_fallbacks=sum(not value for value in has_text),
                      model_initializations=0, generated_embeddings=0, generated_ocr=0,
                      fusion_seconds=fusion_seconds, evaluation_seconds=evaluation_seconds,
                      timing_note='Cached score fusion only; excludes OCR, encoding, base scoring and initialization',
                      runner_sha256=file_hash(Path(__file__)))
        for name, value in [('config', config), ('metrics', metrics), ('by_type', by_type),
                            ('paired_comparisons', comparisons)]:
            write_json(folder / (name + '.json'), value)
        temporary = folder / 'predictions.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in predictions))
        temporary.replace(folder / 'predictions.jsonl')
        bm_r1 = sum(p['correct'] for p in bm_predictions) / len(bm_predictions)
        im_r1 = sum(p['correct'] for p in image_predictions) / len(image_predictions)
        row = dict(id=identifier, model=model, bm25_weight=ALPHA, queries=len(predictions),
                   references=len(dataset.references), image_R1=im_r1, bm25_R1=bm_r1)
        row.update(metrics['photo_average'])
        row.update(delta_R1_vs_bm25=row['recall_at_1']-bm_r1,
                   delta_R1_vs_image=row['recall_at_1']-im_r1,
                   rescued_vs_bm25=comparisons['versus_bm25']['rescued'],
                   harmed_vs_bm25=comparisons['versus_bm25']['harmed'],
                   rescued_vs_image=comparisons['versus_image']['rescued'],
                   harmed_vs_image=comparisons['versus_image']['harmed'],
                   fusion_seconds=fusion_seconds)
        ci = comparisons['versus_bm25']['intervals']['photo_average']['recall_at_1']
        row.update(delta_R1_vs_bm25_ci95_low=ci['low'], delta_R1_vs_bm25_ci95_high=ci['high'])
        rows.append(row)
        print(f"{identifier} {model}: R@1={row['recall_at_1']:.1%}; "
              f"vs BM25 rescued={row['rescued_vs_bm25']}, harmed={row['harmed_vs_bm25']}", flush=True)
    temporary = output_root / 'summary.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_root / 'summary.csv')
    lines = ['# BM25 + image retrieval: fixed equal-weight fusion', '',
             f'{dataset.full_protocol}: {len(dataset.photos):,} photos × {len(dataset.references):,} reference rows. S = 0.5 Z(image) + 0.5 Z(BM25).',
             'No training or weight search. All rows use PP-OCRv6 BM25 scores and the existing image-only scores.', '',
             '| ID | Image model | Image R@1 | BM25 R@1 | Fusion R@1 | R@5 | R@10 | MRR | Rescued vs BM25 | Harmed vs BM25 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row, label in zip(rows, LABELS, strict=True):
        lines.append(f"| {row['id']} | {label} | {row['image_R1']:.2%} | {row['bm25_R1']:.2%} | "
                     f"{row['recall_at_1']:.2%} | {row['recall_at_5']:.2%} | {row['recall_at_10']:.2%} | "
                     f"{row['mrr']:.4f} | {row['rescued_vs_bm25']} | {row['harmed_vs_bm25']} |")
    lines += ['', f'Paired R@1 changes relative to BM25 ({len(set(dataset.groups.values()))} source groups, 10,000 bootstrap draws, seed 20260924):', '']
    for row in rows:
        lines.append(f"- {row['id']}: {100*row['delta_R1_vs_bm25']:+.2f} percentage points; "
                     f"95% CI [{100*row['delta_R1_vs_bm25_ci95_low']:+.2f}, {100*row['delta_R1_vs_bm25_ci95_high']:+.2f}].")
    lines += ['', 'Candidate ties use formal row order; exact SHA-256 duplicates collapse after scoring.',
              'Missing OCR tokens fall back to image retrieval. Both source-score signatures and inventories are checked.',
              'Each model directory includes config, metrics, predictions, type breakdowns and paired comparisons.',
              'Reported fusion time covers only combining cached score matrices, not end-to-end retrieval.',
              'The full cohort contains historical development data and is not an independent held-out test.', '',
              'Reproduce: `python -B -m evaluation.bm25_image_fusion`', '']
    (output_root / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
