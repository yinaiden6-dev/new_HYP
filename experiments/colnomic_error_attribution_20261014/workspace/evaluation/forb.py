"""Five frozen baselines on the complete official FORB Subset."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from .data import digest, file_hash, read_json, write_json
from .forb_data import PROTOCOL, metrics
from .forb_encoding import encode_worker, merge_banks, encoding_progress
from .forb_scoring import (ShardedEmbeddingBank, BM25Index, dense_scores, maxsim_scores,
                           run_score_worker, merge_score_chunks)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'other_datasets/FORB'
MODELS = ('dinov3', 'siglip2', 'Qwen3-VL-Embedding-2B', 'colnomic7b', 'bm25')


def score_context(data, model):
    config = read_json(data / 'baseline_config.json')
    execution_path = data / 'maxsim_execution.json'
    if model == 'colnomic7b' and execution_path.exists():
        execution = read_json(execution_path)
        for key in ('query_batch', 'reference_batch', 'token_block'):
            if not isinstance(execution[key], int) or execution[key] <= 0:
                raise ValueError(f'Invalid MaxSim execution block: {key}')
            config[key] = execution[key]
        if execution.get('backend'):
            from .forb_maxsim_fast import BACKEND
            if execution['backend'] != BACKEND:
                raise ValueError(f"Unsupported MaxSim backend: {execution['backend']}")
            config['maxsim_backend'] = BACKEND
    manifest = read_json(data / 'cache/manifest.json')
    query_ids = [r['id'] for r in manifest['photos']]
    reference_ids = [r['id'] for r in manifest['references']]
    recipe = dict(protocol=PROTOCOL, model=model, dataset=file_hash(data / 'cache/manifest.json'))
    if model == 'bm25':
        from .forb_ocr import verify_cache
        verify_cache(data)
        queries = read_json(data / 'cache/ocr/photo.json')
        references = read_json(data / 'cache/ocr/ref.json')
        recipe.update(bm25=config['bm25'], query_ocr=file_hash(data / 'cache/ocr/photo.json'),
                      reference_ocr=file_hash(data / 'cache/ocr/ref.json'))
        query_texts = [queries['entries'][key]['text'] for key in query_ids]
        reference_texts = [references['entries'][key]['text'] for key in reference_ids]
        return manifest, config, recipe, query_ids, reference_ids, query_texts, reference_texts
    base = data / 'cache/image_embeddings' / model
    query = ShardedEmbeddingBank(base / 'photo/index.json')
    reference = ShardedEmbeddingBank(base / 'ref/index.json')
    if query.ids != query_ids or reference.ids != reference_ids:
        raise ValueError('Embedding order differs from official manifest')
    recipe.update(model_spec=config['models'][model], query=file_hash(base / 'photo/index.json'),
                  reference=file_hash(base / 'ref/index.json'),
                  arithmetic='FP32 mean over valid query-token maxima' if model == 'colnomic7b' else 'FP32 normalized dot product')
    if model == 'colnomic7b':
        recipe['score_blocks'] = {key: config[key] for key in ('query_batch', 'reference_batch', 'token_block')}
        if config.get('maxsim_backend'):
            recipe['maxsim_backend'] = dict(name=config['maxsim_backend'],
                source_sha256=file_hash(ROOT / 'evaluation/forb_maxsim_fast.py'))
    return manifest, config, recipe, query_ids, reference_ids, query, reference


def score_worker(data, model, device, max_tasks=None):
    manifest, config, recipe, query_ids, reference_ids, query, reference = score_context(data, model)
    if model == 'bm25':
        index = BM25Index(reference, **config['bm25'])
        calculate = lambda first, stop: index.score(query[first:stop])
    elif model == 'colnomic7b':
        if config.get('maxsim_backend'):
            from .forb_maxsim_fast import MaxSimEngine
            engine = MaxSimEngine(reference, device=device, query_batch=config['query_batch'],
                reference_batch=config['reference_batch'], token_block=config['token_block'])
            print(f'colnomic7b: gallery cache ready; GPU bytes={engine.gpu_cache_bytes}, '
                  f'CPU bytes={engine.cpu_cache_bytes}', flush=True)
            calculate = lambda first, stop: engine.score(query[first:stop])
        else:
            calculate = lambda first, stop: maxsim_scores(query[first:stop], reference, device=device,
                query_batch=config['query_batch'], reference_batch=config['reference_batch'], token_block=config['token_block'])
    else:
        query = torch.stack(list(query)).to(device)
        reference = torch.stack(list(reference)).to(device)
        calculate = lambda first, stop: dense_scores(query[first:stop], reference, device=device)
    directory = data / 'cache/scores' / model
    return run_score_worker(directory, query_ids, reference_ids, recipe, calculate,
                            chunk_size=config['score_chunk_size'], max_chunks=max_tasks,
                            progress=lambda *args: print(f'{model}: score progress {args}', flush=True))


def finalize(data, model):
    lock_path = data / 'cache/intermediate' / f'finalize_{model}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest, config, recipe, query_ids, reference_ids, query, reference = score_context(data, model)
        directory = data / 'cache/scores' / model
        score_meta = merge_score_chunks(directory, query_ids, reference_ids, recipe,
                                        chunk_size=config['score_chunk_size'])
        matrix = np.load(directory / 'scores.npy', mmap_mode='r')
        valid_query = valid_reference = None
        if model == 'bm25':
            from .scoring import tokenize
            valid_query = [bool(tokenize(t)) for t in query]
            valid_reference = [bool(tokenize(t)) for t in reference]
        report, predictions = metrics(matrix, manifest, valid_query, valid_reference)
        mode = 'bm25' if model == 'bm25' else 'image'
        result = data / 'results' / model / mode
        write_json(result / 'metrics.json', report)
        write_json(result / 'config.json', dict(protocol=PROTOCOL, model=model, mode=mode,
            specification=config['bm25'] if model == 'bm25' else config['models'][model],
            dataset_sha256=file_hash(data / 'cache/manifest.json'), scoring=recipe,
            score_metadata=score_meta, score_path=str((directory / 'scores.npy').relative_to(data)),
            host=socket.gethostname(), metrics_source=file_hash(ROOT / 'evaluation/forb_data.py'),
            labels_used_only_for_evaluation=True, training='none',
            empty_ocr=config['empty_ocr'], tie_break=config['tie_break']))
        temporary = result / 'predictions.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in predictions))
        temporary.replace(result / 'predictions.jsonl')
        print(f'{model}: {report}', flush=True)
    summarize(data)


def summarize(data):
    result = data / 'results'
    result.mkdir(parents=True, exist_ok=True)
    with (result / '.summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for model in MODELS:
            mode = 'bm25' if model == 'bm25' else 'image'
            path = result / model / mode / 'metrics.json'
            if not path.exists():
                continue
            report = read_json(path)
            values = report.get('query_average', report)
            row = dict(model=model, mode=mode)
            for name in ['queries', 'references', 'R@1', 'R@5', 'R@10', 'MRR', 'mAP@1', 'mAP@5', 'mAP@100']:
                row[name] = values.get(name, report.get(name))
            rows.append(row)
        if rows:
            temp = result / 'summary.csv.tmp'
            with temp.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            temp.replace(result / 'summary.csv')
        write_json(result / 'status.json', dict(protocol=PROTOCOL, completed=[r['model'] for r in rows],
            pending=[m for m in MODELS if m not in [r['model'] for r in rows]], updated_unix=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--worker')
    parser.add_argument('--max-tasks', type=int)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ['encode', 'merge-embeddings', 'score', 'finalize', 'summarize', 'status']:
        actions.add_argument('--' + name, action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not args.summarize and not args.status and not args.model:
        parser.error('--model is required for this action')
    if args.encode:
        encode_worker(args.data, args.model, args.device, args.worker, args.max_tasks)
    elif args.merge_embeddings:
        merge_banks(args.data, args.model)
    elif args.score:
        score_worker(args.data, args.model, args.device, args.max_tasks)
    elif args.finalize:
        finalize(args.data, args.model)
    elif args.summarize:
        summarize(args.data)
    else:
        print(json.dumps({model: encoding_progress(args.data, model) for model in MODELS if model != 'bm25'}, indent=2))


if __name__ == '__main__':
    main()
