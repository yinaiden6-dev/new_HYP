"""Incrementally fill OCR and embedding banks without publishing evaluation results."""
import argparse
from pathlib import Path

import torch

from .cache import Cache
from .data import Dataset, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model', default='all')
    parser.add_argument('--view', choices=('all', 'image', 'text', 'ocr'), default='all')
    parser.add_argument('--population', choices=('all', 'photos', 'references'), default='all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cache-only', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    dataset = Dataset(args.root)
    settings = read_json(dataset.root / 'configs/baselines.json')
    if args.model != 'all' and args.model not in settings['models']:
        parser.error(f'Unknown model: {args.model}')
    populations = ('photos', 'references') if args.population == 'all' else (args.population,)
    models = list(settings['models']) if args.model == 'all' else [args.model]
    reports = {}
    for model in [None] if args.view == 'ocr' else models:
        cache = Cache(dataset, settings, args.device, cache_only=args.cache_only)
        try:
            if args.view == 'ocr':
                for population in populations:
                    cache.ocr(population)
            else:
                views = ('image', 'text') if args.view == 'all' else (args.view,)
                for view in views:
                    if view == 'text' and 'text_model' not in settings['models'][model]:
                        continue
                    for population in populations:
                        bank = cache.embeddings(model, view, population)
                        print(f'Ready {model}/{view}/{population}: {len(bank["ids"])}', flush=True)
            reports[model or 'ocr'] = cache.cost
        finally:
            cache.release()
    if args.report:
        write_json(args.report, dict(status='complete', photos=len(dataset.photos),
                                    references=len(dataset.references), models=reports))


if __name__ == '__main__':
    main()
