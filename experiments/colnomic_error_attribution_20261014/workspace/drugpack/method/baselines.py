import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, WORK, RESULTS, evaluation_group, read, save, sha
from .ranking import baseline_ranks, deduplicated_rank, summarize

FOLDERS = dict(medicine=ROOT/'results',grozi=ROOT/'other_datasets/GroZi-3.2k/results',
               milk=ROOT/'other_datasets/MILK10K/results',forb32=ROOT/'other_datasets/FORB/results')


def imported(dataset, split):
    manifest = read(WORK/dataset/'manifest.json')
    rows = [r for r in manifest['rows'] if r['split']==split]
    raw = baseline_ranks(manifest)
    labels = read(WORK/dataset/'labels.json')
    results = {}
    for path in sorted(FOLDERS[dataset].rglob('predictions.jsonl')):
        relative = path.parent.relative_to(FOLDERS[dataset]).as_posix()
        kind = 'existing_baseline'
        if relative.startswith('diagnostics/'):
            continue
        if relative.startswith('foreground/'):
            route = relative.split('/')[1]
            if route not in ('sam3_auto','human_boxes'):
                continue
            latest = read(FOLDERS[dataset]/'foreground'/route/'latest.json')
            if path.parent != ROOT/latest['result']:
                continue
            relative = route + '/colnomic7b'
            kind = 'mixed_region_diagnostic' if route == 'human_boxes' else 'automatic_foreground_baseline'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        by_id = {r.get('photo_id',r.get('query_id')):r for r in records}
        if set(by_id) != {r['original_id'] for r in manifest['rows']}:
            continue
        config_path = path.parent/'config.json'
        config = read(config_path)
        region = {}
        if kind == 'mixed_region_diagnostic':
            region_path = ROOT/config['region_manifest']
            assert sha(region_path)==config['region_manifest_sha256']
            photos = read(region_path)['photos']
            by_photo = {r['photo_id']:r for r in photos}
            region = dict(source=str(region_path),source_sha256=sha(region_path),
                          full_counts=dict(Counter(r['annotation_source'] for r in photos)),
                          split_counts=dict(Counter(by_photo[r['original_id']]['annotation_source'] for r in rows)),
                          interpretation='Human-confirmed boxes plus retained SAM3 defaults; not a fully human-annotated region upper bound.')
        if dataset == 'medicine':
            assert config['dataset_sha256'] == manifest['source']['inventory'], str(path)
        predictions = []
        for row in rows:
            source = by_id[row['original_id']]
            result = dict(id=row['id'],group=evaluation_group(dataset,row),identity=row['identity'],domain=row.get('domain'),
                          rank=source['rank'],raw_rank=raw[row['id']],
                          target_in_candidates=bool(set(row['candidates']) & set(labels[row['id']])))
            if 'positive_ranks' in source and all(r is not None for r in source['positive_ranks']):
                result['positive_ranks'] = dict(zip(map(str,source['positive_indices']),source['positive_ranks']))
            predictions.append(result)
        metrics = summarize(predictions)
        if dataset.startswith('forb'):
            metrics['official'] = read(path.parent/'metrics.json')
            for name,value in metrics['official'].items():
                if name.startswith('mAP'):
                    metrics['micro'][name] = value
                    metrics['macro'][name] = float(np.mean([v[name] for v in metrics['official']['by_domain'].values()]))
        results[relative] = dict(metrics=metrics,source=str(path),source_sha256=sha(path),
                                 config_sha256=sha(config_path),predictions=predictions,kind=kind,region_provenance=region)
    assert results
    save(RESULTS/'baselines'/dataset/f'{split}.json',results)
    return results


def fusion_rows(manifest, scores, split, raw):
    labels = read(WORK/'medicine/labels.json')
    first = {}
    canonical = [first.setdefault(r['sha256'],i) for i,r in enumerate(manifest['references'])]
    rows = []
    for row in manifest['rows']:
        if row['split'] != split:
            continue
        rows.append(dict(id=row['id'],group=row['group'],identity=row['identity'],
                         rank=deduplicated_rank(scores[row['query_index']],canonical,labels[row['id']]),
                         raw_rank=raw[row['id']],target_in_candidates=bool(set(row['candidates']) & set(labels[row['id']]))))
    return rows


def tuned_fusion(phase):
    from evaluation.scoring import tokenize, zscore
    from evaluation.bm25_image_fusion import load_source
    from evaluation.data import Dataset
    dataset = Dataset(ROOT)
    manifest = read(WORK/'medicine/manifest.json')
    raw = baseline_ranks(manifest)
    text_path = ROOT/'cache/scores/bm25/bm25.pt'
    text, _, _ = load_source(dataset,'bm25','bm25')
    text_scores = text['scores']
    assert [r['photo_id'] for r in dataset.photos] == [r['original_id'] for r in manifest['rows']]
    ocr = read(ROOT/'cache/ocr/photos.json')
    available = torch.tensor([bool(tokenize(ocr[r['original_id']]['text'])) for r in manifest['rows']])
    all_ocr = zscore(text_scores)
    destination = RESULTS/'baseline_selection.json'
    frozen = read(destination) if destination.exists() else {}
    results = {}
    for model in ('colpali','colnomic3b','colnomic7b','Qwen3-VL-Embedding-2B'):
        path = ROOT/f'cache/scores/{model}/image.pt'
        bank, _, _ = load_source(dataset,model,'image')
        image = zscore(bank['scores'])
        if phase == 'validation' and model not in frozen:
            candidates = []
            for i in range(21):
                alpha = i/20
                scores = (1-alpha)*image + alpha*all_ocr
                scores[~available] = image[~available]
                predictions = fusion_rows(manifest,scores.numpy(),'val',raw)
                metrics = summarize(predictions,False)
                candidates.append(dict(alpha=alpha,macro=metrics['macro'],micro=metrics['micro']))
            best = max(candidates,key=lambda c:(c['macro']['R@1'],c['macro']['MRR'],-abs(c['alpha']-.5)))
            frozen[model] = dict(alpha=best['alpha'],validation=best,grid=candidates,
                 selection='Medicine validation package macro R@1 then macro MRR; ties closest to 0.5.',
                 image_sha256=sha(path),text_sha256=sha(text_path),manifest=manifest['signature'])
            save(destination,frozen)
        assert model in frozen and frozen[model]['image_sha256']==sha(path) and frozen[model]['text_sha256']==sha(text_path)
        alpha = frozen[model]['alpha']
        scores = (1-alpha)*image + alpha*all_ocr
        scores[~available] = image[~available]
        split = 'val' if phase == 'validation' else 'test'
        predictions = fusion_rows(manifest,scores.numpy(),split,raw)
        results[model] = dict(alpha=alpha,metrics=summarize(predictions),predictions=predictions)
    save(RESULTS/'baselines'/'medicine'/f'tuned_fusion_{split}.json',results)
    print('FUSION',split,{m:(v['alpha'],v['metrics']['micro']) for m,v in results.items()},flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['validation','final'],required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.phase == 'final':
        assert (RESULTS/'selection.json').exists(), 'Freeze model selection first'
    datasets = ('medicine',) if args.phase=='validation' else tuple(FOLDERS)
    for dataset in datasets:
        split = ('val' if args.phase=='validation' else 'test') if dataset=='medicine' else 'transfer'
        results = imported(dataset,split)
        print(dataset,split,{m:round(v['metrics']['micro']['R@1']*100,2) for m,v in results.items()},flush=True)
    tuned_fusion(args.phase)


if __name__ == '__main__':
    main()
