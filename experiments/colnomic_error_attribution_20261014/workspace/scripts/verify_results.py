"""Independently check current full-cohort outputs against saved score matrices."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.data import Dataset, digest, file_hash, read_csv, read_json, write_json
from evaluation.evaluate import IDS, FUSIONS


def verify_fusion_matrix(matrix, image, text, has_text):
    """Independent FP64 formula, with a per-row FP32 rounding error budget."""
    def normalized(array):
        raw=np.asarray(array,dtype=np.float64)
        std=raw.std(axis=1,keepdims=True)
        z=np.divide(raw-raw.mean(axis=1,keepdims=True),np.maximum(std,1e-12),
            out=np.zeros_like(raw),where=std>0)
        # Mean reduction error is amplified when the score spread is narrow.
        # Bound both mean and standard-deviation rounding; constant rows stay zero.
        error=4*np.finfo(np.float32).eps*(np.max(np.abs(raw),axis=1,keepdims=True)/np.maximum(std,1e-12)
            +np.max(np.abs(z),axis=1,keepdims=True)+1)
        return z,np.where(std>0,error,0)
    a,ea=normalized(image);b,eb=normalized(text)
    expected=(a+b)*.5;roundoff=(ea+eb)*.5
    empty=~np.asarray(has_text,dtype=bool)
    expected[empty]=a[empty];roundoff[empty]=ea[empty]
    bound=2e-5+2e-5*np.abs(expected)+roundoff
    delta=np.abs(np.asarray(matrix,dtype=np.float64)-expected)
    assert np.isfinite(delta).all() and (delta<=bound).all(), 'Fusion differs beyond the per-query FP32 rounding bound'


def check_summary(rows, report):
    ranks = np.array([r['rank'] if r['rank'] is not None else np.inf for r in rows])
    values = np.column_stack([ranks <= 1, ranks <= 5, ranks <= 10, 1/ranks])
    packages = sorted({r['package_id'] for r in rows})
    assert report['queries'] == len(rows)
    assert report['packages'] == len(packages)
    assert report['source_groups'] == len({r['group_id'] for r in rows})
    assert report['correct'] == int((ranks == 1).sum())
    assert report['failures'] == int(np.isinf(ranks).sum())
    expected = {'photo_average': values.mean(0) if rows else [None]*4,
                'package_average': np.mean([values[[r['package_id'] == p for r in rows]].mean(0)
                                             for p in packages], axis=0) if rows else [None]*4}
    for average, metrics in expected.items():
        for key, value in zip(('recall_at_1', 'recall_at_5', 'recall_at_10', 'mrr'), metrics):
            if value is None:
                assert report[average][key] is None
            else:
                assert abs(report[average][key] - value) < 1e-12


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready',action='store_true',help='Verify completed results and report pending configurations')
    parser.add_argument('--read-only', action='store_true', help='Verify without rewriting verification.json')
    args=parser.parse_args()
    torch.set_num_threads(2)
    dataset = Dataset(ROOT)
    reference_ids = [r['reference_id'] for r in dataset.references]
    sha_of = {r['reference_id']: r['sha256'] for r in dataset.references}
    expected_photos = [r['photo_id'] for r in dataset.photos]
    nq,nr=len(expected_photos),len(reference_ids)
    ng=len(set(dataset.groups.values()))
    unique=len(set(sha_of.values()))
    settings = read_json(ROOT/'configs/baselines.json')
    image_meta = read_json(ROOT/'cache/image_embeddings/metadata.json')
    text_meta = read_json(ROOT/'cache/ocr_embeddings/metadata.json')
    methods = {}
    predictions = {}
    pending=[]
    for (model, mode), identifier in sorted(IDS.items(), key=lambda pair: pair[1]):
        folder = ROOT/'results'/model/mode
        if not (folder/'metrics.json').is_file():
            if args.ready:
                pending.append(identifier);continue
            raise FileNotFoundError(f'Missing result: {folder}')
        config = read_json(folder/'config.json')
        metrics = read_json(folder/'metrics.json')
        by_type = read_json(folder/'by_type.json')
        rows = [json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
        assert config['id'] == identifier and config['protocol'] == dataset.full_protocol
        assert config['queries'] == nq and config['references'] == nr
        assert dataset.matches_full_inventory(config['dataset_sha256'])
        assert config['annotation_sha256'] == dataset.annotation_sha256
        assert config['active_difference'] is None
        assert [r['photo_id'] for r in rows] == expected_photos
        assert all(n >= 0 for n in config['cost']['generated'].values())
        assert all(config['cost'][key] >= 0 for key in ('model_loading_seconds', 'encoding_seconds', 'ocr_inference_seconds'))
        filename = f'{mode}_alpha_0.5' if mode in FUSIONS else mode
        score = torch.load(ROOT/f'cache/scores/{model}/{filename}.pt', map_location='cpu', weights_only=True)
        matrix = score['scores'].numpy()
        assert matrix.shape == (nq, nr) and np.isfinite(matrix).all()
        if model == 'bm25':
            assert score['signature'] == digest(config['scoring'])
        else:
            # Storage paths may move without changing the pinned model or recipe.
            storage_keys = {'checkpoint', 'base_checkpoint'}
            identity = lambda value: {k:v for k,v in value.items() if k not in storage_keys}
            assert identity(config['model_spec']) == identity(settings['models'][model])
            assert config['score_signature'] == score['signature']
            assert score['recipe'] == config['scoring']
            if mode in FUSIONS:
                assert config['alpha'] == .5 and score['recipe']['alpha'] == .5
                text_mode='ocr_to_image' if mode=='fusion' else 'ocr_to_ocr'
                image_score=torch.load(ROOT/f'cache/scores/{model}/image.pt',map_location='cpu',weights_only=True)
                text_score=torch.load(ROOT/f'cache/scores/{model}/{text_mode}.pt',map_location='cpu',weights_only=True)
                assert score['recipe']['image']==image_score['signature']
                assert score['recipe']['text']==text_score['signature']
                ocr=read_json(ROOT/'cache/ocr/photos.json')
                verify_fusion_matrix(matrix,image_score['scores'].numpy(),text_score['scores'].numpy(),
                    [bool(ocr[p]['text'].strip()) for p in expected_photos])
            else:
                meta = text_meta if mode.startswith('ocr_to_') else image_meta
                assert score['recipe']['query'] == meta['models'][model]['files']['photos']['sha256']
                reference_meta=text_meta if mode=='ocr_to_ocr' else image_meta
                assert score['recipe']['reference'] == reference_meta['models'][model]['files']['references']['sha256']
                assert score['recipe']['scorer_sha256'] == file_hash(ROOT/'evaluation/scoring.py')
                assert score['query_ids'] == expected_photos and score['reference_ids'] == reference_ids
        # Numpy stable sorting provides a separate implementation from the runner's torch ranking.
        order = np.argsort(-matrix, axis=1, kind='stable')
        for photo, row, ranking in zip(dataset.photos, rows, order, strict=True):
            package = photo['package_id']
            assert row['package_id'] == package and row['group_id'] == dataset.groups[package]
            assert row['type'] == dataset.labels[photo['photo_id']]
            assert row['positive_reference_ids'] == sorted(dataset.positives[package])
            if row['failure'] is not None:
                ocr=read_json(ROOT/'cache/ocr/photos.json')
                assert mode.startswith('ocr_to_') and row['failure']=='empty_ocr'
                assert not ocr[row['photo_id']]['text'].strip()
                assert row['rank'] is None and not row['correct'] and not row['top_reference_ids']
                assert row['selected_reference_id'] is None
                continue
            positive = {sha_of[r] for r in dataset.positives[package]}
            seen = set()
            top = []
            rank = None
            for index in ranking:
                ref = reference_ids[index]
                sha = sha_of[ref]
                if sha in seen:
                    continue
                seen.add(sha)
                if len(top) < 10:
                    top.append(ref)
                if rank is None and sha in positive:
                    rank = len(seen)
                if rank is not None and len(top) == 10:
                    break
            assert row['rank'] == rank and row['correct'] == (rank == 1)
            assert row['top_reference_ids'] == top and row['selected_reference_id'] == top[0]
            assert row['failure'] is None
        check_summary(rows, metrics)
        assert metrics['reference_rows'] == nr and metrics['unique_reference_images'] == unique
        ci = metrics['confidence_intervals']
        assert (ci['groups'], ci['repetitions'], ci['seed']) == (ng, 10000, 20260924)
        for key, report in by_type.items():
            conditions = key.split('/')
            selected = [r for r in rows if all(
                r['type'].get('occlusion')==v.split('=',1)[1] if v.startswith('occlusion=')
                else v in r['type'].values() for v in conditions)]
            check_summary(selected, report)
        assert len(by_type) == (11 if any('occlusion' in r['type'] for r in rows) else 8)
        methods[identifier] = dict(model=model, mode=mode, predictions=len(rows),
                                   independently_verified_ranks=True, photo_average=metrics['photo_average'],
                                   package_average=metrics['package_average'])
        predictions[identifier] = rows
        print(identifier, 'verified', flush=True)
    assert len(methods)+len(pending)==len(IDS)
    summary = read_csv(ROOT/'results/summary.csv')
    assert [r['id'] for r in summary] == sorted(methods)
    for row in summary:
        for average in ('photo', 'package'):
            for key, value in methods[row['id']][average+'_average'].items():
                assert abs(float(row[average+'_'+key])-value) < 1e-12
    comparisons = read_json(ROOT/'results/paired_comparisons.json')
    for name, report in comparisons.items():
        first, second = [predictions[n] for n in name.split('_minus_')]
        assert report['first_queries'] == report['second_queries'] == nq
        assert report['rescued'] == sum(a['correct'] and not b['correct'] for a,b in zip(first,second))
        assert report['harmed'] == sum(b['correct'] and not a['correct'] for a,b in zip(first,second))
        assert report['intervals']['groups'] == ng
    # The main CSV must also include both foreground routes and all BM25-image fusions.
    from evaluation.foreground import verify_rankings, score_signature
    from evaluation.bm25_image_fusion import MODELS
    extra_paths={}
    for route,identifier in [('human_boxes','D01'),('sam3_auto','D03')]:
        latest=ROOT/'results/foreground'/route/'latest.json'
        if latest.exists():
            extra_paths[identifier]=ROOT/read_json(latest)['result']
    for i,model in enumerate(MODELS):
        extra_paths[f'F{i+1:02d}']=ROOT/'results/bm25_image_fusion'/model/'alpha_0.5'
    layout_rows = [r for r in read_csv(ROOT/'results/all_results.csv') if r['family']=='bm25_layout_fusion']
    for row in layout_rows:
        if row['id'] in extra_paths or row['id'] in methods:
            raise ValueError('Duplicate layout result ID: ' + row['id'])
        extra_paths[row['id']] = ROOT/row['result']
    for identifier in ['D01','D03',*[f'F{i+1:02d}' for i in range(len(MODELS))],
                       *[r['id'] for r in layout_rows]]:
        folder=extra_paths.get(identifier)
        if folder is None or not (folder/'metrics.json').exists():
            if args.ready:
                pending.append(identifier);continue
            raise FileNotFoundError(f'Missing result: {identifier}')
        config=read_json(folder/'config.json')
        assert config['id']==identifier and config['protocol']==dataset.full_protocol
        assert dataset.matches_full_inventory(config['dataset_sha256'])
        assert config['annotation_sha256']==dataset.annotation_sha256
        metrics=read_json(folder/'metrics.json')
        rows=[json.loads(s) for s in (folder/'predictions.jsonl').read_text().splitlines()]
        if identifier.startswith('D'):
            manifest_path=ROOT/config['region_manifest']
            spec=read_json(manifest_path.parent/'config.json')
            score=torch.load(manifest_path.parent/'scores.pt',map_location='cpu',weights_only=True)
            assert score['signature']==score_signature(spec,read_json(manifest_path),config['reference_cache_sha256'])
            if identifier=='D01':
                assert all(r['status']=='confirmed' for r in spec['annotations']['records'].values())
                assert spec['annotations']['records']==read_json(ROOT/'dataset/annotations/subject_boxes.json')['records']
        else:
            score=torch.load(ROOT/config['score_path'],map_location='cpu',weights_only=True)
            assert score['signature']==config['score_signature']==digest(config['scoring'])
            if config['mode'] in ('bm25_layout_fusion','bm25_relative_layout_fusion'):
                if config['mode']=='bm25_relative_layout_fusion':
                    from evaluation.bm25_relative_layout_fusion import verify_result
                else:
                    from evaluation.bm25_layout_fusion import verify_result
                verify_result(ROOT,dataset,folder)
        assert tuple(score['scores'].shape)==(nq,nr)
        verify_rankings(score['scores'],rows,dataset)
        assert all(r['type']==dataset.labels[r['photo_id']] for r in rows)
        check_summary(rows,metrics)
        for key,report in read_json(folder/'by_type.json').items():
            conditions=key.split('/')
            selected=[r for r in rows if all(r['type'].get('occlusion')==v.split('=',1)[1]
                if v.startswith('occlusion=') else v in r['type'].values() for v in conditions)]
            check_summary(selected,report)
        assert metrics['confidence_intervals']['groups']==ng
        methods[identifier]=dict(model=config.get('model',config.get('settings',{}).get('model')),
            mode=config.get('mode',config.get('route')),predictions=len(rows),independently_verified_ranks=True,
            photo_average=metrics['photo_average'],package_average=metrics['package_average'])
        print(identifier,'verified',flush=True)
    decision_rows=[r for r in read_csv(ROOT/'results/all_results.csv') if r['family']=='frozen_decision_head']
    for row in decision_rows:
        from evaluation.content_box_ce_polish18 import verify_result as verify_decision_result
        assert row['id'] not in methods
        report=verify_decision_result(ROOT,dataset,ROOT/row['result'])
        methods[row['id']]=report
        print(row['id'],'fixed-head decisions verified; full-ranking metrics unavailable',flush=True)
    all_summary=read_csv(ROOT/'results/all_results.csv')
    assert sorted(r['id'] for r in all_summary)==sorted(methods)
    assert len(methods)+len(pending)==len(IDS)+2+len(MODELS)+len(layout_rows)+len(decision_rows)
    for row in all_summary:
        assert int(row['queries'])==nq and int(row['references'])==nr
        for key,value in methods[row['id']]['photo_average'].items():
            if value is None:
                assert row[key]=='' and methods[row['id']]['ranking_scope']=='decision_only'
            else:
                assert abs(float(row[key])-value)<1e-12
    if not args.read_only:
        write_json(ROOT/'results/verification.json', dict(status='partial' if pending else 'passed',
            pending_configurations=pending,completed_configurations_verified=True,
            verified_utc=datetime.now(timezone.utc).isoformat(),
            dataset_sha256=dataset.hashes,annotation_sha256=dataset.annotation_sha256,
            configurations=len(methods),queries_per_configuration=nq,references=nr,unique_reference_images=unique,
            source_groups=ng,paired_comparisons=len(comparisons),methods=methods))
    print(f'PASS: {len(methods)} full-cohort results; pending: {pending}', flush=True)


if __name__ == '__main__':
    main()
