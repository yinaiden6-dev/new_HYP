"""Verify current core cache coverage and preservation of historical encoding rows."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.data import Dataset, digest, file_hash, read_csv, read_json, write_json
from evaluation.scoring import fuse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    dataset = Dataset(ROOT)
    snapshot = args.snapshot.resolve()
    before = snapshot / 'before'
    recorded = read_json(snapshot / 'before_files.json')
    specs = read_json(ROOT / 'configs/baselines.json')['models']
    ids = [row['photo_id'] for row in dataset.photos]
    reference_ids = [row['reference_id'] for row in dataset.references]
    old_ids = [row['photo_id'] for row in read_csv(before / 'cache/image_embeddings/manifests/photos.csv')]
    positions = {identifier: position for position, identifier in enumerate(ids)}
    old_positions = [positions[identifier] for identifier in old_ids]
    report = dict(status='passed', photos=len(ids), references=len(reference_ids),
                  historical_photos=len(old_ids), added_photos=len(ids)-len(old_ids),
                  banks={}, scores={}, ocr={})
    for population, rows, key in [('photos', dataset.photos, 'photo_id'),
                                  ('references', dataset.references, 'reference_id')]:
        relative = f'cache/ocr/{population}.json'
        current = read_json(ROOT / relative)
        old = read_json(before / relative)
        assert set(current) == {row[key] for row in rows}
        for row in rows:
            entry = current[row[key]]
            assert entry['image_sha256'] == row['sha256']
            assert entry['image_path'] == 'dataset/' + row['path']
            assert isinstance(entry['text'], str)
        for identifier, entry in old.items():
            assert {k: v for k, v in entry.items() if k != 'image_path'} == {
                k: v for k, v in current[identifier].items() if k != 'image_path'}
        if population == 'references':
            assert file_hash(ROOT / relative) == recorded[relative]['sha256']
        report['ocr'][population] = dict(records=len(current), historical_text_preserved=len(old),
                                        generated=len(current)-len(old))
    for folder, view in [('image_embeddings', 'image'), ('ocr_embeddings', 'text')]:
        metadata = read_json(ROOT / 'cache' / folder / 'metadata.json')
        assert metadata['status'] == 'complete'
        for model in specs:
            if view == 'text' and 'text_model' not in specs[model]:
                continue
            relative = f'cache/{folder}/{model}/photos.pt'
            current = torch.load(ROOT / relative, weights_only=True, map_location='cpu', mmap=True)
            old = torch.load(before / relative, weights_only=True, map_location='cpu', mmap=True)
            assert current['ids'] == ids
            assert len(current['embeddings']) == len(ids)
            for i, identifier in enumerate(old['ids']):
                j = positions[identifier]
                assert torch.equal(old['embeddings'][i], current['embeddings'][j])
                for field in ['sha256', 'source_sha256', 'provenance'] + (
                        ['text_sha256', 'has_text', 'input_token_counts'] if view == 'text' else []):
                    assert old[field][i] == current[field][j]
            reference_path = f'cache/{folder}/{model}/references.pt'
            assert file_hash(ROOT / reference_path) == recorded[reference_path]['sha256']
            assert file_hash(ROOT / relative) == metadata['models'][model]['files']['photos']['sha256']
            report['banks'][f'{model}/{view}'] = dict(photos=len(ids), references=len(reference_ids),
                reused_photos=len(old['ids']), added_photos=len(ids)-len(old['ids']), reference_bytes_preserved=True)
            del old, current
    for relative in sorted(recorded):
        if not relative.startswith('cache/scores/') or not relative.endswith('.pt'):
            continue
        path = ROOT / relative
        value = torch.load(path, weights_only=True, map_location='cpu')
        old = torch.load(before / relative, weights_only=True, map_location='cpu')
        scores = value['scores']
        assert scores.shape == (len(ids), len(reference_ids)) and torch.isfinite(scores).all()
        assert value['query_ids'] == ids and value['reference_ids'] == reference_ids
        assert digest(value['recipe']) == value['signature']
        assert torch.equal(scores[old_positions], old['scores']), f'Historical score rows changed: {relative}'
        recipe = value['recipe']
        pieces = Path(relative).parts
        model, mode = pieces[2], Path(relative).stem
        if model in specs and mode in ('image', 'image_crisp', 'ocr_to_image', 'ocr_to_image_crisp', 'ocr_to_ocr'):
            query_folder = 'image_embeddings' if mode.startswith('image') else 'ocr_embeddings'
            reference_folder = 'ocr_embeddings' if mode == 'ocr_to_ocr' else 'image_embeddings'
            assert recipe['query'] == file_hash(ROOT / f'cache/{query_folder}/{model}/photos.pt')
            assert recipe['reference'] == file_hash(ROOT / f'cache/{reference_folder}/{model}/references.pt')
        if model == 'bm25':
            assert recipe['query_ocr'] == file_hash(ROOT / 'cache/ocr/photos.json')
            assert recipe['reference_ocr'] == file_hash(ROOT / 'cache/ocr/references.json')
            assert recipe['query_order'] == digest(ids) and recipe['reference_order'] == digest(reference_ids)
        if model in specs and mode in ('fusion_alpha_0.5', 'ocr_fusion_alpha_0.5'):
            text_mode = 'ocr_to_image' if mode == 'fusion_alpha_0.5' else 'ocr_to_ocr'
            image = torch.load(path.with_name('image.pt'), weights_only=True, map_location='cpu')
            text = torch.load(path.with_name(text_mode + '.pt'), weights_only=True, map_location='cpu')
            bank = torch.load(ROOT / f'cache/ocr_embeddings/{model}/photos.pt', weights_only=True, map_location='cpu', mmap=True)
            assert recipe['image'] == image['signature'] and recipe['text'] == text['signature']
            assert torch.equal(scores, fuse(image['scores'], text['scores'], bank['has_text'], .5))
        if model == 'bm25_image_fusion':
            base_model = pieces[3]
            image_path = ROOT / f'cache/scores/{base_model}/image.pt'
            bm25_path = ROOT / 'cache/scores/bm25/bm25.pt'
            assert recipe['image_score_sha256'] == file_hash(image_path)
            assert recipe['bm25_score_sha256'] == file_hash(bm25_path)
        report['scores'][str(Path(relative).relative_to('cache/scores'))] = dict(
            shape=list(scores.shape), reused_rows=len(old_ids), added_rows=len(ids)-len(old_ids),
            historical_rows_identical=True, sha256=file_hash(path))
        print('Verified', relative, flush=True)
    assert len(report['banks']) == len(specs) + sum('text_model' in spec for spec in specs.values())
    assert len(report['scores']) == 27
    write_json(snapshot / 'verification.json', report)
    print('PASSED', len(report['banks']), 'banks;', len(report['scores']), 'score caches', flush=True)


if __name__ == '__main__':
    main()
