"""Verify final text-only OCR banks against OCR, image-bank manifests and dataset lists."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import time
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.data import validate_encoding_manifest
CACHE = ROOT / 'cache/ocr_embeddings'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready',action='store_true',help='Verify the models already merged while others are building')
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    meta = json.loads((CACHE / 'metadata.json').read_text())
    assert args.ready or meta['status'] == 'complete', 'Embedding generation is not complete yet'
    image_meta = json.loads((ROOT / 'cache/image_embeddings/metadata.json').read_text())
    assert set(meta['models']).issubset(image_meta['models'])
    specs=json.loads((ROOT/'configs/baselines.json').read_text())['models']
    expected={name for name,spec in specs.items() if 'text_model' in spec}
    assert args.ready or set(meta['models']) == expected
    report = dict(status=meta['status'], models={}, total_records=0, total_nonempty_embeddings=0,
                  all_ocr_text_retained=True, images_supplied=False)
    inputs = {}
    for pop in ('photos', 'references'):
        manifest = ROOT / f'dataset/lists/{pop}.csv'
        with manifest.open() as f:
            rows = list(csv.DictReader(f))
        ocr_path = ROOT / f'cache/ocr/{pop}.json'
        ocr = json.loads(ocr_path.read_text())
        id_key = 'photo_id' if pop == 'photos' else 'reference_id'
        assert set(ocr) == {r[id_key] for r in rows}
        recorded = validate_encoding_manifest(ROOT, pop, CACHE / f'manifests/{pop}.csv')
        validate_encoding_manifest(ROOT, pop, ROOT / f'cache/image_embeddings/manifests/{pop}.csv')
        inputs[pop] = (rows, ocr, sha256(CACHE / f'manifests/{pop}.csv'), sha256(ocr_path), id_key)
    for name, model_meta in meta['models'].items():
        image_spec = image_meta['models'][name]['model']
        assert all(model_meta['model'][k] == v for k,v in image_spec.items())
        assert model_meta['runtime']['script_sha256']
        for relative, digest in model_meta['runtime']['input_files'].items():
            source = ROOT / relative
            if relative in ('dataset/lists/photos.csv', 'dataset/lists/references.csv'):
                source = CACHE / 'manifests' / Path(relative).name
            assert sha256(source) == digest
        results = {}
        for pop, (rows, ocr, csv_hash, ocr_hash, id_key) in inputs.items():
            path = CACHE / name / f'{pop}.pt'
            bank = torch.load(path, map_location='cpu', weights_only=True)
            image_bank = torch.load(ROOT / f'cache/image_embeddings/{name}/{pop}.pt',
                                    map_location='cpu', weights_only=True, mmap=True)
            for field in ('ids','sha256','source_sha256'):
                assert bank[field] == image_bank[field], f'Image/OCR alignment mismatch: {name}/{pop}/{field}'
            assert len(bank['embeddings']) == len(image_bank['embeddings'])
            del image_bank
            assert bank['schema'] == 'release.ocr_embeddings.v1'
            assert bank['view'] == 'text' and bank['population'] == pop
            assert bank['model'] == model_meta['model']
            assert bank['ids'] == [r[id_key] for r in rows]
            assert len(set(bank['ids'])) == len(rows)
            recorded = validate_encoding_manifest(ROOT, pop, CACHE / bank['manifest'], bank)
            original_paths = {r[id_key]: 'dataset/' + r['path'] for r in recorded}
            assert bank['sha256'] == [r['sha256'] for r in rows]
            assert bank['source_sha256'] == [r.get('source_sha256',r['sha256']) for r in rows]
            assert bank['ocr_source_sha256'] == ocr_hash and bank['text_field'] == 'text'
            assert bank['contract'] == model_meta['contract']
            contract = bank['contract']
            assert contract['truncation'] is False and contract['chunking'] is False
            assert contract['images_supplied'] is False
            assert len(bank['embeddings']) == len(bank['provenance']) == len(rows)
            assert len(bank['has_text']) == len(bank['text_sha256']) == len(bank['input_token_counts']) == len(rows)
            empty_ids = []
            lengths = []
            for i, row in enumerate(rows):
                text = ocr[row[id_key]]['text']
                has_text = bool(text.strip())
                assert bank['has_text'][i] == has_text
                digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
                assert bank['text_sha256'][i] == digest
                assert bank['provenance'][i] == dict(kind='generated_text' if has_text else 'empty_text',
                    ocr_id=row[id_key], text_sha256=digest)
                assert ocr[row[id_key]]['image_sha256'] == row['sha256']
                assert ocr[row[id_key]]['image_path'] in (original_paths[row[id_key]], 'dataset/'+row['path'])
                emb = bank['embeddings'][i]
                assert torch.isfinite(emb).all()
                tokens = bank['input_token_counts'][i]
                assert 0 <= tokens <= contract['context_limit']
                if name == 'Qwen3-VL-Embedding-2B':
                    assert emb.shape == (2048,) and emb.dtype == torch.float32
                else:
                    assert emb.shape == (tokens,128) and emb.dtype == torch.float16
                if has_text:
                    assert tokens > 0
                    assert ((emb.float().norm(dim=-1)-1).abs() < 0.03).all()
                else:
                    assert tokens == 0 and not torch.count_nonzero(emb)
                    empty_ids.append(row[id_key])
                lengths.append(tokens)
            digest = sha256(path)
            assert digest == model_meta['files'][pop]['sha256']
            assert path.stat().st_size == model_meta['files'][pop]['bytes']
            counts = model_meta['counts'][pop]
            assert counts['records'] == len(rows)
            assert counts['empty_ids'] == empty_ids
            assert counts['nonempty_embeddings'] == len(rows)-len(empty_ids)
            assert counts['max_input_tokens'] == max(lengths)
            results[pop] = dict(records=len(rows), nonempty_embeddings=len(rows)-len(empty_ids),
                empty_ids=empty_ids, order_matches_image_manifest=True, finite=True, normalized_nonempty=True,
                input_token_min=min(lengths), input_token_max=max(lengths), sha256=digest)
            report['total_records'] += len(rows)
            report['total_nonempty_embeddings'] += len(rows)-len(empty_ids)
            print(name, pop, len(rows), 'verified', flush=True)
        report['models'][name] = results
    assert report['total_records'] == meta['total_records']
    assert report['total_nonempty_embeddings'] == meta['total_nonempty_embeddings']
    report['seconds'] = round(time.monotonic()-started,2)
    (CACHE / 'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'].upper(), report['total_records'], 'records verified', flush=True)


if __name__ == '__main__':
    main()
