"""Incremental PP-OCRv6 extraction on the fixed GroZi retrieval subset."""
import argparse
import fcntl
import os
from pathlib import Path
import time

from .data import digest, file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'other_datasets/GroZi-3.2k'


def run(data, device):
    from .ocr import OCR
    metadata = read_json(data / 'retrieval_subset/metadata.json')
    configuration = dict(config=read_json(ROOT / 'configs/ppocrv6.json'),
                         models=read_json(ROOT / 'configs/ppocrv6_models.json'),
                         implementation=file_hash(ROOT / 'evaluation/ocr.py'))
    signature = digest(dict(configuration=configuration,
                            subset=file_hash(data / 'retrieval_subset/metadata.json')))
    engine = None
    loading_seconds = 0.
    for population, key, records in [('photo', 'photo_id', metadata['photos']),
                                     ('ref', 'reference_id', metadata['references'])]:
        path = data / 'cache/ocr' / (population + '.json')
        bank = read_json(path) if path.exists() else dict(schema='grozi.ocr.v1',
            signature=signature, configuration=configuration, entries={},
            inference_seconds=0., complete=False, population=population)
        if bank['signature'] != signature:
            raise ValueError(f'Stale OCR cache: {path}')
        for index, row in enumerate(records):
            identifier = row[key]
            image_path = data / 'retrieval_subset' / row['image_path']
            if file_hash(image_path) != row['sha256']:
                raise ValueError(f'Changed image: {image_path}')
            if identifier in bank['entries']:
                if bank['entries'][identifier]['image_sha256'] != row['sha256']:
                    raise ValueError(f'Stale OCR entry: {identifier}')
                continue
            if engine is None:
                start = time.monotonic()
                engine = OCR(ROOT, device)
                loading_seconds = time.monotonic() - start
            entry, elapsed = engine.recognize(dict(path=row['image_path'], sha256=row['sha256']), image_path)
            entry['image_path'] = 'retrieval_subset/' + row['image_path']
            entry['inference_seconds'] = elapsed
            bank['entries'][identifier] = entry
            bank['inference_seconds'] += elapsed
            if (index + 1) % 8 == 0 or index + 1 == len(records):
                write_json(path, bank)
                print(f'OCR/{population}: {index+1}/{len(records)}, {bank["inference_seconds"]:.1f}s', flush=True)
        assert set(bank['entries']) == {r[key] for r in records}
        bank['entries'] = {r[key]: bank['entries'][r[key]] for r in records}
        bank.update(complete=True, ids=list(bank['entries']), count=len(records),
                    image_path_base='GroZi-3.2k directory',
                    empty_ids=[k for k, v in bank['entries'].items() if not v['text'].strip()])
        write_json(path, bank)
    write_json(data / 'cache/ocr/metadata.json', dict(signature=signature,
        configuration=configuration, loading_seconds=loading_seconds,
        source=file_hash(Path(__file__)), complete=True,
        files={p: file_hash(data / 'cache/ocr' / f'{p}.json') for p in ('photo', 'ref')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--device', default='gpu:0')
    args = parser.parse_args()
    lock = args.data / 'cache/intermediate/ocr.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.data, args.device)
