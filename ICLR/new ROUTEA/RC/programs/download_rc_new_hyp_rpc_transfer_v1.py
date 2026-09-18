#!/usr/bin/env python3
"""Receive the frozen RPC image list; decode only for integrity, never score."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import zipfile

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_new_hyp_rpc_transfer_v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert json.loads(p.read_text()) == data, 'IMMUTABLE_RECEIPT'
        return
    with p.open('x') as f:
        json.dump(data, f, sort_keys=True, indent=2)
        f.write('\n')


def receive(row, manifest):
    path = Path(row['image_path'])
    assert path.parent == OUT / 'images' and path.name == row['asset_id'] + '.jpg'
    receipt = OUT / 'downloads' / (row['asset_id'] + '.json')
    if receipt.exists():
        r = json.loads(receipt.read_text())
        assert r['image_sha256'] == sha(path.read_bytes())
        return r
    relative = manifest['dataset_path_prefix'] + row['source_filename']
    url = manifest['endpoint_prefix'] + urllib.parse.quote(relative, safe='')
    error = None
    for attempt in range(3):
        try:
            if path.exists():
                data = path.read_bytes()
            else:
                request = urllib.request.Request(url, headers={'User-Agent': 'RouteA-public-research-intake/1.0'})
                with urllib.request.urlopen(request, timeout=25) as response:
                    blob = response.read(20000001)
                assert len(blob) <= 20000000, 'DOWNLOAD_SIZE_BOUND'
                if blob.startswith(b'PK\x03\x04'):
                    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                        files = archive.infolist()
                        assert len(files) == 1 and Path(files[0].filename).name == row['source_filename']
                        assert files[0].file_size <= 20000000, 'ARCHIVE_SIZE_BOUND'
                        data = archive.read(files[0])
                else:
                    data = blob
            with Image.open(io.BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                assert image.width > 0 and image.height > 0
                pixel_sha = sha(str(image.size).encode() + b'|RGB|' + image.tobytes())
                size = list(image.size)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open('xb') as f:
                    f.write(data)
            result = dict(asset_id=row['asset_id'], role=row['role'], image_path=str(path),
                          source_filename=row['source_filename'], source_url=url,
                          image_sha256=sha(data), oriented_RGB_sha256=pixel_sha,
                          bytes=len(data), size=size)
            write_json(receipt, result)
            return result
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(row['asset_id'] + ': ' + str(error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    assert 1 <= args.workers <= 8
    validation = json.loads((OUT / 'metadata_validation.json').read_text())
    for b in validation['manifest_sources'].values():
        assert sha(Path(b['path']).read_bytes()) == b['sha256'], 'MANIFEST_DRIFT'
    assert sha(Path(validation['plan']['path']).read_bytes()) == validation['plan']['sha256']
    manifest = json.loads((OUT / 'download_manifest.json').read_text())
    records = manifest['records']
    assert len(records) == 800 and len({r['asset_id'] for r in records}) == 800
    if args.limit is not None:
        assert 1 <= args.limit <= 800
        records = records[:args.limit]
    errors, completed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(receive, r, manifest): r['asset_id'] for r in records}
        for future in as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as exc:
                errors.append(dict(asset_id=futures[future], error=str(exc)))
            if (len(completed) + len(errors)) % 50 == 0:
                print(json.dumps(dict(received=len(completed), errors=len(errors), requested=len(records))), flush=True)
    if errors:
        print(json.dumps(dict(status='DOWNLOAD_INCOMPLETE', errors=errors)), flush=True)
        raise SystemExit(1)
    if args.limit is None:
        write_json(OUT / 'download_receipt.json', dict(status='RPC800_PUBLIC_IMAGES_RECEIVED_NOT_SCORED',
                   count=800, queries=600, references=200,
                   files=sorted(completed, key=lambda r: r['asset_id']),
                   manifest_sha256=sha((OUT / 'download_manifest.json').read_bytes()),
                   program_sha256=sha(Path(__file__).read_bytes()),
                   image_visual_inspections=0, model_inferences=0, HYP_GO_claimed=False))
    print(json.dumps(dict(status='RECEIVED', count=len(completed), scores_computed=0)), flush=True)


if __name__ == '__main__':
    main()
