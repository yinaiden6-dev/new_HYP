#!/usr/bin/env python3
"""Download, verify and restore the exact accepted ColNomic audit archive."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.request


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(path, row):
    return path.is_file() and path.stat().st_size == row['bytes'] and sha256(path) == row['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--parts-dir', type=Path, default=Path('.archive_parts'))
    parser.add_argument('--archive', type=Path, default=Path('colnomic_full_audit.tar.gz'))
    parser.add_argument('--extract-to', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).parent / 'archive_parts_manifest.json').read_text())
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    args.parts_dir.mkdir(parents=True, exist_ok=True)

    def obtain(row):
        path = args.parts_dir / row['name']
        if checked(path, row):
            print('VERIFIED_EXISTING_PART', row['name'], flush=True)
            return
        if not args.download:
            raise RuntimeError(f'Missing or invalid part: {path}; use --download')
        temporary = path.with_name(path.name + '.partial')
        request = urllib.request.Request(row['download_url'], headers={'User-Agent': 'ColNomic-archive-restore'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
            while data := response.read(8 * 2**20):
                output.write(data)
        if not checked(temporary, row):
            raise RuntimeError(f'Downloaded SHA256/size mismatch: {row["name"]}')
        temporary.replace(path)
        print('DOWNLOADED_AND_VERIFIED_PART', row['name'], flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(obtain, manifest['parts']))
    expected = {'bytes': manifest['archive_bytes'], 'sha256': manifest['archive_sha256']}
    if not checked(args.archive, expected):
        if args.archive.exists():
            raise RuntimeError(f'Refusing to replace an existing invalid archive: {args.archive}')
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.archive.with_name(args.archive.name + '.partial')
        digest = hashlib.sha256()
        size = 0
        with temporary.open('wb') as output:
            for row in manifest['parts']:
                with (args.parts_dir / row['name']).open('rb') as source:
                    while data := source.read(8 * 2**20):
                        output.write(data)
                        digest.update(data)
                        size += len(data)
        if size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
            raise RuntimeError('Concatenated archive SHA256/size mismatch')
        temporary.replace(args.archive)
    args.archive.with_name(args.archive.name + '.sha256').write_text(expected['sha256'] + '  ' + args.archive.name + '\n')
    print('VERIFIED_COMPLETE_ARCHIVE', args.archive, expected['sha256'], flush=True)
    if args.extract_to:
        if args.extract_to.exists() and (not args.extract_to.is_dir() or any(args.extract_to.iterdir())):
            raise RuntimeError('--extract-to must name an empty directory')
        args.extract_to.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.archive, 'r:gz') as archive:
            archive.extractall(args.extract_to, filter='data')
        print('EXTRACTED_WORKSPACE', args.extract_to, flush=True)
        print('Run from its root: python scripts/colnomic_mechanism/package_full_audit.py --verify-extracted --axes --links')


if __name__ == '__main__':
    main()
