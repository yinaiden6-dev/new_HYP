"""Verify this dataset or link it to an independently extracted DailyMed copy."""
import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

from splits import historical_inputs, verify as verify_splits
from photo_metadata import PhotoStore, validate_photo


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def load_packages(root):
    packages = []
    photo_rows = {p['photo_id']: p for p in read_csv(root / 'lists/photos.csv')}
    for entry in read_csv(root / 'lists/packages.csv'):
        package = json.loads((root / entry['metadata_path']).read_text())
        package['directory'] = entry['directory']
        package['metadata_path'] = entry['metadata_path']
        # Metadata paths are local to a package; normalized paths use the dataset root.
        for photo in package['photos']:
            validate_photo(photo)
            row = photo_rows[Path(photo['image']).stem]
            photo.update(photo_id=row['photo_id'], file=photo['image'], sha256=row['sha256'],
                         source_sha256=row['source_sha256'], design_id=row['design_id'])
        for asset in [package['design'], *package['photos'], *package['documents']]:
            asset['file'] = entry['directory'] + '/' + asset['file']
        packages.append(package)
    return packages


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def header(path):
    fields = {'id': 'document_id', 'setId': 'set_id', 'versionNumber': 'version'}
    result = {}
    depth = -1
    with path.open('rb') as stream:
        for event, node in ET.iterparse(stream, events=('start', 'end')):
            if event == 'start':
                depth += 1
                name = node.tag.rsplit('}', 1)[-1]
                if depth == 1 and name in fields:
                    value = node.get('value' if name == 'versionNumber' else 'root')
                    result[fields[name]] = value.lower() if value else value
                if len(result) == len(fields):
                    break
            else:
                depth -= 1
    return result


def check_reference_metadata(root, references, documents):
    entries = json.loads((root / 'references/metadata.json').read_text())
    if not isinstance(entries, list):
        return ['Reference metadata must be an image list']
    errors = []
    if len(entries) != len(references):
        errors.append('Reference metadata count differs')
    for entry, row in zip(entries, references):
        identifier = row['reference_id']
        if set(entry) != {'image', 'dailymed'}:
            errors.append('Unexpected reference metadata fields: ' + identifier)
        if entry['image'] != 'images/' + identifier + '.jpg' or 'references/' + entry['image'] != row['path']:
            errors.append('Unexpected reference image path: ' + identifier)
        label = entry['dailymed']
        if not isinstance(label, dict) or set(label) != {'id', 'image', 'url', 'type'}:
            errors.append('Expected one DailyMed source object: ' + identifier)
            continue
        if not all(isinstance(label[k], str) and label[k] for k in label):
            errors.append('Incomplete DailyMed locator: ' + identifier)
            continue
        url = urlsplit(label['url'])
        query = parse_qs(url.query)
        try:
            UUID(label['id'])
            if set(query) != {'setid'} or len(query['setid']) != 1:
                raise ValueError('Expected one set ID')
            UUID(query['setid'][0])
        except ValueError:
            errors.append('Invalid DailyMed identifier: ' + identifier)
        if (url.scheme, url.netloc, url.path) != ('https', 'dailymed.nlm.nih.gov', '/dailymed/drugInfo.cfm'):
            errors.append('Invalid DailyMed label URL: ' + identifier)
        if label['type'] != label['type'].upper():
            errors.append('Reference type capitalization differs: ' + identifier)
        doc = documents.get(label['id'])
        if doc:
            if label['type'] != doc['document_type']['displayName'].upper():
                errors.append('Reference type differs from packaged XML: ' + identifier)
            if query.get('setid') != [doc['set_id']]:
                errors.append('Reference URL differs from packaged XML: ' + identifier)
    return errors


def verify(root, *, reconstruct=True):
    packages = load_packages(root)
    references = read_csv(root / 'lists/references.csv')
    collection_paths = {name: name + '.csv' for name in ('photos', 'train', 'val', 'test')}
    collection_paths.update(train_25='train_subsets/25.csv', train_50='train_subsets/50.csv')
    if (root / 'lists/review.csv').exists():
        collection_paths['review'] = 'review.csv'
    collections = {name: read_csv(root / 'lists' / path)
                   for name, path in collection_paths.items()}
    tables = dict(photos=[p for group in packages for p in group['photos']],
                  designs=[group['design'] for group in packages],
                  documents=[d for group in packages for d in group['documents']])
    errors = []
    for name, key in [('photos', 'photo_id'), ('designs', 'design_id')]:
        values = [r[key] for r in tables[name]]
        if len(values) != len(set(values)):
            errors.append('Duplicate IDs in ' + name)
    if len({p['package_id'] for p in packages}) != len(packages):
        errors.append('Duplicate package IDs')
    photos = {r['photo_id']: r for r in tables['photos']}
    designs = {r['design_id']: r for r in tables['designs']}
    docs = {r['document_id']: r for r in tables['documents']}
    errors.extend(check_reference_metadata(root, references, docs))
    for row in tables['documents']:
        if any(row[k] != docs[row['document_id']][k] for k in ('sha256', 'set_id', 'version')):
            errors.append('Conflicting copies of XML document ' + row['document_id'])
    if len({r['source_sha256'] for r in photos.values()}) != len(photos):
        errors.append('Duplicate source photograph bytes')
    expected_files = {r['file'] for rows in tables.values() for r in rows}
    expected_files.update(r['path'] for r in references)
    expected_files.update(p['metadata_path'] for p in packages)
    expected_files.add('references/metadata.json')
    actual_files = {p.relative_to(root).as_posix() for folder in ('packages', 'references')
                    for p in (root / folder).rglob('*') if p.is_file()}
    if actual_files != expected_files:
        errors.append('Asset directories differ from manifest file lists')

    def check_file(row):
        path = root / row['file']
        if not path.is_file():
            return 'Missing file: ' + row['file']
        if digest(path) != row['sha256']:
            return 'Checksum mismatch: ' + row['file']

    assets = [r for rows in tables.values() for r in rows] + [dict(file=r['path'], sha256=r['sha256']) for r in references]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for problem in pool.map(check_file, assets):
            if problem:
                errors.append(problem)
    package_lookup = {p['package_id']: p for p in packages}
    for package in packages:
        design = package['design']
        local_docs = {d['document_id']: d for d in package['documents']}
        if set(local_docs) != {a['document_id'] for a in design['xml_links']}:
            errors.append('Package XML inventory differs: ' + package['package_id'])
        if any(p['design_id'] != design['design_id'] for p in package['photos']):
            errors.append('Photograph stored under a different design')
        for association in design['xml_links']:
            doc = local_docs.get(association['document_id'])
            if not doc or (association['xml_reference'] is not None and association['xml_reference'] not in doc['media_references']):
                errors.append('Broken XML link: ' + design['design_id'])
    for row in tables['documents']:
        actual = header(root / row['file'])
        if any(actual.get(key) != row[key] for key in ('document_id', 'set_id', 'version')):
            errors.append('XML identity mismatch: ' + row['document_id'])
        if dict(ET.parse(root / row['file']).getroot().find('{*}code').attrib) != row['document_type']:
            errors.append('XML document type mismatch: ' + row['document_id'])
    for name, rows in collections.items():
        ids = [r['photo_id'] for r in rows]
        if len(ids) != len(set(ids)) or not set(ids) <= set(photos):
            errors.append('Invalid collection IDs: ' + name)
        for row in rows:
            photo = photos[row['photo_id']]
            package = package_lookup[row['package_id']]
            if (row['path'] != photo['file'] or row['sha256'] != photo['sha256'] or
                    row['design_id'] != photo['design_id'] or row['category'] != package['category'] or
                    row['identity'] != package['identity'] or row['source_sha256'] != photo['source_sha256'] or
                    row['design_id'] != package['design']['design_id']):
                errors.append('Collection metadata differs: ' + row['photo_id'])
    if {r['photo_id'] for r in collections['photos']} != set(photos):
        errors.append('Photo list does not cover every package photograph')
    if [int(r['row']) for r in references] != list(range(len(references))):
        errors.append('Reference row order differs')
    if [r['reference_id'] for r in references] != [f'reference_{i:06d}' for i in range(len(references))]:
        errors.append('Reference IDs must be consecutive and match row order')
    linked = defaultdict(list)
    for row in references:
        if row['package_id']:
            linked[row['package_id']].append(row['reference_id'])
    for package in packages:
        expected = linked[package['package_id']]
        if len(expected) != 1 or package.get('reference_id') not in expected:
            errors.append('Package must identify exactly one reference: ' + package['package_id'])
    if len({r['path'] for r in references}) != len(references) or len({r['reference_id'] for r in references}) != len(references):
        errors.append('Reference paths and IDs must be distinct for each entry')
    for row in references:
        if row['design_id'] != 'design_' + row['sha256'][:16]:
            errors.append('Reference design ID differs from original image bytes')
        if row['package_id']:
            package = package_lookup[row['package_id']]
            if row['sha256'] != package['design']['sha256'] or row['category'] != package['category']:
                errors.append('Reference package association differs')
    targets = {(r['identity'], r['sha256']) for r in references}
    if any((p['identity'], p['design']['sha256']) not in targets for p in packages):
        errors.append('A package design is absent from the full reference inventory')
    counts = dict(packages=len(packages), photos=len(photos), reference_entries=len(references), xml_documents=len(docs))
    release = json.loads((root / 'release.json').read_text())
    if counts != release['counts']:
        errors.append('Release counts differ from file inventories')
    if release['collections']['references'].get('metadata') != 'references/metadata.json':
        errors.append('Release reference metadata path differs')
    if dict(Counter(p['category'] for p in packages)) != release['package_categories']:
        errors.append('Release category counts differ')
    errors.extend(verify_splits(root, reconstruct=reconstruct))
    try:
        PhotoStore(root).verify_active()
    except (ValueError, FileNotFoundError) as error:
        errors.append(str(error))
    for name, entry in release['collections'].items():
        if len(read_csv(root / entry['file'])) != entry['count']:
            errors.append('Release collection count differs: ' + name)
    result = dict(status='passed' if not errors else 'failed',
                  counts=counts, collections={name: len(rows) for name, rows in collections.items()},
                  categories=release['package_categories'], errors=errors,
                  split_reconstruction='checked' if reconstruct else 'skipped by explicit option',
                  skipped_historical_inputs=[] if reconstruct else sorted(historical_inputs(
                      json.loads((root / 'lists/protocol.json').read_text()))))
    print(json.dumps(result, indent=2))
    return bool(errors)


def link(root, upstream, output):
    packages = load_packages(root)
    docs = {d['document_id']: d for p in packages for d in p['documents']}
    wanted_sets = {r['set_id'] for r in docs.values()}
    found = defaultdict(list)
    versions = defaultdict(set)
    parse_errors = []
    def inspect_xml(path):
        try:
            return path, header(path), None
        except (ET.ParseError, OSError) as error:
            return path, {}, str(error)

    # XML root IDs are authoritative; directory names are not required to match.
    paths = (p for p in upstream.rglob('*') if p.suffix.lower() == '.xml' and p.is_file())
    with ThreadPoolExecutor(max_workers=8) as pool:
        for path, fields, error in pool.map(inspect_xml, paths):
            if error:
                parse_errors.append(dict(file=str(path), error=error))
                continue
            if fields.get('set_id') in wanted_sets:
                versions[fields['set_id']].add(fields.get('version'))
            if fields.get('document_id') in docs:
                found[fields['document_id']].append((path, fields))
    design_links = defaultdict(list)
    for design in (p['design'] for p in packages):
        for association in design['xml_links']:
            design_links[association['document_id']].append((design, association))
    records = []
    for doc_id, doc in sorted(docs.items()):
        matches = []
        for path, fields in found[doc_id]:
            row = dict(file=str(path), sha256=digest(path),
                       identifiers_match=all(fields[k] == doc[k] for k in ('document_id', 'set_id', 'version')))
            row['bytes_match'] = row['sha256'] == doc['sha256']
            row['designs'] = []
            for design, association in design_links[doc_id]:
                candidate = path.parent / association['image_file']
                exists = candidate.is_file()
                row['designs'].append(dict(design_id=design['design_id'], file=str(candidate),
                                           status='missing' if not exists else
                                           ('matched' if digest(candidate) == design['sha256'] else 'different_bytes')))
            matches.append(row)
        status = 'matched' if any(m['identifiers_match'] and m['bytes_match'] for m in matches) else ('different_bytes' if matches else 'missing_version')
        records.append(dict(document_id=doc_id, set_id=doc['set_id'], version=doc['version'],
                            status=status, versions_present=sorted(versions[doc['set_id']]), matches=matches))
    design_counts = Counter(d['status'] for r in records for m in r['matches'] for d in m['designs'])
    result = dict(counts=dict(Counter(r['status'] for r in records)), design_counts=dict(design_counts), documents=records, parse_errors=parse_errors)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), counts=result['counts'], design_counts=dict(design_counts), parse_errors=len(parse_errors)), indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest='command', required=True)
    audit = subparsers.add_parser('verify')
    audit.add_argument('--skip-split-reconstruction', action='store_true',
                       help='Check frozen splits and assets without regenerating the old allocation')
    join = subparsers.add_parser('link')
    join.add_argument('upstream', type=Path, help='Directory containing extracted DailyMed XML and media')
    join.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'link':
        if not args.upstream.is_dir():
            parser.error('Upstream directory does not exist')
        return link(args.root, args.upstream, args.output)
    return verify(args.root, reconstruct=not args.skip_split_reconstruction)


if __name__ == '__main__':
    raise SystemExit(main())
