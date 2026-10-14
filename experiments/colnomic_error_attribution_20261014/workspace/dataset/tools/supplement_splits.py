"""Extend frozen splits with new source groups, preserving existing memberships."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from allocate_splits import apportion, csv_bytes, json_bytes
from splits import checksum, read_csv


def source_groups(packages, photos, previous_groups):
    """Join source identities transitively, including previously frozen groups."""
    parent = {p['package_id']: p['package_id'] for p in packages}

    def find(pid):
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid

    owners = {}

    def associate(pid, key):
        if key in owners:
            left, right = find(pid), find(owners[key])
            parent[max(left, right)] = min(left, right)
        else:
            owners[key] = pid

    for row in previous_groups:
        associate(row['package_id'], ('legacy_group', row['group_id']))
    for package in packages:
        pid = package['package_id']
        associate(pid, ('identity', package['identity']))
        associate(pid, ('design', package['design']['sha256']))
        for doc in package['documents']:
            for field in ('document_id', 'set_id'):
                associate(pid, (field, doc[field]))
            for product in doc['products']:
                if product.get('code'):
                    associate(pid, ('product', product['code_system'], product['code']))
    for row in photos:
        for field in ('sha256', 'source_sha256'):
            associate(row['package_id'], (field, row[field]))
    members = defaultdict(list)
    for pid in sorted(parent):
        members[find(pid)].append(pid)
    previous = {r['package_id']: r for r in previous_groups}
    groups, roles = {}, {}
    for ids in members.values():
        old = [previous[pid] for pid in ids if pid in previous]
        old_ids = {r['group_id'] for r in old}
        if len(old_ids) > 1:
            raise ValueError('Supplement links distinct frozen groups: ' + ', '.join(ids))
        if old:
            gid = old[0]['group_id']
            roles[gid] = old[0]['split']
        else:
            gid = 'group_' + hashlib.sha256('\n'.join(ids).encode()).hexdigest()[:16]
        for pid in ids:
            groups[pid] = gid
    return groups, roles


def select_groups(ids, count, categories, seed, tag):
    if not 0 <= count <= len(ids):
        raise ValueError('New-group budget conflicts with frozen memberships')
    buckets = defaultdict(list)
    for gid in sorted(ids):
        buckets[categories[gid]].append(gid)
    quotas = apportion(count, {c: len(v) for c, v in buckets.items()})
    return {gid for category, values in buckets.items()
            for gid in sorted(values, key=lambda g: (
                hashlib.sha256(f'{seed}:{tag}:{g}'.encode()).hexdigest(), g))[:quotas[category]]}


def generate(root, rules):
    root = Path(root)
    previous = root / rules['previous_lists']
    if checksum(previous / 'protocol.json') != rules['previous_protocol_sha256']:
        raise ValueError('Pre-supplement protocol changed')
    old = json.loads((previous / 'protocol.json').read_text())
    for name, expected in old['output_sha256'].items():
        if checksum(previous / name) != expected:
            raise ValueError('Pre-supplement split changed: ' + name)
    for name, expected in rules['source_sha256'].items():
        if checksum(root / name) != expected:
            raise ValueError('Supplement inventory changed: ' + name)
    photos = read_csv(root / 'lists/photos.csv')
    package_rows = read_csv(root / 'lists/packages.csv')
    packages = [json.loads((root / r['metadata_path']).read_text()) for r in package_rows]
    old_groups = read_csv(previous / 'groups.csv')
    group_of, assigned = source_groups(packages, photos, old_groups)
    frozen_ids = set(assigned)
    members = defaultdict(list)
    for row in package_rows:
        members[group_of[row['package_id']]].append(row)
    if any(len({r['category'] for r in rows}) != 1 for rows in members.values()):
        raise ValueError('A source group spans categories; an explicit stratum is required')
    categories = {gid: rows[0]['category'] for gid, rows in members.items()}
    targets = apportion(len(members), rules['group_percentages'])
    splits = {name: {gid for gid, role in assigned.items() if role == name}
              for name in ('train', 'val', 'test', 'review')}
    available = set(members) - frozen_ids
    for name in ('train', 'val', 'test'):
        chosen = select_groups(available, targets[name] - len(splits[name]),
                               categories, rules['seed'], name)
        splits[name].update(chosen)
        available -= chosen
    if available:
        raise ValueError('Unassigned supplement groups remain')
    assigned = {gid: name for name, ids in splits.items() for gid in ids}
    groups = [dict(package_id=r['package_id'], group_id=group_of[r['package_id']],
                   split=assigned[group_of[r['package_id']]]) for r in package_rows]
    old_group_of = {r['package_id']: r['group_id'] for r in old_groups}
    extra_train = splits['train'] - frozen_ids
    # Extend both existing nested budgets by the same fixed new-group order.
    new_order = sorted(extra_train, key=lambda g: (
        hashlib.sha256(f"{rules['seed']}:budget:{g}".encode()).hexdigest(), g))
    for percent in (25, 50):
        chosen = {old_group_of[r['package_id']]
                  for r in read_csv(previous / f'train_subsets/{percent}.csv')}
        target = (len(splits['train']) * percent + 50) // 100
        extra = target - len(chosen)
        if not 0 <= extra <= len(new_order):
            raise ValueError('Training subset budget conflicts with frozen groups')
        splits[f'train_{percent}'] = chosen | set(new_order[:extra])
    files, entries = {}, {}
    for name, ids in splits.items():
        rows = [r for r in photos if group_of[r['package_id']] in ids]
        filename = f'train_subsets/{name[6:]}.csv' if name.startswith('train_') else name + '.csv'
        files[filename] = csv_bytes(rows, list(photos[0]))
        entries[name] = dict(file='lists/' + filename, photos=len(rows),
                             packages=len({r['package_id'] for r in rows}), groups=len(ids),
                             categories=dict(Counter(r['category'] for r in rows)))
    files['groups.csv'] = csv_bytes(groups, ['package_id', 'group_id', 'split'])
    added = set(group_of) - set(old_group_of)
    report = dict(protocol=rules['name'], previous_protocol=old['name'],
                  target_group_counts=targets, collections=entries,
                  training_reference_count=entries['train']['packages'],
                  preserved_packages=len(old_groups), changed_packages=[],
                  added_packages=[r for r in groups if r['package_id'] in added],
                  scene_balance={'rule': 'No scene balancing; supplement scene labels are pending.'},
                  source_grouping='Transitive identity, design hash, photo hashes, document ID, Set ID, product codes and frozen groups.',
                  allocation='Preserve existing roles; allocate only new groups by category and seeded SHA-256 order to total 70/15/15 group targets.')
    files['allocation_report.json'] = json_bytes(report)
    protocol = dict(old)
    protocol.update(name=rules['name'], seed=rules['seed'], frozen_date=rules['date'],
                    finalized_date=rules['date'], inventory_version=rules['inventory_version'],
                    photo_count=len(photos), group_count=len(members),
                    validation_group_count=len(splits['val']), collections=entries,
                    allocation=report['allocation'],
                    training_subsets='Preserve earlier nested group budgets; append new training groups in seeded order to rounded 25%/50% source-group targets.',
                    selection=f"Add {len(photos) - old['photo_count']} unchanged photographs and {len(added)} packaging designs; reference IDs, image bytes and gallery order remain fixed.",
                    historical_exposure=f"Original {old['photo_count']} photos have development/inspection history. Supplement photos were supplied on {rules['date']}; earlier use and capture-session independence are unknown.",
                    current_evaluation=dict(name=f'Full-{len(photos)}', photos=len(photos),
                        references=old['reference_count'],
                        unique_reference_images=old['current_evaluation']['unique_reference_images'],
                        source_groups=len(members), frozen_encoders=True, query_list='photos.csv',
                        historical_splits_used_for_evaluation=False,
                        status='pending_supplement_encoding_and_evaluation'),
                    previous_protocol=dict(name=old['name'], directory=rules['previous_lists'],
                        protocol_sha256=rules['previous_protocol_sha256'],
                        groups_sha256=checksum(previous / 'groups.csv')))
    protocol['source_sha256'] = dict(rules['source_sha256'])
    protocol['source_sha256']['lists/allocation.json'] = checksum(root / 'lists/allocation.json')
    for name in ('protocol.json', 'groups.csv'):
        protocol['source_sha256'][rules['previous_lists'] + '/' + name] = checksum(previous / name)
    protocol['output_sha256'] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    protocol['output_sha256']['positives.csv'] = checksum(root / 'lists/positives.csv')
    files['protocol.json'] = json_bytes(protocol)
    return files
