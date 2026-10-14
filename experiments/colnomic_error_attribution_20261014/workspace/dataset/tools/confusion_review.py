"""Human-reviewed design confusion pairs and a derived validation subset.

Selections are annotation metadata, never replacements for the formal gallery,
positive mappings, photo labels, or historical train/val/test assignments.
"""
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .photo_metadata import Conflict, read_csv, sha, write_json
else:
    from photo_metadata import Conflict, read_csv, sha, write_json

CATEGORIES = [
    dict(id='name_series', name='名称 / 系列相近', description='同品牌、同系列，或名称相近，需结合设计区分。'),
    dict(id='strength_count', name='规格 / 数量差异', description='剂量、浓度、容量、片数等不同，主体设计相近。'),
    dict(id='text_similar', name='文字内容相近', description='名称或说明文字高度重合，单靠文字容易混淆。'),
    dict(id='visual_layout', name='配色 / 版式相近', description='包装配色、图案、布局或外观相近。'),
    dict(id='source_variant', name='同来源设计变体', description='同来源的不同设计、包装形式或版本；具体关系需人工核实。'),
    dict(id='other', name='其他混淆原因', description='不属于以上类型，可在备注中说明。'),
]
CATEGORY_IDS = {v['id'] for v in CATEGORIES}
STATUSES = ('pending', 'confusing', 'not_confusing', 'same_design', 'unsure')
FINAL_STATUSES = {'confusing', 'not_confusing', 'same_design'}


def revision(record):
    return sha(json.dumps(record, sort_keys=True, ensure_ascii=False).encode())


def inventory_signature(root):
    return {name: sha((Path(root) / 'lists' / name).read_bytes())
            for name in ('photos.csv', 'references.csv', 'positives.csv', 'groups.csv')}


class ConfusionReview:
    def __init__(self, store):
        self.store = store
        self.path = store.root / 'annotations/confusion_metadata.json'
        self.candidate_path = store.root / 'review/confusion/candidates.json'
        self.inventory = inventory_signature(store.root)
        rows = read_csv(store.root / 'lists/references.csv')
        self.all_refs = {r['reference_id']: r for r in rows}
        self.positive = {r['package_id']: r['reference_id']
                         for r in read_csv(store.root / 'lists/positives.csv')}
        positive_ids = set(self.positive.values())
        groups = defaultdict(list)
        for row in rows:
            groups[row['sha256']].append(row['reference_id'])
        self.canonical, self.aliases, self.refs = {}, {}, {}
        for ids in groups.values():
            canonical = min(ids, key=lambda rid: (rid not in positive_ids, rid))
            self.refs[canonical] = self.all_refs[canonical]
            self.aliases[canonical] = sorted(ids)
            self.canonical.update({rid: canonical for rid in ids})
        self.positive = {p: self.canonical[r] for p, r in self.positive.items()}
        self.owners = defaultdict(list)
        for pid, rid in self.positive.items():
            self.owners[rid].append(pid)
        self.photos_by_package = defaultdict(list)
        for photo in store.photos.values():
            self.photos_by_package[photo['package_id']].append(photo['photo_id'])
        self.names = {}
        for pid in store.packages:
            package, _ = store.package(pid)
            names = list(dict.fromkeys(p['name'] for d in package.get('documents', []) for p in d.get('products', [])))
            self.names[pid] = ' / '.join(names) or package['name']
        self._catalog_stamp = None
        self._catalog = {}

    def catalog(self):
        stamp = self.candidate_path.stat().st_mtime_ns if self.candidate_path.exists() else None
        if stamp != self._catalog_stamp:
            data = json.loads(self.candidate_path.read_text()) if stamp else {}
            if data and (data.get('schema') != 'release.confusion_candidates.v1' or data.get('inventory') != self.inventory):
                raise ValueError('混淆推荐与当前数据清单不一致，请重新生成推荐。')
            self._catalog = data
            self._catalog_stamp = stamp
        return self._catalog

    def read(self):
        empty = dict(schema='release.confusion_metadata.v1', inventory=self.inventory,
                     categories=CATEGORIES, pairs={}, packages={}, designs={})
        if not self.path.exists():
            return empty
        data = json.loads(self.path.read_text())
        if data.get('schema') != empty['schema'] or data.get('inventory') != self.inventory:
            raise ValueError('混淆标注与当前数据清单不一致；请先核对图片 ID 与哈希。')
        return data

    def reference(self, rid):
        rid = self.canonical[rid]
        row = self.refs[rid]
        info = self.catalog().get('references', {}).get(rid, {})
        return dict(reference_id=rid, sha256=row['sha256'], title=info.get('title', row.get('identity', rid)),
                    category=row.get('category', ''), original_filename=row.get('original_filename', ''),
                    package_ids=self.owners[rid], photo_count=sum(len(self.photos_by_package[p]) for p in self.owners[rid]),
                    aliases=self.aliases[rid], ocr_preview=info.get('ocr_preview', ''),
                    image_url='/reference-image/' + rid, thumb_url='/reference-thumb/' + rid)

    def pair_key(self, left, right):
        left, right = sorted((self.canonical[left], self.canonical[right]))
        if left == right:
            raise ValueError('同一设计或字节相同的图片不能标为混淆对。')
        return left + '__' + right

    def decision(self, data, left, right):
        key = self.pair_key(left, right)
        record = data['pairs'].get(key)
        return dict(pair_id=key, revision=revision(record), status=record['status'] if record else 'pending',
                    categories=record['categories'] if record else [], note=record['note'] if record else '')

    def subset(self, data):
        confirmed = {k: v for k, v in data['pairs'].items() if v['status'] == 'confusing'}
        auto_packages, external, adjacency = set(), set(), defaultdict(set)
        for row in confirmed.values():
            a, b = row['reference_ids']
            adjacency[a].add(b); adjacency[b].add(a)
            for rid in (a, b):
                auto_packages.update(self.owners[rid])
                if not self.owners[rid]:
                    external.add(rid)
        selected = set(auto_packages)
        for pid, row in data['packages'].items():
            if row['selection'] == 'include': selected.add(pid)
            elif row['selection'] == 'exclude': selected.discard(pid)
        # A separate design shortlist can explicitly include or dismiss an extra design.
        for rid, row in data['designs'].items():
            if row['selection'] == 'include': external.add(rid)
            elif row['selection'] == 'exclude': external.discard(rid)
        external = {r for r in external if not self.owners[r]}
        components, visited = [], set()
        for first in sorted(adjacency):
            if first in visited: continue
            stack, component = [first], []
            while stack:
                rid = stack.pop()
                if rid in visited: continue
                visited.add(rid); component.append(rid); stack.extend(adjacency[rid] - visited)
            components.append(sorted(component))
        # Current photo exclusions are respected without changing the formal list.
        active = {p for p, row in self.store.records().items() if not row['excluded']}
        photos = [p for p, row in self.store.photos.items() if row['package_id'] in selected and p in active]
        return dict(package_ids=sorted(selected), photo_ids=photos,
                    positive_reference_ids=sorted({self.positive[p] for p in selected}),
                    confirmed_pair_ids=sorted(confirmed), connected_groups=components,
                    additional_design_reference_ids=sorted(external),
                    gallery='full references.csv; exact SHA duplicates collapsed',
                    query_rule='all retained photos of selected packages',
                    group_rule='connected components for grouping; only confirmed edges assert confusion',
                    split_policy='existing source-group splits unchanged; this is a curated subset, not a new held-out split')

    def proposals(self, pid, data):
        anchor = self.positive[pid]
        proposals = {p['reference_id']: p for p in self.catalog().get('packages', {}).get(pid, [])}
        for row in data['pairs'].values():
            if anchor in row['reference_ids']:
                other = next(r for r in row['reference_ids'] if r != anchor)
                proposals.setdefault(other, dict(reference_id=other, proposed_categories=[], evidence=['人工添加']))
        return proposals

    def review_progress(self, pid, data):
        anchor = self.positive[pid]
        statuses = Counter(data['pairs'].get(self.pair_key(anchor, rid), {}).get('status', 'pending')
                           for rid in self.proposals(pid, data))
        finalized = sum(statuses[s] for s in FINAL_STATUSES)
        return dict(total=sum(statuses.values()), finalized=finalized, pending=statuses['pending'],
                    unsure=statuses['unsure'], confusing=statuses['confusing'],
                    not_confusing=statuses['not_confusing'], same_design=statuses['same_design'],
                    saved=finalized + statuses['unsure'])

    def summary(self, data=None):
        if data is None:
            with self.store.locked(): data = self.read()
        subset = self.subset(data)
        package_rows = []
        selected = set(subset['package_ids'])
        for pid in self.store.packages:
            record = data['packages'].get(pid)
            progress = self.review_progress(pid, data)
            package_rows.append(dict(package_id=pid, name=self.names[pid], reference_id=self.positive[pid],
                photo_count=len(self.photos_by_package[pid]), included=pid in selected,
                selection=record['selection'] if record else 'auto', revision=revision(record),
                reviewed=bool(record and record.get('reviewed')), note=record.get('note', '') if record else '',
                proposal_count=progress['total'], review_progress=progress))
        counts = Counter(p['status'] for p in data['pairs'].values())
        return dict(categories=CATEGORIES, packages=package_rows,
                    counts=dict(packages=len(selected), photos=len(subset['photo_ids']),
                                additional_designs=len(subset['additional_design_reference_ids']),
                                reviewed_pairs=sum(v for k, v in counts.items() if k != 'pending'),
                                confirmed_pairs=counts['confusing'], not_confusing_pairs=counts['not_confusing'],
                                same_design_pairs=counts['same_design'], unsure_pairs=counts['unsure'],
                                completed_packages=sum(p['review_progress']['total'] > 0 and
                                    p['review_progress']['finalized'] == p['review_progress']['total'] for p in package_rows),
                                reviewed_packages=sum(p['reviewed'] for p in package_rows)),
                    subset=subset, proposals_ready=bool(self.catalog()),
                    reference_rows=len(self.all_refs), unique_references=len(self.refs),
                    metadata_path='dataset/annotations/confusion_metadata.json')

    def package(self, pid):
        anchor = self.positive[pid]
        with self.store.locked(): data = self.read()
        # Always keep human-reviewed pairs visible after a proposal rebuild.
        proposals = self.proposals(pid, data)
        candidates = [dict(reference=self.reference(rid), proposal=p, decision=self.decision(data, anchor, rid),
                           design=self.design_state(data, rid)) for rid, p in proposals.items()]
        return dict(package_id=pid, anchor=self.reference(anchor), candidates=candidates,
                    photos=[dict(photo_id=p, thumb_url='/thumb/' + p) for p in self.photos_by_package[pid]])

    def design_state(self, data, rid):
        rid = self.canonical[rid]
        row = data['designs'].get(rid)
        auto = any(v['status'] == 'confusing' and rid in v['reference_ids'] for v in data['pairs'].values())
        choice = row['selection'] if row else 'auto'
        return dict(selection=choice, included=choice == 'include' or (choice == 'auto' and auto),
                    revision=revision(row), note=row.get('note', '') if row else '')

    def browse(self, query='', scope='all', offset=0, limit=36, package_id=None, status='all'):
        if scope not in ('all', 'existing', 'other', 'selected'): raise ValueError('Invalid gallery scope')
        if status not in ('all', 'finalized', *STATUSES): raise ValueError('Invalid review status')
        if offset < 0 or not 1 <= limit <= 72 or len(query) > 200: raise ValueError('Invalid pagination or query')
        anchor = self.positive[package_id] if package_id else None
        with self.store.locked(): data = self.read()
        words = query.casefold().split()
        selected = set(self.subset(data)['additional_design_reference_ids']) if scope == 'selected' else set()
        rows, review_counts = [], Counter()
        catalog = self.catalog().get('references', {})
        for rid in sorted(self.refs):
            if scope == 'existing' and not self.owners[rid]: continue
            if scope == 'other' and self.owners[rid]: continue
            if scope == 'selected' and rid not in selected: continue
            row = self.refs[rid]; info = catalog.get(rid, {})
            haystack = ' '.join([rid, *self.aliases[rid], *self.owners[rid], row.get('identity', ''),
                                 info.get('title', ''), info.get('search_text', '')]).casefold()
            if not all(word in haystack for word in words): continue
            current_status = data['pairs'].get(self.pair_key(anchor, rid), {}).get('status', 'pending') if anchor and rid != anchor else 'self'
            review_counts[current_status] += 1
            if status == 'all' or current_status == status or (status == 'finalized' and current_status in FINAL_STATUSES):
                rows.append(rid)
        items = []
        for rid in rows[offset:offset+limit]:
            items.append(dict(reference=self.reference(rid), design=self.design_state(data, rid),
                              decision=self.decision(data, anchor, rid) if anchor and anchor != rid else None))
        return dict(items=items, total=len(rows), offset=offset, limit=limit, review_counts=dict(review_counts))

    def save(self, data):
        data['updated_at'] = datetime.now(timezone.utc).isoformat()
        data['validation_subset'] = self.subset(data)
        write_json(self.path, data)

    def update_pair(self, body):
        if not isinstance(body, dict) or set(body) != {'left', 'right', 'revision', 'status', 'categories', 'note'}:
            raise ValueError('Expected left, right, revision, status, categories and note')
        if body['status'] not in STATUSES: raise ValueError('Invalid pair status')
        cats = body['categories']
        if not isinstance(cats, list) or any(not isinstance(c, str) or c not in CATEGORY_IDS for c in cats):
            raise ValueError('Invalid confusion categories')
        if body['status'] == 'confusing' and not cats: raise ValueError('请至少选择一种混淆原因。')
        self.validate_note(body)
        key = self.pair_key(body['left'], body['right'])
        refs = key.split('__')
        if not any(self.owners[r] for r in refs): raise ValueError('混淆对至少需包含一个已有实拍的包装。')
        with self.store.locked():
            data = self.read()
            if revision(data['pairs'].get(key)) != body['revision']:
                raise Conflict('这对设计已在其他窗口修改，请重新读取后再保存。')
            data['pairs'][key] = dict(reference_ids=refs, image_sha256=[self.refs[r]['sha256'] for r in refs],
                status=body['status'], categories=sorted(set(cats)), note=body['note'].strip(),
                source='human', updated_at=datetime.now(timezone.utc).isoformat())
            self.save(data)
            return self.decision(data, *refs)

    @staticmethod
    def validate_note(body):
        if not isinstance(body.get('revision'), str) or not isinstance(body.get('note'), str) or len(body['note']) > 2000:
            raise ValueError('Invalid revision or note')

    def update_package(self, pid, body):
        if pid not in self.store.packages: raise KeyError(pid)
        if not isinstance(body, dict) or set(body) != {'revision', 'selection', 'reviewed', 'note'}:
            raise ValueError('Expected revision, selection, reviewed and note')
        if body['selection'] not in ('auto', 'include', 'exclude') or type(body['reviewed']) is not bool:
            raise ValueError('Invalid package selection')
        self.validate_note(body)
        with self.store.locked():
            data = self.read()
            if revision(data['packages'].get(pid)) != body['revision']:
                raise Conflict('这个包装已在其他窗口修改，请重新读取后再保存。')
            data['packages'][pid] = dict(selection=body['selection'], reviewed=body['reviewed'], note=body['note'].strip(),
                                        updated_at=datetime.now(timezone.utc).isoformat())
            self.save(data)
            return self.summary(data)

    def update_design(self, rid, body):
        rid = self.canonical[rid]
        if self.owners[rid]: raise ValueError('已有实拍包装请使用验证子集选项。')
        if not isinstance(body, dict) or set(body) != {'revision', 'selection', 'note'}:
            raise ValueError('Expected revision, selection and note')
        if body['selection'] not in ('auto', 'include', 'exclude'): raise ValueError('Invalid design selection')
        self.validate_note(body)
        with self.store.locked():
            data = self.read()
            if revision(data['designs'].get(rid)) != body['revision']:
                raise Conflict('这个设计的选择已变化，请重新读取后再保存。')
            data['designs'][rid] = dict(selection=body['selection'], note=body['note'].strip(),
                image_sha256=self.refs[rid]['sha256'], updated_at=datetime.now(timezone.utc).isoformat())
            self.save(data)
            return self.design_state(data, rid)

    def export(self):
        with self.store.locked():
            data = self.read()
            data['validation_subset'] = self.subset(data)
            return data
