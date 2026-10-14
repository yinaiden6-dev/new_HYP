"""Analyze every retrieval configuration; review each (photo, wrong design) once.

Only saved predictions and manual review are used. No model or network is called.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
from functools import cached_property
import hashlib
import json
from pathlib import Path
import threading
import uuid

ROOT = Path(__file__).resolve().parents[2]
MAIN = {
    'E0': '数据／评测错误', 'E1': '参考缺失', 'E2': '候选召回失败（未进 Top-10）',
    'E3': '排序失败', 'E4': '文字误伤', 'E5': '视觉误伤', 'E6': '融合构造异常', 'E7': '未确定',
}
CONFUSION = {
    'C0': '实际同一设计／重复参考', 'C1': '同产品不同包装版本',
    'C2': '同品牌同系列不同口味／剂量／规格／数量', 'C3': '同品牌不同产品',
    'C4': '不同品牌同类别', 'C5': '不同品牌但外观／文字／形状相似',
    'C6': '无明显关联', 'C7': '无法判断',
}
CAUSES = {
    '图像质量': ['模糊', '反光', '过曝', '低分辨率', '极端透视', '旋转'],
    '可见性': ['目标过小', '截断', '遮挡', '侧面/背面', '多个商品'],
    '场景干扰': ['货架背景', '价签', '邻近商品', '背景文字'],
    '细粒度与文字': ['区分区域过小', 'OCR漏字', 'OCR错字', '规格数字混淆', '通用词主导'],
    '输入处理': ['裁剪丢失信息', '框过松', 'SAM漏分割', '白底域偏移', '图文证据冲突'],
}
FACES = {'front': '正面', 'back': '背面', 'side': '侧面', 'top_bottom': '顶面／底面', 'unclear': '无法判断'}
CLUES = {'present': '出现', 'absent': '未出现', 'uncertain': '无法判断'}
READABILITY = {'readable': '可读', 'partial': '部分可读', 'unreadable': '不可读', 'not_present': '线索未出现', 'uncertain': '无法判断'}
PATTERNS = ['ALL_FAIL', 'VISUAL_ONLY', 'TEXT_ONLY', 'FUSION_RESCUE', 'FUSION_BREAK',
            'PREPROCESS_SENSITIVE', 'MODEL_SPECIFIC', 'UNSTABLE_CONFUSER']
TRANSITIONS = ['RESCUE', 'BREAK', 'WRONG_HOLD', 'WRONG_SWITCH', 'CORRECT_HOLD']
FUSIONS = {
    'B10': ('B01', 'B06'), 'B11': ('B02', 'B07'), 'B12': ('B03', 'B08'),
    'B13': ('B04', 'B09'), 'B26': ('B24', 'B25'), 'B23': ('B04', 'B20'),
    'F01': ('B01', 'B05'), 'F02': ('B02', 'B05'), 'F03': ('B03', 'B05'),
    'F04': ('B04', 'B05'), 'F05': ('B24', 'B05'),
}


def controlled_comparisons():
    edges = []
    def add(family, pairs):
        edges.extend(dict(family=family, baseline=a, variant=b) for a, b in pairs)
    add('图像模型差异', [('B03', b) for b in ['B01', 'B02', 'B04', 'B21', 'B22', 'B24']])
    add('OCR查询', [('B01', 'B06'), ('B02', 'B07'), ('B03', 'B08'), ('B04', 'B09'), ('B24', 'B25')])
    add('图像与OCR融合', [(a, b) for b in ['B10', 'B11', 'B12', 'B13', 'B26'] for a in FUSIONS[b]])
    add('CRISP', [('B01', 'B14'), ('B02', 'B15'), ('B03', 'B16'), ('B06', 'B17'),
                  ('B07', 'B18'), ('B08', 'B19'), ('B24', 'B27'), ('B25', 'B28')])
    add('OCR→OCR', [('B04', 'B20'), ('B04', 'B23'), ('B20', 'B23')])
    add('裁剪与分割', [('B03', 'D01'), ('B03', 'D03')])
    add('BM25融合', [(a, b) for b in ['F01', 'F02', 'F03', 'F04', 'F05'] for a in FUSIONS[b]])
    return edges


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise')
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def rank_bucket(rank):
    if rank is None:
        return 'missing'
    if rank == 1:
        return '1'
    if rank <= 5:
        return '2-5'
    if rank <= 10:
        return '6-10'
    return '>10'


def transition(left, right):
    if left['correct'] and right['correct']:
        return 'CORRECT_HOLD'
    if not left['correct'] and right['correct']:
        return 'RESCUE'
    if left['correct'] and not right['correct']:
        return 'BREAK'
    return 'WRONG_HOLD' if left['selected_design_id'] == right['selected_design_id'] else 'WRONG_SWITCH'


class Conflict(ValueError):
    pass


class Analysis:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.output = self.root / 'results/error_analysis'
        self.review = self.root / 'dataset/review/error_analysis'
        self.review.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()
        paths = [self.root / 'results/all_results.csv']
        paths += [self.root / f'dataset/lists/{name}.csv' for name in ['photos', 'references', 'positives', 'groups']]
        self.configs = {r['id']: r for r in read_csv(paths[0])}
        self.photos = {r['photo_id']: r for r in read_csv(paths[1])}
        self.refs = {r['reference_id']: r for r in read_csv(paths[2])}
        self.positives = defaultdict(set)
        for row in read_csv(paths[3]):
            self.positives[row['package_id']].add(row['reference_id'])
        sha_designs = defaultdict(set)
        self.design_refs = defaultdict(list)
        for r in self.refs.values():
            sha_designs[r['sha256']].add(r['design_id'])
            self.design_refs[r['design_id']].append(r['reference_id'])
        if any(len(ids) != 1 for ids in sha_designs.values()) or len(sha_designs) != len(self.design_refs):
            raise ValueError('Design IDs must match the SHA-unique gallery used for evaluation')
        positive_ids = set().union(*self.positives.values())
        self.canonical = {design: min(ids, key=lambda r: (r not in positive_ids, r)) for design, ids in self.design_refs.items()}
        self.predictions = {}
        self.processed = {}
        for cid, cfg in self.configs.items():
            prediction_path = self.safe_path(cfg['result'] + '/predictions.jsonl')
            config_path = self.safe_path(cfg['result'] + '/config.json')
            paths.extend([prediction_path, config_path])
            rows = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
            pred = {r['photo_id']: r for r in rows}
            if len(pred) != len(rows) or set(pred) != set(self.photos):
                raise ValueError('Incomplete or duplicate predictions: ' + cid)
            for pid, p in pred.items():
                rid = p.get('selected_reference_id')
                p['selected_design_id'] = self.refs.get(rid, {}).get('design_id')
                positive = self.positives[self.photos[pid]['package_id']]
                p['positive_design_ids'] = sorted({self.refs[r]['design_id'] for r in positive})
                consistent = (set(p.get('positive_reference_ids', [])) == positive
                    and p.get('package_id') == self.photos[pid]['package_id']
                    and p['correct'] == (p['selected_design_id'] in p['positive_design_ids'])
                    and ((p.get('rank') == 1) == p['correct']))
                p['data_issue'] = not consistent
                if p.get('rank') is not None and (type(p['rank']) is not int or p['rank'] < 1):
                    p['data_issue'] = True
            errors = sum(not p['correct'] for p in pred.values())
            if errors != round(len(self.photos) * (1 - float(cfg['recall_at_1']))):
                raise ValueError('Predictions differ from result summary: ' + cid)
            self.predictions[cid] = pred
            config = json.loads(config_path.read_text())
            if config.get('region_manifest'):
                path = self.safe_path(config['region_manifest'])
                paths.append(path)
                self.processed[cid] = {r['photo_id']: r for r in json.loads(path.read_text())['photos']}
        self.inputs = {str(p.relative_to(self.root)): file_hash(p) for p in paths}
        self.signature = digest(self.inputs)
        self.edges = [e for e in controlled_comparisons() if e['baseline'] in self.configs and e['variant'] in self.configs]
        self.baselines = {cid: next((e['baseline'] for e in self.edges if e['variant'] == cid), None) for cid in self.configs}
        self.cases = {}
        self.photo_cases = defaultdict(list)
        for cid, predictions in self.predictions.items():
            for pid, p in predictions.items():
                if p['correct']:
                    continue
                design = p['selected_design_id'] or 'missing_reference'
                key = pid + '__' + design
                case = self.cases.setdefault(key, dict(case_id=key, photo_id=pid,
                    wrong_design_id=design, wrong_reference_id=self.canonical.get(design),
                    correct_design_ids=p['positive_design_ids'],
                    correct_reference_ids=sorted(self.positives[self.photos[pid]['package_id']]), configurations=[]))
                case['configurations'].append(cid)
        for case in self.cases.values():
            case['configurations'].sort()
            case['primary_config'] = 'F03' if 'F03' in case['configurations'] else (
                'B03' if 'B03' in case['configurations'] else case['configurations'][0])
            self.photo_cases[case['photo_id']].append(case['case_id'])
        self.patterns = {pid: self.photo_patterns(pid) for pid in self.photos}
        self.annotation_path = self.review / 'annotations.json'
        with self.locked():
            if self.annotation_path.exists():
                self._read_annotations()
            else:
                write_json(self.annotation_path, dict(schema='drugpack.unique_error_review.v1',
                    input_signature=self.signature, labels={}))

    def safe_path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Path outside project')
        return path

    @contextmanager
    def locked(self):
        with self._thread_lock, (self.review / '.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _read_annotations(self):
        data = json.loads(self.annotation_path.read_text())
        if data['input_signature'] != self.signature:
            raise ValueError('Predictions or inventories changed; preserve and migrate the previous manual review before continuing')
        return data

    def annotations(self):
        with self.locked():
            return self._read_annotations()['labels']

    def auto_main(self, cid, pid):
        p = self.predictions[cid][pid]
        if p['correct']:
            return None
        if p['data_issue']:
            return 'E0'
        if not p['positive_design_ids']:
            return 'E1'
        if cid in FUSIONS:
            image, text = FUSIONS[cid]
            if image in self.predictions and text in self.predictions:
                a, b = self.predictions[image][pid], self.predictions[text][pid]
                if a['data_issue'] or b['data_issue']:
                    return 'E7'
                if a['correct'] and b['correct']:
                    return 'E6'
                if a['correct'] and not b['correct']:
                    return 'E4'
                if b['correct'] and not a['correct']:
                    return 'E5'
        rank = p.get('rank')
        if rank is None or not p['selected_design_id']:
            return 'E7'
        return 'E2' if rank > 10 else 'E3'

    def photo_patterns(self, pid):
        rows = {cid: p[pid] for cid, p in self.predictions.items()}
        correct = [cid for cid, p in rows.items() if p['correct']]
        visual = {cid for cid, c in self.configs.items() if c['mode'] in ('image', 'image_crisp', 'manual', 'sam3')}
        text = {cid for cid, c in self.configs.items() if c['mode'] in ('bm25', 'ocr_to_image', 'ocr_to_image_crisp', 'ocr_to_ocr')}
        rescue, broken = [], []
        for fused, (a, b) in FUSIONS.items():
            if not {fused, a, b} <= rows.keys():
                continue
            if rows[fused]['correct'] and not rows[a]['correct'] and not rows[b]['correct']:
                rescue.append(fused)
            if not rows[fused]['correct'] and (rows[a]['correct'] or rows[b]['correct']):
                broken.append(fused)
        wrong = Counter(p['selected_design_id'] or 'missing_reference' for p in rows.values() if not p['correct'])
        families = sorted({self.configs[c]['model'] for c in correct})
        tags = []
        if not correct:
            tags.append('ALL_FAIL')
        if correct and set(correct) <= visual:
            tags.append('VISUAL_ONLY')
        if correct and set(correct) <= text:
            tags.append('TEXT_ONLY')
        if rescue:
            tags.append('FUSION_RESCUE')
        if broken:
            tags.append('FUSION_BREAK')
        if {'B03', 'D01', 'D03'} <= rows.keys() and len({(rows[c]['correct'], rows[c]['selected_design_id']) for c in ['B03', 'D01', 'D03']}) > 1:
            tags.append('PREPROCESS_SENSITIVE')
        if len(families) == 1:
            tags.append('MODEL_SPECIFIC')
        if len(wrong) > 1:
            tags.append('UNSTABLE_CONFUSER')
        return dict(tags=tags, correct_configs=correct, correct_models=families,
            wrong_design_counts=dict(wrong), fusion_rescue=rescue, fusion_break=broken,
            top1_design_counts=dict(Counter(p['selected_design_id'] or 'missing_reference' for p in rows.values())))

    def summary(self, labels=None):
        labels = self.annotations() if labels is None else labels
        configs = []
        for cid, cfg in self.configs.items():
            cases = [c for c in self.cases.values() if cid in c['configurations']]
            configs.append(dict(id=cid, model=cfg['model'], mode=cfg['mode'], errors=len(cases),
                reviewed=sum(c['case_id'] in labels and labels[c['case_id']]['confirmed'] for c in cases)))
        return dict(queries=len(self.photos), configurations=configs,
            prediction_rows=len(self.photos) * len(self.configs), error_occurrences=sum(len(c['configurations']) for c in self.cases.values()),
            unique_cases=len(self.cases), error_photos=len(self.photo_cases),
            confirmed=sum(v['confirmed'] for v in labels.values()),
            controlled_comparisons=len(self.edges), pattern_counts={t: sum(t in p['tags'] for p in self.patterns.values()) for t in PATTERNS},
            taxonomy=dict(main=MAIN, confusion=CONFUSION, causes=CAUSES, faces=FACES, clues=CLUES, readability=READABILITY),
            input_signature=self.signature)

    def browse(self, config='F03', status='pending', pattern='', query=''):
        if config != 'all' and config not in self.configs:
            raise ValueError('Unknown configuration')
        if status not in ('all', 'pending', 'confirmed') or pattern not in ['', *PATTERNS]:
            raise ValueError('Invalid filter')
        labels = self.annotations()
        result = []
        for key, case in self.cases.items():
            if config != 'all' and config not in case['configurations']:
                continue
            confirmed = labels.get(key, {}).get('confirmed', False)
            if status == 'pending' and confirmed or status == 'confirmed' and not confirmed:
                continue
            if pattern and pattern not in self.patterns[case['photo_id']]['tags']:
                continue
            if query and query.lower() not in (' '.join([key, self.photos[case['photo_id']]['package_id'],
                    self.photos[case['photo_id']].get('identity', ''), self.refs.get(case['wrong_reference_id'], {}).get('identity', '')])).lower():
                continue
            result.append(dict(case_id=key, photo_id=case['photo_id'], wrong_design_id=case['wrong_design_id'],
                configurations=case['configurations'], confirmed=confirmed))
        result.sort(key=lambda c: ('F03' not in c['configurations'], -len(c['configurations']), c['case_id']))
        return result

    @cached_property
    def ocr(self):
        text = {}
        for population in ['photos', 'references']:
            path = self.root / f'cache/ocr/{population}.json'
            if path.exists():
                text.update({k: v.get('text', '') for k, v in json.loads(path.read_text()).items()})
        return text

    def detail(self, key):
        case = self.cases[key]
        pid = case['photo_id']
        labels = self.annotations()
        label = labels.get(key, {})
        rows = []
        for cid, predictions in self.predictions.items():
            p = predictions[pid]
            baseline = self.baselines[cid]
            same = cid in case['configurations']
            rows.append(dict(id=cid, model=self.configs[cid]['model'], mode=self.configs[cid]['mode'],
                correct=p['correct'], rank=p.get('rank'), selected_reference_id=p.get('selected_reference_id'),
                selected_design_id=p['selected_design_id'], same_wrong_pair=same,
                auto_main=self.auto_main(cid, pid),
                main=label.get('main_by_config', {}).get(cid, self.auto_main(cid, pid)) if same else self.auto_main(cid, pid),
                baseline=baseline, transition=transition(self.predictions[baseline][pid], p) if baseline else None,
                channels=list(FUSIONS.get(cid, ())), processed=cid in self.processed))
        references = [dict(self.refs[r], ocr=self.ocr.get(r, '')) for r in case['correct_reference_ids']]
        wrong = dict(self.refs[case['wrong_reference_id']], ocr=self.ocr.get(case['wrong_reference_id'], '')) if case['wrong_reference_id'] else None
        comparisons = [dict(e, transition=transition(self.predictions[e['baseline']][pid], self.predictions[e['variant']][pid])) for e in self.edges]
        return dict(**case, photo=dict(self.photos[pid], ocr=self.ocr.get(pid, '')), wrong=wrong,
            correct=references, predictions=rows, comparisons=comparisons, patterns=self.patterns[pid],
            annotation=label, revision=label.get('revision', 0),
            other_cases=self.photo_cases[pid],
            processed={cid: m[pid] for cid, m in self.processed.items()})

    def validate_label(self, key, label):
        case = self.cases[key]
        if type(label.get('confirmed')) is not bool:
            raise ValueError('Invalid confirmation')
        confirmed = label['confirmed']
        if label.get('confusion') not in CONFUSION and not (not confirmed and label.get('confusion') == ''):
            raise ValueError('请选择一个混淆关系')
        faces = label.get('visible_faces')
        if not isinstance(faces, list) or (confirmed and not faces) or len(faces) != len(set(faces)) or not set(faces) <= FACES.keys() or ('unclear' in faces and len(faces) > 1):
            raise ValueError('请选择可见面；无法判断不能与其他面并选')
        if (label.get('clue_presence') not in CLUES and not (not confirmed and label.get('clue_presence') == '')) or (label.get('clue_readability') not in READABILITY and not (not confirmed and label.get('clue_readability') == '')):
            raise ValueError('请选择区分线索是否出现、是否可读')
        if (label['clue_presence'] == 'absent') != (label['clue_readability'] == 'not_present') and (confirmed or all([label['clue_presence'], label['clue_readability']])):
            raise ValueError('线索未出现时，可读性应选择线索未出现')
        common = set().union(*(set(v) for k, v in CAUSES.items() if k != '输入处理'))
        causes = label.get('causes')
        if not isinstance(causes, list) or len(causes) != len(set(causes)) or not set(causes) <= common:
            raise ValueError('Invalid apparent cause labels')
        config_causes = label.get('config_causes', {})
        if not isinstance(config_causes, dict) or not config_causes.keys() <= set(case['configurations']):
            raise ValueError('Invalid configuration-specific causes')
        for cid, values in config_causes.items():
            if not isinstance(values, list) or len(values) != len(set(values)) or not set(values) <= set(CAUSES['输入处理']):
                raise ValueError('Invalid input-processing cause')
            allowed = {'图文证据冲突'} if cid in FUSIONS else set()
            if cid == 'D01':
                allowed |= {'裁剪丢失信息', '框过松'}
            if cid == 'D03':
                allowed |= {'SAM漏分割', '白底域偏移'}
            if not set(values) <= allowed:
                raise ValueError('输入处理标签与配置不符: ' + cid)
        mains = label.get('main_by_config')
        if not isinstance(mains, dict) or mains.keys() != set(case['configurations']) or not set(mains.values()) <= MAIN.keys():
            raise ValueError('每个相关配置都需要一个主错误类型')
        for cid, value in mains.items():
            auto = self.auto_main(cid, case['photo_id'])
            if value not in ('E0', 'E1', 'E7', auto):
                raise ValueError('主类型不符合该配置的排名或融合优先级: ' + cid)
        note = label.get('note', '')
        if not isinstance(note, str) or len(note) > 4000 or type(label.get('confirmed')) is not bool:
            raise ValueError('Invalid note or confirmation')
        return {k: label[k] for k in ['confusion', 'visible_faces', 'clue_presence', 'clue_readability',
                'causes', 'main_by_config', 'confirmed']} | dict(config_causes=config_causes, note=note)

    def save(self, key, label, revision):
        clean = self.validate_label(key, label)
        with self.locked():
            self.verify_inputs()
            data = self._read_annotations()
            old = data['labels'].get(key, {})
            if type(revision) is not int or revision != old.get('revision', 0):
                raise Conflict('该案例已在另一个页面修改，请刷新后再保存')
            clean.update(revision=revision + 1, updated_utc=datetime.now(timezone.utc).isoformat(),
                primary_config=self.cases[key]['primary_config'],
                main_error_type=clean['main_by_config'][self.cases[key]['primary_config']], actor='human')
            data['labels'][key] = clean
            write_json(self.annotation_path, data)
            self._export_labels(data['labels'])
        return clean

    def classified(self, cid, pid, labels):
        p = self.predictions[cid][pid]
        key = pid + '__' + (p['selected_design_id'] or 'missing_reference')
        label = labels.get(key, {})
        confirmed = label.get('confirmed', False)
        main = label.get('main_by_config', {}).get(cid, self.auto_main(cid, pid))
        return dict(config_id=cid, photo_id=pid, case_id=key, wrong_design_id=p['selected_design_id'],
            wrong_reference_id=p.get('selected_reference_id'), correct_rank=p.get('rank'),
            rank_bucket=rank_bucket(p.get('rank')), auto_main_type=self.auto_main(cid, pid),
            main_type=main, main_type_source='human_confirmed' if confirmed else ('human_draft' if label else 'rule_suggestion'),
            annotation_status='confirmed' if confirmed else 'pending', confusion=label.get('confusion', ''),
            visible_faces='|'.join(label.get('visible_faces', [])), clue_presence=label.get('clue_presence', ''),
            clue_readability=label.get('clue_readability', ''), apparent_causes='|'.join(label.get('causes', [])),
            config_causes='|'.join(label.get('config_causes', {}).get(cid, [])), note=label.get('note', ''))

    def _export_labels(self, labels):
        classified = [self.classified(cid, pid, labels) for cid, pp in self.predictions.items() for pid, p in pp.items() if not p['correct']]
        if classified:
            write_csv(self.output / 'classified_errors.csv', classified, list(classified[0]))
            write_csv(self.output / 'F03_errors.csv', [r for r in classified if r['config_id'] == 'F03'], list(classified[0]))
        unique = []
        for key, c in self.cases.items():
            label = labels.get(key, {})
            unique.append(dict(case_id=key, photo_id=c['photo_id'], wrong_design_id=c['wrong_design_id'],
                wrong_reference_id=c['wrong_reference_id'], correct_reference_ids='|'.join(c['correct_reference_ids']),
                configurations='|'.join(c['configurations']), occurrences=len(c['configurations']),
                primary_config=c['primary_config'],
                main_type=label.get('main_error_type', self.auto_main(c['primary_config'], c['photo_id'])),
                annotation_status='confirmed' if label.get('confirmed') else 'pending',
                confusion=label.get('confusion', ''), visible_faces='|'.join(label.get('visible_faces', [])),
                clue_presence=label.get('clue_presence', ''), clue_readability=label.get('clue_readability', ''),
                apparent_causes='|'.join(label.get('causes', [])), config_causes=json.dumps(label.get('config_causes', {}), ensure_ascii=False),
                main_types=json.dumps(label.get('main_by_config', {}), ensure_ascii=False), note=label.get('note', '')))
        if unique:
            write_csv(self.output / 'unique_errors.csv', unique, list(unique[0]))
        summaries = []
        for cid, cfg in self.configs.items():
            selected = [r for r in classified if r['config_id'] == cid]
            automatic = Counter(r['auto_main_type'] for r in selected)
            final = Counter(r['main_type'] for r in selected if r['annotation_status'] == 'confirmed')
            provisional = Counter(r['main_type'] for r in selected)
            buckets = Counter(rank_bucket(p.get('rank')) for p in self.predictions[cid].values())
            row = dict(config_id=cid, model=cfg['model'], mode=cfg['mode'], queries=len(self.photos),
                correct=sum(p['correct'] for p in self.predictions[cid].values()), errors=len(selected),
                reviewed=sum(r['annotation_status'] == 'confirmed' for r in selected),
                baseline_id=self.baselines[cid] or '', recall_at_1=cfg['recall_at_1'])
            row.update({f'auto_{e}': automatic[e] for e in MAIN})
            row.update({f'confirmed_{e}': final[e] for e in MAIN})
            row.update({f'provisional_{e}': provisional[e] for e in MAIN})
            row.update({f'rank_{b}': buckets[b] for b in ['1', '2-5', '6-10', '>10', 'missing']})
            for t in TRANSITIONS:
                row[t] = sum(transition(self.predictions[self.baselines[cid]][pid], p) == t for pid, p in self.predictions[cid].items()) if self.baselines[cid] else ''
            summaries.append(row)
        write_csv(self.output / 'configuration_summary.csv', summaries, list(summaries[0]))
        write_json(self.output / 'review_progress.json', self.summary(labels))

    def verify_inputs(self):
        if any(file_hash(self.safe_path(name)) != value for name, value in self.inputs.items()):
            raise ValueError('Source files changed during analysis')

    def export(self):
        self.output.mkdir(parents=True, exist_ok=True)
        # Pin source files again so an export cannot silently mix evaluation versions.
        self.verify_inputs()
        with self.locked():
            labels = self._read_annotations()['labels']
            self._export_labels(labels)
            predictions = []
            for cid, pp in self.predictions.items():
                for pid, p in pp.items():
                    baseline = self.baselines[cid]
                    predictions.append(dict(config_id=cid, photo_id=pid, package_id=p['package_id'], correct=p['correct'],
                        correct_rank=p.get('rank'), rank_bucket=rank_bucket(p.get('rank')), auto_main_type=self.auto_main(cid, pid),
                        selected_reference_id=p.get('selected_reference_id'), selected_design_id=p['selected_design_id'],
                        positive_reference_ids='|'.join(p.get('positive_reference_ids', [])),
                        baseline_id=baseline or '', transition=transition(self.predictions[baseline][pid], p) if baseline else '',
                        photo_patterns='|'.join(self.patterns[pid]['tags']),
                        correct_configuration_count=len(self.patterns[pid]['correct_configs']),
                        top1_agreement_count=self.patterns[pid]['top1_design_counts'].get(p['selected_design_id'], 0),
                        wrong_design_count=len(self.patterns[pid]['wrong_design_counts']), data_issue=p['data_issue']))
            write_csv(self.output / 'all_predictions.csv', predictions, list(predictions[0]))
            transitions, comparisons = [], []
            for e in self.edges:
                counts = Counter()
                for pid in self.photos:
                    a, b = self.predictions[e['baseline']][pid], self.predictions[e['variant']][pid]
                    t = transition(a, b)
                    counts[t] += 1
                    transitions.append(dict(**e, photo_id=pid, transition=t, baseline_correct=a['correct'], variant_correct=b['correct'],
                        baseline_rank=a.get('rank'), variant_rank=b.get('rank'),
                        baseline_design_id=a['selected_design_id'], variant_design_id=b['selected_design_id']))
                comparisons.append(dict(**e, queries=len(self.photos), **{t: counts[t] for t in TRANSITIONS},
                    net_rescue=counts['RESCUE']-counts['BREAK'],
                    recall_delta=(counts['RESCUE']-counts['BREAK'])/len(self.photos)))
            if transitions:
                write_csv(self.output / 'controlled_transitions.csv', transitions, list(transitions[0]))
                write_csv(self.output / 'controlled_comparisons.csv', comparisons, list(comparisons[0]))
            patterns = [dict(photo_id=pid, tags='|'.join(p['tags']), correct_configurations='|'.join(p['correct_configs']),
                correct_models='|'.join(p['correct_models']), wrong_design_count=len(p['wrong_design_counts']),
                fusion_rescue='|'.join(p['fusion_rescue']), fusion_break='|'.join(p['fusion_break']),
                top1_design_counts=json.dumps(p['top1_design_counts'], sort_keys=True)) for pid, p in self.patterns.items()]
            write_csv(self.output / 'cross_model_cases.csv', patterns, list(patterns[0]))
            write_json(self.output / 'manifest.json', dict(schema='drugpack.error_analysis.v1',
                created_utc=datetime.now(timezone.utc).isoformat(), inputs=self.inputs,
                **self.summary(labels), priority='E0/E1 → E4/E5/E6 → E2/E3 → E7',
                fusion_channels={cid: list(channels) for cid, channels in FUSIONS.items()},
                notes=['No AI calls. Automatic labels are pending human review.',
                    'All ranks are one-based over SHA-unique designs, using full-gallery scoring.',
                    'E2 means outside Top-10, not absent from a scored candidate shortlist.',
                    'E4/E5/E6 generalize the F03 controlled-channel rules to each explicitly listed fusion.',
                    'WRONG_SWITCH is not a rescue; controlled comparisons are directional.',
                    'MODEL_SPECIFIC uses the model column, including its fusion variants.',
                    'PREPROCESS_SENSITIVE compares top-1 correctness and selected design, not ranks.',
                    'Visible tags describe observations or possible reasons, not proven internal causality.']))
            self._write_report(comparisons, labels)
        return self.summary(labels)

    def _write_report(self, comparisons, labels):
        summary = self.summary(labels)
        f03 = [self.classified('F03', pid, labels) for pid, p in self.predictions.get('F03', {}).items() if not p['correct']]
        counts = Counter(r['auto_main_type'] for r in f03)
        lines = ['# 检索错误分析', '',
            f"共 {len(self.configs)} 组 × {len(self.photos)} 张照片 = {summary['prediction_rows']:,} 条预测；"
            f"{summary['error_occurrences']:,} 次错误，去重为 {summary['unique_cases']:,} 个 `(photo_id, wrong_design_id)` 错误对，涉及 {summary['error_photos']:,} 张照片。", '',
            '## 主系统 F03', '',
            f"F03 共 {len(f03)} 个错误对，人工确认 {sum(r['annotation_status']=='confirmed' for r in f03)} 个。", '',
            '| 自动主类型建议 | 数量 |', '| --- | ---: |']
        lines += [f'| {e} {MAIN[e]} | {counts[e]} |' for e in MAIN]
        lines += ['', '这些是规则建议。身份歧义、真值问题和精确参考缺失仍需看图确认。', '',
            '## 配置家族受控对照', '', '| 因素 | 对照 | 救回 | 误伤 | 两者错且换设计 | 净救回 |', '| --- | --- | ---: | ---: | ---: | ---: |']
        lines += [f"| {r['family']} | {r['baseline']} → {r['variant']} | {r['RESCUE']} | {r['BREAK']} | {r['WRONG_SWITCH']} | {r['net_rescue']} |" for r in comparisons]
        lines += ['', '## 跨模型案例', '', '| 类型 | 照片数 |', '| --- | ---: |']
        lines += [f'| {t} | {summary["pattern_counts"][t]} |' for t in PATTERNS]
        lines += ['', '## 全部配置附表与人工校验', '',
            '- `all_predictions.csv`：全部预测、排名桶、对应基线转移、一致／分歧信息。',
            '- `configuration_summary.csv`：35 组错误主类型建议、人工确认数量、排名桶和主对照转移。',
            '- `controlled_comparisons.csv` / `controlled_transitions.csv`：受控对照汇总及逐图结果。',
            '- `unique_errors.csv`：去重后的人工查看单位；`classified_errors.csv`：标签映射回全部错误。',
            '- `F03_errors.csv`：主系统错误；`cross_model_cases.csv`：跨模型模式，标签可重叠。', '',
            '运行 `/home/zyin/envs/labelstudio/bin/python -B dataset/tools/review_errors.py --port 8766`，打开 http://127.0.0.1:8766/。', '',
            '人工数据保存在 `dataset/review/error_analysis/annotations.json`。每个错误对只看一次，'
            '包装关系、可见面和线索共享；各配置主类型及输入处理原因单独保存。保存后更新分类附表，'
            '刷新完整报告可运行 `dataset/tools/error_analysis.py` 或页面中的“更新报告”。', '',
            '唯一错误对的代表主类型优先取 F03，其次 B03，再其次第一个相关配置；同时保留每个相关配置的主类型，避免把不同排名或融合行为混为一类。',
            '图像模型对照以 B03 为固定锚点；对应基线的单列只列一个主对照，所有指定双通道对照都保留在受控对照表。',
            'E4/E5/E6 按各融合配置的明确基础通道推广，优先于 E2/E3。排名来自全图库打分，E2 只表示未进 Top-10。',
            '严格评测指标保持原始真值；E0/E1 人工标签不自动改变 Recall 或删除样本。',
            'VISUAL_ONLY 包括原图、CRISP 图像和前景处理；TEXT_ONLY 包括 BM25、OCR 查询及 OCR→OCR；融合成功会阻止这两个“仅”标签。',
            'FUSION_RESCUE/BREAK 可在同一照片的不同融合配置同时出现；MODEL_SPECIFIC 按模型名称划分，'
            'PREPROCESS_SENSITIVE 包含错向不同设计的变化；WRONG_SWITCH 不计救回。', '']
        (self.output / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    result = Analysis(args.root).export()
    print(json.dumps({k: result[k] for k in ['queries', 'prediction_rows', 'error_occurrences', 'unique_cases', 'confirmed', 'controlled_comparisons', 'pattern_counts']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
