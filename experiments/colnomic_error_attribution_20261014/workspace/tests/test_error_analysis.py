"""Controlled comparisons, SHA aliases, shared review, and local API protection."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dataset/tools'))
from error_analysis import Analysis, CAUSES, Conflict, FUSIONS, read_csv, transition, write_csv, write_json
from review_errors import ReviewServer


class ErrorAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        ids = [f'B{i:02}' for i in range(1, 29)] + ['D01', 'D03'] + [f'F{i:02}' for i in range(1, 6)]
        refs = []
        for n, color in enumerate(['red', 'blue', 'blue', 'green']):
            path = self.root / f'dataset/references/ref{n}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (80, 120), color).save(path)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            refs.append(dict(reference_id=f'ref{n}', row=n, path=str(path.relative_to(self.root / 'dataset')),
                design_id='design_' + sha[:16], sha256=sha, identity=['correct', 'wrong blue', 'wrong blue alias', 'wrong green'][n]))
        self.refs = {r['reference_id']: r for r in refs}
        self.photos = []
        for n in range(8):
            path = self.root / f'dataset/packages/photo{n}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (100, 160), 'orange').save(path)
            self.photos.append(dict(photo_id=f'photo{n}', path=str(path.relative_to(self.root / 'dataset')),
                package_id=f'package{n}', design_id=refs[0]['design_id'],
                identity=f'package name {n}', sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        lists = self.root / 'dataset/lists'
        write_csv(lists / 'photos.csv', self.photos, list(self.photos[0]))
        write_csv(lists / 'references.csv', refs, list(refs[0]))
        write_csv(lists / 'positives.csv', [dict(package_id=p['package_id'], reference_id='ref0') for p in self.photos], ['package_id', 'reference_id'])
        write_csv(lists / 'groups.csv', [dict(package_id=p['package_id'], group_id='group' + p['photo_id']) for p in self.photos], ['package_id', 'group_id'])
        configs = []
        for cid in ids:
            mode = ('bm25_image_fusion' if cid.startswith('F') else 'manual' if cid == 'D01' else 'sam3' if cid == 'D03'
                    else 'bm25' if cid == 'B05' else 'ocr_fusion' if cid == 'B23' else 'fusion' if cid in FUSIONS
                    else 'ocr_to_ocr' if cid == 'B20' else 'ocr_to_image' if cid in ['B06', 'B07', 'B08', 'B09', 'B25']
                    else 'ocr_to_image_crisp' if cid in ['B17', 'B18', 'B19', 'B28']
                    else 'image_crisp' if cid in ['B14', 'B15', 'B16', 'B27'] else 'image')
            rows = [dict(photo_id=p['photo_id'], package_id=p['package_id'], group_id='g', rank=1, correct=True,
                    selected_reference_id='ref0', top_reference_ids=['ref0', 'ref1', 'ref3'], positive_reference_ids=['ref0']) for p in self.photos]
            def wrong(index, ref='ref1', rank=12):
                rows[index].update(correct=False, rank=rank, selected_reference_id=ref)
            wrong(6)
            wrong(7, 'ref2' if cid in ['B05', 'F03'] else 'ref1')
            if cid == 'B03':
                for n in [1, 3, 4, 5]:
                    wrong(n)
            if cid == 'B05':
                for n in [0, 3, 4, 5]:
                    wrong(n, 'ref3' if n == 3 else 'ref1')
            if cid == 'F03':
                for n, rank in [(0, 20), (1, 3), (2, 5), (3, 12), (4, 2)]:
                    wrong(n, 'ref3' if n == 3 else 'ref1', rank)
            if cid == 'D03':
                wrong(0)
            result = 'results/' + cid
            path = self.root / result
            path.mkdir(parents=True)
            (path / 'predictions.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in rows))
            config = dict(id=cid)
            if cid in ['D01', 'D03']:
                manifest = f'cache/{cid}/manifest.json'
                config['region_manifest'] = manifest
                write_json(self.root / manifest, dict(photos=[dict(photo_id=p['photo_id'], query_path='dataset/' + p['path'], fallback=False) for p in self.photos]))
            write_json(path / 'config.json', config)
            configs.append(dict(id=cid, family='test', model='bm25' if cid == 'B05' else ('colnomic7b' if cid in ['B03', 'D01', 'D03', 'F03'] else cid),
                mode=mode, queries=8, references=4, recall_at_1=sum(p['correct'] for p in rows) / 8, result=result))
        write_csv(self.root / 'results/all_results.csv', configs, list(configs[0]))
        self.store = Analysis(self.root)
        self.original = {p: p.read_bytes() for p in lists.glob('*.csv')}

    def tearDown(self):
        self.temp.cleanup()

    def case_key(self, photo='photo0', ref='ref1'):
        return photo + '__' + self.refs[ref]['design_id']

    def label(self, key, confirmed=True):
        c = self.store.cases[key]
        return dict(confusion='C2', visible_faces=['back', 'side'], clue_presence='absent',
            clue_readability='not_present', causes=['侧面/背面', '反光'], config_causes={},
            main_by_config={cid: self.store.auto_main(cid, c['photo_id']) for cid in c['configurations']},
            confirmed=confirmed, note='人工可见证据')

    def test_priority_includes_fusion_construction_and_ranks(self):
        self.assertEqual([self.store.auto_main('F03', f'photo{i}') for i in range(6)], ['E4', 'E5', 'E6', 'E2', 'E3', None])
        a = self.store.predictions['F03']['photo4']
        a['rank'] = None
        self.assertEqual(self.store.auto_main('F03', 'photo4'), 'E7')
        a['data_issue'] = True
        self.assertEqual(self.store.auto_main('F03', 'photo4'), 'E0')

    def test_sha_alias_is_one_case_and_not_wrong_switch(self):
        key = self.case_key('photo7')
        self.assertEqual(len(self.store.photo_cases['photo7']), 1)
        self.assertEqual(len(self.store.cases[key]['configurations']), 35)
        self.assertEqual(transition(self.store.predictions['B03']['photo7'], self.store.predictions['F03']['photo7']), 'WRONG_HOLD')
        self.assertEqual(transition(self.store.predictions['B03']['photo3'], self.store.predictions['F03']['photo3']), 'WRONG_SWITCH')

    def test_all_controlled_rows_and_cross_model_patterns(self):
        result = self.store.export()
        self.assertEqual(result['prediction_rows'], 35 * 8)
        self.assertEqual(result['controlled_comparisons'], 44)
        self.assertIn('ALL_FAIL', self.store.patterns['photo6']['tags'])
        self.assertIn('FUSION_RESCUE', self.store.patterns['photo5']['tags'])
        self.assertIn('FUSION_BREAK', self.store.patterns['photo0']['tags'])
        self.assertIn('PREPROCESS_SENSITIVE', self.store.patterns['photo0']['tags'])
        self.assertIn('UNSTABLE_CONFUSER', self.store.patterns['photo3']['tags'])
        rows = read_csv(self.store.output / 'controlled_comparisons.csv')
        for row in rows:
            self.assertEqual(sum(int(row[t]) for t in ['RESCUE', 'BREAK', 'WRONG_HOLD', 'WRONG_SWITCH', 'CORRECT_HOLD']), 8)
            self.assertEqual(int(row['net_rescue']), int(row['RESCUE']) - int(row['BREAK']))
        self.assertEqual(len(read_csv(self.store.output / 'controlled_transitions.csv')), 44 * 8)

    def test_save_one_pair_maps_to_configs_and_preserves_inputs(self):
        self.store.export()
        key = self.case_key('photo0')
        label = self.label(key)
        label['config_causes'] = {'D03': ['SAM漏分割'], 'F03': ['图文证据冲突']}
        self.store.save(key, label, 0)
        fresh = Analysis(self.root)
        self.assertEqual(fresh.summary()['confirmed'], 1)
        rows = [r for r in read_csv(self.store.output / 'classified_errors.csv') if r['case_id'] == key]
        self.assertEqual(len(rows), len(self.store.cases[key]['configurations']))
        self.assertTrue(all(r['annotation_status'] == 'confirmed' and r['confusion'] == 'C2' for r in rows))
        self.assertEqual(next(r for r in rows if r['config_id'] == 'F03')['main_type'], 'E4')
        self.assertEqual(next(r for r in rows if r['config_id'] == 'D03')['config_causes'], 'SAM漏分割')
        self.assertEqual(next(r for r in rows if r['config_id'] == 'B05')['config_causes'], '')
        with self.assertRaises(Conflict):
            self.store.save(key, label, 0)
        for path, original in self.original.items():
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_labels_and_priority_override_are_rejected(self):
        key = self.case_key()
        label = self.label(key)
        for change in [dict(confusion='C0,C1'), dict(causes=['内部机制猜测']), dict(visible_faces=['back', 'unclear']),
                       dict(config_causes={'B05': ['SAM漏分割']}), dict(clue_readability='readable')]:
            with self.assertRaises(ValueError):
                self.store.validate_label(key, label | change)
        label['main_by_config']['F03'] = 'E2'
        with self.assertRaises(ValueError):
            self.store.validate_label(key, label)
        label['main_by_config']['F03'] = 'E0'
        self.store.validate_label(key, label)

    def test_draft_is_not_confirmation_and_changed_sources_block_save(self):
        key = self.case_key()
        label = self.label(key, False) | dict(confusion='', visible_faces=[], clue_presence='', clue_readability='', causes=[])
        self.store.save(key, label, 0)
        self.assertEqual(self.store.summary()['confirmed'], 0)
        rows = [r for r in read_csv(self.store.output / 'classified_errors.csv') if r['case_id'] == key]
        self.assertTrue(all(r['annotation_status'] == 'pending' and r['main_type_source'] == 'human_draft' for r in rows))
        path = self.root / 'dataset/lists/groups.csv'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Source files changed'):
            self.store.save(key, self.label(key), 1)
        with self.assertRaisesRegex(ValueError, 'migrate'):
            Analysis(self.root)

    def test_local_api_auth_conflict_images_and_paths(self):
        server = ReviewServer(('127.0.0.1', 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            def request(path, data=None, headers=None):
                payload = json.dumps(data).encode() if data is not None else None
                r = urllib.request.Request(base + path, data=payload, headers=headers or {})
                with urllib.request.urlopen(r) as response:
                    return response.read()
            self.assertEqual(json.loads(request('/api/state'))['token'], server.token)
            key = self.case_key()
            payload = dict(case_id=key, revision=0, annotation=self.label(key))
            with self.assertRaises(urllib.error.HTTPError) as error:
                request('/api/save', payload)
            self.assertEqual(error.exception.code, 403)
            with self.assertRaises(urllib.error.HTTPError) as error:
                request('/api/save', payload, {'X-Review-Token': server.token, 'Origin': 'https://evil.example'})
            self.assertEqual(error.exception.code, 403)
            result = json.loads(request('/api/save', payload, {'X-Review-Token': server.token, 'Origin': base}))
            self.assertEqual(result['annotation']['revision'], 1)
            with self.assertRaises(urllib.error.HTTPError) as error:
                request('/api/save', payload, {'X-Review-Token': server.token})
            self.assertEqual(error.exception.code, 409)
            with Image.open(io.BytesIO(request('/image/input/D03/photo0'))) as image:
                self.assertEqual(image.size, (100, 160))
            with self.assertRaises(urllib.error.HTTPError):
                request('/image/photo/../../etc/passwd')
            with self.assertRaises(urllib.error.HTTPError) as error:
                request('/api/state', headers={'Host': 'evil.example'})
            self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
