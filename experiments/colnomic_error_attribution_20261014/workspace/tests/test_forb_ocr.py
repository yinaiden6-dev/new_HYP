"""Distributed OCR work must resume safely without changing official row IDs."""
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from evaluation import forb_ocr
from evaluation.data import read_json, write_json


class ForbOCRTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)
        self.calls = []
        self.fail_at = None
        rows = []
        for index in range(3):
            contents = f'image {index}'.encode()
            path = self.data / f'{index}.jpg'
            path.write_bytes(contents)
            rows.append(dict(index=index, id=f'i{index}', domain='bookcovers',
                             image_path=path.name, sha256=hashlib.sha256(contents).hexdigest()))
        manifest = dict(photos=[dict(rows[0], index=0, id='q0'), dict(rows[1], index=1, id='q1')],
                        references=[dict(rows[0], index=0, id='r0'),
                                    dict(rows[2], index=1, id='r1'), dict(rows[0], index=2, id='r2')])
        write_json(self.data / 'cache/manifest.json', manifest)
        write_json(self.data / 'baseline_config.json',
                   dict(ocr=read_json(forb_ocr.ROOT / 'configs/ppocrv6.json')))
        case = self

        class FakeOCR:
            def __init__(self, root, device):
                pass

            def recognize(self, row, image_path):
                if case.fail_at is not None and len(case.calls) == case.fail_at:
                    raise RuntimeError('simulated worker interruption')
                case.calls.append(row['sha256'])
                text = '' if image_path.name == '1.jpg' else image_path.stem
                lines = [] if not text else [dict(text=text, normalized_text=text, confidence=.9,
                    polygon=[[0., 0.], [9., 0.], [9., 9.], [0., 9.]], bbox=[0., 0., 9., 9.])]
                return dict(image_path=row['path'], image_sha256=row['sha256'], width=10, height=10,
                    stored_size=[10, 10], exif_orientation=1,
                    coordinate_system='EXIF-transposed image, pixel centers, top-left origin',
                    text=text, normalized_text=text, lines=lines), .5

        self.addCleanup(patch.stopall)
        patch.dict('sys.modules', {'evaluation.ocr': types.SimpleNamespace(OCR=FakeOCR)}).start()

    def test_resume_deduplicates_inference_but_preserves_every_gallery_row(self):
        self.fail_at = 1
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            forb_ocr.run(self.data, 'cpu', 'interrupted', task_size=2)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            forb_ocr.merge(self.data, task_size=2)
        self.fail_at = None
        forb_ocr.run(self.data, 'cpu', 'resumed', task_size=2)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(set(self.calls)), 3)
        metadata = forb_ocr.merge(self.data, task_size=2)
        self.assertEqual(metadata['inference_seconds'], 1.5)
        self.assertEqual(metadata['populations']['photo']['empty_ids'], ['q1'])
        ref = read_json(self.data / 'cache/ocr/ref.json')
        self.assertEqual(ref['ids'], ['r0', 'r1', 'r2'])
        self.assertEqual(ref['count'], 3)
        self.assertEqual(ref['unique_images'], 2)
        self.assertEqual(ref['entries']['r2']['index'], 2)
        self.assertEqual(ref['entries']['r0']['image_sha256'], ref['entries']['r2']['image_sha256'])
        forb_ocr.verify_cache(self.data, task_size=2)
        forb_ocr.run(self.data, 'cpu', 'finished', task_size=2)
        self.assertEqual(len(self.calls), 3)

    def test_worker_skips_busy_task_and_releases_its_locks(self):
        inputs = forb_ocr.load_inputs(self.data)
        forb_ocr.task_plan(inputs, 2)
        first_lock = forb_ocr.task_path(inputs, 0).with_suffix('.lock')
        with forb_ocr.lock(first_lock) as acquired:
            self.assertTrue(acquired)
            result = forb_ocr.run(self.data, 'cpu', 'second-worker', task_size=2)
            self.assertEqual(result['locked_tasks_skipped'], 1)
            self.assertEqual(result['generated_images'], 1)
        forb_ocr.run(self.data, 'cpu', 'first-worker', task_size=2)
        self.assertEqual(len(self.calls), 3)
        with forb_ocr.lock(first_lock) as acquired:
            self.assertTrue(acquired)

    def test_single_image_smoke_can_resume_without_changing_partition(self):
        result = forb_ocr.run(self.data, 'cpu', 'smoke', max_images=1)
        self.assertEqual(result['generated_images'], 1)
        self.assertEqual(result['completed_tasks'], 0)
        forb_ocr.run(self.data, 'cpu', 'resume')
        self.assertEqual(len(self.calls), 3)
        forb_ocr.merge(self.data)
        forb_ocr.verify_cache(self.data)

    def test_modified_task_or_image_fails_closed(self):
        forb_ocr.run(self.data, 'cpu', 'worker')
        inputs = forb_ocr.load_inputs(self.data)
        path = forb_ocr.task_path(inputs, 0)
        task = read_json(path)
        sha = task['image_hashes'][0]
        task['entries'][sha]['text'] = 'tampered'
        write_json(path, task)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            forb_ocr.merge(self.data)
        task['entries'][sha]['text'] = '0'
        write_json(path, task)
        (self.data / '0.jpg').write_bytes(b'changed image')
        with self.assertRaisesRegex(ValueError, 'differs from manifest'):
            forb_ocr.merge(self.data)

    def test_config_and_task_partition_cannot_silently_change(self):
        inputs = forb_ocr.load_inputs(self.data)
        forb_ocr.task_plan(inputs, 2)
        with self.assertRaisesRegex(ValueError, 'pinned to 2'):
            forb_ocr.task_plan(inputs, 3)
        config = read_json(self.data / 'baseline_config.json')
        config['ocr']['rotations_ccw'] = [0]
        write_json(self.data / 'baseline_config.json', config)
        with self.assertRaisesRegex(ValueError, 'differs from released'):
            forb_ocr.load_inputs(self.data)


if __name__ == '__main__':
    unittest.main()
