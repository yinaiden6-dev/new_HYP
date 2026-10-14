"""Exercise resumable FORB workers and banks without loading a GPU model."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

import torch

from evaluation import forb_encoding as encoding
from evaluation.data import file_hash, read_json, write_json
from evaluation.encoders import Encoder


def storage_bytes(path):
    with zipfile.ZipFile(path) as archive:
        return sum(info.file_size for info in archive.infolist() if '/data/' in info.filename)


class Fixture:
    def __init__(self, data, unique=12, task_size=8, photo_indices=None, ref_indices=None):
        self.data = Path(data)
        self.data.mkdir(parents=True, exist_ok=True)
        self.rows = []
        for i in range(unique):
            path = self.data / f'image_{i:04d}.jpg'
            path.write_bytes(f'fake image content {i}'.encode())
            self.rows.append(dict(image_path=path.name, sha256=file_hash(path), domain='a'))
        photo_indices = list(range(unique // 2)) if photo_indices is None else photo_indices
        ref_indices = list(range(unique // 2, unique)) if ref_indices is None else ref_indices
        manifest = {}
        for population, indices in [('photos', photo_indices), ('references', ref_indices)]:
            manifest[population] = [dict(self.rows[j], id=f'{population}:{i}', index=i)
                                    for i, j in enumerate(indices)]
        write_json(self.data / 'cache/manifest.json', manifest)
        write_json(self.data / 'baseline_config.json', dict(models={
            'dinov3': {'image_contract': {'version': 1}},
            'colnomic7b': {'image_contract': {'version': 1}}},
            encoding_task_size=task_size, dense_batch=1))
        self.calls = []
        self.call_lock = threading.Lock()
        self.fail_after = None
        self.barrier = None

    def factory(self, model, spec, root, device):
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        fixture = self

        class FakeEncoder:
            def encode_images(self, paths):
                return [self.encode_image(path) for path in paths]

            def encode_image(self, path):
                with fixture.call_lock:
                    if fixture.fail_after is not None and len(fixture.calls) >= fixture.fail_after:
                        raise RuntimeError('simulated interrupted worker')
                    fixture.calls.append(Path(path).name)
                if fixture.barrier is not None:
                    time.sleep(.005)
                position = int(Path(path).stem.split('_')[-1])
                if model == 'dinov3':
                    value = torch.zeros(1024, dtype=torch.float32)
                    value[position % 1024] = 1.
                else:
                    value = torch.zeros((8, 128), dtype=torch.float16)
                    value[:, position % 128] = 1.
                return value

            def release(self):
                pass

        return FakeEncoder()


class ForbEncodingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_resume_starts_after_last_saved_image(self):
        fixture = Fixture(self.root, unique=13, task_size=16)
        fixture.fail_after = 8
        with patch.object(encoding, 'DenseEncoder', fixture.factory):
            with self.assertRaisesRegex(RuntimeError, 'simulated interrupted'):
                encoding.encode_worker(self.root, 'dinov3', device='cpu', worker='interrupted')
            path, marker, _ = encoding.task_paths(self.root, 'dinov3', 0)
            self.assertFalse(marker.exists())
            partial = torch.load(path, weights_only=True)
            self.assertEqual(len(partial['embeddings']), 8)
            self.assertEqual(read_json(self.root / 'cache/intermediate/workers/interrupted.json')['status'], 'failed')
            fixture.fail_after = None
            self.assertEqual(encoding.encode_worker(self.root, 'dinov3', device='cpu', worker='resumed'), 1)
            # A third invocation must hit cache without constructing an encoder.
        with patch.object(encoding, 'DenseEncoder', side_effect=AssertionError('unexpected model load')):
            self.assertEqual(encoding.encode_worker(self.root, 'dinov3', device='cpu', worker='cached'), 0)
        self.assertEqual(Counter(fixture.calls), Counter(r['image_path'] for r in fixture.rows))
        self.assertEqual(read_json(marker)['workers'], ['interrupted', 'resumed'])
        self.assertTrue(encoding.encoding_progress(self.root, 'dinov3')['complete'])

    def test_two_concurrent_workers_claim_each_task_once(self):
        fixture = Fixture(self.root, unique=16, task_size=4)
        fixture.barrier = threading.Barrier(2)
        with patch.object(encoding, 'DenseEncoder', fixture.factory):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(encoding.encode_worker, self.root, 'dinov3', 'cpu', worker)
                           for worker in ('worker-a', 'worker-b')]
                completed = [future.result(timeout=30) for future in futures]
        self.assertTrue(all(count > 0 for count in completed))
        self.assertEqual(sum(completed), 4)
        self.assertEqual(Counter(fixture.calls), Counter(r['image_path'] for r in fixture.rows))
        for first in range(0, 16, 4):
            _, marker, _ = encoding.task_paths(self.root, 'dinov3', first)
            self.assertEqual(len(read_json(marker)['workers']), 1)

    def test_sha_reuse_restores_every_original_id_order_and_dense_storage(self):
        fixture = Fixture(self.root, unique=6, task_size=4,
                          photo_indices=[2, 0, 2, 1],
                          ref_indices=[5, 2, 4, 0, 3, 1] * 24)
        with patch.object(encoding, 'DenseEncoder', fixture.factory):
            encoding.encode_worker(self.root, 'dinov3', device='cpu', worker='dense')
        self.assertEqual(len(fixture.calls), 6)
        encoding.merge_banks(self.root, 'dinov3')
        manifest = read_json(self.root / 'cache/manifest.json')
        for side, population in [('photo', 'photos'), ('ref', 'references')]:
            folder = self.root / 'cache/image_embeddings/dinov3' / side
            index = read_json(folder / 'index.json')
            self.assertEqual(index['ids'], [r['id'] for r in manifest[population]])
            self.assertEqual(index['image_sha256'], [r['sha256'] for r in manifest[population]])
            self.assertEqual(index['unique_images'], 6)
            restored_ids, restored_vectors = [], []
            for info in index['shards']:
                path = folder / info['path']
                shard = torch.load(path, weights_only=True)
                restored_ids.extend(shard['ids'])
                restored_vectors.extend(shard['embeddings'])
                self.assertEqual(file_hash(path), info['sha256'])
                self.assertEqual(storage_bytes(path), len(shard['ids']) * 1024 * 4)
            self.assertEqual(restored_ids, index['ids'])
            for row, vector in zip(manifest[population], restored_vectors, strict=True):
                expected = int(Path(row['image_path']).stem.split('_')[-1])
                self.assertEqual(int(vector.argmax()), expected)

    def test_model_signature_and_manifest_hash_changes_reject_completed_cache(self):
        fixture = Fixture(self.root, unique=4, task_size=4)
        with patch.object(encoding, 'DenseEncoder', fixture.factory):
            encoding.encode_worker(self.root, 'dinov3', device='cpu', worker='original')
            settings_path = self.root / 'baseline_config.json'
            original_settings = read_json(settings_path)
            changed = read_json(settings_path)
            changed['models']['dinov3']['image_contract']['version'] = 2
            write_json(settings_path, changed)
            with self.assertRaisesRegex(ValueError, 'Stale completed encoding task'):
                encoding.encode_worker(self.root, 'dinov3', device='cpu', worker='changed-model')
            write_json(settings_path, original_settings)
            manifest_path = self.root / 'cache/manifest.json'
            manifest = read_json(manifest_path)
            manifest['photos'][0]['sha256'] = '0' * 64
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, 'Stale completed encoding task'):
                encoding.encoding_progress(self.root, 'dinov3')

    def test_second_merge_preserves_index_and_shard_bytes_and_timestamps(self):
        fixture = Fixture(self.root, unique=6, task_size=4)
        with patch.object(encoding, 'DenseEncoder', fixture.factory):
            encoding.encode_worker(self.root, 'dinov3', device='cpu', worker='original')
        encoding.merge_banks(self.root, 'dinov3')
        folder = self.root / 'cache/image_embeddings/dinov3'
        paths = sorted(path for path in folder.rglob('*') if path.is_file())
        before = {str(p.relative_to(folder)): (file_hash(p), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in paths}
        with patch.object(encoding, 'save_tensor', side_effect=AssertionError('rewrote a valid shard')):
            with patch.object(encoding, 'write_json', side_effect=AssertionError('rewrote a valid index')):
                encoding.merge_banks(self.root, 'dinov3')
        after = {str(p.relative_to(folder)): (file_hash(p), p.stat().st_size, p.stat().st_mtime_ns)
                 for p in paths}
        self.assertEqual(before, after)

    def test_existing_merge_rejects_same_size_shard_checksum_change(self):
        fixture = Fixture(self.root, unique=6, task_size=4)
        with patch.object(encoding, 'DenseEncoder', fixture.factory):
            encoding.encode_worker(self.root, 'dinov3', device='cpu', worker='original')
        encoding.merge_banks(self.root, 'dinov3')
        shard = self.root / 'cache/image_embeddings/dinov3/ref/000000.pt'
        original = shard.read_bytes()
        changed = bytearray(original)
        changed[-50] ^= 1
        shard.write_bytes(changed)
        self.assertEqual(shard.stat().st_size, len(original))
        with self.assertRaisesRegex(ValueError, 'Changed merged embedding shard'):
            encoding.merge_banks(self.root, 'dinov3')

    def test_stale_partial_signature_rejected_before_resuming(self):
        fixture = Fixture(self.root, unique=10, task_size=16)
        fixture.fail_after = 8
        with patch.object(encoding, 'DenseEncoder', fixture.factory):
            with self.assertRaises(RuntimeError):
                encoding.encode_worker(self.root, 'dinov3', device='cpu', worker='original')
            path, _, _ = encoding.task_paths(self.root, 'dinov3', 0)
            partial = torch.load(path, weights_only=True)
            partial['signature'] = 'wrong-signature'
            encoding.save_tensor(path, partial)
            fixture.fail_after = None
            with self.assertRaisesRegex(ValueError, 'Stale partial encoding'):
                encoding.encode_worker(self.root, 'dinov3', device='cpu', worker='resumed')
        self.assertEqual(len(fixture.calls), 8)

    def test_changed_image_and_same_size_task_tampering_are_rejected(self):
        fixture = Fixture(self.root, unique=4, task_size=4)
        first = self.root / fixture.rows[0]['image_path']
        original = first.read_bytes()
        first.write_bytes(b'different image')
        with patch.object(encoding, 'DenseEncoder', fixture.factory):
            with self.assertRaisesRegex(ValueError, 'Image changed'):
                encoding.encode_worker(self.root, 'dinov3', device='cpu', worker='changed-image')
            self.assertEqual(fixture.calls, [])
            first.write_bytes(original)
            encoding.encode_worker(self.root, 'dinov3', device='cpu', worker='original')
        path, _, _ = encoding.task_paths(self.root, 'dinov3', 0)
        content = bytearray(path.read_bytes())
        content[-50] ^= 1
        path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, 'checksum differs'):
            encoding.merge_banks(self.root, 'dinov3')

    def test_multivector_shards_do_not_serialize_whole_task_storage(self):
        fixture = Fixture(self.root, unique=140, task_size=128,
                          photo_indices=[0, 1], ref_indices=list(range(2, 140)))
        with patch.object(encoding, 'Encoder', fixture.factory):
            encoding.encode_worker(self.root, 'colnomic7b', device='cpu', worker='multi')
        encoding.merge_banks(self.root, 'colnomic7b')
        for side in ('photo', 'ref'):
            folder = self.root / 'cache/image_embeddings/colnomic7b' / side
            index = read_json(folder / 'index.json')
            for info in index['shards']:
                path = folder / info['path']
                shard = torch.load(path, weights_only=True)
                expected = len(shard['ids']) * 8 * 128 * 2
                self.assertEqual(storage_bytes(path), expected)
                self.assertTrue(all(v.untyped_storage().nbytes() == v.numel() * v.element_size()
                                    for v in shard['embeddings']))

    def test_real_multivector_extraction_gives_each_image_its_own_storage(self):
        # Exercise the actual Encoder method, supplying only a fake model output.
        vectors = torch.zeros((2, 256, 128), dtype=torch.float32)
        vectors[:, :, 0] = 1.

        class Inputs(dict):
            def to(self, device):
                return self

        encoder = Encoder.__new__(Encoder)
        encoder.name = 'colnomic7b'
        encoder.device = 'cpu'
        encoder.model = lambda **kwargs: vectors
        mask = torch.zeros((2, 256), dtype=torch.int64)
        mask[0, :8] = 1
        mask[1, :12] = 1
        values, counts = encoder.encode_prepared(Inputs(attention_mask=mask))
        self.assertEqual(counts, [8, 12])
        for value, count in zip(values, counts, strict=True):
            self.assertEqual(value.dtype, torch.float16)
            self.assertEqual(value.untyped_storage().nbytes(), count * 128 * 2)
        self.assertNotEqual(values[0].untyped_storage().data_ptr(), values[1].untyped_storage().data_ptr())


if __name__ == '__main__':
    unittest.main()
