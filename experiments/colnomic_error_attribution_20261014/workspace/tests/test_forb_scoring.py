import fcntl
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from evaluation.data import file_hash
from evaluation.forb_scoring import (BM25Index, ShardedEmbeddingBank, dense_scores,
    maxsim_scores, merge_score_chunks, run_score_worker, verify_scores)
from evaluation.scoring import bm25_scores, multivector_scores


def _concurrent_worker(directory, counts, guard, barrier, queue):
    torch.set_num_threads(1)
    barrier.wait()

    def score(start, stop):
        with guard:
            for row in range(start, stop):
                counts[row] += 1
        time.sleep(.02)
        return torch.tensor([[float(i), -float(i)] for i in range(start, stop)])

    result = run_score_worker(directory, [f'q{i}' for i in range(8)], ['a', 'b'],
                              {'model': 'concurrency-test'}, score, chunk_size=1)
    queue.put(result)


class ForbScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_maxsim_negative_padding_asymmetry_and_fp64(self):
        queries = [torch.tensor([[1., 0.], [0., 1.], [0., 0.]]),
                   torch.tensor([[1., 0.]])]
        references = [torch.tensor([[-1., 0.], [0., 0.]]),
                      torch.tensor([[1., 0.], [0., 1.]])]
        actual = maxsim_scores(queries, references, 'cpu', query_batch=2,
                               reference_batch=2, token_block=1)
        torch.testing.assert_close(actual, torch.tensor([[-.5, 1.], [-1., 1.]]))
        self.assertTrue(verify_scores(queries, references, actual, 'maxsim')['passed'])
        reverse = maxsim_scores(references, queries, 'cpu', token_block=1)
        self.assertFalse(torch.equal(actual, reverse.T))

    def test_maxsim_all_tokens_and_independent_block_shapes(self):
        generator = torch.Generator().manual_seed(735)
        queries = [F.normalize(torch.randn(n, 5, generator=generator), dim=-1).half()
                   for n in (1, 5, 8)]
        references = [F.normalize(torch.randn(n, 5, generator=generator), dim=-1).half()
                      for n in (2, 7, 3, 9)]
        queries[1] = torch.cat([queries[1], torch.zeros(2, 5).half()])
        references[0] = torch.cat([references[0], torch.zeros(3, 5).half()])
        shared, _ = multivector_scores(queries, references, 'cpu', query_batch=2,
                                      reference_batch=3, token_block=3)
        for qb, rb, tb in [(1, 1, 1), (2, 3, 3), (4, 8, 32)]:
            actual = maxsim_scores(queries, references, 'cpu', query_batch=qb,
                                   reference_batch=rb, token_block=tb)
            torch.testing.assert_close(actual, shared, atol=1e-6, rtol=1e-6)
            self.assertLess(verify_scores(queries, references, actual, 'maxsim')['max_abs_error'], 1e-6)

    def test_maxsim_invalid_inputs_and_block_size(self):
        valid = [torch.ones(2, 3)]
        for invalid in [torch.zeros(3, 3), torch.full((2, 3), float('nan')),
                        torch.ones(2, 4), torch.ones(3)]:
            with self.assertRaises(ValueError):
                maxsim_scores(valid, [invalid], device='cpu')
        with self.assertRaises(ValueError):
            maxsim_scores(valid, valid, 'cpu', token_block=0)

    def test_dense_matches_fp64_and_uses_every_reference(self):
        generator = torch.Generator().manual_seed(921)
        q = F.normalize(torch.randn(5, 13, generator=generator), dim=-1)
        r = F.normalize(torch.randn(9, 13, generator=generator), dim=-1)
        actual = dense_scores(q, r, 'cpu', query_batch=2, reference_batch=4)
        torch.testing.assert_close(actual, q @ r.T)
        self.assertTrue(verify_scores(q, r, actual, 'dense')['passed'])
        with self.assertRaises(ValueError):
            dense_scores(q * 2, r, 'cpu')

    def test_bm25_global_statistics_repeats_unicode_empty_and_fp64(self):
        references = ['Milk milk_tea 商品', '', 'TEA, bread', 'milk bread bread', 'rare']
        queries = ['milk milk tea', '', '商品 unknown', 'rare milk', 'bread']
        index = BM25Index(references)
        actual = torch.cat([index.score(queries[:2]), index.score(queries[2:])])
        self.assertTrue(torch.equal(actual, bm25_scores(queries, references)))
        self.assertTrue(torch.equal(actual[1], torch.zeros(len(references))))
        self.assertTrue(verify_scores(queries, references, actual, 'bm25')['passed'])
        self.assertTrue(torch.equal(BM25Index(['', '']).score(['milk', '']), torch.zeros(2, 2)))

    def _bank(self, root):
        values = F.normalize(torch.arange(1, 16, dtype=torch.float32).reshape(5, 3), dim=-1)
        ids = [f'image{i}' for i in range(5)]
        shards = []
        for start, stop in [(0, 2), (2, 5)]:
            path = root / f'{start}.pt'
            torch.save(dict(ids=ids[start:stop], signature='bank1',
                            embeddings=values[start:stop].clone()), path)
            shards.append(dict(path=path.name, start=start, stop=stop, sha256=file_hash(path)))
        index = dict(ids=ids, signature='bank1', shards=shards)
        (root / 'index.json').write_text(json.dumps(index))
        return index, values

    def test_sharded_bank_cross_shard_slices_and_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, values = self._bank(root)
            bank = ShardedEmbeddingBank(root / 'index.json', cache_shards=1)
            torch.testing.assert_close(torch.stack(bank[1:4]), values[1:4])
            torch.testing.assert_close(bank[-1], values[-1])
            torch.testing.assert_close(torch.stack(bank[::-2]), values.flip(0)[::2])
            torch.testing.assert_close(dense_scores(values, bank, 'cpu', reference_batch=3),
                                       values @ values.T)
            self.assertEqual(len(bank._cache), 1)
            with self.assertRaises(IndexError):
                bank[5]

    def test_sharded_bank_rejects_corrupt_file_and_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, _ = self._bank(root)
            with (root / '0.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                ShardedEmbeddingBank(root / 'index.json')[0]
            index['shards'][0]['sha256'] = file_hash(root / '0.pt')
            index['ids'][0] = 'different'
            (root / 'index.json').write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, 'metadata'):
                ShardedEmbeddingBank(root / 'index.json')[0]
            index['shards'][1]['start'] = 3
            (root / 'index.json').write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, 'consecutive'):
                ShardedEmbeddingBank(root / 'index.json')

    def test_worker_resume_full_cache_merge_and_recipe_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qids, rids = ['q0', 'q1', 'q2'], ['r0', 'r1', 'r2', 'r3']
            matrix = torch.arange(12).reshape(3, 4).float()
            calls = []

            def score(start, stop):
                calls.append((start, stop))
                return matrix[start:stop]

            result = run_score_worker(root, qids, rids, {'model': 'test'}, score,
                                      chunk_size=2, max_chunks=1)
            self.assertFalse(result['complete'])
            with self.assertRaises(FileNotFoundError):
                merge_score_chunks(root)
            self.assertFalse((root / 'scores.npy').exists())
            result = run_score_worker(root, qids, rids, {'model': 'test'}, score, chunk_size=2)
            self.assertTrue(result['complete'])
            self.assertEqual(calls, [(0, 2), (2, 3)])
            metadata = merge_score_chunks(root, qids, rids, {'model': 'test'}, chunk_size=2)
            np.testing.assert_array_equal(np.load(root / 'scores.npy', mmap_mode='r'), matrix.numpy())
            self.assertEqual(metadata['query_ids'], qids)
            self.assertEqual(metadata['reference_ids'], rids)
            self.assertEqual(metadata['scores_sha256'], file_hash(root / 'scores.npy'))
            self.assertEqual(merge_score_chunks(root), metadata)
            warm = run_score_worker(root, qids, rids, {'model': 'test'}, score, chunk_size=2)
            self.assertEqual(warm['computed'], 0)
            with self.assertRaisesRegex(ValueError, 'Stale'):
                run_score_worker(root, qids, rids[::-1], {'model': 'test'}, score, chunk_size=2)

    def test_worker_exception_and_busy_lock_recover_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qids, rids = ['q0', 'q1'], ['r0']

            def fail(start, stop):
                raise RuntimeError('interrupted')

            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                run_score_worker(root, qids, rids, 'test', fail, chunk_size=1)
            self.assertEqual(list((root / 'chunks').glob('*.pt')), [])
            with (root / 'locks/000000-000001.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                result = run_score_worker(root, qids, rids, 'test',
                    lambda start, stop: torch.ones(stop - start, 1), chunk_size=1)
                self.assertEqual(result['busy'], 1)
                self.assertEqual(result['computed'], 1)
                self.assertFalse(result['complete'])
            result = run_score_worker(root, qids, rids, 'test',
                lambda start, stop: torch.ones(stop - start, 1), chunk_size=1)
            self.assertTrue(result['complete'])
            self.assertEqual(result['computed'], 1)

    def test_worker_rejects_incomplete_gallery_and_tampered_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'Invalid returned'):
                run_score_worker(root, ['q'], ['a', 'b'], 'test',
                                  lambda start, stop: torch.ones(1, 1))
            run_score_worker(root, ['q'], ['a', 'b'], 'test', lambda start, stop: torch.ones(1, 2))
            path = next((root / 'chunks').glob('*.pt'))
            chunk = torch.load(path, weights_only=True)
            chunk['scores'][0, 0] = -9
            torch.save(chunk, path)
            with self.assertRaisesRegex(ValueError, 'Invalid score chunk'):
                merge_score_chunks(root)

    def test_concurrent_workers_claim_each_query_once(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context('spawn')
            counts = context.Array('i', 8)
            guard = context.Lock()
            barrier = context.Barrier(3)
            queue = context.Queue()
            processes = [context.Process(target=_concurrent_worker,
                args=(directory, counts, guard, barrier, queue)) for _ in range(3)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=45)
                if process.is_alive():
                    process.terminate()
                    process.join()
                self.assertEqual(process.exitcode, 0)
            results = [queue.get(timeout=5) for _ in processes]
            self.assertEqual(sum(r['computed'] for r in results), 8)
            self.assertEqual(list(counts), [1] * 8)
            metadata = merge_score_chunks(directory)
            self.assertEqual(metadata['shape'], [8, 2])
            expected = np.array([[float(i), -float(i)] for i in range(8)], dtype=np.float32)
            np.testing.assert_array_equal(np.load(Path(directory) / 'scores.npy'), expected)


if __name__ == '__main__':
    unittest.main()
