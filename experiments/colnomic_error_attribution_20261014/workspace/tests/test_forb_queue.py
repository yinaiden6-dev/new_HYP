"""Scheduler resource tests use only CPU subprocesses and temporary lock files."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from scripts.forb_gpu_queue import Queue, claim


class QueueLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def queue(self, host='test43', gpu='0', role='single'):
        q = Queue.__new__(Queue)
        q.folder, q.hostname = self.folder, host
        q.args = argparse.Namespace(gpus=gpu, role=role)
        q.naps, q.calls = [], []
        q.nap = lambda phase: (q.naps.append(phase), time.sleep(.01))
        q.update = lambda **fields: None
        q.event = lambda *args, **fields: None
        q.finalized = lambda model: True
        q.ocr_status = lambda: ({'complete': True}, [], 32)
        q.banks_ready = lambda model: False
        q.forb = lambda *args, **kwargs: q.calls.append((args, kwargs))
        return q

    def test_host_names_are_independent_and_gpu_order_is_numeric(self):
        a, b = self.queue(), self.queue(host='test45')
        with a.gpu_leases('1,0', 'test') as descriptors:
            self.assertEqual([Path(os.readlink(f'/proc/self/fd/{fd}')).name for fd in descriptors],
                             ['test43_gpu_0.lock', 'test43_gpu_1.lock'])
            with claim(self.folder / 'gpu_leases/test43_gpu_0.lock') as acquired:
                self.assertFalse(acquired)
            with b.gpu_leases('0', 'test') as other:
                self.assertEqual(len(other), 1)
            with a.gpu_leases('', 'cpu') as cpu:
                self.assertEqual(cpu, [])

    def test_other_process_waits_until_entire_pair_is_available(self):
        q = self.queue()
        marker = self.folder / 'acquired'
        code = '''import argparse,sys,time
from pathlib import Path
from scripts.forb_gpu_queue import Queue
q=Queue.__new__(Queue);q.folder=Path(sys.argv[1]);q.hostname='test43'
q.nap=lambda phase: time.sleep(.01)
with q.gpu_leases('1,0','test'):
 Path(sys.argv[2]).write_text('leased')
'''
        with q.gpu_leases('1', 'test'):
            child = subprocess.Popen([sys.executable, '-c', code, str(self.folder), str(marker)])
            time.sleep(.15)
            self.assertFalse(marker.exists())
        self.assertEqual(child.wait(timeout=5), 0)
        self.assertTrue(marker.exists())

    def test_inherited_lease_survives_scheduler_descriptor_close(self):
        q = self.queue()
        with q.gpu_leases('0', 'test') as descriptors:
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.3)'],
                                     pass_fds=tuple(descriptors))
        path = self.folder / 'gpu_leases/test43_gpu_0.lock'
        with claim(path) as acquired:
            self.assertFalse(acquired)
        self.assertEqual(child.wait(timeout=5), 0)
        with claim(path) as acquired:
            self.assertTrue(acquired)

    def test_role_prerequisites_and_busy_tasks_do_not_load_model(self):
        q = self.queue()
        q.finalized = lambda model: False
        self.assertFalse(q.assist_colnomic_pair_encoding())
        q.args.role = 'ocr'
        q.ocr_status = lambda: ({'complete': False}, [], 32)
        self.assertFalse(q.assist_colnomic_pair_encoding())
        q.ocr_status = lambda: ({'complete': True}, [], 32)
        q.encoding_status = lambda model: ({'complete': False}, ['busy'])
        q.available = lambda locks: False
        self.assertFalse(q.assist_colnomic_pair_encoding())
        self.assertEqual(q.calls, [])

    def test_peer_pair_claim_causes_wait_instead_of_model_start(self):
        q = self.queue(gpu='1')
        with claim(self.folder / 'gpu_leases/test43_pair_0_1.lock'):
            self.assertTrue(q.assist_colnomic_pair_encoding())
        self.assertEqual(q.calls, [])
        self.assertIn('claimed_by_peer', q.naps[0])

    def test_pair_encoder_uses_adjacent_gpus_then_cpu_merge(self):
        q = self.queue(gpu='3')
        responses = iter([({'complete': False}, ['free']), ({'complete': True}, [])])
        q.encoding_status = lambda model: next(responses)
        q.available = lambda locks: True
        self.assertTrue(q.assist_colnomic_pair_encoding())
        self.assertEqual(q.calls, [(('colnomic7b', 'encode'), {'gpus': '2,3'}),
                                   (('colnomic7b', 'merge-embeddings'), {'gpus': ''})])

    def test_cpu_ocr_finalization_is_attempted_before_long_pair_encoding(self):
        q = self.queue()
        q.finalized = lambda model: model != 'bm25'
        q.try_finish_ocr_bm25 = lambda: q.calls.append('cpu_bm25')
        def enter_pair():
            q.calls.append('pair')
            raise StopIteration
        q.assist_colnomic_pair_encoding = enter_pair
        with self.assertRaises(StopIteration):
            q.assist_colnomic_and_ocr()
        self.assertEqual(q.calls, ['cpu_bm25', 'pair'])

    def test_finalizing_results_does_not_request_a_gpu(self):
        q = self.queue()
        q.data, q.identity, q.name = self.folder, {}, 'synthetic'
        q.finalized = lambda model: False
        q.score_status = lambda model: ({'complete': True}, [])
        metadata = q.data / 'cache/scores/bm25/metadata.json'
        metadata.parent.mkdir(parents=True)
        metadata.write_text('{}')
        self.assertTrue(q.try_finalize('bm25'))
        self.assertEqual(q.calls, [(('bm25', 'finalize'), {'gpus': ''})])

    def test_merge_is_not_repeated_after_peer_publishes_banks(self):
        q = self.queue()
        q.banks_ready = lambda model: True
        self.assertTrue(q.merge_embeddings_once('colnomic7b'))
        self.assertEqual(q.calls, [])
        q.banks_ready = lambda model: False
        with claim(self.folder / 'merge_colnomic7b.lock'):
            self.assertFalse(q.merge_embeddings_once('colnomic7b'))
        self.assertEqual(q.calls, [])


if __name__ == '__main__':
    unittest.main()
