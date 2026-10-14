"""Dense cache reuse and the OCR-to-OCR scoring route."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from threading import Event
from unittest.mock import patch

import torch

from evaluation.evaluate import neural_scores
from evaluation.dense import ParallelDenseEncoder


class ParallelEncodingTests(unittest.TestCase):
    def test_out_of_order_devices_preserve_rows_and_short_batches(self):
        completed=Event()
        instances=[]
        class Replica:
            def __init__(self,name,config,root,device):
                self.released=False;instances.append(self)
            def encode_images(self,paths):
                if paths[0]==0:
                    if not completed.wait(3):
                        raise RuntimeError('Other replica was not scheduled')
                else:
                    completed.set()
                return list(paths)
            def release(self):
                self.released=True
        with patch('evaluation.dense.DenseEncoder',Replica):
            encoder=ParallelDenseEncoder('dinov3',{},'.',devices=['cuda:0','cuda:1'])
            try:
                self.assertEqual(encoder.encode_images(list(range(5))),list(range(5)))
                self.assertEqual(encoder.encode_images([7]),[7])
                self.assertEqual(encoder.encode_images([]),[])
            finally:
                encoder.release()
        self.assertTrue(all(replica.released for replica in instances))

    def test_replica_failure_reaches_caller(self):
        class Replica:
            def __init__(self,*args): pass
            def encode_images(self,paths):
                if 2 in paths: raise RuntimeError('encoding failed')
                return paths
            def release(self): pass
        with patch('evaluation.dense.DenseEncoder',Replica):
            encoder=ParallelDenseEncoder('dinov3',{},'.',devices=['cuda:0','cuda:1'])
            try:
                with self.assertRaisesRegex(RuntimeError,'encoding failed'):
                    encoder.encode_images([0,1,2,3])
            finally:
                encoder.release()


class TextScoringTests(unittest.TestCase):
    def test_ocr_references_and_missing_document(self):
        with tempfile.TemporaryDirectory() as directory:
            calls=[]
            def embeddings(model,view,population):
                calls.append((view,population))
                self.assertEqual(view,'text')
                if population=='photos':
                    return dict(ids=['q'],embeddings=torch.tensor([[1.,0.]]),
                                has_text=[True],_file_sha256='query')
                return dict(ids=['r','empty'],embeddings=torch.tensor([[-1.,0.],[0.,0.]]),
                            has_text=[True,False],_file_sha256='reference')
            cache=SimpleNamespace(root=Path(directory),embeddings=embeddings,release=lambda:None)
            args=SimpleNamespace(refresh_scores=False,device='cpu',query_batch=2,reference_batch=16,token_block=32)
            results,_=neural_scores(cache,'Qwen3-VL-Embedding-2B','text',args,reference_view='text')
            score=results['ocr_to_ocr']
            self.assertEqual(calls,[('text','photos'),('text','references')])
            self.assertEqual(score['scores'].argmax(-1).tolist(),[0])
            self.assertEqual(score['recipe']['reference_view'],'text')
            self.assertTrue((cache.root/'cache/scores/Qwen3-VL-Embedding-2B/ocr_to_ocr.pt').exists())
            warm,_=neural_scores(cache,'Qwen3-VL-Embedding-2B','text',args,reference_view='text')
            self.assertTrue(warm['ocr_to_ocr']['cache_hit'])


if __name__=='__main__':
    unittest.main()
