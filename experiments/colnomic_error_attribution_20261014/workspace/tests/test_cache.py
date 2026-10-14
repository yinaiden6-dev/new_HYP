"""Cold cache, incremental repair, invalidation and model-free warm reads."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch
from evaluation.cache import Cache, save_tensor
from evaluation.data import file_hash, write_json


class FakeEncoder:
    calls=[]
    loads=0
    def __init__(self,name,config,root,device):
        type(self).loads+=1
    def encode_image(self,path):
        type(self).calls.append(str(path))
        return torch.nn.functional.normalize(torch.arange(1,129,dtype=torch.float32).repeat(3,1),dim=-1).half()
    def release(self):
        pass


class FakeDenseEncoder(FakeEncoder):
    def encode_images(self,paths):
        type(self).calls.extend(map(str,paths))
        vector=torch.nn.functional.normalize(torch.arange(1,1025,dtype=torch.float32),dim=0)
        return [vector.clone() for _ in paths]


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'dataset/lists').mkdir(parents=True)
        rows=[]
        for i in range(2):
            path=self.root/f'dataset/{i}.png';path.write_bytes(bytes([i]))
            rows.append(dict(photo_id=f'p{i}',path=path.name,sha256=file_hash(path),identity='x'))
        lists=self.root/'dataset/lists'
        with (lists/'photos.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        self.ds=SimpleNamespace(root=self.root,photos=rows,references=[],lists=lists,
            records=lambda p:rows,id_key=lambda p:'photo_id',verify_images=lambda p:None)
        self.config={'models':{'colpali':dict(family='colpali',image_model={'revision':'one'},
            image_contract={'recipe':'unit','view':'image','text':''})}}
        self.path=self.root/'cache/image_embeddings/colpali/photos.pt'
        FakeEncoder.calls=[];FakeEncoder.loads=0
    def tearDown(self):
        self.tmp.cleanup()
    def run_cache(self):
        return Cache(self.ds,self.config,device='cpu',encoder_factory=FakeEncoder).embeddings('colpali','image','photos')
    def test_cold_warm_and_one_missing_row(self):
        cold=self.run_cache()
        self.assertEqual(len(FakeEncoder.calls),2)
        calls=FakeEncoder.loads
        warm=self.run_cache();self.assertEqual(FakeEncoder.loads,calls)
        self.assertTrue(torch.equal(cold['embeddings'][0],warm['embeddings'][0]))
        partial=torch.load(self.path,weights_only=True)
        for key in ('ids','paths','sha256','source_sha256','identities','embeddings','provenance'):
            partial[key]=partial[key][:1]
        save_tensor(self.path,partial)
        repaired=self.run_cache()
        self.assertEqual(len(FakeEncoder.calls),3)
        self.assertTrue(torch.equal(cold['embeddings'][0],repaired['embeddings'][0]))
        self.assertEqual(repaired['ids'],['p0','p1'])
    def test_checkpoint_change_invalidates_all_rows(self):
        self.run_cache()
        self.config['models']['colpali']['image_model']['revision']='two'
        self.run_cache();self.assertEqual(len(FakeEncoder.calls),4)
    def test_image_change_invalidates_only_one_row(self):
        self.run_cache();self.ds.photos[1]['sha256']='changed'
        self.run_cache();self.assertEqual(len(FakeEncoder.calls),3)
    def test_cache_only_does_not_initialize_encoder(self):
        with self.assertRaisesRegex(ValueError,'Missing/stale'):
            Cache(self.ds,self.config,cache_only=True,encoder_factory=FakeEncoder).embeddings('colpali','image','photos')
        self.assertEqual(FakeEncoder.loads,0)

    def test_label_and_path_changes_reuse_immutable_bank(self):
        self.run_cache()
        before = file_hash(self.path)
        loads = FakeEncoder.loads
        self.ds.photos[0].update(path='new-package/0.png', identity='corrected-design')
        warm = Cache(self.ds, self.config, cache_only=True, encoder_factory=FakeEncoder).embeddings('colpali','image','photos')
        self.assertEqual(file_hash(self.path), before)
        self.assertEqual(FakeEncoder.loads, loads)
        self.assertEqual(warm['paths'][0], 'dataset/new-package/0.png')
        self.assertEqual(warm['identities'][0], 'corrected-design')

    def test_dense_cold_cache_and_model_free_warm_read(self):
        spec=self.config['models']['colpali']
        self.config['models']={'siglip2':dict(spec,family='siglip2')}
        FakeDenseEncoder.calls=[];FakeDenseEncoder.loads=0
        cold=Cache(self.ds,self.config,device='cpu',encoder_factory=FakeDenseEncoder).embeddings('siglip2','image','photos')
        self.assertEqual(cold['embeddings'].shape,(2,1024))
        self.assertEqual(len(FakeDenseEncoder.calls),2)
        warm=Cache(self.ds,self.config,cache_only=True,encoder_factory=FakeDenseEncoder).embeddings('siglip2','image','photos')
        self.assertEqual(FakeDenseEncoder.loads,1)
        torch.testing.assert_close(warm['embeddings'],cold['embeddings'])
        self.assertFalse((self.root/'cache/image_embeddings/siglip2/photos.partial.pt').exists())

    def test_dense_interrupted_batch_resumes_saved_rows(self):
        spec=self.config['models']['colpali']
        self.config['models']={'dinov3':dict(spec,family='dinov3')}
        rows=[dict(self.ds.photos[0],photo_id=f'p{i}',path=f'{i}.png') for i in range(129)]
        positions=list(range(len(rows)))
        class InterruptedEncoder(FakeDenseEncoder):
            def encode_images(self,paths):
                if paths[0].name=='128.png':
                    raise RuntimeError('interrupted')
                return super().encode_images(paths)
        interrupted=Cache(self.ds,self.config,device='cpu',encoder_factory=InterruptedEncoder,image_batch_size=128)
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            interrupted.dense_images('dinov3','photos',rows,positions,spec['image_contract'])
        FakeDenseEncoder.calls=[]
        resumed=Cache(self.ds,self.config,device='cpu',encoder_factory=FakeDenseEncoder,image_batch_size=256)
        values=resumed.dense_images('dinov3','photos',rows,positions,spec['image_contract'])
        self.assertEqual(sorted(values),positions)
        self.assertEqual(FakeDenseEncoder.calls,[str(self.root/'dataset/128.png')])

    def test_multivector_interruption_preserves_completed_image(self):
        class InterruptedEncoder(FakeEncoder):
            def encode_image(self,path):
                if path.name=='1.png':
                    raise RuntimeError('interrupted')
                return super().encode_image(path)
        cache=Cache(self.ds,self.config,device='cpu',encoder_factory=InterruptedEncoder)
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            cache.embeddings('colpali','image','photos')
        saved=self.path.parent/'photos.partial/p0.pt'
        first=torch.load(saved,weights_only=True)['embedding']
        self.assertFalse(self.path.exists())
        FakeEncoder.calls=[]
        resumed=self.run_cache()
        self.assertEqual(FakeEncoder.calls,[str(self.root/'dataset/1.png')])
        self.assertTrue(torch.equal(first,resumed['embeddings'][0]))
        self.assertFalse(saved.parent.exists())


if __name__=='__main__':
    unittest.main()
