"""Foreground preparation keeps failures and resumes without re-segmenting."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from evaluation.data import file_hash, write_json
from evaluation.foreground import prepare, retrieve, manual_snapshot, Sam3Segmenter, select_sam3_top1
from evaluation.metrics import evaluate


class ForegroundTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        photos=[]
        for i in range(2):
            path=self.root/f'dataset/{i}.png';path.parent.mkdir(exist_ok=True)
            Image.new('RGB',(100,80),'red').save(path)
            photos.append(dict(photo_id=f'photo_{i}',path=f'{i}.png',sha256=file_hash(path)))
        self.dataset=SimpleNamespace(root=self.root,photos=photos,hashes={'photos.csv':'inventory'})
        self.dataset.matches_full_inventory=lambda hashes: hashes==self.dataset.hashes
        self.directory=self.root/'cache/foreground/sam3_auto/test';self.directory.mkdir(parents=True)
        self.spec=dict(route='sam3',settings=dict(margin=.05),sam=dict(path='synthetic fixture'),device='cpu')

    def tearDown(self):
        self.temp.cleanup()

    def test_original_resolution_fallback_and_resume(self):
        class Segmenter:
            def __init__(self,*args):self.calls=0
            def __call__(self,image):
                self.calls+=1
                if self.calls==2:return None,dict(candidates=0,fallback_reason='no_detection')
                mask=Image.new('L',image.size);mask.paste(255,(20,10,60,50))
                return mask,dict(candidates=1,confidence=.8)
        manifest=prepare(self.dataset,self.directory,self.spec,lambda *a,**k:None,Segmenter)
        self.assertEqual(manifest['counts'],dict(total=2,cropped=1,fallback=1))
        self.assertEqual(manifest['photos'][0]['crop_bounds'],[18,8,62,52])
        self.assertEqual(manifest['photos'][1]['query_path'],'dataset/1.png')
        with Image.open(self.root/manifest['photos'][0]['query_path']) as crop:
            self.assertEqual(crop.size,(44,44));self.assertEqual(crop.getpixel((0,0)),(255,255,255))
        def forbidden(*args):raise AssertionError('Cached preparation must not load SAM')
        shutil.rmtree(self.directory/'intermediate/regions')
        again=prepare(self.dataset,self.directory,self.spec,lambda *a,**k:None,forbidden)
        self.assertEqual(again,manifest)
        (self.root/manifest['photos'][0]['query_path']).write_bytes(b'corrupt crop')
        with self.assertRaisesRegex(ValueError,'modified'):
            prepare(self.dataset,self.directory,self.spec,lambda *a,**k:None,forbidden)

    def test_incomplete_manual_regions_are_explicit(self):
        data=dict(schema='release.subject_boxes.v1',photos_sha256='inventory',records={})
        write_json(self.root/'dataset/annotations/subject_boxes.json',data)
        with self.assertRaisesRegex(ValueError,'2 photos still need boxes'):
            manual_snapshot(self.dataset,False)
        self.assertEqual(manual_snapshot(self.dataset,True),data)
        spec=dict(self.spec,route='manual',annotations=data)
        manifest=prepare(self.dataset,self.directory,spec,lambda *a,**k:None)
        self.assertEqual(manifest['counts']['fallback'],2)
        self.assertEqual(manifest['fallback_reasons'],{'human_pending':2})

    def test_default_boxes_work_without_per_photo_confirmation(self):
        records = {p['photo_id']: dict(photo_id=p['photo_id'], image_sha256=p['sha256'],
            bbox=[20, 10, 60, 50], width=100, height=80) for p in self.dataset.photos}
        write_json(self.root/'dataset/annotations/subject_box_prelabels.json',dict(
            schema='release.subject_box_prelabels.v1', photos_sha256='inventory', records=records))
        snapshot = manual_snapshot(self.dataset,False)
        self.assertTrue(all(r['status']=='accepted' and r['source']=='sam3' for r in snapshot['records'].values()))
        self.assertFalse((self.root/'dataset/annotations/subject_boxes.json').exists())
        spec = dict(self.spec, route='manual', annotations=snapshot)
        manifest = prepare(self.dataset,self.directory,spec,lambda *a,**k:None)
        self.assertEqual(manifest['counts'], dict(total=2,cropped=2,fallback=0))
        self.assertTrue(all(r['annotation_source']=='sam3' for r in manifest['photos']))

    def test_stable_sam_selection_and_no_silent_inference_errors(self):
        segmenter=Sam3Segmenter.__new__(Sam3Segmenter)
        segmenter.official=False;segmenter.device='cpu'
        segmenter.config=dict(sam_prompt='product package',sam_confidence_threshold=.2,sam_mask_threshold=.5)
        masks=torch.zeros(1,2,80,100);masks[0,0,10:50,20:60]=1;masks[0,1,20:60,30:70]=1
        segmenter.model=lambda **kw:SimpleNamespace(pred_logits=torch.tensor([[1.,1.]]),
            pred_boxes=torch.zeros(1,2,4),pred_masks=masks,presence_logits=None)
        class Inputs(dict):
            def to(self,device):return self
        class Processor:
            def __call__(self,**kw):
                assert kw['text']=='product package'
                return Inputs()
            def post_process_instance_segmentation(self,outputs,**kwargs):
                assert outputs.pred_masks.shape[1]==1
                return [dict(scores=outputs.pred_logits[0].sigmoid(),masks=outputs.pred_masks[0])]
        segmenter.processor=Processor()
        mask,info=segmenter(Image.new('RGB',(100,80)))
        self.assertEqual(info['selected_index'],0);self.assertEqual(mask.getbbox(),(20,10,60,50))
        self.assertEqual(info['candidates'],2)
        def failed(**kw):raise RuntimeError('inference bug')
        segmenter.model=failed
        with self.assertRaisesRegex(RuntimeError,'inference bug'):
            segmenter(Image.new('RGB',(100,80)))

    def test_top1_matches_hf_full_postprocessing(self):
        from transformers.models.sam3.image_processing_sam3 import Sam3ImageProcessor
        processor=Sam3ImageProcessor()
        generator=torch.Generator().manual_seed(71)
        for presence in (None,torch.tensor([[.8]])):
            for logits in (torch.tensor([[-9.,1.,4.,4.,-8.]]),torch.full((1,5),-20.)):
                outputs=SimpleNamespace(pred_logits=logits,pred_boxes=torch.rand(1,5,4,generator=generator),
                    pred_masks=torch.randn(1,5,9,11,generator=generator),presence_logits=presence)
                selected,details=select_sam3_top1(outputs,.2)
                self.assertEqual(selected.pred_masks.shape[1],1)
                full=processor.post_process_instance_segmentation(outputs,threshold=.2,
                    mask_threshold=.5,target_sizes=[(101,153)])[0]
                limited=processor.post_process_instance_segmentation(selected,threshold=.2,
                    mask_threshold=.5,target_sizes=[(101,153)])[0]
                self.assertEqual(details['candidates'],len(full['scores']))
                if len(full['scores']):
                    index=int(torch.argsort(full['scores'],descending=True,stable=True)[0])
                    self.assertEqual(details['selected_index'],index)
                    self.assertTrue(torch.equal(limited['scores'][0],full['scores'][index]))
                    self.assertTrue(torch.equal(limited['masks'][0],full['masks'][index]))
                else:
                    self.assertEqual(len(limited['scores']),0)

    def test_retrieval_keeps_fallback_and_full_gallery(self):
        # First photo is cropped and matches ref 1; the second falls back to its
        # original ref-0 vector. Both rank-one outcomes must survive ID alignment.
        dataset=self.dataset
        for i,photo in enumerate(dataset.photos):photo['package_id']=f'package_{i}'
        dataset.references=[dict(reference_id=f'ref_{i}',sha256=f'hash_{i}') for i in range(2)]
        dataset.groups={f'package_{i}':f'group_{i}' for i in range(2)}
        dataset.positives={'package_0':{'ref_1'},'package_1':{'ref_0'}}
        dataset.labels={r['photo_id']:dict(placement='tabletop',background='single_object') for r in dataset.photos}
        dataset.annotation_sha256='scene labels'
        first=torch.zeros(1,128,dtype=torch.float16);first[0,0]=1
        second=torch.zeros_like(first);second[0,1]=1
        reference=dict(ids=['ref_0','ref_1'],embeddings=[first,second],_file_sha256='reference bank')
        original=dict(ids=['photo_0','photo_1'],embeddings=[first,first])
        control_path=self.root/'results/colnomic7b/image'
        write_json(control_path/'config.json',dict(dataset_sha256=dataset.hashes,annotation_sha256=dataset.annotation_sha256))
        control=evaluate(torch.tensor([[1.,0.],[1.,0.]]),dataset)[0]
        (control_path/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in control))
        rows=[dict(photo_id='photo_0',query_path='dataset/0.png',query_sha256=dataset.photos[0]['sha256'],fallback=False,fallback_reason=None),
              dict(photo_id='photo_1',query_path='dataset/1.png',query_sha256=dataset.photos[1]['sha256'],fallback=True,fallback_reason='no_detection')]
        manifest=dict(photos=rows,counts=dict(total=2,cropped=1,fallback=1),fallback_reasons={'no_detection':1})
        write_json(self.directory/'manifest.json',manifest)
        spec=dict(route='sam3',device='cpu',settings=dict(model='colnomic7b',query_batch=1,reference_batch=1,token_block=4),
                  baselines=dict(models={'colnomic7b':dict(image_model={},image_contract={})}))
        class FakeCache:
            def __init__(self,*a,**kw):pass
            def embeddings(self,model,view,pop):return reference if pop=='references' else original
        class FakeEncoder:
            def __init__(self,*a,**kw):pass
            def encode_image(self,path):return second
            def release(self):pass
        with patch('evaluation.foreground.Cache',FakeCache),patch('evaluation.encoders.Encoder',FakeEncoder):
            retrieve(dataset,self.directory,spec,manifest,lambda *a,**kw:None)
        result=self.root/'results/foreground/sam3_auto/test'
        metrics=json.loads((result/'metrics.json').read_text())
        paired=json.loads((result/'paired_vs_B03.json').read_text())
        predictions=[json.loads(r) for r in (result/'predictions.jsonl').read_text().splitlines()]
        self.assertEqual(metrics['queries'],2);self.assertEqual(metrics['reference_rows'],2)
        self.assertEqual(metrics['correct'],2);self.assertEqual(paired['rescued'],1);self.assertEqual(paired['harmed'],0)
        self.assertEqual(predictions[1]['foreground'],dict(fallback=True,reason='no_detection'))
        stages=[]
        shutil.rmtree(self.directory/'intermediate')
        bank_hash = file_hash(self.directory/'photo_embeddings.pt')
        with patch('evaluation.foreground.Cache',FakeCache), \
                patch('evaluation.encoders.Encoder',side_effect=AssertionError('Reuse existing vectors')), \
                patch('evaluation.foreground.evaluate',side_effect=AssertionError('Bank-only must not evaluate')):
            retrieve(dataset,self.directory,spec,manifest,lambda stage,*a,**kw:stages.append(stage),embedding_only=True)
        bank=torch.load(self.directory/'photo_embeddings.pt',map_location='cpu',weights_only=True)
        self.assertEqual(bank['ids'],original['ids'])
        self.assertTrue(torch.equal(bank['embeddings'][0],second))
        self.assertEqual(stages[-1],'embedded')
        info=json.loads((self.directory/'bank.json').read_text())
        self.assertEqual(info['new_embeddings'],0)
        self.assertEqual(info['bank_sha256'],file_hash(self.directory/'photo_embeddings.pt'))
        self.assertEqual(bank_hash, info['bank_sha256'])
        with patch('evaluation.foreground.Cache',FakeCache), \
                patch('evaluation.encoders.Encoder',side_effect=AssertionError('No new encoding')), \
                patch('evaluation.foreground.multivector_scores',side_effect=AssertionError('No new scoring')):
            retrieve(dataset,self.directory,spec,manifest,lambda *a,**kw:None)
        self.assertFalse((self.directory/'intermediate').exists())
        manifest['photos'][0]['query_sha256'] = 'changed crop'
        with patch('evaluation.foreground.Cache',FakeCache), self.assertRaisesRegex(ValueError, 'Stale foreground'):
            retrieve(dataset,self.directory,spec,manifest,lambda *a,**kw:None,embedding_only=True)
