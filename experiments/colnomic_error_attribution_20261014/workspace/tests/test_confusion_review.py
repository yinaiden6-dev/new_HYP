"""Confusion decisions, subset integrity, concurrent edits and HTTP isolation."""
import hashlib
import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image
import test_photo_review as fixture
from photo_metadata import Conflict, PhotoStore, write_json
from confusion_review import ConfusionReview, inventory_signature
from review_photos import ReviewServer


class ConfusionReviewTest(unittest.TestCase):
    def setUp(self):
        fixture.PhotoReviewTest.setUp(self)
        refs=[]
        for i in range(4):
            rid=f'reference_{i:06d}';path=self.root/f'references/images/{rid}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(160,100),['#308088','#cd6849','#51713d','#308088'][i]).save(path)
            refs.append(dict(reference_id=rid,row=str(i),path=path.relative_to(self.root).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),identity=f'Example medicine {i}',category='human/otc',original_filename=f'example{i}.jpg'))
        fixture.csv_file(self.root/'lists/references.csv',list(refs[0]),refs)
        fixture.csv_file(self.root/'lists/positives.csv',['package_id','reference_id'],[
            dict(package_id='package_001',reference_id=refs[0]['reference_id']),dict(package_id='package_002',reference_id=refs[1]['reference_id'])])
        for row in self.store.photos.values():Image.new('RGB',(120,80),'#7060a0').save(self.root/row['path'])
        for row in self.store.packages.values():Image.new('RGB',(160,100),'#308088').save(self.root/row['design_path'])
        self.review=ConfusionReview(self.store)
        write_json(self.review.candidate_path,dict(schema='release.confusion_candidates.v1',inventory=inventory_signature(self.root),
            references={r['reference_id']:dict(title=r['identity'],ocr_preview='Example 20 mg',search_text='Example 20 mg') for r in refs},
            packages={'package_001':[dict(reference_id=refs[1]['reference_id'],proposed_categories=['strength_count'],evidence=['规格候选']),
                dict(reference_id=refs[2]['reference_id'],proposed_categories=['visual_layout'],evidence=['视觉候选'])]}))
        self.original={p:p.read_bytes() for p in [*(self.root/'lists').glob('*.csv'),*(self.root/'packages').rglob('metadata.json')]}

    tearDown=fixture.PhotoReviewTest.tearDown

    def body(self,left='reference_000000',right='reference_000001',**updates):
        d=self.review.read();v=self.review.decision(d,left,right)
        return dict(dict(left=left,right=right,revision=v['revision'],status='confusing',categories=['strength_count'],note='different strengths'),**updates)

    def test_proposals_do_not_select_and_pairs_are_symmetric(self):
        self.assertEqual(self.review.summary()['counts']['packages'],0)
        self.assertFalse(self.review.path.exists())
        body=self.body();self.review.update_pair(body)
        data=self.review.export()
        self.assertEqual(data['validation_subset']['package_ids'],['package_001','package_002'])
        self.assertEqual(len(data['validation_subset']['photo_ids']),4)
        self.assertEqual(len(data['pairs']),1)
        a=self.review.decision(data,'reference_000000','reference_000001')
        b=self.review.decision(data,'reference_000001','reference_000000')
        self.assertEqual(a,b)
        with self.assertRaises(Conflict):self.review.update_pair(body)
        self.review.update_pair(self.body(left='reference_000001',right='reference_000000',status='not_confusing'))
        self.assertEqual(self.review.summary()['counts']['packages'],0)
        self.assertEqual(len(self.review.read()['pairs']),1)
        self.assertEqual(ConfusionReview(self.store).export(),self.review.export())

    def test_extra_designs_do_not_become_queries_and_can_be_dismissed(self):
        self.review.update_pair(self.body(right='reference_000002'))
        s=self.review.export()['validation_subset']
        self.assertEqual(s['package_ids'],['package_001']);self.assertEqual(len(s['photo_ids']),3)
        self.assertEqual(s['additional_design_reference_ids'],['reference_000002'])
        state=self.review.design_state(self.review.read(),'reference_000002')
        self.review.update_design('reference_000002',dict(revision=state['revision'],selection='exclude',note='do not acquire'))
        self.assertEqual(self.review.summary()['counts']['additional_designs'],0)
        self.review.update_pair(self.body(status='confusing'))
        state=self.review.summary()['packages'][0]
        self.review.update_package('package_001',dict(revision=state['revision'],selection='exclude',reviewed=True,note='skip'))
        self.assertEqual(self.review.export()['validation_subset']['package_ids'],['package_002'])
        self.assertEqual(self.review.summary()['counts']['reviewed_packages'],1)
        for p,data in self.original.items():self.assertEqual(p.read_bytes(),data,str(p))

    def test_duplicate_rejection_and_input_validation(self):
        self.assertEqual(len(self.review.refs),3)
        with self.assertRaises(ValueError):self.review.update_pair(self.body(right='reference_000003'))
        for changes in [dict(status='invented'),dict(categories=[]),dict(categories=['invented']),dict(note=123)]:
            with self.assertRaises(ValueError):self.review.update_pair(self.body(**changes))
        self.assertFalse(self.review.path.exists())
        self.review.update_pair(self.body(status='same_design'))
        self.assertEqual(self.review.summary()['counts']['packages'],0)
        self.assertEqual(self.review.browse(query='reference_000003')['items'][0]['reference']['reference_id'],'reference_000000')
        self.assertEqual(self.review.browse(query='20 mg',scope='other')['total'],1)
        self.assertEqual(self.review.browse(scope='existing')['total'],2)

    def test_independent_pair_edits_merge_and_stale_inventory_stops(self):
        first=self.body();second=self.body(right='reference_000002')
        self.review.update_pair(first);self.review.update_pair(second)
        self.assertEqual(len(self.review.read()['pairs']),2)
        self.assertEqual(self.review.summary()['counts']['confirmed_pairs'],2)
        self.assertEqual(len(self.review.export()['validation_subset']['connected_groups']),1)
        data=self.review.read();data['inventory']['photos.csv']='stale';write_json(self.review.path,data)
        with self.assertRaises(ValueError):self.review.summary()

    def test_review_progress_separates_pending_unsure_and_finished(self):
        initial=self.review.summary()['packages'][0]['review_progress']
        self.assertEqual((initial['total'],initial['pending'],initial['finalized']),(2,2,0))
        self.review.update_pair(self.body(status='unsure'))
        progress=self.review.summary()['packages'][0]['review_progress']
        self.assertEqual((progress['pending'],progress['unsure'],progress['finalized']),(1,1,0))
        self.assertEqual(self.review.summary()['counts']['completed_packages'],0)
        self.review.update_pair(self.body(status='not_confusing'))
        progress=self.review.summary()['packages'][0]['review_progress']
        self.assertEqual((progress['pending'],progress['unsure'],progress['finalized']),(1,0,1))
        # Reverse direction also counts as reviewed, without another manual checkbox.
        self.assertEqual(self.review.summary()['counts']['completed_packages'],1)
        self.assertEqual(self.review.browse(package_id='package_001',status='finalized')['total'],1)
        self.assertEqual(self.review.browse(package_id='package_001',status='pending')['total'],1)
        self.review.update_pair(self.body(status='pending'))
        self.assertEqual(self.review.summary()['counts']['reviewed_pairs'],0)
        self.assertEqual(self.review.summary()['packages'][0]['review_progress']['pending'],2)

    def test_http_session_conflicts_gallery_and_existing_editor(self):
        server=ReviewServer(('127.0.0.1',0),self.store)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url=f'http://127.0.0.1:{server.server_port}'
        try:
            for path in ['/confusion','/confusion_review.js','/confusion_review.css','/api/confusion/state',
                         '/api/confusion/package/package_001','/api/confusion/browse?scope=other&package=package_001',
                         '/reference-image/reference_000002','/reference-thumb/reference_000002','/api/state','/']:
                with urllib.request.urlopen(url+path) as response:self.assertEqual(response.status,200,path)
            body=json.dumps(self.body()).encode()
            for headers,expected in [({},403),({'X-Review-Token':server.token,'Origin':'http://evil.invalid'},403),
                                     ({'X-Review-Token':server.token},200),({'X-Review-Token':server.token},409)]:
                request=urllib.request.Request(url+'/api/confusion/pair',data=body,headers=headers)
                try:
                    with urllib.request.urlopen(request) as response:self.assertEqual(response.status,expected)
                except urllib.error.HTTPError as error:self.assertEqual(error.code,expected)
            with urllib.request.urlopen(url+'/api/confusion/export') as response:
                self.assertEqual(len(json.load(response)['validation_subset']['photo_ids']),4)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)


if __name__=='__main__':unittest.main()
