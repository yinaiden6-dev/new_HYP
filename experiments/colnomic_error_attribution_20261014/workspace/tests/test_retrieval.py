"""Numerical and ranking edge cases for the full-gallery protocol."""
import unittest
from types import SimpleNamespace
import torch
from evaluation.scoring import multivector_scores, bm25_scores, fuse
from evaluation.metrics import evaluate, bootstrap


class ScoringTests(unittest.TestCase):
    def test_blocked_crisp_matches_direct_equations(self):
        torch.manual_seed(92)
        qs = [torch.nn.functional.normalize(torch.randn(n,8),dim=-1) for n in (7,3,0)]
        rs = [torch.nn.functional.normalize(torch.randn(n,8),dim=-1) for n in (5,9,4)]
        # Zero padding and all-negative similarities must not create false maxima.
        qs[0] = torch.cat((qs[0],torch.zeros(2,8)))
        rs[1] = torch.cat((rs[1],torch.zeros(3,8)))
        actual,crisp = multivector_scores(qs,rs,device='cpu',query_batch=2,reference_batch=2,token_block=2)
        for i,q in enumerate(qs):
            q = q[q.norm(dim=-1)>0]
            for j,r in enumerate(rs):
                r = r[r.norm(dim=-1)>0]
                if not len(q):
                    self.assertEqual(actual[i,j],0);self.assertEqual(crisp[i,j],0);continue
                sim = q@r.T
                w = (r-r.mean(0)).norm(dim=-1);w *= len(r)/w.sum()
                expected = (sim*w).max(-1).values.mean()-(sim*w).mean()+sim.max(0).values.mean()
                torch.testing.assert_close(actual[i,j],sim.max(-1).values.mean())
                torch.testing.assert_close(crisp[i,j],expected)

    def test_negative_matches_and_uniform_reference(self):
        plain,crisp = multivector_scores([torch.tensor([[1.,0.]])],
            [torch.tensor([[-1.,0.],[0.,0.]])],device='cpu')
        self.assertEqual(plain.item(),-1)
        self.assertEqual(crisp.item(),-1)

    def test_bm25_numbers_and_empty_text(self):
        s = bm25_scores(['100 mg','10 mg',''],['MEDICINE 10 mg','MEDICINE 100 mg'])
        self.assertEqual(s.argmax(-1)[:2].tolist(),[1,0]);self.assertEqual(s[2].sum(),0)

    def test_fixed_fusion_missing_ocr(self):
        image=torch.tensor([[1.,2.,3.],[3.,2.,1.]])
        text=torch.tensor([[3.,2.,1.],[0.,0.,0.]])
        out=fuse(image,text,[True,False])
        self.assertTrue(torch.equal(out[0],torch.zeros(3)))
        self.assertEqual(out[1].argmax(),0)

    def test_byte_dedup_ties_multiple_positives_and_empty_query(self):
        refs=[dict(reference_id=f'r{i}',sha256=s) for i,s in enumerate(['a','a','b','c'])]
        photos=[dict(photo_id=f'p{i}',package_id='pkg') for i in range(2)]
        ds=SimpleNamespace(photos=photos,references=refs,groups={'pkg':'g'},
            positives={'pkg':{'r1','r3'}},labels={p['photo_id']:dict(placement='tabletop',background='single_object') for p in photos})
        rows,metrics,types=evaluate(torch.zeros(2,4),ds,valid=[True,False])
        self.assertEqual(rows[0]['rank'],1)
        self.assertEqual(rows[0]['top_reference_ids'],['r0','r2','r3'])
        self.assertIsNone(rows[1]['rank']);self.assertEqual(rows[1]['top_reference_ids'],[])
        self.assertEqual(metrics['photo_average']['recall_at_1'],0.5)
        self.assertIsNone(types['handheld']['photo_average']['mrr'])
        interval=bootstrap(rows,rows,repetitions=100)['photo_average']['mrr']
        self.assertEqual(interval,dict(low=0.,high=0.))


if __name__=='__main__':
    unittest.main()
