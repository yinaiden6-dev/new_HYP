import copy
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'programs'))
import run_rc_internal_m_v4_probe_v1 as P


class ProbeReplay(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(91)
        self.row=dict(query_id='fixture',candidate_ids=list(range(128)),candidate_identities=[str(i) for i in range(128)],
                      M=rng.uniform(.001,.3,128).tolist(),raw_scores=rng.normal(size=128).tolist())
        self.row['winner_index']=int(np.argmax(self.row['raw_scores']))
        self.row['challenger_positions']=[i for i in range(128) if i!=self.row['winner_index']]
        self.L=rng.uniform(.2,.8,128).tolist()

    def test_all_three_head_arithmetic(self):
        for kind,n in [('INTERNAL3',3),('ADDITIVE4',4),('PRODUCT5',5)]:
            theta=np.linspace(-.2,.9,n).tolist()
            a=P.numpy_decision(self.row,self.L,theta,kind);b=P.V.choose(self.row,self.L,theta,kind)
            np.testing.assert_allclose(a['logits'],b['logits'],atol=1e-12,rtol=1e-12)
            self.assertEqual(a['prediction_id'],b['prediction_id'])

    def test_internal_head_does_not_use_M(self):
        row=copy.deepcopy(self.row);row['M']=row['M'][1:]+row['M'][:1]
        self.assertEqual(P.numpy_decision(row,self.L,[1,2,3]),P.numpy_decision(self.row,self.L,[1,2,3]))

    def test_zero_logits_hold(self):
        d=P.numpy_decision(self.row,self.L,[0,0,0]);self.assertFalse(d['switched'])
        self.assertEqual(d['prediction_position'],self.row['winner_index'])

    def test_source_and_axis_tampering(self):
        rec=dict(authority={'sha256':'a'},snapshot={'sha256':'b'},query_id='fixture',candidate_ids=self.row['candidate_ids'],
                 M=self.row['M'],raw_scores=self.row['raw_scores'],L=self.L,decision=P.V.choose(self.row,self.L,[1,2,3]),
                 training_updates=0,held_label_reads=0)
        P.verify_prediction(rec,self.row,rec['authority'],rec['snapshot'])
        for key,value in [('snapshot',{'sha256':'wrong'}),('candidate_ids',list(reversed(self.row['candidate_ids']))),('training_updates',1)]:
            d=copy.deepcopy(rec);d[key]=value
            with self.assertRaises(RuntimeError):P.verify_prediction(d,self.row,rec['authority'],rec['snapshot'])


if __name__=='__main__':unittest.main()
