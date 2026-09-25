"""Boundary regressions for frozen external inference; no images or GPU needed."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'programs'))
import run_rc_postllm_grozi_external_v1 as X
import join_rc_postllm_h593_v1 as J


class ExternalContract(unittest.TestCase):
    def row(self):
        axis=list(range(128));raw=torch.linspace(-2,2,128,dtype=torch.float64)
        q={'query_id':'example','candidate_physical_rows':axis,'candidate_raw_scores':raw,
           'raw_ranked_physical_rows':list(reversed(axis))}
        r={'candidates':[{'old_scores':{'visibility_mass':(i+1)/256}} for i in axis]}
        return X.make_row(q,r,[f'id{i}' for i in axis])

    def test_physical_axis_and_no_target_input(self):
        r=self.row()
        self.assertEqual(r['winner_index'],127)
        self.assertEqual(r['challenger_positions'],list(range(127)))
        self.assertEqual(r['candidate_identities'][13],'id13')
        self.assertFalse({'target_id','target_positions','raw_correct'}&r.keys())

    def test_strict_hold_and_independent_scalar_actions(self):
        r=self.row();L=torch.linspace(.1,.9,128,dtype=torch.float64)
        for kind,theta in [('INTERNAL3',[0.,0.,0.]),('INTERNAL3',[.3,2.,-.1]),
                           ('ADDITIVE4',[.3,.7,2.,-.1]),('PRODUCT5',[.3,.4,.7,2.,-.1])]:
            actual=X.H.V.choose(r,L,theta,kind)
            expected=J.numpy_decision(r,L.tolist(),theta,kind)
            self.assertLessEqual(J.compare_decision(actual,expected),1e-10)
        zero=X.H.V.choose(r,L,[0.,0.,0.])
        self.assertFalse(zero['switched'])
        self.assertEqual(zero['prediction_position'],127)

    def test_intervention_preserves_candidate_axis(self):
        r=self.row();axis=r['candidate_ids'].copy();orig=r['M'].copy()
        masses,const=X.H.row_condition(r,'POST_REAL','constant')
        self.assertTrue(const);self.assertEqual(masses,orig)
        masses,const=X.H.row_condition(r,'POST_REAL','shuffled')
        self.assertFalse(const);self.assertEqual(masses,orig[1:]+orig[:1])
        self.assertEqual(r['candidate_ids'],axis);self.assertEqual(r['M'],orig)


if __name__=='__main__':unittest.main()
