"""Independent action reconstruction and fail-closed qualification."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'programs'))
import join_rc_postllm_h593_v1 as J


class Qualification(unittest.TestCase):
    def row(self):
        return dict(raw_scores=[float(128-i) for i in range(128)], M=[.001+.5*i/128 for i in range(128)],
            winner_index=0, challenger_positions=list(range(1,128)), candidate_ids=list(range(128)),
            candidate_identities=[str(i) for i in range(128)])

    def test_numpy_independent_against_original_torch(self):
        import torch
        import run_rc_internal_m_condition_scale_v4 as V
        torch.set_num_threads(2)
        rng = np.random.default_rng(991); row = self.row()
        for kind, dim in [('INTERNAL3',3),('ADDITIVE4',4),('PRODUCT5',5)]:
            for _ in range(6):
                l = rng.random(128); t = rng.normal(size=dim)
                original = V.choose(row,l,t,kind); independent = J.numpy_decision(row,l,t,kind)
                self.assertLess(J.compare_decision(original, independent),1e-10)

    def test_strict_zero_hold_and_first_positive_tie(self):
        row=self.row(); l=np.ones(128)
        self.assertEqual(J.numpy_decision(row,l,[0,0,0])['prediction_position'],0)
        self.assertEqual(J.numpy_decision(row,l,[0,0,-1])['prediction_position'],0)
        self.assertEqual(J.numpy_decision(row,l,[0,0,1])['prediction_position'],1)
        row['winner_index']=64; row['challenger_positions']=[i for i in range(128) if i!=64]
        self.assertEqual(J.numpy_decision(row,l,[0,0,1])['prediction_position'],0)

    def test_complete_axis_and_finite_inputs_required(self):
        with self.assertRaisesRegex(RuntimeError,'FULL128'):
            J.numpy_decision(self.row(),np.ones(127),[0,1,0])
        x=np.ones(128);x[34]=np.nan
        with self.assertRaisesRegex(RuntimeError,'FINITE'):
            J.numpy_decision(self.row(),x,[0,1,0])

    def test_missing_fold_seals_fail_before_label_access(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'INCOMPLETE_BEFORE_LABEL_JOIN'):
                J.required_seals(directory)
            self.assertFalse((Path(directory)/'all_predictions_prelabel_seal.json').exists())

    def test_decision_tampering_rejected(self):
        x=J.numpy_decision(self.row(),np.ones(128),[0,0,1]);y=dict(x)
        y['logits']=list(x['logits']);y['logits'][8]+=.001
        with self.assertRaisesRegex(RuntimeError,'INDEPENDENT_NUMPY'):
            J.compare_decision(y,x)


if __name__=='__main__':
    unittest.main()
