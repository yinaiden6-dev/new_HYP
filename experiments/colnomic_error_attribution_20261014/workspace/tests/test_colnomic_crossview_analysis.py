import importlib.util
import unittest
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('crossview_analysis',Path(__file__).parents[1]/'scripts/colnomic_mechanism/analyze_controlled_crossview_layouts.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class CrossviewDecompositionTests(unittest.TestCase):
    def test_fixed_winner_feature_change_and_argmax_switch_conserve(self):
        e=torch.eye(128)[:2]
        sham=e[[0,0]];edited=e[[1,0]]
        r=c.decompose_query_change(sham,edited,e)
        self.assertEqual(r['delta'],0)
        self.assertEqual(r['fixed_sham_winner_feature_delta'],-.5)
        self.assertEqual(r['argmax_switch_delta'],.5)
        self.assertEqual(r['switch_tokens'],1)
        self.assertEqual(r['max_token_conservation_error'],0)

    def test_padded_zero_tokens_removed_with_original_axis_preserved(self):
        e=torch.zeros((4,128));e[1,0]=1;e[3,1]=1
        selected,indices=c.valid_tokens(e)
        self.assertEqual(indices.tolist(),[1,3])
        self.assertEqual(selected.shape,(2,128))

    def test_tied_winners_use_stable_first_reference_token(self):
        e=torch.eye(128)[:2]
        r=c.decompose_query_change(e[:1],e[1:2],e[[0,0,1]])
        self.assertEqual(r['per_token']['base_winner'].tolist(),[0])
        self.assertEqual(r['per_token']['edited_winner'].tolist(),[2])
        self.assertEqual(r['fixed_sham_winner_feature_delta'],-1)
        self.assertEqual(r['argmax_switch_delta'],1)


if __name__=='__main__':unittest.main()
