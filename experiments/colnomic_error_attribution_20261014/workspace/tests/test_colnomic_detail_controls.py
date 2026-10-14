import unittest
import numpy as np
import torch
from scripts.colnomic_mechanism.analyze_detail_controls import region_weights, traced_similarity


class DetailControlsTests(unittest.TestCase):
    def test_soft_partition_conserves_template_and_roi(self):
        spec=dict(exif_orientation=1,layout=dict(start=2,count=4,width=2,height=2,total=6))
        source=dict(numeric_bbox=[0,0,50,50],blank_bbox=[50,0,100,50])
        weights,visual=region_weights(spec,source,[0,0,100,100],[100,100],np.arange(6))
        np.testing.assert_allclose(weights.sum(1),1)
        np.testing.assert_array_equal(visual,[False,False,True,True,True,True])
        self.assertEqual(weights[0,4],1)
        self.assertEqual(weights[2,0],1)
        self.assertEqual(weights[3,1],1)
        self.assertEqual(weights[4,2],1)

    def test_partial_patch_region_is_soft(self):
        spec=dict(exif_orientation=1,layout=dict(start=0,count=1,width=1,height=1,total=1))
        source=dict(numeric_bbox=[0,0,25,100],blank_bbox=[75,0,100,100])
        weights,_=region_weights(spec,source,[0,0,100,100],[100,100],np.array([0]))
        np.testing.assert_allclose(weights[0],[.25,.25,.5,0,0])

    def test_zero_padding_never_wins_and_ties_stable(self):
        q=torch.tensor([[1.,0.],[0.,0.],[0.,1.]])
        r=torch.tensor([[0.,0.],[1.,0.],[1.,0.],[0.,1.]])
        traced=traced_similarity(q,r)
        np.testing.assert_array_equal(traced['qi'],[0,2])
        np.testing.assert_array_equal(traced['winner'],[1,3])
        self.assertEqual(traced['second'][0],2)
        self.assertAlmostEqual(traced['score'],1.)

    def test_region_outside_snapshot_rejected(self):
        spec=dict(exif_orientation=1,layout=dict(start=0,count=1,width=1,height=1,total=1))
        source=dict(numeric_bbox=[0,0,100,100],blank_bbox=[0,0,0,0])
        with self.assertRaises(ValueError):region_weights(spec,source,[0,0,50,50],[100,100],np.array([0]))


if __name__=='__main__':unittest.main()
