"""Independent fusion checks remain strict for narrow score distributions."""
import unittest
import numpy as np
import torch
from evaluation.scoring import fuse
from scripts.verify_results import verify_fusion_matrix


class FusionVerificationTest(unittest.TestCase):
    def test_narrow_scores_and_empty_ocr_fallback(self):
        generator=torch.Generator().manual_seed(49)
        image=torch.randn(3,5405,generator=generator,dtype=torch.float32)*.03+.24
        text=torch.randn(3,5405,generator=generator,dtype=torch.float32)*.00065+.24
        image[2]=0.;text[2]=0.
        has_text=[True,False,True]
        matrix=fuse(image,text,has_text,.5).numpy()
        verify_fusion_matrix(matrix,image.numpy(),text.numpy(),has_text)
        matrix[0,100]+=.005
        with self.assertRaisesRegex(AssertionError,'rounding bound'):
            verify_fusion_matrix(matrix,image.numpy(),text.numpy(),has_text)

    def test_wrong_fusion_weight_is_rejected(self):
        image=torch.tensor([[1.,2.,5.]],dtype=torch.float32)
        text=torch.tensor([[5.,3.,1.]],dtype=torch.float32)
        matrix=fuse(image,text,[True],.7).numpy()
        with self.assertRaises(AssertionError):
            verify_fusion_matrix(matrix,image.numpy(),text.numpy(),[True])


if __name__=='__main__':
    unittest.main()
