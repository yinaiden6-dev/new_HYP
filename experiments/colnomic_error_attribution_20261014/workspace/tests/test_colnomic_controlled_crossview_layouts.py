import importlib.util
import unittest
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('crossview',Path(__file__).parents[1]/'scripts/colnomic_mechanism/controlled_crossview_layouts.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class CrossviewGeometryTests(unittest.TestCase):
    def source(self,h,width=1000,step=35,center=(.5,.5)):
        return dict(alignment=dict(homography=np.asarray(h).tolist()),display_width=width,
            shift_pixels=step,center_normalized=list(center))

    def test_180_degree_camera_rotation_swaps_B_direction_labels(self):
        a=self.source(np.eye(3));b=self.source([[-1,0,1],[0,-1,1],[0,0,1]])
        result=c.physical_correspondence(a,b)
        self.assertTrue(result['B_direction_label_swapped'])
        self.assertEqual(result['B_corresponding_sign'],-1)
        for move in result['physical_displacement_checks']:
            self.assertGreater(np.dot(move['A_GT_displacement'],move['B_GT_displacement']),0)

    def test_crossview_magnitude_difference_is_reported_not_rejected(self):
        result=c.physical_correspondence(self.source(np.eye(3),step=20),self.source(np.eye(3),step=40))
        self.assertGreater(result['physical_displacement_checks'][0]['magnitude_error'],0)
        self.assertFalse(result['physical_displacement_checks'][0]['crossview_magnitude_equality_required'])

    def test_repeated_word_at_different_physical_location_rejected(self):
        with self.assertRaisesRegex(ValueError,'same GT position'):
            c.physical_correspondence(self.source(np.eye(3),center=(.3,.3)),self.source(np.eye(3),center=(.6,.6)))

    def test_vertical_crossview_axis_not_silently_called_same_left(self):
        a=self.source(np.eye(3));b=self.source([[0,-1,1],[1,0,0],[0,0,1]])
        with self.assertRaisesRegex(ValueError,'x direction'):
            c.physical_correspondence(a,b)


if __name__=='__main__':unittest.main()
