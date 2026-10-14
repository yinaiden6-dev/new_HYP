import numpy as np
from scripts.colnomic_mechanism.symmetric_layouts import variants
from scripts.colnomic_mechanism.display_pixels import display_image
from PIL import Image,ImageOps


def test_equal_magnitude_relocation_conserves_pixels_and_controls_leave_glyph():
    original=np.full((30,100,3),200,np.uint8)
    original[12:18,42:50]=[10,20,30]
    box=(40,10,52,20)
    images,step,allowed,block,background=variants(original,box,np.array([0,0,100,30]),[])
    assert step>0
    for arm,offset in [('left',-step),('right',step),('sham',0)]:
        np.testing.assert_array_equal(images[arm][10:20,40+offset:52+offset],block)
        np.testing.assert_array_equal(images[arm][~allowed],original[~allowed])
    for arm in ['editor_left','editor_right']:
        np.testing.assert_array_equal(images[arm][10:20,40:52],block)
    a=sorted(map(tuple,images['left'][allowed]))
    b=sorted(map(tuple,images['right'][allowed]))
    assert a==b


def test_display_pixels_match_all_valid_exif_orientations_without_metadata_rewrite():
    pixels=np.arange(3*4*3,dtype=np.uint8).reshape(3,4,3)
    for orientation in range(1,9):
        image=Image.fromarray(pixels)
        image.getexif()[274]=orientation
        actual=np.asarray(display_image(image))
        expected=np.asarray(ImageOps.exif_transpose(image))
        np.testing.assert_array_equal(actual,expected)
        assert image.getexif()[274]==orientation
