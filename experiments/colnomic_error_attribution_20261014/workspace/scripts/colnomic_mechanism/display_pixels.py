"""EXIF display pixels without reserializing unrelated malformed EXIF tags."""
from PIL import Image

DISPLAY_TRANSFORMS={2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,
    4:Image.Transpose.FLIP_TOP_BOTTOM,5:Image.Transpose.TRANSPOSE,
    6:Image.Transpose.ROTATE_270,7:Image.Transpose.TRANSVERSE,8:Image.Transpose.ROTATE_90}


def display_image(source):
    orientation=int(source.getexif().get(274,1))
    if orientation not in range(1,9):raise ValueError('Invalid EXIF orientation')
    image=source.transpose(DISPLAY_TRANSFORMS[orientation]) if orientation in DISPLAY_TRANSFORMS else source.copy()
    image=image.convert('RGB')
    image.info.clear()  # Derived display only; original bytes/metadata stay intact.
    return image
