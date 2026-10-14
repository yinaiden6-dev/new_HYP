"""Original-resolution region crops shared by human and automatic diagnostics."""
import math

from PIL import Image, ImageOps


def open_photo(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert('RGB')


def padded_box(box, size, margin=0.05):
    width, height = size
    if len(box) != 4 or any(not math.isfinite(v) for v in box):
        raise ValueError('Invalid coordinates')
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Region lies outside the original image')
    if not 0 <= margin <= 1:
        raise ValueError('Invalid crop margin')
    pad = max(x1-x0, y1-y0)*margin
    return [max(0, math.floor(x0-pad)), max(0, math.floor(y0-pad)),
            min(width, math.ceil(x1+pad)), min(height, math.ceil(y1+pad))]


def crop_region(image, box, margin=0.05, mask=None):
    bounds = padded_box(box, image.size, margin)
    if mask is not None:
        if mask.size != image.size:
            raise ValueError('Mask must match the oriented original resolution')
        image = Image.composite(image, Image.new('RGB', image.size, 'white'), mask.convert('L'))
    return image.crop(bounds), bounds
