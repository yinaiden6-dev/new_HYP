#!/usr/bin/env python3
"""Exact pixel-center polygon rasterization for the fixed OUTCOME-0212 audit.

Coordinates are the exact rationals represented by the annotation's binary64
numbers. Pixel centers are (column + 1/2, row + 1/2). Boundary is included.
Multiple target polygons are unioned. This utility performs no model inference.
"""
from __future__ import annotations

from fractions import Fraction
import math


def fraction_polygons(polygons):
    result = []
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError("POLYGON_HAS_FEWER_THAN_THREE_VERTICES")
        points = []
        for point in polygon:
            if len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
                raise ValueError("INVALID_POLYGON_VERTEX")
            points.append(tuple(Fraction.from_float(float(v)) for v in point))
        result.append(points)
    if not result:
        raise ValueError("EMPTY_TARGET_POLYGON_UNION")
    return result


def point_in_polygon_fraction(point, polygon):
    """Return (inside_or_boundary, on_boundary) using exact ray crossing."""
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if cross == 0 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True, True
        if (y1 > y) != (y2 > y):
            intersection = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersection:
                inside = not inside
    return inside, False


def _ceil(value):
    return -((-value.numerator) // value.denominator)


def _floor(value):
    return value.numerator // value.denominator


def rasterize_target_mask(polygons, width, height):
    """Return bool[height,width]; exact scanlines, without per-pixel rounding."""
    import numpy as np
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("INVALID_RASTER_SIZE")
    exact = fraction_polygons(polygons)
    mask = np.zeros((height, width), dtype=np.bool_)
    half = Fraction(1, 2)

    def interval(row, left, right):
        start = max(0, _ceil(left - half))
        stop = min(width - 1, _floor(right - half))
        if start <= stop:
            mask[row, start:stop + 1] = True

    for polygon in exact:
        minimum_y = min(p[1] for p in polygon)
        maximum_y = max(p[1] for p in polygon)
        first_row = max(0, _ceil(minimum_y - half))
        last_row = min(height - 1, _floor(maximum_y - half))
        edges = list(zip(polygon, polygon[1:] + polygon[:1]))
        for row in range(first_row, last_row + 1):
            y = Fraction(2 * row + 1, 2)
            intersections = []
            for (x1, y1), (x2, y2) in edges:
                if (y1 > y) != (y2 > y):
                    intersections.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
                # Include boundary centers, including excluded upper endpoints
                # and horizontal edges at a pixel-center row.
                if y1 == y2:
                    if y == y1:
                        interval(row, min(x1, x2), max(x1, x2))
                elif min(y1, y2) <= y <= max(y1, y2):
                    x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    column = x - half
                    if column.denominator == 1 and 0 <= column.numerator < width:
                        mask[row, column.numerator] = True
            intersections.sort()
            if len(intersections) % 2:
                raise ValueError("ODD_SCANLINE_CROSSING_COUNT")
            for index in range(0, len(intersections), 2):
                interval(row, intersections[index], intersections[index + 1])
    return mask


def target_polygons(annotation):
    if annotation.get("is_complete") is not True:
        raise ValueError("TARGET_ANNOTATION_INCOMPLETE")
    result = []
    for region in annotation.get("regions", []):
        if region.get("label") == "target":
            result.extend(region.get("polygons") or ([region["points"]] if region.get("points") else []))
    fraction_polygons(result)
    return result


def synthetic_e0():
    import numpy as np
    cases = [
        [[[0., 0.], [4., 0.], [4., 3.], [0., 3.]]],
        [[[.5, .5], [3.5, .5], [.5, 2.5]]],
        [[[.5, .5], [3.5, .5], [3.5, 1.5], [1.5, 1.5], [1.5, 2.5], [.5, 2.5]]],
        [[[-1., -1.], [1.5, -1.], [1.5, 1.5], [-1., 1.5]], [[2.5, 1.5], [5., 1.5], [5., 4.], [2.5, 4.]]],
    ]
    for polygons in cases:
        actual = rasterize_target_mask(polygons, 4, 3)
        exact = fraction_polygons(polygons)
        expected = np.array([[any(point_in_polygon_fraction((Fraction(2 * x + 1, 2), Fraction(2 * y + 1, 2)), p)[0] for p in exact) for x in range(4)] for y in range(3)], dtype=np.bool_)
        if not np.array_equal(actual, expected):
            raise RuntimeError("SYNTHETIC_EXACT_PIXEL_CENTER_ORACLE_MISMATCH")
    return {"status": "EXACT_PIXEL_CENTER_RASTER_E0_PASS", "cases": len(cases), "model_inference_count": 0}


if __name__ == "__main__":
    import json
    print(json.dumps(synthetic_e0(), sort_keys=True))
