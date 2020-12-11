import math
from collections import namedtuple
from itertools import islice

from .boundingbox import BoxMode, BoundingBox

Point = namedtuple('Point', ['x', 'y'])


def ellipse_to_bbox(center, radius, angle, mode=BoxMode.XYXY, relative=None, absolute=None):
    # From https://stackoverflow.com/questions/87734/how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse#answer-14163413

    funcs = [math.cos, math.sin]

    u = Point(*(radius.x * func(angle) for func in funcs))
    v = Point(*(radius.y * func(angle + math.pi / 2) for func in funcs))

    tx = math.sqrt(u.x * u.x + v.x * v.x)
    ty = math.sqrt(u.y * u.y + v.y * v.y)

    return BoundingBox(
        [center.x - tx, center.y - ty, center.x + tx, center.y + ty],
        relative=relative,
        absolute=absolute
    ).to(mode)


def mask_to_bbox(mask, mode=BoxMode.XYXY, relative=None, absolute=None):
    xmin, ymin, xmax, ymax = mask[0][0], mask[0][1], mask[0][0], mask[0][1]

    for x, y in islice(mask, 1, None):
        if x < xmin:
            xmin = x
        elif x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        elif y > ymax:
            ymax = y

    return BoundingBox(
        [xmin, ymin, xmax, ymax],
        relative=relative,
        absolute=absolute
    ).to(mode)
