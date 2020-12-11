import math
from itertools import islice

from collections import namedtuple

from .boundingbox import BoxMode, BoundingBox
from ..utils import minmax


Point = namedtuple('Point', ['x', 'y'])


def ellipse_to_bbox(center, radius, angle, mode=BoxMode.XYXY, relative=None, absolute=None):
    # From https://stackoverflow.com/questions/87734/how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse#answer-14163413

    ux = radius.x * math.cos(angle)
    uy = radius.x * math.sin(angle)

    angle += math.pi / 2
    vx = radius.y * math.cos(angle)
    vy = radius.y * math.sin(angle)

    tx = math.sqrt(ux * ux + vx * vx)
    ty = math.sqrt(uy * uy + vy * vy)

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
