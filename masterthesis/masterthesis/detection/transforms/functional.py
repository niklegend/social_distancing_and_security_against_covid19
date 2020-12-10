import math

import torch
import torchvision.transforms.functional as T
from PIL import Image

from ..boundingbox import BoundingBox, BoxMode
from ...utils import minmax


def resize(img, boxes, size, interpolation=Image.BILINEAR):
    new_img = T.resize(img, size=size, interpolation=interpolation)
    new_bboxes = [bbox.resize(img.size, size) for bbox in boxes]

    return new_img, new_bboxes


def hflip(img, bboxes):
    new_img = T.hflip(img)
    new_bboxes = [bbox_hflip(bbox, img.size) for bbox in bboxes]
    return new_img, new_bboxes


def bbox_hflip(bbox, img_size):
    return bbox_translate(bbox, (img_size[0] - 1, None))


def vflip(img, bboxes):
    new_img = T.vflip(img)
    new_bboxes = [bbox_vflip(bbox, img.size) for bbox in bboxes]
    return new_img, new_bboxes


def bbox_vflip(bbox, img_size):
    return bbox_translate(bbox, (None, img_size[1] - 1))


def bbox_translate(bbox, translation):
    new_bbox = bbox.to(BoxMode.CXCYWH)

    tx, ty = translation

    if tx:
        new_bbox[0] = tx - new_bbox[0]
    if ty:
        new_bbox[1] = ty - new_bbox[1]

    return new_bbox.to(bbox.mode)


def rotate(img, bboxes, angle):
    img_bbox = BoundingBox([0, 0, *img.size])
    new_img = img.rotate(angle)

    new_bboxes = []

    for bbox in bboxes:
        new_bbox = bbox_rotate(bbox.to(BoxMode.XYXY), math.radians(-angle), img.size)
        if new_bbox in img_bbox:
            new_bboxes.append(new_bbox.to(bbox.mode))

    return new_img, new_bboxes


def bbox_rotate(bbox, theta, size):
    c = math.cos(theta)
    s = math.sin(theta)

    # Image centroid
    cx, cy = [x / 2 for x in size]

    def rotate_point(x, y):
        a = x - cx
        b = y - cy
        return c * a - s * b + cx, s * a + c * b + cy

    xmin, ymin, xmax, ymax = bbox.to(BoxMode.XYXY)

    x0, y0 = rotate_point(xmin, ymin)
    x1, y1 = rotate_point(xmax, ymin)
    x2, y2 = rotate_point(xmin, ymax)
    x3, y3 = rotate_point(xmax, ymax)

    xmin, xmax = minmax([x0, x1, x2, x3])
    ymin, ymax = minmax([y0, y1, y2, y3])

    return BoundingBox([xmin, ymin, xmax, ymax]).to(bbox.mode)


def crop(img, bboxes, top, left, height, width):
    new_img = T.crop(img, top, left, height, width)

    crop_bbox = BoundingBox([left, top, left + width, top + height])

    new_bboxes = [bbox_crop(bbox, left, top) for bbox in bboxes if bbox in crop_bbox]

    return new_img, new_bboxes


def bbox_crop(bbox, left, top):
    new_bbox = bbox.to(BoxMode.CXCYWH)

    new_bbox[0] -= left
    new_bbox[1] -= top

    return new_bbox.to(bbox.mode)


def center_crop(img, bboxes, output_size):
    src_w, src_h = img.size
    tgt_h, tgt_w = output_size
    crop_top = int(round((src_h - tgt_h) / 2.))
    crop_left = int(round((src_w - tgt_w) / 2.))
    return crop(img, bboxes, crop_top, crop_left, tgt_h, tgt_w)


def resized_crop(img, bboxes, top, left, height, width, size, interpolation=Image.BILINEAR):
    new_img, new_bboxes = crop(img, bboxes, top, left, height, width)
    new_img, new_bboxes = resize(new_img, new_bboxes, size, interpolation)
    return new_img, new_bboxes


def to_tensor(img, bboxes):
    return T.to_tensor(img), torch.tensor(bboxes)


def normalize_boxes(img, bboxes):
    img_size = img.size
    return [bbox.normalize(img_size) for bbox in bboxes]
