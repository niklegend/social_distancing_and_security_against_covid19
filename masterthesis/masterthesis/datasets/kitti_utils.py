import logging
import os
from abc import ABC
from collections import UserList, defaultdict
from typing import List, Union

import imagesize
from PIL import Image

from .utils import LineReader
from ..detection.boundingbox import BoundingBox

Arithmetic = Union[float, int]


def is_string(x):
    return isinstance(x, str)


def is_arithmetic(x):
    return any([
        isinstance(x, int),
        isinstance(x, float)
    ])


def is_array(items, expected_len, item_fn):
    return all([
        any([isinstance(items, list), isinstance(items, UserList)]),
        len(items) == expected_len,
        all([item_fn(item) for item in items])
    ])


# https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt


def create_annotation(
        class_name: str,
        truncated: Arithmetic = None,
        occluded: Arithmetic = None,
        alpha: Arithmetic = None,
        bbox: Union[List[Arithmetic], BoundingBox] = None,
        dimensions: List[Arithmetic] = None,
        location: List[Arithmetic] = None,
        rotation_y: Arithmetic = None,
        score: Arithmetic = None
):
    annotation = {}

    def put(key, value, assert_fn, default=None, optional=False):
        nonlocal annotation
        if value:
            assert assert_fn(value)
        else:
            if not optional and default is None:
                raise ValueError('Value is required.')
            value = default
        if value is not None:
            annotation[key] = value

    put('type', class_name, is_string)
    put('truncated', truncated, is_arithmetic, 0)
    put('occluded', occluded, is_arithmetic, 0)
    put('alpha', alpha, is_arithmetic, 0)
    put('bbox', bbox, lambda x: is_array(x, 4, is_arithmetic), [0] * 4)
    put('dimensions', dimensions, lambda x: is_array(x, 3, is_arithmetic), [0] * 3)
    put('location', location, lambda x: is_array(x, 3, is_arithmetic), [0] * 3)
    put('rotation_y', rotation_y, is_arithmetic, 0)
    put('score', score, is_arithmetic, optional=True)

    return annotation


def write_annotation(out_file, annotation, truncated=False):
    columns = []
    tostr = (lambda x: str(int(x))) if truncated else str

    def write(key):
        if key in annotation:
            value = annotation[key]
            if isinstance(value, list) or isinstance(value, UserList):
                for item in value:
                    columns.append(tostr(item))
            elif isinstance(value, str):
                columns.append(value)
            else:
                columns.append(tostr(value))

    write('type')
    write('truncated')
    write('occluded')
    write('alpha')
    write('bbox')
    write('dimensions')
    write('location')
    write('rotation_y')
    write('score')

    out_file.write(' '.join(columns))


def read_annotation(annotation_str):
    reader = LineReader(annotation_str, r'[\t ]+')
    assert reader.has_columns(15)

    annotation = {}

    def read(key, count=1, func=None, finalizer=None):
        nonlocal annotation
        if reader.has_columns(count):
            value = reader(func=func, count=count)
            if finalizer:
                value = finalizer(value)
            annotation[key] = value

    read('type')
    read('truncated', func=float)
    read('occluded', func=int)
    read('alpha', func=float)
    read('bbox', 4, func=float, finalizer=BoundingBox)
    read('dimensions', 3, func=float)
    read('location', 3, func=float)
    read('rotation_y', func=float)
    read('score', func=float)

    return annotation


def read_annotation_file(annotation_path):
    with open(annotation_path, 'r') as f:
        annotations = []
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            annotations.append(read_annotation(line))
        return annotations


class ToKittiBaseConverter(ABC):

    def __init__(self, kitti_images_dir, kitti_labels_dir, limit, kitti_image_size, strict=False, verbose=False):
        self.kitti_images_dir = kitti_images_dir
        self.kitti_labels_dir = kitti_labels_dir
        self.limit = limit
        self.kitti_image_size = kitti_image_size
        self.strict = strict
        self.verbose = verbose

        self.count = defaultdict(lambda: 0)

        os.makedirs(self.kitti_labels_dir, exist_ok=True)
        os.makedirs(self.kitti_images_dir, exist_ok=True)

    def write_example(self, image_path, class_names, bboxes):
        assert len(bboxes) == len(class_names), f'The number of bounding boxes ({len(bboxes)}) differs from the ' \
                                                f'number of classes ({len(class_names)}).'

        if len(bboxes) > 0:
            annotations = []
            img_size = imagesize.get(image_path)

            img_bbox = BoundingBox([0, 0, *img_size])

            filename = os.path.splitext(os.path.basename(image_path))[0]
            local_count = defaultdict(lambda: 0)

            for class_name, bbox in zip(class_names, bboxes):
                bbox = BoundingBox(bbox)

                if self.count[class_name] < self.limit[class_name]:
                    # check if bounding box is valid and within image bounds
                    if bbox and bbox in img_bbox:
                        local_count[class_name] += 1

                        if self.kitti_image_size:
                            bbox = bbox.resize(img_size, self.kitti_image_size)

                        # Append KITTI annotation
                        annotations.append(create_annotation(class_name, bbox=bbox))
                    else:
                        w, h = img_size
                        self.log(image_path, w, h, bbox.data)
                        if self.verbose:
                            logging.warning(f'{bbox} is not a valid bounding box (image size {w}x{h})')
                elif self.verbose:
                    logging.info(f'Category limit reached for \'{class_name}\' category')

            # If strict mode is enabled, write example only if all annotations were correct and within category limits
            if not self.strict or len(bboxes) == len(annotations):
                # Update category count
                for k, v in local_count.items():
                    self.count[k] += v

                # Save image
                img = Image.open(image_path).convert('RGB')

                if self.kitti_image_size:
                    img = img.resize(self.kitti_image_size)

                img.save(os.path.join(self.kitti_images_dir, filename + '.jpg'), 'JPEG')

                # Save annotations
                with open(os.path.join(self.kitti_labels_dir, filename + '.txt'), 'w') as f:
                    for annotation in annotations:
                        write_annotation(f, annotation, truncated=True)
                        f.write('\n')

    def log(self, image_path, w, h, bbox):
        raise NotImplementedError('log is not implemented.')
