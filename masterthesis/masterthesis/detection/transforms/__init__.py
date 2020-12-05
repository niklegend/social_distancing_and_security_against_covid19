import random
from abc import ABC

from PIL import Image
from torch import nn

from . import functional as D


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img, bboxes)
        return img, bboxes


class Resize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, bboxes):
        return D.resize(img, bboxes, self.size, self.interpolation)


class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, bboxes):
        return D.rotate(img, bboxes, self.angle)


class HorizontalFlip(object):

    def __call__(self, img, bboxes):
        return D.hflip(img, bboxes)


class VerticalFlip(object):

    def __call__(self, img, bboxes):
        return D.vflip(img, bboxes)


class Crop(object):

    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img, bboxes):
        return D.crop(img, bboxes, self.top, self.left, self.height, self.width)


class CenterCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img, bboxes):
        return D.center_crop(img, bboxes, self.output_size)


class ResizedCrop(object):

    def __init__(self, top, left, height, width, size, interpolation=Image.BILINEAR):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, bboxes):
        return D.resized_crop(img, bboxes, self.top, self.left, self.height, self.width, self.size,
                              self.interpolation)


class RandomTransform(nn.Module, ABC):

    def __init__(self, p=0.5):
        super(RandomTransform, self).__init__()
        self.p = p

    def forward(self, img, bboxes):
        if random.random() < self.p:
            return self._forward(img, bboxes)
        return img, bboxes

    def _forward(self, img, bboxes):
        raise NotImplementedError('_forward(img, bboxes) is not implemented!')


class RandomHorizontalFlip(RandomTransform):

    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__(p)

    def _forward(self, img, bboxes):
        return D.hflip(img, bboxes)


class RandomVerticalFlip(RandomTransform):

    def __init__(self, p=0.5):
        super(RandomVerticalFlip, self).__init__(p)

    def _forward(self, img, bboxes):
        return D.vflip(img, bboxes)


class RandomCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img, bboxes):
        # left, top, width, height
        img_w, img_h = img.size
        width, height = self.size
        if img_w == height and img_h == width:
            top, left = 0, 0
        else:
            top, left = random.randint(0, img_h - width), random.randint(0, img_w - height)

        return D.crop(img, bboxes, top, left, height, width)


class ToTensor(object):

    def __call__(self, img, bboxes):
        return D.to_tensor(img, bboxes)


class NormalizeBoxes(object):

    def __call__(self, img, bboxes):
        return D.normalize_boxes(img, bboxes)


class CategoryToId(object):

    def __init__(self, category_to_id):
        self.category_to_id = category_to_id

    def __call__(self, target):
        return self.category_to_id[target]
