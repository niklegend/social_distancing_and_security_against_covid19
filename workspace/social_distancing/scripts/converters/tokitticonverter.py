import os

from masterthesis.datasets.kitti_utils import ToKittiBaseConverter


class _Category(type):

    @property
    def PERSON(self): return 'Person'


class Category(metaclass=_Category):
    pass


class ToKittiConverter(ToKittiBaseConverter):

    def __init__(self, base_dir, limit, kitti_image_size=None, strict=False, verbose=False, stage='train'):
        super(ToKittiConverter, self).__init__(
            kitti_images_dir=os.path.join(base_dir, stage, 'images'),
            kitti_labels_dir=os.path.join(base_dir, stage, 'labels'),
            limit=limit,
            kitti_image_size=kitti_image_size,
            strict=strict,
            verbose=verbose
        )

        self.train = stage == 'train'

    def log(self, image_path, w, h, bbox):
        from . import log
        log(image_path, w, h, bbox.data)

    @property
    def person_limit(self):
        return self.limit[Category.PERSON]

    @property
    def count_person(self):
        return self.count[Category.PERSON]

    @staticmethod
    def get_stage_string(is_train):
        return 'train' if is_train else 'test'
