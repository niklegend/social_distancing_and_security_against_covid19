import os

from masterthesis.datasets.kitti_utils import ToKittiBaseConverter


class _Category(type):

    @property
    def MASK(self): return 'Mask'

    @property
    def NO_MASK(self): return 'No-Mask'


class Category(metaclass=_Category):
    pass


class ToKittiConverter(ToKittiBaseConverter):

    def __init__(self, base_dir, limit, kitti_image_size=None, verbose=False, stage='train'):
        super(ToKittiConverter, self).__init__(
            kitti_images_dir=os.path.join(base_dir, stage, 'images'),
            kitti_labels_dir=os.path.join(base_dir, stage, 'labels'),
            limit=limit,
            kitti_image_size=kitti_image_size,
            verbose=verbose,
            strict=True
        )

        self.train = stage == 'train'

    def log(self, image_path, w, h, bbox):
        from . import log
        log(image_path, w, h, bbox)

    @property
    def mask_limit(self):
        return self.limit[Category.MASK]

    @property
    def no_mask_limit(self):
        return self.limit[Category.NO_MASK]

    @property
    def count_mask(self):
        return self.count[Category.MASK]

    @property
    def count_no_mask(self):
        return self.count[Category.NO_MASK]

    @staticmethod
    def get_stage_string(is_train):
        return 'train' if is_train else 'test'
