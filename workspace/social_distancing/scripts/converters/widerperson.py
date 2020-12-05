import os
from enum import IntEnum

from masterthesis.datasets.utils import LineReader

from .tokitticonverter import ToKittiConverter, Category


class WiderPersonCategory(IntEnum):
    PEDESTRIAN = 1
    RIDER = 2
    PARTIALLY_VISIBLE = 3
    IGNORE_REGION = 4
    CROWD = 5


class WiderPersonToKittiConverter(ToKittiConverter):

    def __init__(
            self,
            kitti_base_dir,
            kitti_image_size,
            widerperson_base_dir,
            limit,
            stage,
            verbose
    ):
        super(WiderPersonToKittiConverter, self).__init__(
            kitti_base_dir,
            limit,
            kitti_image_size,
            verbose=verbose,
            stage=stage
        )

        self.annotations_dir = os.path.join(widerperson_base_dir, 'Annotations')
        self.images_dir = os.path.join(widerperson_base_dir, 'Images')

        filename = 'train' if stage == 'train' else 'val'
        with open(os.path.join(widerperson_base_dir, filename + '.txt'), 'r') as f:
            self.image_ids = list(map(lambda x: x.strip(), f.readlines()))

    def __call__(self):
        for image_id in self.image_ids:
            image_filename = image_id + '.jpg'
            if self.count_person < self.person_limit:
                bboxes = []
                class_names = []

                with open(os.path.join(self.annotations_dir, image_filename + '.txt'), 'r') as f:
                    # Read number of annotations
                    count = int(f.readline())
                    for _ in range(count):
                        reader = LineReader(f.readline().strip(), r'[\t ]+')

                        # Read class label
                        category = reader(func=int)

                        if category not in [
                            WiderPersonCategory.CROWD,
                            WiderPersonCategory.IGNORE_REGION
                        ]:
                            # Read bbox
                            bbox = reader(4, func=float)

                            bboxes.append(bbox)
                            class_names.append(Category.PERSON)

                if bboxes:
                    self.write_example(
                        image_path=os.path.join(self.images_dir, image_filename),
                        class_names=class_names,
                        bboxes=bboxes
                    )

        return self.count_person
