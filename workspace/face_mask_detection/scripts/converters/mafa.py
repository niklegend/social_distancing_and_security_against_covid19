import os

import scipy.io

from .tokitticonverter import ToKittiConverter, Category


class MafaToKittiConverter(ToKittiConverter):

    def __init__(
            self,
            kitti_base_dir,
            kitti_image_size,
            annotations_path,
            images_dir,
            limit,
            stage,
            verbose
    ):
        super(MafaToKittiConverter, self).__init__(kitti_base_dir, limit, kitti_image_size, verbose, stage)

        self.annotations_path = annotations_path
        self.data = scipy.io.loadmat(self.annotations_path)
        self.images_dir = images_dir
        self.len_dataset = len(self.data["label_train"][0]) if self.train else len(self.data["LabelTest"][0])

    def __call__(self):
        for i in range(0, self.len_dataset):
            self.extract_labels(i=i)

        return self.count_mask, self.count_no_mask

    def extract_labels(self, i):
        class_names = []
        bboxes = []
        if self.train:
            train_image = self.data["label_train"][0][i]
            image_name = str(train_image[1]).strip("['']")  # Test [0]
            for i in range(0, len(train_image[2])):
                _bbox_label = train_image[2][i]  # Test[1][0]
                _category_id = _bbox_label[12]  # Occ_Type: For Train: 13th, 10th in Test
                _occlusion_degree = _bbox_label[13]
                # bbox = [_bbox_label[0], _bbox_label[1], _bbox_label[0]+_bbox_label[2], _bbox_label[1]+_bbox_label[3]]
                _left = int(_bbox_label[0])
                _top = int(_bbox_label[1])
                _width = int(_bbox_label[2])
                _height = int(_bbox_label[3])
                bbox = [_left, _top, _left + _width, _top + _height]
                category_name = None

                if _category_id != 3 and _occlusion_degree > 2:
                    category_name = Category.MASK  # Faces with Mask
                elif _category_id == 3 and _occlusion_degree < 2:
                    category_name = Category.NO_MASK  # Faces without Mask

                if category_name and self.count[category_name] < self.limit[category_name]:
                    class_names.append(category_name)
                    bboxes.append(bbox)
        else:
            test_image = self.data["LabelTest"][0][i]
            image_name = str(test_image[0]).strip("['']")  # Test [0]
            for i in range(0, len(test_image[1])):
                _bbox_label = test_image[1][i]  # Test[1][0]
                # Occ_Type: For Train: 13th, 10th in Test
                # In test Data: refer to Face_type, 5th
                _face_type = _bbox_label[4]  # Face Type
                _occ_type = _bbox_label[9]
                _occ_degree = _bbox_label[10]
                _left = int(_bbox_label[0])
                _top = int(_bbox_label[1])
                _width = int(_bbox_label[2])
                _height = int(_bbox_label[3])
                bbox = [_left, _top, _left + _width, _top + _height]
                category_name = None

                if _face_type == 1 and _occ_type != 3 and _occ_degree > 2:
                    category_name = Category.MASK
                elif _face_type == 2:
                    category_name = Category.NO_MASK

                if category_name and self.count[category_name] < self.limit[category_name]:
                    class_names.append(category_name)
                    bboxes.append(bbox)

        if bboxes:
            self.write_example(
                image_path=os.path.join(self.images_dir, image_name),
                class_names=class_names,
                bboxes=bboxes)
