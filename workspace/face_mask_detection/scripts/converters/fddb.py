import math
import os
import re

from masterthesis.detection.utils import ellipse_to_bbox, Point
from .tokitticonverter import ToKittiConverter, Category


class FddbToKittiConverter(ToKittiConverter):

    def __init__(
            self,
            kitti_base_dir,
            kitti_image_size,
            base_dir,
            labels_dir,
            limit,
            verbose
    ):
        super(FddbToKittiConverter, self).__init__(kitti_base_dir, limit, kitti_image_size, verbose, 'train')

        self.base_dir = base_dir
        self.labels_dir = labels_dir

    def __call__(self):
        for root, dirs, files in os.walk(self.labels_dir):
            for file in files:
                if file.endswith('ellipseList.txt'):
                    file_name = os.path.join(root, file)
                    self.mat2data(read_file=file_name)

        return self.count_mask, self.count_no_mask

    def mat2data(self, read_file):
        strings = ("2002/", "2003/")
        with open(read_file, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines)):
                line = lines[i]
                if any(s in line for s in strings) and self.count_no_mask < self.no_mask_limit:
                    image_file_location = line.strip('\n')
                    num_faces_line = re.search(r"(\d+).*?", lines[i + 1])
                    num_faces = int(num_faces_line.group(1))
                    image_name = image_file_location + '.jpg'
                    category_name = Category.NO_MASK
                    bboxes = []
                    class_names = []
                    for j in range(1, num_faces + 1):
                        faces = str(lines[i + j + 1]).split()

                        face_annotations = faces[:5]

                        major_axis_radius = float(face_annotations[0])
                        minor_axis_radius = float(face_annotations[1])
                        angle = float(face_annotations[2])
                        cx = float(face_annotations[3])
                        cy = float(face_annotations[4])

                        bbox = ellipse_to_bbox(Point(cx, cy), Point(major_axis_radius, minor_axis_radius), angle)
                        bboxes.append(bbox)
                        class_names.append(category_name)
                    if bboxes:
                        self.write_example(
                            image_path=os.path.join(self.base_dir, image_name),
                            class_names=class_names,
                            bboxes=bboxes
                        )
        return self.count_mask, self.count_no_mask
