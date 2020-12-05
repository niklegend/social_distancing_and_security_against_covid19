import os
from xml.etree import ElementTree

from .tokitticonverter import ToKittiConverter, Category


class KaggleToKittiConverter(ToKittiConverter):

    def __init__(
            self,
            kitti_base_dir,
            kitti_image_size,
            images_dir,
            labels_dir,
            limit,
            verbose
    ):
        super(KaggleToKittiConverter, self).__init__(kitti_base_dir, limit, kitti_image_size, verbose, 'train')

        self.images_dir = images_dir
        self.labels_dir = labels_dir

    def __call__(self):
        image_extensions = ['.jpeg', '.jpg', '.png']
        for image_name in os.listdir(self.images_dir):
            _, ext = os.path.splitext(image_name)
            if ext.lower() in image_extensions:
                labels_xml = self.get_image_metafile(image_file=image_name)
                if os.path.isfile(labels_xml):
                    labels = ElementTree.parse(labels_xml).getroot()
                    bboxes = []
                    class_names = []
                    for object_tag in labels.findall("object"):
                        cat_name = object_tag.find("name").text

                        category = Category.MASK if cat_name == 'mask' else Category.NO_MASK

                        if self.count[category] < self.limit[category]:
                            xmin = int(object_tag.find("bndbox/xmin").text)
                            xmax = int(object_tag.find("bndbox/xmax").text)
                            ymin = int(object_tag.find("bndbox/ymin").text)
                            ymax = int(object_tag.find("bndbox/ymax").text)
                            bbox = [xmin, ymin, xmax, ymax]
                            class_names.append(category)
                            bboxes.append(bbox)
                    if bboxes:
                        self.write_example(
                            image_path=os.path.join(self.images_dir, image_name),
                            class_names=class_names,
                            bboxes=bboxes)

        return self.count_mask, self.count_no_mask

    def get_image_metafile(self, image_file):
        image_name = os.path.splitext(image_file)[0]
        return os.path.join(self.labels_dir, str(image_name + '.xml'))
