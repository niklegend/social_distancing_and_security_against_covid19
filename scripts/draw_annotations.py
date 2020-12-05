import argparse
import os

import cv2
from masterthesis.datasets import kitti_utils as kitti
from masterthesis.detection import utils
from masterthesis.detection.boundingbox import BoxMode
from matplotlib import pyplot as plt


def draw_annotations(image_path, label_path, output_path=None):
    img = cv2.imread(image_path)

    boxes = []

    annotations = kitti.read_annotation_file(label_path)
    for annotation in annotations:
        boxes.append(annotation['bbox'].to(BoxMode.XYXY))

    out_img = img.copy()

    utils.draw_detections_on_image_array(
        img,
        boxes
    )

    plt.imshow(out_img[::-1])
    plt.show()

    if os.output_path and image_path != output_path:
        basedir = os.path.dirname(output_path)
        os.makedirs(basedir, exist_ok=True)

        cv2.imwrite(output_path, out_img)


def main(args):
    image_path = args.image_path
    label_path = args.label_path
    output_path = args.output_path

    draw_annotations(image_path, label_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image-path', required=True)
    parser.add_argument('-l', '--label-path', required=True)
    parser.add_argument('-o', '--output-path')

    args = parser.parse_args()

    main(args)
