# Adapted from https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py

import argparse
import json
import os

import imagesize
from masterthesis.datasets import kitti_utils as kitti
from masterthesis.utils import TimeIt


def kitti_to_json_data(kitti_split_dir):
    images_dir = os.path.join(kitti_split_dir, 'images')
    labels_dir = os.path.join(kitti_split_dir, 'labels')

    data = []

    for filename in os.listdir(images_dir):
        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt')

        width, height = imagesize.get(image_path)
        annotations = []

        kitti_annotations = kitti.read_annotation_file(label_path)
        for annotation in kitti_annotations:
            annotations.append({
                'class': annotation['type'],
                'bbox': annotation['bbox'].data
            })

        if annotations:
            data.append({
                'filename': filename,
                'width': width,
                'height': height,
                'annotations': annotations
            })

    return data


def kitti_to_json(kitti_split_dir, output_path):
    with TimeIt(f'JSON annotation file created at {output_path}'):
        data = kitti_to_json_data(kitti_split_dir)

        if data:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w') as f:
                for row in data:
                    json_str = json.dumps(row, separators=(',', ':'))
                    f.write(json_str)
                    f.write('\n')


def main(args):
    splits = os.listdir(args.input_dir)

    for split in splits:
        kitti_split_dir = os.path.join(args.input_dir, split)
        output_path = os.path.join(args.output_dir, split + '.json')

        kitti_to_json(kitti_split_dir, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('KITTI to CSV converter')

    parser.add_argument(
        '-i', '--input-dir',
        help='Path to the KITTI dataset base directory.',
        required=True
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Path to the the output directory.',
        required=True
    )

    main(parser.parse_args())
