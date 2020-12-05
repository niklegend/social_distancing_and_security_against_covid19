# Adapted from https://github.com/NVIDIA-AI-IOT/face-mask-detection/blob/master/data2kitti.py

import argparse
import json
import logging
import os
import sys

import converters
from masterthesis.utils import TimeIt

DEFAULT_KITTI_IMAGE_SIZE = (960, 544)


actions = ['train', 'test', 'check_labels']


def print_summary(d, label=None):
    print('-' * 40)
    if label:
        print(label)
        print('=' * 40)
    for k, v in d.items():
        print(f'{k} count: {v}')
    print('-' * 40)


def data_to_kitti(
        kitti_base_dir,
        widerperson_base_dir=None,
        kitti_image_size=None,
        category_limit=sys.maxsize,
        label_filename='000_1OC3DT',
        action=None,
        verbose=False
):
    if action not in actions:
        raise ValueError(f'Unrecognized action: \'{action}\'')
    if kitti_image_size is None:
        kitti_image_size = DEFAULT_KITTI_IMAGE_SIZE

    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    # To check if labels are converted in right format
    if action == 'check_labels':
        if not label_filename:
            raise RuntimeError('Label filename cannot be empty when checking labels.')

        with TimeIt():
            from converters.check_labels import test_labels
            # Check from train directory
            test_labels(kitti_base_dir=kitti_base_dir + '/train/', file_name=label_filename)
    else:
        from converters import Category

        is_train = action == 'train'

        category_limit_dict = {
            Category.PERSON: category_limit,
        }

        def update_statistics(label=None):
            nonlocal category_limit_dict

            category_limit_dict[Category.PERSON] -= count_people

            print_summary({k: category_limit - v for k, v in category_limit_dict.items()}, label=label)

        with TimeIt(f'{action} dataset conversion complete'):
            previous = False

            def new_line():
                nonlocal previous
                if previous:
                    print()
                previous = True

            # ----------------------------------------
            # WiderPerson Dataset Conversion
            # ----------------------------------------
            if widerperson_base_dir:
                from converters.widerperson import WiderPersonToKittiConverter

                with TimeIt(f'WiderPerson {action} dataset conversion complete'):
                    new_line()
                    print(f'Converting WiderPerson {action} dataset to KITTI...')

                    converter = WiderPersonToKittiConverter(
                        kitti_base_dir=kitti_base_dir,
                        kitti_image_size=kitti_image_size,
                        widerperson_base_dir=widerperson_base_dir,
                        limit=category_limit_dict,
                        stage=action,
                        verbose=verbose
                    )

                    count_people = converter()
                    update_statistics('Summary after processing WiderPerson')


def main(args):
    if args.check_labels:
        action = 'check_labels'
    elif args.train:
        action = 'train'
    else:
        action = 'test'

    logs_path = args.logs_path

    if logs_path:
        converters.logs = []

    data_to_kitti(
        kitti_base_dir=args.kitti_base_dir,
        widerperson_base_dir=args.widerperson_base_dir,
        kitti_image_size=args.kitti_image_size,
        category_limit=args.category_limit,
        label_filename=args.label_filename,
        action=action,
        verbose=args.verbose
    )

    if logs_path and converters.logs:
        with open(logs_path, 'w') as f:
            json.dump(converters.logs, f, indent=2)
            f.write('\n')
            print(f'Logs written to {logs_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--widerperson-base-dir', help='path to WiderPerson dataset train and validation images', type=str,
                        default=None)
    parser.add_argument('--kitti-base-dir', help='path to save converted data set', type=str, required=True)
    parser.add_argument('--category-limit', default=sys.maxsize, help='data limit for TLT', type=int)
    parser.add_argument('--kitti-image-size', nargs=2, default=DEFAULT_KITTI_IMAGE_SIZE, help='TLT input dimensions',
                        type=int)
    parser.add_argument('--label-filename', default='000_1OC3DT',
                        help='File name for label checking', type=str)

    parser.add_argument('--logs-path', default=os.getcwd(), help='Path to the logs file.')

    parser.add_argument('--verbose', action='store_true')

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--train', help='Convert Training dataset to KITTI', action='store_true')
    data_group.add_argument('--test', help='Convert test dataset to KITTI', action='store_true')
    data_group.add_argument('--check-labels', dest='check_labels', help='Check if Converted dataset is right',
                            action='store_true')

    args = parser.parse_args()

    main(args)
