import argparse
import os

from masterthesis.utils import TimeIt
from torchvision.datasets.utils import download_and_extract_archive

BASE_URL = 'http://download.tensorflow.org/models/object_detection'


def tf1_model_url(model_name):
    return BASE_URL + '/' + model_name + '.tar.gz'


tf1_model_names = {
    'ssd_mobilenet_v1_coco_2018_01_28',
    'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03',
    'ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18',
    'ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18',
    'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03',
    'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
    'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
    'ssd_mobilenet_v2_coco_2018_03_29',
    'ssd_mobilenet_v2_quantized_300x300_coco_2018_09_14',
    'ssdlite_mobilenet_v2_coco_2018_05_09',
    'ssd_inception_v2_coco_2018_01_28',
    'faster_rcnn_inception_v2_coco_2018_01_28',
    'faster_rcnn_resnet50_coco_2018_01_28',
    'faster_rcnn_resnet50_lowproposals_coco_2018_01_28',
    'rfcn_resnet101_coco_2018_01_28',
    'faster_rcnn_resnet101_coco_2018_01_28',
    'faster_rcnn_resnet101_lowproposals_coco_2018_01_28',
    'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28',
    'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28',
    'faster_rcnn_nas_coco_2018_01_28',
    'faster_rcnn_nas_lowproposals_coco_2018_01_28',
    'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28',
    'mask_rcnn_inception_v2_coco_2018_01_28',
    'mask_rcnn_resnet101_atrous_coco_2018_01_28',
    'mask_rcnn_resnet50_atrous_coco_2018_01_28',
    'faster_rcnn_resnet101_kitti_2018_01_28',
    'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28',
    'faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28',
    'facessd_mobilenet_v2_quantized_320x320_open_image_v4',
    'faster_rcnn_resnet101_fgvc_2018_07_19',
    'faster_rcnn_resnet50_fgvc_2018_07_19',
    'faster_rcnn_resnet101_ava_v2.1_2018_04_30'
}
"""
JavaScript function used to retrieve model names in python set syntax from tables at 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md.

(function () {
    console.log('{\n' + Array.from(document.getElementsByTagName('tbody'))
        .flatMap((table) => Array.from(table.rows).map((row) => row.cells[0]))
        .map((cell) => cell.getElementsByTagName('a')[0].getAttribute('href'))
        .map((url) => url.split('/'))
        .map((splits) => splits[splits.length - 1].split('.')[0])
        .map(filename => '    \'' + filename + '\'')
        .join(',\n') + '\n}');
})();
"""


def tf2_model_url(model_name):
    model_date = tf2_model_name_to_model_date[model_name]
    return BASE_URL + '/tf2/' + model_date + '/' + model_name + '.tar.gz'


tf2_model_name_to_model_date = {
    'centernet_hg104_512x512_coco17_tpu-8': '20200713',
    'centernet_hg104_512x512_kpts_coco17_tpu-32': '20200711',
    'centernet_hg104_1024x1024_coco17_tpu-32': '20200713',
    'centernet_hg104_1024x1024_kpts_coco17_tpu-32': '20200711',
    'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8': '20200711',
    'centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8': '20200711',
    'centernet_resnet101_v1_fpn_512x512_coco17_tpu-8': '20200711',
    'centernet_resnet50_v2_512x512_coco17_tpu-8': '20200711',
    'centernet_resnet50_v2_512x512_kpts_coco17_tpu-8': '20200711',
    'efficientdet_d0_coco17_tpu-32': '20200711',
    'efficientdet_d1_coco17_tpu-32': '20200711',
    'efficientdet_d2_coco17_tpu-32': '20200711',
    'efficientdet_d3_coco17_tpu-32': '20200711',
    'efficientdet_d4_coco17_tpu-32': '20200711',
    'efficientdet_d5_coco17_tpu-32': '20200711',
    'efficientdet_d6_coco17_tpu-32': '20200711',
    'efficientdet_d7_coco17_tpu-32': '20200711',
    'ssd_mobilenet_v2_320x320_coco17_tpu-8': '20200711',
    'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8': '20200711',
    'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8': '20200711',
    'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8': '20200711',
    'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8': '20200711',
    'ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8': '20200711',
    'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8': '20200711',
    'ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8': '20200711',
    'ssd_resnet152_v1_fpn_640x640_coco17_tpu-8': '20200711',
    'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8': '20200711',
    'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8': '20200711',
    'faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8': '20200711',
    'faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8': '20200711',
    'faster_rcnn_resnet101_v1_640x640_coco17_tpu-8': '20200711',
    'faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8': '20200711',
    'faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8': '20200711',
    'faster_rcnn_resnet152_v1_640x640_coco17_tpu-8': '20200711',
    'faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8': '20200711',
    'faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8': '20200711',
    'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8': '20200711',
    'faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8': '20200711',
    'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8': '20200711',
    'extremenet': '20200711'
}
"""
JavaScript function used to retrieve model names and model dates in python dictionary syntax from 
table at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md.

(function () {
    console.log('{\n' + Array.from(document.getElementsByTagName('tbody'))
        .flatMap((table) => Array.from(table.rows).map((row) => row.cells[0]))
        .map((cell) => cell.getElementsByTagName('a')[0].getAttribute('href'))
        .map((url) => url.split('/'))
        .map((splits) => `'${splits[splits.length - 1].split('.')[0]}': '${splits[splits.length - 2]}'`)
        .map(entry => '    ' + entry)
        .join(',\n') + '\n}');
})();
"""

tf2_model_names = tf2_model_name_to_model_date.keys()


def download_model(url, download_root):
    os.makedirs(os.path.dirname(download_root), exist_ok=True)

    with TimeIt(f'Pre-trained model has been downloaded at {download_root}'):
        download_and_extract_archive(
            url,
            download_root=download_root,
            remove_finished=True
        )


def main(args):
    model_name = args.model_name
    download_root = args.download_root

    if args.tf1 and model_name in tf1_model_names:
        download_model(tf1_model_url(model_name), download_root)
    elif args.tf2 and model_name in tf2_model_names:
        download_model(tf2_model_url(model_name), download_root)
    else:
        raise ValueError(f'Could not find \'{model_name}\'')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', help='Name of the model to be downloaded.')
    parser.add_argument(
        '-d', '--download-root',
        help='Directory where the model will be downloaded',
        default=os.getcwd()
    )

    tf_group = parser.add_mutually_exclusive_group(required=True)
    tf_group.add_argument(
        '--tf1',
        action='store_true',
        help='Download TensorFlow Object Detection API 1.13.0 pre-trained model.'
    )
    tf_group.add_argument(
        '--tf2',
        action='store_true',
        help='Download TensorFlow Object Detection API 2.3.0 pre-trained model.'
    )

    args = parser.parse_args()

    main(args)
