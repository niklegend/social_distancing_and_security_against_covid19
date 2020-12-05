import argparse
import logging
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from argparse import Namespace
from masterthesis.detection.utils import visualize_detections
from masterthesis.utils import TimeIt
from masterthesis.utils.demo import run_on_video


# INFO and WARNING messages are not printed
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Range(object):

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __contains__(self, item):
        return self.start <= item < self.end

    def __iter__(self):
        yield self

    def __repr__(self):
        return 'Range(start={0}, end={1})'.format(self.start, self.end)


tf.get_logger().setLevel(logging.ERROR)

DEFAULT_MIN_SCORE_THRESHOLD = 0.6

DEFAULT_NMS_MAX_OUTPUT_SIZE = 100
DEFAULT_NMS_IOU_THRESHOLD = 0.5
DEFAULT_NMS_SCORE_THRESHOLD = 0.005

DISPLAY_NAMES = {
    1: 'Mask',
    2: 'No mask'
}

COLORS = {
    1: (0, 255, 0),
    2: (255, 0, 0)
}


def __run_on_image(
        detect_fn,
        img,
        min_score_threshold=DEFAULT_MIN_SCORE_THRESHOLD,
        nms_args=None
):
    # img shape = (H, W, D) D order = (B, G, R) in cv2
    img = img[:, :, ::-1]  # convert BGR -> RGB
    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    # print(detections)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections').numpy().item())
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    # detections['num_detections'] = num_detections

    boxes = detections['detection_boxes']
    classes = detections['detection_classes'].astype(np.uint8)
    scores = detections['detection_scores']

    if nms_args:
        selected_indices = tf.image.non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size=nms_args.max_output_size,
            iou_threshold=nms_args.iou_threshold,
            score_threshold=nms_args.score_threshold
        )

        boxes = boxes[selected_indices]
        classes = classes[selected_indices]
        scores = scores[selected_indices]

    # print(f'detection_boxes: {boxes.shape}')
    # print(f'detection_classes: {classes}')
    # print(f'detection_classes: {classes.shape}')
    # print(f'detection_scores: {scores}')
    # print(f'detection_scores: {scores.shape}')

    out_img = img.copy()

    visualize_detections(
        out_img,
        boxes=boxes,
        classes=classes,
        scores=scores,
        display_names=DISPLAY_NAMES,
        colors=COLORS,
        min_score_threshold=min_score_threshold,
        use_normalized_coordinates=True
    )

    return out_img[:, :, ::-1]  # convert RGB -> BGR


def main(args):
    print('Loading saved model...')
    with TimeIt('Saved model has been loaded successfully'):
        model = tf.saved_model.load(args.saved_model_dir)
        detect_fn = model.signatures['serving_default']

    output_dir = args.output_dir

    filepath = args.image_path if args.image_path else args.video_path
    output_path = os.path.join(output_dir, 'annotated_' + os.path.basename(filepath))

    nms_args = None if args.nms_disabled else Namespace(
        max_output_size=args.nms_max_output_size,
        iou_threshold=args.nms_iou_threshold,
        score_threshold=args.nms_score_threshold,
    )

    def run_on_image(x):
        return __run_on_image(
            detect_fn=detect_fn,
            img=x,
            min_score_threshold=args.min_score_threshold,
            nms_args=nms_args
        )

    print(f'Running inference on {filepath}')
    with TimeIt(f'Inference results have been saved to {output_path}'):
        if args.image_path:
            img = cv2.imread(filepath)
            out_img = run_on_image(img)

            cv2.imwrite(output_path, out_img)

        elif args.video_path:
            run_on_video(
                filepath,
                run_on_image=run_on_image,
                output_path=output_path
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--saved-model-dir', required=True)

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video-path')
    input_group.add_argument('--image-path')

    parser.add_argument('--output-dir', default=os.getcwd())

    ########################################
    # Visualization arguments
    ########################################

    parser.add_argument(
        '--min-score-threshold',
        default=DEFAULT_MIN_SCORE_THRESHOLD,
        type=float,
        choices=Range(0, 1)
    )

    ########################################
    # Non-maximum=suppression arguments
    ########################################

    parser.add_argument(
        '--nms-disabled',
        action='store_true'
    )

    parser.add_argument(
        '--nms-max-output-size',
        default=DEFAULT_NMS_MAX_OUTPUT_SIZE,
        type=int,
        choices=Range(1, sys.maxsize)
    )

    parser.add_argument(
        '--nms-iou-threshold',
        default=DEFAULT_NMS_IOU_THRESHOLD,
        type=float,
        choices=Range(0, 1)
    )

    parser.add_argument(
        '--nms-score-threshold',
        default=DEFAULT_NMS_SCORE_THRESHOLD,
        type=float,
        choices=Range(0, 1)
    )

    args = parser.parse_args()
    main(args)
