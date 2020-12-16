import random

import cv2
import numpy as np
from PIL import ImageColor
from colormap import hex2rgb

from . import find_largest_divisor

__colormap = None

supported_color_modes = [
    'hex',
    'rgb',
    'bgr'
]


def random_color(color_mode='hex', normalized=False):
    assert color_mode in supported_color_modes, f'Invalid color mode: {color_mode}'
    if normalized:
        assert color_mode in ['rgb', 'bgr']

    global __colormap

    if not __colormap:
        __colormap = ImageColor.colormap.values()

    color = random.choice(__colormap)

    if color_mode == 'hex':
        return color

    color = hex2rgb(color, normalise=normalized)

    if color_mode == 'bgr':
        color = color[::-1]

    return color


def draw_box_on_image_array(
        img,
        box,
        color,
        line_thickness=3
):
    xmin, ymin, xmax, ymax = box
    cv2.rectangle(
        img,
        (xmin, ymin),
        (xmax, ymax),
        color,
        line_thickness
    )


def draw_label_on_image_array(
        img,
        text,
        top_left,
        font,
        color,
        bg_color=None,
        font_scale=1,
        thickness=2
):
    # From https://stackoverflow.com/questions/60674501#answer-65146731
    xmin, ymin = top_left
    text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]

    if bg_color:
        xmax, ymax = xmin + text_width - 1, ymin + text_height + 1
        draw_box_on_image_array(img, [xmin, ymin, xmax, ymax], bg_color, -1)

    cv2.putText(
        img,
        text,
        (xmin, ymin + text_height + font_scale - 1),
        font,
        font_scale,
        color,
        thickness
    )


def draw_detections_on_image_array(
        img,
        boxes,
        classes=None,
        scores=None,
        display_names=None,
        colors=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=None,
        min_score_threshold=None,
        line_thickness=3,
        color_mode='bgr'
):
    if boxes.shape[0] == 0:
        return

    color_mode = color_mode.lower()
    assert all([
        color_mode in ['rgb', 'bgr'],
        boxes.shape[0] > 0,
        scores is None or boxes.shape[0] == scores.shape[0]
    ])

    agnostic_mode = classes is None

    if not agnostic_mode:
        assert all([
            len(boxes) == len(classes),
            display_names is not None,
            len(display_names) > 0
        ])

        if colors is not None:
            assert len(display_names) == len(colors)
        else:
            colors = [
                random_color(color_mode=color_mode)
                for _ in range(len(display_names))
            ]
    else:
        assert all([
            display_names is None,
            colors is None or len(colors) == 1
        ])

    if use_normalized_coordinates:
        height, width, _ = img.shape
        boxes[:, ::2] *= width  # xmin and xmax
        boxes[:, 1::2] *= height  # ymin and ymax

    boxes = boxes.astype(np.uint16)

    drawn_boxes = 0

    check_score = min_score_threshold is not None and scores is not None
    check_max_boxes_to_draw = max_boxes_to_draw is not None

    for idx in range(boxes.shape[0]):
        if check_score and scores[idx] < min_score_threshold:
            continue

        if check_max_boxes_to_draw and drawn_boxes >= max_boxes_to_draw:
            break

        # Get class index
        class_idx = None if agnostic_mode else classes[idx]

        # Get color
        if agnostic_mode:
            color = colors[0] if colors else (255, 255, 255)
        else:
            color = colors[class_idx] if colors else random_color(color_mode=color_mode)

        # Process bounding box
        box = boxes[idx]

        draw_box_on_image_array(img, box, color, line_thickness)

        # Process display name
        text = None if agnostic_mode else display_names[class_idx]

        # Process score
        if scores is not None:
            score = f'{scores[idx] * 100:.2f}%'
            if text:
                text += ' ' + score
            else:
                text = score

        if text:
            draw_label_on_image_array(
                img,
                text,
                box[:2],
                cv2.FONT_HERSHEY_SIMPLEX,
                color=color,
                bg_color=(0, 0, 0)
            )

        drawn_boxes += 1


def make_grid(img_array, num_rows=None, num_cols=None, priority='columns'):
    # https://stackoverflow.com/questions/42040747#answer-42041135
    assert priority in ['rows', 'columns']
    num_images, height, width, _ = img_array.shape

    if num_rows is None and num_cols is None:
        n = find_largest_divisor(num_images)
        if n >= num_images // 2:
            num_cols = n
        else:
            num_rows = n

    if num_rows is not None:
        num_cols = num_images // num_rows
    elif num_cols is not None:
        num_rows = num_images // num_cols

    assert num_images == num_rows * num_cols

    if priority == 'columns':
        if num_rows > num_cols:
            num_rows, num_cols = num_cols, num_rows
    elif num_cols > num_rows:
        num_cols, num_rows = num_rows, num_cols

    return img_array \
        .reshape(num_rows, num_cols, height, width, -1) \
        .swapaxes(1, 2) \
        .reshape(height * num_rows, width * num_cols, -1)
