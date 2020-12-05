import random

import numpy as np
from PIL import ImageColor
from bbox_visualizer import bbox_visualizer as bbv
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


def draw_detections_on_image_array(
        img,
        boxes,
        classes=None,
        scores=None,
        display_names=None,
        colors=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=None,
        min_score_threshold=0.5,
        line_thickness=3,
        color_mode='bgr'
):
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

    for idx in range(boxes.shape[0]):
        if scores is not None and scores[idx] < min_score_threshold:
            continue

        if max_boxes_to_draw is not None and drawn_boxes >= max_boxes_to_draw:
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

        # Draw bounding box on image
        img[:, :, :] = bbv.draw_rectangle(
            img=img,
            bbox=box,
            bbox_color=color,
            thickness=line_thickness
        )

        # Process display name
        label = None if agnostic_mode else display_names[class_idx]

        # Process score
        if scores is not None:
            score = f'{round(scores[idx] * 100, ndigits=2)}%'
            if label:
                label += ' ' + score
            else:
                label = score

        # Draw label on image
        if label:
            img[:, :, :] = bbv.add_label(
                img=img,
                label=label,
                bbox=box,
                text_bg_color=(0, 0, 0),
                text_color=color,
                top=False
            )

        drawn_boxes += 1


def make_grid(img_array, num_rows=None, num_cols=None):
    batch_size, height, width, _ = img_array.shape

    assert num_rows or num_cols, 'At least num_rows or num_cols must be passed, or both.'

    if num_rows is not None:
        num_cols = batch_size // num_rows
    elif num_cols is not None:
        num_rows = batch_size // num_cols
    assert batch_size == num_rows * num_cols

    return img_array \
        .reshape(num_rows, num_cols, height, width, -1) \
        .swapaxes(1, 2) \
        .reshape(height * num_rows, width * num_cols, -1)
