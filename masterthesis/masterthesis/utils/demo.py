import time

import cv2

from . import FpsCounter, TimeIt
from .visualization_utils import draw_detections_on_image_array


def run_on_image_fn(
        infer_fn,
        display_names=None,
        colors=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=None,
        min_score_threshold=None,
        line_thickness=3,
        color_mode='bgr'
):
    def func(img):
        boxes, classes, scores = infer_fn(img)

        draw_detections_on_image_array(
            img=img,
            boxes=boxes,
            classes=classes,
            scores=scores,
            display_names=display_names,
            colors=colors,
            use_normalized_coordinates=use_normalized_coordinates,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_threshold=min_score_threshold,
            line_thickness=line_thickness,
            color_mode=color_mode
        )

        return img

    return func


def run_on_video(video_path, run_on_image, output_path=None):
    cap = cv2.VideoCapture(video_path)

    writer = None
    if output_path:
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, int(fourcc), int(fps), (int(width), int(height)))

    with FpsCounter() as counter:
        start_time = time.time()
        total_frames = 0

        while cap.isOpened():
            # Capture frame-by-frame
            grabbed, frame = cap.read()

            if not grabbed:
                break

            out_img = run_on_image(frame)

            if writer:
                writer.write(out_img)

            total_frames += 1

            fps = counter.update()
            if fps:
                print(f'Inference is running at {fps:.2f} FPS')

        elapsed_time = time.time() - start_time

    # When everything done, release the capture
    cap.release()
    if writer:
        writer.release()

    print(f'Ran inference on {total_frames} frames in {TimeIt.format_elapsed(elapsed_time)}.')
    print(f'Average FPS: {total_frames / elapsed_time:.2f}')
