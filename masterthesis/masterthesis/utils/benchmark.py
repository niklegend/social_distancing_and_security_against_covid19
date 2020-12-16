import time

import cv2
import numpy as np

from . import FpsCounter, TimeIt, time_it

preprocess_times = []
detect_times = []
postprocess_times = []
draw_times = []

read_times = []
write_times = []
image_times = []
frame_times = []


def print_statistics(times, label, total_frames=None):
    if len(times):
        output = f'Average {label} time: {TimeIt.format_elapsed(np.mean(times), ndigits=9)}'
        if total_frames:
            output += f', {label} estimated FPS: {total_frames / np.sum(times):.2f}'

        print(output)


def run_on_video(video_path, run_on_image, output_path=None):
    global detect_times, preprocess_times, postprocess_times, draw_times
    preprocess_times = []
    detect_times = []
    postprocess_times = []
    draw_times = []

    global read_times, write_times, image_times, frame_times
    read_times = []
    write_times = []
    image_times = []
    frame_times = []

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
            frame_st = time.time()

            # Capture frame-by-frame
            elapsed_time, (grabbed, frame) = time_it(cap.read)
            read_times.append(elapsed_time)

            if not grabbed:
                break

            elapsed_time, out_img = time_it(run_on_image, frame)
            image_times.append(elapsed_time)

            if writer:
                write_times.append(time_it(writer.write, out_img))

            total_frames += 1

            fps = counter.update()
            if fps:
                print(f'Inference is running at {fps:.2f} FPS')

            frame_times.append(time.time() - frame_st)

        elapsed_time = time.time() - start_time

    # When everything done, release the capture
    cap.release()
    if writer:
        writer.release()

    print()
    print(f'Ran inference on {total_frames} frames in {TimeIt.format_elapsed(elapsed_time)}.')
    print(f'Average FPS: {total_frames / elapsed_time:.2f}')
    print()
    print_statistics(read_times, 'read')
    print_statistics(preprocess_times, 'pre-processing')
    print_statistics(detect_times, 'detection', total_frames=total_frames)
    print_statistics(preprocess_times, 'post-processing')
    print_statistics(draw_times, 'draw')
    print_statistics(write_times, 'write')
    print()
    print_statistics(image_times, 'image', total_frames=total_frames)
    print_statistics(frame_times, 'frame', total_frames=total_frames)
