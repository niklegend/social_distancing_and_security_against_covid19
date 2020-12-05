import cv2

from masterthesis.utils import FpsCounter


def run_on_video(video_path, run_on_image, output_path=None):
    cap = cv2.VideoCapture(video_path)

    writer = None
    if output_path:
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, int(fourcc), int(fps), (int(width), int(height)))

    total_fps = 0.0
    count = 0

    with FpsCounter() as counter:
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                break

            out_img = run_on_image(frame)

            if writer:
                writer.write(out_img)

            fps = counter.update()
            if fps:
                print(f'Inference is running at {fps:.2f} FPS')

                total_fps += fps
                count += 1

    print(f'Average FPS during inference: {total_fps / count:.2f}')

    # When everything done, release the capture
    cap.release()
    if writer:
        writer.release()
