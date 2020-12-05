import argparse

import cv2
import torch
from detectron2.config import get_cfg
from detectron2.data import Metadata
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from masterthesis.utils import FpsCounter, TimeIt


def run_on_image(img, predictor, metadata=None, show=False):
    model_outputs = predictor(img)

    visualizer = Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        scale=0.5,
        instance_mode=ColorMode.SEGMENTATION
    )

    out = visualizer.draw_instance_predictions(model_outputs['instances'].to('cpu'))

    out_img = out.get_image()[:, :, ::-1]

    return out_img


def run_on_video(video_path, predictor, metadata):
    cap = cv2.VideoCapture(video_path)

    counter = FpsCounter()
    counter.reset()

    status = True

    while status:
        # Capture frame-by-frame
        status, frame = cap.read()

        if status:
            out_img = run_on_image(frame, predictor, metadata=metadata)

            cv2.imshow('image', out_img)

            count = counter.update()
            if count:
                print(f'Running at {count:.2f} FPS')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                status = False

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main(args):
    config_path = args.config_path
    weights_path = args.weights_path

    path = args.image_path

    metadata = Metadata(
        thing_classes=['No mask', 'Mask'],
        # Red color for 'No mask' class and green for 'Mask' class
        thing_colors=[(255, 0, 0), (0, 255, 0)]
    )

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    # Now we change it a little bit for inference:
    cfg.MODEL.WEIGHTS = weights_path  # path to the pre-trained-model archive
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.6  # set a custom confidence testing threshold

    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    with TimeIt('Predictor successfully instantiated'):
        predictor = DefaultPredictor(cfg)

    if path:
        img = cv2.imread(path)
        out_img = run_on_image(img, predictor, metadata, show=True)
        cv2.imshow('image', out_img)
        cv2.waitKey(0)
    else:
        run_on_video(0, predictor, metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-path', help='Path to an image')

    parser.add_argument('--config-path', help='Path to the model config', required=True)
    parser.add_argument('--weights-path', help='Path to the model weights', required=True)

    args = parser.parse_args()

    main(args)
