import random
from typing import *

import torch
import torchvision.transforms.functional as T
from PIL import Image


class AddGaussianNoise(object):

    def __init__(self, sigma: Optional[float] = 0.09, mean: Optional[float] = 0, std: Optional[float] = 1):
        self.sigma = sigma
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(x).normal_(self.mean, self.std)
        y = x + self.sigma * noise
        y.clamp_(min=0., max=1.)
        return y

    def __repr__(self):
        return '%s(sigma=%.4f, mean=%.4f, std=%.4f)' % (self.__class__.__name__, self.sigma, self.mean, self.std)


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    :param image: image, a PIL Image
    :return: distorted image
    """
    new_img = image

    distortions = [T.adjust_brightness,
                   T.adjust_contrast,
                   T.adjust_saturation,
                   T.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_img = d(new_img, adjust_factor)

    return new_img


def transform(image: Image, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None) -> torch.Tensor:
    new_img = image

    # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
    new_img = photometric_distort(new_img)

    # Convert PIL image to Torch tensor
    new_img = T.to_tensor(new_img)

    if mean is not None and std is not None:
        # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
        new_img = T.normalize(new_img, mean=mean, std=std)

    return new_img
