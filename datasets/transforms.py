from collections.abc import Sequence
from typing import Dict, Optional, Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


def _pair(value, default: Optional[Tuple[float, float]] = None):
    if value is None:
        return default
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != 2:
            raise ValueError("Augmentation limits must contain exactly two values")
        return value[0], value[1]
    return value, value


def _odd_kernel_range(value):
    low, high = _pair(value, (3, 3))
    low, high = int(low), int(high)
    if low > high:
        low, high = high, low
    low = max(3, low)
    high = max(low, high)
    if low % 2 == 0:
        low += 1
    if high % 2 == 0:
        high -= 1
    if high < low:
        high = low
    return low, high


def _rotate_limit(value):
    if isinstance(value, Sequence) and not isinstance(value, str):
        return _pair(value)
    value = float(value)
    return -value, value


def _image_compression(quality_lower, quality_upper, prob):
    quality = (int(quality_lower), int(quality_upper))
    quality = (max(1, min(100, quality[0])), max(1, min(100, quality[1])))
    if quality[0] > quality[1]:
        quality = (quality[1], quality[0])

    try:
        return A.ImageCompression(quality_lower=quality[0], quality_upper=quality[1], p=prob)
    except TypeError:
        return A.ImageCompression(quality_range=quality, p=prob)


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        image = np.array(image)
        return self.transform(image=image)["image"]


def build_transforms(config: Dict, train: bool):
    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {})
    image_size = int(data_cfg.get("image_size", 256))
    mean = data_cfg.get("mean", [0.5, 0.5, 0.5])
    std = data_cfg.get("std", [0.5, 0.5, 0.5])

    steps = []
    if train and aug_cfg.get("enabled", True):
        horizontal_flip = float(aug_cfg.get("horizontal_flip", 0.5))
        if horizontal_flip > 0:
            steps.append(A.HorizontalFlip(p=horizontal_flip))

        rotate_limit = aug_cfg.get("rotate_limit", 0)
        if rotate_limit:
            steps.append(
                A.Rotate(
                    limit=_rotate_limit(rotate_limit),
                    border_mode=0,
                    p=float(aug_cfg.get("rotate_prob", 1.0)),
                )
            )

        blur_prob = float(aug_cfg.get("blur_prob", 0.0))
        if blur_prob > 0:
            steps.append(
                A.GaussianBlur(
                    blur_limit=_odd_kernel_range(aug_cfg.get("blur_limit", [3, 3])),
                    p=blur_prob,
                )
            )

    steps.append(A.Resize(height=image_size, width=image_size))

    if train and aug_cfg.get("enabled", True):
        brightness_limit = aug_cfg.get("brightness_limit", None)
        contrast_limit = aug_cfg.get("contrast_limit", None)
        if brightness_limit is not None or contrast_limit is not None:
            steps.append(
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=_pair(brightness_limit, (0.0, 0.0)),
                            contrast_limit=_pair(contrast_limit, (0.0, 0.0)),
                        ),
                        A.FancyPCA(),
                        A.HueSaturationValue(),
                    ],
                    p=float(aug_cfg.get("brightness_prob", 0.0)),
                )
            )
        else:
            color_jitter = aug_cfg.get("color_jitter", None)
            if color_jitter:
                steps.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=float(color_jitter.get("brightness", 0.0)),
                        contrast_limit=float(color_jitter.get("contrast", 0.0)),
                        p=1.0,
                    )
                )

        quality_lower = aug_cfg.get("quality_lower", None)
        quality_upper = aug_cfg.get("quality_upper", None)
        if quality_lower is not None and quality_upper is not None:
            steps.append(
                _image_compression(
                    quality_lower,
                    quality_upper,
                    float(aug_cfg.get("quality_prob", 1.0)),
                )
            )

    steps.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return AlbumentationsTransform(A.Compose(steps))
