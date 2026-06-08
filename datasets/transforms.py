import random
from collections.abc import Sequence
from io import BytesIO
from typing import Dict, Optional, Tuple

from torchvision import transforms


def _pair(value, default: Optional[Tuple[float, float]] = None):
    if value is None:
        return default
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != 2:
            raise ValueError("Augmentation limits must contain exactly two values")
        return value[0], value[1]
    return value, value


def _delta_limit_to_factor(value):
    low, high = _pair(value)
    low = max(0.0, 1.0 + float(low))
    high = max(0.0, 1.0 + float(high))
    if low > high:
        low, high = high, low
    return low, high


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


class RandomGaussianBlurRange:
    def __init__(self, kernel_limit, sigma=(0.1, 2.0)):
        self.kernel_low, self.kernel_high = _odd_kernel_range(kernel_limit)
        self.sigma = sigma

    def __call__(self, image):
        kernel_size = random.randrange(self.kernel_low, self.kernel_high + 1, 2)
        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=self.sigma)(image)


class RandomJPEGCompression:
    def __init__(self, quality_lower: int, quality_upper: int):
        quality_lower, quality_upper = int(quality_lower), int(quality_upper)
        if quality_lower > quality_upper:
            quality_lower, quality_upper = quality_upper, quality_lower
        self.quality_lower = max(1, min(100, quality_lower))
        self.quality_upper = max(1, min(100, quality_upper))

    def __call__(self, image):
        quality = random.randint(self.quality_lower, self.quality_upper)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        mode = "RGB" if image.mode == "P" else image.mode
        from PIL import Image

        with Image.open(buffer) as compressed:
            return compressed.convert(mode)


def build_transforms(config: Dict, train: bool):
    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {})
    image_size = int(data_cfg.get("image_size", 256))
    mean = data_cfg.get("mean", [0.5, 0.5, 0.5])
    std = data_cfg.get("std", [0.5, 0.5, 0.5])

    steps = [transforms.Resize((image_size, image_size))]
    if train and aug_cfg.get("enabled", True):
        if aug_cfg.get("horizontal_flip", 0.5) > 0:
            steps.append(transforms.RandomHorizontalFlip(p=float(aug_cfg.get("horizontal_flip", 0.5))))
        rotate = float(aug_cfg.get("rotate_limit", 0))
        if rotate > 0:
            steps.append(transforms.RandomRotation(degrees=(-rotate, rotate)))

        blur_prob = float(aug_cfg.get("blur_prob", 0.0))
        if blur_prob > 0:
            steps.append(
                transforms.RandomApply(
                    [RandomGaussianBlurRange(aug_cfg.get("blur_limit", [3, 3]))],
                    p=blur_prob,
                )
            )

        brightness_limit = aug_cfg.get("brightness_limit", None)
        contrast_limit = aug_cfg.get("contrast_limit", None)
        if brightness_limit is not None or contrast_limit is not None:
            color_jitter = {}
            if brightness_limit is not None:
                color_jitter["brightness"] = _delta_limit_to_factor(brightness_limit)
            if contrast_limit is not None:
                color_jitter["contrast"] = _delta_limit_to_factor(contrast_limit)
            steps.append(
                transforms.RandomApply(
                    [transforms.ColorJitter(**color_jitter)],
                    p=float(aug_cfg.get("brightness_prob", 0.0)),
                )
            )
        else:
            color_jitter = aug_cfg.get("color_jitter", None)
            if color_jitter:
                steps.append(transforms.ColorJitter(**color_jitter))

        quality_lower = aug_cfg.get("quality_lower", None)
        quality_upper = aug_cfg.get("quality_upper", None)
        if quality_lower is not None and quality_upper is not None:
            steps.append(
                transforms.RandomApply(
                    [RandomJPEGCompression(quality_lower, quality_upper)],
                    p=float(aug_cfg.get("quality_prob", 1.0)),
                )
            )

    steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(steps)
