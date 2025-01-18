from typing import Optional, Tuple

import torch

from torchvision.transforms import (
    Normalize,
    Compose,
    InterpolationMode,
    Resize,
    CenterCrop,
)

from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


def image_transform(
    image_size: int,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that
        # Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)

    def convert_from_uint8_to_float(image: torch.Tensor) -> torch.Tensor:
        if image.dtype == torch.uint8:
            return image.to(torch.float32) / 255.0
        else:
            return image

    return Compose(
        [
            convert_from_uint8_to_float,
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            normalize,
        ]
    )
