import numpy as np

from PIL import Image
from numpy import ndarray
from typing import Tuple

from .array import torch
from .array import arr_type


def to_rgb(
    image: Image.Image,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    if image.mode == "CMYK":
        return image.convert("RGB")
    split = image.split()
    if len(split) < 4:
        return image.convert("RGB")
    background = Image.new("RGB", image.size, color)
    background.paste(image, mask=split[3])
    return background


def to_uint8(normalized_img: arr_type) -> arr_type:
    if isinstance(normalized_img, ndarray):
        return (np.clip(normalized_img * 255.0, 0.0, 255.0)).astype(np.uint8)
    return torch.clamp(normalized_img * 255.0, 0.0, 255.0).to(torch.uint8)
