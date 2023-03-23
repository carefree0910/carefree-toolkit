import numpy as np

from io import BytesIO
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


def np_to_bytes(img_arr: ndarray) -> bytes:
    if img_arr.dtype != np.uint8:
        img_arr = to_uint8(img_arr)
    bytes_io = BytesIO()
    Image.fromarray(img_arr).save(bytes_io, format="PNG")
    return bytes_io.getvalue()
