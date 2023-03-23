import numpy as np

from io import BytesIO
from numpy import ndarray
from typing import Tuple

from .array import torch
from .array import arr_type

try:
    from PIL import Image

    TImage = Image.Image
except:
    Image = None
    TImage = None
try:
    import cv2
except:
    cv2 = None


def to_rgb(image: TImage, color: Tuple[int, int, int] = (255, 255, 255)) -> TImage:
    if image.mode == "CMYK":
        return image.convert("RGB")
    split = image.split()
    if len(split) < 4:
        return image.convert("RGB")
    if Image is None:
        raise ValueError("`pillow` is needed for `to_rgb`")
    background = Image.new("RGB", image.size, color)
    background.paste(image, mask=split[3])
    return background


def to_uint8(normalized_img: arr_type) -> arr_type:
    if isinstance(normalized_img, ndarray):
        return (np.clip(normalized_img * 255.0, 0.0, 255.0)).astype(np.uint8)
    return torch.clamp(normalized_img * 255.0, 0.0, 255.0).to(torch.uint8)


def np_to_bytes(img_arr: ndarray) -> bytes:
    if Image is None:
        raise ValueError("`pillow` is needed for `np_to_bytes`")
    if img_arr.dtype != np.uint8:
        img_arr = to_uint8(img_arr)
    bytes_io = BytesIO()
    Image.fromarray(img_arr).save(bytes_io, format="PNG")
    return bytes_io.getvalue()
