import cv2

import numpy as np

from typing import *
from PIL import Image


def draw_contours(
    w: int,
    h: int,
    contours: List[np.ndarray],
    *,
    saving_path: Optional[str] = None,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> Image.Image:
    canvas = np.zeros([h, w, 3]).astype(np.uint8)
    cv2.drawContours(canvas, contours, -1, color, 3)
    im = Image.fromarray(canvas)
    if saving_path is not None:
        im.save(saving_path)
    return im


class Reader:
    def __init__(self, img_path: str, flag: int = cv2.IMREAD_UNCHANGED):
        try:
            self.img_array = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), flag)
        except:
            self.img_array = None

    @property
    def is_valid(self):
        return self.img_array is not None

    def to_gray(self, *, reverse: bool = False) -> Optional[np.ndarray]:
        if not self.is_valid:
            return None
        if len(self.img_array.shape) == 2:
            gray = self.img_array
        elif self.img_array.shape[-1] == 4:
            gray = self.img_array[..., -1]
        else:
            gray = cv2.cvtColor(self.img_array, cv2.COLOR_RGB2GRAY)
            if reverse:
                gray = 255 - gray
        return gray


__all__ = ["draw_contours", "Reader"]
