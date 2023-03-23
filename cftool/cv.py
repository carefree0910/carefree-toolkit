import numpy as np

from io import BytesIO
from numpy import ndarray
from typing import Tuple
from typing import Optional

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


class ImageProcessor:
    @staticmethod
    def _calculate_cdf(histogram: ndarray) -> ndarray:
        cdf = histogram.cumsum()
        normalized_cdf = cdf / float(cdf.max())
        return normalized_cdf

    @staticmethod
    def _calculate_lookup(source_cdf: ndarray, reference_cdf: ndarray) -> ndarray:
        lookup_table = np.zeros(256, np.uint8)
        lookup_val = 0
        for source_index, source_val in enumerate(source_cdf):
            for reference_index, reference_val in enumerate(reference_cdf):
                if reference_val >= source_val:
                    lookup_val = reference_index
                    break
            lookup_table[source_index] = lookup_val
        return lookup_table

    # accept uint8 inputs
    @classmethod
    def match_histograms(
        cls,
        source: ndarray,
        reference: ndarray,
        mask: Optional[ndarray] = None,
        *,
        strength: float = 1.0,
    ) -> ndarray:
        if cv2 is None:
            raise ValueError("`cv2` is needed for `match_histograms`")
        if strength <= 0.0:
            return source
        rev_mask = None if mask is None else ~mask
        transformed_channels = []
        for channel in range(source.shape[-1]):
            source_channel = source[..., channel]
            reference_channel = reference[..., channel]
            if mask is None:
                src_ch = source_channel
                ref_ch = reference_channel
            else:
                src_ch = source_channel[mask]
                ref_ch = reference_channel[mask]
            source_hist, _ = np.histogram(src_ch, 256, [0, 256])
            reference_hist, _ = np.histogram(ref_ch, 256, [0, 256])
            source_cdf = cls._calculate_cdf(source_hist)
            reference_cdf = cls._calculate_cdf(reference_hist)
            lookup = cls._calculate_lookup(source_cdf, reference_cdf)
            if 0.0 < strength < 1.0:
                for i, value in enumerate(lookup):
                    lookup[i] = round(value * strength + i * (1.0 - strength))
            transformed = cv2.LUT(source_channel, lookup)
            if rev_mask is not None:
                transformed[rev_mask] = reference_channel[rev_mask]
            transformed_channels.append(transformed)
        return np.stack(transformed_channels, axis=2)
