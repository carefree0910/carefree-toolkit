import math
import base64

import numpy as np

from io import BytesIO
from abc import abstractmethod
from numpy import ndarray
from typing import Any
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional
from typing import NamedTuple
from dataclasses import dataclass

from .misc import safe_execute
from .misc import shallow_copy_dict
from .misc import WithRegister
from .array import to_torch
from .types import torch
from .types import torchvision
from .types import arr_type
from .geometry import Matrix2D
from .geometry import Matrix2DProperties

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


padding_modes: Dict[str, Type["Padding"]] = {}


class ReadImageResponse(NamedTuple):
    image: np.ndarray
    alpha: Optional[np.ndarray]
    original: TImage
    original_size: Tuple[int, int]


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


def to_alpha_channel(image: TImage) -> TImage:
    if image.mode == "RGBA":
        return image.split()[3]
    return image.convert("L")


def np_to_bytes(img_arr: ndarray) -> bytes:
    if Image is None:
        raise ValueError("`pillow` is needed for `np_to_bytes`")
    if img_arr.dtype != np.uint8:
        img_arr = to_uint8(img_arr)
    bytes_io = BytesIO()
    Image.fromarray(img_arr).save(bytes_io, format="PNG")
    return bytes_io.getvalue()


def restrict_wh(w: int, h: int, max_wh: int) -> Tuple[int, int]:
    max_original_wh = max(w, h)
    if max_original_wh <= max_wh:
        return w, h
    wh_ratio = w / h
    if wh_ratio >= 1:
        return max_wh, round(max_wh / wh_ratio)
    return round(max_wh * wh_ratio), max_wh


def get_suitable_size(n: int, anchor: int) -> int:
    if n <= anchor:
        return anchor
    mod = n % anchor
    return n - mod + int(mod > 0.5 * anchor) * anchor


def read_image(
    image: Union[str, TImage],
    max_wh: Optional[int],
    *,
    anchor: Optional[int],
    to_gray: bool = False,
    to_mask: bool = False,
    resample: Any = Image.LANCZOS,
    normalize: bool = True,
    padding_mode: Optional[str] = None,
    padding_mask: Optional[TImage] = None,
    padding_kwargs: Optional[Dict[str, Any]] = None,
    to_torch_fmt: bool = True,
) -> ReadImageResponse:
    if Image is None:
        raise ValueError("`pillow` is needed for `read_image`")
    if isinstance(image, str):
        image = Image.open(image)
    alpha = None
    original = image
    if image.mode == "RGBA":
        alpha = padding_mask = image.split()[3]
    if not to_mask and not to_gray:
        if padding_mask is None or padding_mode is None:
            image = to_rgb(image)
        else:
            padding = Padding.make(padding_mode, {})
            padding_kw = shallow_copy_dict(padding_kwargs or {})
            padding_kw.update(dict(image=image, alpha=padding_mask))
            image = safe_execute(padding.pad, padding_kw)
    else:
        if to_mask and to_gray:
            raise ValueError("`to_mask` & `to_gray` should not be True simultaneously")
        if to_mask and image.mode == "RGBA":
            image = alpha
        else:
            image = image.convert("L")
    original_w, original_h = image.size
    if max_wh is None:
        w, h = original_w, original_h
    else:
        w, h = restrict_wh(original_w, original_h, max_wh)
    if anchor is not None:
        w, h = map(get_suitable_size, (w, h), (anchor, anchor))
    image = image.resize((w, h), resample=resample)
    image = np.array(image)
    if normalize:
        image = image.astype(np.float32) / 255.0
    if alpha is not None:
        alpha = np.array(alpha)[None, None]
        if normalize:
            alpha = alpha.astype(np.float32) / 255.0
    if to_torch_fmt:
        if to_mask or to_gray:
            image = image[None, None]
        else:
            image = image[None].transpose(0, 3, 1, 2)
    return ReadImageResponse(image, alpha, original, (original_w, original_h))


def save_images(arr: arr_type, path: str, n_row: Optional[int] = None) -> None:
    if isinstance(arr, np.ndarray):
        arr = to_torch(arr)
    if n_row is None:
        n_row = math.ceil(math.sqrt(len(arr)))
    torchvision.utils.save_image(arr, path, normalize=True, nrow=n_row)


def to_base64(image: TImage) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def from_base64(base64_string: str) -> TImage:
    if Image is None:
        raise ValueError("`pillow` is needed for `from_base64`")
    base64_string = base64_string.split("base64,")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))


@dataclass
class ImageBox:
    l: int
    t: int
    r: int
    b: int

    @property
    def w(self) -> int:
        return self.r - self.l

    @property
    def h(self) -> int:
        return self.b - self.t

    @property
    def wh_ratio(self) -> float:
        return self.w / self.h

    @property
    def tuple(self) -> Tuple[int, int, int, int]:
        return self.l, self.t, self.r, self.b

    @property
    def matrix(self) -> Matrix2D:
        return Matrix2D.from_properties(
            Matrix2DProperties(x=self.l, y=self.t, w=self.w, h=self.h)
        )

    def copy(self) -> "ImageBox":
        return ImageBox(*self.tuple)

    def crop(self, image: ndarray) -> ndarray:
        return image[self.t : self.b, self.l : self.r]

    def crop_tensor(self, image: torch.Tensor) -> torch.Tensor:
        return image[..., self.t : self.b, self.l : self.r]

    def pad(
        self,
        padding: int,
        *,
        w: Optional[int] = None,
        h: Optional[int] = None,
    ) -> "ImageBox":
        l, t, r, b = self.tuple
        l = max(0, l - padding)
        r += padding
        if w is not None:
            r = min(r, w)
        t = max(0, t - padding)
        b += padding
        if h is not None:
            b = min(b, h)
        return ImageBox(l, t, r, b)

    def to_square(
        self,
        *,
        w: Optional[int] = None,
        h: Optional[int] = None,
        expand: bool = True,
    ) -> "ImageBox":
        l, t, r, b = self.tuple
        bw, bh = r - l, b - t
        diff = abs(bw - bh)
        if diff == 0:
            return self.copy()
        if expand:
            if bw > bh:
                t = max(0, t - diff // 2)
                b = t + bw
                if h is not None:
                    b = min(b, h)
            else:
                l = max(0, l - diff // 2)
                r = l + bh
                if w is not None:
                    r = min(r, w)
        else:
            if bw > bh:
                l += diff // 2
                r = l + bh
                if w is not None:
                    r = min(r, w)
            else:
                t += diff // 2
                b = t + bw
                if h is not None:
                    b = min(b, h)
        return ImageBox(l, t, r, b)

    @classmethod
    def from_mask(cls, uint8_mask: ndarray, threshold: Optional[int]) -> "ImageBox":
        ys, xs = np.where(uint8_mask > (threshold or 0))
        ys, xs = np.where(uint8_mask)
        if len(ys) == 0:
            return cls(0, 0, 0, 0)
        return cls(xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item())


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

    @classmethod
    def get_distances_mat(cls, alpha_channel: ndarray) -> ndarray:
        return cv2.distanceTransform(alpha_channel, cv2.DIST_L2, 3)

    @classmethod
    def paste(cls, fg: TImage, bg: TImage, *, num_fade_pixels: int = 50) -> TImage:
        if Image is None:
            raise ValueError("`pillow` is needed for `harmonization`")
        if fg.mode != "RGBA":
            fg = fg.convert("RGBA")
        if bg.mode != "RGBA":
            bg = bg.convert("RGBA")
        fg_array = np.array(fg)
        alpha_channel = fg_array[:, :, 3]
        mat = cls.get_distances_mat(alpha_channel)
        ring_mask = mat <= num_fade_pixels
        inv_ring_mask = ~ring_mask
        mat[inv_ring_mask] = alpha_channel[inv_ring_mask]
        ring_max = mat[ring_mask].max()
        mat[ring_mask] = 255 * mat[ring_mask] / ring_max
        fg_array[:, :, 3] = mat
        return Image.alpha_composite(bg, Image.fromarray(fg_array))


class Padding(WithRegister):
    d = padding_modes

    @abstractmethod
    def pad(self, image: Image.Image, alpha: Image.Image, **kwargs: Any) -> Image.Image:
        pass


@Padding.register("cv2_ns")
class CV2NS(Padding):
    def pad(
        self,
        image: Image.Image,
        alpha: Image.Image,
        *,
        radius: int = 5,
        **kwargs: Any,
    ) -> Image.Image:
        if cv2 is None:
            raise ValueError("`cv2` is needed for `CV2NS`")
        img_arr = np.array(image.convert("RGB"))[..., ::-1]
        rs = cv2.inpaint(img_arr, np.array(alpha), radius, cv2.INPAINT_NS)
        return Image.fromarray(rs[..., ::-1])


@Padding.register("cv2_telea")
class CV2Telea(Padding):
    def pad(
        self,
        image: Image.Image,
        alpha: Image.Image,
        *,
        radius: int = 5,
        **kwargs: Any,
    ) -> Image.Image:
        if cv2 is None:
            raise ValueError("`cv2` is needed for `CV2Telea`")
        img_arr = np.array(image.convert("RGB"))[..., ::-1]
        rs = cv2.inpaint(img_arr, np.array(alpha), radius, cv2.INPAINT_TELEA)
        return Image.fromarray(rs[..., ::-1])
