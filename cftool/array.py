import math

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from collections import Counter
from multiprocessing.shared_memory import SharedMemory
from numpy.lib.stride_tricks import as_strided

from .misc import random_hash
from .types import torch
from .types import torchvision
from .types import F
from .types import arr_type
from .types import tensor_dict_type


TNormalizeResponse = Union[arr_type, Tuple[arr_type, Dict[str, Any]]]


def is_int(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.integer)


def is_float(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.floating)


def is_string(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, str)


def sigmoid(arr: arr_type) -> arr_type:
    if isinstance(arr, np.ndarray):
        return 1.0 / (1.0 + np.exp(-arr))
    return torch.sigmoid(arr)


def softmax(arr: arr_type) -> arr_type:
    if isinstance(arr, np.ndarray):
        logits = arr - np.max(arr, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(1, keepdims=True)
    return F.softmax(arr, dim=1)


def l2_normalize(arr: arr_type) -> arr_type:
    if isinstance(arr, np.ndarray):
        return arr / np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / arr.norm(dim=-1, keepdim=True)


def normalize(
    arr: arr_type,
    *,
    global_norm: bool = True,
    return_stats: False,
    eps: float = 1.0e-8,
) -> TNormalizeResponse:
    if global_norm:
        arr_mean, arr_std = arr.mean().item(), arr.std().item()
        arr_std = max(eps, arr_std)
        out = (arr - arr_mean) / arr_std
        if not return_stats:
            return out
        return out, dict(mean=arr_mean, std=arr_std)
    if isinstance(arr, np.ndarray):
        arr_mean, arr_std = arr.mean(axis=0), arr.std(axis=0)
        std = np.maximum(eps, arr_std)
    else:
        arr_mean, arr_std = arr.mean(dim=0), arr.std(dim=0)
        std = torch.clip(arr_std, min=eps)
    out = (arr - arr_mean) / std
    if not return_stats:
        return out
    return out, dict(mean=arr_mean.tolist(), std=std.tolist())


def normalize_from(arr: arr_type, stats: Dict[str, Any]) -> arr_type:
    mean, std = stats["mean"], stats["std"]
    return (arr - mean) / std


def recover_normalize_from(arr: arr_type, stats: Dict[str, Any]) -> arr_type:
    mean, std = stats["mean"], stats["std"]
    return arr * std + mean


def min_max_normalize(
    arr: arr_type,
    *,
    global_norm: bool = True,
    return_stats: False,
    eps: float = 1.0e-8,
) -> TNormalizeResponse:
    if global_norm:
        arr_min, arr_max = arr.min().item(), arr.max().item()
        diff = max(eps, arr_max - arr_min)
        out = (arr - arr_min) / diff
        if not return_stats:
            return out
        return out, dict(min=arr_min, diff=diff)
    if isinstance(arr, np.ndarray):
        arr_min, arr_max = arr.min(axis=0), arr.max(axis=0)
        diff = np.maximum(eps, arr_max - arr_min)
    else:
        arr_min, arr_max = arr.min(dim=0).values, arr.max(dim=0).values
        diff = torch.clip(arr_max - arr_min, min=eps)
    out = (arr - arr_min) / diff
    if not return_stats:
        return out
    return out, dict(min=arr_min.tolist(), diff=diff.tolist())


def min_max_normalize_from(arr: arr_type, stats: Dict[str, Any]) -> arr_type:
    arr_min, diff = stats["min"], stats["diff"]
    return (arr - arr_min) / diff


def recover_min_max_normalize_from(arr: arr_type, stats: Dict[str, Any]) -> arr_type:
    arr_min, diff = stats["min"], stats["diff"]
    return arr * diff + arr_min


def quantile_normalize(
    arr: arr_type,
    *,
    q: float = 0.01,
    global_norm: bool = True,
    return_stats: False,
    eps: float = 1.0e-8,
) -> TNormalizeResponse:
    # quantiles
    if isinstance(arr, np.ndarray):
        kw = {"axis": 0}
        quantile_fn = np.quantile
    else:
        kw = {"dim": 0}
        quantile_fn = torch.quantile
    if global_norm:
        arr_min = quantile_fn(arr, q)
        arr_max = quantile_fn(arr, 1.0 - q)
    else:
        arr_min = quantile_fn(arr, q, **kw)
        arr_max = quantile_fn(arr, 1.0 - q, **kw)
    # diff
    if global_norm:
        diff = max(eps, arr_max - arr_min)
    else:
        if isinstance(arr, np.ndarray):
            diff = np.maximum(eps, arr_max - arr_min)
        else:
            diff = torch.clip(arr_max - arr_min, min=eps)
    arr = arr.clip(arr_min, arr_max)
    out = (arr - arr_min) / diff
    if not return_stats:
        return out
    if not global_norm:
        arr_min = arr_min.item()
        diff = diff.item()
    else:
        arr_min = arr_min.tolist()
        diff = diff.tolist()
    return out, dict(min=arr_min, diff=diff)


def quantile_normalize_from(arr: arr_type, stats: Dict[str, Any]) -> arr_type:
    arr_min, diff = stats["min"], stats["diff"]
    return (arr - arr_min) / diff


def recover_quantile_normalize_from(arr: arr_type, stats: Dict[str, Any]) -> arr_type:
    arr_min, diff = stats["min"], stats["diff"]
    return arr * diff + arr_min


def clip_normalize(arr: arr_type) -> arr_type:
    fn = np if isinstance(arr, np.ndarray) else torch
    if arr.dtype == fn.uint8:
        return arr
    return fn.clip(arr, 0.0, 1.0)


# will return at least 2d
def squeeze(arr: arr_type) -> arr_type:
    n = arr.shape[0]
    arr = arr.squeeze()
    if n == 1:
        arr = arr[None, ...]
    return arr


def to_standard(arr: np.ndarray) -> np.ndarray:
    if is_int(arr):
        arr = arr.astype(np.int64)
    elif is_float(arr):
        arr = arr.astype(np.float32)
    return arr


def to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(to_standard(arr))


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def to_device(
    batch: tensor_dict_type,
    device: torch.device,
    **kwargs: Any,
) -> tensor_dict_type:
    return {
        k: v.to(device, **kwargs)
        if isinstance(v, torch.Tensor)
        else [
            vv.to(device, **kwargs) if isinstance(vv, torch.Tensor) else vv for vv in v
        ]
        if isinstance(v, list)
        else v
        for k, v in batch.items()
    }


def iou(logits: arr_type, labels: arr_type) -> arr_type:
    is_numpy = isinstance(logits, np.ndarray)
    num_classes = logits.shape[1]
    if num_classes == 1:
        heat_map = sigmoid(logits)
    elif num_classes == 2:
        heat_map = softmax(logits)[:, [1]]
    else:
        raise ValueError("`IOU` only supports binary situations")
    intersect = heat_map * labels
    union = heat_map + labels - intersect
    kwargs = {"axis" if is_numpy else "dim": tuple(range(1, len(intersect.shape)))}
    return intersect.sum(**kwargs) / union.sum(**kwargs)


def corr(
    predictions: arr_type,
    target: arr_type,
    weights: Optional[arr_type] = None,
    *,
    get_diagonal: bool = False,
) -> arr_type:
    is_numpy = isinstance(predictions, np.ndarray)
    keepdim_kw = {"keepdims" if is_numpy else "keepdim": True}
    norm_fn = np.linalg.norm if is_numpy else torch.norm
    matmul_fn = np.matmul if is_numpy else torch.matmul
    sqrt_fn = np.sqrt if is_numpy else torch.sqrt
    transpose_fn = np.transpose if is_numpy else torch.t

    w_sum = 0.0 if weights is None else weights.sum().item()
    if weights is None:
        mean = predictions.mean(0, **keepdim_kw)
    else:
        mean = (predictions * weights).sum(0, **keepdim_kw) / w_sum
    vp = predictions - mean
    if weights is None:
        kw = keepdim_kw.copy()
        kw["axis" if is_numpy else "dim"] = 0
        vp_norm = norm_fn(vp, 2, **kw)
    else:
        vp_norm = sqrt_fn((weights * (vp**2)).sum(0, **keepdim_kw))
    if predictions is target:
        vp_norm_t = transpose_fn(vp_norm)
        if weights is None:
            mat = matmul_fn(transpose_fn(vp), vp) / (vp_norm * vp_norm_t)
        else:
            mat = matmul_fn(transpose_fn(weights * vp), vp) / (vp_norm * vp_norm_t)
    else:
        if weights is None:
            target_mean = target.mean(0, **keepdim_kw)
        else:
            target_mean = (target * weights).sum(0, **keepdim_kw) / w_sum
        vt = transpose_fn(target - target_mean)
        if weights is None:
            kw = keepdim_kw.copy()
            kw["axis" if is_numpy else "dim"] = 1
            vt_norm = norm_fn(vt, 2, **kw)
        else:
            vt_norm = sqrt_fn((transpose_fn(weights) * (vt**2)).sum(1, **keepdim_kw))
        if weights is None:
            mat = matmul_fn(vt, vp) / (vp_norm * vt_norm)
        else:
            mat = matmul_fn(vt, weights * vp) / (vp_norm * vt_norm)
    if not get_diagonal:
        return mat
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(
            "`get_diagonal` is set to True but the correlation matrix "
            "is not a squared matrix, which is an invalid condition"
        )
    return np.diag(mat) if is_numpy else mat.diag()


def interpolant(arr: arr_type) -> arr_type:
    return arr * arr * arr * (arr * (arr * 6.0 - 15.0) + 10.0)


def perlin_noise_2d(
    shape: Tuple[int, int],
    periods: Tuple[int, int],
    should_tile: Tuple[bool, bool] = (False, False),
    interpolant_fn: Callable[[np.ndarray], np.ndarray] = interpolant,
) -> np.ndarray:
    delta = periods[0] / shape[0], periods[1] / shape[1]
    d = (shape[0] // periods[0], shape[1] // periods[1])
    grid = np.mgrid[: periods[0] : delta[0], : periods[1] : delta[1]] % 1  # type: ignore
    grid = grid.transpose(1, 2, 0)
    # gradients
    angles = 2 * np.pi * np.random.rand(periods[0] + 1, periods[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if should_tile[0]:
        gradients[-1, :] = gradients[0, :]
    if should_tile[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]
    # ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # interpolation
    arr = interpolant_fn(grid)
    n0 = n00 * (1 - arr[:, :, 0]) + arr[:, :, 0] * n10
    n1 = n01 * (1 - arr[:, :, 0]) + arr[:, :, 0] * n11
    return np.sqrt(2) * ((1 - arr[:, :, 1]) * n0 + arr[:, :, 1] * n1)


def fractal_noise_2d(
    shape: Tuple[int, int],
    periods: Tuple[int, int],
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: int = 2,
    should_tile: Tuple[bool, bool] = (False, False),
    interpolant_fn: Callable[[np.ndarray], np.ndarray] = interpolant,
) -> np.ndarray:
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1.0
    for _ in range(octaves):
        noise += amplitude * perlin_noise_2d(
            shape,
            (frequency * periods[0], frequency * periods[1]),
            should_tile,
            interpolant_fn,
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise


def contrast_noise(arr: arr_type) -> arr_type:
    arr = 0.9998 * arr + 0.0001
    arr = arr / (1.0 - arr)
    arr = arr**-2
    arr = 1.0 / (1.0 + arr)
    return arr


def get_one_hot(feature: Union[list, np.ndarray], dim: int) -> np.ndarray:
    """
    Get one-hot representation.

    Parameters
    ----------
    feature : array-like, source data of one-hot representation.
    dim : int, dimension of the one-hot representation.

    Returns
    -------
    one_hot : np.ndarray, one-hot representation of `feature`

    """

    one_hot = np.zeros([len(feature), dim], np.int64)
    one_hot[range(len(one_hot)), np.asarray(feature, np.int64).ravel()] = 1
    return one_hot


def get_indices_from_another(base: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """
    Get `segment` elements' indices in `base`.

    Warnings
    ----------
    All elements in segment should appear in base to ensure validity.

    Parameters
    ----------
    base : np.ndarray, base array.
    segment : np.ndarray, segment array.

    Returns
    -------
    indices : np.ndarray, positions where elements in `segment` appear in `base`

    Examples
    -------
    >>> import numpy as np
    >>> base, segment = np.arange(100), np.random.permutation(100)[:10]
    >>> assert np.allclose(get_indices_from_another(base, segment), segment)

    """
    base_sorted_args = np.argsort(base)
    positions = np.searchsorted(base[base_sorted_args], segment)
    return base_sorted_args[positions]


class UniqueIndices(NamedTuple):
    """
    unique           : np.ndarray, unique values of the given array (`arr`).
    unique_cnt       : np.ndarray, counts of each unique value.
    sorting_indices  : np.ndarray, indices which can (stably) sort the given
                                   array by its value.
    split_arr        : np.ndarray, array which can split the `sorting_indices`
                                   to make sure that. Each portion of the split
                                   indices belong & only belong to one of the
                                   unique values.
    """

    unique: np.ndarray
    unique_cnt: np.ndarray
    sorting_indices: np.ndarray
    split_arr: np.ndarray

    @property
    def split_indices(self):
        return np.split(self.sorting_indices, self.split_arr)


def get_unique_indices(arr: np.ndarray) -> UniqueIndices:
    """
    Get indices for unique values of an array.

    Parameters
    ----------
    arr : np.ndarray, target array which we wish to find indices of each unique value.

    Returns
    -------
    UniqueIndices

    Examples
    -------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3, 2, 4, 1, 0, 1], np.int64)
    >>> # UniqueIndices(
    >>> #   unique          = array([0, 1, 2, 3, 4], dtype=int64),
    >>> #   unique_cnt      = array([1, 3, 2, 1, 1], dtype=int64),
    >>> #   sorting_indices = array([6, 0, 5, 7, 1, 3, 2, 4], dtype=int64),
    >>> #   split_arr       = array([1, 4, 6, 7], dtype=int64))
    >>> #   split_indices   = [array([6], dtype=int64), array([0, 5, 7], dtype=int64),
    >>> #                      array([1, 3], dtype=int64), array([2], dtype=int64),
    >>> #                      array([4], dtype=int64)]
    >>> print(get_unique_indices(arr))

    """
    unique, unique_inv, unique_cnt = np.unique(
        arr,
        return_inverse=True,
        return_counts=True,
    )
    sorting_indices, split_arr = (
        np.argsort(unique_inv, kind="mergesort"),
        np.cumsum(unique_cnt)[:-1],
    )
    return UniqueIndices(unique, unique_cnt, sorting_indices, split_arr)


def get_counter_from_arr(arr: np.ndarray) -> Counter:
    """
    Get `Counter` of an array.

    Parameters
    ----------
    arr : np.ndarray, target array which we wish to get `Counter` from.

    Returns
    -------
    Counter

    Examples
    -------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3, 2, 4, 1, 0, 1], np.int64)
    >>> # Counter({1: 3, 2: 2, 0: 1, 3: 1, 4: 1})
    >>> print(get_counter_from_arr(arr))

    """
    if isinstance(arr, np.ndarray):
        arr = dict(zip(*np.unique(arr, return_counts=True)))
    return Counter(arr)


def allclose(*arrays: np.ndarray, **kwargs) -> bool:
    """
    Perform `np.allclose` to `arrays` one by one.

    Parameters
    ----------
    arrays : np.ndarray, target arrays.
    **kwargs : keyword arguments which will be passed into `np.allclose`.

    Returns
    -------
    allclose : bool

    """
    for i, arr in enumerate(arrays[:-1]):
        if not np.allclose(arr, arrays[i + 1], **kwargs):
            return False
    return True


class StrideArray:
    def __init__(
        self,
        arr: np.ndarray,
        *,
        copy: bool = False,
        writable: Optional[bool] = None,
    ):
        self.arr = arr
        self.shape = arr.shape
        self.num_dim = len(self.shape)
        self.strides = arr.strides
        self.copy = copy
        if writable is None:
            writable = copy
        self.writable = writable

    def __str__(self) -> str:
        return self.arr.__str__()

    def __repr__(self) -> str:
        return self.arr.__repr__()

    def _construct(
        self,
        shapes: Tuple[int, ...],
        strides: Tuple[int, ...],
    ) -> np.ndarray:
        arr = self.arr.copy() if self.copy else self.arr
        return as_strided(
            arr,
            shape=shapes,
            strides=strides,
            writeable=self.writable,
        )

    @staticmethod
    def _get_output_dim(in_dim: int, window: int, stride: int) -> int:
        return (in_dim - window) // stride + 1

    def roll(self, window: int, *, stride: int = 1, axis: int = -1) -> np.ndarray:
        while axis < 0:
            axis += self.num_dim
        target_dim = self.shape[axis]
        rolled_dim = self._get_output_dim(target_dim, window, stride)
        if rolled_dim <= 0:
            msg = f"window ({window}) is too large for target dimension ({target_dim})"
            raise ValueError(msg)
        # shapes
        rolled_shapes = tuple(self.shape[:axis]) + (rolled_dim, window)
        if axis < self.num_dim - 1:
            rolled_shapes = rolled_shapes + self.shape[axis + 1 :]
        # strides
        previous_strides = tuple(self.strides[:axis])
        target_stride = (self.strides[axis] * stride,)
        latter_strides = tuple(self.strides[axis:])
        rolled_strides = previous_strides + target_stride + latter_strides
        # construct
        return self._construct(rolled_shapes, rolled_strides)

    def patch(
        self,
        patch_w: int,
        patch_h: Optional[int] = None,
        *,
        h_stride: int = 1,
        w_stride: int = 1,
        h_axis: int = -2,
    ) -> np.ndarray:
        if self.num_dim < 2:
            raise ValueError("`patch` requires input with at least 2d")
        while h_axis < 0:
            h_axis += self.num_dim
        w_axis = h_axis + 1
        if patch_h is None:
            patch_h = patch_w
        h_shape, w_shape = self.shape[h_axis], self.shape[w_axis]
        if h_shape < patch_h:
            msg = f"patch_h ({patch_h}) is too large for target dimension ({h_shape})"
            raise ValueError(msg)
        if w_shape < patch_w:
            msg = f"patch_w ({patch_w}) is too large for target dimension ({w_shape})"
            raise ValueError(msg)
        # shapes
        patched_h_dim = self._get_output_dim(h_shape, patch_h, h_stride)
        patched_w_dim = self._get_output_dim(w_shape, patch_w, w_stride)
        patched_dim = (patched_h_dim, patched_w_dim)
        patched_dim = patched_dim + (patch_h, patch_w)
        patched_shapes = tuple(self.shape[:h_axis]) + patched_dim
        if w_axis < self.num_dim - 1:
            patched_shapes = patched_shapes + self.shape[w_axis + 1 :]
        # strides
        arr_h_stride, arr_w_stride = self.strides[h_axis], self.strides[w_axis]
        previous_strides = tuple(self.strides[:h_axis])
        target_stride = (arr_h_stride * h_stride, arr_w_stride * w_stride)
        target_stride = target_stride + (arr_h_stride, arr_w_stride)
        latter_strides = tuple(self.strides[w_axis + 1 :])
        patched_strides = previous_strides + target_stride + latter_strides
        # construct
        return self._construct(patched_shapes, patched_strides)

    def repeat(self, k: int, axis: int = -1) -> np.ndarray:
        while axis < 0:
            axis += self.num_dim
        target_dim = self.shape[axis]
        if target_dim != 1:
            raise ValueError("`repeat` can only be applied on axis with dim == 1")
        # shapes
        repeated_shapes = tuple(self.shape[:axis]) + (k,)
        if axis < self.num_dim - 1:
            repeated_shapes = repeated_shapes + self.shape[axis + 1 :]
        # strides
        previous_strides = tuple(self.strides[:axis])
        target_stride = (0,)
        latter_strides = tuple(self.strides[axis + 1 :])
        repeated_strides = previous_strides + target_stride + latter_strides
        # construct
        return self._construct(repeated_shapes, repeated_strides)


class SharedArray:
    def __init__(
        self,
        dtype: Union[type, np.dtype],
        shape: Union[List[int], Tuple[int, ...]],
        data: Optional[np.ndarray] = None,
    ):
        name = random_hash()
        d_size = np.dtype(dtype).itemsize * np.prod(shape)
        self._shm = SharedMemory(create=True, size=d_size, name=name)
        self.value = np.ndarray(shape=shape, dtype=dtype, buffer=self._shm.buf)
        if data is not None:
            self.value[:] = data[:]

    def destroy(self) -> None:
        self._shm.close()
        self._shm.unlink()

    @classmethod
    def from_data(cls, data: np.ndarray) -> "SharedArray":
        return cls(data.dtype, data.shape, data)


def get_label_predictions(logits: np.ndarray, threshold: float) -> np.ndarray:
    # binary classification
    if logits.shape[-1] == 2:
        logits = logits[..., [1]] - logits[..., [0]]
    if logits.shape[-1] == 1:
        logit_threshold = math.log(threshold / (1.0 - threshold))
        return (logits > logit_threshold).astype(int)
    return logits.argmax(1)[..., None]


def get_full_logits(logits: np.ndarray) -> np.ndarray:
    # binary classification
    if logits.shape[-1] == 1:
        logits = np.concatenate([-logits, logits], axis=-1)
    return logits


def make_grid(arr: arr_type, n_row: Optional[int] = None) -> torch.Tensor:
    if isinstance(arr, np.ndarray):
        arr = to_torch(arr)
    if n_row is None:
        n_row = math.ceil(math.sqrt(len(arr)))
    return torchvision.utils.make_grid(arr, n_row)
