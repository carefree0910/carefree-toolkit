try:
    from .cython_utils import *
except ImportError:
    raise


import numpy as np


def c_rolling_min(array: np.ndarray, window: int) -> np.ndarray:
    dtype = array.dtype
    array = array.astype(np.float32)
    array[np.isnan(array)] = np.inf
    return rolling_min(array, window).astype(dtype)


def c_rolling_max(array: np.ndarray, window: int) -> np.ndarray:
    dtype = array.dtype
    array = array.astype(np.float32)
    array[np.isnan(array)] = -np.inf
    return rolling_max(array, window).astype(dtype)


__all__ = ["c_rolling_min", "c_rolling_max"]
