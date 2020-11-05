try:
    from .cython_utils import *
except ImportError:
    raise


import numpy as np


def c_rolling_min(array: np.ndarray, window: int) -> np.ndarray:
    dtype = array.dtype
    return rolling_min(array.astype(np.float32), window).astype(dtype)


def c_rolling_max(array: np.ndarray, window: int) -> np.ndarray:
    dtype = array.dtype
    return rolling_max(array.astype(np.float32), window).astype(dtype)


__all__ = ["c_rolling_min", "c_rolling_max"]
