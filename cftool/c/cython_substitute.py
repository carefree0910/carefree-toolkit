import numpy as np

from ..misc import StrideArray


def naive_rolling_min(array: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    return StrideArray(array).roll(window, axis=axis).min(axis)


def naive_rolling_max(array: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    return StrideArray(array).roll(window, axis=axis).max(axis)


__all__ = ["naive_rolling_min", "naive_rolling_max"]
