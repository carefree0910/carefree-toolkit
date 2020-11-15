import numpy as np

from ..misc import StrideArray


def naive_rolling_min(array: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    while axis < 0:
        axis += len(array.shape)
    return np.nanmin(StrideArray(array).roll(window, axis=axis), axis + 1)


def naive_rolling_max(array: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    while axis < 0:
        axis += len(array.shape)
    return np.nanmax(StrideArray(array).roll(window, axis=axis), axis + 1)


def naive_ema(array: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    shape = list(array.shape)
    while axis < 0:
        axis += len(shape)
    rolled = StrideArray(array).roll(window, axis=axis)
    rolled_axis = axis + 1
    shape.insert(rolled_axis, window)
    new_shape = [1] * len(shape)
    new_shape[rolled_axis] = -1
    ratio = 2.0 / (window + 1.0)
    multipliers = ((1.0 - ratio) ** np.arange(window))[::-1].reshape(new_shape)
    return ratio * np.nansum(rolled * multipliers, rolled_axis)


__all__ = ["naive_rolling_min", "naive_rolling_max", "naive_ema"]
