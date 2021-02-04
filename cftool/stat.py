import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from typing import Any
from typing import Optional
from functools import partial

from .c import ema
from .c import rolling_min
from .c import rolling_max
from .c import rolling_sum
from .c import naive_ema
from .c import naive_rolling_min
from .c import naive_rolling_max
from .c import naive_rolling_sum
from .misc import show_or_save
from .ml.utils import generic_data_type


class RollingStat:
    @staticmethod
    def _sum(
        arr: np.ndarray,
        window: int,
        *,
        mean: bool,
        axis: int = -1,
    ) -> np.ndarray:
        if len(arr.shape) == 1 and axis in (0, -1):
            return rolling_sum(arr, window, mean)
        if rolling_sum is naive_rolling_sum:
            return naive_rolling_sum(arr, window, mean, axis)
        rolling_sum_ = partial(rolling_sum, window=window, mean=mean)
        return np.apply_along_axis(rolling_sum_, axis, arr)

    @staticmethod
    def sum(arr: np.ndarray, window: int, *, axis: int = -1) -> np.ndarray:
        return RollingStat._sum(arr, window, mean=False, axis=axis)

    @staticmethod
    def mean(arr: np.ndarray, window: int, *, axis: int = -1) -> np.ndarray:
        return RollingStat._sum(arr, window, mean=True, axis=axis)

    @staticmethod
    def std(arr: np.ndarray, window: int, *, axis: int = -1) -> np.ndarray:
        mean = RollingStat.mean(arr, window, axis=axis)
        second_order = RollingStat.sum(arr ** 2, window, axis=axis)
        return np.sqrt(second_order / float(window) - mean ** 2)

    @staticmethod
    def min(arr: np.ndarray, window: int, *, axis: int = -1) -> np.ndarray:
        if len(arr.shape) == 1 and axis in (0, -1):
            return rolling_min(arr, window)
        return naive_rolling_min(arr, window, axis)

    @staticmethod
    def max(arr: np.ndarray, window: int, *, axis: int = -1) -> np.ndarray:
        if len(arr.shape) == 1 and axis in (0, -1):
            return rolling_max(arr, window)
        return naive_rolling_max(arr, window, axis)

    @staticmethod
    def ema(arr: np.ndarray, window: int, *, axis: int = -1) -> np.ndarray:
        if len(arr.shape) == 1 and axis in (0, -1):
            return ema(arr, window)
        if ema is naive_ema:
            return naive_ema(arr, window, axis)
        return np.apply_along_axis(partial(ema, window=window), axis, arr)


class DataInspector:
    def __init__(self, data: generic_data_type):
        self._data = np.asarray(data, np.float32)
        self._sorted_data = np.sort(self._data)
        self._num_samples = len(self._data)
        self._mean = self._variance = self._std = None
        self._moments = []
        self._q1 = self._q3 = self._median = None

    def get_moment(self, k: int) -> float:
        if len(self._moments) < k:
            self._moments += [None] * (k - len(self._moments))
        if self._moments[k - 1] is None:
            self._moments[k - 1] = (
                np.sum((self._data - self.mean) ** k) / self._num_samples
            )
        return self._moments[k - 1]

    def get_quantile(self, q: float) -> float:
        if not 0.0 <= q <= 1.0:
            raise ValueError("`q` should be in [0, 1]")
        anchor = self._num_samples * q
        int_anchor = int(anchor)
        if not int(anchor % 1):
            return self._sorted_data[int_anchor]
        dq = self._sorted_data[int_anchor - 1] + self._sorted_data[int_anchor]
        return 0.5 * dq

    @property
    def min(self) -> float:
        return self._sorted_data[0]

    @property
    def max(self) -> float:
        return self._sorted_data[-1]

    @property
    def mean(self) -> float:
        if self._mean is None:
            self._mean = self._data.mean()
        return self._mean

    @property
    def variance(self) -> float:
        if self._variance is None:
            square_sum = np.sum((self._data - self.mean) ** 2)
            self._variance = square_sum / (self._num_samples - 1)
        return self._variance

    @property
    def std(self) -> float:
        if self._std is None:
            self._std = self.variance ** 0.5
        return self._std

    @property
    def skewness(self) -> float:
        n, moment3 = self._num_samples, self.get_moment(3)
        return n ** 2 * moment3 / ((n - 1) * (n - 2) * self.std ** 3)

    @property
    def kurtosis(self) -> float:
        n, moment4 = self._num_samples, self.get_moment(4)
        return n ** 2 * (n + 1) * moment4 / (
            (n - 1) * (n - 2) * (n - 3) * self.std ** 4
        ) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))

    @property
    def median(self) -> float:
        if self._median is None:
            self._median = self.get_quantile(0.5)
        return self._median

    @property
    def q1(self) -> float:
        if self._q1 is None:
            self._q1 = self.get_quantile(0.25)
        return self._q1

    @property
    def q3(self) -> float:
        if self._q3 is None:
            self._q3 = self.get_quantile(0.75)
        return self._q3

    @property
    def range(self) -> float:
        return self._sorted_data[-1] - self._sorted_data[0]

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1

    @property
    def trimean(self) -> float:
        return 0.25 * (self.q1 + self.q3) + 0.5 * self.median

    @property
    def lower_cutoff(self) -> float:
        return self.q1 - 1.5 * self.iqr

    @property
    def upper_cutoff(self) -> float:
        return self.q3 + 1.5 * self.iqr

    def draw_histogram(
        self,
        bin_size: int = 10,
        export_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        bins = np.arange(
            self._sorted_data[0] - self.iqr,
            self._sorted_data[-1] + self.iqr,
            bin_size,
        )
        plt.hist(self._data, bins=bins, alpha=0.5)
        plt.title(f"Histogram (bin_size: {bin_size})")
        show_or_save(export_path, **kwargs)

    def qq_plot(self, export_path: Optional[str] = None, **kwargs) -> None:
        stats.probplot(self._data, dist="norm", plot=plt)
        show_or_save(export_path, **kwargs)

    def box_plot(self, export_path: Optional[str] = None, **kwargs) -> None:
        plt.figure()
        plt.boxplot(self._data, vert=False, showmeans=True)
        show_or_save(export_path, **kwargs)


__all__ = [
    "RollingStat",
    "DataInspector",
]
