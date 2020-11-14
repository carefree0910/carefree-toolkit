import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from .c import rolling_min
from .c import rolling_max
from .c import naive_rolling_min
from .c import naive_rolling_max
from .ml import generic_data_type
from .misc import show_or_save


class RollingStat:
    @staticmethod
    def sum(arr: np.ndarray, window: int, *, axis: int = -1) -> np.ndarray:
        if window > arr.shape[axis]:
            raise ValueError("`window` is too large for current array")
        pad_width = [[0, 0] for _ in range(len(arr.shape))]
        pad_width[axis][0] = 1
        arr = np.pad(arr, pad_width=pad_width, mode="constant", constant_values=0)
        cumsum = np.cumsum(arr, axis=axis)
        cumsum = np.swapaxes(cumsum, axis, -1)
        rolling_sum = cumsum[..., window:] - cumsum[..., :-window]
        return np.swapaxes(rolling_sum, axis, -1)

    @staticmethod
    def mean(arr: np.ndarray, window: int, *, axis: int = -1) -> np.ndarray:
        return RollingStat.sum(arr, window, axis=axis) / float(window)

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
    def ema(arr: np.ndarray, ratio: float, axis: int = -1) -> np.ndarray:
        dim = arr.shape[axis]
        shapes = [1] * len(arr.shape)
        shapes[axis] = -1
        multipliers = (ratio ** np.arange(dim))[::-1].reshape(shapes)
        return (1.0 - ratio) * np.cumsum(arr * multipliers, axis=axis) / multipliers


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
        self, bin_size: int = 10, export_path: str = None, **kwargs
    ) -> None:
        bins = np.arange(
            self._sorted_data[0] - self.iqr,
            self._sorted_data[-1] + self.iqr,
            bin_size,
        )
        plt.hist(self._data, bins=bins, alpha=0.5)
        plt.title(f"Histogram (bin_size: {bin_size})")
        show_or_save(export_path, **kwargs)

    def qq_plot(self, export_path: str = None, **kwargs) -> None:
        stats.probplot(self._data, dist="norm", plot=plt)
        show_or_save(export_path, **kwargs)

    def box_plot(self, export_path: str = None, **kwargs) -> None:
        plt.figure()
        plt.boxplot(self._data, vert=False, showmeans=True)
        show_or_save(export_path, **kwargs)


__all__ = [
    "RollingStat",
    "DataInspector",
]
