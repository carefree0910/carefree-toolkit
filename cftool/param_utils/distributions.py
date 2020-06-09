import math
import random

import matplotlib.pyplot as plt

from abc import *


class DistributionBase(metaclass=ABCMeta):
    def __init__(self, lower=None, upper=None, values=None, **kwargs):
        self._lower, self._upper, self._values, self.config = lower, upper, values, kwargs

    @property
    @abstractmethod
    def n_params(self):
        raise NotImplementedError

    @abstractmethod
    def pop(self):
        raise NotImplementedError

    def __str__(self):
        if self._values is not None:
            return f"{type(self).__name__}[{', '.join(map(str, self._values))}]"
        return f"{type(self).__name__}[{self._lower}, {self._upper}]"

    __repr__ = __str__

    def _assert_lower_and_upper(self):
        assert self._lower is not None, "lower should be provided"
        assert self._upper is not None, "upper should be provided"

    def _assert_values(self):
        assert isinstance(self._values, list), "values should be a list"

    def visualize(self, n: int = 100) -> "DistributionBase":
        plt.figure()
        plt.scatter(list(range(n)), sorted(self.pop() for _ in range(n)))
        plt.show()
        return self


class Uniform(DistributionBase):
    @property
    def n_params(self):
        return math.inf

    def pop(self):
        self._assert_lower_and_upper()
        return random.random() * (self._upper - self._lower) + self._lower


class Exponential(Uniform):
    def __init__(self, lower=None, upper=None, values=None, **kwargs):
        super().__init__(lower, upper, values, **kwargs)
        self._assert_lower_and_upper()
        assert self._lower > 0, "lower should be greater than 0 in exponential distribution"
        self._base = self.config.setdefault("base", 2)
        assert self._base > 1, "base should be greater than 1"
        self._log_lower, self._log_upper = map(math.log, [self._lower, self._upper], 2 * [self._base])

    def pop(self):
        lower, upper = self._lower, self._upper
        self._lower, self._upper = self._log_lower, self._log_upper
        result = math.pow(self._base, super().pop())
        self._lower, self._upper = lower, upper
        return result


class Choice(DistributionBase):
    @property
    def n_params(self):
        return len(self._values)

    def pop(self):
        self._assert_values()
        return random.choice(self._values)


__all__ = ["DistributionBase", "Uniform", "Exponential", "Choice"]
