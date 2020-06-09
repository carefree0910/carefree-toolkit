import math
import random

import matplotlib.pyplot as plt

from abc import *


class DistributionBase(metaclass=ABCMeta):
    def __init__(self, lower=None, upper=None, values=None, **kwargs):
        self.lower, self.upper, self.values, self.config = lower, upper, values, kwargs

    @property
    @abstractmethod
    def n_params(self):
        raise NotImplementedError

    @abstractmethod
    def pop(self):
        raise NotImplementedError

    def __str__(self):
        if self.values is not None:
            return f"{type(self).__name__}[{', '.join(map(str, self.values))}]"
        if not self.config:
            config_str = ""
        else:
            config_str = f", {', '.join([f'{k}={self.config[k]}' for k in sorted(self.config)])}"
        return f"{type(self).__name__}[{self.lower:.2f}, {self.upper:.2f}{config_str}]"

    __repr__ = __str__

    def _assert_lower_and_upper(self):
        assert self.lower is not None, "lower should be provided"
        assert self.upper is not None, "upper should be provided"

    def _assert_values(self):
        assert isinstance(self.values, list), "values should be a list"

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
        return random.random() * (self.upper - self.lower) + self.lower


class Exponential(Uniform):
    def __init__(self, lower=None, upper=None, values=None, **kwargs):
        super().__init__(lower, upper, values, **kwargs)
        self._assert_lower_and_upper()
        assert self.lower > 0, "lower should be greater than 0 in exponential distribution"
        self._base = self.config.setdefault("base", 2)
        assert self._base > 1, "base should be greater than 1"
        self.lower, self.upper = map(math.log, [self.lower, self.upper], 2 * [self._base])

    def pop(self):
        return math.pow(self._base, super().pop())


class Choice(DistributionBase):
    @property
    def n_params(self):
        return len(self.values)

    def pop(self):
        self._assert_values()
        return random.choice(self.values)


__all__ = ["DistributionBase", "Uniform", "Exponential", "Choice"]
