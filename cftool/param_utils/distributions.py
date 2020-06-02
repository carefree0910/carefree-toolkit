import math
import random

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


class Uniform(DistributionBase):
    @property
    def n_params(self):
        return math.inf

    def pop(self):
        self._assert_lower_and_upper()
        return random.random() * (self._upper - self._lower) + self._lower


class Exponential(DistributionBase):
    @property
    def n_params(self):
        return math.inf

    def pop(self):
        self._assert_lower_and_upper()
        assert self._lower > 0, "lower should be greater than 0 in exponential distribution"
        ratio = self.config.setdefault("ratio", 2)
        assert ratio > 1, "ratio should be greater than 1"
        diff = math.log(self._upper / self._lower, ratio)
        return math.pow(ratio, random.random() * diff) * self._lower


class Choice(DistributionBase):
    @property
    def n_params(self):
        return len(self._values)

    def pop(self):
        self._assert_values()
        return random.choice(self._values)


__all__ = ["DistributionBase", "Uniform", "Exponential", "Choice"]
