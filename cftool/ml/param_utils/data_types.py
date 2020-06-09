import math
import random

from abc import *
from typing import List, Tuple, Union

from ...misc import prod, Grid
from .distributions import DistributionBase


class DataType(metaclass=ABCMeta):
    def __init__(self, distribution: DistributionBase = None, optional=True, **kwargs):
        self._optional, self._distribution, self.config = optional, distribution, kwargs

    @property
    @abstractmethod
    def n_params(self):
        raise NotImplementedError

    @abstractmethod
    def transform(self, value):
        raise NotImplementedError

    def __str__(self):
        return f"{type(self).__name__}({self._distribution})"

    __repr__ = __str__

    @property
    def distribution_is_inf(self):
        return math.isinf(self._distribution.n_params)

    def _all(self):
        return list(map(self.transform, self._distribution.values))

    def pop(self):
        return self.transform(self._distribution.pop())

    def all(self):
        if math.isinf(self.n_params):
            raise ValueError("'all' method could be called iff n_params is finite")
        return self._all()

    @property
    def lower(self):
        dist_lower = self._distribution.lower
        if dist_lower is None:
            return
        return self.transform(dist_lower)

    @property
    def upper(self):
        dist_upper = self._distribution.upper
        if dist_upper is None:
            return
        return self.transform(dist_upper)

    @property
    def values(self):
        dist_values = self._distribution.values
        if dist_values is None:
            return
        return list(map(self.transform, dist_values))


class Iterable:
    def __init__(self, values: Union[List[DataType], Tuple[DataType]]):
        self._values = values
        self._constructor = list if isinstance(values, list) else tuple

    def __str__(self):
        braces = "[]" if self._constructor is list else "()"
        return f"{braces[0]}{', '.join(map(str, self._values))}{braces[1]}"

    __repr__ = __str__

    def pop(self):
        return self._constructor(v.pop() for v in self._values)

    def all(self, return_values=False):
        grid = Grid([v.all() for v in self._values])
        generator = (self._constructor(v) for v in grid)
        if return_values:
            return list(generator)
        yield from generator

    @property
    def values(self):
        return self._values

    @property
    def n_params(self):
        n_params = prod(v.n_params for v in self._values)
        if math.isinf(n_params):
            return n_params
        return int(n_params)


class Any(DataType):
    @property
    def n_params(self):
        return self._distribution.n_params

    def transform(self, value):
        return value


class Int(DataType):
    @property
    def lower(self):
        return int(math.ceil(self._distribution.lower))

    @property
    def upper(self):
        return int(math.floor(self._distribution.upper))

    @property
    def values(self):
        return list(range(self.lower, self.upper + 1))

    @property
    def n_params(self):
        if self.distribution_is_inf:
            return int(self.upper - self.lower) + 1
        return self._distribution.n_params

    def _all(self):
        if self.distribution_is_inf:
            return list(range(self.lower, self.upper + 1))
        return super()._all()

    def transform(self, value):
        return int(round(value + random.random() * 2e-4 - 1e-4))


class Float(DataType):
    @property
    def n_params(self):
        return self._distribution.n_params

    def transform(self, value):
        return float(value)


class Bool(DataType):
    @property
    def n_params(self):
        if self.distribution_is_inf:
            return 2
        return len(self._all())

    def _all(self):
        return sorted(super()._all())

    def transform(self, value):
        return bool(value)


class String(DataType):
    @property
    def n_params(self):
        return self._distribution.n_params

    def transform(self, value):
        return str(value)


__all__ = ["DataType", "Iterable", "Any", "Int", "Float", "Bool", "String"]
