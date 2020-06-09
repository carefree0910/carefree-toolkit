import math
import random

from abc import *
from typing import *

from .types import *
from .distributions import DistributionBase
from ...misc import prod, Grid


class DataType(metaclass=ABCMeta):
    def __init__(self,
                 distribution: DistributionBase = None,
                 **kwargs):
        self.dist, self.config = distribution, kwargs

    @property
    @abstractmethod
    def n_params(self) -> number_type:
        raise NotImplementedError

    @abstractmethod
    def transform(self, value) -> generic_number_type:
        raise NotImplementedError

    def __str__(self):
        return f"{type(self).__name__}({self.dist})"

    __repr__ = __str__

    @property
    def distribution_is_inf(self) -> bool:
        return math.isinf(self.dist.n_params)

    def _all(self) -> List[generic_number_type]:
        return list(map(self.transform, self.dist.values))

    def pop(self) -> generic_number_type:
        return self.transform(self.dist.pop())

    def all(self) -> List[generic_number_type]:
        if math.isinf(self.n_params):
            raise ValueError("'all' method could be called iff n_params is finite")
        return self._all()

    @property
    def lower(self) -> nullable_number_type:
        dist_lower = self.dist.lower
        if dist_lower is None:
            return
        return self.transform(dist_lower)

    @property
    def upper(self) -> nullable_number_type:
        dist_upper = self.dist.upper
        if dist_upper is None:
            return
        return self.transform(dist_upper)

    @property
    def values(self) -> Union[List[generic_number_type], None]:
        dist_values = self.dist.values
        if dist_values is None:
            return
        return list(map(self.transform, dist_values))


iterable_data_type = Union[List[DataType], Tuple[DataType, ...]]
iterable_generic_number_type = Union[List[generic_number_type], Tuple[generic_number_type, ...]]


class Iterable:
    def __init__(self, values: iterable_data_type):
        self._values = values
        self._constructor = list if isinstance(values, list) else tuple

    def __str__(self):
        braces = "[]" if self._constructor is list else "()"
        return f"{braces[0]}{', '.join(map(str, self._values))}{braces[1]}"

    __repr__ = __str__

    def pop(self) -> iterable_generic_number_type:
        return self._constructor(v.pop() for v in self._values)

    def all(self) -> Iterator[generic_number_type]:
        grid = Grid([v.all() for v in self._values])
        for v in grid:
            yield self._constructor(v)

    def transform(self, value) -> iterable_generic_number_type:
        return self._constructor(v.transform(vv) for v, vv in zip(self._values, value))

    @property
    def values(self) -> iterable_data_type:
        return self._values

    @property
    def n_params(self) -> number_type:
        n_params = prod(v.n_params for v in self._values)
        if math.isinf(n_params):
            return n_params
        return int(n_params)


class Any(DataType):
    @property
    def n_params(self) -> number_type:
        return self.dist.n_params

    def transform(self, value) -> generic_number_type:
        return value


class Int(DataType):
    @property
    def lower(self) -> int:
        return int(math.ceil(self.dist.lower))

    @property
    def upper(self) -> int:
        return int(math.floor(self.dist.upper))

    @property
    def values(self) -> List[int]:
        return list(range(self.lower, self.upper + 1))

    @property
    def n_params(self) -> int:
        if self.distribution_is_inf:
            return int(self.upper - self.lower) + 1
        return self.dist.n_params

    def _all(self) -> List[int]:
        if self.distribution_is_inf:
            return self.values
        return super()._all()

    def transform(self, value) -> int:
        return int(round(value + random.random() * 2e-4 - 1e-4))


class Float(DataType):
    @property
    def n_params(self) -> number_type:
        return self.dist.n_params

    def transform(self, value) -> float:
        return float(value)


class Bool(DataType):
    @property
    def n_params(self) -> int:
        if self.distribution_is_inf:
            return 2
        return len(self._all())

    def _all(self) -> List[bool]:
        return sorted(super()._all())

    def transform(self, value) -> bool:
        return bool(value)


class String(DataType):
    @property
    def n_params(self) -> number_type:
        return self.dist.n_params

    def transform(self, value) -> str:
        return str(value)


__all__ = ["DataType", "Iterable", "Any", "Int", "Float", "Bool", "String"]
