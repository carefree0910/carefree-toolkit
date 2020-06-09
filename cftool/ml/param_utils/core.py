import math
import numpy as np

from typing import Dict, List, Union, Iterator

from .types import *
from .data_types import *
from .distributions import *
from ...misc import prod, Grid

params_type = Union[DataType, Iterable, Dict[str, "params_type"]]
union_nested_type = Union[nested_type, all_nested_type]
union_flattened_type = Union[flattened_type, all_flattened_type]


class ParamsGenerator:
    """
    Parameter generator for param searching, see cftool.ml.hpo.base.HPOBase for usage.

    Parameters
    ----------
    params : params_type, parameter settings.
    * If DataType, then distribution of this DataType must be `Choice`. In this case, we'll simply 'enumerate'
    through the config choices.
    * If Iterable, then each element should be a DataType. It could be nested.
    * If Dict[str, "params_type"], then each key should correspond to a config key, and its value should
    correspond to the desired value distribution. It could be nested.

    Examples
    ----------
    >>> grid = ParamsGenerator({
    >>>     "a": Any(Choice(values=[1, 2, 3])),
    >>>     "c": {"d": Int(Choice(values=[1, 2, 3])), "e": Float(Choice(values=[1, 2]))}
    >>> })
    >>> for param in grid.all():
    >>>     print(param)
    >>> # output : {'a': 1, 'c': {'d': 1, 'e': 1, 'f': 3}}, {'a': 1, 'c': {'d': 1, 'e': 1, 'f': 4}}
    >>> #          {'a': 1, 'c': {'d': 1, 'e': 2, 'f': 3}}, {'a': 1, 'c': {'d': 1, 'e': 2, 'f': 4}}
    >>> #          {'a': 1, 'c': {'d': 2, 'e': 1, 'f': 3}}, {'a': 1, 'c': {'d': 2, 'e': 1, 'f': 4}}
    >>> #          {'a': 1, 'c': {'d': 2, 'e': 2, 'f': 3}}, {'a': 1, 'c': {'d': 2, 'e': 2, 'f': 4}}
    >>> #          ......
    >>> #          {'a': 3, 'c': {'d': 3, 'e': 2, 'f': 3}}, {'a': 3, 'c': {'d': 3, 'e': 2, 'f': 4}}

    """

    def __init__(self, params: params_type):
        self._params = params
        self._delim, self._idx_start = "^_^", "$$$"
        if isinstance(self._params, DataType):
            assert_msg = "distribution must be `Choice` when DataType is used as `params`"
            assert isinstance(self._params.dist, Choice), assert_msg
        self._all_pure_params = self._all_flattened_params = None
        self._sorted_flattened_keys = self._sorted_flattened_offsets = None

    @property
    def n_params(self) -> number_type:
        if self.is_enumerate:
            return self._params.n_params
        def _n_params(params):
            if isinstance(params, (DataType, Iterable)):
                return params.n_params
            assert isinstance(params, dict)
            n_params = prod(_n_params(v) for v in params.values())
            if math.isinf(n_params):
                return n_params
            return int(n_params)
        return _n_params(self._params)

    @property
    def is_enumerate(self) -> bool:
        return isinstance(self._params, (DataType, Iterable))

    @property
    def all_nested_params(self) -> all_nested_type:
        if self._all_pure_params is None:
            def _all(src, tgt):
                for k, v in src.items():
                    if isinstance(v, dict):
                        next_tgt = tgt.setdefault(k, {})
                        _all(v, next_tgt)
                    else:
                        tgt[k] = v.all()
                return tgt
            self._all_pure_params = _all(self._params, {})
        return self._all_pure_params

    @property
    def all_flattened_params(self) -> all_flattened_type:
        if self._all_flattened_params is None:
            self._all_flattened_params = self.flatten_nested_params(self.all_nested_params)
        return self._all_flattened_params

    @property
    def sorted_flattened_key(self) -> List[str]:
        if self._sorted_flattened_keys is None:
            self._sorted_flattened_keys = sorted(self.all_flattened_params)
        return self._sorted_flattened_keys

    @property
    def sorted_flattened_offsets(self) -> List[int]:
        if self._sorted_flattened_offsets is None:
            offsets = []
            for key in self.sorted_flattened_key:
                data_type = self._get_data_type_from(key)
                if not isinstance(data_type, Iterable):
                    offsets.append(1)
                else:
                    offsets.append(len(data_type.values))
            self._sorted_flattened_offsets = offsets
        return self._sorted_flattened_offsets

    def _get_data_type_from(self, flattened_key: str) -> DataType:
        data_type = self._params
        for sub_key in flattened_key.split(self._delim):
            data_type = data_type[sub_key]
        return data_type

    def pop(self) -> nested_type:
        if self.is_enumerate:
            return self._params.pop()
        def _pop(src: dict, tgt: dict):
            for k, v in src.items():
                if isinstance(v, dict):
                    next_tgt = tgt.setdefault(k, {})
                    _pop(v, next_tgt)
                else:
                    tgt[k] = v.pop()
            return tgt
        return _pop(self._params, {})

    def all(self) -> Iterator[nested_type]:
        if self.is_enumerate:
            yield from self._params.all()
        else:
            for flattened_params in Grid(self.all_flattened_params):
                yield self.nest_flattened_params(flattened_params)

    def flatten_nested_params(self,
                              nested_params: union_nested_type) -> union_flattened_type:
        flattened_params = []
        def _flatten_params(d, pre_key: Union[None, str]):
            for name, params in d.items():
                if pre_key is None:
                    next_pre_key = name
                else:
                    next_pre_key = f"{pre_key}{self._delim}{name}"
                if isinstance(params, dict):
                    _flatten_params(params, next_pre_key)
                else:
                    flattened_params.append((next_pre_key, params))
            return flattened_params
        return dict(_flatten_params(nested_params, None))

    def nest_flattened_params(self,
                              flattened_params: flattened_type) -> nested_type:
        sorted_params = sorted(map(
            lambda k, v: (k.split(self._delim), v),
            *zip(*flattened_params.items())
        ), key=len)
        l_start = len(self._idx_start)
        list_traces, nested_params = {}, {}
        for key_list, value in sorted_params:
            if len(key_list) == 1:
                nested_params[key_list[0]] = value
            else:
                last_key = key_list[-1]
                if last_key.startswith(self._idx_start):
                    list_traces.setdefault(tuple(key_list[:-1]), []).append((int(last_key[l_start:]), value))
                else:
                    parent = nested_params.setdefault(key_list[0], {})
                    for key in key_list[1:-1]:
                        parent = parent.setdefault(key, {})
                    parent[last_key] = value
        for list_key_tuple, list_values in list_traces.items():
            if len(list_key_tuple) == 1:
                parent = nested_params
            else:
                parent = nested_params.setdefault(list_key_tuple[0], {})
                for key in list_key_tuple[1:-1]:
                    parent = parent.setdefault(key, {})
            indices, values = zip(*list_values)
            assert sorted(indices) == list(range(len(indices)))
            d = self._params
            for key in list_key_tuple[:-1]:
                d = d[key]
            constructor = d[list_key_tuple[-1]]._constructor
            parent[list_key_tuple[-1]] = constructor(values[i] for i in np.argsort(indices))
        return nested_params

    def flattened2array(self,
                        flattened_params: flattened_type) -> np.ndarray:
        param_list = []
        for key in self.sorted_flattened_key:
            param = flattened_params[key]
            param = list(param) if isinstance(param, (list, tuple)) else [param]
            param_list.extend(param)
        return np.array(param_list, np.float32)

    def array2flattened(self,
                        array: np.ndarray) -> flattened_type:
        cursor = 0
        flattened = {}
        for key, offset in zip(self.sorted_flattened_key, self.sorted_flattened_offsets):
            end = cursor + offset
            data_type = self._get_data_type_from(key)
            if not isinstance(data_type, Iterable):
                value = array[cursor]
            else:
                value = array[cursor:end].tolist()
            flattened[key] = data_type.transform(value)
            cursor = end
        return flattened


__all__ = ["ParamsGenerator"]
