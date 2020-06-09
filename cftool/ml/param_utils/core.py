import math
import numpy as np

from typing import Union

from .data_types import *
from .distributions import *
from ...misc import prod, Grid


class ParamsGenerator:
    """
    Parameter generator for param searching, see cftool.ml.hpo.base.HPOBase for usage.

    Parameters
    ----------
    params : {DataType, Iterable, dict}, parameter settings.
    * If DataType, then distribution of this DataType must be `Choice`. In this case, we'll simply 'enumerate'
    through the config choices.
    * If Iterable, then each element should be a DataType. It could be nested.
    * If dict, then each key should correspond to a config key, and its value should correspond to the
    desired value distribution. It could be nested.

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

    def __init__(self, params: Union[DataType, Iterable, dict]):
        self._params = params
        self._delim, self._idx_start = "^_^", "$$$"
        if isinstance(self._params, DataType):
            assert_msg = "distribution must be `Choice` when DataType is used as `params`"
            assert isinstance(self._params.dist, Choice), assert_msg

    @property
    def n_params(self):
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
    def is_enumerate(self):
        return isinstance(self._params, (DataType, Iterable))

    @staticmethod
    def _append_ax_param(name: str, param: DataType, flattened_params):
        param_dist = param.dist
        if isinstance(param_dist, Choice):
            local_params = param.all()
            if len(local_params) == 1:
                flattened_params.append({"name": name, "type": "fixed", "value": local_params[0]})
            else:
                flattened_params.append({"name": name, "type": "choice", "values": local_params})
        else:
            log_scale = isinstance(param_dist, Exponential)
            if isinstance(param, Int):
                param_type = "int"
            elif isinstance(param, Float):
                param_type = "float"
            elif isinstance(param, Bool):
                param_type = "bool"
            elif isinstance(param, String):
                param_type = "str"
            else:
                raise NotImplementedError
            flattened_params.append({
                "name": name, "type": "range", "bounds": [param.lower, param.upper],
                "parameter_type": param_type, "log_scale": log_scale
            })

    def pop(self):
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

    def all(self):
        if self.is_enumerate:
            yield from self._params.all()
        else:
            def _all(src: dict, tgt: dict):
                for k, v in src.items():
                    if isinstance(v, dict):
                        next_tgt = tgt.setdefault(k, {})
                        _all(v, next_tgt)
                    else:
                        tgt[k] = v.all()
                return tgt
            all_flattened_params_list = self.get_flattened_params(_all(self._params, {}))
            all_flattened_params = dict(all_flattened_params_list)
            for flattened_params in Grid(all_flattened_params):
                yield self.nest_flattened_params(flattened_params)

    def get_flattened_params(self, tgt_params=None, use_ax=False):
        if tgt_params is None:
            tgt_params = self._params
        if isinstance(tgt_params, (DataType, Iterable)):
            raise ValueError("get_flattened_params method requires nested params")
        flattened_params = []
        def _flatten_params(d: dict, pre_key: Union[None, str]):
            for name, params in d.items():
                if pre_key is None:
                    next_pre_key = name
                else:
                    next_pre_key = f"{pre_key}{self._delim}{name}"
                if isinstance(params, dict):
                    _flatten_params(params, next_pre_key)
                elif isinstance(params, Iterable):
                    for i, param in enumerate(params.values):
                        name = f"{next_pre_key}{self._delim}{self._idx_start}{i}"
                        if not use_ax:
                            flattened_params.append((name, param))
                        else:
                            self._append_ax_param(name, param, flattened_params)
                else:
                    if not use_ax:
                        flattened_params.append((next_pre_key, params))
                    else:
                        self._append_ax_param(next_pre_key, params, flattened_params)
            return flattened_params
        return _flatten_params(tgt_params, None)

    def nest_flattened_params(self, flattened_params):
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


__all__ = ["ParamsGenerator"]
