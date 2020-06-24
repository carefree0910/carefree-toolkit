import math
import pprint
import random
import logging

import numpy as np

from typing import *
from tqdm import tqdm
from collections import defaultdict
from abc import abstractmethod, ABCMeta

from ..utils import *
from ...misc import *
from ..param_utils import *
from ...dist import Parallel

hpo_dict: Dict[str, Type["HPOBase"]] = {}
pattern_creator_type = Callable[[np.ndarray, np.ndarray, Dict[str, Any]], pattern_type]


class HPOBase(LoggingMixin, metaclass=ABCMeta):
    def __init__(self,
                 pattern_creator: pattern_creator_type,
                 params: Dict[str, DataType],
                 *,
                 verbose_level: int = 2,
                 **kwargs):
        self._caches = {}
        self._init_config(**kwargs)
        self._creator = pattern_creator
        self.param_generator = ParamsGenerator(params)
        self._verbose_level = verbose_level

    @property
    @abstractmethod
    def is_sequential(self) -> bool:
        pass

    @property
    def last_param(self) -> nested_type:
        return self.param_mapping[self.last_code]

    @property
    def last_patterns(self) -> List[pattern_type]:
        return self.patterns[self.last_code]

    def _init_config(self, **kwargs):
        pass

    def _sample_param(self) -> Union[None, nested_type]:
        if self.is_sequential:
            raise NotImplementedError
        return

    def _get_scores(self, patterns: List[pattern_type]) -> Dict[str, float]:
        key = "core"
        comparer = Comparer({key: patterns}, self.estimators)
        final_scores = comparer.compare(
            self.x_validation, self.y_validation,
            scoring_function=self._estimator_scoring_function,
            verbose_level=6
        ).final_scores
        return {metric: scores[key] for metric, scores in final_scores.items()}

    def _core(self,
              param: nested_type,
              *,
              parallel_run: bool = False) -> List[pattern_type]:
        range_list = list(range(self._num_retry))
        _task = lambda _=0: self._creator(self.x_train, self.y_train, param)
        tqdm_config = {"position": 1, "leave": False}
        if not parallel_run:
            if self._use_tqdm:
                range_list = tqdm(range_list, **tqdm_config)
            local_patterns = [_task() for _ in range_list]
        else:
            local_patterns = Parallel(
                self._num_jobs,
                use_tqdm=self._use_tqdm,
                tqdm_config=tqdm_config
            )(_task, range_list).ordered_results
        return local_patterns

    def search(self,
               x: np.ndarray,
               y: np.ndarray,
               estimators: List[Estimator],
               x_validation: np.ndarray = None,
               y_validation: np.ndarray = None,
               *,
               num_jobs: int = 4,
               num_retry: int = 4,
               num_search: Union[str, int, float] = 10,
               score_weights: Union[Dict[str, float], None] = None,
               estimator_scoring_function: Union[str, scoring_fn_type] = "default",
               use_tqdm: bool = True,
               verbose_level: int = 3) -> "HPOBase":

        if x_validation is None or y_validation is None:
            x_validation, y_validation = x, y

        self.estimators = estimators
        self.x_train, self.y_train = x, y
        self.x_validation, self.y_validation = x_validation, y_validation

        num_params = self.param_generator.num_params
        if isinstance(num_search, str):
            if num_search != "all":
                raise ValueError(f"num_search can only be 'all' when it is a string, '{num_search}' found")
            if num_params == math.inf:
                raise ValueError("num_search is 'all' but we have infinite params to search")
            num_search = num_params
        if num_search > num_params:
            self.log_msg(
                f"`n` is larger than total choices we've got ({num_params}), therefore only "
                f"{num_params} searches will be run", self.warning_prefix, msg_level=logging.WARNING
            )
            num_search = num_params
        num_jobs = min(num_search, num_jobs)

        self._use_tqdm = use_tqdm
        if score_weights is None:
            score_weights = {estimator.type: 1. for estimator in estimators}
        self._score_weights = score_weights
        self._estimator_scoring_function = estimator_scoring_function
        self._num_retry, self._num_jobs = num_retry, num_jobs

        with timeit("Generating Patterns"):
            if self.is_sequential:
                self.patterns, self.param_mapping = {}, {}
                iterator = list(range(num_search))
                if use_tqdm:
                    iterator = tqdm(iterator, position=0)
                for _ in iterator:
                    param = self._sample_param()
                    self.last_code = hash_code(str(param))
                    self.param_mapping[self.last_code] = param
                    self.patterns[self.last_code] = self._core(param, parallel_run=True)
            else:
                if num_params == math.inf:
                    all_params = [self.param_generator.pop() for _ in range(num_search)]
                else:
                    all_params = []
                    all_indices = set(random.sample(list(range(num_search)), k=num_search))
                    for i, param in enumerate(self.param_generator.all()):
                        if i in all_indices:
                            all_params.append(param)
                        if len(all_params) == num_search:
                            break

                codes = list(map(hash_code, map(str, all_params)))
                self.param_mapping = dict(zip(codes, all_params))
                if num_jobs <= 1:
                    if self._use_tqdm:
                        all_params = tqdm(all_params)
                    patterns = list(map(self._core, all_params))
                else:
                    patterns = Parallel(num_jobs)(self._core, all_params).ordered_results
                self.patterns = dict(zip(codes, patterns))
                self.last_code = codes[-1]

        self.comparer = Comparer(self.patterns, estimators)
        self.comparer.compare(
            x_validation, y_validation,
            scoring_function=estimator_scoring_function,
            verbose_level=verbose_level
        )

        weighted_scores = defaultdict(float)
        for metric, scores in self.comparer.final_scores.items():
            for method, score in scores.items():
                weighted_scores[method] += self._score_weights[metric] * score
        sorted_methods = sorted(weighted_scores)
        sorted_methods_scores = [weighted_scores[key] for key in sorted_methods]
        best_method = sorted_methods[np.argmax(sorted_methods_scores).item()]
        self.best_param = self.param_mapping[best_method]

        best_methods = self.comparer.best_methods
        self.best_params = {k: self.param_mapping[v] for k, v in best_methods.items()}
        param_msgs = {k: pprint.pformat(v) for k, v in self.best_params.items()}
        msg = "\n".join(
            sum([[
                "-" * 100,
                f"{k} ({self.comparer.final_scores[k][best_methods[k]]:8.6f})",
                "-" * 100,
                param_msgs[k]
            ] for k in sorted(param_msgs)], [])
            + [
                "-" * 100, f"best ({best_method})", "-" * 100,
                pprint.pformat(self.best_param)
            ]
            + ["-" * 100]
        )
        self.log_block_msg(msg, self.info_prefix, "Best Parameters", verbose_level - 1)

        return self

    @staticmethod
    def make(method: str, *args, **kwargs) -> "HPOBase":
        return hpo_dict[method](*args, **kwargs)

    @classmethod
    def register(cls, name):
        global hpo_dict
        return register_core(name, hpo_dict)


__all__ = ["HPOBase", "hpo_dict"]
