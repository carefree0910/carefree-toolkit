import warnings

from typing import *
from tqdm import tqdm
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from .acquisition import *
from .target_space import *
from ...misc import *


class BayesianOptimization:
    def __init__(self,
                 fn: fn_type,
                 params: params_type,
                 *,
                 gp_params: Dict[str, Any] = None,
                 acquisition: str = "ucb",
                 normalization: Union[str, None] = None,
                 xi: float = 0.01,
                 kappa: float = 2.,
                 kappa_decay: float = 1.,
                 kappa_decay_delay: int = 0):

        self._queue = []
        self._space = TargetSpace(fn, params, normalization=normalization)

        if gp_params is None:
            gp_params = {}
        kernel_params = gp_params.pop("kernel_params", {})
        kernel_params.setdefault("nu", 2.5)
        gp_params.setdefault("alpha", 1e-6)
        gp_params.setdefault("normalize_y", True)
        gp_params.setdefault("n_restarts_optimizer", 5)
        self._gp = GaussianProcessRegressor(
            kernel=Matern(**kernel_params),
            **gp_params
        )

        self._acquisition = Acquisition(self._gp, acquisition, xi, kappa, kappa_decay, kappa_decay_delay)

    @property
    def space(self) -> TargetSpace:
        return self._space

    @property
    def best_result(self) -> Result:
        return self._space.best_result

    @property
    def all_results(self) -> List[Result]:
        return self._space.all_results

    def register(self,
                 param: flattened_type,
                 score: float) -> "BayesianOptimization":
        self._space.register(param, score)
        return self

    def probe(self,
              param: flattened_type,
              *,
              to_queue: bool = True) -> "BayesianOptimization":
        if to_queue:
            self._queue.append(param)
        else:
            self._space.probe(param)
        return self

    def suggest(self,
                num_warmup: int,
                num_iter: int) -> flattened_type:
        if self._space.is_empty:
            return self._space.sample()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.tried_flattened_params, self._space.tried_scores)

        suggestion = self._acquisition.search_max(
            self._space.tried_scores.max(),
            self._space, num_warmup, num_iter
        )

        return self._space.array2param(suggestion)

    def maximize(self,
                 *,
                 init_points: int = 5,
                 num_epoch: int = 20,
                 num_warmup: int = 10000,
                 num_iter: int = 10,
                 use_tqdm: bool = True) -> "BayesianOptimization":

        if not self._queue and self._space.is_empty:
            init_points = max(init_points, 1)
        for _ in range(init_points):
            self._queue.append(self._space.sample())

        counter = 0
        iterator = None if not use_tqdm else tqdm(list(range(len(self._queue) + num_epoch)))
        while self._queue or counter < num_epoch:
            if self._queue:
                x_probe = self._queue.pop(0)
            else:
                self._acquisition.update()
                x_probe = self.suggest(num_warmup, num_iter)
                counter += 1
            self.probe(x_probe, to_queue=False)
            if iterator is not None:
                iterator.update()

        return self


__all__ = ["BayesianOptimization"]
