import numpy as np

from typing import *

from cftool.ml import *
from cftool.ml.hpo import HPOBase
from cftool.ml.param_utils import *
from cftool.optim.gd import GradientDescentMixin


class LinearRegression(GradientDescentMixin):
    def __init__(self, dim, lr, epoch):
        self.w = np.random.random([dim, 1])
        self.b = np.random.random([1])
        self._lr, self._epoch = lr, epoch

    @property
    def parameter_names(self) -> List[str]:
        return ["w", "b"]

    def loss_function(self,
                      x_batch: np.ndarray,
                      y_batch: np.ndarray,
                      batch_indices: np.ndarray) -> Dict[str, Any]:
        predictions = self.predict(x_batch)
        diff = predictions - y_batch
        return {
            "diff": diff,
            "loss": np.abs(diff).mean().item()
        }

    def gradient_function(self,
                          x_batch: np.ndarray,
                          y_batch: np.ndarray,
                          batch_indices: np.ndarray,
                          loss_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        diff = loss_dict["diff"]
        sign = np.sign(diff)
        return {
            "w": (sign * x_batch).mean(0, keepdims=True).T,
            "b": sign.mean(0)
        }

    def fit(self, x, y):
        self.setup_optimizer("adam", self._lr, epoch=self._epoch)
        self.gradient_descent(x, y)
        return self

    def predict(self, x):
        return x.dot(self.w) + self.b


def test():
    dim = 10
    w_true = np.random.random([dim, 1])
    b_true = np.random.random([1])
    x = np.random.random([1000, dim])
    y = x.dot(w_true) + b_true

    def pattern_creator(features, labels, param):
        model = LinearRegression(dim, **param)
        model.show_tqdm = False
        model.fit(features, labels)
        return ModelPattern(init_method=lambda: model)

    params = {
        "lr": Float(Exponential(1e-5, 0.1)),
        "epoch": Int(Choice(values=[2, 20, 200])),
    }

    estimators = list(map(Estimator, ["mae", "mse"]))
    for hpo_method in ["naive", "bo"]:
        hpo = HPOBase.make(hpo_method, pattern_creator, params)
        hpo.search(
            x, y, estimators,
            num_jobs=1, use_tqdm=True, verbose_level=1
        )


if __name__ == '__main__':
    test()
