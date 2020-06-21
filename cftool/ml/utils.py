import os
import math
import logging

import numpy as np
import matplotlib.pyplot as plt

from typing import *
from scipy import interp
from scipy import stats as ss
from sklearn import metrics
from functools import partial

from ..misc import *


class Anneal:
    """
    Util class which can provide annealed numbers with given `method`.
    * Formulas could be found in `_initialize` method.

    Parameters
    ----------
    method : str, indicates which anneal method to be used.
    n_iter : int, indicates how much 'steps' will be taken to reach `ceiling` from `floor`.
    floor : float, indicates the start point of the annealed number.
    ceiling : float, indicates the end point of the annealed number.

    Examples
    --------
    >>> from cftool.ml.utils import Anneal
    >>>
    >>> anneal = Anneal("linear", 50, 0.01, 0.99)
    >>> for i in range(100):
    >>>     # for i == 0, 1, ..., 48, 49, it will pop 0.01, 0.03, ..., 0.97, 0.99
    >>>     # for i == 50, 51, ..., 98, 99, it will pop 0.99, 0.99, ..., 0.99, 0.99
    >>>     print(anneal.pop())

    """

    def __init__(self, method, n_iter, floor, ceiling):
        self._n_iter = max(1, n_iter)
        self._method, self._max_iter = method, n_iter
        self._floor, self._ceiling = floor, ceiling
        self._cache = self._rs = self._cursor = 0
        self._initialize()

    def __str__(self):
        return f"Anneal({self._method})"

    __repr__ = __str__

    def _initialize(self):
        n_anneal = max(1, self._n_iter - 1)
        if self._method == "linear":
            self._cache = (self._ceiling - self._floor) / n_anneal
        elif self._method == "log":
            self._cache = (self._ceiling - self._floor) / math.log(n_anneal)
        elif self._method == "quad":
            self._cache = (self._ceiling - self._floor) / (n_anneal ** 2)
        elif self._method == "sigmoid":
            self._cache = 8 / n_anneal
        self._rs = self._floor - self._cache

    def _update_linear(self):
        self._rs += self._cache

    def _update_log(self):
        self._rs = math.log(self._cursor) * self._cache

    def _update_quad(self):
        self._rs = self._cursor ** 2 * self._cache

    def _update_sigmoid(self):
        self._rs = self._ceiling / (1 + math.exp(4 - self._cursor * self._cache))

    def pop(self):
        self._cursor += 1
        if self._cursor >= self._n_iter:
            return self._ceiling
        getattr(self, f"_update_{self._method}")()
        return self._rs

    def visualize(self):
        rs = [self.pop() for _ in range(self._max_iter)]
        plt.figure()
        plt.plot(range(len(rs)), rs)
        plt.show()
        self._initialize()
        return self


class Metrics(LoggingMixin):
    """
    Util class to calculate a whole variety of metrics.

    Warnings
    ----------
    * Notice that 2-dimensional arrays are desired, not flattened arrays.
    * Notice that first two args of each metric method must be `y` & `pred`.

    Parameters
    ----------
    metric_type : str, indicates which kind of metric is to be calculated.
    config : dict, configuration for the specific metric.
    * e.g. For quantile metric, you need to specify which quantile is to be evaluated.
    verbose_level : int, verbose level of `Metrics`.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from cftool.ml.utils import Metrics
    >>>
    >>> predictions, y_true = map(np.atleast_2d, [[1., 2., 3.], [0., 2., 1.]])
    >>> print(Metrics("mae", {}).metric(y_true.T, predictions.T))  # will be 1.

    """

    sign_dict = {
        "ddr": -1, "quantile": -1, "cdf_loss": -1, "loss": -1,
        "f1_score": 1, "r2_score": 1, "auc": 1, "multi_auc": 1,
        "acc": 1, "mae": -1, "mse": -1, "ber": -1,
        "correlation": 1, "top_k_score": 1
    }
    requires_prob_metrics = {"auc", "multi_auc"}
    optimized_binary_metrics = {"acc", "ber"}
    custom_metrics = {}

    def __init__(self, metric_type=None, config=None, verbose_level=None):
        if config is None:
            config = {}
        self.type, self.config, self._verbose_level = metric_type, config, verbose_level

    def __str__(self):
        return f"Metrics({self.type})"

    __repr__ = __str__

    @property
    def sign(self):
        return Metrics.sign_dict[self.type]

    @property
    def use_loss(self):
        return self.type == "loss"

    @property
    def requires_prob(self):
        return self.type in self.requires_prob_metrics

    def _handle_nan(self, y, pred):
        pred_valid_mask = np.all(~np.isnan(pred), axis=1)
        valid_ratio = pred_valid_mask.mean()
        if valid_ratio == 0:
            self.log_msg("all pred are nan", self.error_prefix, 2, logging.ERROR)
            return None, None
        if valid_ratio != 1:
            self.log_msg(
                f"pred contains nan (ratio={valid_ratio:6.4f})",
                self.error_prefix, 2, logging.ERROR
            )
            y, pred = y[pred_valid_mask], pred[pred_valid_mask]
        return y, pred

    @classmethod
    def add_metric(cls, f, name, sign, requires_prob):
        if name in cls.sign_dict:
            print(f"{LoggingMixin.warning_prefix}'{name}' is already registered in Metrics")
        cls.sign_dict[name] = sign
        cls.custom_metrics[name] = {
            "f": f,
            "sign": sign,
            "requires_prob": requires_prob
        }
        if requires_prob:
            cls.requires_prob_metrics.add(name)

    def metric(self, y, pred):
        if self.type is None:
            raise ValueError("`score` method was called but type is not specified in Metrics")
        y, pred = self._handle_nan(y, pred)
        if y is None or pred is None:
            return float("nan")
        custom_metric_info = self.custom_metrics.get(self.type)
        if custom_metric_info is not None:
            return custom_metric_info["f"](self, y, pred)
        return getattr(self, self.type)(y, pred)

    # config-dependent metrics

    def quantile(self, y, pred):
        q, error = self.config["q"], y - pred
        if isinstance(q, list):
            q = np.array(q, np.float32).reshape([-1, 1])
        return np.maximum(q * error, (q - 1) * error).mean(0).sum()

    def cdf_loss(self, y, pred, yq=None):
        if yq is None:
            eps = self.config.setdefault("eps", 1e-6)
            mask = y <= self.config["anchor"]
            pred = np.clip(pred, eps, 1 - eps)
            cdf_raw = pred / (1 - pred)
            return -np.mean(mask * cdf_raw - np.log(1 - pred))
        q, self.config["q"] = self.config.get("q"), pred
        loss = self.quantile(y, yq)
        if q is None:
            self.config.pop("q")
        else:
            self.config["q"] = q
        return loss

    # static metrics

    @staticmethod
    def f1_score(y, pred):
        return metrics.f1_score(y.ravel(), pred.ravel())

    @staticmethod
    def r2_score(y, pred):
        return metrics.r2_score(y.ravel(), pred.ravel())

    @staticmethod
    def auc(y, pred):
        n_classes = pred.shape[1]
        if n_classes == 2:
            return metrics.roc_auc_score(y.ravel(), pred[..., 1])
        return Metrics.multi_auc(y, pred)

    @staticmethod
    def multi_auc(y, pred):
        n_classes = pred.shape[1]
        y = get_one_hot(y.ravel(), n_classes)
        fpr, tpr = [None] * n_classes, [None] * n_classes
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y[..., i], pred[..., i])
        new_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        new_tpr = np.zeros_like(new_fpr)
        for i in range(n_classes):
            new_tpr += interp(new_fpr, fpr[i], tpr[i])
        new_tpr /= n_classes
        return metrics.auc(new_fpr, new_tpr)

    @staticmethod
    def acc(y, pred):
        return np.mean(y == pred)

    @staticmethod
    def mae(y, pred):
        return np.mean(np.abs(y - pred))

    @staticmethod
    def mse(y, pred):
        return np.mean(np.square(y - pred))

    @staticmethod
    def ber(y, pred):
        mat = metrics.confusion_matrix(y.ravel(), pred.ravel())
        tp = np.diag(mat)
        fp = mat.sum(axis=0) - tp
        fn = mat.sum(axis=1) - tp
        tn = mat.sum() - (tp + fp + fn)
        return 0.5 * np.mean((fn / (tp + fn) + fp / (tn + fp)))

    @staticmethod
    def correlation(y, pred):
        return float(ss.pearsonr(y.ravel(), pred.ravel())[0])

    # auxiliaries

    @staticmethod
    def get_binary_threshold(y, probabilities, metric_type):
        pos_probabilities = probabilities[..., 1]
        fpr, tpr, thresholds = metrics.roc_curve(y, pos_probabilities)
        _, counts = np.unique(y, return_counts=True)
        pos = counts[1] / len(y)
        if metric_type == "ber":
            metric = 0.5 * (1 - tpr + fpr)
        elif metric_type == "acc":
            metric = tpr * pos + (1 - fpr) * (1 - pos)
        else:
            raise NotImplementedError(f"transformation from fpr, tpr -> '{metric_type}' is not implemented")
        metric *= Metrics.sign_dict[metric_type]
        return thresholds[np.argmax(metric)]


def register_metric(name, sign, requires_prob):
    def _register(f):
        Metrics.add_metric(f, name, sign, requires_prob)
        return f
    return _register


estimate_fn_type = Callable[[np.ndarray], np.ndarray]
scoring_fn_type = Callable[[List[float], float, float], float]
predict_method_type = Union[estimate_fn_type, None]


class Estimator(LoggingMixin):
    """
    Util class to estimate the performances of a group of methods, on specific dataset & metric.

    Parameters
    ----------
    metric_type : str, indicates which kind of metric is to be calculated.
    verbose_level : int, verbose level used in `LoggingMixin`.
    **kwargs : used to initialize `Metrics` instance.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from cftool.ml.utils import Estimator
    >>>
    >>> x, y = map(np.atleast_2d, [[1., 2., 3.], [0., 2., 1.]])
    >>> identical = lambda x_: x_
    >>> minus_one = lambda x_: x_ - 1
    >>> # ~~~  [ info ] Results
    >>> # ==========================================================
    >>> # |             identical  |    mae     |  1.000000  |
    >>> # |             minus_one  |    mae     |  0.666667  |  <-
    >>> # ----------------------------------------------------------
    >>> Estimator("mae").estimate(x, y, {"identical": identical, "minus_one": minus_one})

    """

    def __init__(self,
                 metric_type: str,
                 *,
                 verbose_level: int = 2,
                 **kwargs):
        self._reset()
        self._verbose_level = verbose_level
        self._metric = Metrics(metric_type, **kwargs)

    def __str__(self):
        return f"Estimator({self.type})"

    __repr__ = __str__

    @property
    def type(self) -> str:
        return self._metric.type

    @property
    def sign(self) -> int:
        return self._metric.sign

    @property
    def requires_prob(self) -> bool:
        return self._metric.requires_prob

    # Core

    def _reset(self):
        self.raw_metrics = {}
        self.final_scores = {}
        self.best_method = None

    def _default_scoring(self, raw_metrics, mean, std) -> float:
        return mean - self.sign * std

    # API

    def estimate(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 methods: Dict[str, Union[estimate_fn_type, List[estimate_fn_type]]],
                 *,
                 scoring_function: Union[str, scoring_fn_type] = "default",
                 verbose_level: int = 1) -> Dict[str, Dict[str, float]]:
        self._reset()
        if isinstance(scoring_function, str):
            scoring_function = getattr(self, f"_{scoring_function}_scoring")
        for k, v in methods.items():
            if not isinstance(v, list):
                methods[k] = [v]
        self.raw_metrics = {
            name: np.array([self._metric.metric(y, method(x)) for method in sub_methods], np.float32)
            for name, sub_methods in methods.items()
        }
        msg_list = []
        statistics = {}
        best_idx, best_score = -1, -math.inf
        sorted_method_names = sorted(self.raw_metrics)
        for i, name in enumerate(sorted_method_names):
            raw_metrics = self.raw_metrics[name]
            mean, std = raw_metrics.mean().item(), raw_metrics.std().item()
            msg_list.append(f"|  {name:>20s}  |  {self.type:^8s}  |  {mean:8.6f} ± {std:8.6f}  |")
            new_score = scoring_function(raw_metrics, mean, std) * self.sign
            self.final_scores[name] = new_score
            if new_score > best_score:
                best_idx, best_score = i, new_score
            method_statistics = statistics.setdefault(name, {})
            method_statistics["mean"], method_statistics["std"] = mean, std
            method_statistics["score"] = new_score
        self.best_method = sorted_method_names[best_idx]
        msg_list[best_idx] += "  <-  "
        width = max(map(len, msg_list))
        msg_list.insert(0, "=" * width)
        msg_list.append("-" * width)
        self.log_block_msg("\n".join(msg_list), self.info_prefix, "Results", verbose_level)
        return statistics


class ModelPattern(LoggingMixin):
    """
    Util class to create an interface for users to leverage `Comparer` & `HPO` (and more in the future).

    Parameters
    ----------
    init_method : Callable[[], object]
    * If None, then `ModelPattern` will not perform model creation.
    * If Callable, then `ModelPattern` will initialize a model with it.
    train_method : Callable[[object], None]
    * If None, then `ModelPattern` will not perform model training.
    * If Callable, then `ModelPattern` will train the created model (from `init_method`) with it.
    predict_method : Union[str, Callable[[np.ndarray], np.ndarray]]
    * If str, then `ModelPattern` will use `getattr` to get the label predict method of the model obtained
        from above. In this case, `init_method` must be provided (`train_method` is still optional, because you can
        create a trained model in `init_method`).
    * If Callable, then `ModelPattern` will use it for label prediction.
    * Notice that predict_method should return a column vector (e.g. out.shape = [n, 1])
    predict_prob_method : Union[str, Callable[[np.ndarray], np.ndarray]]
    * If str, then `ModelPattern` will use `getattr` to get the probability prediction method of the model obtained
        from above. In this case, `init_method` must be provided (`train_method` is still optional, because you can
        create a trained model in `init_method`).
    * If Callable, then `ModelPattern` will use it for probability prediction.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from cftool.ml.utils import ModelPattern
    >>>
    >>> x, y = map(np.atleast_2d, [[1., 2., 3.], [0., 2., 1.]])
    >>> predict_method = lambda x_: x_ - 1
    >>> init_method = lambda: type("Test", (), {"predict": lambda self, x_: predict_method(x_)})()
    >>> # Will both be [[0., 1., 2.]]
    >>> ModelPattern(init_method=init_method).predict(x)
    >>> ModelPattern(predict_method=predict_method).predict(x)

    """

    def __init__(self,
                 *,
                 init_method: Callable[[], object] = None,
                 train_method: Callable[[object], None] = None,
                 predict_method: Union[str, Callable[[np.ndarray], np.ndarray]] = "predict",
                 predict_prob_method: Union[str, Callable[[np.ndarray], np.ndarray]] = "predict_prob",
                 verbose_level: int = 2):
        if init_method is None:
            self.model = None
        else:
            self.model = init_method()
        if train_method is not None:
            train_method(self.model)
        self._predict_method = predict_method
        self._predict_prob_method = predict_prob_method
        self._verbose_level = verbose_level

    def predict_method(self,
                       requires_prob: bool) -> predict_method_type:
        predict_method = self._predict_prob_method if requires_prob else self._predict_method
        if isinstance(predict_method, str):
            if self.model is None:
                raise ValueError("Either init_method or Callable predict_method is required in ModelPattern")
            predict_method = getattr(self.model, predict_method, None)
        elif self.model is not None:
            self.log_msg(
                "predict_method is Callable but model is also created, which has no effect",
                self.warning_prefix, 2, logging.WARNING
            )
        return predict_method

    def predict(self,
                x: np.ndarray,
                *,
                requires_prob: bool = False) -> np.ndarray:
        return self.predict_method(requires_prob)(x)

    @classmethod
    def repeat(cls, n: int, **kwargs) -> List["ModelPattern"]:
        return [cls(**kwargs) for _ in range(n)]


collate_fn_type = Callable[[List[np.ndarray], bool], np.ndarray]


class EnsemblePattern:
    """
    Util class to create an interface for users to leverage `Comparer` & `HPO` in an ensembled way.

    Parameters
    ----------
    model_patterns : List[ModelPattern], list of `ModelPattern` we want to ensemble from.
    ensemble_method : Union[str, collate_fn_type], ensemble method we use to collate the results.
    * If str, then `EnsemblePattern` will use `getattr` to get the collate function.
        Currently only 'default' is supported, which implements voting for classification
        and averaging for regression.
    * If collate_fn_type, then `EnsemblePattern` will use it to collate the results directly.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from cftool.ml.utils import ModelPattern, EnsemblePattern
    >>>
    >>> x, y = map(np.atleast_2d, [[1., 2., 3.], [0., 2., 1.]])
    >>> identical = lambda x_: x_
    >>> minus_one = lambda x_: x_ - 1
    >>> identical_pattern = ModelPattern(predict_method=identical)
    >>> minus_one_pattern = ModelPattern(predict_method=minus_one)
    >>> # Averaging 'identical' & 'minus_one' -> 'minus_0.5'
    >>> ensemble = EnsemblePattern([identical_pattern, minus_one_pattern])
    >>> # [[0.5 1.5 2.5]]
    >>> print(ensemble.predict(x))

    """

    def __init__(self,
                 model_patterns: List[ModelPattern],
                 ensemble_method: Union[str, collate_fn_type] = "default"):
        self._patterns = model_patterns
        self._ensemble_method = ensemble_method

    def __len__(self):
        return len(self._patterns)

    @property
    def collate_fn(self) -> collate_fn_type:
        if callable(self._ensemble_method):
            return self._ensemble_method
        return getattr(self, f"_{self._ensemble_method}_collate")

    # Core

    @staticmethod
    def _default_collate(arrays: List[np.ndarray],
                         requires_prob: bool) -> np.ndarray:
        predictions = np.array(arrays)
        if not requires_prob and np.issubdtype(predictions.dtype, np.integer):
            max_class = predictions.max() + 1
            predictions = predictions.squeeze(2).T
            counts = np.apply_along_axis(partial(np.bincount, minlength=max_class), 1, predictions)
            return counts.argmax(1).reshape([-1, 1])
        return predictions.mean(0)

    # API

    def predict_method(self,
                       requires_prob: bool) -> predict_method_type:
        predict_methods = list(map(ModelPattern.predict_method, self._patterns, len(self) * [requires_prob]))
        predict_methods = [method for method in predict_methods if method is not None]
        if not predict_methods:
            return
        def _predict(x: np.ndarray):
            predictions = [method(x) for method in predict_methods]
            return self.collate_fn(predictions, requires_prob)
        return _predict

    def predict(self,
                x: np.ndarray,
                *,
                requires_prob: bool = False) -> np.ndarray:
        return self.predict_method(requires_prob)(x)

    @classmethod
    def from_same_methods(cls,
                          n: int,
                          ensemble_method: Union[str, collate_fn_type] = "default",
                          **kwargs):
        return cls([ModelPattern(**kwargs) for _ in range(n)], ensemble_method)


pattern_type = Union[ModelPattern, EnsemblePattern]
patterns_type = Union[pattern_type, List[pattern_type]]


class Comparer(LoggingMixin):
    """
    Util class to compare a group of `patterns_type`s on a group of `Estimator`s.

    Parameters
    ----------
    patterns : Dict[str, Union[patterns_type, Dict[str, patterns_type]]]
    * If values are `patterns_type`, then all estimators will use this only `patterns_type` make predictions.
    * If values are Dict[str, patterns_type], then each estimator will use values.get(estimator.type) to
      make predictions. If corresponding `patterns` does not exist (values.get(estimator.type) is None),
      then corresponding estimation will be skipped.
    estimators : List[Estimator], list of estimators which we are interested in.
    verbose_level : int, verbose level used in `LoggingMixin`.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from cftool.ml.utils import ModelPattern, Estimator, Comparer
    >>>
    >>> x, y = map(np.atleast_2d, [[1., 2., 3.], [0., 2., 1.]])
    >>> identical = lambda x_: x_
    >>> minus_one = lambda x_: x_ - 1
    >>> patterns = {
    >>>     "identical": ModelPattern(predict_method=identical),
    >>>     "minus_one": ModelPattern(predict_method=minus_one)
    >>> }
    >>> estimators = [Estimator("mse"), Estimator("mae")]
    >>> # ~~~  [ info ] Results
    >>> # ==========================================================
    >>> # |             identical  |    mse     |  1.666667  |
    >>> # |             minus_one  |    mse     |  0.666667  |  <-
    >>> # ----------------------------------------------------------
    >>> # ~~~  [ info ] Results
    >>> # ==========================================================
    >>> # |             identical  |    mae     |  1.000000  |
    >>> # |             minus_one  |    mae     |  0.666667  |  <-
    >>> # ----------------------------------------------------------
    >>> comparer = Comparer(patterns, estimators).compare(x, y)
    >>> # {'mse': {'identical': 1.666667, 'minus_one': 0.666666},
    >>> # 'mae': {'identical': 1.0, 'minus_one': 0.666666}}
    >>> print(comparer.scores)
    >>> # {'mse': 'minus_one', 'mae': 'minus_one'}
    >>> print(comparer.best_methods)

    """

    def __init__(self,
                 patterns: Dict[str, Union[patterns_type, Dict[str, patterns_type]]],
                 estimators: List[Estimator],
                 *,
                 verbose_level: int = 2):
        self.patterns = patterns
        self.estimators = dict(zip([estimator.type for estimator in estimators], estimators))
        self._verbose_level = verbose_level

    @property
    def raw_metrics(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {k: v.raw_metrics for k, v in self.estimators.items()}

    @property
    def final_scores(self) -> Dict[str, Dict[str, float]]:
        return {k: v.final_scores for k, v in self.estimators.items()}

    @property
    def best_methods(self) -> Dict[str, str]:
        return {k: v.best_method for k, v in self.estimators.items()}

    def compare(self,
                x: np.ndarray,
                y: np.ndarray,
                *,
                scoring_function: Union[str, scoring_fn_type] = "default",
                verbose_level: int = 1) -> "Comparer":
        for estimator in self.estimators.values():
            methods = {}
            for model_name, patterns in self.patterns.items():
                if isinstance(patterns, dict):
                    patterns = patterns.get(estimator.type)
                if patterns is None:
                    continue
                if not isinstance(patterns, list):
                    patterns = [patterns]
                invalid = False
                predict_methods = []
                requires_prob = estimator.requires_prob
                for pattern in patterns:
                    if pattern is None:
                        invalid = True
                        break
                    predict_methods.append(pattern.predict_method(requires_prob))
                    if predict_methods[-1] is None:
                        invalid = True
                        self.log_msg(
                            f"{estimator} requires probability predictions but {model_name} "
                            f"does not have probability predicting method, skipping",
                            self.warning_prefix, verbose_level, logging.WARNING
                        )
                        break
                if invalid:
                    continue
                methods[model_name] = predict_methods
            estimator.estimate(
                x, y, methods,
                scoring_function=scoring_function,
                verbose_level=verbose_level
            )
        return self


class ScalarEMA:
    """
    Util class to record Exponential Moving Average (EMA) for scalar value.

    Parameters
    ----------
    decay : float, decay rate for EMA.
    * new = (1 - decay) * current + decay * history; history = new

    Examples
    --------
    >>> from cftool.ml.utils import ScalarEMA
    >>>
    >>> ema = ScalarEMA(0.5)
    >>> for i in range(4):
    >>>     print(ema.update("score", 0.5 ** i))  # 1, 0.75, 0.5, 0.3125

    """

    def __init__(self, decay):
        self._decay = decay
        self._ema_records = {}

    def __str__(self):
        return f"ScalarEMA({self._decay})"

    __repr__ = __str__

    def get(self, name):
        return self._ema_records.get(name)

    def update(self, name, new_value):
        history = self._ema_records.get(name)
        if history is None:
            updated = new_value
        else:
            updated = (1 - self._decay) * new_value + self._decay * history
        self._ema_records[name] = updated
        return updated


class Visualizer:
    """
    Visualization class.

    Methods
    ----------
    bar(self, data, classes, categories, save_name="bar_plot", title="",
            padding=1e-3, expand_floor=5, replace=True)
        Make bar plot with given `data`.
        * data : np.ndarray, containing values for the bar plot, where data.shape =
            * (len(categories), ), if len(classes) == 1.
            * (len(classes), len(categories)), otherwise.
        * classes : list(str), list of str which indicates each class.
            * each class will has its own color.
            * len(classes) indicates how many bars are there in one category (side by side).
        * categories : list(str), list of str which indicates each category.
            * a category will be a tick along x-axis.
        * save_name : str, saving name of this bar plot.
        * title : str, title of this bar plot.
        * padding : float, minimum value of each bar.
        * expand_floor : int, when len(categories) > `expand_floor`, the width of the figure will expand.
            * for len(classes) == 1, `expand_floor` will be multiplied by 2 internally.
        * overwrite : bool
            whether overwrite the existing file with the same file name of this plot's saving name.

    function(self, f, x_min, x_max, classes, categories, save_names=None,
             n_sample=1000, expand_floor=5, overwrite=True):
        Make multiple (len(categories)) line plots with given function (`f`)
        * f : function
            * input should be an np.ndarray with shape == (n, n_categories).
            * output should be an np.ndarray with shape == (n, n_categories, n_categories).
        * x_min : np.ndarray, minimum x-values for each line plot.
            * len(x_min) should be len(categories).
        * x_max : np.ndarray, maximum x-values for each line plot.
            * len(x_max) should be len(categories).
        * classes : list(str), list of str which indicates each class.
            * each class will has its own color.
            * len(classes) indicates how many bars are there in one category (side by side).
        * categories : list(str), list of str which indicates each category.
            * every category will correspond to a line plot.
        * save_names : list(str), saving names of these line plots.
        * n_sample : int, sample density along x-axis.
        * expand_floor : int, the width of the figures will be expanded with ratios calculated by:
            expand_ratios = np.maximum(1., np.abs(x_min) / expand_floor, x_max / expand_floor)
        * overwrite : bool
            whether overwrite the existing file with the same file name of this plot's saving name.

    """

    def __init__(self, export_folder):
        self.export_folder = os.path.abspath(export_folder)
        os.makedirs(self.export_folder, exist_ok=True)

    def _get_save_name(self, save_name):
        counter, tmp_save_name = 0, save_name
        while os.path.isfile(os.path.join(self.export_folder, f"{tmp_save_name}.png")):
            counter += 1
            tmp_save_name = f"{save_name}_{counter}"
        return tmp_save_name

    def bar(self, data: np.ndarray, classes: list, categories: list, save_name="bar_plot", title="",
            padding=1e-3, expand_floor=5, overwrite=True):
        n_class, n_categories = map(len, [classes, categories])
        data = [data / data.sum() + padding] if n_class == 1 else data / data.sum(0) + padding
        expand_floor = expand_floor * 2 if n_class == 1 else expand_floor
        colors = plt.cm.Paired([i / n_class for i in range(n_class)])
        x_base = np.arange(1, n_categories + 1)
        expand_ratio = max(1., n_categories / expand_floor)
        fig = plt.figure(figsize=(6.4 * expand_ratio, 4.8))
        plt.title(title)
        n_divide = n_class - 1
        width = 0.35 / max(1, n_divide)
        cls_ratio = 0.5 if n_class == 1 else 1
        for cls in range(n_class):
            plt.bar(x_base - width * (0.5 * n_divide - cls_ratio * cls), data[cls], width=width,
                    facecolor=colors[cls], edgecolor="white",
                    label=classes[cls])
        plt.xticks(
            [i for i in range(len(categories) + 2)],
            [""] + categories + [""]
        )
        plt.legend()
        plt.setp(plt.xticks()[1], rotation=30, horizontalalignment='right')
        plt.ylim(0, 1.2 + padding)
        fig.tight_layout()
        if not overwrite:
            save_name = self._get_save_name(save_name)
        plt.savefig(os.path.join(self.export_folder, f"{save_name}.png"))
        plt.close()

    def function(self, f: Callable[[np.ndarray], np.ndarray], x_min: np.ndarray, x_max: np.ndarray,
                 classes: list, categories: list, save_names=None, n_sample=1000,
                 expand_floor=5, overwrite=True):
        n_class, n_categories = map(len, [classes, categories])
        gaps = x_max - x_min
        x_base = np.linspace(x_min - 0.1 * gaps, x_max + 0.1 * gaps, n_sample)
        f_values = np.split(f(x_base), n_class, axis=1)
        if save_names is None:
            save_names = ["function_plot"] * n_categories
        colors = plt.cm.Paired([i / n_class for i in range(n_class)])
        expand_ratios = np.maximum(1., np.abs(x_min) / expand_floor, x_max / expand_floor)
        for i, (category, save_name, ratio, local_min, local_max, gap) in enumerate(zip(
                categories, save_names, expand_ratios, x_min, x_max, gaps)):
            plt.figure(figsize=(6.4 * ratio, 4.8))
            plt.title(f"pdf for {category}")
            local_base = x_base[..., i]
            for c in range(n_class):
                f_value = f_values[c][..., i].ravel()
                plt.plot(local_base, f_value, c=colors[c], label=f"class: {classes[c]}")
            plt.xlim(local_min - 0.2 * gap, local_max + 0.2 * gap)
            plt.legend()
            if not overwrite:
                save_name = self._get_save_name(save_name)
            plt.savefig(os.path.join(self.export_folder, f"{save_name}.png"))

    @staticmethod
    def visualize1d(method: Callable,
                    x: np.ndarray,
                    y: np.ndarray = None,
                    *,
                    title: str = None,
                    num_samples: int = 100,
                    expand_ratio: float = 0.25,
                    return_canvas: bool = False) -> Union[None, np.ndarray]:
        if x.shape[1] != 1:
            raise ValueError("visualize1d only supports 1-dimensional features")
        plt.figure()
        plt.title(title)
        if y is not None:
            plt.scatter(x, y, c="g", s=20)
        x_min, x_max = x.min(), x.max()
        expand = expand_ratio * (x_max - x_min)
        x0 = np.linspace(x_min - expand, x_max + expand, num_samples).reshape([-1, 1])
        plt.plot(x0, method(x0).ravel())
        return show_or_return(return_canvas)

    @staticmethod
    def visualize2d(method,
                    x: np.ndarray,
                    y: np.ndarray = None,
                    *,
                    title: str = None,
                    dense: int = 200,
                    padding: float = 0.1,
                    return_canvas: bool = False,
                    draw_background: bool = True,
                    extra_scatters: np.ndarray = None,
                    emphasize_indices: np.ndarray = None) -> Union[None, np.ndarray]:
        axis = x.T
        if axis.shape[0] != 2:
            raise ValueError("visualize2d only supports 2-dimensional features")
        nx, ny, padding = dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])  # type: float
        y_min, y_max = np.min(axis[1]), np.max(axis[1])  # type: float
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        def get_base(_nx, _ny):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            n_xf, n_yf = np.meshgrid(_xf, _yf)
            return _xf, _yf, np.c_[n_xf.ravel(), n_yf.ravel()]

        xf, yf, base_matrix = get_base(nx, ny)
        z = method(base_matrix).reshape((nx, ny))

        labels = y.ravel()
        num_labels = y.max().item() + 1
        colors = plt.cm.rainbow([i / num_labels for i in range(num_labels)])[labels]

        plt.figure()
        plt.title(title)
        if draw_background:
            xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
            plt.pcolormesh(xy_xf, xy_yf, z, cmap=plt.cm.Pastel1)
        else:
            plt.contour(xf, yf, z, c='k-', levels=[0])
        plt.scatter(axis[0], axis[1], c=colors)

        if emphasize_indices is not None:
            indices = np.array([False] * len(axis[0]))
            indices[np.asarray(emphasize_indices)] = True
            plt.scatter(axis[0][indices], axis[1][indices], s=80,
                        facecolors="None", zorder=10)
        if extra_scatters is not None:
            plt.scatter(*np.asarray(extra_scatters).T, s=80, zorder=25, facecolors="red")

        return show_or_return(return_canvas)


__all__ = [
    "Anneal", "Metrics", "ScalarEMA", "Visualizer",
    "Estimator", "ModelPattern", "EnsemblePattern", "Comparer",
    "pattern_type", "patterns_type", "estimate_fn_type", "scoring_fn_type", "register_metric"
]
