import os
import math
import logging

import numpy as np
import matplotlib.pyplot as plt

from typing import *
from scipy import interp
from scipy import stats as ss
from sklearn import metrics
from itertools import product

from .misc import *


class Anneal:
    """
    Util class which can provide annealed numbers with given `method`
    * Formulas could be found in `_initialize` method

    Parameters
    ----------
    method : str, indicates which anneal method to be used
    n_iter : int, indicates how much 'steps' will be taken to reach `ceiling` from `floor`
    floor : float, indicates the start point of the annealed number
    ceiling : float, indicates the end point of the annealed number

    Examples
    --------
    >>> anneal = Anneal("linear", 50, 0.01, 0.99)
    >>> for i in range(100):
    >>>     # for i == 0, 1, ..., 48, 49, it will pop 0.01, 0.03, ..., 0.97, 0.99
    >>>     # for i == 50, 51, ..., 98, 99, it will pop 0.99, 0.99, ..., 0.99, 0.99
    >>>     anneal.pop()

    """

    def __init__(self, method, n_iter, floor, ceiling):
        self._n_iter = max(1, n_iter)
        self._method, self._max_iter = method, n_iter
        self._floor, self._ceiling = floor, ceiling
        self._cache = self._rs = self._cursor = 0
        self._initialize()

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
    Util class to calculate a whole variety of metrics

    Warnings
    ----------
    * Notice that 2-dimensional arrays are desired, not flattened arrays
    * Notice that first two args of each metric method must be `y` & `pred`

    Parameters
    ----------
    metric_type : str, indicates which kind of metric is to be calculated
    config : dict
        Configuration for the specific metric
        * e.g. for quantile metric, you need to specify which quantile is to be evaluated
    verbose_level : int, verbose level of Metrics

    Examples
    --------
    >>> import numpy as np
    >>> predictions, y_true = map(np.atleast_2d, [[1., 2., 3.], [0., 2., 1.]])
    >>> print(Metrics("mae", {}).score(y_true.T, predictions.T))  # will be 1.

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

    def score(self, y, pred):
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


class ScalarEMA:
    """
    Util class to record Exponential Moving Average (EMA) for scalar value

    Parameters
    ----------
    decay : float, decay rate for EMA
        * Formula: new = (1 - decay) * current + decay * history; history = new

    Examples
    --------
    >>> ema = ScalarEMA(0.5)
    >>> for i in range(4):
    >>>     print(ema.update("score", 0.5 ** i))  # 1, 0.75, 0.5, 0.3125

    """

    def __init__(self, decay):
        self._decay = decay
        self._ema_records = {}

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


class Grid:
    """
    Util class provides permutation of simple, flattened param dicts
    * For permutation of complex, nested param dicts, please refers to `ParamGenerator` below

    Parameters
    ----------
    param_grid : dict[str, list(int)]
        Indicates param names and corresponding possible values

    Examples
    ----------
    >>> grid = Grid({"a": [1, 2, 3], "b": [1, 2, 3]})
    >>> for param in grid:
    >>>     print(param)
    >>> # output : {'a': 1, 'b': 1}, {'a': 1, 'b': 2}, {'a': 1, 'b': 3}
    >>> #          {'a': 2, 'b': 1}, ..., {'a': 3, 'b': 3}

    """

    def __init__(self, param_grid):
        self._grid = param_grid

    def __iter__(self):
        items = sorted(self._grid.items())
        if not items:
            yield {}
        else:
            keys, values = zip(*items)
            for v in map(list, product(*values)):
                for i, vv in enumerate(v):
                    if isinstance(vv, tuple) and len(vv) == 1:
                        v[i] = vv[0]
                params = dict(zip(keys, v))
                yield params


class Visualizer:
    """
    Visualization class

    Methods
    ----------
    bar(self, data, classes, categories, save_name="bar_plot", title="",
            padding=1e-3, expand_floor=5, replace=True)
        Make bar plot with given `data`
        * data : np.ndarray
            Containing values for the bar plot, where data.shape =
            * (len(categories), ), if len(classes) == 1
            * (len(classes), len(categories)), otherwise
        * classes : list(str), list of str which indicates each class
            * each class will has its own color
            * len(classes) indicates how many bars are there in one category (side by side)
        * categories : list(str), list of str which indicates each category
            * a category will be a tick along x-axis
        * save_name : str, saving name of this bar plot
        * title : str, title of this bar plot
        * padding : float, minimum value of each bar
        * expand_floor : int, when len(categories) > `expand_floor`, the width of the figure will expand
            * for len(classes) == 1, `expand_floor` will be multiplied by 2 internally
        * overwrite : bool
            whether overwrite the existing file with the same file name of this plot's saving name

    function(self, f, x_min, x_max, classes, categories, save_names=None,
             n_sample=1000, expand_floor=5, overwrite=True):
        Make multiple (len(categories)) line plots with given function (`f`)
        * f : function
            * input should be an np.ndarray with shape == (n, n_categories)
            * output should be an np.ndarray with shape == (n, n_categories, n_categories)
        * x_min : np.ndarray, minimum x-values for each line plot
            * len(x_min) should be len(categories)
        * x_max : np.ndarray, maximum x-values for each line plot
            * len(x_max) should be len(categories)
        * classes : list(str), list of str which indicates each class
            * each class will has its own color
            * len(classes) indicates how many bars are there in one category (side by side)
        * categories : list(str), list of str which indicates each category
            * every category will correspond to a line plot
        * save_names : list(str), saving names of these line plots
        * n_sample : int, sample density along x-axis
        * expand_floor : int, the width of the figures will be expanded with ratios calculated by:
            expand_ratios = np.maximum(1., np.abs(x_min) / expand_floor, x_max / expand_floor)
        * overwrite : bool
            whether overwrite the existing file with the same file name of this plot's saving name

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


__all__ = ["Anneal", "Metrics", "ScalarEMA", "Grid", "Visualizer"]
