from .hpo import *
from .utils import *
from .param_utils import *


def register_metric(name, sign, requires_prob):
    def _register(f):
        Metrics.add_metric(f, name, sign, requires_prob)
        return f
    return _register


__all__ = [
    "HPOBase", "hpo_dict",
    "Anneal", "Metrics", "ScalarEMA", "Visualizer",
    "Estimator", "ModelPattern", "EnsemblePattern", "Comparer",
    "ParamsGenerator",
    "DataType", "Iterable", "Any", "Int", "Float", "Bool", "String",
    "DistributionBase", "Uniform", "Exponential", "Choice",
    "register_metric"
]
