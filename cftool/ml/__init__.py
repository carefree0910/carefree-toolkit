from .utils import *


def register_metric(name, sign, requires_prob):
    def _register(f):
        Metrics.add_metric(f, name, sign, requires_prob)
        return f
    return _register


__all__ = [
    "Anneal", "Metrics", "ScalarEMA", "Visualizer", "Tracker",
    "Estimator", "ModelPattern", "EnsemblePattern", "Comparer",
    "register_metric", "DataInspector",
]
