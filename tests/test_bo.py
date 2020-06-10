from cftool.ml.param_utils import *
from cftool.optim.bo import BayesianOptimization

params = {
    "x1": Float(Uniform(-10, 10)),
    "x2": Float(Uniform(-10, 10))
}

def fn(p):
    return -(p["x1"] + 2 * p["x2"] - 7) ** 2 - (2 * p["x1"] + p["x2"] - 5) ** 2

# Ground Truth is [ 1, 3 ]
bo = BayesianOptimization(fn, params).maximize()
print(bo.best_result)
bo.maximize()
print(bo.best_result)
bo.maximize()
print(bo.best_result)
