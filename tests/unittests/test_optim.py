import unittest
import numpy as np

from typing import *
from tqdm import tqdm

from cftool.optim import *


class TestOptim(unittest.TestCase):
    def test_gradient_descent(self):
        def _get_stacked(x_):
            x2, x3 = map(np.power, 2 * [x_], [2, 3])
            return np.hstack([x_, x2, x3])

        class _test(GradientDescentMixin):
            def __init__(self):
                self.w = np.random.random([1, 3])

            @property
            def parameter_names(self) -> List[str]:
                return ["w"]

            def loss_function(self,
                              x_batch: np.ndarray,
                              y_batch: np.ndarray,
                              batch_indices: np.ndarray) -> Dict[str, Any]:
                x_batch_stacked = _get_stacked(x_batch)
                predictions = self.w * x_batch_stacked
                diff = predictions - y_batch
                return {
                    "diff": diff,
                    "x_stacked": x_batch_stacked,
                    "loss": np.abs(diff).mean().item()
                }

            def gradient_function(self,
                                  x_batch: np.ndarray,
                                  y_batch: np.ndarray,
                                  batch_indices: np.ndarray,
                                  loss_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
                diff, x_stacked_ = map(loss_dict.get, ["diff", "x_stacked"])
                return {"w": (np.sign(diff) * x_stacked_).mean(0, keepdims=True)}

        num_try = 50
        for _ in tqdm(list(range(num_try))):
            x = np.random.random([1000, 1])
            x_stacked = _get_stacked(x)
            w_true = np.random.random([1, 3])
            y = x_stacked * w_true
            gd = _test().setup_optimizer("adam", 3e-4, epoch=100)
            gd.show_tqdm = False
            gd.gradient_descent(x, y)
            self.assertTrue(np.allclose(gd.w, w_true, 1e-3, 1e-3))


if __name__ == '__main__':
    unittest.main()
