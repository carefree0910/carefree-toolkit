import unittest
import numpy as np

from cftool.ml import *
from cftool.ml.param_utils import *


class TestML(unittest.TestCase):
    def test_register_metric(self):

        @register_metric("large_is_good", 1, True)
        def _(self_, _, pred):
            threshold = self_.config.get("threshold", 0.5)
            return (pred[..., 1] >= threshold).mean()

        y = np.random.randint(0, 2, 5).reshape([-1, 1])
        predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape([-1, 1])
        predictions = np.hstack([1 - predictions, predictions])

        metric_ins = Metrics("large_is_good")
        metric_ins.config["threshold"] = 0.5
        self.assertEqual(metric_ins.metric(y, predictions), 0.6)
        metric_ins.config["threshold"] = 0.7
        self.assertEqual(metric_ins.metric(y, predictions), 0.4)
        metric_ins.config["threshold"] = 0.9
        self.assertEqual(metric_ins.metric(y, predictions), 0.2)

    def test_param_generator(self):
        params = {
            "a": Any(Choice(values=[[1, 2], [2, 3]])),
            "b": Iterable([
                Int(Choice(values=[5, 6])),
                Float(Choice(values=[5.1, 5.9])),
                String(Choice(values=["b1", "b2"]))
            ]),
            "c": {
                "d": Int(Choice(values=[7, 8])),
                "e": Any(Choice(values=[10, 9])),
                "f": {
                    "g": String(Choice(values=["g1", "g2"])),
                    "h": Int(Choice(values=[13, 14])),
                    "i": {
                        "j": Int(Choice(values=[15, 16])),
                        "k": String(Choice(values=["k1", "k2"]))
                    }
                }
            },
            "d": {
                "d1": Iterable([
                    Int(Choice(values=[17, 18])),
                    Float(Choice(values=[16.9, 17.5, 18.1]))
                ]),
                "d2": Iterable((
                    Int(Choice(values=[19, 20, 21])),
                    Float(Choice(values=[18.5, 19.5, 20.5, 21.5]))
                ))
            }
        }
        pg = ParamsGenerator(params)
        self.assertEqual(pg.num_params, 2 ** 11 * 3 ** 2 * 4)


if __name__ == '__main__':
    unittest.main()
