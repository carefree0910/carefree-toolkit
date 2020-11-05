import math
import unittest
import numpy as np

from cftool.stat import *

test_dict = {}


class TestStat(unittest.TestCase):
    def test_rolling_stat(self):
        n = 10000
        arr = np.arange(n)
        mean_gt = np.arange(1, n - 1)
        std_gt = np.full([n - 2], math.sqrt(2.0 / 3.0))
        self.assertTrue(np.allclose(RollingStat.mean(arr, 3), mean_gt))
        self.assertTrue(np.allclose(RollingStat.std(arr, 3), std_gt))


if __name__ == "__main__":
    unittest.main()
