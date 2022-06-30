import unittest

import numpy as np

from cftool.array import corr
from cftool.array import allclose
from cftool.array import get_one_hot
from cftool.array import to_standard
from cftool.array import get_unique_indices
from cftool.array import get_counter_from_arr
from cftool.array import get_indices_from_another
from cftool.array import StrideArray


class TestArray(unittest.TestCase):
    def test_get_one_hot(self):
        indices = [1, 4, 2, 3]
        self.assertEqual(
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
            get_one_hot(indices, 5).tolist(),
        )

    def test_get_indices_from_another(self):
        base, segment = np.arange(100), np.random.permutation(100)[:10]
        self.assertTrue(np.allclose(get_indices_from_another(base, segment), segment))

    def test_get_unique_indices(self):
        arr = np.array([1, 2, 3, 2, 4, 1, 0, 1], np.int64)
        rs = get_unique_indices(arr)
        self.assertTrue(np.allclose(rs.unique, np.array([0, 1, 2, 3, 4])))
        self.assertTrue(np.allclose(rs.unique_cnt, np.array([1, 3, 2, 1, 1])))
        gt = np.array([6, 0, 5, 7, 1, 3, 2, 4])
        self.assertTrue(np.allclose(rs.sorting_indices, gt))
        self.assertTrue(np.allclose(rs.split_arr, np.array([1, 4, 6, 7])))
        gt_indices_list = list(map(np.array, [[6], [0, 5, 7], [1, 3], [2], [4]]))
        for rs_indices, gt_indices in zip(rs.split_indices, gt_indices_list):
            self.assertTrue(np.allclose(rs_indices, gt_indices))

    def test_counter_from_arr(self):
        arr = np.array([1, 2, 3, 2, 4, 1, 0, 1])
        counter = get_counter_from_arr(arr)
        self.assertTrue(counter[0], 1)
        self.assertTrue(counter[1], 3)
        self.assertTrue(counter[2], 2)
        self.assertTrue(counter[3], 1)
        self.assertTrue(counter[4], 1)

    def test_allclose(self):
        arr = np.random.random(1000)
        self.assertTrue(allclose(*(arr for _ in range(10))))
        self.assertFalse(allclose(*[arr for _ in range(9)] + [arr + 1e-6]))

    def test_stride_array(self):
        arr = StrideArray(np.arange(9).reshape([3, 3]))
        self.assertTrue(
            np.allclose(
                arr.roll(2),
                np.array([[[0, 1], [1, 2]], [[3, 4], [4, 5]], [[6, 7], [7, 8]]]),
            )
        )
        self.assertTrue(
            np.allclose(
                arr.roll(2, axis=0),
                np.array([[[0, 1, 2], [3, 4, 5]], [[3, 4, 5], [6, 7, 8]]]),
            )
        )
        self.assertTrue(
            np.allclose(
                arr.patch(2),
                np.array(
                    [
                        [[[0, 1], [3, 4]], [[1, 2], [4, 5]]],
                        [[[3, 4], [6, 7]], [[4, 5], [7, 8]]],
                    ]
                ),
            )
        )
        arr = StrideArray(np.arange(16).reshape([4, 4]))
        self.assertTrue(
            np.allclose(
                arr.roll(2, stride=2),
                np.array(
                    [
                        [[0, 1], [2, 3]],
                        [[4, 5], [6, 7]],
                        [[8, 9], [10, 11]],
                        [[12, 13], [14, 15]],
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                arr.roll(2, stride=2, axis=0),
                np.array(
                    [[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                arr.patch(2, h_stride=2, w_stride=2),
                np.array(
                    [
                        [[[0, 1], [4, 5]], [[2, 3], [6, 7]]],
                        [[[8, 9], [12, 13]], [[10, 11], [14, 15]]],
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                arr.patch(2, h_stride=1, w_stride=2),
                np.array(
                    [
                        [[[0, 1], [4, 5]], [[2, 3], [6, 7]]],
                        [[[4, 5], [8, 9]], [[6, 7], [10, 11]]],
                        [[[8, 9], [12, 13]], [[10, 11], [14, 15]]],
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                arr.patch(2, h_stride=2, w_stride=1),
                np.array(
                    [
                        [[[0, 1], [4, 5]], [[1, 2], [5, 6]], [[2, 3], [6, 7]]],
                        [[[8, 9], [12, 13]], [[9, 10], [13, 14]], [[10, 11], [14, 15]]],
                    ]
                ),
            )
        )

    def test_to_standard(self) -> None:
        def _check(src: np.dtype, tgt: np.dtype) -> None:
            self.assertEqual(to_standard(np.array([0], src)).dtype, tgt)

        _check(np.float16, np.float32)
        _check(np.float32, np.float32)
        _check(np.float64, np.float32)
        _check(np.int8, np.int64)
        _check(np.int16, np.int64)
        _check(np.int32, np.int64)
        _check(np.int64, np.int64)

    def test_corr(self) -> None:
        pred = np.random.randn(100, 5)
        target = np.random.randn(100, 5)
        weights = np.zeros([100, 1])
        weights[:30] = weights[-30:] = 1.0
        corr00 = corr(pred, pred, weights)
        corr01 = corr(pred, target, weights)
        corr02 = corr(target, pred, weights)
        w_pred = pred[list(range(30)) + list(range(70, 100))]
        w_target = target[list(range(30)) + list(range(70, 100))]
        corr10 = corr(w_pred, w_pred)
        corr11 = corr(w_pred, w_target)
        corr12 = corr(w_target, w_pred)
        self.assertTrue(allclose(corr00, corr10))
        self.assertTrue(allclose(corr01, corr11, corr02.T, corr12.T))


if __name__ == "__main__":
    TestArray().test_corr()
