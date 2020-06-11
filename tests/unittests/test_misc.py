import random
import unittest
import numpy as np

from cftool.misc import *


class TestMisc(unittest.TestCase):
    def test_hashcode(self):
        random_str1, random_str2 = str(random.random()), str(random.random())
        hash11, hash21 = map(hash_code, [random_str1, random_str2])
        hash12, hash22 = map(hash_code, [random_str1, random_str2])
        self.assertEqual(hash11, hash12)
        self.assertEqual(hash21, hash22)
        self.assertNotEqual(hash11, hash22)

    def test_prefix_dict(self):
        prefix = "^_^"
        d = {"a": 1, "b": 2, "c": 3}
        self.assertDictEqual(prefix_dict(d, prefix), {"^_^_a": 1, "^_^_b": 2, "^_^_c": 3})

    def test_update_dict(self):
        src_dict = {"a": 1, "b": {"c": 2}}
        tgt_dict = {"b": {"c": 1, "d": 2}}
        update_dict(src_dict, tgt_dict)
        self.assertDictEqual(tgt_dict, {"a": 1, "b": {"c": 2, "d": 2}})

    def test_grouped(self):
        lst = [1, 2, 3, 4, 5, 6]
        self.assertEqual(grouped(lst, 3), [(1, 2, 3), (4, 5, 6)])

    def test_is_number(self):
        self.assertTrue(is_numeric("1e12"))

    def test_get_one_hot(self):
        indices = [1, 4, 2, 3]
        self.assertEqual(
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
            get_one_hot(indices, 5).tolist()
        )

    def test_incrementer(self):
        sequence = np.random.random(1000)
        incrementer = Incrementer()
        for i, n in enumerate(sequence):
            incrementer.update(n)
            sub_sequence = sequence[:i+1]
            mean, std = incrementer.mean, incrementer.std
            self.assertTrue(np.allclose([mean, std], [sub_sequence.mean(), sub_sequence.std()]))
        window_sizes = [3, 10, 30, 100]
        for window_size in window_sizes:
            incrementer = Incrementer(window_size)
            for i, n in enumerate(sequence):
                incrementer.update(n)
                if i < window_size:
                    sub_sequence = sequence[:i + 1]
                else:
                    sub_sequence = sequence[i-window_size+1:i+1]
                mean, std = incrementer.mean, incrementer.std
                self.assertTrue(np.allclose([mean, std], [sub_sequence.mean(), sub_sequence.std()]))


if __name__ == '__main__':
    unittest.main()
