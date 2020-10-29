import math
import time
import random
import unittest
import numpy as np

from cftool.misc import *

test_dict = {}


class TestMisc(unittest.TestCase):
    def test_timestamp(self):
        t1 = timestamp(simplify=True)
        time.sleep(1)
        t2 = timestamp(simplify=True)
        self.assertEqual(t1, t2)
        t1 = timestamp()
        time.sleep(1.2)
        t2 = timestamp()
        self.assertNotEqual(t1, t2)
        t1 = timestamp(ensure_different=True)
        time.sleep(1e-6)
        t2 = timestamp(ensure_different=True)
        self.assertNotEqual(t1, t2)

    def test_prod(self):
        numbers = range(1, 6)
        self.assertEqual(prod(numbers), 120)

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
        self.assertDictEqual(
            prefix_dict(d, prefix),
            {"^_^_a": 1, "^_^_b": 2, "^_^_c": 3},
        )

    def test_shallow_copy_dict(self):
        d = {"a": {"b": 1}}
        dc1 = shallow_copy_dict(d)
        dc1["a"]["b"] = 2
        self.assertEqual(d["a"]["b"], 1)
        dc2 = d.copy()
        dc2["a"]["b"] = 2
        self.assertEqual(d["a"]["b"], 2)

    def test_update_dict(self):
        src_dict = {"a": 1, "b": {"c": 2}}
        tgt_dict = {"b": {"c": 1, "d": 2}}
        update_dict(src_dict, tgt_dict)
        self.assertDictEqual(tgt_dict, {"a": 1, "b": {"c": 2, "d": 2}})

    def test_fix_float_to_length(self):
        self.assertEqual(fix_float_to_length(1, 8), "1.000000")
        self.assertEqual(fix_float_to_length(1.0, 8), "1.000000")
        self.assertEqual(fix_float_to_length(1.0, 8), "1.000000")
        self.assertEqual(fix_float_to_length(-1, 8), "-1.00000")
        self.assertEqual(fix_float_to_length(-1.0, 8), "-1.00000")
        self.assertEqual(fix_float_to_length(-1.0, 8), "-1.00000")
        self.assertEqual(fix_float_to_length(1234567, 8), "1234567.")
        self.assertEqual(fix_float_to_length(12345678, 8), "12345678")
        self.assertEqual(fix_float_to_length(123456789, 8), "123456789")
        self.assertEqual("+" + fix_float_to_length(math.nan, 8) + "+", "+  nan   +")

    def test_truncate_string_to_length(self):
        self.assertEqual(truncate_string_to_length("123456", 6), "123456")
        self.assertEqual(truncate_string_to_length("1234567", 6), "12..67")
        self.assertEqual(truncate_string_to_length("12345678", 6), "12..78")
        self.assertEqual(truncate_string_to_length("12345678", 7), "12...78")

    def test_grouped(self):
        lst = [1, 2, 3, 4, 5, 6, 7, 8]
        self.assertEqual(grouped(lst, 3), [(1, 2, 3), (4, 5, 6)])
        self.assertEqual(
            grouped(lst, 3, keep_tail=True),
            [(1, 2, 3), (4, 5, 6), (7, 8)],
        )
        unit = 10 ** 5
        lst = list(range(3 * unit + 1))
        gt = [
            tuple(range(unit)),
            tuple(range(unit, 2 * unit)),
            tuple(range(2 * unit, 3 * unit)),
            (3 * unit,),
        ]
        self.assertEqual(grouped(lst, unit, keep_tail=True), gt)
        self.assertEqual(grouped(lst, unit), gt[:-1])

    def test_is_numeric(self):
        self.assertEqual(is_numeric(0x1), True)
        self.assertEqual(is_numeric(1e0), True)
        self.assertEqual(is_numeric("1"), True)
        self.assertEqual(is_numeric("1."), True)
        self.assertEqual(is_numeric("1.0"), True)
        self.assertEqual(is_numeric("1.00"), True)
        self.assertEqual(is_numeric("1.0.0"), False)
        self.assertEqual(is_numeric("Ⅱ"), True)
        self.assertEqual(is_numeric("nan"), True)

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

    def test_register_core(self):
        global test_dict

        def before_register(cls):
            cls.before = "test_before"

        def after_register(cls):
            self.assertEqual(cls.before, "test_before")
            cls.after = "test_after"

        def register(name):
            return register_core(
                name,
                test_dict,
                before_register=before_register,
                after_register=after_register,
            )

        @register("foo")
        class Foo:
            pass

        self.assertIs(Foo, test_dict["foo"])
        self.assertEqual(Foo.after, "test_after")

        @register("foo")
        class Foo2:
            pass

        self.assertIs(Foo, test_dict["foo"])
        self.assertEqual(Foo2.before, "test_before")
        with self.assertRaises(AttributeError):
            _ = Foo2.after

    def test_incrementer(self):
        sequence = np.random.random(1000)
        incrementer = Incrementer()
        for i, n in enumerate(sequence):
            incrementer.update(n)
            sub_sequence = sequence[: i + 1]
            mean, std = incrementer.mean, incrementer.std
            self.assertTrue(
                np.allclose(
                    [mean, std],
                    [sub_sequence.mean(), sub_sequence.std()],
                )
            )
        window_sizes = [3, 10, 30, 100]
        for window_size in window_sizes:
            incrementer = Incrementer(window_size)
            for i, n in enumerate(sequence):
                incrementer.update(n)
                if i < window_size:
                    sub_sequence = sequence[: i + 1]
                else:
                    sub_sequence = sequence[i - window_size + 1 : i + 1]
                mean, std = incrementer.mean, incrementer.std
                self.assertTrue(
                    np.allclose(
                        [mean, std],
                        [sub_sequence.mean(), sub_sequence.std()],
                    )
                )

    def test_check(self):
        @check({"arg1": "int", "arg2": ["int", "odd"]})
        def foo(arg1, arg2):
            pass

        foo(1, 1)
        with self.assertRaises(ValueError):
            foo(1, 2)
        with self.assertRaises(ValueError):
            foo(1.1, 1)
        with self.assertRaises(ValueError):
            foo(1, 1.1)

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


if __name__ == "__main__":
    unittest.main()
