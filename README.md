# carefree-toolkit

`carefree-toolkit` implemented some commonly used functions and modules


## Installation

`carefree-toolkit` requires Python 3.8 or higher.

```bash
pip install carefree-toolkit
```

or

```bash
git clone https://github.com/carefree0910/carefree-toolkit.git
cd carefree-toolkit
pip install -e .
```


## Usages

### **`timeit`**

```python
class timeit(context_error_handler):
    def __init__(self, msg)
```

Timing context manager.

#### Parameters

+ **`msg`** : **str**, name of the context which we want to timeit.

#### Example

```python
import time
from cftool.misc import timeit

# ~~~  [ info ] timing for     sleep 1s     : 1.0002
with timeit("sleep 1s"):
    time.sleep(1)
```


### **`timestamp`**

```python
def timestamp(simplify=False, ensure_different=False) -> str
```

Return current timestamp.

#### Parameters

+ **`simplify`** : **bool**. If `True`, format will be simplified to 'year-month-day'.
+ **`ensure_different`** : **bool**. If `True`, format will include millisecond.

#### Example

```python
from cftool.misc import timestamp

# 2019-09-30_21-49-56
print(timestamp())
# 2019-09-30
print(timestamp(simplify=True))
# 2019-09-30_21-49-56-279768
print(timestamp(ensure_different=True))
```


### **`prod`**

```python
def prod(iterable) -> float
```

Return cumulative production of an **`iterable`**.

#### Parameters

+ **`iterable`** : **iterable**.

#### Example

```python
from cftool.misc import prod

# 120.0
print(prod(range(1, 6)))
```


### **`hash_code`**

```python
def hash_code(code) -> str
```

Return hash code for string **`code`**.

#### Parameters

+ **`code`** : **str**.

#### Example

```python
from cftool.misc import hash_code

# True
hash_code("a") != hash_code("b")
```


### **`prefix_dict`**

```python
def prefix_dict(d, prefix) -> dict
```

Prefix every key in dict **`d`** with **`prefix`**, connected with `'_'`.

#### Parameters

+ **`d`** : **dict**.
+ **`prefix`** : **str**.

#### Example

```python
from cftool.misc import prefix_dict

# {"foo_a": 1, "foo_b": 2}
print(prefix_dict({"a": 1, "b": 2}, "foo"))
```


### **`shallow_copy_dict`**

```python
def shallow_copy_dict(d) -> dict
```

Shallow copy dict **`d`**, nested dict is also supported.

#### Parameters

+ **`d`** : **dict**.

#### Example

```python
from cftool.misc import shallow_copy_dict

d = {"a": 1, "b": {"c": 2, "d": 3}}
sd = shallow_copy_dict(d)
d_copy = d.copy()
d["b"].pop("c")
# {'a': 1, 'b': {'d': 3}}
print(d)
# {'a': 1, 'b': {'c': 2, 'd': 3}}
print(sd)
# {'a': 1, 'b': {'d': 3}}
print(d_copy)
```


### **`update_dict`**

```python
def update_dict(src_dict, tgt_dict) -> dict
```

Update **`tgt_dict`** with **`src_dict`**.

> Changes will happen only on keys which **`src_dict`** holds, and the update procedure will be recursive.

> Changed will happen inplace.

#### Parameters

+ **`src_dict`** : **dict**.
+ **`tgt_dict`** : **str**.

#### Example

```python
from cftool.misc import update_dict

src_dict = {"a": {"b": 1}, "c": 2}
tgt_dict = {"a": {"b": 0, "b1": 1}, "c": 0, "d": 1}
# {"a": {"b": 1, "b1": 1}, "c": 2, "d": 1}
print(update_dict(src_dict, tgt_dict))
```


### **`fix_float_to_length`**

```python
def fix_float_to_length(num, length) -> str
```

Change a float number to string format with fixed length.

#### Parameters

+ **`num`** : **float**.
+ **`length`** : **int**.

#### Example

```python
import math
from cftool.misc import fix_float_to_length

# 1.000000
print(fix_float_to_length(1, 8))
# 1.000000
print(fix_float_to_length(1., 8))
# 1.000000
print(fix_float_to_length(1.0, 8))
# -1.00000
print(fix_float_to_length(-1, 8))
# -1.00000
print(fix_float_to_length(-1., 8))
# -1.00000
print(fix_float_to_length(-1.0, 8))
# 1234567.
print(fix_float_to_length(1234567, 8))
# 12345678
print(fix_float_to_length(12345678, 8))
# 123456789
print(fix_float_to_length(123456789, 8))
# +  nan   +
print("+" + fix_float_to_length(math.nan, 8) + "+")
```


### **`truncate_string_to_length`**

```python
def truncate_string_to_length(string, length) -> str
```

Truncate a string to make sure its length not exceeding a given length.

#### Parameters

+ **`string`** : **str**.
+ **`length`** : **int**.

#### Example

```python
from cftool.misc import truncate_string_to_length

# 123456
print(truncate_string_to_length("123456", 6))
# 12..67
print(truncate_string_to_length("1234567", 6))
# 12..78
print(truncate_string_to_length("12345678", 6))
# 12...78
print(truncate_string_to_length("12345678", 7))
```


### **`grouped`**

```python
def grouped(iterable, n, *, keep_tail) -> list
```

Group an **`iterable`** every **`n`** elements.

#### Parameters

+ **`iterable`** : **iterable**.
+ **`n`** : **int**.
+ **`keep_tail`** : **bool**, whether keep the 'tail' (see example below).

#### Example

```python
from cftool.misc import grouped

# [(0, 1), (2, 3), (4, 5)]
print(grouped(range(6), 2))
# [(0, 1, 2), (3, 4, 5)]
print(grouped(range(6), 3))
# [(0, 1, 2, 3)]
print(grouped(range(6), 4))
# [(0, 1, 2, 3), (4, 5)]
print(grouped(range(6), 4, keep_tail=True))
```


### **`is_number`**

```python
def is_numeric(s) -> bool
```

Check whether string **`s`** is numeric.

#### Parameters

+ **`s`** : **str**.

#### Example

```python
from cftool.misc import is_numeric

# True
print(is_numeric(0x1))
# True
print(is_numeric(1e0))
# True
print(is_numeric("1"))
# True
print(is_numeric("1."))
# True
print(is_numeric("1.0"))
# True
print(is_numeric("1.00"))
# False
print(is_numeric("1.0.0"))
# True
print(is_numeric("nan"))
```


### **`get_one_hot`**

```python
def get_one_hot(feature, dim) -> np.ndarray
```

Get one-hot representation.

#### Parameters

+ **`feature`** : **array-like**, source data of one-hot representation.
+ **`dim`** : **int**, dimension of the one-hot representation. 

#### Example

```python
import numpy as np
from cftool.array import get_one_hot

feature = np.array([0, 1, 0])
# [[1 0], [0 1], [1 0]]
print(get_one_hot(feature, 2))
# [[1 0 0] [0 1 0] [1 0 0]]
print(get_one_hot(feature, 3))
# [[1 0 0] [0 1 0] [1 0 0]]
print(get_one_hot(feature.tolist(), 3))
```


### **`get_indices_from_another`**

```python
def get_indices_from_another(base, segment) -> np.ndarray
```

Get **`segment`** elements' indices in **`base`**. This function will return positions where elements in **`segment`** appear in **`base`**.

> All elements in segment should appear in base to ensure validity.

#### Parameters

+ **`base`** : **np.ndarray**, base array.
+ **`segment`** : **np.ndarray**, segment array. 

#### Example

```python
import numpy as np
from cftool.array import get_indices_from_another

base, segment = np.array([1, 2, 3, 5, 7, 8, 9]), np.array([1, 3, 5, 7, 9])
# [0 2 3 4 6]
print(get_indices_from_another(base, segment))
# [0 1 2 3 4]
print(get_indices_from_another(segment, segment))
# [4 3 2 1 0]
print(get_indices_from_another(segment[::-1], segment))
```


### **`get_unique_indices`**

```python
def get_unique_indices(arr) -> UniqueIndices
```

 Get indices for unique values of an array.

#### Parameters

+ **`arr`** : **np.ndarray**, target array which we wish to find indices of each unique value.
+ **`return_raw`** : **bool**, whether returning raw information.

#### Example

```python
import numpy as np
from cftool.array import get_unique_indices

arr = np.array([1, 2, 3, 2, 4, 1, 0, 1], np.int64)
unique_indices = get_unique_indices(arr)
# UniqueIndices(
#   unique          = array([0, 1, 2, 3, 4], dtype=int64),
#   unique_cnt      = array([1, 3, 2, 1, 1], dtype=int64),
#   sorting_indices = array([6, 0, 5, 7, 1, 3, 2, 4], dtype=int64),
#   split_arr       = array([1, 4, 6, 7], dtype=int64))
#   split_indices   = [array([6], dtype=int64), array([0, 5, 7], dtype=int64), array([1, 3], dtype=int64),
#                      array([2], dtype=int64), array([4], dtype=int64)]
print(get_unique_indices(arr))
```


### And more...

`carefree-toolkit` is well documented, feel free to dive into the codes and explore something you may need!


## License

`carefree-toolkit` is MIT licensed, as found in the [`LICENSE`](https://github.com/carefree0910/carefree-toolkit/blob/master/LICENSE) file.

---
