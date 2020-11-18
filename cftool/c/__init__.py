from .cython_substitute import naive_rolling_sum
from .cython_substitute import naive_rolling_min
from .cython_substitute import naive_rolling_max
from .cython_substitute import naive_ema

try:
    from .cython_wrappers import c_rolling_sum as rolling_sum
    from .cython_wrappers import c_rolling_min as rolling_min
    from .cython_wrappers import c_rolling_max as rolling_max
    from .cython_wrappers import c_ema as ema
except ImportError:
    rolling_sum = naive_rolling_sum
    rolling_min = naive_rolling_min
    rolling_max = naive_rolling_max
    ema = naive_ema


__all__ = [
    "rolling_sum",
    "rolling_min",
    "rolling_max",
    "ema",
    "naive_rolling_sum",
    "naive_rolling_min",
    "naive_rolling_max",
    "naive_ema",
]
