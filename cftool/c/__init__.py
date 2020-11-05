from .cython_substitute import naive_rolling_min
from .cython_substitute import naive_rolling_max

try:
    from .cython_wrappers import c_rolling_min as rolling_min
    from .cython_wrappers import c_rolling_max as rolling_max
except ImportError:
    rolling_min = naive_rolling_min
    rolling_max = naive_rolling_max


__all__ = [
    "rolling_min",
    "rolling_max",
    "naive_rolling_min",
    "naive_rolling_max",
]
