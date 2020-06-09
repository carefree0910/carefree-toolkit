from typing import *

number_type = Union[int, float]
generic_number_type = Union[number_type, Any]
nullable_number_type = Union[number_type, None]
generic_iterable_type = Iterable


__all__ = [
    "number_type", "generic_number_type",
    "nullable_number_type", "generic_iterable_type"
]
