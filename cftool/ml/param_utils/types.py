from typing import *

number_type = Union[int, float]
generic_number_type = Union[number_type, Any]
nullable_number_type = Union[number_type, None]

nested_type = Dict[str, Union[Any, Dict[str, "nested_type"]]]
all_nested_type = Dict[str, Union[List[Any], Dict[str, "all_nested_type"]]]
flattened_type = Dict[str, Any]
all_flattened_type = Dict[str, List[Any]]


__all__ = [
    "number_type", "generic_number_type", "nullable_number_type",
    "nested_type", "all_nested_type", "flattened_type", "all_flattened_type"
]
